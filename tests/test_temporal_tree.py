from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from queryseek.agent.temporal_tree import (
    NodeObservation, SplitConfig, TemporalTree, TreeBudgetExceeded, TreeNavigator, visual_change_split)
from queryseek.agent import tree_observations as cache
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.schemas import VideoFrame
from scripts import probe_scene_tree as probe


def observation(start=0., end=120.):
    times = [start+(i+.5)*(end-start)/12 for i in range(12)]
    frames = [VideoFrame(frame_id=f't{t:.6f}', timestamp_sec=t, image_path='unused', source='uniform') for t in times]
    return NodeObservation(frames, np.repeat(np.eye(3, dtype=np.float32), 4, axis=0))


def test_visual_boundaries_follow_changes_and_preserve_brackets_and_partition():
    spans, info = visual_change_split(0., 120., observation(), SplitConfig(min_children=3, max_children=3))
    assert spans == [(0., 40.), (40., 80.), (80., 120.)]
    assert info['boundary_brackets_sec'] == [[35., 45.], [75., 85.]]
    assert not info['forced_minimum']
    flat = observation()
    flat.embeddings[:] = [1., 0., 0.]
    spans, info = visual_change_split(0., 120., flat, SplitConfig())
    assert len(spans) == 3 and info['forced_minimum']


@pytest.mark.parametrize('kind', ['nan', 'zero', 'wrong_rows', 'out_of_window', 'duplicate_pts'])
def test_bad_features_or_times_cannot_construct_nodes(kind):
    obs = observation()
    if kind == 'nan': obs.embeddings[0, 0] = np.nan
    if kind == 'zero': obs.embeddings[0] = 0
    if kind == 'wrong_rows': obs.embeddings = obs.embeddings[:-1]
    if kind == 'out_of_window': obs.frames[-1].timestamp_sec = 120.
    if kind == 'duplicate_pts': obs.frames[1].timestamp_sec = obs.frames[0].timestamp_sec
    with pytest.raises(ValueError):
        visual_change_split(0., 120., obs, SplitConfig())


def test_lazy_expansion_reuses_nodes_and_budget_failure_does_not_mark_leaf():
    calls = []
    def provider(a,b):
        calls.append((a,b))
        return observation(a,b)
    tree = TemporalTree(120., provider, SplitConfig(leaf_seconds=10., max_observed_nodes=2))
    assert calls == []
    children = tree.expand('r')
    assert tree.expand('r') == children and len(calls) == 1
    tree.expand(children[0])
    with pytest.raises(TreeBudgetExceeded): tree.expand(children[1])
    assert tree.nodes[children[1]].children is None and len(calls) == 2
    leaf = TemporalTree(63., provider)
    assert leaf.expand('r') == () and len(calls) == 2


def test_navigation_multichild_sibling_parent_and_unknown_survive_departure():
    tree = TemporalTree(120., observation, SplitConfig(leaf_seconds=10., max_observed_nodes=4))
    nav = TreeNavigator(tree, max_frames=6, max_moves=4)
    children = nav.expand()
    assert len(children) == 3
    assert nav.state()['children'][1]['window_sec'] == [40., 80.]
    assert len(nav.observe(4)) == 4
    nav.move('zoom_in', children[0])
    assert len(nav.state()['siblings']) == 2 and nav.state()['parent_id'] == 'r'
    grandchildren = nav.expand()
    nav.move('shift', children[2])
    assert len(nav.observe(4)) == 2
    assert nav.observe() == []
    with pytest.raises(ValueError): nav.move('shift', grandchildren[0])
    nav.move('zoom_out')
    assert nav.current_id == 'r' and len(nav.observed) == 6
    assert nav.state()['evidence_status'] == 'unknown'
    with pytest.raises(ValueError): nav.move('zoom_out')
    nav.move('zoom_in', children[0])
    with pytest.raises(TreeBudgetExceeded): nav.move('zoom_out')


def test_short_leaf_can_be_observed_without_enough_frames_to_split():
    def provider(a,b):
        obs = observation(a,b)
        return NodeObservation(obs.frames[:2],obs.embeddings[:2])
    tree = TemporalTree(2., provider)
    assert tree.expand('r') == ()
    nav = TreeNavigator(tree)
    assert len(nav.observe()) == 2
    assert nav.state()['evidence_status'] == 'unknown'


def test_source_cache_roundtrip_model_and_sampling_invalidation_and_tamper(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    asset = asset_from_probe('sample', demo['video'])
    cfg = cache.SamplingConfig(max_frames=6, max_side=96)
    encoder = cache.MockBoundaryEncoder()
    provider = cache.WindowObservationStore(asset.path, asset.duration_sec, asset.sha256, encoder, {'mock': 'v1'}, cfg)
    first, second = provider(0., 60.), provider(0., 60.)
    assert first.cost['encoded_images'] == first.cost['new_frames_extracted'] == 6
    assert second.cost['feature_cache_hit'] and second.cost['encoded_images'] == 0
    assert [f.timestamp_sec for f in first.frames] == [f.timestamp_sec for f in second.frames]
    assert all(0 <= f.timestamp_sec < 60 for f in first.frames)
    different_model = cache.WindowObservationStore(asset.path, asset.duration_sec, asset.sha256, encoder, {'mock': 'v2'}, cfg)(0.,60.)
    assert different_model.cost['frame_cache_hit'] and different_model.cost['encoded_images'] == 6
    different_sampling = cache.WindowObservationStore(asset.path, asset.duration_sec, asset.sha256, encoder, {'mock': 'v1'}, replace(cfg, max_frames=8))(0.,60.)
    assert different_sampling.cost['new_frames_extracted'] == 8
    for p in (tmp_path/'outputs/temporal_tree_embedding_cache').glob('*/features.npy'):
        p.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Feature cache mismatch'): provider(0.,60.)
    with pytest.raises(ValueError): cache.output_path(tmp_path/'outside')
    with pytest.raises(ValueError): provider(0.,61.)


def make_config(requests, labels):
    return {'requests_sha256': sha256_file(requests), 'labels_sha256': sha256_file(labels),
        'model_name': 'ViT-B-32', 'weights_sha256': '0'*64, 'force_quick_gelu': True,
        'sampling': {'max_frames': 8, 'max_side': 96},
        'split': {'leaf_seconds': 10., 'max_observed_nodes': 2},
        'root_comparisons': ['binary_uniform','same_arity_uniform','clip_change']}


def test_mock_build_label_isolation_and_immutable_runs(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    requests, labels, config = (tmp_path/name for name in ['requests.json', 'labels.json', 'config.json'])
    row = {'qid':'x','video_id':'sample','question':'secret query irrelevant to builder',
           'choices':['a','b'],'duration_sec':60.,'video_path':str(demo['video'])}
    cache.write_json(requests,[row])
    cache.write_json(labels,[{'qid':'x','right_answer':'A','gt_intervals_sec':[[12.,24.]]}])
    cfg = make_config(requests,labels)
    cache.write_json(config,cfg)
    output = tmp_path/'outputs/run'
    result = probe.build(requests,config,output,mock=True)
    assert result['completed'] == 1 and result['failed'] == result['new_vlm_calls'] == 0
    trees = (output/'trees.jsonl').read_text('utf-8')
    assert 'secret query' not in trees and 'gt_intervals' not in trees and 'right_answer' not in trees
    with pytest.raises(FileExistsError): probe.build(requests,config,output,mock=True)
    with pytest.raises(ValueError,match='Mock'): probe.evaluate(output,labels,config,tmp_path/'outputs/eval')
    cache.write_json(requests,[{**row,'gt_intervals_sec':[[12.,24.]]}])
    cfg['requests_sha256'] = sha256_file(requests)
    cache.write_json(config,cfg)
    with pytest.raises(ValueError,match='whitelisted'):
        probe.build(requests,config,tmp_path/'outputs/leaky',mock=True)


def test_geometry_evaluation_keeps_failed_queries_and_same_arity_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    run = tmp_path/'outputs/run'
    run.mkdir(parents=True)
    labels, config = tmp_path/'labels.json', tmp_path/'config.json'
    cache.write_json(labels,[{'qid': q, 'gt_intervals_sec':[[20.,40.]]} for q in ['a','b']])
    cfg = {'labels_sha256':sha256_file(labels),'root_comparisons':['binary_uniform','same_arity_uniform','clip_change']}
    cache.write_json(config,cfg)
    cache.write_json(run/'run_manifest.json',{'mock':False,'config_sha256':sha256_file(config)})
    nodes=[{'node_id':'r','children':['r.0','r.1','r.2'],
            'split_info':{'source_times_sec':[10.,30.,100.], 'boundary_brackets_sec':[[10.,30.],[30.,50.]],'forced_minimum':False}}]
    nodes += [{'node_id':f'r.{i}','start_sec':a,'end_sec':b} for i,(a,b) in enumerate([(0.,20.),(20.,40.),(40.,120.)])]
    rows=[{'qid':'a','status':'ok','tree':{'nodes':nodes,'duration_sec':120.}}, {'qid':'b','status':'error'}]
    (run/'trees.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    result=probe.evaluate(run,labels,config,tmp_path/'outputs/eval')
    assert result['n']==2 and result['completed']==1
    assert result['methods']['clip_change']['best_child_union_iou']==.5
    assert result['methods']['same_arity_uniform']['best_child_union_iou']==.25
    assert result['methods']['binary_uniform']['best_child_union_iou']==pytest.approx(1/6)
    assert result['qa_accuracy'] is None


def test_build_preserves_partial_tree_costs_on_failure(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, 'ROOT', tmp_path)
    requests, labels, config = (tmp_path/name for name in ['requests.json','labels.json','config.json'])
    cache.write_json(requests,[{'qid':'x','video_id':'sample','question':'unused','choices':['a','b'],
                              'duration_sec':60.,'video_path':str(demo['video'])}])
    cache.write_json(labels,[])
    cache.write_json(config,make_config(requests,labels))
    def fail_navigation(self):
        raise RuntimeError('This exception text must not enter traces')
    monkeypatch.setattr(probe.TreeNavigator, 'state', fail_navigation)
    output=tmp_path/'outputs/failed'
    result=probe.build(requests,config,output,mock=True)
    assert result['failed']==1 and result['completed']==0
    assert result['encoded_images'] > 0 and result['observed_nodes']==2
    assert not result['observation_costs_complete']
    text=(output/'trees.jsonl').read_text()
    row=json.loads(text)
    assert row['error_stage']=='navigation_tool_check' and row['tree']['observed_nodes']==2
    assert 'exception text' not in text and row['error_locations']
