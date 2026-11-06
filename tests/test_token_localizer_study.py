import copy
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')

from scripts import token_localizer_study as study
from queryseek.retrieval.token_localizer import training_target
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file


def samples():
    rng=np.random.default_rng(1);t=np.arange(12,dtype=float)
    records=[{'qid':i,'source':str(i//2),'times':t,'duration_sec':12.,
              'images':rng.normal(size=(12,512)).astype(np.float32)*.1} for i in range(8)]
    e={'relevant_windows':[[2.,8.]],'relevant_clip_ids':[1,2,3],'saliency_scores':[[3,3,3]]*3}
    targets={r['qid']:training_target(t,12.,e) for r in records}
    text={r['qid']:rng.normal(size=(4+r['qid']%2,512)).astype(np.float32)*.1 for r in records}
    return records,text,targets


def test_preplanned_equal_budget_batches_never_pair_same_source_and_are_reproducible():
    records,_,targets=samples();lookup={r['qid']:r for r in records}
    plan=study.sampling_plan(records,targets,12,3,42)
    assert plan==study.sampling_plan(records,targets,12,3,42)
    assert sum(len(b['qids']) for b in plan)==36
    assert all(len({lookup[q]['source'] for q in b['qids']})==3 for b in plan)
    for batch in plan:
        assert all(targets[q]['timestamp_window'][pos] for q,pos in zip(batch['qids'],batch['positive_positions']))
    with pytest.raises(ValueError):study.sampling_plan(records,targets,1,5,42)


def test_lr_schedule_is_fixed_in_update_units():
    cfg={'warmup_steps':4,'learning_rate':1e-4,'decay_gamma':.1,'decay_steps':[8,12]}
    assert study.learning_rate(cfg,0)==2.5e-5
    assert study.learning_rate(cfg,3)==1e-4
    assert np.isclose(study.learning_rate(cfg,8),1e-5)
    assert np.isclose(study.learning_rate(cfg,12),1e-6)


def test_resume_matches_uninterrupted_optimizer_and_rng_without_repeating_updates(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    records,text,targets=samples();cfg=study.read(study.ROOT/'configs/token_localizer_v1.json')
    cfg={**cfg,'steps':4,'batch_size':2,'checkpoint_every':2,'log_every':1,'warmup_steps':2,
         'decay_steps':[2,3],'capacities':{'small':{'hidden':16,'heads':4,'layers':1,'ffn':32}}}
    plan=study.sampling_plan(records,targets,4,2,12)
    p={'config':cfg,'scale_summary':{'64':{'queries':8,'sources':4}}}
    reference=tmp_path/'reference';resumed=tmp_path/'resumed';reference.mkdir();resumed.mkdir()
    for path in (reference,resumed):write_json_atomic(path/'sampling_plan_64.json',plan)
    baseline=study.fit_arm(reference,p,records,text,targets,'64/small')
    original=study.write_json_atomic
    class Interrupted(Exception):pass
    def write_then_interrupt(path,value):
        original(path,value)
        if path.name=='progress.json' and 'checkpoint_step' in value and value['checkpoint_step']==2:
            raise Interrupted()
    monkeypatch.setattr(study,'write_json_atomic',write_then_interrupt)
    with pytest.raises(Interrupted):study.fit_arm(resumed,p,records,text,targets,'64/small')
    monkeypatch.setattr(study,'write_json_atomic',original)
    result=study.fit_arm(resumed,p,records,text,targets,'64/small')
    assert result['steps']==4 and result['final_tensor_sha256']==baseline['final_tensor_sha256']
    assert result['initial_sha256']==baseline['initial_sha256']
    # Completed arm is read-only and returns the same fit instead of fitting again.
    assert study.fit_arm(resumed,p,records,text,targets,'64/small')==result
    # Simulate interruption after the last checkpoint, before the completion marker.
    (resumed/'models/64/small/fit.json').unlink()
    again=study.fit_arm(resumed,p,records,text,targets,'64/small')
    assert again['final_tensor_sha256']==result['final_tensor_sha256']


def test_signature_separates_query_token_model_code_and_pipeline_configuration():
    signature={'pipeline_version':'2','clip_weights_sha256':'weights','token_cap':32,'length_policy':'upstream_nonzero',
               'manifest_sha256':'manifest','source_sha256':{'model':'source'},'versions':{'torch':'version'}}
    for name in signature:
        changed=copy.deepcopy(signature);changed[name]='different'
        assert study.digest(signature)!=study.digest(changed)


def test_frozen_source_notes_manifest_and_history_tamper_are_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    source=tmp_path/'code.py';source.write_text('original')
    out=tmp_path/'outputs/run';snapshot=out/'source_snapshot';snapshot.mkdir(parents=True)
    (snapshot/'code.py').write_text('original')
    (out/'inference_manifest.json').write_text('[]');(out/'protocol_notes_frozen.md').write_text('frozen')
    weights=tmp_path/'weights';weights.write_bytes(b'weights');historical=tmp_path/'prior.json';historical.write_text('prior')
    p={'source_sha256':{'code.py':sha256_file(source)},'preserved_sha256':{'prior.json':sha256_file(historical)},
       'manifest_sha256':sha256_file(out/'inference_manifest.json'),'protocol_notes_sha256':sha256_file(out/'protocol_notes_frozen.md'),
       'config':{'weight_path':str(weights)},'clip_weights_sha256':sha256_file(weights)}
    write_json_atomic(out/'protocol.json',p);study.verify_protocol(out)
    historical.write_text('modified')
    with pytest.raises(ValueError,match='Previous'):study.verify_protocol(out)


def test_output_path_traversal_rejected():
    with pytest.raises(ValueError):study.checked_output(study.ROOT/'outputs/../../escape')
