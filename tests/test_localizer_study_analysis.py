import pytest
pytest.importorskip('torch')
pytest.importorskip('scipy')
pytest.importorskip('sklearn')
from scripts.analyze_localizer_study import paired_clusters, candidate_diagnostic, per_query, window_score_margin


def test_cluster_pair_preserves_queries_and_reports_percentage_point_units():
    a = {1:{'source':'same','ap':0.,'r1_05':0.},2:{'source':'same','ap':0.,'r1_05':0.}}
    b = {1:{'source':'same','ap':1.,'r1_05':1.},2:{'source':'same','ap':0.,'r1_05':0.}}
    result = paired_clusters(a,b,resamples=50)
    assert result['source_clusters'] == 1 and result['queries'] == 2
    assert result['mAP_delta_pp'] == 50 and result['mAP_ci95_pp'] == [50,50]
    with pytest.raises(ValueError): paired_clusters(a,{1:b[1]})
    with pytest.raises(ValueError): paired_clusters(a,{**b,1:{**b[1],'source':'wrong'}})


def test_candidate_vs_ranking_and_missing_query_denominator_are_separate():
    gt = {1:{'vid':'one_0.0_30.0','relevant_windows':[[2.,4.]]},
          2:{'vid':'two_0.0_30.0','relevant_windows':[[10.,25.]]}}
    pool = {1:[[2.,4.,1.]],2:[[10.,25.,1.]]}
    predictions = {1:[],2:[[10.,25.,1.]]}
    result = candidate_diagnostic(pool,predictions,gt)
    assert result['all_candidate_query_oracle_hits']==2 and result['top5_query_oracle_hits']==1
    assert result['short_pool_hits']==1 and result['short_top5_hits']==0
    assert per_query(predictions,gt)[1]['ap']==0.


def test_posthoc_score_margin_obeys_half_open_windows_and_missing_background():
    assert window_score_margin([0.,2.,0.],[0.,1.,2.],[[1.,2.]])==2.
    assert window_score_margin([1.,2.],[0.,1.],[[0.,2.]]) is None
    assert window_score_margin([1.,2.],[0.,1.],[[.2,.3]]) is None
    with pytest.raises(ValueError): window_score_margin([float('nan')],[0.],[[0.,1.]])
