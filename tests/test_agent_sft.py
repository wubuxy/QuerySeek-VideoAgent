import json

import pytest

from queryseek.agent.sft_policy import AnnotationTeacher, SearchPolicy, SFT_PROMPT
from queryseek.agent.search_policy import rollout
from queryseek.schemas import VideoFrame


def config():
    return {"max_frames":16,"frames_per_turn":4,"max_turns":4,"max_depth":3}


def frames():
    return [VideoFrame(frame_id=f"f{i}",timestamp_sec=float(i),image_path="unused") for i in range(64)]


def test_teacher_recovers_from_wrong_half_and_never_leaks_labels_to_prompt():
    ff=frames();cfg=config()
    teacher=AnnotationTeacher(ff,64.,[[40.,58.]],cfg,"backtracking")
    row=rollout("visible event",64.,ff,"backtracking",(0,),cfg,SearchPolicy(teacher,ff,64.))
    assert row["status"]=="answered"
    assert any(t["decision"]["action"] in {"shift","zoom_out"} for t in row["trace"])
    for x in teacher.examples:
        p=json.loads(x["prompt"][len(SFT_PROMPT):])
        assert "gt_windows" not in p and "target" not in p and "teacher" not in p
    assert all(40.<=next(f.timestamp_sec for f in ff if f.frame_id==i)<58.
               for i in row["trace"][-1]["decision"]["evidence_ids"])


def test_teacher_stops_when_evidence_is_already_sufficient():
    ff=frames();teacher=AnnotationTeacher(ff,64.,[[0.,64.]],config(),"backtracking")
    row=rollout("event",64.,ff,"backtracking",(),config(),SearchPolicy(teacher,ff,64.))
    assert row["status"]=="answered" and len(row["trace"])==1


def test_teacher_abstains_at_exhausted_budget_without_positive_endpoints():
    ff=frames();cfg={**config(),"max_turns":1}
    teacher=AnnotationTeacher(ff,64.,[[60.,64.]],cfg,"backtracking")
    row=rollout("event",64.,ff,"backtracking",(0,),cfg,SearchPolicy(teacher,ff,64.))
    assert row["status"]=="abstained" and len(row["trace"])==1


def test_teacher_rejects_invalid_labels():
    with pytest.raises(ValueError,match="training labels"):
        AnnotationTeacher(frames(),64.,[[60.,65.]],config(),"backtracking")


def test_selection_uses_source_identity_not_labels_or_query_scores():
    from scripts.prepare_agent_sft import select
    cfg={"selection_seed":"test","train_sources":2,"dev_sources":1,"confirmation_sources":1}
    rows=[{"split":split,"entry":{"vid":sid+"_0_64","qid":i,"relevant_windows":[[0,1]]}}
          for i,(split,sid) in enumerate([("train","a"),("train","b"),
                                         ("validation","c"),("validation","d")])]
    a=select(rows,cfg,set())
    for r in rows:r["entry"]["relevant_windows"]=[[20,40]]
    b=select(rows,cfg,set())
    assert {k:[r["entry"]["qid"] for r in v] for k,v in a.items()} == {
        k:[r["entry"]["qid"] for r in v] for k,v in b.items()}
    with pytest.raises(ValueError,match="inventory"):
        select(rows,cfg,{"a"})


def test_only_assistant_tokens_receive_loss_and_selected_logits_preserve_gradient():
    import torch
    import torch.nn.functional as F
    from scripts.train_agent_sft import mask_prefix_labels,supervised_positions
    ids=torch.tensor([[2,3,4,5,6,7]])
    labels=mask_prefix_labels(ids,ids[:,:3])
    assert labels.tolist()==[[-100,-100,-100,5,6,7]]
    torch.manual_seed(9)
    full=torch.randn(1,6,11,requires_grad=True)
    reference=F.cross_entropy(full[:,:-1].reshape(-1,11),labels[:,1:].reshape(-1))
    reference.backward();grad=full.grad.clone();full.grad=None
    positions=supervised_positions(labels)
    optimized=F.cross_entropy(full[:,positions-1].reshape(-1,11),labels[:,positions].reshape(-1))
    optimized.backward()
    assert torch.allclose(reference,optimized)
    assert torch.allclose(grad,full.grad)
    assert not torch.any(full.grad[:,:2])


def test_assistant_boundary_mismatch_is_not_silently_truncated():
    import torch
    from scripts.train_agent_sft import mask_prefix_labels
    with pytest.raises(ValueError,match="boundary"):
        mask_prefix_labels(torch.tensor([[1,2,3]]),torch.tensor([[1,4]]))


def test_checkpoint_selection_uses_only_dev_r1_then_iou_then_earlier_epoch():
    from scripts.run_agent_sft_stages import checkpoint_key
    m={"groups":{"backtracking/hash_half":{"n":2,"r1_at_05":.5,"top1_iou":.4},
                 "backtracking/mirrored_half":{"n":2,"r1_at_05":1.,"top1_iou":.8}}}
    assert checkpoint_key(m,1)==(.75,.6000000000000001,-1)
    assert checkpoint_key(m,1)>checkpoint_key(m,2)


def test_sft_replay_checks_complete_grid_and_rejects_duplicate(tmp_path,monkeypatch):
    from scripts import eval_agent_sft as ev
    from scripts.gpu_agent_experiment import MockBackend,score,sha
    from queryseek.agent.search_policy import initial_branch
    ff=frames();cfg={**config(),"bootstrap_seed":12,"bootstrap_replicates":20}
    sample={"qid":"dev_1","source":"s","stage":"dev","video_id":"s_0_64",
            "query":"event","duration_sec":64.,"gt_windows":[[10.,50.]],
            "frames":[f.model_dump() for f in ff]}
    package=tmp_path/"package";package.mkdir()
    out=tmp_path/"eval";out.mkdir()
    (package/"manifest.json").write_text(json.dumps({"samples":[sample]}))
    protocol={"stage":"dev","config":cfg,"manifest_sha256":sha(package/"manifest.json")}
    (out/"protocol.json").write_text(json.dumps(protocol))
    path=initial_branch(sample["video_id"],sample["query"]);rows=[]
    for method,condition,initial in [
        ("fixed_uniform","global",()),("forward_only","hash_half",path),
        ("backtracking","hash_half",path),("forward_only","mirrored_half",(1-path[0],)),
        ("backtracking","mirrored_half",(1-path[0],))]:
        backend=MockBackend()
        row=rollout("event",64.,ff,method,initial,cfg,SearchPolicy(backend,ff,64.))
        row.update(condition=condition,generation_stats=backend.calls)
        rows.append(score(row,sample,initial))
    (out/"records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out/"metrics.json").write_text(json.dumps(ev.metrics(rows,cfg)))
    monkeypatch.setattr(ev,"within_outputs",lambda p:p)
    assert ev.replay(package,out)["records"]==5
    with (out/"records.jsonl").open("a") as handle:handle.write("\n"+json.dumps(rows[0]))
    with pytest.raises(ValueError,match="Duplicate"):ev.replay(package,out)


def test_sft_contrast_clusters_starts_and_preserves_failures():
    from scripts.compare_agent_sft import compare_rows
    import copy
    base=[]
    for source in ["a","b"]:
        for condition in ["hash_half","mirrored_half"]:
            base.append({"qid":source,"source":source,"method":"backtracking","condition":condition,
                "gt_windows":[[0,4]],"initial_window":[4,8],"recovery_eligible":True,
                "top1_iou":0.,"r1_at_05":0,"r1_at_07":0,"status":"error","unique_frames":4,
                "logical_image_presentations":4,"trace":[],"generation_stats":[]})
    trained=copy.deepcopy(base)
    trained[0].update(top1_iou=1.,r1_at_05=1,r1_at_07=1,status="answered")
    result=compare_rows(base,trained,{"bootstrap_seed":12,"bootstrap_replicates":100})
    assert result["sft_minus_base_source_average"]["backtracking"]["sources"]==2
    assert result["sft_minus_base_source_average"]["backtracking"]["r1_at_05"]["mean"]==.25
    assert result["groups"]["sft"]["backtracking"]["statuses"]=={"answered":1,"error":3}
    with pytest.raises(ValueError,match="unmatched"):
        compare_rows(base,trained[:-1],{"bootstrap_seed":12,"bootstrap_replicates":100})
