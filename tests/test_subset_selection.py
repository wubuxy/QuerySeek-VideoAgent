from scripts.make_qvhighlights_subset import select_entries, source_video_id


def test_fresh_subset_excludes_other_clips_of_old_source():
    entries = [
        {"qid": 1, "vid": "old_source_0.0_150.0"},
        {"qid": 2, "vid": "old_source_150.0_300.0"},
        {"qid": 3, "vid": "new_source_0.0_150.0"},
        {"qid": 4, "vid": "new_source_150.0_300.0"},
        {"qid": 5, "vid": "another_source_0.0_150.0"},
    ]
    selected = select_entries(entries, 10, 9, ["old_source_300.0_450.0"], True)
    identities = [source_video_id(e["vid"]) for e in selected]
    assert len(selected) == 2 and len(set(identities)) == 2
    assert "old_source" not in identities
    assert selected == select_entries(entries, 10, 9, ["old_source_300.0_450.0"], True)


def test_plan_only_never_downloads(tmp_path, monkeypatch):
    import json
    from scripts import make_qvhighlights_subset as subset
    gt = tmp_path / "gt.jsonl"
    gt.write_text(json.dumps({"qid": 10, "vid": "video_0.0_150.0", "query": "event", "relevant_windows": [[1, 5]]}), encoding="utf-8")
    monkeypatch.setattr(subset, "_download_hf", lambda *a: (_ for _ in ()).throw(AssertionError("No download")))
    out = tmp_path / "plan"
    monkeypatch.setattr(subset.sys, "argv", ["subset", "--gt", str(gt), "--out-dir", str(out), "--n", "1", "--plan-only", "--source-disjoint"])
    subset.main()
    assert json.loads((out / "selection.json").read_text(encoding="utf-8"))["entries"][0]["qid"] == 10
    assert not (out / "dataset.json").exists()
