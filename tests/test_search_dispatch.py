from types import SimpleNamespace

import numpy as np
import pytest

from queryseek.config import Config
from queryseek.retrieval import dispatch
from queryseek.schemas import VideoFrame


@pytest.mark.parametrize("bad_metadata", ["model", "frame_order"])
def test_dispatch_rejects_mismatched_embedding_cache(monkeypatch, bad_metadata):
    cfg = Config.load()
    frames = [VideoFrame(frame_id="a", timestamp_sec=1, image_path="a.jpg"),
              VideoFrame(frame_id="b", timestamp_sec=2, image_path="b.jpg")]
    index = SimpleNamespace(frames=frames, cache_dir="unused")
    emb = SimpleNamespace(model_id="openclip:current:weights")
    meta = {"backend": "openclip", "model_id": emb.model_id, "frame_ids": ["a", "b"]}
    if bad_metadata == "model":
        meta["model_id"] = "openclip:old:weights"
    else:
        meta["frame_ids"] = ["b", "a"]
    monkeypatch.setattr(dispatch, "build_embedder", lambda *a, **k: emb)
    monkeypatch.setattr(dispatch, "load_embeddings", lambda *a: (np.zeros((2, 1)), meta))
    with pytest.raises(RuntimeError, match="重新 ingest"):
        dispatch.embedder_for(index, cfg, False)


def test_curve_dispatch_real_cache_without_api(monkeypatch):
    cfg = Config.load()
    frames = [VideoFrame(frame_id=str(i), timestamp_sec=i*4, image_path="unused") for i in range(5)]
    index = SimpleNamespace(frames=frames, asset=SimpleNamespace(duration_sec=20))
    emb = SimpleNamespace(model_id="test", encode_texts=lambda q: np.array([[1.]]))
    monkeypatch.setattr(dispatch, "embedder_for", lambda *a: (emb, np.array([[0], [.1], [.9], [.8], [0]])))
    monkeypatch.setattr(dispatch, "build_vlm", lambda *a, **k: pytest.fail("No API client should be created"))
    result = dispatch.search_index(cfg, index, "v", "event", "curve", mock=False)
    assert result.windows and result.embedding_cache_hit and result.vlm_calls == 0
