"""OpenCLIP 图文编码 Adapter（复用 open_clip，代码 MIT；权重许可按模型单独核对）。

- CPU 推理：帧已由 ingest 缩图到 max_frame_side_px，批量编码并缓存（说明书 §4）。
- 模型与权重下载发生在首次使用时（HF Hub）；失败时抛 EmbeddingUnavailableError，
  不阻塞 ingest 主流程。
- CLIP 文本塔对中文支持有限：中文查询按 UTF-8 字节 BPE 处理，效果弱于英文——
  这是已知限制（docs/DECISIONS.md），不假装中文语义完美。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """OpenCLIP 不可用（未安装 / 权重下载失败）。"""


class OpenCLIPEmbedding:
    """ViT-B-32（默认）图文编码。"""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 device: str = "cpu", max_side: int = 512,
                 force_quick_gelu: bool | None = None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.max_side = max_side
        self.force_quick_gelu = pretrained == "openai" if force_quick_gelu is None else force_quick_gelu
        self._tokenizer = None  # _load() 内赋值，必须先声明（防止覆盖）
        self._model, self._preprocess = self._load()
        self.dim = int(self._model.visual.output_dim)
        self.model_id = f"openclip:{model_name}:{pretrained}:activation={'quickgelu' if self.force_quick_gelu else 'model-default'}"

    def _load(self):
        try:
            import open_clip  # noqa: PLC0415
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "open-clip-torch 未安装：pip install -e '.[embedding]'，或使用 --mock"
            ) from exc
        try:
            import torch  # noqa: PLC0415

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=self.device,
                force_quick_gelu=self.force_quick_gelu,
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer(self.model_name)
            self._tokenizer = tokenizer
            return model, preprocess
        except Exception as exc:  # noqa: BLE001 — 权重下载/加载失败需明确报错
            raise EmbeddingUnavailableError(
                f"OpenCLIP 加载失败（模型 {self.model_name!r} / 权重 {self.pretrained!r}）: {exc}"
            ) from exc

    @staticmethod
    def is_available() -> bool:
        try:
            import open_clip  # noqa: PLC0415,F401
            import torch  # noqa: PLC0415,F401

            return True
        except ImportError:
            return False

    def _preprocess_batch(self, paths: list[str]):
        import torch  # noqa: PLC0415

        from PIL import Image

        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            if max(im.size) > self.max_side:
                im.thumbnail((self.max_side, self.max_side))
            imgs.append(self._preprocess(im))
        return torch.stack(imgs).to(self.device)

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch  # noqa: PLC0415

        toks = self._tokenizer(texts).to(self.device)
        with torch.inference_mode():
            feats = self._model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return self._norm(feats.float().cpu().numpy())

    def encode_images(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        import torch  # noqa: PLC0415

        out = []
        for i in range(0, len(paths), batch_size):
            batch = self._preprocess_batch(paths[i : i + batch_size])
            with torch.inference_mode():
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.float().cpu().numpy())
        return self._norm(np.concatenate(out, axis=0)) if out else np.zeros((0, self.dim))
