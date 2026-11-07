# Third-party components and attribution

## 2026-11-07 查询表示与证据绑定回答

沿用已记录版本/许可证的 OpenCLIP 3.3.0（MIT，https://github.com/mlfoundations/open_clip）、Qwen3-VL-8B-Instruct、Transformers 4.57.6、bitsandbytes 0.46.1 与 lm-format-enforcer；未复制或修改上游代码、未增加依赖或模型下载。直接使用 OpenCLIP 官方 tokenizer 分块、官方文本编码和 Transformers 原生 video/约束生成接口。新增部分为问题表示、候选 ID 绑定、弃答协议和评测胶水；许可证与版权头位置沿用下方组件记录。详见 `docs/EVAL_QUERY_READER_V1.md`。

## 2026-11-07 问题引导补看

沿用已记录版本的FFmpeg、OpenCLIP ViT-B-32/openai、VTS四个MIT边界函数、Qwen3-VL-8B/NF4及Transformers原生video接口；无新第三方代码拷贝、模型下载、依赖安装或上游修改。新增部分仅为区域融合排序、观察预算/上下文分配及评测胶水；候选定位复用本项目已有E4曲线。TFVTG与FV-Action仅作为诊断设计依据，未复制代码或宣称复现其方法。来源链接、适用边界见 `docs/EVAL_QUERY_REFINEMENT_V1.md`。

## 2026-11-07 视觉变化时间树

VTS https://github.com/CeeZh/VTS 固定commit `184310c5eec2193060a5c95bbda4ad443dae4e25`，MIT，Copyright (c) 2026 CeeZh。从 `infer/utils/segmentation.py`（SHA256 `d8a2bd7c4f54be62a2b2fb079bd87fabdb394dea90a6441477209e017c1c2dac`）抽取 `_compute_frame_deltas`、`_detect_boundaries_topk`、`_detect_boundaries_threshold`、`_boundaries_to_segments` 四函数至 `queryseek/vendor/vts/boundaries.py`，函数体未改；删除其余解码/服务依赖和未用功能，新增来源模块头。完整版权与MIT许可在同目录LICENSE，随wheel分发。
媒体抽帧继续使用现有FFmpeg真实源PTS实现，OpenCLIP 3.3.0（MIT，https://github.com/mlfoundations/open_clip）直接调用，无上游修改；新时间节点/导航/哈希缓存为本项目胶水。CLIP模型复用此前本机缓存的 `timm/vit_base_patch32_clip_224.openai` revision `a6f597a30f7b82c51704746581f9a4e41421e878` 的OpenAI CLIP权重（MIT），SHA256 `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`，不分发权重；显式QuickGELU。timm1.0.30（Apache-2.0，https://github.com/huggingface/pytorch-image-models）、ftfy6.3.1（Apache-2.0，https://github.com/rspeer/python-ftfy）为OpenCLIP依赖，直接安装/调用、不修改；包内许可证在各自dist-info/licenses。NumPy/Pillow/Torch复用，版本写入运行清单。
算法适配差异：实际源PTS、bin中心采样、本项目编码器、独立缓存与有限节点预算；不称完整VTS复现或论文成绩。详细协议见 `docs/EVAL_SCENE_TREE_V1.md`。
服务器额外补齐ftfy所需wcwidth0.2.14（MIT，https://github.com/jquast/wcwidth），未修改，许可证位于对应dist-info。安装后pip check通过；Torch2.5.1/torchvision0.20.1保持原版本。

Project code is MIT; upstream code, datasets and weights retain their own terms. No videos, official annotations, model weights or paper PDFs are included in this release.

## Vendored source

**UniVTG** — https://github.com/showlab/UniVTG — commit `32659ac7aeba21742a63274f30eba785fc57e247` — MIT, Copyright (c) 2023 Kevin Lin.

- Source modules: `model/univtg.py`, `model/transformer_encoder_droppath.py`, `model/position_encoding.py`, `model/matcher.py`, `utils/span_utils.py`.
- Included under `queryseek/vendor/univtg/`; original license: `queryseek/vendor/univtg/LICENSE`; source/copyright attribution appears at each vendored Python file's beginning.
- Changes: package-relative import paths and provenance headers. Project adapters and masking fixes are separate modules under `queryseek/retrieval/`.
- Usage: optional frozen-feature temporal localization studies, including the original model/loss components. No released localization weights are included; the CLIP-only adaptation is not a full UniVTG reproduction.

## Installed dependencies (called through APIs or subprocesses, no upstream modifications)

Versions below describe recorded environments, not a promise of the newest version. Public CPU verification versions are separately captured in `results/release_environment.json`. Installed distributions retain their own LICENSE / COPYING / NOTICE files and copyright headers; they are not vendored here.

| Component / URL | Recorded version | License | Usage and original license location |
|---|---|---|---|
| [FFmpeg / ffprobe](https://ffmpeg.org/) | 9.0 full Windows build | LGPL/GPL depending on build | Probe/frame/clip subprocesses; original distribution license and build configuration |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | 0.7.1 | BSD-3-Clause | Scene boundaries; package LICENSE |
| [OpenCV](https://opencv.org/) | 5.0.0.93 | Apache-2.0 | Scene detection backend; distribution license |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | 0.2.2 | Apache-2.0 | Subtitle retrieval; distribution LICENSE |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | 3.3.0 | MIT | Embeddings; distribution LICENSE; pretrained weights have separate terms |
| [NumPy](https://numpy.org/) | CPU 2.5.3 / GPU experiment 2.2.6 | BSD-3-Clause and bundled notices | Numeric operations; numpy dist-info LICENSE |
| [PyTorch](https://pytorch.org/) | CPU 2.14.1 / GPU experiment 2.5.1+cu121 | BSD-3-Clause and bundled notices | Models/training; torch dist-info LICENSE |
| [torchvision](https://github.com/pytorch/vision) | CPU 0.29.1 / GPU experiment 0.20.1+cu121 | BSD-3-Clause | Preprocessing; torchvision LICENSE |
| [scikit-learn](https://scikit-learn.org/) | 1.9.1 | BSD-3-Clause | Candidate rankers; sklearn distribution LICENSE |
| [SciPy](https://scipy.org/) | 1.18.1 | BSD-3-Clause | Optional matching/studies; scipy LICENSE |
| [joblib](https://joblib.readthedocs.io/) | 1.6.0 | BSD-3-Clause | Research model persistence; joblib LICENSE |
| [threadpoolctl](https://github.com/joblib/threadpoolctl) | 3.7.0 | BSD-3-Clause | Thread limits; distribution LICENSE |
| [Pydantic](https://github.com/pydantic/pydantic) | CPU 2.14.0 / GPU experiment 2.14.1 | MIT | Time/schema validation; distribution LICENSE |
| [Typer](https://typer.tiangolo.com/) | 0.27.3 | MIT | CLI; distribution LICENSE |
| [Rich](https://github.com/Textualize/rich) | 15.0.0 | MIT | Console formatting; distribution LICENSE |
| [PyYAML](https://pyyaml.org/) | 6.0.3 | MIT | Configuration; distribution LICENSE |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 1.2.4 | BSD-3-Clause | Optional environment loading; distribution LICENSE |
| [srt](https://github.com/cdown/srt) | 3.5.3 | MIT | Subtitle parsing; distribution LICENSE |
| [HTTPX](https://github.com/encode/httpx) | 0.28.1 | BSD-3-Clause | Compatible visual API transport; distribution LICENSE |
| [LangGraph](https://github.com/langchain-ai/langgraph) | See release environment | MIT | State graph; package LICENSE |
| [LangGraph SQLite checkpointer](https://github.com/langchain-ai/langgraph) | See release environment | MIT | State persistence; package LICENSE |
| [Streamlit](https://github.com/streamlit/streamlit) | 1.65.0 | Apache-2.0 | Demo UI; distribution LICENSE |
| [pytest / pytest-cov](https://docs.pytest.org/) | 9.1.1 / 7.1.0 | MIT | Tests and coverage; distribution LICENSE |
| [Pillow](https://github.com/python-pillow/Pillow) | 12.3.0 | MIT-CMU | Image loading; pillow dist-info licenses/LICENSE |
| [Transformers](https://github.com/huggingface/transformers) | GPU experiment 4.57.6 | Apache-2.0 | Qwen model/processor/Trainer; dist-info licenses/LICENSE |
| [Accelerate](https://github.com/huggingface/accelerate) | GPU experiment 1.12.0 | Apache-2.0 | Model loading; dist-info licenses/LICENSE |
| [PEFT](https://github.com/huggingface/peft) | GPU experiment 0.17.1 | Apache-2.0 | QLoRA; peft dist-info licenses/LICENSE |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | GPU experiment 0.46.1 | MIT | NF4 quantization; dist-info licenses/LICENSE and NOTICE.md |
| [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) | GPU experiment 0.11.3 | MIT | Valid action decoding; distribution LICENSE |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | CPU 2.2.0 / GPU experiment 0.36.2 | Apache-2.0 | Versioned model assets; distribution LICENSE |
| [safetensors](https://github.com/huggingface/safetensors) | GPU experiment 0.8.0 | Apache-2.0 | Checkpoints; distribution LICENSE |
| [Matplotlib](https://matplotlib.org/) | Recorded plot environment 3.10.7 | Matplotlib License | Optional result figures; distribution LICENSE |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Optional, >=1.0 | MIT | Optional CPU ASR; package LICENSE |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Optional; not required by Mock | Unlicense (bundled components separately licensed) | User-authorized dataset acquisition; upstream LICENSE |

## Model and dataset boundaries

- **Qwen3-VL-8B-Instruct**: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Model card declares Apache-2.0. Used as a frozen base with language-layer adapters in the recorded experiment. Neither base nor adapter is distributed here.
- **OpenAI CLIP ViT-B/32**: https://github.com/openai/CLIP (MIT code/model usage record); loaded through OpenCLIP with matching QuickGELU configuration. No weights/tokenizer assets are bundled.
- **QVHighlights / Moment-DETR**: https://github.com/jayleicn/moment_detr, commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`. Code MIT; annotations CC BY-NC-SA 4.0; source-video terms separate. `queryseek/eval/qvhighlights.py` independently implements compatible temporal metrics; no official evaluation source is copied. Empty predictions explicitly score zero. No benchmark annotations/media are distributed.
- Historical research download scripts also reference external mirrors / VideoCC metadata. Their paths do not grant permission to redistribute content; media and metadata must be obtained under the original terms. These assets are not included in this repository.

## Papers and design references (not copied implementations)

- **VideoTree**: https://github.com/Ziyang412/VideoTree, historical reference commit `605ec8e917f9a0f95d37a73a0b50bf0e8cfe076e`, MIT. Hierarchical/query-adaptive observation reference; no source copied.
- **VideoMind**: https://github.com/yeliudev/VideoMind, historical reference commit `6721d2564eeb9ea8ab6c0ab8eaa39bd231ee8927`, BSD-3-Clause. Role/verification reference; no Chain-of-LoRA implementation copied.
- **VTS**: https://github.com/CeeZh/VTS, reference commit `184310c5eec2193060a5c95bbda4ad443dae4e25`, MIT. Navigation/detour supervision reference; no official code or checkpoint bundled. Code license does not determine checkpoint or source-video permissions.
- **VideoAgent**: https://arxiv.org/abs/2403.10517. Read for the future roadmap; no source or assets copied.

## Grounded QA metric references (2026-11-06)

- CG-Bench: https://github.com/CG-Bench/CG-Bench, commit `f826608752443a13b21f214c2d9c4156c526f030`. Read `run/stat_with_key.py` for the acc@IoU definition; no code copied or modified. The code repository has no separate license file in the inspected tree. The dataset card declares MIT with separate research-use terms for source videos. No benchmark media or annotations are distributed here.
- VLMEvalKit: https://github.com/open-compass/VLMEvalKit, commit `3c192143007e4eb3f5c1a36558a4f8c1451980cb`, Apache-2.0; original notice in upstream `LICENSE`. Read the interval-set IoU definition and threshold convention in `vlmeval/dataset/utils/cgbench.py`; no upstream code copied or executed. Our evaluation glue uses strict seconds/JSON and keeps failed items in its denominator. See `docs/EVAL_GROUNDED_QA_R1.md`.

Upstream source additions require an updated URL, revision, license, modification description and preserved copyright/license location before publication.

## 2026-11-07 原生视频输入诊断

复用 Transformers 4.57.6（https://github.com/huggingface/transformers/tree/v4.57.6 ，Apache-2.0）的 Qwen3VLProcessor、Qwen3VLVideoProcessor 与 VideoMetadata；用于已抽帧视频的时域合并、像素处理和时间标签。上游源码未修改/未复制；版权头位于包内 `models/qwen3_vl/processing_qwen3_vl.py`、`video_processing_qwen3_vl.py`，许可证在对应 dist-info/licenses/LICENSE。仍复用原 FFmpeg、Pillow、NumPy、Torch 与约束解码库，版本及许可沿用本文件。源PTS的内部整数时间基适配、逐输入检查和实验运行器为本项目胶水实现。
