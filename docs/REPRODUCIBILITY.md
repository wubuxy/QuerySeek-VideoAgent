# 复现范围

## 1. 可以直接验证

从 README 的安装与 Mock 命令开始。视频 fixture 统一由 `scripts/make_synthetic_fixture.py` 生成；测试不需要真实 API Key。
工程测试使用临时目录；发布核心 CI 显式禁用 dotenv。完整研究测试需要额外 torch/scipy/scikit-learn 等依赖，运行范围与结果见 RELEASE_VALIDATION.md。
历史环境版本记录在 `results/release_environment.json`；它是实际发布验证环境快照，不是跨平台锁文件。

## 2. 需要自行准备才能运行

真实 E4 索引需要 OpenCLIP 权重及合法获得的视频。SFT/定位研究脚本还需要相应数据、来源划分、输入 manifest 和固定模型 revision。
通用研究配置保留原始相对路径用于说明历史协议；不存在的 `outputs/...` 不会由公开仓库自动生成。历史研究工具不是一键训练套件。
数据准备与模型下载须单独审阅，运行 Mock 不会触发正式 SFT。

`prepare_agent_sft.py` 依赖原冻结实验库存与排除来源，故单有聚合结果无法重建原始 544 来源输入包。
自行获得数据后可依代码建立新的来源隔离实验；样本选择或模型配置变化时应报告为新实验，不宣称精确重现归档成绩。

## 3. 公开来源核对

`results/provenance.json` 记录直接复制源码的 SHA256、聚合指标文件 SHA256、基座 revision 和历史 adapter SHA256。
`results/publication_manifest.json` 记录本次发布文件 SHA256（不包含其自身），可用于核对公开快照。
聚合结果未经新推理重算；完整逐任务/图像/训练日志保留在本地，未公开。

## 4. 有意排除的内容

不发布原始视频、真实抽帧、官方标注、模型/适配器、个人运行记录、凭据和论文 PDF。
病例复核配置及其三份依赖测试从公开快照排除，清单见 provenance；其他历史研究工具可能引用这些未公开文件。
只有发布中实际通过的测试计入发布验证，不把排除用例称作通过。

## 5. 视觉变化时间树

`python -m pytest tests/test_temporal_tree.py tests/test_frame_stdin_isolation.py -q` 可离线验证合成视频、真实FFmpeg、缓存和导航。真实构建需OpenCLIP权重与已探测的无标签请求；`scripts/probe_scene_tree.py build` 和 `evaluate` 分离。
`configs/scene_tree_probe_v1.json` 保留归档输入哈希；公开包不分发其原始8来源输入。新数据须另存新配置和哈希。`results/scene_tree_probe_20261107.json` 给出聚合指标、源码/归档SHA、成本口径与本机重放验证；不宣称完整VTS复现。

## 6. 问题引导局部补看

`python -m pytest tests/test_query_refinement.py -q` 使用合成视频跑通检索→独立oracle准备→Mock推理，真实指标入口拒绝Mock。`scripts/probe_query_refinement.py` 提供prepare/context/run/evaluate分离入口；CLIP缓存身份、请求、标签、准备产物和源码均按冻结哈希检查。
真实运行前还需用 `scripts.run_grounded_qa_baseline.verify_model_snapshot` 核验本地Qwen模型快照；该校验已在归档执行前完成。数据/模型不分发，原始8来源配置不是公开下载入口。更换输入须另存配置。协议中的事前版本保留在源包哈希中，公开文档追加实测结果；见 [记录](EVAL_QUERY_REFINEMENT_V1.md)。

## 7. 查询表示与证据绑定回答

`python -m pytest tests/test_query_reader.py -q` 通过合成视频验证规划→oracle接口→检索→候选绑定Mock流程。`scripts/probe_query_reader.py` 的plan-oracle/prepare/answer/evaluate四阶段按配置、源码和输入哈希隔离；真实生成核对模型快照，评估入口拒绝Mock。本轮复用既有开发样本/帧缓存，不分发媒体、标签或模型。更换输入须另存配置；不能把Mock结果当准确率。事前协议原字节保留在source包，当前文档追加结果，见 [协议与结果](EVAL_QUERY_READER_V1.md)。

## 8. 候选内观察与严格证据契约

`python -m pytest tests/test_evidence_packets.py tests/test_evidence_contract.py tests/test_anchor_observations.py -q` 使用合成视频覆盖展示/引用一致、候选反序、三态控制、一次补看、旧联合对照复用和Mock/真实隔离。真实parser测试还需lm-format-enforcer==0.11.3，无需模型。两个研究入口分别为 `scripts/probe_evidence_packets.py` 与 `scripts/probe_evidence_contract.py`，后者保留v1记录，冻结旧准备/联合预测SHA并复用，记录新增核验成本。实际模型快照与源码逐SHA校验；更换数据必须另存配置。媒体、模型和私人复核页不分发，事前协议与实测结果分别留存。参见 [严格契约协议](EVAL_EVIDENCE_CONTRACT_V2.md)。

## 9. 按证据需求观察

`python -m pytest tests/test_evidence_needs.py -q` 使用统一合成fixture，无网络。真实parser需要lm-format-enforcer==0.11.3。研究入口 `scripts/probe_evidence_needs.py` 的run拒收标签，evaluate另读冻结标签；需已获许可的原数据、固定准备SHA及本地模型，详见 [协议](EVAL_EVIDENCE_NEEDS_V1.md)。裁切复用Pillow，PNG图像及父图/源PTS/库版本共同记入缓存清单。

## 10. 固定证据的模型诊断

`python -m pytest tests/test_model_diagnostic.py -q` 无需GPU或网络，使用统一fixture；真实研究按 `scripts/probe_model_diagnostic.py` 的prepare、两精度run、evaluate分阶段执行，详见[冻结协议](EVAL_MODEL_DIAGNOSTIC_V1.md)。推理与标签评估分开，所有实际图像、张量与提示摘要留档，完整原始媒体只保存在私有环境。

复杂核验跟进入口为 `scripts/probe_verifier_diagnostic.py`，依赖已冻结首轮准备目录和配置，通过独立配置执行NF4自由解码/BF16受约束核验，evaluate重用首轮NF4结果。测试：`python -m pytest tests/test_verifier_diagnostic.py -q`。协议与原始意图统计边界见[跟进记录](EVAL_VERIFIER_DIAGNOSTIC_V1.md)。

## 11. 精简阅读与新来源

入口 `scripts/probe_compact_reader.py` 分prepare/run/evaluate，模型推理不读标签；准备使用真实PTS的oracle缓存，明确区别于在线检索。`queryseek.eval.compact_reader.select_sources`实现冻结元数据规则。运行 `python -m pytest tests/test_compact_reader.py -q` 可用统一合成fixture无网络验证。详见[实验协议](EVAL_COMPACT_READER_V1.md)。
