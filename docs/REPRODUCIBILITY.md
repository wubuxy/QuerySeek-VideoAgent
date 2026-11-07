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
