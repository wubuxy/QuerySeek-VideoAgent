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
