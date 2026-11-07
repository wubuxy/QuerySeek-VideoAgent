# 公开快照发布验证

日期：2026-11-06。本页原始发布验证只运行 CPU/Mock；随后完成的真实长视频开发试运行另见 [R1 记录](EVAL_GROUNDED_QA_R1.md)，不混入原始发布测试统计。

## 本地公开套件

在独立公开快照目录执行，使用现有本地 Python 环境，禁用 dotenv 与在线模型加载：

```bash
python -m pytest -q -p no:cacheprovider
```

实测：**427 passed / 6 skipped，134.21 秒**。

跳过原因：1 项需要固定本地 GPU 模型分词器；1 项 Windows symlink 权限不足；4 项需要未分发的官方 Moment-DETR 评测源码作对照。
不把这些跳过项称作本轮通过。原本地历史套件 450/2 与公开套件不同；具体病例审阅配置及依赖它们的三份测试文件未分发。

## 发布内容

保留源码、合成 fixture 生成器、可公开测试/通用配置、聚合结果与哈希。
运行生成的 `outputs/`、`data/`、合成视频不会提交。图片、原视频、标签、权重、密钥、论文 PDF 不在发布范围。
原始源码直接复制；公开包额外在 setuptools package-data 中包含 vendored UniVTG LICENSE，保持分发许可完整。

静态凭据扫描无命中；这是发布检查，不能解释为绝对安全保证。
README 的 Mock 命令另行执行并记录在 `results/mock_quickstart.json`；这些耗时仅描述合成样本，不能用作真实模型延迟。
GitHub Actions 仅运行声明的核心 Mock 子集；其远程状态以实际 Actions 页面为准。

首次远程检查发现发布 `.gitignore` 的 `models/` 也排除了 `queryseek/models/` 源码。
已将权重与运行目录规则限制为仓库根目录，补齐该源码包；本地源码与研究结果未变。
后续发布核对还检查源码清单是否全部入 Git，并从 Git 导出的干净目录验证核心流程。

## 2026-11-07 输入诊断增量

公开工作树相关测试28 passed / 1 skipped（3.99秒）：grounded QA评测、全片基线、输入诊断、帧提取时间与错误日志。唯一跳过项需要本地Qwen processor；服务器实际processor检查已在23项输入测试中通过，帧提取回归另6项通过。
这不是重新执行427项发布全套，也不是Mock准确率。64次真实推理的聚合记录、源码SHA及备份验证见 `results/grounded_qa_input_study_20261107.json`；媒体、标注和画面不分发。CI新增合成输入诊断与缺PTS时保留日志的回归。

## 2026-11-07 视觉变化时间树与完整回归

新增惰性多分支树、源PTS/特征缓存、独立导航工具和标签隔离结构检查；FFmpeg三个抽帧入口隔离stdin，服务器合成复现确认修复。相关协议和真实结果见 [时间树记录](EVAL_SCENE_TREE_V1.md)。
本次在独立公开工作树重新执行完整pytest：**464 passed / 7 skipped，143.56秒**。跳过为固定本地tokenizer、Qwen processor各1项，Windows symlink权限1项，未分发官方Moment-DETR源码对照4项；不把跳过称作通过。服务器相关52项通过（11.29秒），包括真实FFmpeg管道输入回归。CI增加时间树、缓存完整性、标签隔离、失败成本、导航和stdin回归。
VTS的4个MIT函数保留原函数体、来源与完整LICENSE，UniVTG/VTS许可证均纳入setuptools package-data。真实8来源结果完成备份及本机精确重放；无新VLM调用或训练。公开聚合见 `results/scene_tree_probe_20261107.json`。

## 2026-11-07 问题引导补看完整回归

公开工作树重新执行完整pytest：**473 passed / 7 skipped，129.39秒**；跳过原因同上轮。新增9项覆盖等预算、源PTS去重、候选选帧、真实合成FFmpeg、标签隔离、Mock/真实禁止混用、输入篡改、失败分母和联合评测。CI增加该9项。
服务器相关35 passed/1 skipped（15.94秒），随后设置本地Qwen processor执行输入专项5项全部通过（23.20秒），补验该跳过项。真实24组检索、32次生成全成功；242文件归档校验，24曲线和24组选帧、指标/成本均在本机精确重放。完整记录见 [问题引导补看](EVAL_QUERY_REFINEMENT_V1.md)。
