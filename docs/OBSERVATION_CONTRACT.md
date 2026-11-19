# 观察状态与预算计量 v1

2026-11-19：补齐 P0 成本与来源契约，为 P1 局部辨识和预算曲线准备统一接口。本批没有训练或实现新学习控制器，没有新增真实模型准确率结果。前端布局保留，仅接入统计和记录下载。

## 一次任务记录什么

工作台的直接检索和 LangGraph 节点共享 `ObservationSession`。`costs.json` 保存预算、逻辑调用、每次 HTTP 尝试和汇总；`observations.json` 保存追加式来源记录，成功任务的 `result.json.observation_records` 包含同一快照。观察后失败也保留已发生的成本和记录。旧任务不回填数据。

| 字段 | 含义 |
|---|---|
| `unique_source_frames` | 已尝试提交的不同源画面；身份为视频 SHA256 + 源时间（六位小数，与现有 FFmpeg 接口精度一致），不是 frame ID 或像素哈希 |
| `unique_observed_source_frames` | 成功取得可解析模型结果的不同源画面；不意味着内容判断正确 |
| `image_presentations` | 图片累计提交次数；同帧重看、HTTP 重试均重复计入。连接失败仍计为尝试提交，不声称后端实际处理 |
| `model_calls` | 已准入的适配器调用；分批检查分别计数，文本核验也计数 |
| `http_attempts` | 真实 HTTP 尝试，包含重试；Mock 为 0 |
| `input_tokens / output_tokens / total_tokens` | 后端逐次报告的用量；缺失、非法或有尝试未报告时汇总为 `null`，逐次已知值仍保留 |
| `visual_input_tokens / text_input_tokens` | 仅后端明确提供图像 token 拆分时填写；不按帧数估算 |
| `preparation_sec / query_elapsed_sec / elapsed_sec` | 准备、查询和整个工作台任务耗时；查询计时从索引准备完毕开始 |
| `peak_gpu_memory_bytes` | 目前未测，`null` |

Mock 不进行模型推理，HTTP 和 token 为 0，仅用于验证流程。未接入传输计量的自定义适配器会令 `transport_accounting_complete=false`、HTTP/token 为未知；其图片数只是适配器输入数。当前工厂支持的真实路径是 `CompatibleAPIVLM`。

现有 `verify_window` 只发送文本证据，因此记录为 `verify_window_text`，不计图片；`verify_span` 才发送图片。旧结果的 `vlm_calls`、`used_frame_ids` 为兼容字段，不能替代新汇总。

成本范围是 `vlm_only`。全片 CLIP 编码、索引读取、解码分项和显存仍需独立测量；原索引帧数/在线编码/缓存命中字段保留，不能拿 VLM 帧数充当全部视觉计算成本。没有价格表或账单时不计算费用。

## 能约束哪些预算

`Workspace.execute(..., budget=16, budget_options={...})` 保存有效预算，并纳入请求幂等判断。修改预算后必须使用新的请求 ID。

- `max_unique_frames`：源画面上限，与现有 `budget` 一致；相同源时间的不同 ID 合并计数。
- `max_model_calls`、`max_http_attempts`：调用与传输尝试分别限制；重试前检查。
- `max_image_presentations`：可选累计图片提交上限，含重试。
- `max_output_tokens_per_call`：传给后端的 `max_tokens` 上限，默认沿用配置的 512；不是总 token 上限。
- `query_deadline_sec`：禁止超时后发起新调用/重试；不强行中断在途 HTTP 请求，也不限制索引准备时间。
- `max_total_tokens`：当前真实后端没有精确图片 token 预估器。指定严格总上限会在请求前以 `token_preflight_unavailable` 拒绝；不能靠事后统计保证上限。Mock 无 token 消耗。

拒绝原因进入成本记录。现有检索流程可能返回已有候选/回退结果，同时标记 `budget_exhausted`；这不等于语义定位成功。16 帧与 512 输出 token 仍是原型默认值，尚无实验依据认定最优。

## 来源标签与假设更新

每条记录包含 `source`、`kind`、`status`、创建轮次、实际已观察帧引用、可选模型与分数类型。仅程序读取的元信息可标为 `program / metadata / measured`。来源包括视觉模型、纯文本模型、Mock、字幕、检索和程序；来源不等于事实性。

视觉输出记为 `model_visual / observation / observed`，含义是“记录了模型对这些图片的输出”。核验即使返回 `confirmed` 也仍为 `tentative`；模型自报分数标为 `model_self_report`，不当成正确概率。字幕只是字幕记录，检索候选只是 `retrieval_hint`，不自动登记为模型已经看过的图片。

`ObservationState` 支持 `hypothesis` 和 `gap`。假设更新通过 `revise()`，要求后续轮次新观察到的源画面；旧画面别名、重复引用、此前已看但未引用的图片不能单独支持升级。更新追加带 `supersedes` 的记录，旧记录保留，禁止倒填未来证据。`supported/refuted` 也只是证据支持状态，所有记录的 `semantic_verified=false`。

本批实现状态接口和现有输出的来源记录，尚未增加猜测语句生成提示词、假设评分实验或新动作选择器；这些需在 P1/P2 单独验证。

## 已执行的离线预算矩阵

使用 `scripts/make_synthetic_fixture.py` 新生成 60 秒素材，uniform/adaptive × 8/16/32/64 帧 × 每批 4/8 帧 × 输出上限 128/512，共 **32/32 任务完成**。每任务最多 8 次调用；全部遵守源画面与调用上限，无真实 HTTP 或 token 消耗。完整 pytest：**273 passed / 6 skipped**；仅含本轮发布文件与原 Git 基线的独立副本专项：**77 passed**（观察/矩阵/工作台/模型/Agent/页面），不依赖未提交的重组文件。

| 方法 | 帧预算 | 每批帧数 | 不同源画面 | 累计图片提交 | 模型调用 |
|---|---:|---:|---:|---:|---:|
| uniform | 16 | 4 | 16 | 16 | 4 |
| uniform | 16 | 8 | 16 | 16 | 2 |
| adaptive | 16 | 4 | 14 | 16 | 4 |
| adaptive | 32 | 4 | 20 | 24 | 6 |
| uniform | 64 | 8 | 30 | 30 | 4 |

表格为 512 上限的工程示例；128 上限下也通过预算检查。Mock 不模拟生成截断，不能据此比较 token 上限对质量的影响。adaptive 的重复观察解释了不同源画面为何少于累计提交。素材粗索引只有 30 帧，所以 64 帧预算不意味着使用 64 帧。**准确率、最优预算与真实延迟收益：TODO / 未测。**

本机原始产物（不提交 Git）：`outputs/cost_contract_job_20261119_090453_5a10c6/runs/observation_probes/job_20261119_090454_9eb72a/`，包含 `protocol.json`、`results.json`；各任务目录另有完整成本/来源记录。协议保存视频/SRT/Mock 提示摘要、配置、模型 ID、源码摘要及依赖版本；真实服务模型 revision 未知时为 `null`。摘要不替代完整冻结源码。

复跑示例（使用新的输出根目录，避免覆盖现有索引）：

```powershell
$env:QUERYSEEK_LOAD_DOTENV = '0'
$env:QUERYSEEK_OUTPUT_ROOT = 'outputs/probe_next/runs'
$env:QUERYSEEK_CACHE_ROOT = 'outputs/probe_next/cache'
$env:QUERYSEEK_DATA_ROOT = 'outputs/probe_next/data'
.venv/Scripts/python.exe scripts/make_synthetic_fixture.py --out-dir outputs/probe_next/fixture
.venv/Scripts/python.exe scripts/observation_probe.py --video outputs/probe_next/fixture/sample.mp4 --srt outputs/probe_next/fixture/sample.srt --query '寻找讲解系统架构的片段' --mock --budgets 8,16,32,64 --frames-per-call 4,8 --output-tokens 128,512 --max-calls 8
```

运行器仅用于显式输入的单视频成本矩阵，采用直接检索模式，不读取历史确认集或计算定位准确率；Mock ingest 会使用合成素材自带的提示标签。它不是 P1 完整公平对照协议，真实实验还需冻结新开发来源、查询/标注、同一骨干及图片规格，补齐全片检索成本和人工语义核查。
