# 公开快照发布验证

日期：2026-11-06。无新训练、真实 VLM 调用或算法评测。

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
