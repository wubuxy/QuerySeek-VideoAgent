# 保留触发窗口的原始观察帧（2026-11-07）

这是在两轮候选核验完成后发现的采样问题及工程修复。原核验记录不变；修复后的问答准确率 **TODO / 未测**。

## 问题

候选由已观察的高分帧触发，但64秒窗口重新按8个等分区间的中点抽帧，会避开窗口中心。实际20个seed窗口都没有保留触发帧，最近新帧与原帧相差约2.72–4秒。因此“围绕一个线索继续看”可能丢掉原线索本身。此审计只说明信息保留失败，不证明这些帧都包含正确答案。

## 修改与隔离

新增 `queryseek/retrieval/anchor_observations.py` 与 `scripts/repair_packet_anchors.py`：从带视频、索引和图片SHA的原观察缓存找回唯一触发帧，用它替换新窗口中距其最近的一帧，再按真实PTS排序。候选区间、数量与每候选8帧预算保持，其他候选不变。时间仍使用浮点秒和半开区间。

此操作不读答案标注、不重新抽帧、不编码、不调用模型。原v1/v2代码、输入和结果不覆盖，新准备文件使用独立目录及源码/输入/输出哈希清单。新文件不能直接冒充旧实验的准备清单；未来推理必须另立配置和预算。

## 核验

5项单测覆盖帧身份与预算保留、输入不变、幂等、错误触发时间/来源/哈希/窗口拒绝。实际缓存核验与备份结果见公开聚合记录中的 `anchor_retention`；该字段只记录准备层面的检查，准确率留空。

```bash
python -m pytest tests/test_anchor_observations.py -q
python scripts/repair_packet_anchors.py \
  --prepared outputs/evidence_contract_20261107/study/prepared/prepared.json \
  --retrieval outputs/query_refinement_20261107/retrieval/retrieval.jsonl \
  --config configs/evidence_contract_v2.json \
  --cache-root outputs/temporal_tree_frame_cache \
  --output outputs/evidence_anchor_repair_20261107/plan
```

真实复用要求原私有缓存和冻结SHA。输出目录必须尚不存在。现有人工复核页另展示原锚点，明确标注这些图没有用于已完成的v2核验；不能把人观看完整视频或原图的判断计入模型成绩。
