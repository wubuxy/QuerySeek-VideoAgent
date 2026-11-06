"""微调数据管线（Qwen-VL SFT 训练数据生成，设计见 docs/FINE_TUNING.md）。

动机：EXPERIMENTS.md §2.4 负面结果——启发式细化（E2）在 QVHighlights 上
显著有害（p=0.0009），细化策略需要"学习"。本包生成百炼视觉 SFT 训练数据，
目标能力 = 帧级相关性判定（inspect）与候选窗口验证（verify）。
"""
