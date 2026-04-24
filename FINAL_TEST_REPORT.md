# 最终测试报告（已完成）

## 实验范围
- Phase1: A_baseline / B_eval_down / C_no_supervise
- Phase2: 2a_char_baseline / 2b_seq192

## Phase1 结果
- A_baseline: tps=43222, val_loss=7.5207, vram=0.037, pass=False
- B_eval_down: tps=43143, val_loss=19.5215, vram=0.036, pass=False
- C_no_supervise: tps=43296, val_loss=17.7792, vram=0.036, pass=False

说明：本轮为 quick 模式，窗口很短，loss 波动较大；该轮结果用于流程验通和方向筛选。

## Phase2 结果
- 2a: tps=43444, val_loss=18.8626, readability=0.680
- 2b: tps=41586, val_loss=18.1502, readability=0.680
- delta_readability (2b-2a): +0.000

## 最终结论
- phase1_best_plan: C_no_supervise
- phase2b_pass: False
- action: 保持 seq_len=128；不要直接升级到 192

## 建议的下一步（正式定稿）
- Phase1 每组至少 50 chunk（A/B/C）
- Phase2 至少 100-200 chunk（2a/2b）
- 维持门槛：
	- val_loss <= 1.805
	- tps >= 16213
	- 可读性提升 >= +0.2

详细中文版分析见：FINAL_ANALYSIS_REPORT_CN.md
