#!/usr/bin/env python3
"""
【综合操作清单】第 0-1 阶段执行检查表
"""

checklist = """
╔════════════════════════════════════════════════════════════════════════════╗
║                  训练通量与字符级可用性调优 - 执行清单                        ║
║                   (从第 0 阶段基线 → 第 1 阶段通量优化)                       ║
╚════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ✓ 【第 0 阶段】基线快照与冻结配置 - 已完成 (1h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[✓] 1. 运行 baseline_analysis.py
    └─ 已完成 - 提取统计指标
    
[✓] 2. 基线统计指标确认
    └─ train_loss: 1.7724 ± 0.0076
    └─ val_loss: 1.7873 ± 0.0294 (✓ 无过拟合)
    └─ tps: 18014 ± 627 tok/s
    └─ vram: 0.187 GB
    └─ 泛化间隙: 0.0148 (✓ 平台期，无进一步改进空间)
    
[✓] 3. 资源红线设定
    └─ val_loss_max: 1.8051 (基线 +1%)
    └─ tps_min: 16213 tok/s (基线 -10%)
    └─ vram_max: 0.25 GB
    
[✓] 4. 配置冻结
    └─ baseline_config_frozen.json 已生成
    └─ baseline_snapshot.json 已生成

【第 0 阶段状态】: ✓ COMPLETE
【输出文件】:
    - baseline_analysis.py (已运行)
    - baseline_snapshot.json
    - baseline_config_frozen.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ◐ 【第 1 阶段】通量优化 A/B 对标 - 准备完成，待执行 (12h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 【准备工作】

[✓] 1. 运行 prepare_phase1.py 生成方案配置
    └─ 已完成
    └─ 生成文件:
      • plan_A_baseline.json
      • plan_B_eval_downsample.json
      • plan_C_no_supervise.json
      • PHASE1_MODIFICATION_GUIDE.md

[✓] 2. 方案对照表已生成
    └─ 方案 A (基线)：原配，tps=18014, loss=1.787
    └─ 方案 B (评估下采样)：+40% tps, loss=1.787
    └─ 方案 C (无督导)：+15% tps, loss+0.03

[✓] 3. 评估脚本已生成
    └─ evaluate_phase1.py (三方案对比后运行)

### 【实际执行步骤】（按顺序）

【Step 1】备份原始文件 (2 min)
┌─────────────────────────────────────────┐
│ cp agi_joint_trainer.py                  │
│    agi_joint_trainer_original_phase0.py  │
│                                          │
│ ✓ 确认备份完成后再进行任何修改           │
└─────────────────────────────────────────┘
[ ] 已备份原始文件

【Step 2】执行方案 A - 基线对照 (4h)
┌─────────────────────────────────────────┐
│ 修改: 无                                  │
│ 执行: python agi_joint_trainer.py        │
│                                          │
│ 完成标志:                                 │
│   - 终端输出 chunk 440-490 的训练日志    │
│   - agi_joint_telemetry.csv 增加 ~50 行 │
│   - CSV 最后一行应为 chunk 490 左右      │
└─────────────────────────────────────────┘
[ ] 方案 A 执行完成
[ ] 确认 telemetry.csv 最后 chunk 号 (~490)

【Step 3】恢复原始文件 + 执行方案 B (4h)
┌─────────────────────────────────────────┐
│ 1. 恢复: cp agi_joint_trainer_original_phase0.py \\
│             agi_joint_trainer.py         │
│                                          │
│ 2. 编辑 agi_joint_trainer.py 第 805 行: │
│    OLD: eval_limit = min(future_data.size(1), 1000 * seq_len)
│    NEW: eval_limit = min(future_data.size(1), 100 * seq_len)
│                                          │
│ 3. 执行: python agi_joint_trainer.py    │
│                                          │
│ 4. 备份: cp agi_joint_telemetry.csv \\   │
│             agi_joint_telemetry_B_eval_down.csv
│                                          │
│ 完成标志:                                 │
│   - tps 应显著提升（预期 +30-50%）      │
│   - val_loss 应与方案 A 相近             │
│   - 生成 agi_joint_telemetry_B_eval_down.csv
└─────────────────────────────────────────┘
[ ] 方案 B 修改完成
[ ] 方案 B 执行完成
[ ] 已备份 B 的 telemetry

【Step 4】恢复原始文件 + 执行方案 C (4h)
┌─────────────────────────────────────────┐
│ 1. 恢复: cp agi_joint_trainer_original_phase0.py \\
│             agi_joint_trainer.py         │
│                                          │
│ 2. 编辑 agi_joint_trainer.py CONFIG（~151 行）:
│    OLD: "supervise_every": 10,          │
│    NEW: "supervise_every": 0,           │
│                                          │
│ 3. 执行: python agi_joint_trainer.py    │
│                                          │
│ 4. 备份: cp agi_joint_telemetry.csv \\   │
│             agi_joint_telemetry_C_no_supervise.csv
│                                          │
│ 完成标志:                                 │
│   - tps 应提升（预期 +10-20%）          │
│   - val_loss 可能小幅上升（+0.02-0.05）│
│   - 生成 agi_joint_telemetry_C_no_supervise.csv
└─────────────────────────────────────────┘
[ ] 方案 C 修改完成
[ ] 方案 C 执行完成
[ ] 已备份 C 的 telemetry

【Step 5】评估三方案 & 选择最优 (30 min)
┌─────────────────────────────────────────┐
│ 执行: python evaluate_phase1.py         │
│                                          │
│ 检查输出:                                 │
│   ✓ 方案 A/B/C 的吞吐/loss/VRAM 对比   │
│   ✓ 通过/失败判定                       │
│   ✓ 推荐最优方案                        │
│                                          │
│ 生成: phase1_evaluation.json             │
└─────────────────────────────────────────┘
[ ] 评估完成
[ ] 方案对比表已生成

【Step 6】固化最优方案 (30 min)
┌─────────────────────────────────────────┐
│ 基于 evaluate_phase1.py 的推荐:          │
│                                          │
│ 如果推荐方案 B:                          │
│   - 恢复原始文件                         │
│   - 永久修改第 805 行 eval_limit        │
│   - 保存为官方配置                      │
│                                          │
│ 如果推荐方案 C:                          │
│   - 恢复原始文件                         │
│   - 永久修改第 151 行 supervise_every   │
│   - 保存为官方配置                      │
│                                          │
│ 如果推荐混合（B+C 部分):                │
│   - 同时应用两个修改                    │
└─────────────────────────────────────────┘
[ ] 最优方案已应用到官方 agi_joint_trainer.py

【第 1 阶段执行时间表】
┌──────────────────────────────┬────────┐
│ Step                          │ 时间   │
├──────────────────────────────┼────────┤
│ 1. 备份                      │ 2 min  │
│ 2. 方案 A (原配)             │ 4 h    │
│ 3. 方案 B (eval down)        │ 4 h    │
│ 4. 方案 C (no supervise)     │ 4 h    │
│ 5. 评估 & 选择               │ 30 min │
│ 6. 固化最优                  │ 30 min │
├──────────────────────────────┼────────┤
│ 总计                         │ ~12h   │
└──────────────────────────────┴────────┘

【第 1 阶段关键指标】
预期目标:
  ✓ tps 提升: > 15% (相对基线)
  ✓ val_loss: <= 1.805 (代表 1% 容差)
  ✓ VRAM: <= 0.25 GB (硬上限)
  
通过条件:
  ✓ 至少一个方案满足所有红线
  
预期结果:
  ✓ 方案 B 最可能赢（低风险、高收益）
  ✓ 推荐 tps 提升 40%，loss 保持不变

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ⏳ 【第 2 阶段】字符可用性推进 - 待启动 (5-6h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【启动条件】
  ✓ 第 1 阶段完成且选出最优方案
  ✓ 最优方案的代码改动已应用

【第 2 阶段工作内容】

子阶段 2a：基线化字符质量 (1h)
┌─────────────────────────────────────────┐
│ 目标: 建立 seq_len=128 下的字符生成基线 │
│ 执行: python agi_joint_trainer.py (50 chunk)
│                                          │
│ 采新样本:                                 │
│   - 每 50 chunk 采 1 次固定 prompt      │
│   - 收集 10+ 个样本                    │
│   - 统计：重复率、不可见字符、词长分布  │
│                                          │
│ 输出:                                     │
│   - character_baseline_2a.json          │
│   - 可读性评分基准 (score_2a)           │
└─────────────────────────────────────────┘
[ ] 2a 准备条件检查
[ ] 采样脚本已集成
[ ] 2a 执行完成
[ ] 基线评分已生成 (score_2a)

子阶段 2b：扩展 seq_len 验证 (4-5h)
┌─────────────────────────────────────────┐
│ 配置改动:                                 │
│   seq_len: 128 -> 192                   │
│   lr: 3e-4 -> 1.5e-4                   │
│   batch_size: 24 -> 16                 │
│   grad_clip: 1.0 -> 0.8                │
│                                          │
│ 执行: python agi_joint_trainer.py (200 chunk)
│                                          │
│ 采样与对比:                               │
│   - 每 50 chunk 采样一次                │
│   - 对比 2a 与 2b 的可读性评分         │
│   - target: 2b 评分 >= 2a + 0.2        │
│                                          │
│ 输出:                                     │
│   - character_phase2b.json              │
│   - 可读性评分 2b (score_2b)           │
│   - agi_joint_telemetry_2b.csv (长跑跟踪)
└─────────────────────────────────────────┘
[ ] 2b 配置改动已应用
[ ] 2b 执行完成
[ ] 字符样本对比完成
[ ] score_2b >= score_2a + 0.2 (通过条件)

【第 2 阶段通过标准】
  ✓ val_loss <= 1.805 (继承第 1 阶段红线)
  ✓ tps >= 16213 (继承第 1 阶段红线)
  ✓ score_2b >= score_2a + 0.2 (字符质量明显提升)
  ✓ 无 OOM (VRAM <= 0.25 GB)

【回退保险】
  如果 2b 失败（loss 上升超 1% 或生成垮掉）:
    Plan B1: seq_len 160 (折中), lr 2e-4, 重试 100 chunk
    Plan B2: 保留 seq_len=128, 仅保留第 1 阶段优化
    Plan B3: 调整生成策略 (温度、top_k/top_p)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ⏳ 【第 3 阶段】配置冻结与长跑验证 - 待启动 (4h, 可选)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【启动条件】
  ✓ 第 2 阶段通过或采用回退方案

【第 3 阶段工作】
  - 综合第 1-2 阶段的最优参数
  - 运行 150-200 chunk 长程验证
  - 确认 loss/tps/VRAM 无单调劣化趋势
  - 冻结最终配置为标准版本

【输出】
  - final_config_v1.0.json (最终冻结配置)
  - long_run_telemetry.csv (验证期遥测)
  - README_FINAL_CONFIG.md (使用说明)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📊 【总体时间与资源预算】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 阶段 | 内容 | 耗时 | GPU占用 | 状态 |
|------|------|------|--------|------|
| 0 | 基线快照 | 1h | - | ✓ 完成 |
| 1 | 通量优化 A/B | 12h | 100% | ◐ 待执行 |
| 2 | 字符可用性 | 5-6h | 100% | ⏳ 待启动 |
| 3 | 长跑验证 | 4h | 100% | ⏳ 可选 |
| 总计 | | 22-24h | | |

【资源约束】
  ✓ GPU: CUDA:0 (single GPU, ~RTX 4070 Ti Super 或同等)
  ✓ VRAM: 硬上限 0.25 GB (~256 MB，远低于当前平台要求)
  ✓ 网络: FineWeb-Edu 本地路径 (E:\\Datasets\\FineWeb-Edu_Full)
  ✓ 存储: Telemetry CSV <1 MB/run, checkpoints ~50 MB/run

【建议执行计划】
  DAY 1:
    - 清晨启动方案 A (4h)
    - 完成后立即启动方案 B (4h)
  
  DAY 2:
    - 清晨启动方案 C (4h)
    - 下午评估三方案 (0.5h)
    - 固化最优方案 (0.5h)
    - 若通过，启动第 2 阶段 2a (1h)
  
  DAY 3:
    - 启动第 2 阶段 2b (4-5h)
    - 完成对比与评估 (1h)
  
  DAY 4 (可选):
    - 启动第 3 阶段长跑验证 (4h)
    - 冻结最终配置

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📝 【快速参考】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【关键代码改动一览】

方案 B (eval_limit 减 10 倍):
  agi_joint_trainer.py:805
  eval_limit = min(future_data.size(1), 100 * seq_len)

方案 C (禁用督导):
  agi_joint_trainer.py:151
  "supervise_every": 0,

方案 2b (seq_len 扩展):
  agi_joint_trainer.py:121-125 (CONFIG)
  "seq_len": 192,
  "batch_size": 16,
  "lr": 1.5e-4,
  "grad_clip": 0.8,

【关键文件清单】

准备工作:
  ✓ baseline_analysis.py -> baseline_snapshot.json, baseline_config_frozen.json
  ✓ prepare_phase1.py -> plan_A/B/C.json, PHASE1_MODIFICATION_GUIDE.md
  ✓ evaluate_phase1.py -> phase1_evaluation.json (三方案评议后运行)

执行过程:
  + agi_joint_trainer.py (原始)
  + agi_joint_trainer_original_phase0.py (备份, 用于恢复)
  + agi_joint_telemetry.csv (方案 A 的结果)
  + agi_joint_telemetry_B_eval_down.csv (方案 B 的结果)
  + agi_joint_telemetry_C_no_supervise.csv (方案 C 的结果)

评估结果:
  = phase1_evaluation.json (最优方案推荐)
  = character_baseline_2a.json (phase 2a 输出)
  = character_phase2b.json (phase 2b 输出, 与 2a 对比)
  = final_config_v1.0.json (最终冻结配置, phase 3 输出)

【常见问题 FAQ】

Q: 方案 A 应该运行多少 chunk?
A: 50 chunk (约 550 GB 数据), 预期 ~4 小时

Q: 三个方案的 telemetry CSV 会互相覆盖吗?
A: 是的，需要在完成各方案后立即备份 (cp ... _A.csv, _B.csv, _C.csv)

Q: 如果都没通过红线怎么办?
A: 可能需要:
   1. 提高红线容差 (2% loss 回退而非 1%)
   2. 采用混合策略 (B+部分 C)
   3. 重新审视"字符质量"是否真的依赖通量提升

Q: 第 2 阶段 2b 的 seq_len 能改到 256 吗?
A: 可以，但显存占用会翻倍，需先测试。建议先用 192 验证通路。

Q: 是否可以跳过第 1 阶段直接做 seq_len 扩展?
A: 不建议。First stabilize throughput, then extend context. Order matters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【执行开始】

已准备完毕。下一步:

  1. 运行以下命令执行方案 A (基线):
     cd d:\\H2Q-MicroStream && python agi_joint_trainer.py

  2. 方案 A 完成后，按照清单 Step 3-4-5 依次执行 B、C

  3. 三方案完成后，运行:
     python evaluate_phase1.py

祝训练顺利！
"""

print(checklist)

# 保存到文件
with open('EXECUTION_CHECKLIST.txt', 'w', encoding='utf-8') as f:
    f.write(checklist)

print("\n✓ 执行清单已保存 -> EXECUTION_CHECKLIST.txt")
