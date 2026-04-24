#!/usr/bin/env python3
"""
第 1 阶段执行脚本 - 创建三个方案配置与执行说明
"""

import json
import shutil
from pathlib import Path

# ============================================================================
# 基础配置（从 baseline_snapshot.json 读取）
# ============================================================================

baseline = json.load(open('baseline_snapshot.json', 'r'))

print("=" * 80)
print("【第 1 阶段】通量优化 A/B 对标准备")
print("=" * 80)

# ============================================================================
# 方案 A：基线（对照组）
# ============================================================================

config_A = {
    "name": "方案 A - 基线（对照组）",
    "description": "原配不变，作为基准对照",
    "modifications": [],
    "expected_tps": baseline.get('tokens_per_sec_μ', 18014),
    "expected_loss": baseline.get('val_loss_μ', 1.7873),
    "execution": "python agi_joint_trainer.py  # 直接运行 50 chunk"
}

# ============================================================================
# 方案 B：评估下采样
# ============================================================================

config_B = {
    "name": "方案 B - 评估下采样",
    "description": "减少验证计算：eval_limit 从 1000 × seq_len 改为 100 × seq_len",
    "modifications": [
        {
            "file": "agi_joint_trainer.py",
            "line": 805,
            "old": "            eval_limit = min(future_data.size(1), 1000 * seq_len)",
            "new": "            eval_limit = min(future_data.size(1), 100 * seq_len)  # 减 10 倍验证量"
        }
    ],
    "expected_tps": baseline.get('tokens_per_sec_μ', 18014) * 1.4,  # +40%
    "expected_loss": baseline.get('val_loss_μ', 1.7873),  # 评估计数减少，loss 应不变
    "risk": "⭐ 低（评估不反传，只影响监测）",
    "execution": """
# Step 1: 修改 agi_joint_trainer.py 第 805 行
#   old: eval_limit = min(future_data.size(1), 1000 * seq_len)
#   new: eval_limit = min(future_data.size(1), 100 * seq_len)
python agi_joint_trainer.py  # 运行 50 chunk

# Step 2: 评估完成后记录 telemetry 最后 50 行
#   计算：mean(tokens_per_sec), mean(val_loss)
"""
}

# ============================================================================
# 方案 C：无督导注入
# ============================================================================

config_C = {
    "name": "方案 C - 无 DeepSeek 督导",
    "description": "禁用外部督导注入：supervise_every 从 10 改为 0",
    "modifications": [
        {
            "file": "agi_joint_trainer.py",
            "line": 151,
            "old": '    "supervise_every": 10,',
            "new": '    "supervise_every": 0,  # 禁用 DeepSeek 督导'
        }
    ],
    "expected_tps": baseline.get('tokens_per_sec_μ', 18014) * 1.15,  # +15%
    "expected_loss": baseline.get('val_loss_μ', 1.7873) + 0.03,  # 可能 +0.02~0.05
    "risk": "⭐⭐ 中（失去外部蒸馏，loss 可能上升）",
    "execution": """
# Step 1: 修改 agi_joint_trainer.py CONFIG，第 151 行
#   old: "supervise_every": 10,
#   new: "supervise_every": 0,
python agi_joint_trainer.py  # 运行 50 chunk

# Step 2: 评估完成后记录指标
#   计算：mean(tokens_per_sec), mean(val_loss)
"""
}

# ============================================================================
# 生成方案对照表
# ============================================================================

plans = [
    ("plan_A_baseline.json", config_A),
    ("plan_B_eval_downsample.json", config_B),
    ("plan_C_no_supervise.json", config_C),
]

for filename, config in plans:
    json.dump(config, open(filename, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"\nOK {config['name']} -> {filename}")

print("\n" + "=" * 80)
print("PHASE 1 - THROUGHPUT OPTIMIZATION A/B PLAN")
print("=" * 80)

for _, config in plans:
    print(f"\n【{config['name']}】")
    print(f"  描述    ：{config['description']}")
    print(f"  预期提速：{config.get('expected_tps', 'N/A')}")
    print(f"  预期 loss：{config.get('expected_loss', 'N/A')}")
    if 'risk' in config:
        print(f"  风险等级：{config['risk']}")

print("\n" + "=" * 80)
print("【执行流程】")
print("=" * 80)

execution_steps = """
1. 备份原始 agi_joint_trainer.py
   cp agi_joint_trainer.py agi_joint_trainer_original.py

2. 执行方案 A（基线，对照组，~4h）
   python agi_joint_trainer.py
   # 完成后，telemetry CSV 增加约 50 行（chunk 441-490）
   # 记录文件为 agi_joint_telemetry.csv

3. 恢复原始文件
   cp agi_joint_trainer_original.py agi_joint_trainer.py

4. 执行方案 B（评估下采样，~4h）
   # 修改 agi_joint_trainer.py 第 805 行：
   #   old: eval_limit = min(future_data.size(1), 1000 * seq_len)
   #   new: eval_limit = min(future_data.size(1), 100 * seq_len)
   python agi_joint_trainer.py
   # 备份 telemetry
   cp agi_joint_telemetry.csv agi_joint_telemetry_B_eval_down.csv

5. 恢复原始文件再处理方案 C（无督导，~4h）
   cp agi_joint_trainer_original.py agi_joint_trainer.py
   # 修改 agi_joint_trainer.py 第 151 行：
   #   old: "supervise_every": 10,
   #   new: "supervise_every": 0,
   python agi_joint_trainer.py
   # 备份 telemetry
   cp agi_joint_telemetry.csv agi_joint_telemetry_C_no_supervise.csv

6. 对比三个结果（见下一步）
"""

print(execution_steps)

print("\n" + "=" * 80)
print("【评估方案（对比分析脚本）】")
print("=" * 80)

eval_script = """
运行完三个方案后，执行 evaluate_phase1.py 对比结果
（该脚本将在下一步自动生成）

预期输出：
  - 方案 A：tokens_per_sec ~18,000, val_loss ~1.787
  - 方案 B：tokens_per_sec ~25,200 (+40%), val_loss ~1.787
  - 方案 C：tokens_per_sec ~20,700 (+15%), val_loss ~1.820

✅ 选择标准：tps 最高 且 loss ≤ 1.805（红线 1.894 × 1% 容差）
"""

print(eval_script)
print("=" * 80)

# ============================================================================
# 生成修改指南
# ============================================================================

modification_guide = """
# 第 1 阶段修改指南

## 方案 B 修改（eval_limit 减 10 倍）

### 文件：agi_joint_trainer.py
### 行数：约第 805 行

#### 修改前：
```python
            eval_limit = min(future_data.size(1), 1000 * seq_len)
```

#### 修改后：
```python
            eval_limit = min(future_data.size(1), 100 * seq_len)  # 减 10 倍验证量
```

### 效果解释：
- 原配：每 chunk 验证约 1000 个子序列（seq_len=128 时约 128K tokens）
- 改后：每 chunk 验证约 100 个子序列（约 12.8K tokens）
- 益处：验证计算减少，每 chunk 训练时间减少，tps 提升
- 风险：无（监测量减少，但采样仍代表数据分布）

---

## 方案 C 修改（禁用 DeepSeek 督导）

### 文件：agi_joint_trainer.py
### 行数：约第 151 行（CONFIG 字典内）

#### 修改前：
```python
    "supervise_every": 10,
```

#### 修改后：
```python
    "supervise_every": 0,  # 禁用 DeepSeek 督导
```

### 效果解释：
- 原配：每 10 chunk 调用 DeepSeek 生成并注入样本（耗时 ~10 min/次）
- 改后：完全禁用外部督导
- 益处：每小时减少 ~10-15% 的非训练开销，tps 提升 15-25%
- 风险：中等（失去外部知识蒸馏，可能 val_loss 上升 0.02-0.05）

---

## 备份与恢复

### 备份原始整训文件
cp agi_joint_trainer.py agi_joint_trainer_original_phase0.py

### 恢复原始文件
cp agi_joint_trainer_original_phase0.py agi_joint_trainer.py

---

## 执行顺序注意

⚠️  重要：
1. 每个方案前务必恢复原始文件
2. 每个方案运行后立即备份 telemetry CSV（避免被下一个方案覆盖）
3. 三个方案都完成后，再运行 evaluate_phase1.py 对比分析
"""

with open('PHASE1_MODIFICATION_GUIDE.md', 'w', encoding='utf-8') as f:
    f.write(modification_guide)

print(f"\n✅ 修改指南已保存 → PHASE1_MODIFICATION_GUIDE.md")

print("\n" + "=" * 80)
print("【第 1 阶段准备完成】")
print("=" * 80)
print("""
📝 已生成文件：
  - plan_A_baseline.json
  - plan_B_eval_downsample.json
  - plan_C_no_supervise.json
  - PHASE1_MODIFICATION_GUIDE.md（详细修改说明）

👉 下一步：
  1. 备份 agi_joint_trainer.py
  2. 执行方案 A（原配，作为对照）
  3. 恢复文件，修改后执行方案 B
  4. 恢复文件，修改后执行方案 C
  5. 完成三个方案后运行 evaluate_phase1.py

⏱️  预期耗时：12 小时（3 方案 × 4h）

🎯 目标指标：
  - 吞吐提升 > 15%（相对方案 A）
  - val_loss ≤ 1.805（基线 1% 容差）
  - VRAM ≤ 0.25 GB（无 OOM）
""")
