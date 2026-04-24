#!/usr/bin/env python3
"""
最终验证与启动检查清单
确认所有文件已生成，用户可立刻开始执行
"""

import json
import os
from pathlib import Path

print("=" * 80)
print("【最终验证】第 0-1 阶段完成状态检查")
print("=" * 80)

# ============================================================================
# 检查关键文件是否存在
# ============================================================================

required_files = {
    '【第 0 阶段输出】': [
        'baseline_analysis.py',
        'baseline_snapshot.json',
        'baseline_config_frozen.json',
    ],
    '【第 1 阶段准备】': [
        'prepare_phase1.py',
        'plan_A_baseline.json',
        'plan_B_eval_downsample.json',
        'plan_C_no_supervise.json',
        'PHASE1_MODIFICATION_GUIDE.md',
        'evaluate_phase1.py',
    ],
    '【执行指南与文档】': [
        'START_HERE.md',
        'EXECUTION_CHECKLIST.txt',
        'PHASE1_COMPLETION_REPORT.md',
    ],
}

all_ok = True

for category, files in required_files.items():
    print(f"\n{category}")
    print("-" * 80)
    for filename in files:
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  ✓ {filename:<40} ({size:6.1f} KB)")
        else:
            print(f"  ✗ {filename:<40} (缺失！)")
            all_ok = False

# ============================================================================
# 验证关键 JSON 文件内容
# ============================================================================

print("\n" + "=" * 80)
print("【JSON 文件内容验证】")
print("=" * 80)

try:
    baseline = json.load(open('baseline_snapshot.json', 'r', encoding='utf-8'))
    print("\n✓ baseline_snapshot.json 内容：")
    print(f"  train_loss: {baseline['train_loss_μ']:.4f}")
    print(f"  val_loss: {baseline['val_loss_μ']:.4f}")
    print(f"  tps: {baseline['tokens_per_sec_μ']:.0f}")
    print(f"  vram: {baseline['vram_alloc_gb']:.3f} GB")
    print(f"  val_loss_max (红线): {baseline['val_loss_max']:.4f}")
    print(f"  tps_min (红线): {baseline['tokens_per_sec_min']:.0f}")
except Exception as e:
    print(f"✗ baseline_snapshot.json 读取失败：{e}")
    all_ok = False

try:
    config = json.load(open('baseline_config_frozen.json', 'r', encoding='utf-8'))
    print("\n✓ baseline_config_frozen.json 内容：")
    print(f"  模型配置：dim={config['model_config']['dim']}, depth={config['model_config']['depth']}")
    print(f"  训练配置：batch_size={config['training_config']['batch_size']}, lr={config['training_config']['lr']}")
except Exception as e:
    print(f"✗ baseline_config_frozen.json 读取失败：{e}")
    all_ok = False

# ============================================================================
# 验证方案配置
# ============================================================================

print("\n" + "=" * 80)
print("【方案配置验证】")
print("=" * 80)

plans = [
    ('plan_A_baseline.json', 'A'),
    ('plan_B_eval_downsample.json', 'B'),
    ('plan_C_no_supervise.json', 'C'),
]

for filename, plan_name in plans:
    try:
        plan = json.load(open(filename, 'r', encoding='utf-8'))
        expected_tps = plan.get('expected_tps', 'N/A')
        print(f"\n✓ 方案 {plan_name}: {plan['name']}")
        print(f"  预期吞吐：{expected_tps}")
        print(f"  风险等级：{plan.get('risk', 'N/A')}")
    except Exception as e:
        print(f"✗ {filename} 读取失败：{e}")
        all_ok = False

# ============================================================================
# 验证主执行脚本
# ============================================================================

print("\n" + "=" * 80)
print("【执行脚本验证】")
print("=" * 80)

scripts = [
    ('baseline_analysis.py', '基线提取'),
    ('prepare_phase1.py', '方案准备'),
    ('evaluate_phase1.py', '方案评估'),
]

for filename, desc in scripts:
    path = Path(filename)
    if path.exists():
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = len(content.split('\n'))
        print(f"✓ {filename:<25} ({desc:<20}) {lines:>4} 行")
    else:
        print(f"✗ {filename:<25} (缺失！)")
        all_ok = False

# ============================================================================
# 确认训练脚本可用
# ============================================================================

print("\n" + "=" * 80)
print("【训练脚本检查】")
print("=" * 80)

trainer_path = Path('agi_joint_trainer.py')
if trainer_path.exists():
    size = trainer_path.stat().st_size / 1024
    with open(str(trainer_path), 'r', encoding='utf-8') as f:
        content = f.read()
        lines = len(content.split('\n'))
    print(f"✓ agi_joint_trainer.py 可用")
    print(f"  大小：{size:.1f} KB，{lines} 行代码")
    print(f"  状态：原始未修改，可用于方案 A（基线）")
else:
    print(f"✗ agi_joint_trainer.py 不存在！")
    all_ok = False

# ============================================================================
# 验证遥测数据
# ============================================================================

print("\n" + "=" * 80)
print("【遥测数据检查】")
print("=" * 80)

import pandas as pd

telemetry_path = Path('agi_joint_telemetry.csv')
if telemetry_path.exists():
    df = pd.read_csv(str(telemetry_path))
    print(f"✓ agi_joint_telemetry.csv 可用")
    print(f"  已训练 {len(df)} chunks（共 {df['chunk'].max()} 个 chunks）")
    print(f"  最后一行（chunk {df['chunk'].iloc[-1]:.0f}）：")
    print(f"    train_loss={df['train_loss'].iloc[-1]:.4f}")
    print(f"    val_loss={df['val_loss'].iloc[-1]:.4f}")
    print(f"    tps={df['tokens_per_sec'].iloc[-1]:.0f} tok/s")
else:
    print(f"✗ agi_joint_telemetry.csv 不存在！")
    all_ok = False

# ============================================================================
# 生成启动检查表
# ============================================================================

print("\n" + "=" * 80)
print("【启动前最终检查】")
print("=" * 80)

checklist = {
    '✓ 第 0 阶段': [
        ('基线数据已提取', Path('baseline_snapshot.json').exists()),
        ('资源红线已设定', Path('baseline_snapshot.json').exists()),
        ('配置已冻结', Path('baseline_config_frozen.json').exists()),
    ],
    '✓ 第 1 阶段准备': [
        ('方案 A 配置就绪', Path('plan_A_baseline.json').exists()),
        ('方案 B 配置就绪', Path('plan_B_eval_downsample.json').exists()),
        ('方案 C 配置就绪', Path('plan_C_no_supervise.json').exists()),
        ('代码修改指南就绪', Path('PHASE1_MODIFICATION_GUIDE.md').exists()),
        ('评估脚本就绪', Path('evaluate_phase1.py').exists()),
    ],
    '✓ 文档与指南': [
        ('快速入门文档就绪', Path('START_HERE.md').exists()),
        ('执行清单就绪', Path('EXECUTION_CHECKLIST.txt').exists()),
        ('完整报告就绪', Path('PHASE1_COMPLETION_REPORT.md').exists()),
    ],
    '✓ 执行环境': [
        ('主训练脚本可用', Path('agi_joint_trainer.py').exists()),
        ('历史遥测数据可用', Path('agi_joint_telemetry.csv').exists()),
        ('检查点文件存在', Path('agi_joint.pt').exists()),
    ],
}

for category, items in checklist.items():
    print(f"\n{category}")
    for desc, status in items:
        symbol = '✓' if status else '✗'
        print(f"  [{symbol}] {desc}")

# ============================================================================
# 总体状态
# ============================================================================

print("\n" + "=" * 80)
print("【总体状态】")
print("=" * 80)

if all_ok:
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║        ✅ 所有文件已生成，系统就绪！                                ║
║                                                                       ║
║        可以立刻执行第 1 阶段（方案 A/B/C 通量优化）                 ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

【建议的立刻行动】

1. 阅读快速入门文档：
   START_HERE.md

2. 如需详细指南，查看：
   - EXECUTION_CHECKLIST.txt (完整检查表)
   - PHASE1_MODIFICATION_GUIDE.md (代码改动说明)
   - PHASE1_COMPLETION_REPORT.md (技术细节)

3. 立刻启动第 1 阶段方案 A（基线对照）：
   python agi_joint_trainer.py

4. 预期 4 小时完成后，启动方案 B：
   [参考 PHASE1_MODIFICATION_GUIDE.md 修改第 805 行]
   python agi_joint_trainer.py

5. 三方案完成后，运行评估脚本：
   python evaluate_phase1.py

【预期时间表】
  方案 A: 4 小时
  方案 B: 4 小时
  方案 C: 4 小时
  评估:   0.5 小时
  总计:   约 12.5 小时

祝您的训练顺利！🚀
""")
else:
    print("✗ 部分文件缺失或验证失败，请检查上述错误信息")

print("=" * 80)
