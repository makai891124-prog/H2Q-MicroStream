#!/usr/bin/env python3
"""
第 1 阶段评估脚本 - 对比三个方案的结果
"""

import pandas as pd
import json
from pathlib import Path

print("=" * 80)
print("【第 1 阶段评估】三方案对比分析")
print("=" * 80)

# ============================================================================
# 加载基线配置
# ============================================================================

baseline = json.load(open('baseline_snapshot.json', 'r'))
baseline_tps = baseline.get('tokens_per_sec_μ', 18014)
baseline_loss = baseline.get('val_loss_μ', 1.7873)
red_line_loss = baseline.get('val_loss_max', 1.8051)
red_line_tps = baseline.get('tokens_per_sec_min', 16213)

print(f"\n基线参考：tps={baseline_tps:.0f}, val_loss={baseline_loss:.4f}")
print(f"资源红线：tps_min={red_line_tps:.0f}, loss_max={red_line_loss:.4f}")

# ============================================================================
# 配置三个 telemetry 文件路径
# ============================================================================

telemetry_files = {
    'A_baseline': 'agi_joint_telemetry.csv',
    'B_eval_down': 'agi_joint_telemetry_B_eval_down.csv',
    'C_no_supervise': 'agi_joint_telemetry_C_no_supervise.csv',
}

results = {}

for plan_name, csv_file in telemetry_files.items():
    path = Path(csv_file)
    
    if not path.exists():
        print(f"\n警告：文件不存在 {csv_file}（跳过）")
        continue
    
    df = pd.read_csv(csv_file)
    
    # 取最后 50 chunk 作为评估窗口
    window = df.tail(50)
    
    tps_mean = window['tokens_per_sec'].mean()
    tps_std = window['tokens_per_sec'].std()
    loss_mean = window['val_loss'].mean()
    loss_std = window['val_loss'].std()
    vram_mean = window['vram_alloc_gb'].mean()
    
    # 计算相对改进
    tps_gain_pct = (tps_mean / baseline_tps - 1) * 100
    loss_delta = loss_mean - baseline_loss
    
    # pass/fail 判定
    tps_pass = tps_mean >= red_line_tps
    loss_pass = loss_mean <= red_line_loss
    vram_pass = vram_mean <= 0.25
    overall_pass = tps_pass and loss_pass and vram_pass
    
    results[plan_name] = {
        'tps': tps_mean,
        'tps_std': tps_std,
        'tps_gain_pct': tps_gain_pct,
        'loss': loss_mean,
        'loss_std': loss_std,
        'loss_delta': loss_delta,
        'vram': vram_mean,
        'tps_pass': tps_pass,
        'loss_pass': loss_pass,
        'vram_pass': vram_pass,
        'overall_pass': overall_pass,
    }
    
    print(f"\n" + "=" * 80)
    print(f"【{plan_name}】")
    print("=" * 80)
    print(f"  吞吐量(tok/s)   : {tps_mean:7.0f} ± {tps_std:5.0f}  ({tps_gain_pct:+6.1f}%)  {'PASS' if tps_pass else 'FAIL'}")
    print(f"  验证损失(val)   : {loss_mean:7.4f} ± {loss_std:7.4f}  ({loss_delta:+7.4f})   {'PASS' if loss_pass else 'FAIL'}")
    print(f"  VRAM占用(GB)   : {vram_mean:7.3f}                         {'PASS' if vram_pass else 'FAIL'}")
    print(f"  综合判定        : {'✓ 通过' if overall_pass else '✗ 未通过'}")

# ============================================================================
# 最优方案选择
# ============================================================================

print("\n" + "=" * 80)
print("【最优方案选择】")
print("=" * 80)

# 找最佳方案：tps 最高 且所有指标都通过
best_plan = None
best_tps = baseline_tps

for plan_name, stats in results.items():
    if stats['overall_pass'] and stats['tps'] > best_tps:
        best_plan = plan_name
        best_tps = stats['tps']

if best_plan:
    best_stats = results[best_plan]
    print(f"\n✓ 推荐方案：{best_plan}")
    print(f"  吞吐收益    : +{best_stats['tps_gain_pct']:6.1f}% ({best_stats['tps']:.0f} tok/s)")
    print(f"  Loss 变化   : {best_stats['loss_delta']:+7.4f} （{best_stats['loss']:.4f}）")
    print(f"  VRAM        : {best_stats['vram']:.3f} GB")
else:
    print(f"\n⚠  无方案通过所有红线。分析备选方案:")
    
    # 如果都没通过，列出最接近的
    for plan_name, stats in results.items():
        violations = []
        if not stats['tps_pass']:
            violations.append(f"tps 低 {red_line_tps - stats['tps']:.0f} tok/s")
        if not stats['loss_pass']:
            violations.append(f"loss 高 {stats['loss'] - red_line_loss:.4f}")
        if not stats['vram_pass']:
            violations.append(f"vram 高 {stats['vram'] - 0.25:.3f} GB")
        
        print(f"\n  {plan_name}: " + "; ".join(violations) if violations else "可接受")

# ============================================================================
# 保存结果到文件
# ============================================================================

summary = {
    'baseline': {
        'tps': baseline_tps,
        'loss': baseline_loss,
        'vram': 0.187,
    },
    'red_lines': {
        'tps_min': red_line_tps,
        'loss_max': red_line_loss,
        'vram_max': 0.25,
    },
    'results': results,
    'best_plan': best_plan,
}

json.dump(summary, open('phase1_evaluation.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

print(f"\n\n✓ 评估结果已保存 -> phase1_evaluation.json")

# ============================================================================
# 下一步建议
# ============================================================================

print("\n" + "=" * 80)
print("【下一步】")
print("=" * 80)

if best_plan:
    print(f"""
推荐配置：{best_plan}
预期效果：
  - 吞吐提升 {results[best_plan]['tps_gain_pct']:+.1f}%
  - Loss 变化 {results[best_plan]['loss_delta']:+.4f}
  
后续步骤：
  1. 固化 {best_plan} 的代码改动
  2. 启动第 2 阶段（字符可用性推进）
     - 2a：基线化字符质量（seq_len=128）
     - 2b：扩展 seq_len 到 192 或 256
  3. 对比生成质量，评估可用性改善
  
预期 Phase 2 耗时：5-6 小时
目标指标：
  - 字符可读性评分 >= 0.7
  - val_loss <= {red_line_loss:.4f}
  - 吞吐 >= {red_line_tps:.0f} tok/s
""")
else:
    print("""
所有方案都未完全通过红线。可能的原因：
  1. 基线数据波动大（可重复实验）
  2. 改动存在问题（检查代码修改）
  3. 资源约束（降低 batch_size 或 seq_len）

建议：
  - 重新运行通过率最高的方案，确认稳定性
  - 或调整红线，使用更宽松的容差（e.g., 2% loss 回退）
  - 或采用混合策略（e.g., 方案 B + 部分 C）
""")

print("=" * 80)
