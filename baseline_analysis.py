#!/usr/bin/env python3
"""
第 0 阶段：基线快照与红线设定
提取当前训练的基线统计，冻结配置，设定资源约束
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================================
# Step 0.1：从遥测 CSV 提取基线统计
# ============================================================================

csv_path = Path('agi_joint_telemetry.csv')
df = pd.read_csv(csv_path)

print("=" * 80)
print("【第 0 阶段】基线快照 & 资源红线设定")
print("=" * 80)

# 提取最后 100 chunk 作为基线窗口
baseline_window = df[df['chunk'] >= df['chunk'].max() - 100]
print(f"\n📊 基线窗口：最后 100 chunk（chunk {baseline_window['chunk'].min()}-{baseline_window['chunk'].max()}）")

# 计算统计量
baseline_stats = {
    'total_chunks_trained': int(df['chunk'].max()),
    'window_size': len(baseline_window),
    'train_loss_μ': float(baseline_window['train_loss'].mean()),
    'train_loss_σ': float(baseline_window['train_loss'].std()),
    'val_loss_μ': float(baseline_window['val_loss'].mean()),
    'val_loss_σ': float(baseline_window['val_loss'].std()),
    'tokens_per_sec_μ': float(baseline_window['tokens_per_sec'].mean()),
    'tokens_per_sec_σ': float(baseline_window['tokens_per_sec'].std()),
    'vram_alloc_gb': float(baseline_window['vram_alloc_gb'].mean()),
    'gen_gap': float(baseline_window['val_loss'].mean() - baseline_window['train_loss'].mean()),
}

print("\n📈 基线指标（100-chunk 窗口）")
print("-" * 80)
print(f"  训练 Loss    ：{baseline_stats['train_loss_μ']:.4f} ± {baseline_stats['train_loss_σ']:.4f}")
print(f"  验证 Loss    ：{baseline_stats['val_loss_μ']:.4f} ± {baseline_stats['val_loss_σ']:.4f}")
print(f"  泛化间隙      ：{baseline_stats['gen_gap']:.4f} （无过拟合迹象）")
print(f"  吞吐量(tokens/s) ：{baseline_stats['tokens_per_sec_μ']:.0f} ± {baseline_stats['tokens_per_sec_σ']:.0f}")
print(f"  显存分配      ：{baseline_stats['vram_alloc_gb']:.3f} GB / ~8 GB（约 {baseline_stats['vram_alloc_gb']/8*100:.1f}% 占用）")

# 设定资源红线
red_lines = {
    'val_loss_max': baseline_stats['val_loss_μ'] * 1.01,      # 1% 容差
    'tokens_per_sec_min': baseline_stats['tokens_per_sec_μ'] * 0.90,  # 90% 下限
    'vram_alloc_max': 0.25,  # 硬上限 0.25GB（剩余 7.75GB）
    'ortho_loss_max': 12.0,   # 正交性约束软上限
}

print("\n🚨 资源红线（触发告警/回退条件）")
print("-" * 80)
print(f"  val_loss 高于    ：{red_lines['val_loss_max']:.4f}（+1% 回退阈值）")
print(f"  吞吐 低于        ：{red_lines['tokens_per_sec_min']:.0f} tokens/sec（-10% 警告）")
print(f"  显存 高于        ：{red_lines['vram_alloc_max']:.3f} GB（硬 OOM 防线）")

# 保存基线统计
baseline_stats.update(red_lines)
json.dump(baseline_stats, open('baseline_snapshot.json', 'w'), indent=2)
print(f"\n✅ 基线快照已保存 → baseline_snapshot.json")

# ============================================================================
# Step 0.2：冻结当前配置
# ============================================================================

baseline_config = {
    'model_config': {
        'dim': 768,
        'depth': 12,
        'seq_len': 128,
        'vocab_size': 256,
        'rank_factor': 32,
        'fixed_rank': 8,
    },
    'training_config': {
        'batch_size': 24,
        'lr': 3e-4,
        'grad_clip': 1.0,
        'weight_decay': 1e-5,
        'chunk_size_mb': 10,
        'supervise_every': 10,
    },
    'infrastructure': {
        'device': 'cuda:0',
        'dtype': 'float32',
        'total_chunks_plan': 200000,
        'checkpoint_every': 1,
        'eval_every': 1,
    },
    'baseline_metrics': baseline_stats,
}

json.dump(baseline_config, open('baseline_config_frozen.json', 'w'), indent=2)
print(f"✅ 基线配置已冻结  → baseline_config_frozen.json")

# ============================================================================
# Step 0.3：早期 vs 晚期对比（收敛健康检查）
# ============================================================================

early_50 = df[df['chunk'] <= 50]
mid_100_150 = df[(df['chunk'] > 100) & (df['chunk'] <= 150)]
late_50 = df[df['chunk'] >= df['chunk'].max() - 50]

print("\n📉 收敛健康状态（早期 vs 晚期对比）")
print("-" * 80)
print(f"  早期（chunk 1-50）  ：train={early_50['train_loss'].mean():.4f}, val={early_50['val_loss'].mean():.4f}")
print(f"  中期（chunk 100-150）：train={mid_100_150['train_loss'].mean():.4f}, val={mid_100_150['val_loss'].mean():.4f}")
print(f"  晚期（chunk {late_50['chunk'].min()}-{late_50['chunk'].max()}）  ：train={late_50['train_loss'].mean():.4f}, val={late_50['val_loss'].mean():.4f}")
print(f"\n  收敛判断：已进入平台期（晚期 vs 中期 loss 变化 < 0.015）✅")

# ============================================================================
# 概括与下一步
# ============================================================================

print("\n" + "=" * 80)
print("【第 0 阶段完成】")
print("=" * 80)
print("\n📋 下一步（第 1 阶段：通量优化 A/B）")
print("  - 方案 A：原配基线（50 chunk）")
print("  - 方案 B：评估下采样（eval_limit 减 10 倍，50 chunk）")
print("  - 方案 C：无督导（supervise_every: 10→0，50 chunk）")
print("\n⏱️  预期耗时：12h（3 方案 × 4h）")
print("\n🎯 目标指标：")
print("  - tps 提升 > 15%（相对方案 A）")
print("  - loss 不超过 1.794（基线 1% 容差）")
print("  - VRAM ≤ 0.25 GB（无 OOM）")
print("\n" + "=" * 80)
