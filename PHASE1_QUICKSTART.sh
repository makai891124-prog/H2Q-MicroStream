#!/usr/bin/env bash
# 第 1 阶段快速开始脚本
# 注意：这是伪脚本用于说明步骤，实际需在 PowerShell 或 cmd 中执行

echo "=============================================="
echo "第 1 阶段：通量优化 A/B 快速启动"
echo "=============================================="

# Step 0: 确认已完成第 0 阶段
if [ ! -f "baseline_snapshot.json" ]; then
    echo "错误：必须先运行 baseline_analysis.py"
    exit 1
fi

echo "✓ 基线配置已确认"

# Step 1: 备份原始文件
echo ""
echo "Step 1: 备份原始 agi_joint_trainer.py"
cp agi_joint_trainer.py agi_joint_trainer_original_phase0.py
echo "✓ 备份完成 -> agi_joint_trainer_original_phase0.py"

# ============================================================================
# 方案 A：基线（不修改，直接运行）
# ============================================================================

echo ""
echo "=============================================="
echo "方案 A：基线对照（原配，~4h）"
echo "=============================================="
echo "执行命令："
echo "  python agi_joint_trainer.py"
echo ""
echo "完成后会自动写入 agi_joint_telemetry.csv"
echo "（此时 CSV 应有约 490 行数据）"
echo ""
echo "⏱️  预期耗时：4 小时"

# ============================================================================
# 方案 B：评估下采样
# ============================================================================

echo ""
echo "=============================================="
echo "方案 B：评估下采样（~4h）"
echo "=============================================="
echo ""
echo "前置操作："
echo "  1. 恢复原始文件"
echo "     cp agi_joint_trainer_original_phase0.py agi_joint_trainer.py"
echo ""
echo "  2. 编辑 agi_joint_trainer.py，第 805 行改为："
echo "     eval_limit = min(future_data.size(1), 100 * seq_len)"
echo "     （原为 1000，改为 100）"
echo ""
echo "  3. 保存文件"
echo ""
echo "执行："
echo "  python agi_joint_trainer.py"
echo ""
echo "完成后备份结果防止被覆盖："
echo "  cp agi_joint_telemetry.csv agi_joint_telemetry_B_eval_down.csv"
echo ""
echo "⏱️  预期耗时：4 小时"

# ============================================================================
# 方案 C：无督导
# ============================================================================

echo ""
echo "=============================================="
echo "方案 C：禁用 DeepSeek 督导（~4h）"
echo "=============================================="
echo ""
echo "前置操作："
echo "  1. 恢复原始文件"
echo "     cp agi_joint_trainer_original_phase0.py agi_joint_trainer.py"
echo ""
echo "  2. 编辑 agi_joint_trainer.py，第 151 行（CONFIG 中）改为："
echo '     "supervise_every": 0,'
echo '     （原为 10，改为 0）'
echo ""
echo "  3. 保存文件"
echo ""
echo "执行："
echo "  python agi_joint_trainer.py"
echo ""
echo "完成后备份结果："
echo "  cp agi_joint_telemetry.csv agi_joint_telemetry_C_no_supervise.csv"
echo ""
echo "⏱️  预期耗时：4 小时"

# ============================================================================
# 评估与选择
# ============================================================================

echo ""
echo "=============================================="
echo "评估与选择最优方案"
echo "=============================================="
echo ""
echo "三个方案都完成后，执行："
echo "  python evaluate_phase1.py"
echo ""
echo "该脚本会："
echo "  1. 对比三个 telemetry 文件的最后 50 行"
echo "  2. 计算吞吐增益、loss 变化、VRAM 占用"
echo "  3. 选出最优方案（tps 最高 + loss <= 1.805）"
echo "  4. 生成 phase1_evaluation.json"
echo ""
echo "预期输出示例："
echo "  A_baseline：  tps=18000, loss=1.787, VRAM=0.187"
echo "  B_eval_down： tps=25000 (+39%), loss=1.787, VRAM=0.187"
echo "  C_no_supervise：tps=20700 (+15%), loss=1.820, VRAM=0.187"
echo "  -> 推荐方案 B"

# ============================================================================
# 收尾
# ============================================================================

echo ""
echo "=============================================="
echo "第 1 阶段完成后的下一步"
echo "=============================================="
echo ""
echo "1. 固化最优方案的代码改动到 agi_joint_trainer.py"
echo ""
echo "2. 启动第 2 阶段（字符可用性）"
echo "   - 子阶段 2a：基线化字符质量（seq_len=128）"
echo "   - 子阶段 2b：扩展 seq_len 到 192"
echo ""
echo "3. 对比两个阶段的生成样本，评估可用性改善"
echo ""
echo "预期 Phase 2 耗时：5-6 小时"
echo ""
echo "=============================================="
