
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
