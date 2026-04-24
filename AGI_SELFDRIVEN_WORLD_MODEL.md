# AGI Self-Driven World-Model Evolution System (H2Q Edition)

## 1. 目标定义
- 目标1：连续自治学习，保持时间箭头一致性与无监督结构涌现。
- 目标2：把外部开放知识源（数学论文、GitHub 代码、数据集描述）转成统一字节流并持续训练。
- 目标3：以拓扑指标驱动保存与策略自校正，而不是固定 epoch。

## 2. 资料源与通道
- arXiv：最新 math/cs.AI 摘要流（结构先验 + 新概念注入）。
- Hugging Face：高下载数据集元信息（任务拓扑与语义分布提示）。
- GitHub：核心仓库 README 与元数据（算法与工程范式注入）。

对应实现：
- 语料构建器：[build_open_corpus.py](build_open_corpus.py)
- 语料产物：[data/open_corpus/open_corpus.bin](data/open_corpus/open_corpus.bin)
- 来源清单：[data/open_corpus/source_manifest.json](data/open_corpus/source_manifest.json)

## 3. 进化核心（现有 H2Q）
- 主引擎：[local_evolution_daemon.py](local_evolution_daemon.py)
- 约束：cuda:0 强锁定、while True 连续流、训练后即时 zero_grad(set_to_none=True)。
- 遥测：每 1000 步记录 T_Step / EMA Loss / Sparsity / SVD Entropy / VRAM。
- 相变保存：EMA 创新低且 Sparsity > 0.5 时保存 best topology。

## 4. 自驱动层（新增）
- 编排器：[world_model_autopilot.py](world_model_autopilot.py)
- 循环：抓语料 -> 训练周期 -> 指标分析 -> 下一轮策略更新。
- 自校正规则（当前版本）：
  - 稀疏度长期低：降低学习率以增强结构收敛。
  - 熵过低接近坍缩：小幅提高学习率以恢复模态活性。

## 5. 世界模型最小闭环
- 感知层：开放语料增量输入（arXiv/HF/GitHub）。
- 动力学层：H2Q 拓扑注意力 + Rank-8 投影。
- 记忆层：best topology 权重 + 紧急快照。
- 元认知层：autopilot 指标诊断与策略调整。

## 6. 推荐执行序列
1. 生成开放语料：
   - python build_open_corpus.py --arxiv-max 300 --hf-max 80 --target-mb 256
2. 运行 24h 守护：
   - python local_evolution_daemon.py --source data/open_corpus/open_corpus.bin --seq-len 1024 --telemetry-every 1000 --print-every 1000 --svd-every 1000 --cache-clear-every 10000 --telemetry-csv evolution_telemetry_24h.csv
3. 跑多周期自驱动：
   - python world_model_autopilot.py --cycles 3 --steps-per-cycle 20000 --seq-len 1024

## 7. 判据（是否出现“新东西”）
- 拓扑相变事件频率提升（phase-save 次数上升）。
- 在高 sparsity 区间，EMA Loss 仍持续下降。
- SVD entropy 长期非零且不过快坍缩。
- 同显存占用下，单位时间有效步数稳定。

## 8. 风险与下一步
- 风险1：开放语料中噪声较高，可能导致局部损失振荡。
- 风险2：仅摘要而非全文时，知识密度受限。
- 下一步：接入 arXiv PDF 全文解析、代码仓深层文件抽样与目标导向任务回放。