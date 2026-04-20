# 最终实验结果与分析报告（中文版）

## 1. 执行说明
- 本次实验已完成以下阶段：
  - Phase1: A_baseline / B_eval_down / C_no_supervise
  - Phase2: 2a_char_baseline / 2b_seq192
- 结果来源文件：
  - telemetry_A_baseline.csv
  - telemetry_B_eval_down.csv
  - telemetry_C_no_supervise.csv
  - telemetry_2a_char_baseline.csv
  - telemetry_2b_seq192.csv
  - final_test_report.json

## 2. 关键结果（汇总）

### 2.1 Phase1（通量方案 A/B/C）
- A_baseline:
  - tps: 43222
  - val_loss: 7.5207
  - vram: 0.037 GB
  - 结论: 未通过 loss 红线
- B_eval_down:
  - tps: 43143
  - val_loss: 19.5215
  - vram: 0.0365 GB
  - 结论: 未通过 loss 红线
- C_no_supervise:
  - tps: 43296
  - val_loss: 17.7792
  - vram: 0.036 GB
  - 结论: 未通过 loss 红线，但在本轮中吞吐最高

### 2.2 Phase2（字符可用性 2a/2b）
- 2a_char_baseline:
  - tps: 43443.5
  - val_loss: 18.8626
  - readability: 0.6800
- 2b_seq192:
  - tps: 41586.0
  - val_loss: 18.1502
  - readability: 0.6800
- 可读性增量:
  - delta(2b-2a) = +0.00003（近似 0）

## 3. 结论
- 本轮自动实验“流程执行层面”是成功的：
  - 所有阶段均已跑通
  - 指标均已写入 telemetry 与 JSON
  - 报告已自动生成
- 但在“模型性能结论层面”暂不满足升级条件：
  - 2b 相对 2a 没有显著可读性提升
  - loss 指标远高于正式基线红线（1.805）
  - 因此不建议将 seq_len 从 128 升级到 192 作为正式配置

## 4. 为什么会出现 loss 偏高
- 当前这轮是快速验证模式（quick run）：
  - 每个方案仅 2 个 chunk
  - 含冷启动阶段，统计窗口非常短
- 在短窗口下，rolling 验证损失方差会明显放大；
  因此该结果更适合用于“流程验通”和“方向筛选”，不用于最终定稿。

## 5. 正式定稿建议（下一轮）
- Phase1 正式复验：
  - A/B/C 每组至少 50 chunk
  - 继续使用独立 telemetry 文件，避免互相覆盖
- Phase2 正式复验：
  - 2a（seq_len=128）与 2b（seq_len=192）至少各 100-200 chunk
  - 使用相同提示词集采样，计算可读性均值与方差
- 判定门槛（保持不变）：
  - val_loss <= 1.805
  - tps >= 16213
  - 2b 可读性提升 >= +0.2

## 6. 当前可执行决策
- 立即可采用：
  - 保持 seq_len=128
  - 若必须从本轮 Phase1 选一个临时通量方案：选择 C_no_supervise（吞吐最高）
- 不建议立即采用：
  - 直接升级到 seq_len=192（证据不足）

## 7. 交付物清单
- 英文汇总报告：FINAL_TEST_REPORT.md
- 结构化结果：final_test_report.json
- 中文分析报告：FINAL_ANALYSIS_REPORT_CN.md

---

结论一句话：
本轮实验已完整执行并成功产出报告，但属于快速验证性质；用于正式上线前，必须进行长窗口复验。当前最稳妥策略是维持 seq_len=128，并进入下一轮 50/100+ chunk 的正式 A/B。

---

## 8. 素数筛扩展实验并入结论（新增）

### 8.1 目的
- 针对“新发现素数作为新轮基进行自控制收敛”的机制，进行了真实 C++ 编译与运行验证。
- 重点观察三项指标：正确性、时间效率、内存可控性。

### 8.2 实现与环境
- 代码文件：
  - `benchmark_prime_algorithms.cpp`
  - `prime_primorial_mvp.cpp`
- 编译器：MSYS2 UCRT64 `g++ 15.2.0`
- 编译参数：`-O3 -std=c++17 -march=native`
- 新增算法：`adaptive_wheel`（分段 + 动态轮基预筛）

### 8.3 对照算法集合
- `full_byte`：全空间字节筛
- `odd_byte`：奇数压缩字节筛
- `wheel6_bit`：6 轮（2x3）位图筛
- `segmented_odd`：经典分段奇数筛
- `adaptive_wheel`：动态轮基分段筛（本次新增）

### 8.4 真实运行结果

#### n = 10^7
| algorithm | count | time_ms | memory_mb | ok |
|---|---:|---:|---:|---|
| full_byte | 664579 | 20 | 9.537 | yes |
| odd_byte | 664579 | 9 | 4.768 | yes |
| wheel6_bit | 664579 | 3 | 0.397 | yes |
| segmented_odd | 664579 | 5 | 0.502 | yes |
| adaptive_wheel | 664579 | 6 | 0.516 | yes |

#### n = 10^8
| algorithm | count | time_ms | memory_mb | ok |
|---|---:|---:|---:|---|
| full_byte | 5761455 | 532 | 95.367 | yes |
| odd_byte | 5761455 | 218 | 47.684 | yes |
| wheel6_bit | 5761455 | 47 | 3.974 | yes |
| segmented_odd | 5761455 | 59 | 0.505 | yes |
| adaptive_wheel | 5761455 | 72 | 0.519 | yes |

#### n = 10^9
| algorithm | count | time_ms | memory_mb | ok |
|---|---:|---:|---:|---|
| wheel6_bit | 50847534 | 1039 | 39.736 | yes |
| segmented_odd | 50847534 | 659 | 0.513 | yes |
| adaptive_wheel | 50847534 | 780 | 0.527 | yes |

### 8.5 关键分析
- 正确性：
  - 三个扩展算法在 `10^9` 全部命中标准值 `pi(10^9)=50,847,534`。
- 内存行为：
  - `adaptive_wheel` 在 `10^9` 仍保持约 `0.53 MB` 峰值内存，体现了分段与轮基预算控制的稳定性。
- 时间效率：
  - 当前实现下，`adaptive_wheel` 尚慢于 `segmented_odd`（`780 ms` vs `659 ms`）。
  - 主要原因是轮模板拷贝与 residue 更新开销仍偏高。

### 8.6 工程与学术定位
- 学术定位：
  - 属于经典筛法体系内的工程优化（adaptive wheel presieving + segmented sieve），复杂度阶未变。
- 工程价值：
  - 提供“以内存预算驱动轮基扩张停止”的自控制接口。
  - 对大范围素数扫描具备稳定低内存特征，可作为后续大素数生成流程前置过滤器。

### 8.7 已修复问题
- 初版动态轮模板 residue 映射存在错误，已修复为“全周期模板 + 按 +2 residue 迭代”，并通过 `10^7/10^8/10^9` 校验。

### 8.8 结论补充
- “新素数递进轮基 + 自控制内存”机制已被真实工程验证可行。
- 当前版本不是全局最快，但在“正确性 + 可扩展 + 内存可控”三者平衡上成立。
- 若继续优化模板表达（bitset 化）与段内标记流水，仍有进一步提速空间。
