# 仓库未跟踪文件清点与整理建议

## 1. 目标

本报告用于区分当前工作区中的未跟踪文件，明确哪些更适合进入版本控制，哪些应通过 `.gitignore` 排除，减少仓库噪声并提高后续复现性。

## 2. 建议提交到仓库的内容

### 2.1 源码与脚本

- Python 源码：
  - `agi_joint_trainer.py`
  - `baseline_analysis.py`
  - `build_open_corpus.py`
  - `evaluate_phase1.py`
  - `execution_checklist.py`
  - `final_verification.py`
  - `finalize_test_report.py`
  - `generate_completion_report.py`
  - `local_evolution_daemon.py`
  - `prepare_phase1.py`
  - `run_all_tests_and_report.py`
  - `test_agi_quick.py`
  - `world_model_autopilot.py`
- `H2Q-Single/` 中除权重、缓存和敏感配置以外的 `.py` 文件。

### 2.2 文档与报告

- `AGI_SELFDRIVEN_WORLD_MODEL.md`
- `EXECUTION_CHECKLIST.txt`
- `FINAL_TEST_REPORT.md`
- `PHASE1_COMPLETION_REPORT.md`
- `PHASE1_MODIFICATION_GUIDE.md`
- `PHASE1_QUICKSTART.sh`
- `START_HERE.md`
- `FINAL_ANALYSIS_REPORT_CN.md`
- `REPO_CLEANUP_REPORT_CN.md`

### 2.3 可复现配置与快照

- `baseline_config_frozen.json`
- `baseline_snapshot.json`
- `plan_A_baseline.json`
- `plan_B_eval_downsample.json`
- `plan_C_no_supervise.json`

这些文件如果用于说明实验配置和流程，应优先纳入版本控制。

## 3. 建议忽略的内容

### 3.1 运行环境与缓存

- `.venv/`
- `__pycache__/`
- `*.pyc`

### 3.2 编译产物

- `*.exe`
- 其它原生编译中间产物（`.dll`、`.obj`、`.o` 等）

### 3.3 大模型权重与大型二进制数据

- `*.pt`
- `*.bin`

包括但不限于：

- `agi_joint.pt`
- `agi_joint_best.pt`
- `best_*.pt`
- `ckpt_*.pt`
- `h2q_evolution_best_topology.pt`
- `h2q_evolution_emergency.pt`
- `corpus_mix_256mb.bin`
- `data/open_corpus/open_corpus.bin`

### 3.4 日志与生成输出

- `*.log`
- `telemetry_*.csv`
- `evolution_telemetry*.csv`
- `agi_joint_telemetry.csv`
- `final_test_report.json`
- `autopilot_state.json`

### 3.5 敏感内容

- `.secrets/`
- `H2Q-Single/secret_config.py`
- `*.key`
- `*.secret`
- `*.token`

## 4. 需要人工判断的内容

以下文件不应盲目忽略，也不一定全部提交，建议按用途再确认：

- `autopilot_hypotheses.jsonl`
  - 如果是研究输入样本，应提交。
  - 如果是运行时生成状态，应忽略。
- `autopilot_report.md`
  - 如果是人工整理报告，应提交。
  - 如果每次运行都会重写，应转为生成物并忽略。
- `data/open_corpus/open_corpus.txt`
  - 如果用于小规模复现实验，可提交。
  - 如果体积持续增长且可脚本生成，建议只保留 manifest。
- `data/open_corpus/source_manifest.json`
  - 推荐提交，便于追溯语料来源。

## 5. 本次整理动作

- 已补充更干净的 `.gitignore`。
- 已将 prime 实验代码与报告整理到 `experiments/prime_sieve/`。
- 已保留总报告在根目录，避免主入口丢失。

## 6. 下一步建议

1. 单独提交“源码与文档”一批。
2. 单独提交“配置快照”一批。
3. 继续把生成数据、日志和权重统一排除，保持仓库主干轻量。