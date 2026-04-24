# Binary STA cuda_ext 切换决策报告

## 编译环境
- PyTorch: 2.5.1+cu121  (CUDA 12.1)
- Device:  NVIDIA GeForce RTX 4070 Ti SUPER
- nvcc:    Build cuda_12.1.r12.1/compiler.32415258_0
- cl.exe:  用法: cl [ 选项... ] 文件名... [ /link 链接选项... ]
- ninja:   C:\Users\makai\AppData\Local\Programs\Python\Python310\Scripts\ninja.EXE
- CUDA_HOME: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
- **can_compile: True**

## 3×3 协议结果

| steps | variant | TPS | last_loss | p99_ms | VRAM_MB | backend_eff |
|------:|---------|----:|----------:|-------:|--------:|-------------|
| 10 | sta_v2 | 19114 | 5.12985 | 104.34 | 39.0 | n/a |
| 10 | packbits | 33421 | 4.89967 | 21.85 | 32.9 | packbits |
| 10 | cuda_ext | 63076 | 5.04453 | 10.49 | 25.7 | cuda_ext |
| 30 | sta_v2 | 23023 | 4.33032 | 32.96 | 39.0 | n/a |
| 30 | packbits | 34963 | 4.17004 | 18.92 | 32.9 | packbits |
| 30 | cuda_ext | 66126 | 4.23985 | 10.14 | 25.7 | cuda_ext |
| 60 | sta_v2 | 24012 | 3.65187 | 26.56 | 39.0 | n/a |
| 60 | packbits | 34645 | 3.61378 | 19.33 | 32.9 | packbits |
| 60 | cuda_ext | 66306 | 3.65371 | 10.51 | 25.7 | cuda_ext |

## cuda_ext vs packbits 差值

| steps | loss_delta | tps_ratio | p99_ratio | vram_delta_MB | verdict |
|------:|-----------:|----------:|----------:|--------------:|---------|
| 10 | 0.144854 | 1.8873 | 0.4802 | -7.14 | ❌ FAIL |
| 30 | 0.069813 | 1.8913 | 0.5362 | -7.14 | ❌ FAIL |
| 60 | 0.039933 | 1.9139 | 0.5437 | -7.14 | ✅ PASS |

## 门槛定义
- loss_delta ≤ 0.05
- TPS 提升 ≥ 110%（vs packbits）
- p99 回退 ≤ 20%
- VRAM 额外开销 ≤ 50.0 MB

## 最终结论

**整体判决: KEEP_PACKBITS**

⚠️  保持 packbits 为默认（cuda_ext 未完全满足门槛）
未通过的判据:
  • steps=10: loss_delta=+0.1449>0.05
  • steps=30: loss_delta=+0.0698>0.05

下一步建议:
  1. 检查 cuda_ext kernel 是否正确实现（数值精度）
  2. 若仅 TPS 不达标，考虑优化 kernel 内存访问
  3. 可降低门槛（修改本脚本顶部的 THRESHOLD_* 常量）