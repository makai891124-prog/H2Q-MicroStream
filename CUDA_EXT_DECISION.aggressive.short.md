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
| 10 | sta_v2 | 20938 | 5.12985 | 98.62 | 39.0 | n/a |
| 10 | packbits | 42845 | 4.89967 | 18.17 | 32.9 | packbits |
| 10 | cuda_ext | 78389 | 5.04453 | 8.39 | 25.7 | cuda_ext |
| 30 | sta_v2 | 25841 | 4.33032 | 24.06 | 39.0 | n/a |
| 30 | packbits | 43969 | 4.17004 | 13.78 | 32.9 | packbits |
| 30 | cuda_ext | 80618 | 4.23985 | 7.96 | 25.7 | cuda_ext |
| 60 | sta_v2 | 26582 | 3.65187 | 21.81 | 39.0 | n/a |
| 60 | packbits | 44249 | 3.61378 | 12.81 | 32.9 | packbits |
| 60 | cuda_ext | 75186 | 3.65371 | 8.18 | 25.7 | cuda_ext |

## cuda_ext vs packbits 差值

| steps | loss_delta | tps_ratio | p99_ratio | vram_delta_MB | verdict |
|------:|-----------:|----------:|----------:|--------------:|---------|
| 10 | 0.144854 | 1.8296 | 0.4621 | -7.14 | ❌ FAIL |
| 30 | 0.069813 | 1.8335 | 0.5774 | -7.14 | ❌ FAIL |
| 60 | 0.039933 | 1.6992 | 0.6386 | -7.14 | ✅ PASS |

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