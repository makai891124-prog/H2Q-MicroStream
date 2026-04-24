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
| 40 | sta_v2 | 21714 | 3.84423 | 101.91 | 39.0 | n/a |
| 40 | packbits | 34996 | 3.70104 | 24.51 | 32.9 | packbits |
| 40 | cuda_ext | 35610 | 3.70104 | 18.34 | 32.9 | cuda_ext |
| 120 | sta_v2 | 23713 | 3.47319 | 28.14 | 39.0 | n/a |
| 120 | packbits | 35159 | 3.50826 | 18.86 | 32.9 | packbits |
| 120 | cuda_ext | 35716 | 3.50826 | 19.34 | 32.9 | cuda_ext |
| 240 | sta_v2 | 25910 | 3.09994 | 24.59 | 39.0 | n/a |
| 240 | packbits | 42769 | 3.36275 | 14.69 | 32.9 | packbits |
| 240 | cuda_ext | 43248 | 3.36275 | 14.72 | 32.9 | cuda_ext |

## cuda_ext vs packbits 差值

| steps | loss_delta | tps_ratio | p99_ratio | vram_delta_MB | verdict |
|------:|-----------:|----------:|----------:|--------------:|---------|
| 40 | 0.0 | 1.0176 | 0.7485 | 0.0 | ❌ FAIL |
| 120 | 0.0 | 1.0158 | 1.0257 | 0.0 | ❌ FAIL |
| 240 | 0.0 | 1.0112 | 1.0019 | 0.0 | ❌ FAIL |

## 门槛定义
- loss_delta ≤ 0.05
- TPS 提升 ≥ 110%（vs packbits）
- p99 回退 ≤ 20%
- VRAM 额外开销 ≤ 50.0 MB

## 最终结论

**整体判决: KEEP_PACKBITS**

⚠️  保持 packbits 为默认（cuda_ext 未完全满足门槛）
未通过的判据:
  • steps=40: tps_ratio=1.018<1.1
  • steps=120: tps_ratio=1.016<1.1
  • steps=240: tps_ratio=1.011<1.1

下一步建议:
  1. 检查 cuda_ext kernel 是否正确实现（数值精度）
  2. 若仅 TPS 不达标，考虑优化 kernel 内存访问
  3. 可降低门槛（修改本脚本顶部的 THRESHOLD_* 常量）