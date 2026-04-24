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
| 40 | sta_v2 | 22566 | 3.84423 | 102.93 | 39.0 | n/a |
| 40 | packbits | 37931 | 3.70104 | 19.06 | 32.9 | packbits |
| 40 | cuda_ext | 71530 | 3.74549 | 9.36 | 25.7 | cuda_ext |
| 120 | sta_v2 | 26610 | 3.47319 | 22.90 | 39.0 | n/a |
| 120 | packbits | 43511 | 3.50826 | 13.59 | 32.9 | packbits |
| 120 | cuda_ext | 72654 | 3.50744 | 8.69 | 25.7 | cuda_ext |
| 240 | sta_v2 | 26911 | 3.09994 | 22.68 | 39.0 | n/a |
| 240 | packbits | 42903 | 3.36275 | 14.63 | 32.9 | packbits |
| 240 | cuda_ext | 70340 | 3.31563 | 9.23 | 25.7 | cuda_ext |

## cuda_ext vs packbits 差值

| steps | loss_delta | tps_ratio | p99_ratio | vram_delta_MB | verdict |
|------:|-----------:|----------:|----------:|--------------:|---------|
| 40 | 0.04445 | 1.8858 | 0.4909 | -7.14 | ✅ PASS |
| 120 | -0.000827 | 1.6698 | 0.6394 | -7.14 | ✅ PASS |
| 240 | -0.047122 | 1.6395 | 0.6308 | -7.14 | ✅ PASS |

## 门槛定义
- loss_delta ≤ 0.05
- TPS 提升 ≥ 110%（vs packbits）
- p99 回退 ≤ 20%
- VRAM 额外开销 ≤ 50.0 MB

## 最终结论

**整体判决: SWITCH**

✅ 建议将 cuda_ext 设为默认 binary_backend。
所有 step 时长均满足:
  • loss delta ≤ 0.05 (损失无显著回退)
  • TPS gain ≥ 110% (相比 packbits 明显提速)
  • p99 回退 ≤ 20% (尾部延迟可接受)
  • VRAM 额外开销 ≤ 50.0 MB
实施: sta_core_v2.py 中 binary_backend 默认值改为 'cuda_ext'