# EXPERIMENT_REPORT

## Diagnosis
- **Device**: NVIDIA GeForce RTX 4070 Ti SUPER
- **Torch Version**: 2.5.1+cu121
- **Torch CUDA Version**: 12.1
- **Steps**: [10, 30, 60]
- **CUDA Extension Compiled**: True

## File: cuda_ext_protocol_result.conservative.short.json
- **Overall Decision**: KEEP_PACKBITS

| Steps | Verdict | Loss Delta | TPS Ratio | P99 Ratio | VRAM Delta (MB) |
|-------|---------|------------|-----------|-----------|-----------------|
| 10 | FAIL | 0.144854 | 1.8873 | 0.4802 | -7.14 |
| 30 | FAIL | 0.069813 | 1.8913 | 0.5362 | -7.14 |
| 60 | PASS | 0.039933 | 1.9139 | 0.5437 | -7.14 |

## File: cuda_ext_protocol_result.conservative.long.json
- **Overall Decision**: KEEP_PACKBITS

| Steps | Verdict | Loss Delta | TPS Ratio | P99 Ratio | VRAM Delta (MB) |
|-------|---------|------------|-----------|-----------|-----------------|
| 40 | FAIL | 0.000000 | 1.0159 | 0.8125 | 0.00 |
| 120 | FAIL | 0.000000 | 0.9770 | 1.1430 | 0.00 |
| 240 | FAIL | 0.000000 | 0.9733 | 0.9947 | 0.00 |

## File: cuda_ext_protocol_result.aggressive.short.json
- **Overall Decision**: KEEP_PACKBITS

| Steps | Verdict | Loss Delta | TPS Ratio | P99 Ratio | VRAM Delta (MB) |
|-------|---------|------------|-----------|-----------|-----------------|
| 10 | FAIL | 0.144854 | 1.8296 | 0.4621 | -7.14 |
| 30 | FAIL | 0.069813 | 1.8335 | 0.5774 | -7.14 |
| 60 | PASS | 0.039933 | 1.6992 | 0.6386 | -7.14 |

## File: cuda_ext_protocol_result.aggressive.long.json
- **Overall Decision**: SWITCH

| Steps | Verdict | Loss Delta | TPS Ratio | P99 Ratio | VRAM Delta (MB) |
|-------|---------|------------|-----------|-----------|-----------------|
| 40 | PASS | 0.044450 | 1.8858 | 0.4909 | -7.14 |
| 120 | PASS | -0.000827 | 1.6698 | 0.6394 | -7.14 |
| 240 | PASS | -0.047122 | 1.6395 | 0.6308 | -7.14 |

## Final Recommendation
建议保持默认 packbits 并使用 conservative inference-long 策略 (Recommend keep default packbits and use conservative inference-long policy)
