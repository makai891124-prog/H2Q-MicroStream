# Final Binary STA Report

## Acceptance
- core_checks: {'small_correctness': True, 'evolution_check': True, 'cuda_dummy_forward': True, 'cuda_stress_2048': True}
- h2q_binary_sta: loss=5.578939, topology_sparsity=0.492188
- trainer_binary_sta: loss=30.544941, sta_sparsity_mean=0.484375

## Benchmark 1024
- sta_v2: avg_time_ms=1.754, peak_vram_mb=209.58, last_sparsity=0.000000
- binary_sta: avg_time_ms=7.803, peak_vram_mb=132.32, last_sparsity=0.499512

## Benchmark 2048
- sta_v2: avg_time_ms=1.678, peak_vram_mb=169.42, last_sparsity=0.000000
- binary_sta: avg_time_ms=14.700, peak_vram_mb=88.63, last_sparsity=0.499756

## Analysis
binary_sta completed three-level acceptance: core correctness/evolution/GPU long sequence passed; h2q_evolution and agi_joint_trainer are both configurable and passed smoke tests; benchmark provides direct comparison with sta_v2 for further default path decisions.
