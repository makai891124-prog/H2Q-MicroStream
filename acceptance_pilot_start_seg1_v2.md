# Dynamic Acceptance Report

- UTC: 2026-04-20T06:41:38.287521+00:00
- Telemetry: evolution_telemetry_pilot_start_seg1.csv
- Verdict: **REJECT**

## Gate A (base thresholds)
- pass: False
- ema_last=2.844601 <= val_loss_max=1.805137
- vram_last_gb=0.017821 <= vram_max_gb=0.250000
- tokens_per_sec=139222.08 >= tokens_per_sec_min=16213.00
- tps_available=True

## Gate B (window stability)
- pass: False
- max_loss_cv=N/A (threshold=0.100000)
- max_sparsity_cv=N/A (threshold=0.200000)
- window=20

## Gate C (topology state)
- pass: False
- sparsity_peak=0.024895 (target=0.500000)
- svd_last=2.891209 (min=1.600000)
- phase_trigger_count=0 (max=3)

## Gate D (hypothesis support)
- pass: False
- checked=0, supported=0, support_rate=0.000000

