# Multi-seed Multi-duration Training Protocol

- device: cuda:0
- config: {'seeds': [11, 23, 37], 'step_budgets': [40, 120, 240], 'batch_size': 4, 'seq_len': 128, 'dim': 128, 'layers': 4, 'lr': 0.0003, 'binary_backend': 'cuda_ext', 'corpus': 'data/open_corpus/open_corpus.txt'}

## Summary
- steps=40
  sta_v2: last_loss=4.014441, tps=21352.50, vram=39.03, p99=107.879
  binary: last_loss=3.913444, tps=34745.02, vram=32.88, p99=21.794
  delta: loss=-0.100997, tps_ratio=1.6272x, vram=-6.15, p99=-86.085
- steps=120
  sta_v2: last_loss=3.454201, tps=23147.64, vram=39.03, p99=33.286
  binary: last_loss=3.575951, tps=38048.85, vram=32.88, p99=21.842
  delta: loss=+0.121749, tps_ratio=1.6437x, vram=-6.15, p99=-11.445
- steps=240
  sta_v2: last_loss=3.013451, tps=25442.41, vram=39.03, p99=26.839
  binary: last_loss=3.292596, tps=42337.05, vram=32.88, p99=15.459
  delta: loss=+0.279145, tps_ratio=1.6640x, vram=-6.15, p99=-11.380

## CUDA Hotspots: STA_V2
- -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
-                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
- -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
-                                                aten::mm         4.58%      49.987ms         4.58%      49.987ms      28.337us      55.115ms         4.75%      55.115ms      31.244us           0 b           0 b     165.75 Mb     165.75 Mb          1764  
-                                         aten::transpose         3.76%      41.026ms         4.00%      43.706ms      13.149us      45.845ms         3.95%      66.968ms      20.147us           0 b           0 b           0 b           0 b          3324  
-                                  aten::_index_put_impl_         3.54%      38.615ms         7.43%      81.150ms     140.885us      44.930ms         3.87%      96.011ms     166.686us           0 b        -168 b           0 b    -480.00 Kb           576  
-                                               aten::mul         3.46%      37.748ms         3.46%      37.748ms      15.420us      43.243ms         3.73%      43.243ms      17.665us         120 b         120 b     319.03 Mb     319.03 Mb          2448  
-                                                 aten::t         3.67%      40.036ms         7.15%      78.014ms      26.535us      40.922ms         3.53%     101.438ms      34.503us           0 b           0 b           0 b           0 b          2940  
-                                        aten::as_strided         0.51%       5.544ms         0.51%       5.544ms       0.887us      39.679ms         3.42%      39.679ms       6.347us           0 b           0 b           0 b           0 b          6252  
-                                           aten::reshape         3.18%      34.745ms         3.76%      41.053ms      14.435us      35.439ms         3.05%      63.214ms      22.227us           0 b           0 b           0 b           0 b          2844  
-                                              aten::view         0.61%       6.632ms         0.61%       6.632ms       2.117us      30.386ms         2.62%      30.386ms       9.702us           0 b           0 b           0 b           0 b          3132  
-                                       aten::result_type         0.07%     769.600us         0.07%     769.600us       0.153us      29.589ms         2.55%      29.589ms       5.899us           0 b           0 b           0 b           0 b          5016  
-                                             aten::copy_         2.48%      27.113ms         2.48%      27.113ms      15.912us      28.812ms         2.48%      28.812ms      16.908us           0 b           0 b           0 b           0 b          1704  
-                                             MmBackward0         3.41%      37.188ms        10.64%     116.133ms     197.505us      27.473ms         2.37%     123.194ms     209.514us           0 b           0 b      83.25 Mb           0 b           588  
-                               Optimizer.step#AdamW.step         3.52%      38.431ms        13.88%     151.526ms      12.627ms      25.051ms         2.16%     151.323ms      12.610ms         268 b         -48 b     658.00 Kb      -3.86 Mb            12  
-                                              aten::add_         1.11%      12.142ms         1.11%      12.142ms       7.076us      20.304ms         1.75%      20.304ms      11.832us           0 b           0 b           0 b           0 b          1716  
-                                             aten::empty         0.70%       7.597ms         0.70%       7.597ms       2.699us      20.175ms         1.74%      20.175ms       7.167us      94.31 Kb      92.31 Kb     322.46 Mb     322.46 Mb          2815  
-                                            aten::matmul         1.92%      20.957ms         4.46%      48.660ms      82.754us      19.497ms         1.68%      56.608ms      96.272us           0 b           0 b      82.50 Mb           0 b           588  
- -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
- Self CPU time total: 1.092s

## CUDA Hotspots: Binary
- -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
-                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
- -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
-                                                aten::mm         4.09%      26.918ms         4.09%      26.918ms      28.395us      51.246ms         7.10%      51.246ms      54.057us           0 b           0 b     128.25 Mb     128.25 Mb           948  
-                                        aten::as_strided         0.53%       3.493ms         0.53%       3.493ms       0.864us      26.490ms         3.67%      26.490ms       6.549us           0 b           0 b           0 b           0 b          4045  
-                                       aten::result_type         0.07%     443.000us         0.07%     443.000us       0.143us      25.601ms         3.55%      25.601ms       8.269us           0 b           0 b           0 b           0 b          3096  
-                                             aten::copy_         6.67%      43.947ms         6.67%      43.947ms      60.533us      24.248ms         3.36%      24.248ms      33.399us           0 b           0 b           0 b           0 b           726  
-                                                 aten::t         3.53%      23.253ms         6.97%      45.902ms      28.300us      23.162ms         3.21%      55.663ms      34.318us           0 b           0 b           0 b           0 b          1622  
-                                         aten::transpose         3.43%      22.623ms         3.64%      24.015ms      13.979us      22.347ms         3.10%      33.873ms      19.717us           0 b           0 b           0 b           0 b          1718  
-                                           aten::reshape         2.45%      16.149ms         2.93%      19.288ms      13.488us      20.927ms         2.90%      29.647ms      20.732us           0 b           0 b           0 b           0 b          1430  
-                                             aten::index         2.56%      16.857ms         4.79%      31.560ms      74.964us      19.777ms         2.74%      38.501ms      91.451us           0 b           0 b      49.70 Mb      49.67 Mb           421  
-                               aten::_local_scalar_dense        10.88%      71.694ms        10.88%      71.694ms      60.964us      18.654ms         2.58%      18.654ms      15.862us           0 b           0 b           0 b           0 b          1176  
-                                          aten::_to_copy         1.88%      12.400ms         8.29%      54.627ms     107.533us      18.354ms         2.54%      43.421ms      85.474us           0 b           0 b     113.14 Mb           0 b           508  
-                                               aten::bmm         1.32%       8.722ms         1.32%       8.722ms      45.425us      17.418ms         2.41%      17.418ms      90.719us           0 b           0 b      36.00 Mb      36.00 Mb           192  
-                                           aten::nonzero         3.43%      22.588ms         3.89%      25.650ms     346.622us      16.236ms         2.25%      18.856ms     254.811us           0 b           0 b      67.00 Kb           0 b            74  
-                                               aten::add         1.25%       8.236ms         1.25%       8.236ms      13.177us      15.759ms         2.18%      15.759ms      25.214us           0 b           0 b      63.30 Mb      63.30 Mb           625  
-                                             aten::empty         0.54%       3.525ms         0.54%       3.525ms       2.358us      15.126ms         2.10%      15.126ms      10.118us      96.21 Kb      96.21 Kb     356.07 Mb     356.07 Mb          1495  
-                                              aten::view         0.63%       4.123ms         0.63%       4.123ms       2.046us      14.700ms         2.04%      14.700ms       7.295us           0 b           0 b           0 b           0 b          2015  
- -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
- Self CPU time total: 658.858ms

## Recommendation Rule
- If binary keeps loss within +0.05 and gives >=1.2x throughput in medium and long budgets, switch default to binary.
- If p99 latency regresses >20% while throughput gain <10%, keep sta_v2 as default and continue kernel-level tuning.
