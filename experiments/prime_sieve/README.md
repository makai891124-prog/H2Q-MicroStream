# Prime Sieve Experiments

本目录收纳与“Primorial Base / Adaptive Wheel”素数筛实验直接相关的源码、报告与复现说明，避免根目录继续堆积编译产物和实验稿件。

## 目录结构

- `src/prime_primorial_mvp.cpp`
  - 6 轮（2x3）位图筛 MVP。
- `src/benchmark_prime_algorithms.cpp`
  - 多算法对照基准，包含 `full_byte`、`odd_byte`、`wheel6_bit`、`segmented_odd`、`adaptive_wheel`。
- `reports/ADAPTIVE_WHEEL_ANALYSIS_REPORT_CN.md`
  - 动态轮基扩展实验的详细中文分析报告。

## 编译

Windows + MSYS2 UCRT64:

```powershell
& 'C:/msys64/ucrt64/bin/g++.exe' -O3 -std=c++17 -march=native `
  'd:/H2Q-MicroStream/experiments/prime_sieve/src/prime_primorial_mvp.cpp' `
  -o 'd:/H2Q-MicroStream/experiments/prime_sieve/prime_primorial_mvp.exe'

& 'C:/msys64/ucrt64/bin/g++.exe' -O3 -std=c++17 -march=native `
  'd:/H2Q-MicroStream/experiments/prime_sieve/src/benchmark_prime_algorithms.cpp' `
  -o 'd:/H2Q-MicroStream/experiments/prime_sieve/benchmark_prime_algorithms.exe'
```

## 运行

```powershell
$env:PATH = 'C:/msys64/ucrt64/bin;' + $env:PATH
& 'd:/H2Q-MicroStream/experiments/prime_sieve/prime_primorial_mvp.exe'
& 'd:/H2Q-MicroStream/experiments/prime_sieve/benchmark_prime_algorithms.exe'
```

## 当前结论摘要

- `wheel6_bit` 在 `10^8` 下速度和内存都很强。
- `adaptive_wheel` 在 `10^9` 下保持了低内存，但当前实现仍慢于 `segmented_odd`。
- 下一步优化重点应放在 `wheel_pattern` 的 bitset 化、段内首址缓存和更大的 segment 调参。