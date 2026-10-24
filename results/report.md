# AT-VMD Experiment Results

Five seeds: 2021-2025. Values are mean +/- population standard deviation
across seeds (ddof=0); lower MSE and MAE are better. Input length 336,
chronological 60/20/20 split, train-only z-score standardization, batch 64.
Training: 10 epochs maximum, patience 5; forecaster LR 1e-3 and encoder LR 1e-4.
These are recorded experimental results, not new runs performed for this release.

## Forecasting Comparison

### ENTSO-E Belgium

Each cell contains MSE / MAE.

| Model | H=96 | H=192 | H=336 |
|---|---|---|---|
| AT-VMD | 0.1273 +/- 0.0091 / 0.2534 +/- 0.0106 | 0.1682 +/- 0.0163 / 0.2994 +/- 0.0171 | 0.2187 +/- 0.0072 / 0.3469 +/- 0.0055 |
| VMDNet | 0.1426 +/- 0.0049 / 0.2703 +/- 0.0058 | 0.1901 +/- 0.0045 / 0.3176 +/- 0.0033 | 0.2273 +/- 0.0154 / 0.3577 +/- 0.0151 |
| DU-VMD | 0.1410 +/- 0.0053 / 0.2739 +/- 0.0060 | 0.2011 +/- 0.0062 / 0.3281 +/- 0.0056 | 0.2337 +/- 0.0047 / 0.3619 +/- 0.0052 |
| DLinear | 0.1693 +/- 0.0007 / 0.2838 +/- 0.0006 | 0.2206 +/- 0.0055 / 0.3264 +/- 0.0054 | 0.2908 +/- 0.0071 / 0.3754 +/- 0.0058 |
| PatchTST | 0.1626 +/- 0.0052 / 0.2824 +/- 0.0037 | 0.1963 +/- 0.0041 / 0.3209 +/- 0.0037 | 0.2297 +/- 0.0086 / 0.3521 +/- 0.0065 |
| iTransformer | 0.1333 +/- 0.0012 / 0.2630 +/- 0.0008 | 0.1841 +/- 0.0017 / 0.3173 +/- 0.0016 | 0.2853 +/- 0.0044 / 0.3890 +/- 0.0023 |
| Persistence | 1.6056 / 1.0265 | 1.6313 / 1.0312 | 1.7671 / 1.0720 |

### ENTSO-E DE-LU

Each cell contains MSE / MAE.

| Model | H=96 | H=192 | H=336 |
|---|---|---|---|
| AT-VMD | 0.1101 +/- 0.0031 / 0.2240 +/- 0.0026 | 0.1289 +/- 0.0036 / 0.2472 +/- 0.0050 | 0.1559 +/- 0.0078 / 0.2822 +/- 0.0052 |
| VMDNet | 0.1344 +/- 0.0026 / 0.2436 +/- 0.0019 | 0.1729 +/- 0.0097 / 0.2816 +/- 0.0110 | 0.1940 +/- 0.0089 / 0.3116 +/- 0.0090 |
| DU-VMD | 0.1222 +/- 0.0014 / 0.2348 +/- 0.0016 | 0.1565 +/- 0.0137 / 0.2699 +/- 0.0139 | 0.1876 +/- 0.0121 / 0.3044 +/- 0.0118 |
| DLinear | 0.1514 +/- 0.0026 / 0.2510 +/- 0.0038 | 0.1794 +/- 0.0006 / 0.2783 +/- 0.0007 | 0.2275 +/- 0.0015 / 0.3198 +/- 0.0016 |
| PatchTST | 0.1192 +/- 0.0011 / 0.2331 +/- 0.0019 | 0.1466 +/- 0.0012 / 0.2647 +/- 0.0024 | 0.1734 +/- 0.0029 / 0.2944 +/- 0.0017 |
| iTransformer | 0.1158 +/- 0.0020 / 0.2300 +/- 0.0022 | 0.1368 +/- 0.0037 / 0.2559 +/- 0.0027 | 0.1484 +/- 0.0016 / 0.2740 +/- 0.0020 |
| Persistence | 1.5583 / 1.0127 | 1.5308 / 1.0004 | 1.6320 / 1.0359 |

### AEMO NSW

Each cell contains MSE / MAE.

| Model | H=96 | H=192 | H=336 |
|---|---|---|---|
| AT-VMD | 0.3395 +/- 0.0043 / 0.4040 +/- 0.0047 | 0.3939 +/- 0.0194 / 0.4426 +/- 0.0134 | 0.4242 +/- 0.0199 / 0.4644 +/- 0.0125 |
| VMDNet | 0.3617 +/- 0.0112 / 0.4216 +/- 0.0081 | 0.4082 +/- 0.0080 / 0.4540 +/- 0.0049 | 0.4421 +/- 0.0136 / 0.4761 +/- 0.0088 |
| DU-VMD | 0.3655 +/- 0.0083 / 0.4205 +/- 0.0048 | 0.4170 +/- 0.0150 / 0.4560 +/- 0.0098 | 0.4400 +/- 0.0211 / 0.4746 +/- 0.0140 |
| DLinear | 0.3673 +/- 0.0003 / 0.4086 +/- 0.0000 | 0.4083 +/- 0.0006 / 0.4339 +/- 0.0001 | 0.4612 +/- 0.0010 / 0.4648 +/- 0.0004 |
| PatchTST | 0.3976 +/- 0.0072 / 0.4326 +/- 0.0043 | 0.4727 +/- 0.0071 / 0.4719 +/- 0.0041 | 0.4865 +/- 0.0103 / 0.4869 +/- 0.0037 |
| iTransformer | 0.3518 +/- 0.0071 / 0.4079 +/- 0.0086 | 0.4288 +/- 0.0065 / 0.4644 +/- 0.0065 | 0.4615 +/- 0.0112 / 0.4869 +/- 0.0085 |
| Persistence | 1.6612 / 1.0104 | 1.6981 / 1.0194 | 1.7604 / 1.0370 |

### PJM / EIA-930

Each cell contains MSE / MAE.

| Model | H=96 | H=192 | H=336 |
|---|---|---|---|
| AT-VMD | 0.1896 +/- 0.0040 / 0.3295 +/- 0.0028 | 0.2203 +/- 0.0170 / 0.3567 +/- 0.0147 | 0.2143 +/- 0.0070 / 0.3551 +/- 0.0035 |
| VMDNet | 0.1903 +/- 0.0103 / 0.3319 +/- 0.0108 | 0.2267 +/- 0.0147 / 0.3668 +/- 0.0153 | 0.2250 +/- 0.0098 / 0.3644 +/- 0.0067 |
| DU-VMD | 0.1902 +/- 0.0072 / 0.3315 +/- 0.0084 | 0.2349 +/- 0.0116 / 0.3703 +/- 0.0099 | 0.2181 +/- 0.0050 / 0.3564 +/- 0.0036 |
| DLinear | 0.1755 +/- 0.0003 / 0.3125 +/- 0.0005 | 0.2256 +/- 0.0009 / 0.3579 +/- 0.0004 | 0.2594 +/- 0.0003 / 0.3877 +/- 0.0001 |
| PatchTST | 0.2017 +/- 0.0023 / 0.3385 +/- 0.0029 | 0.2701 +/- 0.0074 / 0.3929 +/- 0.0059 | 0.3080 +/- 0.0084 / 0.4182 +/- 0.0098 |
| iTransformer | 0.2004 +/- 0.0099 / 0.3340 +/- 0.0081 | 0.2371 +/- 0.0070 / 0.3619 +/- 0.0059 | 0.2370 +/- 0.0128 / 0.3701 +/- 0.0104 |
| Persistence | 0.5677 / 0.5837 | 0.6325 / 0.6164 | 0.6716 / 0.6412 |

### ENTSO-E Poland (supplementary)

Each cell contains MSE / MAE.

| Model | H=96 | H=192 | H=336 |
|---|---|---|---|
| AT-VMD | 0.1472 +/- 0.0033 / 0.2542 +/- 0.0032 | 0.1793 +/- 0.0072 / 0.2801 +/- 0.0046 | 0.2277 +/- 0.0225 / 0.3235 +/- 0.0194 |
| VMDNet | 0.1687 +/- 0.0081 / 0.2694 +/- 0.0079 | 0.2275 +/- 0.0084 / 0.3086 +/- 0.0070 | 0.2254 +/- 0.0083 / 0.3236 +/- 0.0064 |
| DU-VMD | 0.1658 +/- 0.0051 / 0.2670 +/- 0.0051 | 0.2237 +/- 0.0108 / 0.3083 +/- 0.0062 | 0.2486 +/- 0.0165 / 0.3357 +/- 0.0090 |
| DLinear | 0.2052 +/- 0.0029 / 0.2769 +/- 0.0026 | 0.2554 +/- 0.0007 / 0.3107 +/- 0.0006 | 0.3296 +/- 0.0088 / 0.3677 +/- 0.0093 |
| PatchTST | 0.1690 +/- 0.0037 / 0.2720 +/- 0.0025 | 0.2300 +/- 0.0045 / 0.3195 +/- 0.0040 | 0.2826 +/- 0.0039 / 0.3651 +/- 0.0038 |
| iTransformer | 0.1517 +/- 0.0022 / 0.2610 +/- 0.0016 | 0.1982 +/- 0.0006 / 0.3062 +/- 0.0008 | 0.2614 +/- 0.0040 / 0.3533 +/- 0.0029 |
| Persistence | 1.7274 / 1.0585 | 1.7420 / 1.0616 | 1.8896 / 1.1083 |

Persistence repeats the last observed value across the forecast horizon.
It is deterministic and has no seed standard deviation.

## Component Ablation

ENTSO-E Belgium, H=336; same five seeds.

| Variant | MSE | MAE |
|---|---|---|
| AT-VMD | 0.2187 +/- 0.0072 | 0.3469 +/- 0.0055 |
| w/o task gradient | 0.2236 +/- 0.0082 | 0.3527 +/- 0.0103 |
| w/o learned initial frequencies | 0.2356 +/- 0.0109 | 0.3633 +/- 0.0080 |
| w/o window-conditional initial frequencies | 0.2383 +/- 0.0182 | 0.3684 +/- 0.0151 |
| w/o reconstruction regularizer | 0.2438 +/- 0.0165 | 0.3719 +/- 0.0116 |
| w/ adaptive bandwidth | 0.2196 +/- 0.0134 | 0.3485 +/- 0.0135 |
| w/ two sweeps | 0.2357 +/- 0.0198 | 0.3623 +/- 0.0216 |

The fixed-frequency variant uses uniform initialization; the global variant
learns one initialization shared across windows. The task-gradient variant
detaches the modes and post-sweep frequencies on the prediction path only.

## Downstream Forecaster Compatibility

Skill (%) = 100 * (1 - AT-VMD error / raw-forecaster error), computed from
five-seed mean errors per dataset, then averaged equally over the four main
datasets. Positive values favor AT-VMD with that forecaster; negative values
favor the raw forecaster. Poland is excluded from this paper summary.

| AT-VMD forecaster | Horizon | MSE skill (%) | MAE skill (%) |
|---|---|---|---|
| TCN | 96 | +6.32 | +3.23 |
| TCN | 192 | +6.36 | +1.52 |
| TCN | 336 | +2.44 | -0.56 |
| DLinear | 96 | +0.65 | +0.22 |
| DLinear | 192 | +1.09 | +0.67 |
| DLinear | 336 | +0.56 | +0.19 |
| LSTM | 96 | -7.74 | -4.19 |
| LSTM | 192 | -2.93 | -1.72 |
| LSTM | 336 | -1.12 | -0.64 |

## Computational Efficiency

ENTSO-E Belgium, H=336; NVIDIA GeForce RTX 5070 Laptop GPU, PyTorch 2.11.0+cu128.
Latency is the median at batch 1; throughput uses batch 64. Memory is peak
incremental CUDA allocation during forward, in MiB (bytes / 1024^2),
not total process/GPU memory. Host-to-device transfer is excluded.

| Model | Parameters | Latency (ms) | Throughput (windows/s) | Memory B=1 (MiB) | Memory B=64 (MiB) |
|---|---|---|---|---|---|
| AT-VMD | 422,232 | 2.251 | 11780.0 | 0.844 | 55.067 |
| DU-VMD | 415,704 | 9.186 | 5560.2 | 0.844 | 54.912 |
| VMDNet | 415,700 | 18.447 | 1369.4 | 0.827 | 53.999 |
| TCN forecaster | 136,721 | 0.481 | 69811.0 | 0.542 | 31.819 |
| DLinear | 226,464 | 0.107 | 572238.4 | 0.007 | 0.410 |
