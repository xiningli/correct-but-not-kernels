# Kernel engineering pilot report

<!-- BEGIN GENERATED GATES -->
| Gate | Status | Record |
|---|---|---|
| C0 | PASS | `runs/gates/c0-20261004T224213.417987Z-7b8a8dac.json` |
| C1 | PASS | `runs/gates/c1-20261005T023604.299506Z-353e8c5c.json` |
| C2 | PASS | `runs/gates/c2-20261005T020938.981676Z-f6b7812e.json` |
| C3 | PASS | `runs/gates/c3-20261005T022109.670740Z-bd951734.json` |
| C4 | PASS | `runs/gates/c4-20261005T032853.993235Z-2fac6866.json` |
| C5 | PASS | `runs/gates/c5-20261005T035544.370523Z-bb47b38c.json` |
| C6 | PASS | `runs/gates/c6-20261005T165633.936013Z-88fa5e5d.json` |

C2: CV across 30 median estimates = 0.0145%; maximum within-pair raw CV = 7.40%. Raw samples were retained, including outliers.
C3: 4/4 deliberate cheats pass naive verification and are rejected by the hardened evaluator.
C4 gemma-4-12B-it: mean P_signal = 0.9880; tasks with nonzero pass@8 = 1.
C4 gemma-4-E4B-it: mean P_signal = 0.8515; tasks with nonzero pass@8 = 1.
<!-- END GENERATED GATES -->
## Base-policy TRAIN signal history

| Attempt | Model | Task | Correct / n | pass@1 | pass@8 | P_signal |
|---|---|---|---:|---:|---:|---:|
| 1 | gemma-4-12B-it | level1/1_Square_matrix_multiplication_ | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-12B-it | level1/2_Standard_matrix_multiplication_ | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-12B-it | level1/40_LayerNorm | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-12B-it | level1/88_MinGPTNewGelu | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-E4B-it | level1/1_Square_matrix_multiplication_ | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-E4B-it | level1/2_Standard_matrix_multiplication_ | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-E4B-it | level1/40_LayerNorm | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 1 | gemma-4-E4B-it | level1/88_MinGPTNewGelu | 32 / 32 | 1.0000 | 1.0000 | 0.0000 |
| 2 | gemma-4-12B-it | level1/40_LayerNorm | 20 / 32 | 0.6250 | 1.0000 | 0.9880 |
| 2 | gemma-4-12B-it | level1/88_MinGPTNewGelu | 0 / 32 | 0.0000 | 0.0000 | 0.0000 |
| 2 | gemma-4-E4B-it | level1/40_LayerNorm | 7 / 32 | 0.2188 | 0.8972 | 0.8972 |
| 2 | gemma-4-E4B-it | level1/88_MinGPTNewGelu | 0 / 32 | 0.0000 | 0.0000 | 0.0000 |
| 3 | gemma-4-12B-it | level1/40_LayerNorm | 20 / 32 | 0.6250 | 1.0000 | 0.9880 |
| 3 | gemma-4-E4B-it | level1/40_LayerNorm | 6 / 32 | 0.1875 | 0.8515 | 0.8515 |

Attempt 1: 714.0 rollouts/hour end to end, including model loading/generation and serial evaluation.
Attempt 2: 469.0 rollouts/hour end to end, including model loading/generation and serial evaluation.
Attempt 3: 375.4 rollouts/hour end to end, including model loading/generation and serial evaluation.

These are TRAIN selection measurements. Attempt 2 changes the prompt and task subset; it is not a paired model-improvement experiment.

Each cell is an estimate with a 95% percentile bootstrap interval (2,000 task-cluster resamples).
One generated program per task is a noisy pilot estimate. Fresh input checks are not counted as independent programs.
Speed is the geometric mean conditional on correctness; changing the successful task set can change it.
Empirical bootstrap intervals at all-pass/all-fail boundaries can be degenerate.

| Policy | Level | Tasks | Correctness (fast0) | fast1 | Correct-only speedup | Detector fraction |
|---|---|---:|---|---|---|---|
| base | level1 | 20 | 0.6000 [0.3500, 0.8000] | 0.3000 [0.1000, 0.5000] | 1.0004 [0.9994, 1.0016] | 0.0000 [0.0000, 0.0000] |
| base | level2 | 100 | 0.5100 [0.4100, 0.6100] | 0.2000 [0.1300, 0.2800] | 1.0273 [1.0031, 1.0568] | 0.0100 [0.0000, 0.0300] |
| C-step-10 | level1 | 20 | 0.7000 [0.5000, 0.9000] | 0.2000 [0.0500, 0.4000] | 1.0012 [0.9946, 1.0093] | 0.0000 [0.0000, 0.0000] |
| C-step-10 | level2 | 100 | 0.6700 [0.5800, 0.7700] | 0.2100 [0.1300, 0.3000] | 0.9983 [0.9681, 1.0265] | 0.0000 [0.0000, 0.0000] |
| C-step-20 | level1 | 20 | 0.7500 [0.5500, 0.9500] | 0.1500 [0.0000, 0.3000] | 1.0074 [0.9992, 1.0205] | 0.0000 [0.0000, 0.0000] |
| C-step-20 | level2 | 100 | 0.6100 [0.5100, 0.7100] | 0.2500 [0.1700, 0.3400] | 1.0386 [0.9860, 1.1289] | 0.0200 [0.0000, 0.0500] |
| C-step-30 | level1 | 20 | 0.8000 [0.6000, 0.9500] | 0.4000 [0.2000, 0.6000] | 1.0020 [0.9993, 1.0069] | 0.0500 [0.0000, 0.1500] |
| C-step-30 | level2 | 100 | 0.6900 [0.6000, 0.7800] | 0.2400 [0.1600, 0.3300] | 1.0623 [0.9943, 1.1954] | 0.0100 [0.0000, 0.0300] |
| C | level1 | 20 | 0.7000 [0.5000, 0.9000] | 0.4000 [0.2000, 0.6000] | 1.0284 [1.0001, 1.0894] | 0.0000 [0.0000, 0.0000] |
| C | level2 | 100 | 0.7100 [0.6200, 0.8000] | 0.2900 [0.2000, 0.3800] | 1.0476 [0.9716, 1.1810] | 0.0200 [0.0000, 0.0500] |
| S-step-10 | level1 | 20 | 0.6500 [0.4000, 0.8500] | 0.2000 [0.0500, 0.4000] | 1.0031 [0.9994, 1.0098] | 0.0000 [0.0000, 0.0000] |
| S-step-10 | level2 | 100 | 0.6300 [0.5300, 0.7300] | 0.2100 [0.1300, 0.2900] | 1.0087 [0.9596, 1.0559] | 0.0200 [0.0000, 0.0500] |
| S-step-20 | level1 | 20 | 0.8000 [0.6000, 0.9500] | 0.3500 [0.1500, 0.5500] | 0.8646 [0.6008, 1.0596] | 0.0000 [0.0000, 0.0000] |
| S-step-20 | level2 | 100 | 0.7500 [0.6700, 0.8300] | 0.3000 [0.2100, 0.3900] | 1.0297 [1.0039, 1.0634] | 0.0200 [0.0000, 0.0500] |
| S-step-30 | level1 | 20 | 0.8500 [0.7000, 1.0000] | 0.4500 [0.2500, 0.6500] | 0.9993 [0.9977, 1.0005] | 0.0000 [0.0000, 0.0000] |
| S-step-30 | level2 | 100 | 0.7200 [0.6300, 0.8100] | 0.2600 [0.1800, 0.3400] | 1.0859 [0.9963, 1.2589] | 0.0100 [0.0000, 0.0300] |
| S | level1 | 20 | 0.8500 [0.7000, 1.0000] | 0.3500 [0.1500, 0.5500] | 0.9993 [0.9979, 1.0003] | 0.0000 [0.0000, 0.0000] |
| S | level2 | 100 | 0.7300 [0.6400, 0.8100] | 0.2900 [0.2000, 0.3800] | 1.0060 [0.9870, 1.0249] | 0.0100 [0.0000, 0.0300] |

| Paired correctness change vs base | Level | Estimate [95% CI] |
|---|---|---|
| C | level1 | 0.1000 [-0.1000, 0.3000] |
| C | level2 | 0.2000 [0.0700, 0.3200] |
| S | level1 | 0.2500 [0.0000, 0.5500] |
| S | level2 | 0.2200 [0.0900, 0.3400] |

These short pilots estimate effects; detector flags require source/runtime review before claims of exploitation.
K1/K4 evidence is given by paired correctness intervals above. K2 requires reading speed and correctness together;
conditional speed alone is insufficient to claim an improved general kernel engineer.
