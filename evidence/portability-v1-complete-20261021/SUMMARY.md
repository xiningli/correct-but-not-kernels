# Consumer GPU replay: verified completion

Completed 4,368 attempts on 78 common tasks in 5.13 hours.
Excluded 22 tasks by frozen reference/storage limits: {'reference_total_storage_cap': 6, 'disk_limit': 13, 'reference_output_storage_cap': 3}.

| Policy family | Attempts | RTX PRO 6000 pass | RTX 5070 Ti pass | Local unknown |
|---|---:|---:|---:|---:|
| base | 624 | 54.01% | 54.01% | 223 |
| C | 1872 | 71.63% | 71.69% | 355 |
| SFT | 1872 | 71.42% | 71.47% | 364 |

Same 78 task grids on both platforms. Unknowns remain in denominators.
C and SFT each average the three equally sized training-seed evaluations.

| Local contrast | Percentage-point change | 98.333% task-bootstrap interval |
|---|---:|---:|
| C − base | +17.68 | [+12.39, +23.13] |
| SFT − base | +17.47 | [+12.39, +22.49] |
| C − SFT | +0.21 | [-2.88, +3.47] |

Strict custom-kernel successes: 0 / 4368.
Numerical three-state disagreements: 8; details are in audit.json.
The pass gains persist on this common resource-qualified subset. This is same-architecture platform replication.
It does not establish new-task transfer, acceleration, improved custom-kernel skill or RL superiority to SFT.
The 22 exclusions reflect temporary/output-storage policy; they do not establish that these tasks cannot run on a 16GB GPU.
No new generation, training or GPU rerun was used in this audit.

Metrics, input bindings, summary counts, bootstrap effects and terminal archive hash verified.
Reproduce with `python3 -m eval.portability_audit --prepared PREPARED --run RUN --out NEW_DIRECTORY`.
