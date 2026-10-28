# Frozen-program consumer GPU replication

This add-on tests whether numerical correctness gains observed on an RTX PRO
6000 Blackwell remain observable on an RTX 5070 Ti. Both devices have compute
capability 12.0: this is same-architecture platform replication. It does not
establish cross-architecture generalization or isolate hardware from driver,
container, resource-budget and host differences.

The task is queued after local runtime controls and before any new model
generation. It uses spare local GPU capacity; it does not require AutoDL to
be powered on. The independent-defect repair study remains the main research
priority. Its future frozen patches can receive a separate second-device check.

## Frozen scope and comparison

- All 100 original KernelBench Level-2 tasks, eight original samples per task.
- Shared base, three correctness-reward RL seeds (`C`) and three SFT seeds:
  5,600 logical attempts before common reference-based resource exclusions.
- No regenerated outputs, edited programs, checkpoint selection, or filtering
  by previous success. Format errors remain in the denominator.
- The primary matched source is the 5,600-program v4 decomposition. Reuse its
  exact four input seeds, two forward calls per case, numerical tolerance and
  observation worker. Legacy v1.1 scores are preserved as historical context,
  not treated as measurements under an identical evaluator.
- Host mathematical computation was legal in the original prompts. Report
  numerical and strict custom-kernel outcomes separately. The former alone
  cannot substantiate improved custom CUDA implementation ability.

## Qualification and outcomes

The plan is frozen before local candidate outcomes. A task qualifies only if
the reference runs at all eight original seeds, its four cases are finite and
repeatable, and it fits the declared 16GB-workstation resource envelope.
The same exclusion applies to every policy and sample of that task. Keep the
original input shapes and dtypes. Report coverage and all exclusion reasons.
Unexpected reference infrastructure errors stop the run for investigation.

Candidate OOM, timeout, compile errors and format errors are retained. Numerical
outcomes that were never observed are `null`, not asserted to be wrong numerical
values. The operational numerical pass rate counts only observed passes over
all attempts on qualifying tasks. Report unknown counts alongside it.

Report local and remote pass rates, both-device passes, and remote-pass to
local-not-pass transitions. For C−base, SFT−base and C−SFT, average the three
trained seeds within each task and use task-cluster bootstrap intervals at
98.333% confidence (three comparisons). Also report the paired change in each
effect between devices. Shared base is counted once; repeated samples and
training seeds are not extra independent tasks. Intervals condition on these
checkpoints and tasks. Do not infer equivalence from non-significance.

Only complete task grids enter aggregate rates and intervals. Intermediate
reports are descriptive; the schedule and stopping rules never depend on
whether the method looks better. Analyze discordant programs after completion,
separating numerical, compiler, resource and synchronization issues; a
mechanism requires a reproducer. There is no speed endpoint in this version.

## Execution and artifacts

The image is pinned by Docker image ID. Read-only mounts reuse the existing
Python 3.12 / Torch 2.13.0+cu130 / Triton 3.7.1 environment, with CUDA toolkit
13.0.88 in the image. A short-lived unprivileged container receives only its
program, input tensors, output-shape contract and trusted observation runner.
It receives no reference implementation, oracle outputs, model weights,
credentials, other project directories, host PID namespace or Docker socket.
Network, memory, process, CPU, file, disk and wall-time limits apply.
Container/observer controls are new evidence, not a reuse of an old C1 PASS.
This is bounded operational containment, not a proof against arbitrary native
attacks or GPU driver vulnerabilities.

The controller pauses before starting a worker when another GPU compute
process is present. It stops only its own containers, writes partial/terminal
reports, and leaves the workstation and unrelated services running.

```bash
python3 -m research.portability_v1.prepare --out runs/portability-v1-20261020/prepared-r3
/mnt/fastdata/glomergence/.venv/bin/python -m research.portability_v1.run \
  --prepared runs/portability-v1-20261020/prepared-r3 \
  --out runs/portability-v1-20261020/run-r3
```

`manifest.json` binds inputs and evaluator source. `progress.json`,
`eligibility.json`, append-only `metrics.jsonl`, `REPORT.json`, `REPORT.md` and
`evidence.tgz` record execution. Resume only the same preparation and output
directory. Saved scores are not regenerated; a changed implementation requires
a new preparation revision. Identical source/seed executions may be shared
within a task, with every logical policy attempt still recorded.

Related work:
[KernelBench §4.4](https://arxiv.org/html/2502.10517v1#S4.SS4)
already studies cross-hardware behavior. The contribution here is testing
retention of the project's intervention effects and explaining any failures.
[NVIDIA's device table](https://developer.nvidia.com/cuda/gpus)
places both GPUs at compute capability 12.0.
