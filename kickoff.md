# Build: post-train Gemma 4 into a CUDA kernel engineer with verifiable RL — a Paper Track submission for the Gemma 4 Developer Agent Competition

You are the build agent for a research project. Read this whole document before touching anything. Work in phases, pass each gate before starting the next, and stop at the stop points in §6.

**Deadline context.** This project targets the Kaggle "Google — The Gemma 4 Developer Agent Paper Track"
(https://www.kaggle.com/competitions/gemma-4-developer-agent-paper). Final submission: **2026-11-12, 11:59 PM UTC**.
The deliverable is a 3,000-word Kaggle Writeup (original, unpublished research; non-archival) plus optional public
notebook/code. Judged 0–5 on Novelty, Quality, Relevance, Verifiability, Clarity. Everything in this document is
scoped to be finishable, with the scope-cut order in §6, before that date.

## 0. Operator and working style

Operator: Xining Li — software engineer (Python/Go, distributed systems, ML infra), M.A. Statistics. This is the
operator's second project in this series; the first (glomergence, `/mnt/fastdata/glomergence`) built a GRPO pipeline
that post-trained Qwen2.5-Coder-14B to synthesize distributed protocols verified by Jepsen-style testing. Its
negative results (RL converges to reward-hacking strategy classes) motivate this pivot to a domain with a positive,
measurable objective.

Hardware plan: local dev box (RTX 5070 Ti 16 GB, mostly occupied — smoke tests only) + a rented AutoDL GPU node
(1× RTX PRO 6000) for all real runs.

How we work. These rules caught real bugs on the previous project; they are not optional.

- **Gates before runs.** No training run starts before its gate passes. Every gate is an executable script under
  `tools/gate_*.py` that prints PASS/FAIL and writes a JSON record to `runs/gates/`.
- **Failure ledger.** Every bug, dead end, and surprising result goes in `LEDGER.md`: date, symptom, root cause,
  the rule it produced.
- **Nothing enters the codebase that a hypothesis does not need.** Every module and every config flag names the
  hypothesis (K1–K4) it serves, in its docstring. If you cannot name one, do not add it; propose it under Roadmap
  in `REPORT_STATUS.md`.
- **Every number has a script.** Any number that could end up in the writeup is produced by a script under `eval/`
  from files under `runs/`. Never typed by hand.
- **Kill criteria.** Every training run declares, in its config, the conditions under which it is aborted, and
  `train/launch.py` enforces them.
- **Report what you see.** When a command fails, paste the actual error. Never guess a number, never round a
  failure up to a pass, never make a surprising result disappear by changing what is measured. §5 says which
  surprises are findings and which are bugs.

## 1. Research question and hypotheses

**Setting.** The policy is a Gemma 4 instruction-tuned model (primary: `google/gemma-4-12B-it`, trainable with LoRA
on one RTX PRO 6000 GPU; secondary: `google/gemma-4-E4B-it`, the consumer-hardware story). Tasks come from KernelBench
(github.com/ScalingIntelligence/KernelBench): given a PyTorch reference module, write a replacement (custom CUDA /
Triton / fused PyTorch) that is **correct** and **faster than eager PyTorch**. The environment compiles and runs the
generated code in a sandbox, checks correctness against the reference on fresh random inputs, and measures speedup
with controlled GPU timing. That verdict is the reward. Training is GRPO + LoRA (TRL, colocated vLLM) — the stack
validated on the previous project.

**The catch (carried over).** The evaluator is a proxy. Correctness is checked on a finite set of random inputs
with a tolerance; speedup is measured under one timing protocol. A policy can raise reward by exploiting the
evaluator instead of writing a better kernel — this actually happened to Sakana's AI CUDA Engineer. So the
evaluator's adversarial robustness is a first-class research object here, not an implementation detail.

**Question.** Can RL post-training turn a consumer-scale Gemma 4 into a kernel engineer that is correct and
measurably faster than eager PyTorch on held-out tasks — and what does it do to the evaluator on the way there?

**Hypotheses** (pre-registered; verdicts go in `REPORT.md` as confirmed / refuted / underpowered):

- **K1 — RL raises correctness.** GRPO with a correctness-only reward raises held-out correctness (fast1-style
  pass rate) over the base model on KernelBench Level 1 tasks.
- **K2 — speedup reward buys speed, at a price we can measure.** A reward with a speedup term raises geometric-mean
  speedup on held-out tasks relative to the correctness-only arm; report the correctness trade-off explicitly.
- **K3 — evaluator exploitation is structured.** As reward rises, some fraction of programs exploit the evaluator
  (constant/precomputed outputs, tolerance abuse, input-buffer aliasing, timing games). These exploits cluster
  into a small taxonomy detectable from source + runtime behavior, and each detector has a countermeasure in the
  evaluator. (This is the differentiating contribution: the taxonomy, the detectors, and the hardened evaluator
  are a reusable resource.)
- **K4 — generalization across task difficulty.** Gains on held-out Level 1 tasks transfer partially to unseen
  Level 2 tasks; report per-level transfer, not one aggregate number.

**Not in scope now** (do not build): multi-turn revise-on-failure episodes (Kevin-style; a stretch arm, §6 scope
order), TPU/JAX, models other than Gemma 4, full KernelBench Level 3, distributed/multi-GPU kernels.

Related work to read before designing the reward (skim, one hour max): KernelBench (arXiv:2502.10517) for task
definitions and the fast_p metric; Kevin (arXiv:2507.11948) for multi-turn RL and reward design; CUDA-L1
(arXiv:2507.14111) for contrastive RL with speedup rewards and their speedup measurement protocol; Sakana's AI
CUDA Engineer + follow-up critiques for real evaluator-exploitation examples; METR's RE-Kernel writeup for timing
methodology on shared GPUs. The writeup's positioning: first RL post-training of the Gemma 4 family for kernel
engineering, consumer-hardware angle, and an evaluator-hardening contribution.

## 2. What exists — check before writing code

This is a fresh repo. The previous project at `/mnt/fastdata/glomergence` is the reuse source. Read its
`LEDGER.md` and `REPORT_STATUS.md` first — two days of GPU debugging are encoded there. List what you reuse and
what conflicts at the top of `REPORT_STATUS.md` before changing anything.

Reusable (copy, don't symlink; credit in the file header):

- `scripts/setup_node.sh`, `scripts/gpu_env.sh` — AutoDL bring-up: uv-managed Python 3.12, vLLM/TRL pins, CUDA ≥ 13
  driver requirement, npmmirror/Tsinghua mirrors, **ModelScope (not hf-mirror) for model weights**.
- `train/launch.py`, `train/configs/*.yaml` — TRL GRPO + LoRA adapter with kill-criteria enforcement, colocated
  vLLM with sleep mode. Ledger rules it embodies: `model_init_kwargs={"dtype": bfloat16}`; re-eval uses
  `eval_gpu_memory_utilization` (default 0.85), not the training share; micro-batch per GPU;
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `tools/gate_g4.py`'s `gemma4` preset — Gemma 4 sampling settings (temperature 1.0, top_p 0.95, top_k 64) and the
  `Gemma4ForConditionalGeneration` multimodal memory note (skip vision/audio encoders).
- `env/reward.py`'s extractor — strips Gemma 4's `<|channel>thought…<channel|>` reasoning channel before parsing code.
- `eval/report.py` — bootstrap-CI table generation; saturation kill criterion judged on outcome pass rate
  (`env/valid ≥ 0.95` for 5 steps), the version that works under continuous rewards.
- The `Schedule`/held-out-grid *pattern*: fixed held-out task set committed before the first training run.

Known landmine to fix on day one: the 2026-09-29 Gemma 4 download **silently produced no weights** (vLLM later
failed with "Cannot find any model weights"). Gate C0 must verify `*.safetensors` exist and load, not that the
download command exited 0.

What is genuinely new (do not underestimate):

- **The GPU is in the eval loop.** The previous env ran Maelstrom on CPU while the GPU trained. Here every rollout
  compiles and runs CUDA on the same GPU as training. Colocated vLLM sleep mode is validated; the eval side (CUDA
  context setup/teardown per rollout, memory leaks from generated code, timing while vLLM sleeps) is not.
- **Sandboxing GPU code.** Generated kernels need `/dev/nvidia*` and CUDA libraries, so the previous
  seccomp+Landlock design does not transfer directly. See G1.
- **Timing is a measurement problem.** Noisy neighbours, clock boosts, cache state. See G2.

Inventory the machine(s) per the previous project's REPORT_STATUS format: GPU/VRAM, CPU/RAM/disk, driver and CUDA
versions, PyTorch/triton/vLLM versions, whether Hugging Face / ModelScope are reachable.

## 3. Target architecture

```
repo/
  env/
    tasks.py           # KernelBench task loader: reference module, input generator, level/tags
    heldout.json       # fixed held-out task split, committed before the first training run   (K1–K4)
    run_kernel.py      # compile + run one generated program in the sandbox -> RunResult
    sandbox.py         # isolation for GPU code: no network, RO fs except workdir, CPU/mem/wall/GPU-mem caps
    timing.py          # CUDA-event timing protocol: warmup, L2 flush, median-of-N, CV logging (K2)
    correctness.py     # fresh-input checks, tolerance policy, output-persistence probes       (K3)
    reward.py          # RunResult -> scalar reward + logged components (correct, speedup, format, crash, hacks)
    hacks/detectors.py # evaluator-exploit detectors (source + behavior)                       (K3)
  tasks/
    prompt.md          # task spec shown to the policy: API contract, what "correct" and "faster" mean
  gold/
    cheats/*.py        # deliberate evaluator exploits used to validate detectors             (K3)
  train/
    configs/*.yaml     # one per run: model, task distribution, reward arm, GRPO hparams, kill criteria
    launch.py          # TRL GRPO adapter; enforces kill criteria
  eval/
    heldout.py         # checkpoint -> held-out tasks, n samples per task
    taxonomy.py        # classify failures and exploits                                        (K3)
    report.py          # runs/ -> REPORT.md tables with bootstrap CIs
  tools/gate_c0.py … gate_c6.py
  tests/               # unit + integration; sandbox and timing tests run real CUDA
  runs/<run_id>/       # config.yaml, git_sha, seed, versions.json, metrics.jsonl, checkpoints/, heldout/  (git-ignored)
  LEDGER.md  REPORT_STATUS.md  REPORT.md
```

**Design decisions already made.** Change any of these only with a ledger entry saying why.

- The policy writes **Python files that may embed CUDA C++/Triton**, compiled at eval time. One file per program,
  a fixed contract (`ModelNew(nn.Module)` per KernelBench convention).
- **Single-turn** episodes: prompt → program → compile/run → reward. Multi-turn is the stretch arm.
- **Reward arms.** Arm C (correctness only): 1.0 correct, 0.0 incorrect, −0.1 format/compile/crash/timeout.
  Arm S (correctness + speedup): 0 if incorrect; if correct, a bounded monotone function of speedup vs eager
  baseline (define it in `env/reward.py` with a comment justifying the shape; log raw speedup separately).
  Components always logged, even when unused by the arm.
- **Held-out split is fixed before the first training run** and committed as `env/heldout.json`: a subset of
  Level 1 held out from training, plus Level 2 tasks never trained on (K4). Nobody trains on it, tunes on it, or
  reads per-item results while choosing hyperparameters.
- **The baseline is eager PyTorch, measured in the same session, same GPU, same protocol as the candidate.**
  Never compare against published baselines measured elsewhere.
- **Training reward uses one eval per rollout; evaluation uses n ≥ 4 fresh-input correctness checks and the full
  timing protocol per sample.** GRPO tolerates reward noise; the writeup's numbers do not.
- **Parallel rollout evals are isolated**: separate processes, separate CUDA contexts, GPU memory verified
  reclaimed between rollouts.

## 4. Phases and gates

### Phase 0 — environment and measurement (dev box + one AutoDL boot)

- **C0 — stack and weights are real.** AutoDL node up via `scripts/setup_node.sh --gpu`; vLLM serves
  `google/gemma-4-12B-it` and answers a prompt; `*.safetensors` verified on disk for both 12B and E4B (the 09-29
  failure mode). KernelBench tasks load through `env/tasks.py`; its reference modules run on this GPU.
  PASS = all of the above, from scripts, not from memory.
- **C1 — the sandbox holds.** Hostile programs under `tests/hostile/`: opens a socket; writes outside its workdir;
  fork-bombs; allocates GPU memory until OOM and exits (next rollout must see full VRAM); launches an infinite
  kernel (wall-clock cap must fire and the GPU must be usable afterwards); tries to read the task's reference
  implementation or the timing harness at runtime (oracle-read probe — harness files are 0600, the sandbox blocks
  it). PASS = 6/6 contained with the correct failure component in `reward.py`.
- **C2 — timing is a measurement, not a vibe.** `env/timing.py` runs the reference kernel 30× interleaved with a
  known-fast variant (e.g., a torch.compile'd or hand-fused version where legal) and reports median, IQR, and
  CV for both. PASS = the fast variant measures faster in 30/30 pairs, and reference CV ≤ 5% (if the box can't
  reach 5%, document the floor and widen n — the number goes in the writeup's limitations, not under the rug).
- **C3 — the evaluator can see a cheat.** Write ≥ 4 deliberate exploits under `gold/cheats/`: constant/precomputed
  output; correct only at the default input distribution; returns a view of the input buffer (output-persistence
  violation); timing exploit (async work invisible to the timer). Each must score high under a naive evaluator
  and be caught by `env/correctness.py` + `env/hacks/detectors.py`. PASS = every cheat caught, and the reference
  passes every detector. If a cheat is uncatchable, the reward arm it breaks is redesigned now, ledgered —
  never after RL finds it.

### Phase 1 — signal and pilot (AutoDL)

- **C4 — the base model has signal.** Sample gemma-4-12B-it (model-card settings: temp 1.0, top_p 0.95, top_k 64)
  on the training-distribution tasks, n = 32 per task; also E4B for the secondary arm. Record pass@1, pass@8,
  correctness, speedup distribution of correct samples, median output length, P_signal per GRPO group (previous
  project's metric: probability a group of 8 contains both a pass and a fail). PASS = pass@8 > 0 on enough tasks
  that the training distribution has P_signal ≥ ~0.2. If 12B has none, try E4B and a narrowed task subset; if
  still none, stop and report — RL on zero signal is not a run.
- **C5 — one GRPO step end to end.** Previous project's adapter on the new env: group of G programs per prompt,
  each compiled/run on this GPU while vLLM sleeps, group-relative advantages, one optimizer step, checkpoint,
  reload, re-eval. PASS = a resumable run of ≥ 3 steps with metrics on disk and GPU memory verified stable.
- **C6 — pilots and the paper's curves.** Two runs, arm C and arm S, on the C4-validated task distribution,
  ≤ 8 GPU-hours each. Eval cadence every 10 steps on the held-out split plus the always-score-the-final-policy
  rule (previous ledger). Kill criteria in every config: outcome saturation (pass rate ≥ 0.95 for 5 steps);
  frac_reward_zero_std ≥ 0.95 with mean reward ≥ 0.95 for 5 steps; KL above threshold; median length collapse or
  truncation spike; any sandbox violation; detector fire rate above threshold (that one is a *finding* trigger —
  freeze the checkpoint, don't silently continue). PASS = runs end by completion or a declared criterion, and
  `eval/report.py` produces base-vs-trained tables (correctness, geo-mean speedup, per-level K4 transfer,
  detector fire rates) with bootstrap CIs.

## 5. Findings versus bugs

- Training reward rises while held-out correctness stays flat or falls → **finding**. Snapshot, dump 20 failed
  programs, run `taxonomy.py`, write it up.
- A detector fires on a trained policy's program → **finding (K3)**. Add the exploit to the taxonomy; if it's
  novel, add a detector and a countermeasure, then decide with the operator whether to restart the arm with the
  hardened evaluator (both results go in the writeup).
- Speedup rises while correctness stays high but the "kernels" wrap torch.compile or call vendor libraries the
  task forbids → **task-spec bug**. Tighten `tasks/prompt.md` and the detector; ledger it.
- Timing CV degrades mid-run (noisy neighbour on the rental box) → **measurement problem**. Re-measure affected
  checkpoints; never compare numbers across protocols.
- Correctness gains come entirely from the format/crash term shrinking → **reward-weighting bug**, not learning.

## 6. Hard rules, budget, and scope cuts

Hard rules:

1. LLM-generated code runs only inside the C1 sandbox. No exceptions for quick tests.
2. Never train on, tune on, or inspect per-item results from the held-out split while choosing hyperparameters.
3. Every run directory contains `config.yaml`, `git_sha`, `seed`, `versions.json`, `metrics.jsonl`.
   A run without these is not a run.
4. Commit at every gate pass, with the gate's JSON record in the commit. Never commit `runs/`, checkpoints, or
  model weights.
5. Do not add tasks, reward terms, turns, or models beyond this document. Propose them under Roadmap in
   `REPORT_STATUS.md`.

Budget: AutoDL spending needs operator approval before provisioning (target ≈ 25–30 GPU-hours total on one
80 GB card, ≈ ¥250–375 at previous rates). Any run expected to exceed 1 GPU-hour needs operator go.

**Scope-cut order (deadline is 2026-11-12; cut from the top when squeezed):** E4B secondary arm → K4 Level-2
transfer evaluation (report Level 1 only) → arm S speedup reward (report arm C + K3 hardening only) → multi-turn
stretch arm (already optional). Never cut: C1–C3 evaluator gates, the K3 taxonomy, bootstrap CIs.

Suggested pace: week 1 C0–C3; week 2 C4–C5; weeks 3–4 C6 pilots + hardening iteration; week 5 analysis and
3,000-word writeup (leave 5+ full days for writing; the writeup is the deliverable, not an afterthought).

Stop and ask the operator before: spending money; a training run over ~1 GPU-hour; continuing after a gate has
failed twice; changing the held-out split once C6 has started; any sign of sandbox escape; reintroducing
multi-turn episodes or non-Gemma models (needs a new kickoff).

## 7. This session, in order

1. Inventory (§2) + reuse/conflict list → `REPORT_STATUS.md`. Read the glomergence `LEDGER.md` first.
2. Scaffold the tree in §3 with stubs and a passing `pytest` that, for now, only checks imports.
3. Copy and adapt the reusable stack (§2); bring up the AutoDL node; pass **C0**.
4. Build `env/sandbox.py`, `run_kernel.py`; pass **C1**.
5. Build `env/timing.py`, `correctness.py`, `hacks/detectors.py`; write `gold/cheats/`; pass **C2** and **C3**.
6. Stop. Write the session summary (§8) and wait. Do not start C4–C6 unless the operator says go.

## 8. Session summary format

Write it to `REPORT_STATUS.md`, then repeat it in chat.

- Gates: table of C0–C6 with PASS / FAIL / not run and the JSON record path.
- Numbers: timing noise floor; detector coverage over `gold/cheats/`; base-model pass@1/pass@8 and P_signal
  (once C4 runs); rollouts/hour.
- Ledger entries added this session, one line each.
- Decisions you made that this document did not make for you, with reasons.
- Open questions for the operator, ranked by how much they block the next gate.
- The exact command to resume.
