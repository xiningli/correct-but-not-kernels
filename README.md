# Correct, But Not Kernels

Code, receipts and paper for *Correct, But Not Kernels: RL on a permissive verifier
raised Gemma 4's correctness, not its kernel writing*, a Gemma 4 Developer Agent
Paper Track submission.

- Paper: [paper/writeup.pdf](paper/writeup.pdf), built from [paper/writeup.md](paper/writeup.md)
  by `python -m research.paper_v1.build_latex`.
- Data bundle (CC BY 4.0): https://huggingface.co/datasets/xiningli/correct-but-not-kernels-data.
  Download it as `release-data/` in this checkout, then check every number in the paper with
  `python -m research.paper_v1.claims --data release-data/data` (expects 0 missing) and rebuild
  Sections 4.1–4.3 with `python -m research.paper_v1.reproduce --data release-data/data --out <new dir>`.
- Licence: Apache-2.0 ([LICENSE](LICENSE)); third-party material: [NOTICE](NOTICE).

Tests: run `bash scripts/setup_node.sh` (fetches the pinned KernelBench), then
`python -m pytest tests research` with PyTorch installed. Tests whose inputs are not
committed (the data bundle, upstream Liger-Kernel snapshots, one local evidence archive)
skip with a reason; a clean checkout gives 225 passed, 6 skipped.

## Project history: Gemma 4 CUDA kernel engineer

Research kickoff: [kickoff.md](kickoff.md). Current evidence and session decisions:
[REPORT_STATUS.md](REPORT_STATUS.md). Failures: [LEDGER.md](LEDGER.md).

**Current evidence, audited 2026-10-18:** fine-tuning improves numerical correctness
under the original protocol, which permits high-level PyTorch computation. It has
not established improved custom CUDA/Triton kernel generation, an RL advantage
over rejection-sampled SFT, or broad runtime acceleration.

| Completed study | Result | Scope |
|---|---|---|
| Three-seed confirmation, 9,600 programs | Level-2 base 54.875%; C 72.000%; S 71.667%; SFT 71.250% | Known tasks, new draws; training used one LayerNorm task |
| Strict replay of 5,600 saved programs | 0 satisfy both numerical and custom-kernel requirements | New diagnostic endpoint on old outputs, not new strict-prompt generations |
| Host-interface development, 84 programs | Full program 1/42; fixed host 16/42 | Six development graphs; 15/42 if every length-capped output counts as failure |
| Iterative repair, two development defects | 12B: 1/4, 1/4, 0/4; 31B: 2/4, 2/4, 3/4 | Resampling / verdict / detailed diagnostics; no independent task confirmation |

The 31B diagnostic-only extra repair passes the retained GPU suite, but its tanh
formula has a separately demonstrated CPU float32 overflow risk. The 7 successful
31B episodes contain only 3 distinct successful programs because first draws are
shared. These are development leads, not reliable cross-repository repair claims.

Read the [full audit and its limitations](REPORT_STATUS.md) before quoting results.
[REPORT.md](REPORT.md) is the original C6 pilot report, not the final project report.
Audit receipts are in [evidence/full-audit-20261018/](evidence/full-audit-20261018/);
the read-only reconstruction entry is
[research/full_audit_20261018.py](research/full_audit_20261018.py).

The original training/evaluation lives in `env/`, `train/`, `eval/` and `tools/`.
Strict replay and host-interface studies live in `research/reliability_v2/`;
diagnostic repair in `research/diagnostics_v1/`, `research/diagnostics_v2/` and
`research/diagnostics_resume_v6/`. `research/scoped_repair/` is an unintegrated
function-editing prototype with CPU checks, not a completed model experiment.
Forty-eight proposed new graphs remain unsealed and unevaluated by models.
Only two reduced Liger defects have been model-tested; the ExecuTorch candidate
does not yet provide cross-repository evidence.

Completed protocols and source seals remain immutable. New studies need separate
versions; do not silently alter historical scores, prompts, splits or evaluators.

**Completed add-on, verified 2026-10-21:** [consumer GPU replication](evidence/portability-v1-complete-20261021/SUMMARY.md)
replayed 4,368 saved programs on 78 resource-qualified tasks using the local RTX 5070 Ti.
Numerical pass rates: Base 54.01%, C/RL 71.69%, SFT 71.47%.
Gains persisted on the common subset, with no clear C/RL advantage over SFT.
Strict custom-kernel successes remain 0/4,368. The 22 exclusions follow temporary/output-storage
limits; they do not establish GPU infeasibility. This is same-architecture
platform replication of existing programs. Independent-defect repair remains
the main study. [Current evidence and limits](REPORT_STATUS.md).

## Local preparation

```bash
bash scripts/setup_node.sh
.venv/bin/python -m tools.preflight
```

Optional trusted-reference CUDA smoke using the existing GPU stack, without modifying it:

```bash
.venv/bin/python -m tools.preflight --gpu-python /mnt/fastdata/glomergence/.venv/bin/python
```

The tool prints a new `runs/preflight/<timestamp>/preflight.json` path. Generate the
status numbers from that directory:

```bash
.venv/bin/python -m eval.preflight_report runs/preflight/<timestamp> --write-status
```

The independent `.venv` contains only development dependencies. GPU smoke reuses
the previous project's installed interpreter read-only. A standalone machine
needs `setup_node.sh --gpu`. No model weights are downloaded during local setup.

For all CPU checks, use an interpreter containing PyTorch and pytest, from the
repository checkout:

```bash
python -m pytest -q tests research/reliability_v2 research/diagnostics_v1 research/diagnostics_v2 research/diagnostics_resume_v6 research/scoped_repair -m 'not cuda'
```

The audit passed 179 checks with the workstation's PyTorch interpreter. The
development-only `.venv` cannot run the complete suite without PyTorch, and default
pytest discovery omits the research directories. Package discovery in the frozen
`pyproject.toml` also omits `tasks` and `research`; do not assume an installed wheel
provides these entry points. Public artifact packaging remains unfinished: Git
alone excludes the raw runs, upstream task archives and model weights required
for complete reproduction.

## AutoDL deployment

The commands below describe the original setup workflow. Historical C0 on
2026-10-04 passed on an RTX PRO 6000 Blackwell Server Edition (97,887 MiB), with
CUDA toolkit 13.0.88 and g++ 11.4.0: both weight/config/tokenizer checks, a real
12B `READY` response and two trusted CUDA references. See
[the C0 evidence](evidence/gates/c0-20261004T224213.417987Z-7b8a8dac.json).
Later nodes, storage capacity and study preparations are recorded separately in
REPORT_STATUS. The last observed data disk was 470 GiB; this audit could not reach
SSH and does not certify current power, disk, GPU or billing state.

For development dependencies and CPU checks only:

```bash
cd /root/autodl-tmp/cuda-gemma4-llm
bash scripts/run_autodl.sh prepare
```

After GPU attachment, one command checks resources, completes setup, verifies the
existing snapshots and runs C0 (cached files are reused):

```bash
cd /root/autodl-tmp/cuda-gemma4-llm
bash scripts/run_autodl.sh c0
```

To pre-download while the expanded node is still in no-GPU mode:

```bash
bash scripts/run_autodl.sh prefetch
```

`prefetch` requires disk space and CPU memory, installs GPU wheels with low
concurrency, and downloads both ModelScope snapshots. It checks safetensors
payload bounds and index consistency without loading tensors or models. This
structural check is not C0; C0 repeats validation, loads tensors/metadata and
performs actual vLLM generation after the GPU boot. This original download path
does not prepare the later 31B continuation; use its versioned preparation records.

`c0` first checks GPU capacity, driver/toolkit, compiler, data-disk headroom and
container memory quota. A readiness failure writes a preflight record and exits
before installs/downloads; it is not a formal gate attempt. It also checks the
twice-failed gate stop rule before spending time on downloads. It stops after C0.
No C1–C6 or training is automatically started. Use `bash scripts/run_autodl.sh status`
for inventory. All stages log to `runs/setup/` and reject concurrent setup commands.

Managed Python, uv, package caches, model caches and compilation caches live in
the project's ignored `.cache/` on the data disk; final snapshots live in `models/`.
The matching FlashInfer cubin wheel is kept in `.cache/wheels/`; GPU setup checks
its pinned SHA-256 and installs from that local directory. On another machine,
copy that wheel as well if GitHub is unreachable; without a local copy, setup
uses the [official FlashInfer wheel index](https://flashinfer.ai/whl/flashinfer-cubin/).
GPU-specific JIT compilation and actual model loading still require the GPU boot.
The image Conda environment and shell profiles are preserved. The CPU `prepare`
stage never imports torch or downloads model weights.

For a new machine, prepare the transfer artifacts from a clean committed checkout:

```bash
.venv/bin/python -m tools.package_autodl
```

Transfer the printed directory's `source.bundle`, `kernelbench.tar.gz`,
`manifest.json`, and `SHA256SUMS` together. In that directory on the node:

```bash
sha256sum -c SHA256SUMS
git clone source.bundle /root/autodl-tmp/cuda-gemma4-llm
git -C /root/autodl-tmp/cuda-gemma4-llm switch -c autodl
tar xzf kernelbench.tar.gz --no-same-owner -C /root/autodl-tmp/cuda-gemma4-llm
```

Do not extract over an existing changed checkout. This bundle includes the exact
public KernelBench checkout and its Git metadata, so setup needs no GitHub access.
It excludes environments, runs, credentials and weights. The source bundle retains
the frozen split's Git history. A manifest records both revisions and artifact hashes.

`setup_node.sh` uses domestic Python/PyPI mirrors on AutoDL, verifies the pinned
KernelBench revision, and records inventory. Downloads come from ModelScope,
never hf-mirror. The downloader validates actual tensor payloads and offline
config/tokenizer loading. C0 independently repeats validation of both snapshots,
requires a real 12B vLLM response, then runs official CUDA reference modules in
separate processes. C1–C3 were subsequently implemented and passed their recorded
isolation, timing and deliberate-cheat controls. These bounded controls are not
a general proof of containment or evaluator correctness.

Historical gate receipts are under `evidence/gates/`, with raw evidence in ignored
`runs/`. C0–C6 and R2/R3 are complete. For a new execution, follow the applicable
versioned study protocol and current operator instructions, retaining fresh
controls and failures; an old PASS is not a current-machine readiness check.

## Frozen benchmark

`env/kernelbench.lock.json` records the upstream revision and per-task SHA-256.
`env/heldout.json` records the deterministic split before any model evaluation.
Before initial evaluation, only identities/source hashes were inspected for held-out
tasks. The researcher has since observed their outcomes; they are no longer unseen
research tasks for a new confirmation claim.
All Level 2 tasks are excluded from training. Re-running the split generator refuses
to overwrite either file. Normal setup restores the existing split, never regenerates it.

Generated programs use the study's validated isolated evaluator. The
reference loader accepts only content-verified, pinned upstream reference modules;
it is not an evaluator or an isolation mechanism for candidate code.

## Provenance and source checks

Setup and GPU environment scripts, the Gemma sampling settings and code extractor
are adapted from `/mnt/fastdata/glomergence` at
`e602f63207fe4e67eeda4dff7b53d49e92cc318b`, credited in file headers. The old project's
Maelstrom runtime, reward, results and CPU sandbox were not copied into this evaluator.
Training, reporting and later repair runners are implemented; consult each
completed study's source seal for its exact code version.

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench): upstream code and MIT license live in the pinned checkout.
- [12B model card](https://huggingface.co/google/gemma-4-12B-it) and [E4B model card](https://huggingface.co/google/gemma-4-E4B-it): sampling configuration checked 2026-10-04.
- [Official paper track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper): consult the current track rules for submission requirements. The project audit establishes neither eligibility nor award prospects.

Reward definitions and historical decisions are retained in `env/reward.py`,
the frozen experiment configurations and LEDGER. The latest audit distinguishes
confirmed numerical results from strict-kernel and repair hypotheses still needing
independent experiments.
