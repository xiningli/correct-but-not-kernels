# Failure ledger

## 2026-10-21 — Completed consumer replay: storage coverage and discordant outcomes

- All 4,368 policy attempts on 78 qualifying tasks completed. The 22 exclusions
  are reference temporary/output-storage limits, not established GPU OOM.
  Rule: report coverage and apply the same reference-only exclusions to every
  policy; do not attribute this incomplete coverage to 16GB GPU capability.
- Eight numerical three-state outcomes differ: one remote pass becomes a local
  numerical failure, three remote sandbox-blocked attempts become local passes,
  and four remote numerical failures become local illegal-memory-access crashes.
  Rule: preserve unknown outcomes and platform differences; logs alone do not
  establish a causal transfer mechanism. No diagnostic reruns were performed.
- Strict custom-kernel success remains zero. Numerical gains retain the original
  prompt's high-level PyTorch scope; this replication does not repair that limit
  or establish RL superiority to SFT.
- Recomputed metrics and task-bootstrap effects match the terminal report;
  frozen inputs and the terminal evidence archive hash verify. Details and
  executable verification are linked in REPORT_STATUS.md.

## 2026-10-20 — Consumer GPU replay: compiler controls before model outputs

- Revision r2 stopped at its trusted Triton positive control with zero candidate
  attempts. Docker's temporary mount was not executable, so loading Triton's
  compiled helper failed with `failed to map segment from shared object`.
  Revision r3 explicitly permits execution in the bounded temporary mount;
  the network, read-only inputs/root filesystem and resource limits remain.
  Preserve r2 outcome, logs and archive; an environment failure is not a model
  correctness failure.
- Independent r3 preflight passed Triton and CUDA C++ positive controls, the
  PyTorch numerical-versus-contract distinction, wrong values, input mutation
  and fresh-input failure controls. The full controller repeats its controls
  under a new preparation before any saved model program is evaluated.
- Initial r1 preparation was not launched. Pre-launch review added the mounted
  environment's Ninja executable to PATH and corrected the mutation control's
  expected numerical result: two forwards perform two mutations.
- Preparation verifies every member of the original v1.1 source seal, the
  historical plan and generated-job hashes. Later-study scripts added outside
  those members are not modifications of frozen training code. The original
  seal and its stricter directory-enumeration guard are unchanged.

## 2026-10-17 — Completed 31B developer experiment: shared successes and API failures

- All 12 episodes completed. Detailed diagnostics resolved 3/4, independent
  resampling 2/4 and verdict-only revision 2/4. Two successful first candidates
  were reused across strategies; the seven logical successes are three distinct
  generated programs. Rule: do not count shared first draws as independent
  evidence for a feedback intervention.
- One diagnostic branch repaired the previously unsuccessful cross-entropy
  developer case on attempt four after repeated missing `tl.tanh` errors. The
  corresponding independent/verdict branches failed. Seventeen of 22 physical
  candidates received the same missing-API diagnosis. Rule: preserve both the
  positive trace and repeated failures; one seed does not establish an average
  causal diagnostic effect or reliable transfer.
- The successful custom tanh code uses exponentials that can overflow for
  sufficiently large absolute float32 inputs. This source inspection motivates
  a numerical-domain review before broader claims; no new test was executed and
  the frozen visible/withheld scores remain unchanged.
- Both archived reports reproduce byte-identically and all source seals match.
  The backup was verified before acknowledgment, followed by loss of SSH.
  No shutdown-return receipt was captured. Rule: experiment completion and
  backup verification do not independently verify platform power/billing.

## 2026-10-17 — 31B continuation launch recovered without rerolling outcomes

- The first inline launcher invocation failed before an active controller or
  model output existed. A compiled launcher file replaced the inline command;
  its exclusive launch receipt and active-run guard allowed one controller.
  Experimental code and the prepared protocol were unchanged.
- At 90% memory utilization, the 31B engine initialized with 24.83 GiB KV cache,
  versus the prior 20.07 GiB allocation. The first four request payloads match
  the stopped parent exactly except their preparation path. Rule: distinguish
  infrastructure recovery from improved model repair results; no completed
  generation was yet observed in the startup snapshot.

## 2026-10-16 — 31B startup cache budget, after completed 12B arm

- The 31B BF16 weights loaded, but vLLM rejected its 24K context: 20.64 GiB
  KV cache required versus 20.07 GiB available at 85% memory utilization.
  No 31B completion or score exists. Rule: classify engine initialization
  separately from model repair failures; retain the startup log and cost.
- The stopped archive was verified locally before acknowledgment. SSH was
  subsequently lost without a captured shutdown-return receipt. Rule: report
  backup success while leaving physical power/billing unverified.
- All 12 completed 12B episodes are retained: independent 1/4, verdict 1/4,
  detailed diagnostics 0/4; both successes on the same stride defect. Rule:
  replicas/branches are not independent defects, and this small development
  result does not establish the effect of diagnostics or model scaling.
- Continuation is isolated in a new namespace and bound to the exact parent
  archive. Only the ungenerated 31B arm will run, at 90% memory utilization;
  original generation/evaluation rules and 12B outcomes stay unchanged.
  Rule: never reroll completed outcomes to recover an unrelated startup error.
- Initial local regression invocation used the lightweight project environment,
  which lacks torch; the existing runner test could not import it. Repeating the
  relevant suite in the workstation's installed torch environment passed all
  19 tests. No remote execution was attempted by these CPU tests.

## 2026-10-16 — Larger-model and iterative repair controls

- Changing both model and retry count in one condition would confound them.
  v6 runs both model sizes under all three strategies, sharing the exact first
  candidate. The output/attempt caps match; real tokens, latency and shared costs
  are reported rather than calling them equal compute.
- A revised program's visible compiler errors may guide the next attempt, but
  raw logs can expose harness details or arbitrary model-generated text. Only
  closed API-error patterns, validated numerical facts and contract operation
  names enter feedback. Withheld outcomes are evaluated after generation and
  cannot reopen an episode or select a different patch.
- Initial code compilation caught an unclosed report-list bracket; this was
  corrected before deployment or any GPU generation. State-machine integration
  tests check exact first-draw sharing, four-attempt bounds, malformed attempts,
  visible stopping, hidden-score isolation and physical/logical cost accounting.
- The first CPU-node inventory assumed `nvidia-smi` was executable; no-GPU mode
  returns a permission error. GPU availability probes now treat that as normal
  preparation state, not a model failure. No GPU experiment has run in v6.
- The first monitor probe succeeded, but automatic staging exposed a working-
  directory bug: the SSH helper imported the project namespace from `/root`.
  It now explicitly changes to the project directory before executing Python.
  The failed launch produced no staging preparation or GPU output; the initial
  CPU validation receipt is retained and current-source validation repeated.
- Existing disk headroom could not hold BF16 31B weights plus evaluator scratch.
  Only reconstructible uv cache was cleaned after confirming that installed
  packages do not symlink into it; no checkpoint or research artifact was removed.
  Slow sequential weight download was stopped at its contiguous partial offset
  and resumed using four bounded ranges. Range/length/SHA checks reject incorrect
  resume responses, with CPU tests covering assembly and unsupported Range replies.
- Two real network read timeouts showed that a five-retry budget shared by an
  entire 50 GB shard was too brittle. Each bounded segment now has at most five
  network retries before the existing shard-level stop logic applies. A partial-
  body timeout test checks exact replacement bytes and the retry bound. CPU
  staging/download workers were stopped, their receipts preserved, and current-
  source validation repeated before resuming; no GPU result existed at this fix.

## 2026-10-16 — Completed strict replay limits the original capability claim

- All 5,600 saved programs were replayed; no program satisfied both numerical
  correctness and the strict custom-compute contract. There were 4,236 contract
  violations. PyTorch computation was legal originally. Rule: retain the original
  numerical result under its actual specification, but do not describe it as
  demonstrated improvement in writing correct custom CUDA/Triton kernels.
- Diagnostic repair completed 84 attempts with two successes: both base samples
  on the same stride defect with grouped facts. C and SFT each resolved 0/36.
  Rule: keep independent-defect count at two, disclose all failures, and do not
  infer general diagnostic gains or statistically established fine-tuning harm.
- The repair failures included 49 numerical, 19 crash, 11 format and 3 contract
  outcomes. Only two completions hit the output cap. Rule: inspect failure
  mechanisms before adding more generation budget or training; do not treat a
  failed repair as proof that its available diagnostic was sufficient.
- Both local archives pass hash/size checks; exact grids and byte-identical
  reports reproduce. Backup acknowledgment preceded loss of SSH connectivity.
  No shutdown-return receipt was captured. Rule: mark the experiments complete
  and backups verified while leaving platform power/billing unverified.

## 2026-10-16 — Diagnostic evidence interventions, with information held explicit

- The operator selected diagnostic availability/use as the main research question.
  New work lives in `research/diagnostics_v1`; the running v4 and frozen v1.1
  sources remain unchanged. Existing SFT/RL models serve as controls, not a
  requirement that RL must win. Rule: implement evidence interventions without
  changing or relabeling past evaluations.
- Initial CPU check failed with `AssertionError: assert '/pull/' not in
  task['source']`: besides the task's PR locator, the preserved kernel contained
  historical PR-198 links in comments. The public projection now removes PR
  locators only in comments and asserts identical executable AST. License text
  remains intact; private upstream archives are unchanged. The repaired checks
  passed. This reduces explicit answer locators, not pretraining contamination.
- A flat diagnostic and a grouped diagnostic must contain exactly the same
  measured facts before attributing a difference to presentation. Arbitrary raw
  stderr is not information-equivalent to a curated summary. Rule: use a closed
  visible-fact schema, preserve its hash across both rich conditions, and keep
  withheld scores and gold fixes outside prompt construction.
- A failed repair with detailed feedback does not prove intrinsic inability:
  the feedback may still omit the key fact. Rule: review evidence relevance and
  add independent defects/strong-model controls before claiming that mechanism.
- The initial frozen-source verification command used an unsupported option:
  `replication.py: error: unrecognized arguments: --directory runs/replication-v1.1`.
  The documented `--out` option passed; no frozen source changed.
- New GPU work must wait for the live predecessor to exit after its report and
  verified archive. A task-specific handoff defers only its shutdown. CPU tests
  reject missing/mismatched backup receipts, corrupted archives, and acquisition
  while the predecessor still runs. The successor owns final report/backup and
  shutdown; a failed queue must not power off a still-running predecessor.
- Initial remote preparation validation stopped with `Prepared source seal
  changed`: `task_inference.py` and its test are local-only post-hoc statistics
  files, absent remotely and unused by the diagnostic pipeline. Uploading them
  would alter the running v4 glob-based inventory. The v5 seal now explicitly
  excludes these two unrelated files. All runtime dependencies remain sealed;
  preparation was repeated into a new directory, and all 53 CPU tests and
  frozen-source verification passed again. No GPU experiment started with the
  mismatched manifest.
- Before model generation, the actual Gemma tokenizer measured the original
  programs at 5,284 and 1,175 tokens including code fences. The initial 4,096
  output cap would require compressing the longer program even for a tiny fix.
  All conditions now receive 8,192 output tokens and 24,576 context; a new
  preflight requires the unchanged original to fit. Preparations `a` and `b`
  remain preserved, with `c` used for execution. No model outcomes were observed
  before this protocol correction.
- Final CPU validation passed 54 local and 14 remote checks. The successor was
  queued at 05:13:25 UTC, after checking the exact predecessor PID, source seal,
  and absence of an existing shutdown handoff. Both local monitors are active;
  the successor is waiting, with no GPU controls or model repairs yet. Remote
  receipts and JUnit were copied locally and their identities verified.

## 2026-10-14 — A p-value requires an explicit sampling model

- Supplementary task-cluster bootstrap testing gives C-minus-SFT raw p=0.518535
  (Bonferroni p=1), conditional on the saved policies. This neither demonstrates
  equivalence nor estimates variability across future training runs. Tasks were
  curated, not randomly drawn from all CUDA problems. Rule: declare the inferential
  target and task-independence/representativeness assumptions; do not call
  nonparametric inference assumption-free or count programs as independent tasks.
- The original pre-registered analysis reported percentile intervals. The new
  null-centered studentized p-values were requested after observing results and
  remain a labeled post-hoc supplement. Report all three planned contrasts and
  preserve the original intervals. Zero Monte Carlo exceedances are a numerical
  resolution limit, not p=0. All-equal task outcomes cannot estimate variance and
  are returned as untestable instead of evidence of equality.

## 2026-10-13 — Strict transfer development floor

- All 36 development attempts completed, but base, C-20261021 and SFT-20261021
  each scored 0/12 under the strict custom-kernel contract. The archived report
  and complete grid reproduce locally. The 24 crashes include invalid APIs,
  Triton integer-pointer launches, extension linkage errors and stride failures;
  other failures include forbidden PyTorch computation, output contract and
  format errors. This is a development finding, not a reason to relax scoring
  or to claim the three-seed population effect is zero. Prior permissive-contract
  correctness gains cannot substitute for strict CUDA evidence. Rule: retain this
  floor result and establish usable development signal with symmetric interfaces
  and visible-feedback trials before a large confirmatory evaluation.
- The backup watcher acknowledged a verified terminal archive and exited before
  collecting the controller's subsequent shutdown-return record. SSH later timed
  out. Rule: distinguish verified run completion/backup from unverified platform
  power state; a future watcher should attempt to collect the shutdown receipt.

## 2026-10-13 — Strict observer and transfer development


- The first strict Triton positive exited with `sandbox_blocked` / SIGSYS before
  candidate scoring. A trusted-only strace probe identified a socket created by
  lazy profiler/Dynamo initialization. The new strict launcher initializes that
  trusted runtime before TSYNC; candidate code still begins after Landlock, UID
  drop and the byte-identical original seccomp filter. Seven runtime-contract
  controls and all six inherited hostile/GPU-recovery probes subsequently passed.
  Rule: a missing observer is infrastructure failure, never model failure or
  evidence of compliance. Keep initial failure records.
- Ordinary `.cuda()` may compact a strided CPU view. The new worker moves each
  backing storage once and reconstructs original strides/offsets, retaining shared
  storage identity. Real upstream independent-stride failures reproduce and the
  original upstream fix passes under this path. Rule: input transport must not
  remove the property an evaluation claims to test.
- The operator authorized multi-turn CUDA repair evaluation beyond the original
  kickoff scope. Its budgets, visible-only patch selection, real-vs-reduced-vs-
  injected provenance, and behavioral mechanism limits are a new protocol draft;
  original experiment code and scores stay sealed. No positive outcome is assumed.
- Development sampler initially failed before any generation with
  `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`. Its new
  controller had omitted `scripts/gpu_env.sh`, so the installed pinned compiler
  was absent from PATH. Reuse the existing environment launcher and fail fast
  on missing `ninja`/`nvcc` before loading weights. The zero-score failed run is
  backed up; its delayed shutdown is explicitly deferred before the new run.


- **2026-10-13 · All-policy replay resolved the 21 ENOSPC misclassifications.**
  All 80 saved L1Norm programs were replayed after 12 matching controls passed.
  The 21 cases flagged by the prior log audit all changed from crash to correct
  (C seven, S six, SFT eight); the other 59 result components were unchanged.
  Local verification matched identities/seeds, code/score/report hashes and
  the exact set of recovered cases to the audit. Original v1.1 scores remain
  immutable; primary Level-2 comparisons are unaffected. **Rule:** report the
  repaired measurement separately, including all-policy coverage and the
  changed environment; do not count recovery as additional model learning.
  Terminal evidence was verified locally before the acknowledged shutdown
  command returned 0. A later fresh container was idle, not a resumed run.

- **2026-10-12 · Calibration scratch cleanup resolved the full-size control block.**
  On restart, the helper hashed and removed 14,763,975,122 bytes from the
  explicitly passed preceding C3 run; free bytes rose from 49,524,518,912 to
  64,288,628,736. The new C3 scratch was treated the same way. With unchanged
  evaluator code, limits and replay manifest, all 12 controls then passed,
  including both full-size four-input reference/custom cases. The 80-program
  replay began at 22:37:22 UTC. This is validation of the operations repair,
  not evidence yet of improved model outcomes. **Rule:** clean reproducible
  calibration tensors with receipts between stages and retain the failed
  attempt. The live watcher also used `controls/checks.json` instead of the
  actual `controls/cases/checks.json`; fixing that read-only path restored
  intermediate control visibility without changing any experiment result.

- **2026-10-12 · Streaming full-size control correctly stopped on low free disk.**
  Fresh C0–C3 and ten small v2 controls passed. The full-size reference control
  reported `Insufficient evaluator storage: free=15165579264,
  required=17181107920`; the 80-program replay never began. The failed run was
  reported, archived and verified locally before its shutdown grace elapsed.
  C3 calibration retains large reproducible input copies and outputs, which
  were not removed before budgeting the next stage. Exact reclaimed bytes
  require the remote cleanup receipt and are not inferred from this error.
  **Rule:** account for preceding gate scratch as well as current evaluator
  scratch. Hash and clean only named tensors in explicitly passed C3 runs,
  retain all original evidence, and check free space again before full-size
  controls. The new guard must not be weakened to force a PASS. This repair
  changes operations only; evaluator code, frozen manifest, programs and seeds
  remain unchanged. CPU cleanup self-test passed; GPU retry is pending.

- **2026-10-12 · New GPU boot requires preventing a stale shutdown action.**
  The old archive recovery service was still downloading with unconditional
  shutdown on completion. The operator then authorized and attached the GPU
  for a follow-up. Stopped the service, retained its partial transfer, added an
  explicit active-follow-up check before acknowledgment/shutdown, and resumed.
  **Rule:** a prior study's backup completion must not terminate a newer
  authorized workload. Transfer/hash verification still runs; shutdown moves
  to the current experiment supervisor. The GPU boot also changed driver
  580.95.05 to 595.58.03, so revalidate C0–C3 and disclose environment changes.

- **2026-10-12 · Old tensor staging can exceed remaining data-disk capacity.**
  The replay node has about 60 GiB free. L1Norm has approximately 8 GiB per
  input/output, with four trials; the old path retains private and sandbox
  copies plus all outputs. The new version consumes one staged case at a
  time and retains outputs for trusted-parent comparison. The deterministic
  storage-budget unit test requires 48 GiB including one-output scratch and
  8 GiB headroom for four equal 8 GiB cases, instead of input duplication.
  **Rule:** validate actual free space and owned same-filesystem staging,
  maintain correctness checks in the trusted parent, and run full-size valid
  reference/custom controls before attributing replay failures to the model.
  GPU validation is pending at this entry; CPU tests are not a GPU PASS.

- **2026-10-12 · Retrospective audit found output-storage errors counted as crashes.**
  `research/reliability_v2/audit.py` parses all 9,600 final programs and archived
  logs without executing candidates. Its r2 report identifies 21
  `safetensors.save_file` failures with `No space left on device (os error 28)`,
  all on Level-1 L1Norm in the third training seed (SFT eight, C seven, S six).
  Numerical comparisons were never reached. Historical free-space/quota cause
  and these candidates' correctness remain unresolved. The supervisor's work
  allowance is not a guarantee of actual free space, and the private input
  copy also consumes storage outside the sandbox work tree. **Rule:** preserve
  original scores and disclose the measurement limitation; make storage
  preflight and harness-I/O attribution part of the next evaluator. Replay
  all saved programs for the affected task across every policy under a
  separate manifest, not only trained failures. Six additional Level-2
  `File too large` logs involve candidate convolution padding that differs
  from the reference; an oversized candidate output can be a model mistake.
  Check shape/dtype before serialization and distinguish these cases from
  lack of space for a valid output. The 21 ENOSPC cases do not enter the
  original Level-2 primary comparison; no original result was changed.

- **2026-10-12 · API/name error reduction emerged as a descriptive follow-up hypothesis.**
  Per 960 attempts, base has 188 API/import/name traceback hints; means across
  saved seeds are C 16.33, S 23.67 and SFT 31.67. Exact-forward AST matches do
  not increase for C (base 94, C mean 92). Neither observation establishes a
  learning mechanism or the runtime implementation. **Rule:** test an explicit
  API-scaffold prompt equally across base and trained policies on sealed new
  tasks, alongside actual custom-kernel execution checks; do not claim that
  every API exception is hallucination or that source markers prove execution.
  New audit code intentionally lives outside `eval/` and the other v1.1 sealed
  trees to preserve reproducibility. All reported audit numbers are scripted
  from archived `runs/` evidence; namespace separation is not an exception to
  the requirement that every reported number have a reproducible script.
  The first audit's generic `error:` regex also matched I/O errors; r2 uses a
  narrower compiler hint plus a storage category and retains the first output.

- **2026-10-12 · Terminal study completed, but whole-archive download was interrupted.**
  R3 and the controller completed successfully; the terminal report preceded
  a 30,779,004,695-byte remote archive and an authorized shutdown command with
  return code 0. The old watcher used non-resumable SCP with a 30-minute limit;
  the supervisor also bounded its backup wait. The first local partial file was
  only 150,405,120 bytes. A restarted container became reachable around 17:15
  UTC, and the watcher began another copy from zero. Neither copy had a verified
  archive receipt or caused an acknowledgment. Stopped that watcher, retained
  the incomplete files explicitly as `.part`, and retrieved and hash-verified
  the small terminal reports, gate and shutdown evidence first. Replaced only
  the operational backup wrapper with persistent, resumable SFTP; it prioritizes
  the last seed's evidence/policy, checks the entire archive hash before use or
  acknowledgment, and retains the user's authorized shutdown after recovery.
  **Rule:** separate successful experiments and remote preservation from local
  backup completion. A shutdown timer must not be treated as a backup receipt;
  large archives need resumable transfer and small critical records first.
  No experiment code, frozen protocol, original verdict or result was changed.

- **2026-10-10 · Workstation monitoring vanished with temporary SSH state; remote work continued.**
  The saved monitor observation stopped at 01:41 UTC. At the 06:36 check its
  process and all `/tmp/cuda-gemma4-westb*` files were absent; workstation uptime
  was about 76 minutes. Restored the exact previously verified host-key pin in
  persistent storage and reauthenticated. The same GPU and original remote
  supervisor/worker were still running: SFT-20261022 had finished and C-20261022
  was training. No training restart or score rewrite was performed. Installed
  a restricted task-specific SSH public key (private key stays locally protected;
  no password saved), scoped SSH alias and enabled user systemd monitor/backup
  services. Fresh key authentication and new status observations passed.
  **Rule:** temporary control sockets alone cannot support unattended backup
  across workstation restarts; persist host pinning and scoped authentication,
  restart monitoring on user login, bound its observation lifetime, and preserve
  the prior outage evidence. A local monitoring outage is not a remote failure.

- **2026-10-09 · The SSH master disappeared; the local watcher then used the wrong known-hosts source.**
  The last successful monitor observation was 09:09 UTC. After timeouts, its
  fallback SSH connection reported host-key verification failure and the watcher
  exited after 20 failed attempts. The original master disappearance cause is
  unknown. Reconnecting at 16:31 UTC with the existing explicit pinned host-key
  file succeeded; the pinned key and GPU UUID were unchanged, and original
  supervisor/worker PIDs 1693/1706 had continued. Four formal evaluations had
  finished without a new stop. Preserved the failed watcher script/log/status,
  restored authentication without saving the password, added the explicit
  known-hosts file with strict checking, and restarted the local monitor.
  Its status now includes R3 jobs and distinguishes a pending gate from an
  unstarted study. **Rule:** distinguish transport/authentication failure from
  workload failure, retain host-key pinning, and verify live processes and
  durable results before claiming a shutdown, lost work or completion.

- **2026-10-08 · S-20261021 hit a detector stop at step 15; NaNs confound persistence detection.**
  One of 120 TRAIN programs, generated at sampling step 14 / call 5 / index 1,
  produced a nonfinite output, received reward 0 and `output_persistence`, and
  triggered the configured stop after optimizer step 15. Its sandbox completed
  normally. Source review found incorrect LayerNorm reductions and a mask that
  leaves output elements unwritten. The detector compares snapshots with
  `torch.equal`; a trusted CPU probe on remote PyTorch 2.13.0 confirmed that even
  bytewise unchanged NaNs fail that comparison. Local probing first found no
  Torch installation in the lightweight environment, so the hand-authored probe
  ran on remote CPU only. This demonstrates a detector confound, not proof of
  the exact candidate's flag cause or exploitation. Original output tensors
  had already been cleaned, leaving hashes and the nonfinite verdict.
  **Rule:** retain the frozen stop, policy and all scores; evaluate the actual
  stopped checkpoint and report its shorter training exposure. Do not silently
  patch the active study or resume past a declared stop. Any corrected detector
  needs separately versioned validation, including unchanged nonfinite outputs
  and real output mutation controls. The current controller continues to SFT
  after the stopped policy's final evaluation, as preregistered.

- **2026-10-07 · Workstation execution failed before two requested status checks could run.**
  Both the local monitor read and the SSH probe failed at command startup with
  `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. A ledger write
  also failed in that environment; this entry was added after access recovered.
  No SSH authentication attempt or remote stop was issued by those failed checks.
  Reconnection at 20:45 UTC found the original supervisor still running: the first
  R3 C seed had reached step 40 and its final evaluation was progressing.
  **Rule:** report an unavailable observation channel separately from a remote
  workload failure, retain the last verified timestamp, and verify live state
  after access is restored. Do not bypass the active execution restrictions.

- **2026-10-06 · Documentation bundle fetch requested a branch the bundle did not export.**
  Git returned `fatal: couldn't find remote ref master`; the incremental bundle
  exported `HEAD`. Inspected `git bundle list-heads`, fetched `HEAD`, and then
  fast-forwarded successfully; the frozen source verified unchanged. **Rule:**
  consume the bundle's advertised ref, and record deployment only after success.

- **2026-10-06 · SSH vanished during R2 and the operator subsequently rebooted.**
  Reconnection found the same host/GPU, no experiment workers and no terminal
  report, archive or shutdown request from the interrupted controller. Base had
  960 valid scores; C had 66 valid scores and all 960 generations. Cause of the
  interruption is unconfirmed; the new cgroup's zero OOM counters cannot explain
  the previous container. Preserved a clearly post-hoc interruption report and
  hash-verified metadata/results backup before resuming. **Rule:** absence of SSH
  does not establish completion or successful shutdown; inspect durable records.
- **2026-10-06 · Abrupt interruption retained 81,603,348,284 bytes of tensor scratch.**
  The unscored C attempt retained private inputs and sandbox work; an earlier
  deployment-invalidated audit also retained private inputs. Recorded a size/mtime
  manifest and backed up programs, seeds, requests, scores and logs before
  deleting only those named regenerable files. **Rule:** reclaim abandoned scratch
  only after verifying no live worker and preserving reconstruction evidence;
  retain completed model failures and original interrupted attempt directories.
- **2026-10-06 · The standard `all` entry point repeats a passed R1 on resume.**
  A hashed operational supervisor under `runs/deploy/` dispatches the unchanged
  `Study.reevaluate()` then `Study.confirm()` after checking R1, the source seal,
  ownership and the same GPU UUID. It retains watchdog, report, archive and
  shutdown behavior, plus the existing PASS-evidence commit workflow. The local
  watcher targets the new controller explicitly. **Rule:** preserve completed
  audit evidence on the same validated node; orchestration changes do not amend
  the frozen experimental source, sampling grid or resource caps.

- **2026-10-06 · Tar deployment restored a private study directory with workstation ownership.**
  `runs/replication-v1.1` was UID 1000, mode 0700. The existing sandbox only
  grants traversal through root-owned ancestors, so Torch/dill bootstrap raised
  `FileNotFoundError: No usable temporary directory found` before candidate
  source execution. C0–C3 passed in root-created directories and did not exercise
  this imported ancestor. Interrupted the control attempt, retained every row,
  recorded its invalid infrastructure scope, restored root ownership, and
  verified `CUDA_TRITON_OK` under the actual study directory. Started a fresh
  audit with unchanged source/configuration. **Rule:** unpack deployment packages
  with `tar --no-same-owner`; verify the imported study's ownership before launch.
  Do not weaken Landlock or relabel bootstrap crashes as model correctness.
- **2026-10-06 · The ad-hoc ownership-repair smoke tried to rewrite its own result.**
  `env.sandbox.run` had already written `result.json`, so the extra `write_json`
  raised `FileExistsError`. Reading the existing record and log confirmed
  component `ok`, return code 0 and `CUDA_TRITON_OK`. **Rule:** use the sandbox's
  authoritative result instead of attempting a duplicate immutable write.
- **2026-10-06 · Raw calibration paths also matched the archive work-directory exclusion.**
  The C2 `work/timings.json` was copied into an archive-safe node-evidence file
  with a source/hash manifest, while its original copy was retained and verified
  on the workstation. **Rule:** audit archive inclusion patterns for raw evidence,
  not only for expensive disposable tensor/compiler files.

- **2026-10-06 · The next operator-provided endpoint was another physical GPU.**
  The westb host has a new GPU UUID and driver 595.58.03. Reran R0 and C0–C3
  before v1.1's GPU audit; the fixed tokenizer then passed the real KL probe.
  **Rule:** tie startup evidence to the actual host and protocol, and distinguish
  a historical R1 FAIL from an in-progress retry. Copy raw calibration JSON/logs
  into the study before terminal archival so the timing summary stays auditable.
- **2026-10-06 · Verification again ran before a recursive SCP had finished.**
  The local manifest read raised `FileNotFoundError` while the transfer session
  was still running. No experiment was affected and no copy was acknowledged.
  **Rule:** inspect the yielded session result, not merely completion of the
  wait tool call; run checksum verification only after an actual exit code 0.

- **2026-10-06 · Local R0 invocation supplied a relative study path.**
  Artifact validation finished, but the record formatter raised
  `ValueError: 'runs/replication-v1.1' is not in the subpath of '/mnt/fastdata/cuda-gemma4-llm'`
  before a gate record was emitted. **Rule:** supply the absolute study path to
  `tools.gate_r0`; the remote operational wrapper already does this. Re-running
  the local command with the absolute path preserves the frozen source.

- **2026-10-06 · R1 on the clone failed on the tokenizer return type before training.**
  The actual error was `TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'`.
  Pinned Transformers 5.17.0 defaults `apply_chat_template(return_dict=True)`;
  both KL replay and SFT assumed the older token-list default. The old unit stub
  returned a list unconditionally and hid this API boundary. **Rule:** explicitly
  request token IDs through a shared helper, test the mapping default, and run
  the saved tokenizer on CPU. Eight retained TRAIN completions confirmed identical
  prompt tokens and completion-only labels after the fix. R1 remains FAIL until
  a real GPU retry passes; v1 evidence is preserved and v1.1 is a new source seal.
- **2026-10-06 · The early blockage guard powered off before interactive repair.**
  R0 and C0–C3 passed; the R1 failure triggered the authorized terminal archive
  and shutdown request. The local watcher verified the complete archive before
  SSH disappeared. The operator then required a report before shutdown. **Rule:**
  generate a terminal report before archiving on both success and failure,
  distinguish incomplete research from completed confirmation, and preserve
  the earlier shutdown evidence. The local recovery report is post-hoc; no
  claim is made that it existed on the remote host before that shutdown.

- **2026-10-06 UTC · First phase-2 deployment fetch raced its file transfer.**
  `git fetch` reported that `phase2-source.bundle` did not appear to be a Git
  repository because the SCP process was still running. The original staged C6
  evidence had already been saved in a named Git stash. Awaiting SCP completion
  and retrying the fetch produced a clean fast-forward; source verification and
  all 88 CPU tests then passed remotely. **Rule:** a yielded transfer is not a
  completed transfer; wait for its exit before consuming its artifacts.

- **2026-10-05 · Review caught overlapping SFT sample namespaces before any run.**
  A draft used `training_seed + offset + sample_index`; adjacent training seeds
  would therefore share almost their entire candidate pool. Replace arithmetic
  windows with task/sample namespaced SHA-256 seed derivation. Also separate
  TRAIN candidate, TRAIN validation and held-out input namespaces. Regression
  tests check non-overlap and reject undeclared sampling/input seeds in results.
  No experimental results were produced with the draft scheme.

- **2026-10-05 · Phase-2 preparation recovered a successful terminal backup.**
  The workstation watcher copied the 7,143,496,613-byte C6 archive and recorded
  its expected hash before shutdown. A fresh SHA-256 computation matched the
  remote manifest. Earlier assistant terminal failures (`bwrap: loopback:
  Failed RTM_NEWADDR: Operation not permitted`) prevented inspection, not the
  already-running watcher. **Rule:** distinguish an unavailable observation
  channel from a failed experiment or failed backup.
- **2026-10-05 · C6 automatic evidence commit failed after archiving.**
  Remote log: `Author identity unknown` and `fatal: unable to auto-detect email
  address`; shutdown still returned 0. C6 evidence and report were staged and
  preserved in the archive, then committed locally as `77e230a`. Configure Git
  identity only in this project, and preserve the remote staged state before
  deployment. Do not reinterpret an evidence-commit failure as failed training.
- **2026-10-05 · Phase-2 protocol deliberately separates uncertainty sources.**
  The original task set has been observed. New seeds are independent generation/
  training replications, not a new test set. Keep C0–C6 results intact; use R0–R3
  gates and a content-hashed protocol. Match SFT candidate budget and optimizer
  updates but report its extra TRAIN-only learning-rate selection cost. Task
  bootstrap intervals are conditional on checkpoints and cannot measure the
  population variance of training seeds. Do not select a checkpoint on held-out
  results or turn nonsignificance into equivalence.
- **2026-10-05 · KL source inspection narrows the audit but not the cause.**
  Pinned TRL computes `exp(ref_logp-policy_logp)-(ref_logp-policy_logp)-1` and
  disables the PEFT adapter for its reference. C step 34's logged KL was small;
  step 35 triggered the declared stop. The pilot did not preserve original
  completion-token log probabilities. Add a GPU adapter-disable invariance check
  and a clearly labelled retokenized, post-update replay. A successful replay
  cannot establish the exact historical cause or invalidate the recorded stop.

- **2026-10-05 · Freeze the initial pilot budget before held-out evaluation.**
  C and S each get at most 40 optimizer steps, full held-out evaluation every 10
  steps and at their actual final policy, within the kickoff's per-arm wall budget.
  Stop before a new segment if its reserved final-evaluation window would be lost.
  Retain all evaluated policy adapters and the latest two full optimizer states;
  retaining every full optimizer snapshot would consume storage without adding
  evaluated policies. All hyperparameters come from TRAIN/C5 evidence; no held-out
  outcomes have been inspected. **Rule:** reserve time and disk for final scoring
  and a verified backup before starting a rented-GPU pilot.

- **2026-10-05 · Shutdown implementation audit found a shell-format landmine.**
  AutoDL's `/usr/bin/shutdown` is plain shell text without a shebang. A shell can
  invoke it, but Python's direct executable call would fail with `ENOEXEC`.
  Run that exact platform script through `/bin/bash`; retain its return code/log.
  This was discovered by read-only inspection, before any shutdown attempt.
  **Rule:** verify the execution format of the unattended terminal action.

- **2026-10-05 · C5 second attempt made one saved optimizer step, but its learning
  signal was numerically suppressed.** The metric file records sequence importance
  weights near zero and a correspondingly tiny gradient despite mixed rewards.
  Pinned TRL defaults to `sequence_mask`: log-probability differences accumulate
  across long programs before exponentiation. Use its supported `token_truncate`
  correction, retain the default upper cap, and declare a mean-weight collapse
  guard in the config. This changes the pre-pilot optimizer plumbing, not reward
  or held-out selection. Also use Gemma's approved `<turn|>` stop token as the
  trainer tokenizer EOS: vLLM stopped at that token while TRL checked `<eos>` and
  reported normal endings as clipped. Stop the resumed process deliberately
  (observed exit `-9`) and retain its checkpoint/metrics as invalid pilot evidence.
  The user's autonomous-decision instruction authorizes a recorded compatibility
  retry after this second C5 failure. **Rule:** step counts alone do not establish
  an effective GRPO update; inspect correction weights, gradients and termination.

- **2026-10-05 · C5 first attempt failed before any optimizer step.** Actual error:
  `ValueError: There is no module or parameter named 'embed_vision.pos_embedding'
  in Gemma4UnifiedForConditionalGeneration.` TRL streams HF named parameters,
  including multimodal aliases that differ from the checkpoint names accepted by
  vLLM. This task has text-only prompts and LoRA only in the language model.
  Keep TRL's merge/unmerge and sync every language-model/head parameter; omit only
  explicitly named frozen multimodal namespaces. Reject unknown namespaces or
  trainable nontext weights. Record each sync's loaded/skipped names. **Rule:**
  standalone inference success does not validate live PEFT weight synchronization.

- **2026-10-05 · The fusion retry revealed a usable LayerNorm curriculum but the
  initial config demanded two successful tasks.** That extra task-count rule is
  not in kickoff's numeric signal criterion; kickoff explicitly permits narrowing
  the TRAIN subset. GELU generations contain invalid Triton APIs/autotune arguments,
  while LayerNorm has mixed successful and failed programs. Preserve both broad
  attempts, then validate LayerNorm alone with the same sampling, reward, timing
  and P_signal threshold. The operator's latest “自动决策 跑完或者卡住关机” authorizes
  this documented continuation past the two-failure consultation point. The gate
  records that authorization explicitly; default failure-stop enforcement stays.
  **Rule:** a TRAIN curriculum change must remain visible; never call it evidence
  of improved policy quality, and never select using held-out outcomes.
- **2026-10-05 · Local report CLI still printed `C6 is not implemented; see
  REPORT_STATUS.md. No execution performed.`** An old second `main` definition
  shadowed the implemented entry point even though import/statistics tests passed.
  Remove it and run the actual CLI against saved C4 records. **Rule:** validate
  executable reporting paths, not only their imported functions.

- **2026-10-05 · First C4 failed because both base policies had zero mixed-outcome
  signal.** The saved TRAIN generations mainly repeat eager reference operators;
  the prompt described correctness/timing but omitted a direct instruction to
  optimize. This is a pre-training task-prompt defect and a recorded saturation
  observation, not an RL result. Explicitly request fusion/efficient GPU kernels
  and narrow the retry to the two existing fusion tasks, LayerNorm and GELU.
  PyTorch remains legal; reward, tolerances, split, sampling settings and gates
  stay unchanged. **Rule:** preserve the first baseline and retry C4 before any
  optimizer step; stop if the second attempt still fails.

- **2026-10-05 · Ninja then died with `Signals.SIGSYS: 31`.** `strace` identified
  its `posix_spawn` child calling `setpgid(0, 0)`. Permit compiler process groups,
  while preserving the UID drop, `setsid`/namespace restrictions and rlimits.
  The watchdog now freezes, kills and reaps all dedicated-UID descendants,
  including distinct groups. The fork-bomb probe creates separate groups and C1
  must be re-run. **Rule:** compiler compatibility changes require corresponding
  containment coverage; never disable filtering for compilation.

- **2026-10-05 · Trusted inline CUDA smoke failed with `RuntimeError: Ninja is
  required to load C++ extensions (pip install ninja to get it)`.** Ninja was
  installed in the project venv, but the sanitized sandbox PATH omitted its bin
  directory. Add that already read-only, allowlisted directory to PATH. **Rule:**
  validate inline CUDA compilation as well as Triton before policy sampling.

- **2026-10-05 · Initial training scope decision, before C4 outcomes.** Many locked
  Level-1 elementwise tasks use 4096×393216 tensors; serializing repeated inputs
  and outputs would dominate an initial signal run. Start C4 on four existing
  TRAIN tasks (square/standard matmul, LayerNorm, MinGPT GELU), retaining their
  original shapes. This is a narrower training distribution, not a split or
  benchmark modification. Both Gemma models get 32 samples/task. **Rule:** select
  initial scope using only TRAIN source/resource needs, never held-out outcomes.
- **2026-10-05 · C3 confirmed the uniform-positive-input loophole.** Its deliberate
  default-distribution implementation passes naive verification and fails fresh
  signed inputs. The MinGPT GELU prompt now declares the full signed domain, and
  its training/held-out input protocol mixes original/signed/wider values without
  changing shapes. **Rule:** freeze this hardening before base-policy sampling.

- **2026-10-05 · The first trusted CUDA sandbox compatibility smoke failed** with
  `cudaGetDeviceCount ... Error 304: OS call failed or operation not supported on this OS`.
  Syscall tracing identified CUDA reads of `/proc/self/maps`, `/proc/devices`,
  `mmap_min_addr`, writes to its own thread `comm`, and a required AF_UNIX socket
  during initialization. `/proc/self` had also been resolved in the supervisor,
  rather than the child. This is a compatibility probe, before
  formal C1 and before hostile/generated execution. **Rule:** no hostile GPU test
  until a normal CUDA/Triton worker succeeds under the identical restrictions.
- **2026-10-05 · Design adaptation for CUDA initialization.** Landlock, UID drop
  and resource limits precede trusted Torch initialization; a seccomp TSYNC filter
  then denies socket creation/network/escape in every existing thread before any
  candidate is imported or executed. Only its own procfs task metadata is writable,
  in addition to the workdir and GPU devices. No reference/expected outputs enter
  the worker. **Rule:** do not allow candidate networking to make CUDA initialize.
- **2026-10-05 · Triton compilation hit `OSError: [Errno 18] Invalid cross-device link`.**
  Landlock ABI 1 forbids cross-directory rename even within the same filesystem.
  The trusted worker adapts Triton's cache writer to create its temporary file in
  the destination directory and preserves atomic `os.replace`. Normal CUDA/Triton
  smoke then printed `CUDA_TRITON_OK`. **Rule:** retain Landlock; adapt cache layout,
  never fall back to an unrestricted compiler or candidate execution.

Format: date · symptom · root cause · rule. Keep failures and superseded records.

- **2026-10-04 · Initial file access failed with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.**
  The tool execution sandbox could not initialize; the attempted unsandboxed read was rejected by the session's
  approval policy. The operator subsequently supplied an unrestricted environment. **Rule:** distinguish tool
  infrastructure failures from project gate failures; do not invent a document read or gate result.
- **2026-10-04 · Reuse audit found an hf-mirror fallback despite the kickoff's ModelScope requirement.**
  The older `setup_node.sh` still altered `~/.bashrc` and used hf-mirror; its success did not validate weights.
  **Rule:** use explicit ModelScope directories; check safetensors payload/index consistency, then load every
  tensor and offline config/tokenizer. C0 additionally requires a real 12B vLLM answer.
- **2026-10-04 · An apparently small ReLU reference has a large default allocation.**
  Pinned KernelBench source uses large benchmark shapes. **Rule:** inspect official reference input sizes before
  local smoke; use the two committed training-split matmul tasks, without changing benchmark shapes. This is
  plumbing validation, never a policy correctness or speedup score.
- **2026-10-04 · Kickoff asks for a gate record in commits and also forbids committing `runs/`.**
  These storage instructions conflict literally. **Rule:** ignore all `runs/`; copy passing gate JSON byte-for-byte
  to tracked `evidence/gates/`, print its SHA-256, and commit that evidence at a real gate pass.
- **2026-10-04 · First combined patch was rejected: `invalid patch: multiple operations target .../tools/gate_c0.py`.**
  Delete and add operations targeted the same file in one patch. No files in that patch were changed.
  **Rule:** update an existing file in one operation; inspect state after a rejected patch.
- **2026-10-04 · Integrated local preflight failed with `ModuleNotFoundError: No module named 'torch'` after standalone CUDA smoke passed.**
  `Path.resolve()` followed `.venv/bin/python` to uv's base interpreter, bypassing the virtual environment.
  **Rule:** make interpreter paths absolute without resolving symlinks. The failed record is retained at
  `runs/preflight/20261004T195138.576726Z/preflight.json`; this is not a C0 gate failure.

- **2026-10-04 · Staged whitespace check reported `new blank line at EOF` in scaffold files.**
  The scaffold generator appended an extra empty line. **Rule:** normalize generated file endings and run
  `git diff --cached --check` after staging newly created files; the unstaged check does not inspect untracked files.

- **2026-10-04 · AutoDL SSH probe returned `bash: line 1: python3: command not found`.**
  Noninteractive SSH did not include `/root/miniconda3/bin` or `/usr/local/cuda/bin` in PATH, although both
  image Python and CUDA were installed. **Rule:** inspect explicit image paths; do not infer missing software
  from the login shell's PATH. Project jobs use the explicit venv interpreter and `scripts/gpu_env.sh`.
- **2026-10-04 · AutoDL no-GPU host tools showed 208 CPUs and host-scale RAM, but cgroups allowed only 0.5 CPU / 2 GiB.**
  `lscpu` and `free` expose host resources in this container. **Rule:** inventory cgroup CPU/memory/pid limits;
  do not size compilation, model loading or rollout parallelism from host totals. Recheck on the GPU boot.

- **2026-10-04 · First AutoDL prepare stopped with `fatal: detected dubious ownership in repository .../third_party/KernelBench`.**
  Root's tar extraction restored the workstation UID from the benchmark archive, so Git correctly refused the
  imported repository. **Rule:** normalize archive owners and extract with `--no-same-owner`; fix ownership only
  on the freshly deployed benchmark checkout. Preserve the failed `runs/setup/prepare-20261004T201610-1409.log`.
  The CPU setup failure is not a C0 gate failure. Create an `autodl` branch after cloning the HEAD-only source bundle.

- **2026-10-04 · No-GPU downloader unexpectedly mapped PyTorch libraries and used about 0.5 GiB RSS.**
  The legacy `modelscope` import pulls in torch even though structural validation is standard-library-only.
  **Rule:** use the pinned `modelscope_hub.HubApi` file-download API; a hub-only import probe used about 40 MiB
  RSS with `torch` absent from `sys.modules`. The initial API probes returned
  `TypeError: HubApi.list_repo_files() missing 1 required positional argument: 'repo_type'` and
  `ImportError: cannot import name 'snapshot_download' from 'modelscope_hub'`; inspect installed signatures
  instead of assuming the legacy API is re-exported. No model was instantiated and no OOM was observed.
- **2026-10-04 · ModelScope range requests retried `RemoteDisconnected` / `Read timed out`, and SSH returned `Broken pipe`.**
  The connection interruption did not terminate the two remote download processes; partial files continued
  growing after reconnection. The cause of the network interruption is unconfirmed. **Rule:** retain range
  files for resume, verify source hashes, and make `tee` retain file logging when its SSH output pipe closes.
  These transport problems are not formal gate failures.

- **2026-10-04 · Optional no-GPU Transformers metadata probe was terminated with `Killed` / exit 137.**
  The container still had a 2 GiB RAM quota while both downloads ran. No metadata validation result was
  produced, and the exact SIGKILL source was not established. **Rule:** keep no-GPU verification to bounded
  file/hash/structure checks; defer Transformers config/tokenizer loading to the GPU boot with adequate RAM.
  Preserve `runs/setup/offline-metadata-direct-20261004.log` (empty) and the observed exit status; this was not C0.

- **2026-10-04 · Optional cubin-wheel probe reported `no version of flashinfer-cubin==0.6.18.post1`.**
  The probe used deprecated `--index-url` while the AutoDL environment set `UV_DEFAULT_INDEX` to Tsinghua;
  `--default-index https://flashinfer.ai/whl` resolved the exact version successfully. **Rule:** use uv's current
  index option for an intentional override. Pin the matching FlashInfer Python/cubin pair and the official
  wheel's release SHA-256; keep vLLM and FlashInfer runtime caches on the data disk.

- **2026-10-04 · AutoDL's cubin-wheel request failed after three retries: `client error (Connect)` / `operation timed out`.**
  The official index was reachable but its GitHub release asset was not reachable from the node during this
  attempt. **Rule:** transfer the official wheel from the workstation, compare it with the release SHA-256,
  and make setup prefer the verified `.cache/wheels/` file. Keep the matching package version pin; do not
  disable FlashInfer's version check or downgrade the inference stack to work around a transport failure.

- **2026-10-04 · First GPU-boot NVML sample showed 100% utilization with 0 MiB allocated and no listed processes.**
  The cause of this transient startup reading was not established. After the successful C0 workers exited,
  NVML showed 0% and 0 MiB. **Rule:** do not infer contention or a timing noise floor from a startup sample;
  measure steady-state behavior during C2. C0 made no performance claim.
- **2026-10-04 · Successful C0 generation warned `Triton kernel JIT compilation during inference: kernel_unified_attention`.**
  vLLM selected TRITON_ATTN for Gemma 4's heterogeneous head dimensions; the first inference encountered a
  shape not fully covered by engine warmup. **Rule:** separate initialization/JIT from steady-state timing;
  C2 must establish warmup and measurement behavior independently of C0's successful `READY` response.

The earlier setup entries predate generated candidates and hostile execution.
Current gate outcomes and research limitations are in REPORT_STATUS.md.

- **2026-10-05 · Workstation backup listener exited on one stalled SSH read.**
  The retained log reports `subprocess.TimeoutExpired` for the `archive.json` read
  after 30 seconds (2026-10-05 06:40 UTC). The remote C6 supervisor and both pilot
  arms continued; reconnection at 16:20 UTC found the final S evaluation running.
  **Rule:** handle a status-read timeout as a retryable transport failure, retain
  bounded consecutive-unreachable handling, and enable SSH keepalives. Restart
  the listener in a new provenance directory; keep the failed log. Archive hash
  verification remains mandatory before acknowledging a local copy. This repair
  changes only the workstation listener, not the frozen training or evaluation.
- **2026-10-05 · C6 arm C hit its declared KL stop at step 35.**
  `training-C/kill.json` records `reason: kl`, KL 242.33519557930413 above 0.2;
  the final step-35 adapter was saved and all 120 held-out tasks evaluated.
  The cause of the spike is not established. **Rule:** preserve the stop and
  final-policy evaluation; do not silently resume, raise the threshold, or
  substitute the earlier checkpoint. Arm S subsequently completed 40 steps;
  its final held-out evaluation was still running at this observation.
- **2026-10-06 · Cloned node has a different GPU driver and CPU/RAM allocation.**
  Source and protocol hashes matched, but the new host reports driver 580.95.05,
  22 CPU cores of quota and 110 GiB RAM. **Rule:** rerun C0–C3 on the actual GPU
  before phase-2 measurements; retain the old pilot evidence separately. The
  operational bootstrap is outside the frozen experimental source, has its own
  hash/provenance, commits new PASS evidence and preserves failures before shutdown.
- **2026-10-06 · Clone startup probe used an absent `python3`, then SCP targeted a missing directory.**
  Actual errors were `bash: line 1: python3: command not found` and
  `scp: dest open "/root/autodl-tmp/cuda-gemma4-llm/runs/deploy/": Failure`.
  No workload was launched by the failed syntax check. Created the operations
  directory, completed the transfer, validated syntax, then launched once.
  **Rule:** use the cloned `.venv/bin/python` explicitly and require successful
  transfer completion before executing a dependent remote command.
- **2026-10-06 · The existing pilot backup watcher's 73,000-second horizon is too short for phase 2.**
  **Rule:** the workstation's operational watcher waits up to eight days before
  invoking the existing checksum/acknowledgement transfer on a completed archive.
  This changes backup orchestration only; experimental source and seeds stay sealed.

- **2026-10-15 · Execution environment recovered after bwrap loopback initialization failures.**
  Local read/write and SSH now execute. No new experiment ran during the blocked
  interval. The follow-up preserves original input contracts instead of applying
  the v3 mixed-dtype/noncontiguous domain to old programs. Unknown runtime evidence
  is represented as null; weight initialization is separate from forward kernel
  compliance. **Rule:** separate observation failures and changed task contracts
  from model capability, and retain all attempted programs in denominators.
- **2026-10-15 · Fixed-host development intervention replaces generic API reminders.**
  Both arms use Triton and the same ABI/grid/output specification. The supplied
  wrapper performs allocation/metadata/launch only, and malformed fixed-host
  responses are retained as failures without stripping unwanted host code.
  **Rule:** report actual prompt/output cost as well as the common token cap;
  small developer-task results are not a confirmatory mechanism claim.

- **2026-10-15 · Full-program extractor incompatible with kernel-only intervention.**
  Review before policy generation found `env.reward.extract_program` requires
  `ModelNew`, so the new fixed-host condition would always lose valid kernel-only
  responses. Kept the old extractor/source immutable, added a shared fenced-code
  extractor for both v4 conditions and explicit condition-specific AST contracts.
  Added an end-to-end format regression and routed GPU wrapper controls through
  extraction. The first controller's passing controls/source were backed up;
  no policy output was sampled under the bad integration. **Rule:** control the
  actual completion-to-program path, not only a manually assembled kernel.
- **2026-10-15 · Restricted session returned while remote controller was paused.**
  Both SSH and local reads failed with `bwrap: loopback: Failed RTM_NEWADDR:
  Operation not permitted`; the paused parent could not reach automatic shutdown.
  Once access returned, verified it was stopped, archived evidence, retired it,
  and started a fresh controller. The local watcher now checks process state,
  not just PID existence, and restarts on failure. **Rule:** never report a
  stopped controller as a running experiment; idle GPU time is not progress.
