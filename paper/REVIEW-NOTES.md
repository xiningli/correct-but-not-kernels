# Pre-submission review notes (2026-10-25)

Not part of the submission. Numbers are machine-checked by
`python -m research.paper_v1.claims` (also a test in `research/paper_v1/test_paper.py`).

## Sentences to rewrite in your own voice

All nine sentences flagged on 2026-10-25 (aphorisms, "will not"/"form only"
overreach, "honest partial fusion", "nothing new to find", the 27-word subtitle) were
cut or rewritten on 2026-10-26 while fitting the related-work update into 3,000
words. The remaining prose is still machine-drafted; a full pass in your own voice
is still recommended (the host forum reportedly penalizes purely LLM-written text).

## Reviewer risks by criterion

- **Relevance (SWE / agentic learning).** §4.4 now measures the SWE link: on
  issue-style repair of two Liger-Kernel defects (issue text + buggy source, hidden
  fix and withheld tests, strict contract), the six fine-tuned checkpoints resolve
  0/72 single-shot attempts vs base 2/12; iterative repair 12B 2/12, 31B 7/12. Small
  and development-seen (two defects, one repository): present as a transfer probe,
  never as a repair benchmark or as evidence that fine-tuning harms repair.
- **Novelty (now the largest risk).** Decorative kernels are published: Dr. Kernel
  [15] names "Triton kernels emitted but never executed", detects them by
  instrumenting Triton launches, and reports ~20% → ~3% hacking during RL with the
  check on; TritonRL [14] shows AutoTriton's Level-2 correctness is 1% with a call-level
  verifier and 94% without. Our claim must stay the *controlled counterfactual*: no
  check, pre-registered held-out evaluation, three seeds, replication, RL vs RFT,
  training-time share rising 58% → 73%, and a trace-validated static classifier.
  Do not claim to introduce the phenomenon or the runtime check.
  §4.5 (2026-10-26) adds a per-kernel audit of two public corpora: decoration is rare
  (68 of 16,459 AI CUDA Engineer kernels; 0 of 18,162 KernelBook pairs) but GEMM/conv
  delegation is common (archive L2 2,734 of 4,750 "faster" kernels; L3 296 of 1,161
  launch no custom kernel; KernelBook 59%). Prior coverage: Hari et al. reviewed only
  the best kernel per problem (3 of 57 PyTorch-only). Corpus classes are static,
  checked against the archive's own logged profiles (98/99/90% agreement), not
  re-executed; delegation is legitimate partial fusion, not cheating. Kevin [2] is the
  try/except variant.
- **Discussion 1.7% claim.** Scoped to an instruct model with no kernel SFT and one
  task; [13–15] all cold-start SFT before RL, so do not argue strict rewards fail
  in general.
- **Quality.** §4.2 has 12 unadjusted 95% intervals; only the total row is
  pre-registered. C − SFT decorative (+4.79) and plain (−4.29) are exploratory —
  the text already says "we leave its cause open"; keep it that way.
- **Verifiability.** Repository and dataset links are TODO. The bundle was rebuilt
  clean at 37b73c7; rebuild once more from the final submission commit.
- **Clarity.** "Calls its own kernel" (§4.3, static `reachable`) and "launched"
  (§4.2, runtime) are different measurements; the table header says "static class",
  but a reader can still conflate them.

## Checked and fine

- Base samples, GRPO rollouts: same prompt, temperature 1.0, top_p 0.95, top_k 64,
  4,096 tokens, reasoning off, one fresh check (`train/launch.py:184`,
  SFT candidate `request.json`, `env/run_kernel.py` default `trials=1`).
- "466 mention Triton" was the R3 held-out base count but sat in a LayerNorm
  context; replaced with 613/39 (LayerNorm) and 274/12 (Level 2), both derived
  from receipts.
- 9,600 programs = 10 policies × 120 tasks × 8 (sum of R3 failure components).
