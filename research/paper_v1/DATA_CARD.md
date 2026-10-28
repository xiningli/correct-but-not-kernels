---
license: cc-by-4.0
pretty_name: Correct, But Not Kernels (data bundle)
tags:
- code
- gpu-kernels
- reinforcement-learning
- gemma
---

# Correct, But Not Kernels — data bundle

Data behind the Gemma 4 Paper Track writeup *Correct, But Not Kernels*
(paper and code: https://github.com/xiningli/correct-but-not-kernels, Apache-2.0;
this bundle: CC BY 4.0). Files keep repository-relative `runs/...` paths. Download
the bundle as `release-data/` inside a checkout of the code repository (the frozen R3 loader rejects paths outside it), then, with
PyTorch installed:

```bash
(cd release-data && sha256sum -c SHA256SUMS)
python -m research.paper_v1.reproduce --data release-data/data --out /tmp/paper-recheck
```

This rebuilds the R3 confirmation report (Section 4.1), the kernel-use
decomposition (4.2) and the training-dynamics tables (4.3), and compares each
with the committed receipts. No GPU is needed and nothing is re-executed.

| Contents | Paths under `data/runs/` |
|---|---|
| 5,600 saved Level-2 programs and their replay traces (ATen ops, CUDA kernel names) | `reliability-v2/capability-v4/`, `diagnostics-v5/terminal-audit-20261016/replay/` |
| R3 held-out scores: base and 9 fine-tuned policies, 960 programs each | `autodl-westb-reboot-20261006/*/extracted/...` |
| GRPO training rollouts (6 runs), SFT data and base candidates, checkpoint evaluations | `replication-v1.1/training/`, `replication-v1.1/confirmation/` |
| Worker logs for one seed's base candidates (crash causes) | `replication-v1.1/confirmation/SFT-20261022/...` |
| Consumer-GPU replay attempts | `portability-v1-20261020/run-r3/` |
| Issue-style repair records (Section 4.4) | `diagnostics-v5/terminal-audit-20261016/diagnostics/`, `diagnostics-v6-resume31b/terminal-audit-20261017/` |

Not included: Gemma weights and LoRA adapters (adapter hashes are in the code
repository's audit receipts), KernelBench sources (pinned revision and per-file
hashes in `env/kernelbench.lock.json`; MIT licence upstream), and compile workdirs.
R3 training files come from an archive with SHA-256
`8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74`; `MANIFEST.json`
lists every file's origin, size and hash.

Generated programs are outputs of `google/gemma-4-12B-it`. They were executed only
inside the study's sandboxes. Treat them as untrusted code.

## Public corpora (Section 4.5)

The two public corpora are not redistributed. Download them from their sources
and check the hashes recorded in `evidence/paper-public-audit-20261026/`:

| Corpus | Source | Licence |
|---|---|---|
| AI CUDA Engineer archive, `level_{1,2,3}.parquet` | huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive | CC BY 4.0 |
| KernelBook, `dataset_permissive.parquet` | huggingface.co/datasets/GPUMODE/KernelBook | June 9 Researcher Reciprocity License |

```bash
python -m research.public_audit_v1.sakana_static --data <dir with level_*.parquet> --out sakana-static.json
python -m research.public_audit_v1.kernelbook_static --data dataset_permissive.parquet --out kernelbook-static.json
```

Both scripts only parse source text and the archive's logged profiles; no corpus
code is compiled or executed.
