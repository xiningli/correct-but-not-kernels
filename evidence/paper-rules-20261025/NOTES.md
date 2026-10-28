# Paper Track rules check — 2026-10-25 UTC

Rendered copies of the public Overview and Rules tabs (fetched through a reader
proxy because Kaggle renders client-side) are `overview.md` and `rules.md`.

| Item | Official text (overview.md) |
|---|---|
| Deadline | November 12, 2026, 11:59 PM UTC; drafts and un-submitted Writeups are not judged |
| Length | "3,000 words maximum, reporting original, unpublished research" |
| Required sections | Title and subtitle; Abstract; Introduction; Description of the research, including methods and experiments; Related works and citations |
| Optional | Public notebook (Project Links, no login/paywall); arXiv-ready PDF via Public Project Link |
| Archival | Non-archival |
| Criteria | Novelty, Quality (generality), Relevance (SWE and agentic learning), Verifiability, Clarity; each 0–5, mean of five |
| Awards | Overall Best Paper $15k; Best New Resource $10k; Best New Application $10k |
| Model eligibility | No model requirement on the Paper Track overview or rules. "Submissions can include any related research, as long as it is unpublished." The QAT-variant Gemma 4 requirement reported by third parties belongs to the main competition. |

Not in the official pages: whether title/tables/references count toward 3,000
words (count everything to be safe); the host forum warning that purely
LLM-generated text scores poorly (reported by third-party mirrors, forum 745712).
The rendered Rules tab contained only Kaggle's Foundational Rules; §6c requires
OSI-licensed open-source code if open source is used in the submission.

Relevance risk: the track brief centres on agents that navigate codebases and fix
real software issues. Kernel generation fits "Tuning & Optimization" (RL) and
"Tasks & Benchmarks" (verifier design), but the paper must argue the SWE link.
