# GM V22 — MPS New Base Research (M3 Pro)

**Goal:** a solo base that can clear ≫0.950 OOF (path toward 0.954+ stack / eventually 0.95784).

## Hardware / env
- Apple **M3 Pro** · PyTorch **MPS** · `/tmp/s6e11_mps_venv`
- Matapanino Deotte Storyteller under `/tmp/s6e11_v16/lm_research/matapanino`
- MPS fix: skip per-step `.cpu()` sync (was stalling Metal)

## Throughput (measured)
| Model | batch | rows/s | notes |
|-------|------:|-------:|-------|
| DistilGPT-2 | 16 | **~19** | overnight workhorse |
| GPT-2 small | 8–16 | ~11 | too slow for full 5-fold tonight |

Full GPT-2 + 8ep + orig replica (masaya recipe) is **not** an overnight MPS job.

## Research lever
`DIGIT_TEXT=1` — keep exact CSV floats, append `cents / len / lead / lead2` (Fuad digit-leak hypothesis).

## Live schedule (pid in `run_v22/overnight.log`)
1. **Pilot** — DistilGPT-2, 20k fit rows, 2 ep, vanilla vs digit → pick winner  
2. **Fold0 canary** — full fold0, 2 ep, 30-order OOF vs masaya f0 **0.94888** (gate **≥0.9505**)  
3. **Full5** only if gate passes

## Expectation (honest)
DistilGPT-2 / no-replica canary tests the **digit-text** lever. Clearing 0.955+ solo likely still needs full GPT-2 + orig20k replica on more GPU; this run tells us whether digits move Fold0 before we spend that.

## Artifacts
- Log: `logs/overnight/GM_V22_MPS_NEW_BASE_RESEARCH.log`
- Run: `/tmp/s6e11_v16/lm_research/run_v22/`
- Driver: `scripts/gm_v22_mps_new_base_research.py`
