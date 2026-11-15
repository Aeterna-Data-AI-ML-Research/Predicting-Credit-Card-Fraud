# GM V22 — Local full GPT-2 + replica (speed pivot)

## Why Docker was stopped
Mac Docker = **CPU only** (~1.5 rows/s) → ~**9 days** for 60-ep replica. Too slow.

## New path: host **MPS** (M3 Pro Metal)
- Measured replica ~**10–11 rows/s** (~7× Docker)
- Docker container `s6e11_gpt2_storyteller` **stopped/removed**

## Live schedule (MPS)
| Stage | Setting | Est. time |
|-------|---------|-----------|
| Replica | GPT-2, **20 ep**, batch 8, bf16 | ~**10–12 h** |
| Fold0 classify | **2 ep** from replica | ~**1–1.5 d** |
| Fold0 OOF score | 30 orders | hours |

Full Deotte 60-ep replica ≈ ~31 h on MPS; set `REP_EPOCHS=60` to match the post exactly after the 20-ep canary.

## Resume + monitor (added)
- **Checkpoints every 500 steps** (`SAVE_STEPS=500`); SIGTERM saves + exits cleanly
- Replica resumes from `replica/trainer/checkpoint-*`; classifier from `fits/f0/resume.pt`
- **Watchdog** `scripts/gm_v22_mps_watchdog.sh` — restarts pipeline if it dies before `DONE`
- **CPU/mem monitor** `scripts/gm_v22_mps_monitor.sh` → `logs/overnight/GM_V22_MPS_MONITOR.log` (30s)

## Paths
- Run: `/tmp/s6e11_v16/lm_research/run_mps_gpt2/`
- Script: `scripts/gm_v22_mps_full_gpt2_replica.sh`
- Log: `…/pipeline.log` / `nohup.out`
