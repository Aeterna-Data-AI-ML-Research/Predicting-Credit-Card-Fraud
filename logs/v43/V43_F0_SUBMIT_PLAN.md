# V43 Fold-0 → sequential DS submit plan

**Armed by user:** submit after F0 success · one-at-a-time · wait for LB before next · use remaining daily slots.

## Preflight (now)
| Item | Value |
|---|---|
| Live tip | **0.95569** (Diag_A) |
| Today UTC submits | **6** |
| Daily limit (ladder) | **10** → **~4 left** |
| E4 NOSUBMIT | **quarantined — never submit** |
| F0 status | training (resumable ckpts); canary **EPOCHS=1** for $ budget |
| Balance | ~$4.3 @ A40 $0.59/hr |

## After `meta_f0.json` + gate pass (≥+5e-5, ≥4/5)
1. Rsync `/workspace/out/qwen/*.npy` → `out/v43_fold0_qwen/`
2. Run `gate_fold0_field24_local.py` → must `pass_gate=true`
3. Build tip×Qwen ladder CSVs in `out/v43_f0_submit_ladder/`
4. `V43_ARM_SUBMIT=1` → `v43_f0_sequential_submit_ladder.py`:
   - submit **one** CSV
   - **poll** `publicScore` to `complete`
   - if **UP** → next higher/neighbor weight
   - if **FLAT** → one more probe
   - if **2× DOWN** → stop (cut loss)
   - stop when quota empty

## Hard rules
- Never submit E4 / blacklisted rays
- Never parallel submits
- Never submit before gate pass
- Tip floor: do not replace Diag_A tip with a down-scored file as “best”
