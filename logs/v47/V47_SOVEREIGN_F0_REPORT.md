# V47 SOVEREIGN F0 REPORT — HARD STOP (billing gate)

**NO_SUBMIT=True · 0 submissions · 0 public competitor predictions admitted**

## Step 1 — DONE ($0 local)

- Schema scan: `logs/v47/STEP1_SCHEMA_SCAN.md` — local `orig20k` 20k×26, 25/25 train overlap, both schema flags present; HF/OpenML/Kaggle public searches did **not** surface a sister table that explains tip `0.96153` as a verified sovereign self-train.
- `z_sovereign_v0` (14 self-owned streams, Deotte logit+std nested LR C=0.03):
  - **overall AUC = 0.952251**
  - **Fold-0 va0 AUC = 0.951251** (n=139,992)
  - fold AUCs: `[0.951251, 0.952438, 0.957075, 0.952647, 0.947987]`
  - saved: `preds/v47_sovereign/z_sovereign_v0_{oof,test}.npy`

## Step 2 — BLOCKED (cannot create billable pod)

| Check | Result |
|---|---|
| `get-balance` | **balanceAmount = −0.0008226931** · PREPAID |
| PRE_EXISTING pods | **`u4ky2oq4pohcyt`** · EXITED · A40 · $0.59/hr · name `v43-f0-qwen-field24` · **DO NOT TOUCH** |
| V47 pods created | **0** |
| Active V47 billing | **$0.00/hr** |

Secure stock/price (live): A40 **$0.59/hr** (Low), RTX 4090 **$0.89/hr** (Low), L40S **$1.09/hr** (Low). Hard cap ~$3.50 → ~4.5h dead-man at A40.

Pod package ready (not launched):
- `scripts/v47_sovereign/pod_f0/hgdt_lm_fold0.py` — HGDT dual-stream A1/A2/A3 → `hgdt_f0_{seq,field24}_{llr,std}_{oof,test}.npy`
- `scripts/v47_sovereign/pod_f0/qwen_f0_runner.sh` — Qwen2.5-0.5B 4-ep bf16 SDPA, K=4, `USE_COMPILE=0`
- `scripts/v47_sovereign/pod_f0/on_pod_main.sh` — dead-man `sleep 16200 && remove $POD_ID` + SHA256 manifest

## Step 3 — Gate not runnable (no pod artifacts)

- Gate script: `scripts/v47_sovereign/step3_f0_gate.py`
- Threshold: **ΔAUC_f0 ≥ +8.0e-5** and **≥4/5** subfolds → scale winner(s) to full 5-fold
- Status: **BLOCKED_NO_POD_ARTIFACTS**

## HARD STOP

No Fold-0 GPU canary until prepaid balance is topped up above $0. On top-up: create **only** a new Secure A40/4090/L40S V47 pod (never `u4ky2oq4pohcyt`), arm dead-man, rsync → `preds/v47_pod_f0/`, tear down, run Step3 gate, keep **NO_SUBMIT=True**.
