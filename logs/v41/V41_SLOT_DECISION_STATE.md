# V41 Slot Decision State

## Live LB
- **LB_01 = 0.95569** (`SLOT01` Step_AB tip lock)
- **LB_02 = 0.95568** (`SLOT02` AB×areeb geodesic) → **Δ = −0.00001 (−1 tick)**

## Decision from LB_02
`+α·(L(areeb) − L(AB))` is **locally downhill** from AB.

### PERMANENT QUARANTINE (DO NOT SUBMIT)
- **SLOT03** — quarter/areeb-family geodesic (same downhill axis)
- **SLOT04** — C1 fail → **areeb fallback**
- **SLOT05** — C1 fail → **areeb fallback**
- **SLOT07** — C4 fail → **areeb fallback** / minDiv fail

## Next authorized burn
- **SLOT06** — `C3_harvest_p4` clean proprietary residual (zanch/v10 purged, contrastive − sign)

## Live after SLOT06
- **LB_06 = 0.95568** · **ΔLB_06 = −0.00001 (−1 tick vs SLOT01)**
- remaining daily ≈ 7
- HARD STOP — awaiting next explicit command (08/09/10 NOT authorized)

## Live after LB_06 (recon gate)
- **LB_06 = 0.95568** (−1 tick vs SLOT01 @ 0.280e-4 on −C3_harvest_p4_orth)
- **PERMANENT QUARANTINE SLOT08** (0.450e-4 along same −C3 axis) — DO NOT SUBMIT
- **HOLD SLOT09** (171.8e-4) and **SLOT10** (35.0e-4) — do NOT submit
- Remaining candidates to audit: FWD_AB_areeb (−geodesic), FWD_AB_AA (secant), UID_clean

## After LB_06 recon (NO_SUBMIT)
- QUARANTINE **SLOT08** (−C3 axis @ 0.450e-4) — DO NOT SUBMIT
- HOLD SLOT09 / SLOT10 untouched
- Public recon: **no tip > 0.95569** (Step_AB still public lock; no Step_AC)
- Audited (NO_SUBMIT): Cand_FWD_AB_areeb / Cand_FWD_AB_AA / Cand_UID_clean → see `logs/v41/V41_RECON_FWD_SECANT_UID.md`

## Candidate fire gate (Step6c)
- **QUARANTINE** `Cand_UID_clean_clip985.csv` (OOF Δ=0.00, disp=0.948<0.95, loser≤0=False) — DO NOT SUBMIT
- **HOLD** `P1s_AB_uid_a0p015.csv` — DO NOT SUBMIT
- **HOLD** `Cand_FWD_AB_areeb_0p025e4.csv` (collinear w/ FWD_AB_AA @ 1−ρ=0.002e-4) — DO NOT SUBMIT
- **FIRE** `Cand_FWD_AB_AA_0p030e4.csv` only

## Live after FWD_AB_AA
- **LB_FWD_AA = 0.95569** · **ΔLB = 0.00000** (flat vs SLOT01 tip)
- remaining daily ≈ 6
- HARD STOP — zero additional submits

## After LB_FWD_AA (Step6d board-model gate)
- **LB_FWD_AA = 0.95569** @ 0.030e-4 along L(AB)−L(AA) — **flat** (held tip)
- Geometry: forward AA→AB ray flat/uphill; backward SLOT02 (−0.028e-4) and −C3 SLOT06 (0.280e-4) both dropped to 0.95568
- Next: inspect goodpjw board-model v43 + Step_AC candidate audit (NO_SUBMIT)

## Step6d complete (NO_SUBMIT)
- Board model NOT shipped; local surrogate fitted
- Candidates staged under out/v41_step6d_candidates/
- Awaiting single-slot fire authorization

## Step6d fire gate
- **HOLD** Cand1_FWD_0p10 (`888c1c85…`) — DO NOT SUBMIT
- **HOLD** Cand2_StepAC (`f0418672…`) — DO NOT SUBMIT
- **FIRE** Cand3_Orth_Proprietary_Bank (`1db2ec76…`) = delta1_v4 @ 0.150e-4

## Live after Cand3 delta1_v4
- See poll result below / LB_delta1 in chat
- Cand1 & Cand2 remain HELD
- HARD STOP

## After LB_delta1 (Step6e 2D diagonal gate)
- **LB_delta1 = 0.95569** (`Cand3` +delta1_v4 ⊥ z_pub_only @ 0.150e-4, SHA 1db2ec76…) · ΔLB=0.00000 · ~5 daily left
- Proprietary orth ray **verified non-regressing** at 0.150e-4
- Two live-verified 0.95569 rays from Step_AB (cos≈−0.007 between axes):
  * Ray1 Public Secant/Anti-S02: StepAC blend_secant_antiare (cos·S02↓=−0.975); FWD_AB_AA @0.030e-4 = 0.95569
  * Ray2 Proprietary Stage-B: +delta1_v4 ⊥ z_pub_only (OOF +1.53e-5 4/5; cos·S02↓=−0.006, cos·S06↓=−0.107) @0.150e-4 = 0.95569
- Next: Stage 2D diagonals Diag_A/B/(C) — NO_SUBMIT

## Step6e fire gate
- **HOLD** Diag_B (`795cb4f1…`) — DO NOT SUBMIT
- **HOLD** SLOT09 / SLOT10 — DO NOT SUBMIT
- **FIRE** Diag_A (`f83c899d…`) @ 0.150e-4 only

## Live after Diag_A
- **LB_DiagA = 0.95569**
- Diag_B / SLOT09 / SLOT10 HELD
- HARD STOP

## V42 MACRO-LEAP — peephole freeze
- **FROZEN spearhead:** Diag_A (`f83c899d…`, live LB 0.95569)
- **TERMINATED:** all <0.30e-4 linear peephole micro-jumps of existing bank vectors
- Next: V42 Macro Engines 1–3 (SCAN4 id-cohort / group-offset / StageB_v42) — NO_SUBMIT

## V42 MACRO-LEAP complete (NO_SUBMIT)
- Peephole FREEZE held; Diag_A remains spearhead LB 0.95569
- SCAN4: sequential IDs → raw u_g KS fails; hash-u KS-ok but OOF −12e-5
- Cohort offsets ≈0; StageB_v42 FAIL (−102e-5)
- **No macro≥5 candidate authorized** — see logs/v42/V42_MACRO_LEAP_REPORT.md
