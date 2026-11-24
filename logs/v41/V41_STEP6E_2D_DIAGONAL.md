# V41 Step6e — 2D Diagonal Candidates

**NO_SUBMIT=True** · LB_delta1=0.95569 (prop ray non-regressing @ 0.150e-4)

## Dual live 0.95569 rays
- Ray1 StepAC/anti-S02 · Ray2 +δ1_v4 · cross cos = **-0.0051**

## δ_v4_combo quick check
- δ1_v4: Δe5=1.53 pos=4/5 loser_ok=True disp=0.9995
- δ_combo (d1+d3lin−d3bnd): Δe5=6.94 pos=5/5 loser_ok=False disp=1.0004 · beats_d1=False

## Step_AB internal levers
- Exact Sriram/RECENT/MOVE6/MSTEP **not on disk** — tip-geometry proxies only (see JSON).

## Staged candidates

| Cand | SHA | 1−ρ AB | vs FWD030 | vs Cand3 | vs S02 | vs S06 | cos·S02↓ | cos·S06↓ | OOF Δe5 | pos | Zr | disp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| Diag_A | `f83c899dbe63…` | 0.1500 | 0.1040 | 0.0554 | 0.2507 | 0.4957 | -0.608 | -0.147 | δ1:1.53 | 4/5 | True | 0.9995 |
  - full sha `f83c899dbe63d69ad104f891da1c968a3869c767cb562e81058787f69881a817` · file `Diag_A_0p050AC_0p100d1_tot0p150e4.csv`
| Diag_B | `795cb4f1c919…` | 0.2200 | 0.1669 | 0.0605 | 0.3278 | 0.5781 | -0.556 | -0.146 | δ1:1.53 | 4/5 | True | 0.9995 |
  - full sha `795cb4f1c919775e65c27fd8227b59fff31eace64db202a9302baa22b203c032` · file `Diag_B_0p060AC_0p160d1_tot0p220e4.csv`
| Diag_C | — | — | — | — | — | — | — | — | — | — | — | SKIP: δ_v4_combo does not beat δ1 on OOF+loser (combo Δe5=6.94 loser=False vs δ1 Δe5=1.53 loser=True) |

## STOP
NO_SUBMIT=True. Awaiting single-slot fire authorization.
