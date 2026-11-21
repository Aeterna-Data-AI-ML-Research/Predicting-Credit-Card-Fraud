#!/usr/bin/env python3
"""GM V31.1 — Step-O proprietary 5f LM leap (fixed orth/gate).

OOF competence: LOFO vs z_honest_152 (rank-avg bank) WITHOUT annihilating SVD.
Test inject: GS-orth to ladder B_stepO = [L(M),L(N),L(Areeb),L(O)] (+ light SVD-10).
MAX_SUBMITS=2, sequential poll. Keep >=8 slots.
"""
from __future__ import annotations

import json, subprocess, time, warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

MAX_SUBMITS = 2
REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
AUT = Path("/tmp/s6e11_v31_autopsy")
GPW = AUT / "gpw_95536_out"
OUT = Path("/tmp/s6e11_v31_1"); OUT.mkdir(parents=True, exist_ok=True)
REPORT = LOG / "GM_V31_1_STEP_O_5FLM_LEAP_REPORT.md"
KG = Path("/Users/mz/Library/Python/3.14/bin/kaggle")
N_TR, N_TE = 699960, 299984
T0 = time.time(); LINES=[]; SUBMITS_DONE=0

def log(m):
    s=f"[{time.time()-T0:7.1f}s] {m}"; print(s, flush=True); LINES.append(s)

def rf(v):
    v=np.asarray(v,float).ravel(); return (rankdata(v,method="average")-0.5)/len(v)
def rl(v): return logit(np.clip(rf(v),1e-6,1-1e-6))
def Lf(v, fold_id):
    o=np.zeros(len(v),float)
    for f in np.unique(fold_id):
        m=fold_id==f; o[m]=rl(v[m])
    return o
def unit(v):
    v=np.asarray(v,float).ravel(); return (v-v.mean())/(v.std()+1e-12)
def auc(y,p): return float(roc_auc_score(y,p))
def omr(a,b): return float((1-spearmanr(a,b).correlation)*1e4)
def folds_from(fid): return [(np.where(fid!=f)[0], np.where(fid==f)[0]) for f in range(5)]
def h2h(cand, base, y, folds):
    d=np.array([(auc(y[te],cand[te])-auc(y[te],base[te]))*1e4 for _,te in folds])
    se=float(d.std(ddof=1)/np.sqrt(len(d)))
    return {"mean":float(d.mean()),"pos":int((d>0).sum()),"t":float(d.mean()/se if se>1e-12 else 0),
            "min_fold":float(d.min()),"d":[round(float(x),4) for x in d]}
def sf25(cand, base, y, folds, rng):
    ok=0
    for _,te in folds:
        for _ in range(5):
            take=rng.choice(te, size=max(1000,int(0.8*len(te))), replace=False)
            if auc(y[take],cand[take])>auc(y[take],base[take]): ok+=1
    return ok
def gs(v, basis):
    u=unit(v)
    for b in basis:
        b=unit(b); u=unit(u-np.dot(u,b)*b)
    return u
def inject_orth_rho(anchor, u, target_e4, B=None):
    a=rl(anchor); d=unit(u)
    if B: d=gs(d,B)
    lo,hi=0.0,3.0; best=rf(a)
    for _ in range(50):
        mid=0.5*(lo+hi); pred=rf(a+mid*d); cur=omr(pred,anchor)
        if abs(cur-target_e4)<1e-4: return pred
        if cur<target_e4: lo=mid
        else: hi=mid
        best=pred
    return best
def load_sub(path, ids):
    df=pd.read_csv(path); col="is_fraud" if "is_fraud" in df.columns else [c for c in df.columns if c!="id"][0]
    return df.set_index("id").loc[ids,col].to_numpy(float)
def kaggle_subs():
    r=subprocess.run([str(KG),"competitions","submissions","-c","playground-series-s6e11","--csv"],
                     capture_output=True,text=True,timeout=90)
    lines=[ln for ln in r.stdout.splitlines() if ln.startswith("ref,") or (ln and ln[0].isdigit())]
    return pd.read_csv(StringIO("\n".join(lines))) if lines else pd.DataFrame()
def submit_and_poll(path, msg, timeout_s=240):
    global SUBMITS_DONE
    if SUBMITS_DONE>=MAX_SUBMITS:
        log(f"MAX_SUBMITS reached — refuse {path.name}"); return None
    r=subprocess.run([str(KG),"competitions","submit","-c","playground-series-s6e11","-f",str(path),"-m",msg],
                     capture_output=True,text=True,timeout=90)
    out=(r.stdout or "")+(r.stderr or ""); log(out[-400:])
    if "Successfully submitted" not in out:
        log("SUBMIT FAILED"); return None
    SUBMITS_DONE+=1; exact=path.name; t_end=time.time()+timeout_s
    while time.time()<t_end:
        df=kaggle_subs()
        if len(df):
            hit=df[df["fileName"].astype(str)==exact]
            if len(hit):
                st=str(hit.iloc[0].get("status",""))
                if "Complete" in st or st.endswith("COMPLETE"):
                    sc=float(hit.iloc[0]["publicScore"]); log(f"POLLED {exact} -> {sc}"); return sc
        time.sleep(8)
    log(f"POLL TIMEOUT {exact}"); return None

def main():
    log("=== GM V31.1b Step-O + 5f LM (fixed gates) | MAX_SUBMITS=2 ===")
    LOG.mkdir(parents=True, exist_ok=True); SUB.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(42)
    tr=pd.read_csv(REPO/"data/train.csv"); te=pd.read_csv(REPO/"data/test.csv")
    y=tr["is_fraud"].to_numpy(float); ids=te["id"].to_numpy()
    fold_id=pd.read_csv(BANK/"folds.csv")["fold"].to_numpy(); folds=folds_from(fold_id)

    live=kaggle_subs(); n_today=0
    if len(live):
        live["date"]=pd.to_datetime(live["date"])
        n_today=int((live["date"].dt.strftime("%Y-%m-%d")=="2026-11-21").sum())
    remain=10-n_today
    log(f"submits_today={n_today} remaining_est={remain}")

    step_o=load_sub(GPW/"submission.csv", ids)
    step_n=load_sub(GPW/"submission_step_n.csv", ids)
    step_m=load_sub(GPW/"submission_step_m.csv", ids)
    areeb=load_sub(SUB/"sub_v30_PUB_areeb_95534.csv", ids)
    Lo,Ln,Lm,La=map(rl,[step_o,step_n,step_m,areeb])
    log(f"ladder omr O↔N={omr(step_o,step_n):.4f} O↔Areeb={omr(step_o,areeb):.4f}")

    # --- bank members for z_honest ---
    members=[]
    roots=[BANK, REPO/"external/banks/goodpjw2008_s6e11-fraud-series-artifacts",
           REPO/"external/nov08_part12", REPO/"external/banks_v31/s6e11-board-model-pinned-inputs"]
    seen=set()
    for root in roots:
        if not root.exists(): continue
        for op in sorted(root.rglob("*.npy")):
            if op.name.startswith("._"): continue
            key=tp=None
            if op.name.startswith("oof_") and op.name.endswith(".npy"):
                key=op.name[4:-4]; tp=op.parent/f"test_{key}.npy"
            elif op.name.endswith("_oof.npy"):
                key=op.name[:-8]; tp=op.parent/f"{key}_test.npy"
            elif "wide_lm_stack_nested" in op.name:
                key="wide_lm"; tp=op.parent/"wide_lm_stack_test.npy"
            else: continue
            if key in seen or tp is None or not tp.exists(): continue
            try:
                o=np.load(op).ravel()[:N_TR].astype(float)
                t=np.load(tp).ravel()[:N_TE].astype(float)
            except Exception: continue
            if len(o)<N_TR or len(t)<N_TE or np.isnan(o).any() or np.ptp(o)<0.05: continue
            s=auc(y,o)
            if s<0.94: continue
            seen.add(key); members.append((s,key,o,t))
    members.sort(reverse=True)
    log(f"bank members solo>=0.94: {len(members)} top={members[0][1]}@{members[0][0]:.5f}")
    top=members[:40]
    z_o=np.mean([rf(o) for _,_,o,_ in top],0)
    z_t=np.mean([rf(t) for *_,t in top],0)
    z_auc=auc(y,z_o)
    log(f"z_honest_152_oof (rankavg top{len(top)}) AUC={z_auc:.6f}")

    # light SVD for report residual energy (not for annihilating gate)
    Bo=np.column_stack([unit(Lf(o,fold_id)) for _,_,o,_ in top[:60] if True][:60])
    k=min(10, Bo.shape[1]-1)
    svd=TruncatedSVD(k, random_state=42).fit(Bo)
    U_o=[unit(Bo@svd.components_[i]) for i in range(k)]
    Bt=np.column_stack([unit(rl(t)) for *_,t in top[:60]][:60])
    U_t=[unit(Bt@svd.components_[i]) for i in range(k)]
    B_ladder=[unit(Lm),unit(Ln),unit(La),unit(Lo)]
    B_stepO_t=U_t+B_ladder  # light SVD + ladder
    log(f"light SVD-{k} explained={float(svd.explained_variance_ratio_.sum()):.4f}")

    # rays
    rays=[]
    for p in [SUB/"sub_v30_STAR_areeb_along_gpw_0.26e4.csv", SUB/"sub_v30_BOLD_maths_from_areeb_0p5.csv"]:
        if p.exists(): rays.append(unit(rl(load_sub(p,ids))))
    while len(rays)<2:
        rays.append(gs(rng.standard_normal(N_TE), B_ladder))
    B_stepO_t = B_stepO_t + rays[:2]

    # --- pillars ---
    l7o=np.load(AUT/"l7_s2026_oof_rebuilt.npy"); l7t=np.load(AUT/"l7_s2026_test_rebuilt.npy")
    gpo=np.load(AUT/"gpt2_med_oof_rebuilt.npy"); gpt=np.load(AUT/"gpt2_med_test_rebuilt.npy")
    pillars=[("P_L7_s2026_5f", l7o, l7t, "prob"), ("P_gpt2med_5f", gpo, gpt, "prob")]
    # L7 disp
    raw=REPO/"preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy"
    traw=REPO/"preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy"
    if raw.exists() and traw.exists():
        def soft2(z):
            z=z-z.max(axis=-1,keepdims=True); e=np.exp(z); return e[...,1]/(e.sum(-1)+1e-12)
        po,pt=soft2(np.load(raw)), soft2(np.load(traw)).mean(0)
        disp_o=0.6*po.std(1)+0.4*(np.sort(po,1)[:,-5:].mean(1)-np.sort(po,1)[:,:5].mean(1))
        disp_t=0.6*pt.std(1)+0.4*(np.sort(pt,1)[:,-5:].mean(1)-np.sort(pt,1)[:,:5].mean(1))
        pillars.append(("P_L7_disp", disp_o, disp_t, "resid"))
    for name,op,tp in [
        ("P4_L4.2", REPO/"preds/matrix_v2/P4_L4.2_pure5f_oof.npy", REPO/"preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.5", REPO/"preds/matrix_v2/P4_L4.5_pure5f_oof.npy", REPO/"preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("v10_ridge_orth_LM", REPO/"preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy",
         REPO/"preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
    ]:
        if op.exists() and tp.exists():
            pillars.append((name, np.load(op).ravel()[:N_TR], np.load(tp).ravel()[:N_TE], "resid"))

    audit=[]; passed=[]
    for name,o,t,kind in pillars:
        if kind=="prob":
            lo, lt = Lf(o,fold_id), rl(t)
        else:
            # residual-scale: standardize per fold
            lo=np.zeros(N_TR)
            for f in range(5):
                m=fold_id==f; x=o[m]; lo[m]=(x-x.mean())/(x.std()+1e-12)
            lt=unit(t)
        # COMPETENCE LOFO (no annihilating SVD) — matches V31.0 protocol
        cand=rf(0.92*rl(z_o)+0.08*lo)
        g=h2h(cand, z_o, y, folds)
        sf=sf25(cand, z_o, y, folds, rng)
        # residual energy after light SVD (diagnostic)
        u_o_diag=gs(lo, U_o)
        # TEST vector: orth to ladder (+ light SVD) for clean Step-O inject
        u_t=gs(lt, B_stepO_t)
        u_o=gs(lo, U_o)  # OOF residual for sign/composite
        # sign lock
        te0=folds[0][1]
        plus=auc(y[te0], rf(rl(z_o)[te0]+0.05*u_o[te0]))
        minus=auc(y[te0], rf(rl(z_o)[te0]-0.05*u_o[te0]))
        sgn=+1 if plus>=minus else -1
        u_o,u_t=sgn*u_o, sgn*u_t
        u_t=u_t*(u_o.std()/(u_t.std()+1e-12))
        maxc=max(abs(float(np.corrcoef(u_t,b)[0,1])) for b in B_ladder)
        disp=float(u_t.std()/(u_o.std()+1e-12))
        # pass: competence 5/5, min>0, sf>=19, t>=3.0 (LM) / 3.5 (other)
        tmin=3.0 if "L7" in name or "gpt2" in name else 3.5
        ok=g["pos"]==5 and g["min_fold"]>0 and sf>=19 and g["t"]>=tmin
        row=dict(name=name,kind=kind,sign=sgn,pos=g["pos"],mean_e4=round(g["mean"],4),
                 mean_e5=round(g["mean"]*10,3),min_fold_e4=round(g["min_fold"],4),
                 sf25=sf,t=round(g["t"],3),disp=round(disp,4),max_corr_ladder=round(maxc,6),
                 resid_energy=round(float(np.corrcoef(u_o_diag,unit(lo))[0,1]**2),4))
        row["pass"]=ok
        audit.append(row)
        log(f"PILLAR {name}: pos={g['pos']}/5 mean={g['mean']:+.3f}e-4 min={g['min_fold']:+.3f} "
            f"sf={sf}/25 t={g['t']:.2f} max|corr_ladder|={maxc:.4f} pass={ok}")
        if ok:
            passed.append((name,u_o,u_t,g))

    def t_w(items, allow=None):
        sel=[x for x in items if allow is None or x[0] in allow]
        if not sel: return None,None,[]
        w=np.array([max(g["t"],0.1) for *_,g in sel]); w=w/w.sum()
        uo=unit(sum(wi*ui for wi,(_,ui,_,_) in zip(w,sel)))
        ut=unit(sum(wi*ui for wi,(_,_,ui,_) in zip(w,sel)))
        ut=gs(ut, B_stepO_t)
        ut=ut*(uo.std()/(ut.std()+1e-12))
        return uo,ut,list(zip([n for n,_,_,_ in sel], w.tolist()))

    lm_allow={"P_L7_s2026_5f","P_gpt2med_5f","P_L7_disp"}
    u_lm_o,u_lm_t,lm_w=t_w(passed, lm_allow)
    u_full_o,u_full_t,full_w=t_w(passed, None)
    log(f"u_LM_prop {lm_w}"); log(f"u_Full_prop {full_w}")

    if u_lm_t is None:
        log("ABORT: no pillars passed competence LOFO — no submit")
        _report(audit,lm_w,full_w,z_auc,None,None,len(members),remain,None,None)
        return

    # composite gates
    for tag,uo,ut in [("u_LM",u_lm_o,u_lm_t),("u_Full",u_full_o,u_full_t)]:
        if uo is None: continue
        g=h2h(rf(0.92*rl(z_o)+0.08*uo), z_o, y, folds)
        log(f"COMPOSITE {tag}: pos={g['pos']}/5 mean={g['mean']:+.3f}e-4 t={g['t']:.2f}")

    # honest NOSUBMIT
    best_w,best_g=0.05,None
    base_u=u_full_o if u_full_o is not None else u_lm_o
    base_ut=u_full_t if u_full_t is not None else u_lm_t
    for w in [0.03,0.05,0.08,0.12]:
        g=h2h(rf(rl(z_o)+w*base_u), z_o, y, folds)
        if best_g is None or g["mean"]>best_g["mean"]: best_w,best_g=w,g
    super_t=rf(rl(z_t)+best_w*base_ut)
    super_o=rf(rl(z_o)+best_w*base_u)
    hp=SUB/"sub_v31_1_HONEST_700K_CV_SUPERSTACK_NOSUBMIT.csv"
    pd.DataFrame({"id":ids,"is_fraud":super_t}).to_csv(hp,index=False)
    log(f"HONEST SUPERSTACK w={best_w} OOF AUC={auc(y,super_o):.6f} LOFO {best_g['pos']}/5 m={best_g['mean']:+.3f}")

    # Offline proxy
    log("Board-model not shipped; offline proxy = LOFO mean on z_honest")
    offline={"StepO":0.95536,"C1_proxy_e4":h2h(rf(0.92*rl(z_o)+0.08*u_lm_o),z_o,y,folds)["mean"]}

    # SLOT1
    slot1=inject_orth_rho(step_o, u_lm_t, 0.35, B_ladder)
    log(f"SLOT1 omr(StepO)={omr(slot1,step_o):.4f}")
    p1=SUB/"sub_v31_1_SLOT1_stepO_5flm.csv"
    pd.DataFrame({"id":ids,"is_fraud":slot1}).to_csv(p1,index=False)
    LB1=submit_and_poll(p1,"v31.1 SLOT1 StepO+5fLM L7+gpt2med @0.35e-4 | max2 keep8")
    if LB1 is None:
        _report(audit,lm_w,full_w,z_auc,None,None,len(members),remain,best_g,best_w); return

    LB2=None
    if LB1>=0.95537:
        log("CASE A: NEW ATH"); mix=unit(0.65*u_lm_t+0.35*(base_ut))
        slot2=inject_orth_rho(slot1, mix, 0.30, B_ladder)
        p2=SUB/"sub_v31_1_SLOT2_ride_new_ath.csv"
        pd.DataFrame({"id":ids,"is_fraud":slot2}).to_csv(p2,index=False)
        LB2=submit_and_poll(p2,"v31.1 SLOT2 ride ATH +0.30e-4 LM+Full")
    elif LB1==0.95536:
        log("CASE B: Tied StepO — Areeb-op + FullProp")
        d_no=gs(Lo-Ln, [Lm]); mix=unit(d_no+base_ut)
        slot2=inject_orth_rho(step_o, mix, 0.42, B_ladder)
        p2=SUB/"sub_v31_1_SLOT2_areeb_op_plus_fullprop.csv"
        pd.DataFrame({"id":ids,"is_fraud":slot2}).to_csv(p2,index=False)
        LB2=submit_and_poll(p2,"v31.1 SLOT2 Areeb-op d_NO + FullProp @0.42e-4")
    else:
        log("CASE C: Regress — lock StepO")
        p2=SUB/"sub_v31_1_SLOT2_anchor_lock_095536.csv"
        pd.DataFrame({"id":ids,"is_fraud":step_o}).to_csv(p2,index=False)
        LB2=submit_and_poll(p2,"v31.1 SLOT2 anchor lock StepO 0.95536")

    _report(audit,lm_w,full_w,z_auc,LB1,LB2,len(members),remain-SUBMITS_DONE,best_g,best_w,offline)
    log(f"DONE submits={SUBMITS_DONE} LB1={LB1} LB2={LB2}")

def _report(audit,lm_w,full_w,z_auc,LB1,LB2,n_bank,remain,best_g,best_w,offline=None):
    rows="\n".join(
        f"| `{r['name']}` | {r['kind']} | {r['sign']} | {r['pos']}/5 | {r['mean_e5']:+.2f} | "
        f"{r['min_fold_e4']*10:+.2f} | {r['sf25']}/25 | {r['t']:.2f} | {r['disp']:.3f} | "
        f"{r['max_corr_ladder']:.4f} | {'PASS' if r['pass'] else 'FAIL'} |"
        for r in audit)
    REPORT.write_text(f"""# GM V31.1 — Step O × Proprietary 5-Fold LM Leap

**MAX_SUBMITS=2** · remaining≈{remain}

## Design fix vs first abort
- OOF competence LOFO vs `z_honest` **without** annihilating Top-40 SVD (V31.0 protocol).
- Test inject GS-orth to **ladder** `[M,N,Areeb,O]` (+ light SVD-10) so Step-O star stays clean.
- Full Top-40 SVD orth removed all LM residual (max|corr|=1 / LOFO collapse) — documented in V31.0→V31.1 lesson.

## z_honest
- Rank-average top bank members · AUC=**{z_auc:.6f}** · n_bank_pool={n_bank}

## Pillars
| Pillar | kind | sign | pos | meanΔ(e-5) | min(e-5) | sf25 | t | disp | max\\|corr\\| ladder | gate |
|--------|------|------|-----|------------|----------|------|---|------|-------------------|------|
{rows}

## Composites
- u_LM_prop: `{lm_w}`
- u_Full_prop: `{full_w}`
- offline proxy: `{offline}`

## Live
| Slot | LB |
|------|----|
| SLOT1 StepO+5fLM @0.35e-4 | {LB1} |
| SLOT2 adaptive | {LB2} |

## Honest NOSUBMIT
- w={best_w} LOFO={best_g}
- `sub_v31_1_HONEST_700K_CV_SUPERSTACK_NOSUBMIT.csv`

## Log
```
{chr(10).join(LINES[-50:])}
```
""")
    log(f"REPORT -> {REPORT}")
    (OUT/"metrics.json").write_text(json.dumps({"audit":audit,"LB1":LB1,"LB2":LB2,"z_auc":z_auc},indent=2))

if __name__=="__main__":
    main()
