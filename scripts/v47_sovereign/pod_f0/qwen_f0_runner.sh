#!/usr/bin/env bash
# V47 Fold-0 Qwen2.5-0.5B digit GenBayes — wraps proven V43 trainer, renames to sovereign paths.
# NO_SUBMIT. USE_COMPILE=0. K=4 antithetic (2 fwd + 2 rev).
set -euo pipefail
ROOT="${1:-/workspace/v47}"
DATA="${2:-/workspace/data}"
OUT="${3:-/workspace/preds/v47_pod_f0}"
SRC="${ROOT}/scripts/v43_pod_qwen_perfield_llr_and_tabicl"
mkdir -p "$OUT/qwen"

export TORCH_COMPILE_DISABLE=1
export V43_USE_COMPILE=0
export V43_LAUNCH=1
export V43_FOLD=0
export V43_EPOCHS="${V43_EPOCHS:-4}"
export V43_K_PAIRS=2
export V43_MODEL="${V43_MODEL:-Qwen/Qwen2.5-0.5B}"
export V43_SKIP_TABICL=1
export V43_CKPT_EVERY="${V43_CKPT_EVERY:-500}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TRANSFORMERS_CACHE="$HF_HOME"
export OUT_DIR="$OUT"

cd "$SRC"
python3 qwen_digit_genbayes_24llr.py --data "$DATA" --out "$OUT/qwen" --fold 0

# Canonical V47 names + pack oof to full train length when va-sized
python3 - <<'PY'
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

out = Path(os.environ["OUT_DIR"])
q = out / "qwen"
data = Path(os.environ.get("DATA", "/workspace/data"))
y = pd.read_csv(data / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
_, va_ix = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))[0]

def pick(*globs):
    for g in globs:
        ps = sorted(q.rglob(g))
        if ps:
            return ps[0]
    return None

pairs = [
    (("oof_f0_seq_llr.npy", "*seq*oof*.npy"), "qwen_f0_seq_llr_oof.npy", True),
    (("test_f0_seq_llr.npy", "*seq*test*.npy"), "qwen_f0_seq_llr_test.npy", False),
    (("oof_f0_field24_llr.npy",), "qwen_f0_field24_llr_oof.npy", True),
    (("test_f0_field24_llr.npy",), "qwen_f0_field24_llr_test.npy", False),
    (("oof_f0_field24_std.npy",), "qwen_f0_field24_std_oof.npy", True),
    (("test_f0_field24_std.npy",), "qwen_f0_field24_std_test.npy", False),
]
for globs, name, pack in pairs:
    p = pick(*globs)
    if p is None:
        print(f"MISSING {name} globs={globs}")
        continue
    arr = np.load(p)
    if pack and arr.shape[0] == len(va_ix):
        full = np.zeros((len(y),) + arr.shape[1:], dtype=arr.dtype)
        full[va_ix] = arr
        arr = full
    np.save(out / name, arr)
    print(f"wrote {name} shape={arr.shape} from {p.name}")
PY
