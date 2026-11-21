#!/usr/bin/env python3
"""Local recon: find the public 0.95369 blocker kernel/dataset. STRICT NO_SUBMIT=True.

Does NOT touch any RunPod / A40 training pods.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

NO_SUBMIT = True  # HARD LOCK
VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
OUT = VOL / "external" / "nov12_095369_recon"
LOG = VOL / "logs" / "overnight"
ART = Path(os.environ.get("GM_RECON_ROOT", "/tmp/gm_recon095369"))
for p in (OUT, LOG, ART):
    p.mkdir(parents=True, exist_ok=True)

COMP = "playground-series-s6e11"
T0 = time.time()
AUTHORS = [
    "goodpjw2008",
    "areebbb",
    "amanatar",
    "samanyu1808",
    "megayak",
    "lucifer19",
    "sometimessubodh",
    "smartmanoj",
    "nibeditabal",
    "alexchilton",
    "moranzavdi",
]

# ---- DNS patch (local resolver often blocks api.kaggle.com) ----
_FALLBACK = {"api.kaggle.com": "34.54.168.202", "www.kaggle.com": "35.244.233.98"}
_CACHE = dict(_FALLBACK)
_orig = socket.getaddrinfo


def _doh(name: str):
    try:
        out = subprocess.check_output(
            [
                "curl", "-sS", "-m", "12", "--resolve", "dns.google:443:8.8.8.8",
                f"https://dns.google/resolve?name={name}&type=A",
            ],
            text=True,
        )
        for a in json.loads(out).get("Answer", []):
            if a.get("type") == 1:
                return a["data"]
    except Exception:
        return None
    return None


def _gai(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and host in ("api.kaggle.com", "www.kaggle.com"):
        if host not in _CACHE or True:
            ip = _doh(host) or _FALLBACK.get(host)
            if ip:
                _CACHE[host] = ip
                print(f"DNS_PATCH {host} -> {ip}", flush=True)
                host = ip
    return _orig(host, port, family, type, proto, flags)


socket.getaddrinfo = _gai  # type: ignore

sys.path.insert(0, str(VOL / ".venv_forensics/lib"))
for sp in (VOL / ".venv_forensics/lib").glob("python*/site-packages"):
    sys.path.insert(0, str(sp))


def log(m: str):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def setup_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _as_dict(x):
    if isinstance(x, dict):
        return x
    d = {}
    for k in (
        "ref", "title", "author", "lastRunTime", "totalVotes", "score",
        "url", "kernelId", "id", "subtitle", "lastUpdated", "usabilityRating",
        "totalDownloads", "datasetSize", "fileSize",
    ):
        if hasattr(x, k):
            d[k] = getattr(x, k)
    # pydantic / new API
    if hasattr(x, "to_dict"):
        try:
            d.update(x.to_dict())
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        for k, v in vars(x).items():
            if not k.startswith("_") and k not in d:
                d[k] = v
    return d


def parse_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    s = str(v).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def list_kernels(api, **kw):
    try:
        items = api.kernels_list(**kw)
    except TypeError:
        # older signature
        items = api.kernels_list(page=kw.get("page", 1), page_size=kw.get("page_size", 50),
                                 search=kw.get("search"), competition=kw.get("competition"),
                                 sort_by=kw.get("sort_by"))
    return [_as_dict(x) for x in (items or [])]


def list_datasets(api, **kw):
    try:
        items = api.dataset_list(**kw)
    except TypeError:
        items = api.dataset_list(search=kw.get("search"), page=kw.get("page", 1),
                                 max_size=kw.get("max_size"), file_type=kw.get("file_type"),
                                 license_name=kw.get("license_name"), tag_ids=kw.get("tag_ids"),
                                 sort_by=kw.get("sort_by"))
    return [_as_dict(x) for x in (items or [])]


def safe_ref(d):
    return d.get("ref") or d.get("id") or ""


def pull_kernel(api, ref: str) -> Path | None:
    ref = ref.strip("/")
    if not ref or "/" not in ref:
        return None
    owner, slug = ref.split("/", 1)
    dest = OUT / f"{owner}__{slug}_code"
    if dest.exists() and any(dest.glob("*.ipynb")):
        log(f"  kernel exists: {dest.name}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_pull(ref, path=str(dest), metadata=True)
        log(f"  pulled kernel {ref} -> {dest.name}")
        return dest
    except Exception as e:
        log(f"  kernel pull FAIL {ref}: {e}")
        return None


def pull_dataset(api, ref: str) -> Path | None:
    ref = ref.strip("/")
    if not ref or "/" not in ref:
        return None
    owner, slug = ref.split("/", 1)
    dest = OUT / "datasets" / f"{owner}__{slug}"
    if dest.exists() and any(dest.iterdir()):
        log(f"  dataset exists: {dest.name}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=False)
        log(f"  pulled dataset {ref}")
        return dest
    except Exception as e:
        log(f"  dataset pull FAIL {ref}: {e}")
        return None


FORMULA_RE = re.compile(
    r"(score\s*=|submission|logit|rankdata|0\.953|lm7|L7|megaLM|S10|S9|delta|"
    r"is_fraud|np\.load|pd\.read_csv|dataset_sources|INPUT)",
    re.I,
)
SCORE_RE = re.compile(r"0\.953[0-9]{2}")
LM7_RE = re.compile(r"lm7_test_f0|lm7_oof_f0|compact-lm-seed7|L7", re.I)


def inspect_notebook(path: Path) -> dict:
    info = {
        "path": str(path),
        "formulas": [],
        "scores_mentioned": [],
        "npy_csv": [],
        "uses_lm7_f0_only": None,
        "dataset_mentions": [],
        "cells_interesting": [],
    }
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        info["error"] = str(e)
        return info
    texts = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        texts.append(src)
        if FORMULA_RE.search(src):
            # keep short snippets
            for line in src.splitlines():
                if re.search(r"score\s*=|0\.6\s*\*|0\.12\s*\*|0\.06\s*\*|logit|rankdata", line, re.I):
                    info["formulas"].append(line.strip()[:240])
                if SCORE_RE.search(line):
                    info["scores_mentioned"] += SCORE_RE.findall(line)
                if re.search(r"\.npy|\.csv|kaggle/input", line, re.I):
                    info["npy_csv"].append(line.strip()[:200])
                if re.search(r"datasets?/|goodpjw|megayak|amanatar|areebb|samanyu|lucifer", line, re.I):
                    info["dataset_mentions"].append(line.strip()[:200])
            if LM7_RE.search(src) or "score =" in src or "0.953" in src:
                info["cells_interesting"].append(src[:900])
    blob = "\n".join(texts)
    info["scores_mentioned"] = sorted(set(info["scores_mentioned"]))
    info["uses_lm7_f0_only"] = bool(re.search(r"lm7_test_f0", blob, re.I)) and not bool(
        re.search(r"lm7_test_f[1-4]|lm7_oof_f[1-4]|L7_test_5f|l7_5f", blob, re.I)
    )
    info["mentions_5f_l7"] = bool(re.search(r"lm7_.*f[1-4]|L7_5f|5.?fold.*lm7|compact.?lm.*fold", blob, re.I))
    # dedupe formulas
    seen = set()
    uniq = []
    for f in info["formulas"]:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    info["formulas"] = uniq[:40]
    info["npy_csv"] = list(dict.fromkeys(info["npy_csv"]))[:40]
    return info


def cli_list(cmd: list[str]) -> str:
    env = os.environ.copy()
    kaggle = VOL / ".venv_forensics/bin/kaggle"
    full = [str(kaggle)] + cmd
    try:
        return subprocess.check_output(full, text=True, stderr=subprocess.STDOUT, timeout=90)
    except Exception as e:
        return f"CLI_FAIL {e}"


def main():
    assert NO_SUBMIT is True
    log("======== GM RECON 0.95369 BLOCKER (NO_SUBMIT) ========")
    log("HARD RULE: do not touch A40 pods (rsoju37zhpzaae / smugcqcy2oqzeg)")
    api = setup_api()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

    catalogs = {}

    # 1) API lists
    queries = [
        ("kernels_dateRun", dict(competition=COMP, sort_by="dateRun", page_size=50)),
        ("kernels_voteCount", dict(competition=COMP, sort_by="voteCount", page_size=50)),
        ("kernels_search_09536", dict(search="0.9536", page_size=50)),
        ("kernels_search_09535", dict(search="0.9535", page_size=50)),
        ("kernels_search_s6e11", dict(search="s6e11", page_size=50)),
        ("kernels_comp_hotness", dict(competition=COMP, sort_by="hotness", page_size=50)),
    ]
    for name, kw in queries:
        try:
            catalogs[name] = list_kernels(api, **kw)
            log(f"{name}: {len(catalogs[name])} kernels")
        except Exception as e:
            catalogs[name] = []
            log(f"{name} FAIL: {e}")

    # author filters via search
    for a in AUTHORS:
        key = f"kernels_author_{a}"
        try:
            catalogs[key] = list_kernels(api, search=a, competition=COMP, page_size=20, sort_by="dateRun")
            log(f"{key}: {len(catalogs[key])}")
        except Exception as e:
            catalogs[key] = []
            log(f"{key} FAIL: {e}")

    for name, kw in [
        ("datasets_s6e11", dict(search="s6e11", page_size=50)),
        ("datasets_09536", dict(search="0.9536", page_size=30)),
        ("datasets_compact_lm", dict(search="s6e11 compact lm", page_size=30)),
        ("datasets_strike", dict(search="s6e11 strike", page_size=30)),
    ]:
        try:
            catalogs[name] = list_datasets(api, **kw)
            log(f"{name}: {len(catalogs[name])} datasets")
        except Exception as e:
            catalogs[name] = []
            log(f"{name} FAIL: {e}")

    # CLI mirrors (best-effort)
    cli_dumps = {}
    for label, args in [
        ("cli_kernels_date", ["kernels", "list", "-c", COMP, "--sort-by", "dateRun", "--page-size", "50"]),
        ("cli_kernels_vote", ["kernels", "list", "-c", COMP, "--sort-by", "voteCount", "--page-size", "50"]),
        ("cli_kernels_09536", ["kernels", "list", "-s", "0.9536", "--page-size", "30"]),
        ("cli_kernels_09535", ["kernels", "list", "-s", "0.9535", "--page-size", "30"]),
        ("cli_kernels_s6e11", ["kernels", "list", "-s", "s6e11", "--page-size", "30"]),
        ("cli_datasets_s6e11", ["datasets", "list", "-s", "s6e11", "--page-size", "30"]),
        ("cli_datasets_09536", ["datasets", "list", "-s", "0.9536", "--page-size", "20"]),
    ]:
        cli_dumps[label] = cli_list(args)
        log(f"{label}: {len(cli_dumps[label])} chars")

    # Merge unique kernel refs
    all_k = {}
    for lst in catalogs.values():
        if not isinstance(lst, list):
            continue
        for d in lst:
            if not isinstance(d, dict):
                continue
            ref = safe_ref(d)
            if not ref or "/" not in str(ref):
                continue
            # skip non-kernel looking refs for dataset catalogs
            if "datasets_" in str(lst):
                pass
            all_k[ref] = {**all_k.get(ref, {}), **d}

    # Separate datasets
    all_ds = {}
    for key in catalogs:
        if not key.startswith("datasets_"):
            continue
        for d in catalogs[key]:
            ref = safe_ref(d)
            if ref and "/" in str(ref):
                all_ds[ref] = {**all_ds.get(ref, {}), **d}

    # Recent filter (48h) — also keep high-score title hits
    recent_k = []
    score_hits = []
    for ref, d in all_k.items():
        title = str(d.get("title") or "")
        dt = parse_dt(d.get("lastRunTime") or d.get("lastUpdated") or d.get("updated"))
        is_recent = dt is not None and dt >= cutoff
        has_score = bool(SCORE_RE.search(title) or SCORE_RE.search(ref))
        if "0.9536" in title or "0.95369" in title or "95369" in title.replace(".", ""):
            score_hits.append((ref, d, dt))
        if is_recent or has_score or any(a in ref for a in AUTHORS):
            recent_k.append((ref, d, dt, is_recent, has_score))

    # Prefer pull: recent OR score in title OR top authors updated
    to_pull = []
    for ref, d, dt, is_recent, has_score in recent_k:
        if is_recent or has_score or ("0.953" in str(d.get("title", ""))):
            to_pull.append(ref)
    # always try known leaders' newest
    for a in AUTHORS[:6]:
        for ref, d in all_k.items():
            if ref.startswith(a + "/") and ref not in to_pull:
                dt = parse_dt(d.get("lastRunTime") or d.get("lastUpdated"))
                if dt and dt >= cutoff:
                    to_pull.append(ref)

    to_pull = list(dict.fromkeys(to_pull))[:25]
    log(f"kernels to pull: {len(to_pull)}")

    pulled = []
    inspections = []
    for ref in to_pull:
        dest = pull_kernel(api, ref)
        if not dest:
            continue
        pulled.append(ref)
        for nb in dest.rglob("*.ipynb"):
            if nb.name.startswith("._"):
                continue
            info = inspect_notebook(nb)
            info["kernel_ref"] = ref
            inspections.append(info)
            log(f"  inspected {nb.name}: scores={info['scores_mentioned']} lm7_f0_only={info['uses_lm7_f0_only']} formulas={len(info['formulas'])}")

    # datasets: pull recent / score-named
    ds_pull = []
    for ref, d in all_ds.items():
        title = str(d.get("title") or ref)
        dt = parse_dt(d.get("lastUpdated") or d.get("updated"))
        if (dt and dt >= cutoff) or "0.953" in title or "compact-lm" in title.lower() or "strike" in title.lower():
            ds_pull.append(ref)
    ds_pull = list(dict.fromkeys(ds_pull))[:15]
    log(f"datasets to pull: {len(ds_pull)}")
    ds_pulled = []
    ds_files = {}
    for ref in ds_pull:
        dest = pull_dataset(api, ref)
        if not dest:
            continue
        ds_pulled.append(ref)
        files = []
        for f in dest.rglob("*"):
            if f.is_file() and not f.name.startswith("._"):
                files.append(f"{f.relative_to(dest)} ({f.stat().st_size})")
        ds_files[ref] = files[:80]
        log(f"  dataset {ref}: {len(files)} files")

    # Heuristic: which inspection looks like 0.95369 ladder
    blockers = []
    for info in inspections:
        scores = info.get("scores_mentioned") or []
        if any(s >= "0.9536" for s in scores) or any("0.9536" in c for c in info.get("cells_interesting", [])):
            blockers.append(info)
        if info.get("formulas") and ("0.953" in str(info.get("scores_mentioned")) or info.get("uses_lm7_f0_only")):
            if info not in blockers:
                blockers.append(info)

    # Serialize catalogs (trim)
    def trim_k(d):
        return {
            "ref": safe_ref(d),
            "title": d.get("title"),
            "author": d.get("author") or (safe_ref(d).split("/")[0] if "/" in safe_ref(d) else None),
            "lastRunTime": str(d.get("lastRunTime") or d.get("lastUpdated") or ""),
            "totalVotes": d.get("totalVotes") or d.get("total_votes"),
        }

    summary = {
        "NO_SUBMIT": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_48h": cutoff.isoformat(),
        "n_kernel_refs_seen": len(all_k),
        "n_dataset_refs_seen": len(all_ds),
        "pulled_kernels": pulled,
        "pulled_datasets": ds_pulled,
        "score_title_hits": [(r, trim_k(d), str(dt)) for r, d, dt in score_hits[:30]],
        "inspections": inspections,
        "ds_files": ds_files,
        "catalog_kernels_dateRun": [trim_k(d) for d in catalogs.get("kernels_dateRun", [])[:40]],
        "catalog_kernels_vote": [trim_k(d) for d in catalogs.get("kernels_voteCount", [])[:40]],
        "elapsed_s": time.time() - T0,
    }
    (ART / "recon_095369.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "recon_095369.json").write_text(json.dumps(summary, indent=2, default=str))
    (LOG / "recon_095369_cli.txt").write_text(
        "\n\n".join(f"===== {k} =====\n{v}" for k, v in cli_dumps.items())
    )

    # Markdown report
    md = []
    md.append("# GM Recon — Public `0.95369` Blocker (`NO_SUBMIT=True`)\n\n")
    md.append(f"Generated `{summary['generated_at']}` · elapsed `{summary['elapsed_s']:.0f}s`.\n\n")
    md.append("**Hard rule:** A40 training pods left untouched (`rsoju37zhpzaae` / `smugcqcy2oqzeg`).\n\n")
    md.append("## Catalog — newest competition kernels (dateRun)\n\n")
    md.append("| ref | title | votes | lastRun |\n|---|---|---:|---|\n")
    for d in summary["catalog_kernels_dateRun"]:
        md.append(f"| `{d['ref']}` | {d['title']} | {d['totalVotes']} | {d['lastRunTime']} |\n")
    md.append("\n## Catalog — top voted\n\n")
    md.append("| ref | title | votes | lastRun |\n|---|---|---:|---|\n")
    for d in summary["catalog_kernels_vote"]:
        md.append(f"| `{d['ref']}` | {d['title']} | {d['totalVotes']} | {d['lastRunTime']} |\n")
    md.append("\n## Score-title hits (`0.9536*`)\n\n")
    if score_hits:
        for ref, d, dt in score_hits[:20]:
            md.append(f"- `{ref}` — **{d.get('title')}** (last `{dt}`)\n")
    else:
        md.append("_No kernel titles literally containing 0.9536x in API results._\n")
    md.append(f"\n## Pulled kernels ({len(pulled)})\n\n")
    for r in pulled:
        md.append(f"- `{r}`\n")
    md.append(f"\n## Pulled datasets ({len(ds_pulled)})\n\n")
    for r in ds_pulled:
        md.append(f"- `{r}`\n")
        for f in ds_files.get(r, [])[:15]:
            md.append(f"  - `{f}`\n")
    md.append("\n## Notebook inspections — formulas & L7 dependency\n\n")
    for info in inspections:
        md.append(f"### `{info.get('kernel_ref')}` — `{Path(info['path']).name}`\n\n")
        md.append(f"- Scores mentioned: `{info.get('scores_mentioned')}`\n")
        md.append(f"- Uses **lm7_test_f0 only** (1-fold L7): `{info.get('uses_lm7_f0_only')}`\n")
        md.append(f"- Mentions 5-fold L7: `{info.get('mentions_5f_l7')}`\n")
        if info.get("formulas"):
            md.append("- Formulas / ladder lines:\n")
            for f in info["formulas"][:15]:
                md.append(f"  - `{f}`\n")
        if info.get("npy_csv"):
            md.append("- Inputs:\n")
            for f in info["npy_csv"][:12]:
                md.append(f"  - `{f}`\n")
        md.append("\n")
    md.append("## Verdict / working hypothesis\n\n")
    if any(info.get("uses_lm7_f0_only") for info in inspections):
        md.append(
            "- At least one pulled ladder still depends on **1-fold `lm7_test_f0.npy`** "
            "(Fold-0-only L7). That matches our structural bottleneck motivating `L7_5F_K10`.\n"
        )
    if any(info.get("mentions_5f_l7") for info in inspections):
        md.append("- Found mention of multi-fold L7 — check those notebooks first for a true 0.95369 leap.\n")
    else:
        md.append(
            "- No pulled notebook clearly documents a **5-fold L7** replacement; the public pack "
            "likely still rides 1-fold LM7 + Strike ladder arithmetic (or a new unpublished GBDT/LM dataset).\n"
        )
    high = [i for i in inspections if any(s.startswith("0.9536") for s in (i.get("scores_mentioned") or []))]
    if high:
        md.append("- Notebooks mentioning **0.9536x**:\n")
        for i in high:
            md.append(f"  - `{i.get('kernel_ref')}` scores={i.get('scores_mentioned')}\n")
    else:
        md.append(
            "- **No pulled notebook text yet cites `0.95369` explicitly** — the cluster may be "
            "from private blends of the same public ingredients, or a dataset/kernel landed under a "
            "title without the score string. Re-check dateRun list + dataset files above.\n"
        )
    md.append(f"\nArtifacts: `{OUT}/recon_095369.json`, CLI dump `{LOG}/recon_095369_cli.txt`.\n")
    report = LOG / "GM_RECON_095369_BLOCKER.md"
    report.write_text("".join(md))
    log(f"REPORT -> {report}")
    log("======== RECON DONE (NO_SUBMIT) ========")
    print(json.dumps({
        "pulled_kernels": pulled,
        "pulled_datasets": ds_pulled,
        "n_inspections": len(inspections),
        "report": str(report),
        "NO_SUBMIT": True,
    }, indent=2))


if __name__ == "__main__":
    main()
