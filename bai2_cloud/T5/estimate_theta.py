"""D2: theta = marginal loss of AUPRC_w per 1,000 rows removed from the recent (FIFO) compartment. CPU only (locked
LightGBM), for the NCS to run locally on the development cache. Fixed and hashed (SHA256SUMS) before it is run; it
writes its own SHA-256 and the hashes of its inputs into the output. Never touches ThreeBin or any H2 comparator, so
it reveals nothing about the H2 contrast.

    python estimate_theta.py --bundle <bai2_handoff> --cache-dir <cache> --lgbm-params lgbm_hparams_locked.json \
                             --out theta.json
    python estimate_theta.py --selftest          # synthetic records only (what was run in T5)

Design (design_h2.md section 8; the definition must match the q* derivation of T2 -> [VERIFY against T2 REPORT]):
  * Proposition 1(c): ThreeBin pays for the per-group floor q_min by shrinking the recent compartment by up to
    G q_min rows. theta prices such rows: theta = d AUPRC_w / d k_R (AUPRC_w lost per 1,000 rows removed).
  * Runs: FIFO with k_R in K_GRID (k = 2,000 minus 0 to 800 rows), locked LightGBM, b = 2,000, eta = 0.2, tau = 0,
    A_FP = 1,000, isolation filter with zero tolerated drops, seeds SEEDS; scenario 0 of the feature and prior kinds of
    every root day (the Bai 1 primary chains; recurring chains are excluded because H2 is evaluated on them).
  * Per root day: AUPRC_w averaged over evaluable days, chains and seeds for each k_R; OLS slope over k_R.
  * theta_hat = HL estimate of the root-day slopes, with the exact 95 % interval (signed-rank inversion, corrected
    index; design_h2.md section 3). Not-evaluable days (one-class context, no attack in the sample) are counted.
"""
import argparse
import hashlib
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

K = 2000
K_GRID = (2000, 1800, 1600, 1400, 1200)
SEEDS = (42, 123)
B, ETA, TAU, A_FP = 2000, 0.2, 0, 1000
KINDS = ("feature", "prior")


def signed_rank_null(n):
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1
    for r in range(1, n + 1):
        new = dist.copy()
        new[r:] += dist[:-r]
        dist = new
    return dist / dist.sum()


def hl_with_ci(x, alpha=0.025):
    """HL estimate and exact two-sided (1 - 2 alpha) interval; bounds None if n is too small."""
    x = np.asarray(x, float)
    w = np.sort([(a + b) / 2 for a, b in combinations_with_replacement(x, 2)])
    c = int(np.searchsorted(np.cumsum(signed_rank_null(len(x))), alpha, side="right"))
    if c == 0:
        return float(np.median(w)), None, None
    return float(np.median(w)), float(w[c - 1]), float(w[len(w) - c])


def slope(ks, v):
    ks, v = np.asarray(ks, float) / 1000, np.asarray(v, float)
    return float(np.polyfit(ks, v, 1)[0])


def estimate(records):
    """records: dicts with root (dataset|day), k_R, auprc_w (None if not evaluable)."""
    per_root = {}
    ne = sum(r["auprc_w"] is None for r in records)
    for r in records:
        if r["auprc_w"] is not None:
            per_root.setdefault(r["root"], {}).setdefault(r["k_R"], []).append(r["auprc_w"])
    slopes = {}
    for root, byk in per_root.items():
        ks = sorted(byk)
        if len(ks) >= 3:
            slopes[root] = slope(ks, [np.mean(byk[k]) for k in ks])
    hl, lo, hi = hl_with_ci(list(slopes.values())) if slopes else (None, None, None)
    return {"theta_hat_per_1000_rows": hl, "theta_ci95": None if lo is None else [lo, hi],
            "n_root_days": len(slopes), "root_day_slopes": slopes, "n_not_evaluable_days": ne,
            "k_grid": K_GRID, "seeds": SEEDS}


def run(args):
    h = Path(args.bundle) / "protocol_code"
    sys.path.insert(0, str(h / "replay"))
    sys.path.insert(0, str(h))
    from data import DEV_DATASETS, DayStream          # locked harness, imported read-only
    from models import LightGBMModel
    from policies import FIFO
    from replay import run_replay
    from scenario_stream import ScenarioStream
    man = json.loads((h / "scenario_manifest.json").read_text(encoding="utf-8"))
    params = json.loads(Path(args.lgbm_params).read_text(encoding="utf-8"))
    records = []
    for ds in DEV_DATASETS:
        root = DayStream(ds, cache_dir=args.cache_dir)
        pools = {p["day"]: p["sha256"] for p in man["datasets"][ds]["pools"]}
        for scn in man["scenarios"]:
            if scn["dataset"] != ds or scn["kind"] not in KINDS or not scn["id"].endswith("|0"):
                continue
            stream = ScenarioStream(root, scn, expected_pool_sha256=pools[scn["root_day"]])
            for k_r in K_GRID:
                for seed in SEEDS:
                    recs = run_replay(stream, FIFO(k=k_r, seed=seed), LightGBMModel(params, seed=seed), b=B, tau=TAU,
                                      a_fp=A_FP, eta=ETA, first_day=0, n_days=len(scn["days"]),
                                      isolation="filter", max_filter_drops=0, seed=seed)
                    for r in recs:
                        ok = r["auprc_w"] is not None and r["context_ok"]
                        records.append({"root": f"{ds}|{scn['root_day']}", "scenario": scn["id"], "k_R": k_r,
                                        "seed": seed, "day": r["day"], "auprc_w": r["auprc_w"] if ok else None})
            print(scn["id"], "done", flush=True)
    out = estimate(records)
    out["records"] = records
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["manifest_sha256"] = hashlib.sha256((h / "scenario_manifest.json").read_bytes()).hexdigest()
    out["lgbm_params_sha256"] = hashlib.sha256(Path(args.lgbm_params).read_bytes()).hexdigest()
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("theta_hat_per_1000_rows", "theta_ci95", "n_root_days")}))


def selftest():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(12):                       # true theta = 0.01 per 1,000 rows
        base = rng.uniform(0.5, 0.9)
        for k in K_GRID:
            for s in SEEDS:
                recs.append({"root": f"r{i}", "k_R": k, "auprc_w": base + 0.01 * k / 1000 + rng.normal(0, 0.001)})
    recs.append({"root": "r0", "k_R": 2000, "auprc_w": None})
    out = estimate(recs)
    assert abs(out["theta_hat_per_1000_rows"] - 0.01) < 0.002 and out["n_not_evaluable_days"] == 1
    lo, hi = out["theta_ci95"]
    assert lo <= out["theta_hat_per_1000_rows"] <= hi
    assert hl_with_ci([1.0, 2.0, 3.0])[1] is None          # n = 3: no exact 95 % interval
    print("selftest ok", json.dumps({k: out[k] for k in ("theta_hat_per_1000_rows", "theta_ci95")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bundle")
    ap.add_argument("--cache-dir")
    ap.add_argument("--lgbm-params")
    ap.add_argument("--out", default="theta.json")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
