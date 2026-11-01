"""T5 item 4: the two criteria for the H2 margin epsilon_2 (outline v14.3, Noi dung 1, "Bien hieu ung va ly do").

    python eps2_criteria.py inputs.json out.json      # once, on development-data pilot output (NCS machine)
    python eps2_criteria.py --selftest                # synthetic inputs only (what was run in T5)

NOT run on data in T5. The value is locked with the preregistration, before any H2 run.

(i) Operational criterion, NF-ToN-IoT-v3 only, at the operating recall R_op. For each development day with the
    reference context (inputs.operational.days[]): N_att attack flows of the day, and P = precision of the reference
    model at R_op. A loss of delta AUPRC is modelled as a uniform downward shift of the precision-recall curve by
    delta (AUPRC = integral of precision over recall). Then
      extra false alarms per day at R_op : N_att R_op (1/(P - delta) - 1/P)          -> = F_op  gives delta_FA
      extra missed attack flows per day at the same false-alarm count             -> = M_op  gives delta_M
    delta_FA = P - 1 / (1/P + F_op / (N_att R_op));  delta_M = (P - q (1 - P)) / (1 + q), q = (TP - M_op) / FA,
    TP = N_att R_op, FA = TP (1 - P) / P. Per day eps_op_day = min(delta_FA, delta_M) (the smallest AUPRC loss that is
    operationally meaningful by either count); eps_op = median over days. F_op, M_op and R_op are SOC tolerances the NCS
    fixes in inputs.json BEFORE running (they are not derivable from data).
(ii) Noise criterion: 2 x the seed-to-seed SD of the H2 contrast (ThreeBin - comparator, in the H2 endpoint),
    pooled within (situation, comparator) cells: SD_seed = sqrt(mean of within-cell sample variances). Reported per
    comparator; the criterion uses the largest. Blinded as in the Bai 1 variance pilot: only SDs and counts are written,
    never a mean, a sign or a per-cell value.
Locked value: eps2 = max(eps_op, 2 SD_seed). If it is below the outline's provisional 0.03, the output says so: keeping
or lowering 0.03 is then an NCS decision (the outline forbids loosening a margin to gain power).

inputs.json:
  {"operational": {"dataset": "NF-ToN-IoT-v3", "R_op": 0.9, "F_op": 100, "M_op": 50,
                   "days": [{"day": "...", "N_att": 123456, "precision_at_R_op": 0.83}, ...]},
   "seed_runs": [{"situation": "...", "comparator": "CURE", "seed": 42, "contrast": 0.0123}, ...]}
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS2_OUTLINE = 0.03


def delta_fa(P, n_att, r_op, f_op):
    return P - 1.0 / (1.0 / P + f_op / (n_att * r_op))


def delta_miss(P, n_att, r_op, m_op):
    tp = n_att * r_op
    fa = tp * (1 - P) / P
    if m_op >= tp:
        return P   # the whole curve would have to collapse
    q = (tp - m_op) / fa if fa > 0 else np.inf
    return P if not np.isfinite(q) else (P - q * (1 - P)) / (1 + q)


def operational(op):
    if op["dataset"] != "NF-ToN-IoT-v3":
        raise ValueError("the outline fixes the operational criterion on NF-ToN-IoT-v3")
    r_op, f_op, m_op = op["R_op"], op["F_op"], op["M_op"]
    per_day = []
    for d in op["days"]:
        P, n = d["precision_at_R_op"], d["N_att"]
        if not (0 < P <= 1) or n <= 0:
            per_day.append({"day": d["day"], "status": "not_evaluable"})
            continue
        a, b = delta_fa(P, n, r_op, f_op), delta_miss(P, n, r_op, m_op)
        per_day.append({"day": d["day"], "delta_FA": a, "delta_M": b, "eps_op_day": min(a, b)})
    v = [x["eps_op_day"] for x in per_day if "eps_op_day" in x]
    return {"R_op": r_op, "F_op": f_op, "M_op": m_op, "n_days": len(v),
            "eps_op": float(np.median(v)) if v else None, "per_day": per_day}


def noise(runs):
    cells = defaultdict(list)
    for r in runs:
        cells[(r["comparator"], r["situation"])].append(r["contrast"])
    by_comp = defaultdict(list)
    for (comp, _), v in cells.items():
        if len(v) >= 2:
            by_comp[comp].append(np.var(v, ddof=1))
    out = {c: {"n_cells": len(v), "sd_seed": float(np.sqrt(np.mean(v)))} for c, v in by_comp.items()}
    worst = max((x["sd_seed"] for x in out.values()), default=None)
    return {"per_comparator": out, "sd_seed_max": worst, "eps_noise": None if worst is None else 2 * worst}


def compute(inp):
    op, nz = operational(inp["operational"]), noise(inp["seed_runs"])
    vals = [v for v in (op["eps_op"], nz["eps_noise"]) if v is not None]
    eps2 = max(vals) if len(vals) == 2 else None
    return {"criterion_i_operational": op, "criterion_ii_noise": nz, "eps2_locked_candidate": eps2,
            "outline_provisional": EPS2_OUTLINE,
            "flag": None if eps2 is None or eps2 >= EPS2_OUTLINE else
            "computed value below the outline's 0.03: NCS decides (lowering would loosen the margin)"}


def selftest():
    rng = np.random.default_rng(0)
    days = [{"day": f"syn-{i}", "N_att": int(rng.integers(50_000, 500_000)), "precision_at_R_op": float(p)}
            for i, p in enumerate(rng.uniform(0.6, 0.99, 6))]
    runs = [{"situation": f"s{s}", "comparator": c, "seed": k, "contrast": float(rng.normal(0, 0.004))}
            for s in range(8) for c in ("CURE", "CBRS-family", "ECBRS") for k in range(5)]
    out = compute({"operational": {"dataset": "NF-ToN-IoT-v3", "R_op": 0.9, "F_op": 100, "M_op": 100, "days": days},
                   "seed_runs": runs})
    # closed forms: shifting P by delta_FA adds exactly F_op alarms; by delta_M loses exactly M_op flows at fixed FA
    for d in days:
        P, n = d["precision_at_R_op"], d["N_att"]
        a = delta_fa(P, n, 0.9, 100)
        assert abs(n * 0.9 * (1 / (P - a) - 1 / P) - 100) < 1e-6
        b = delta_miss(P, n, 0.9, 100)
        fa = n * 0.9 * (1 - P) / P
        assert abs((n * 0.9 - fa * (P - b) / (1 - P + b)) - 100) < 1e-6
    assert abs(out["criterion_ii_noise"]["sd_seed_max"] - 0.004) < 0.0015
    assert "mean" not in json.dumps(out["criterion_ii_noise"])
    print("selftest ok", json.dumps({k: out[k] for k in ("eps2_locked_candidate", "flag")}))


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        raw = Path(sys.argv[1]).read_bytes()
        res = compute(json.loads(raw))
        res["inputs_sha256"] = hashlib.sha256(raw).hexdigest()
        res["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("eps2_locked_candidate", "flag")}))
