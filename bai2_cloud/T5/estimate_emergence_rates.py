"""D3: emergence (appearance / re-appearance) rate rho_e and per-row share r_e of attack families, from LABELS ONLY.
CPU, for the NCS to run locally. Fixed and hashed (SHA256SUMS) before it is run; it writes its own SHA-256 into the
output.

    # 1) on the NCS machine, the LOCKED Bai 1 script (unchanged) writes per-day family counts from the cache labels:
    #      python protocol_code/replay/count_days.py            -> protocol_code/replay/out/day_counts.json
    # 2) then
    python estimate_emergence_rates.py day_counts.json emergence_rates.json
    python estimate_emergence_rates.py --selftest               # synthetic day counts (what was run in T5)

Input: day_counts.json of count_days.py: per development dataset, the valid days in time order with
{day, flows, attack_flows, families: {name: flows}}. Only label and family counts are used; no feature is read.

Definitions (design_h2.md section 8; they must match the q* derivation of T2 -> [VERIFY against T2 REPORT]):
  * Day sequence = the valid days of a dataset in time order (the order the replay uses). A family is PRESENT on a day
    if it has >= 1 flow.
  * EMERGENCE EVENT of family f on day t: f present on t and absent on each of the previous D_ABSENT = 3 valid days
    (the outline's recurrence rule). The first D_ABSENT days are a burn-in (left-censored, no event can be scored).
    An event is "new" if f was never present before t, "re-emergence" otherwise.
  * rho_e = events / family-days at risk, where family f is at risk on day t (t > burn-in) if it was absent on the
    previous D_ABSENT days. Poisson MLE with an exact (Garwood) 95 % interval. Reported pooled, per dataset, and for
    rare events only (f < 1 % of the day's attack flows on t).
  * r_e = share of f among ALL flows of the emergence day (labels are queried uniformly from all rows, so this is the
    per-trusted-row probability r of Proposition 1). Reported as median and IQR over events.
  * lambda = trusted policy rows per day = (1 - eta) b = 1,600 (Bai 1 settings), written for the q* function.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

D_ABSENT = 3
RARE_THR = 0.01
B, ETA = 2000, 0.2


def garwood(k, t, alpha=0.05):
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo / t, hi / t


def events_for(days):
    fams = sorted({f for d in days for f, c in d["families"].items() if c > 0})
    pres = np.array([[d["families"].get(f, 0) > 0 for f in fams] for d in days], bool)   # (T, F)
    ev, at_risk = [], 0
    for t in range(D_ABSENT, len(days)):
        absent_before = ~pres[t - D_ABSENT:t].any(axis=0)
        at_risk += int(absent_before.sum())
        for j in np.flatnonzero(absent_before & pres[t]):
            d = days[t]
            c = d["families"][fams[j]]
            ev.append({"day": d["day"], "family": fams[j], "kind": "re-emergence" if pres[:t, j].any() else "new",
                       "r_e": c / d["flows"], "share_of_attack_flows": c / d["attack_flows"],
                       "rare": c / d["attack_flows"] < RARE_THR})
    return ev, at_risk


def summarize(ev, at_risk):
    k = len(ev)
    out = {"events": k, "family_days_at_risk": at_risk}
    if at_risk:
        lo, hi = garwood(k, at_risk)
        out.update(rho_e=k / at_risk, rho_e_ci95=[lo, hi])
    if k:
        r = np.array([e["r_e"] for e in ev])
        out.update(r_e_median=float(np.median(r)), r_e_iqr=[float(np.quantile(r, .25)), float(np.quantile(r, .75))])
    return out


def compute(counts):
    res, all_ev, all_risk = {"per_dataset": {}}, [], 0
    for ds, v in counts.items():
        days = v["days"]
        ev, risk = events_for(days)
        res["per_dataset"][ds] = summarize(ev, risk) | {"n_valid_days": len(days), "event_list": ev}
        all_ev += ev
        all_risk += risk
    res["pooled"] = summarize(all_ev, all_risk)
    res["pooled_rare_events_only"] = summarize([e for e in all_ev if e["rare"]], all_risk)
    res["lambda_trusted_rows_per_day"] = round((1 - ETA) * B)
    res["definitions"] = {"D_ABSENT": D_ABSENT, "RARE_THR": RARE_THR}
    return res


def selftest():
    # 8 synthetic days, two datasets; family "a" absent on days 2-4 and back on 5; "c" new on day 6
    def day(i, fam):
        return {"day": f"d{i}", "flows": 1000, "attack_flows": max(1, sum(fam.values())), "families": fam}
    ds = [day(0, {"a": 50, "b": 400}), day(1, {"a": 40, "b": 300}), day(2, {"b": 500}), day(3, {"b": 500}),
          day(4, {"b": 500}), day(5, {"a": 3, "b": 500}), day(6, {"b": 500, "c": 2}), day(7, {"b": 500})]
    res = compute({"SYN": {"days": ds}})
    ev = res["per_dataset"]["SYN"]["event_list"]
    assert [(e["day"], e["family"], e["kind"]) for e in ev] == [("d5", "a", "re-emergence"), ("d6", "c", "new")], ev
    # at risk (absent on the 3 previous days): t=3 c; t=4 c; t=5 a, c; t=6 c; t=7 none -> 5 family-days
    assert res["per_dataset"]["SYN"]["family_days_at_risk"] == 5
    assert ev[0]["rare"] and abs(ev[0]["r_e"] - 0.003) < 1e-12
    lo, hi = garwood(2, 10)
    assert lo < 0.2 < hi
    print("selftest ok", json.dumps(res["pooled"]))


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        raw = Path(sys.argv[1]).read_bytes()
        res = compute(json.loads(raw))
        res["input_sha256"] = hashlib.sha256(raw).hexdigest()
        res["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res["pooled"]))
