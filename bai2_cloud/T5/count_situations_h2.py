"""T5 item 1: count H2 situations per root day from the scenario manifest ONLY (no data rows are read).

    python count_situations_h2.py <path/to/protocol_code/scenario_manifest.json> [out.json]

Definitions (outline v14.3, Noi dung 1): a family is RARE on a target day if it holds < 1 % of that day's attack flows;
it is RECURRING if it was absent for at least d = 3 days before. The independent unit is the ROOT DAY (outline;
Bai 1 Section 3.3): situations generated from the same root day share its flows.

What the manifest can and cannot tell:
  * rare families per root day: datasets.<ds>.pools[].rare_family_split (exact pool counts);
  * recurring scenarios: scenarios[kind == recurring] with family, absent_days, family_flows and share;
  * per-family flow counts only for families with >= 1,000 flows on the 10 root days that have recurring scenarios.
    Families of the other 6 root days are NOT in the manifest (count_days.py in protocol_code/replay lists them; its
    output is not in the bundle).
Expected labelled rows of a family per day = (1 - eta) * b * (family flows in POLICY pool / POLICY pool size), with the
POLICY pool share of a non-rare family taken equal to its root-day share (pools are stratified by family) and the
exact split for rare families. b = 2,000 and eta = 0.2 are the Bai 1 settings (manuscript Table 2).
Injected rare-recurring situations (design_h2.md option B): within-pool down-weighting of family f to a share s of
the attack flows; the expected rows per day are computed for s in S_GRID.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

B, ETA = 2000, 0.2                    # Bai 1 Table 2
N_POLICY_LABELS = round((1 - ETA) * B)
EVAL_DAY_SIZE = 200_000               # manifest design.EVAL_DAY_SIZE (checked below)
N_EVAL = 10_000                       # Bai 1 evaluation sample per day
MIN_FAMILY_ROWS_EVAL = 10             # Bai 1 per-family evaluability rule (Section 5.5)
S_GRID = (0.0025, 0.005, 0.009)       # injected rare shares (< 0.01 by the outline's definition)
RARE_THR, RECUR_MIN_ABSENT = 0.01, 3  # outline definitions


def main(manifest_path, out_path):
    raw = Path(manifest_path).read_bytes()
    m = json.loads(raw)
    assert m["design"]["EVAL_DAY_SIZE"] == EVAL_DAY_SIZE and abs(m["design"]["ETA"] - ETA) < 1e-12
    by_root = {}
    for ds, r in m["datasets"].items():
        for p in r["pools"]:
            n_att_root = sum(p["attack_flows"].values())
            by_root[(ds, p["day"])] = {
                "dataset": ds, "root_day": p["day"], "n_root_flows": p["n_root"], "attack_flows_root": n_att_root,
                "attack_share_root": n_att_root / p["n_root"],
                "rare_families": {fid: dict(split, share_of_attack_flows=sum(split.values()) / n_att_root)
                                  for fid, split in p["rare_family_split"].items()},
                "_pool": p, "scenarios": defaultdict(int), "recurring": [], "rare_situations": [],
            }
    fam_flows = defaultdict(dict)    # (ds, day) -> {family name: flows} (families >= 1,000 flows only)
    rare_name = {}
    for s in m["scenarios"]:
        key = (s["dataset"], s["root_day"])
        by_root[key]["scenarios"][s["kind"]] += 1
        if s["kind"] == "recurring":
            p = s["params"]
            fam_flows[key][p["family"]] = p["family_flows"]
            if p["rare"]:
                rare_name[(key, str(p["family_id"]))] = p["family"]
    for s in m["scenarios"]:
        key = (s["dataset"], s["root_day"])
        R = by_root[key]
        pool = R["_pool"]
        if s["kind"] == "recurring":
            p = s["params"]
            assert p["absent_days"] >= RECUR_MIN_ABSENT
            rare = p["family_share_of_attack_flows"] < RARE_THR
            if rare:
                split = R["rare_families"][str(p["family_id"])]
                pol_rows = split["policy"]
            else:
                pol_rows = p["family_flows"] * pool["attack_flows"]["policy"] / R["attack_flows_root"]
            R["recurring"].append({
                "scenario": s["id"], "family": p["family"], "absent_days": p["absent_days"],
                "share_of_attack_flows": p["family_share_of_attack_flows"], "rare_and_recurring": rare,
                "expected_labelled_rows_per_day": N_POLICY_LABELS * pol_rows / pool["n_policy_pool"]})
        # rare-family (not necessarily recurring) situations: a root-day rare family present on evaluated days
        for fid in R["rare_families"]:
            if s["kind"] == "feature":
                R["rare_situations"].append({"scenario": s["id"], "status": "rare at root; realised share under "
                                             "feature reweighting not in manifest [VERIFY at export]"})
            elif s["kind"] == "prior":
                t = s["params"]["family_temper"]
                if t == 0.0:
                    R["rare_situations"].append({"scenario": s["id"], "status": "rare (temper 0 keeps family shares)"})
                else:
                    fl = fam_flows[key]
                    if sum(fl.values()) == R["attack_flows_root"]:   # all families of the day are known
                        u = {f: c ** (-t) * c for f, c in fl.items()}
                        share = u[rare_name[(key, fid)]] / sum(u.values())
                        R["rare_situations"].append({"scenario": s["id"], "status":
                                                     f"not rare after temper {t}: share {share:.4f}"
                                                     if share >= RARE_THR else f"rare (share {share:.4f})"})
                    else:
                        R["rare_situations"].append({"scenario": s["id"], "status": "unknown: family counts missing"})
            else:
                p = s["params"]
                st = ("rare and recurring" if p["family_id"] == int(fid) else
                      "rare on pre and reintroduction days; not rare while the other family is absent")
                R["rare_situations"].append({"scenario": s["id"], "status": st})
    # injection feasibility (option B): any family with >= 1,000 flows on a root day with >= 2 such families
    for key, R in by_root.items():
        pool = R["_pool"]
        inj = []
        for f, c in fam_flows.get(key, {}).items():
            A = R["attack_flows_root"]
            row = {"family": f, "root_share": c / A}
            for s in S_GRID:
                w = s * (A - c) / (c * (1 - s))           # weight that gives f the share s of attack flows
                e = {}
                for part in ("eval", "policy"):
                    a_p = pool["attack_flows"][part]
                    n_p = pool[f"n_{part}_pool"]
                    c_p = c * a_p / A
                    e[part] = (w * c_p) / (n_p - c_p + w * c_p)
                row[f"s={s}"] = {
                    "eval_rows_per_day": EVAL_DAY_SIZE * e["eval"],
                    "labelled_rows_per_day": N_POLICY_LABELS * e["policy"],
                    "evaluable_per_family_rule": EVAL_DAY_SIZE * e["eval"] >= MIN_FAMILY_ROWS_EVAL,
                    "down_weight": w if w < 1 else None,
                }
            inj.append(row)
        R["injection_candidates"] = inj
    roots = []
    for key, R in sorted(by_root.items()):
        R.pop("_pool")
        R["scenarios"] = dict(R["scenarios"])
        R["n_recurring_situations"] = len(R["recurring"])
        R["n_recurring_families"] = len({x["family"] for x in R["recurring"]})
        R["n_rare_and_recurring_situations"] = sum(x["rare_and_recurring"] for x in R["recurring"])
        R["n_rare_situations"] = sum(x["status"].startswith("rare") for x in R["rare_situations"])
        R["n_rare_situations_unverified"] = sum("VERIFY" in x["status"] for x in R["rare_situations"])
        roots.append(R)
    summary = {
        "n_root_days": len(roots),
        "root_days_with_rare_family": [r["root_day"] for r in roots if r["rare_families"]],
        "root_days_with_recurring_situations": [r["root_day"] for r in roots if r["n_recurring_situations"]],
        "root_days_with_rare_and_recurring": [r["root_day"] for r in roots if r["n_rare_and_recurring_situations"]],
        "n_recurring_situations": sum(r["n_recurring_situations"] for r in roots),
        "n_recurring_root_family_pairs": sum(r["n_recurring_families"] for r in roots),
        "n_root_days_any_H2_situation": sum(1 for r in roots if r["rare_families"] or r["n_recurring_situations"]),
        "n_root_days_injection_feasible": sum(
            1 for r in roots if any(c["s=0.005"]["evaluable_per_family_rule"] for c in r["injection_candidates"])),
        "n_injection_root_family_pairs_s0.005": sum(
            sum(c["s=0.005"]["evaluable_per_family_rule"] for c in r["injection_candidates"]) for r in roots),
    }
    out = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "settings": {"b": B, "eta": ETA, "policy_labels_per_day": N_POLICY_LABELS, "s_grid": S_GRID},
           "summary": summary, "root_days": roots}
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in roots:
        print(r["dataset"][3:6], r["root_day"], "rare:", {k: round(v["share_of_attack_flows"], 4)
                                                         for k, v in r["rare_families"].items()},
              "recurring:", r["n_recurring_situations"], "fams:", r["n_recurring_families"],
              "rare-situations:", r["n_rare_situations"], "(unverified", r["n_rare_situations_unverified"], ")")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "situations_h2.json")
