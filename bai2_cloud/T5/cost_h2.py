"""T5 item 6: TabFM GPU-hours of the proposed H2 design, per tier, and the cut order.

    python cost_h2.py <path/to/bai2_handoff> situations_h2.json [cost_h2.json]

Timing (bundle context/numbers_addenda.json, cost.*; Bai 1 Table 11): TabFM on an NVIDIA L4 takes a median 58.4 s per
evaluated day at 10,000 query rows and k = 2,000, i.e. 5.84 s per 1,000 rows (fit + predict). H2 needs per-family
recall at an operating threshold, so TabFM must ALSO score the D_cal rows (Bai 1 DEV-010 lesson): about 1,200 rows
(eta b = 400 reserved labels per day x W_V = 3 days; Bai 1 Section 3.6). Rows per scored day = 11,200 (proportional
scaling of the per-row median; the Bai 1 linear fit 4.36 s + 5.25 s per 1,000 rows gives a slightly lower figure,
also reported). LightGBM runs on CPU and is not counted in GPU-hours.

Situations (situations_h2.json): primary H2 situations per root day =
  * natural recurring: one scenario per (root day, family) with d = 3 (the outline's minimum absence);
  * injected rare-recurring (option B): one scenario per (root day, non-dominant family) with >= 10 expected evaluation
    rows per day at the injected share s = 0.009.
TabFM scores only the N_REINTRO = 2 reintroduction days of each situation (the H2 window); LightGBM scores every day.
Budget: no GPU budget for Bai 2 is in the bundle. The script prints the cut sequence for several reference budgets,
including the Bai 1 cap (12 compute units / (1.62 units per hour x 1.1 margin) = 6.73 Colab hours).
"""
import hashlib
import json
import sys
from pathlib import Path

ROWS_EVAL, ROWS_CAL = 10_000, 1_200
WINDOW_DAYS = 2
S_INJ = "s=0.009"


def main(bundle, situations, out_path):
    add = json.loads((Path(bundle) / "context" / "numbers_addenda.json").read_text(encoding="utf-8"))
    sec_per_1000 = add["cost"]["tabfm_L4_seconds_per_1000_rows_median"]
    sec_day = sec_per_1000 * (ROWS_EVAL + ROWS_CAL) / 1000
    sec_day_fit = 4.36 + 5.25 * (ROWS_EVAL + ROWS_CAL) / 1000          # Bai 1 Section 4.5 linear fit
    sit = json.loads(Path(situations).read_text(encoding="utf-8"))
    rec, inj, roots = 0, 0, 0
    for r in sit["root_days"]:
        fams = {x["family"] for x in r["recurring"]}
        n_inj = sum(1 for c in r["injection_candidates"] if c["root_share"] < 0.5 and
                    c[S_INJ]["evaluable_per_family_rule"])
        rec += len(fams)
        inj += n_inj
        roots += bool(fams or n_inj)
    n_sit = rec + inj
    natural_rare = 2      # 2018-02-21: feature scenario 0 and prior scenario 0 (temper 0), descriptive

    def days(policies, noise, seeds, situations=n_sit, window=WINDOW_DAYS):
        return policies * noise * seeds * situations * window

    def tiers(seeds_secondary=5, secondary_noise=2, secondary_tabfm=True, seeds_primary=5):
        t = {
            "A_primary (ThreeBin, CURE, CBRS-family, ECBRS; 2 noise levels)": days(4, 2, seeds_primary),
            "A_descriptive natural rare situations (2018-02-21)": days(4, 2, seeds_primary, natural_rare, 4),
        }
        sec = {
            "B (FIFO, DualFIFO, CBRS, herding, LoCalPFN-kNN [LoCalPFN cost x?, VERIFY])": days(5, secondary_noise,
                                                                                            seeds_secondary),
            "C (TabPFN-2.5 robustness: ThreeBin + best tier-A comparator, ToN only [TabFM timing as proxy, VERIFY]; "
            "binary-label ThreeBin; ARF on CPU = 0)": days(2, 1, min(seeds_secondary, 3), round(n_sit * 6 / roots))
            + days(1, secondary_noise, seeds_secondary),
            "ablations (ThreeBin without R, without F, without H)": days(3, secondary_noise, seeds_secondary),
        }
        if not secondary_tabfm:
            sec = {k: 0 for k in sec}
        t.update(sec)
        return {k: {"tabfm_days": v, "gpu_hours": v * sec_day / 3600} for k, v in t.items()}

    steps = [
        ("base: outline (5 seeds everywhere)", tiers()),
        ("cut 1 (outline): secondary comparisons 5 -> 3 seeds, primary unchanged", tiers(seeds_secondary=3)),
        ("cut 2 (proposed, NCS): secondary TabFM at 0 % noise only", tiers(seeds_secondary=3, secondary_noise=1)),
        ("cut 3 (proposed, NCS): secondary on LightGBM only (no TabFM)", tiers(secondary_tabfm=False)),
        ("cut 4 (NCS decision, changes the outline's 'primary unchanged'): primary 5 -> 3 seeds",
         tiers(secondary_tabfm=False, seeds_primary=3)),
    ]
    out_steps = []
    for name, t in steps:
        tot = sum(v["gpu_hours"] for v in t.values())
        out_steps.append({"step": name, "tiers": t, "total_gpu_hours": tot})
    budgets = {"Bai 1 cap (12 CU = 6.73 Colab h)": 12 / (1.62 * 1.1), "25 GPU-h": 25.0, "50 GPU-h": 50.0,
               "100 GPU-h": 100.0}
    applied = {}
    for bname, b in budgets.items():
        ok = next((s["step"] for s in out_steps if s["total_gpu_hours"] <= b), None)
        applied[bname] = {"budget_gpu_hours": b, "first_step_within_budget": ok or "none: NCS decision needed"}
    out = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "seconds_per_scored_day": sec_day, "seconds_per_scored_day_linear_fit": sec_day_fit,
           "rows_per_scored_day": ROWS_EVAL + ROWS_CAL, "seconds_per_1000_rows": sec_per_1000,
           "situations": {"root_days": roots, "natural_recurring": rec, "injected_rare_recurring": inj,
                          "total_primary": n_sit, "window_days": WINDOW_DAYS},
           "steps": out_steps, "budgets": applied,
           "colab_units_per_gpu_hour_bai1": 1.62}
    Path(out_path).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"situations: {out['situations']}; s/day {sec_day:.1f} (fit {sec_day_fit:.1f})")
    for s in out_steps:
        print(f"\n{s['step']}: TOTAL {s['total_gpu_hours']:.1f} GPU-h")
        for k, v in s["tiers"].items():
            print(f"   {v['tabfm_days']:6d} days  {v['gpu_hours']:6.1f} h  {k}")
    print()
    for k, v in applied.items():
        print(f"{k}: {v['first_step_within_budget']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cost_h2.json")
