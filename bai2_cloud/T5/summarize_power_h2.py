"""Print the markdown tables of design_h2.md section 5 from power_h2_results.json (no other input).

    python summarize_power_h2.py power_h2_results.json > power_h2_tables.md
"""
import json
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
M = ["M1_heavy_tail", "M2_normal", "M3_recurring_empirical", "M4_per_family_empirical"]
TH = [0.0, 0.03, 0.04, 0.05, 0.07, 0.10, 0.20]
f = lambda v: f"{v:.3f}"
print(f"Monte Carlo: {R['meta']['reps']} replicates per cell (SE at 0.05 = {R['meta']['mc_se_at_0.05']:.4f}); "
      f"eps2 = {R['meta']['eps2']}; seed {R['meta']['seed']}; power_h2.py sha256 {R['meta']['script_sha256'][:12]}...\n")
print("### A. Probability that the primary rule (exact HL lower bound >= eps2) is met, one comparator\n")
for cfg in ["n10_H2_design", "n6_ToN_only", "n4_CIC_only", "n16_all_root_days"]:
    print(f"\n**{cfg}** (theta = 0.03 column = false-positive rate at the boundary; theta = 0 = far null)\n")
    print("| model | rho | " + " | ".join(f"θ={t}" for t in TH) + " |")
    print("|---|---|" + "---|" * len(TH))
    for m in M:
        for rho in (0.0, 0.3, 0.6):
            row = {r["theta"]: r for r in R["single_contrast"] if r["config"] == cfg and r["model"] == m and r["rho"] == rho}
            print(f"| {m} | {rho} | " + " | ".join(f(row[t]["p_exact_rule"]) for t in TH) + " |")
print("\n### B. Same rule with the index of the locked Bai 1 HL function (one rank too high), n = 10, theta = eps2\n")
print("| model | rho=0 | rho=0.3 | rho=0.6 |\n|---|---|---|---|")
for m in M:
    v = [r["p_exact_rule_locked_index"] for rho in (0.0, 0.3, 0.6) for r in R["single_contrast"]
         if r["config"] == "n10_H2_design" and r["model"] == m and r["rho"] == rho and r["theta"] == 0.03]
    print(f"| {m} | " + " | ".join(f(x) for x in v) + " |")
print("\n### C. Rules compared, n = 10 (FPR at θ = 0.03; power at θ = 0.07)\n")
print("| model | rho | EXACT FPR | EXACT+BLOCK FPR | T FPR | HBOOT FPR | SIGN FPR | EXACT power | EXACT+BLOCK power | T power | HBOOT power |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in M:
    for rho in (0.0, 0.3, 0.6):
        a = {r["theta"]: r for r in R["rules_n10"] if r["model"] == m and r["rho"] == rho}
        n, p = a[0.03], a[0.07]
        print(f"| {m} | {rho} | {f(n['EXACT'])} | {f(n['EXACT+BLOCK'])} | {f(n['T'])} | {f(n['HBOOT'])} | {f(n['SIGN'])} | "
              f"{f(p['EXACT'])} | {f(p['EXACT+BLOCK'])} | {f(p['T'])} | {f(p['HBOOT'])} |")
print("\n### D. Intersection-union: all m contrasts must meet the rule, n = 10\n")
print("| model | rho | m | " + " | ".join(f"θ={t}" for t in TH) + " |")
print("|---|---|---|" + "---|" * len(TH))
for m in M:
    for rho in (0.0, 0.3, 0.6):
        for k in (3, 6):
            row = {r["theta"]: r for r in R["iut"] if r["model"] == m and r["rho"] == rho and r["m"] == k}
            print(f"| {m} | {rho} | {k} | " + " | ".join(f(row[t]["p_all_m_contrasts"]) for t in TH) + " |")
print("\n### E. Small n: can the exact rule be met at all? (exact, independent days)\n")
print("| n | min one-sided p | can reject at 0.05 | exact level, locked index | exact level, corrected index |")
print("|---|---|---|---|---|")
for n, v in R["small_n_exact_rule"].items():
    print(f"| {n} | {v['min_one_sided_p']:.4f} | {v['test_can_reject_at_alpha']} | {v['exact_level_locked_index']:.4f} | "
          f"{v['exact_level_corrected_index']:.4f} |")
print("\n### F. Alpha calibration under dependence, n = 10 (P(rule met); theta = 0.03 is the FPR)\n")
print("| model | rho | alpha | θ=0.03 single | θ=0.05 single | θ=0.07 single | θ=0.1 single | θ=0.07 IUT m=3 | θ=0.1 IUT m=3 |")
print("|---|---|---|---|---|---|---|---|---|")
for m in M:
    for rho in (0.0, 0.3, 0.6):
        for a in (0.05, 0.025, 0.01):
            r = {x["theta"]: x for x in R["alpha_calibration_n10"] if x["model"] == m and x["rho"] == rho and x["alpha"] == a}
            print(f"| {m} | {rho} | {a} | {f(r[0.03]['p_single'])} | {f(r[0.05]['p_single'])} | {f(r[0.07]['p_single'])} | "
                  f"{f(r[0.1]['p_single'])} | {f(r[0.07]['p_iut_m3'])} | {f(r[0.1]['p_iut_m3'])} |")
