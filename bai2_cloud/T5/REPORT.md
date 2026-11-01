# T5 REPORT: H2 design, power, preregistration draft

Numbers come from scripts here, on bundle files only; no data used.

**Done**
1. **Situations** (manifest only): a rare family occurs on 1 root day; recurring scenarios cover 10 root days. H2 has
   at most **n = 10 independent root days**, and the rare-family part alone has n = 1, so it cannot be tested.
2. **Options:** rare-recurring injection by within-pool down-weighting extends the rare part to the same 10 root days
   without changing the outline. More scenarios add no units
   (`design_h2.md` §2).
3. **Primary test:** exact Wilcoxon/HL on root days at one-sided α = 0.025. It must hold for every tier-A comparator
   at both noise levels. This is a deviation from the outline's mixed model (DEV-H2-01).
4. **ε₂:** `eps2_criteria.py` computes it. It was not run on data.
5. **Power:** error control fails under dependence. At n = 10 and α = 0.05, the false-positive rate is 0.063–0.075 at
   ρ = 0.3 and 0.093–0.134 at ρ = 0.6. At α = 0.025 it is ≤ 0.052 up to ρ = 0.3, but 0.06–0.10 at ρ = 0.6. With
   n = 4 the rule cannot be met at all. Under the normal model, power for the full claim is 0.15 at θ = 0.07 and
   0.54 at θ = 0.10.
6. **Cost:** 161.2 GPU-h at base, 119.3 GPU-h after the outline's cut, 52.4 GPU-h for tier A only.
7. **Preregistration draft:** outcome labels INSUFFICIENT_DATA, CRITERION_NOT_MET, SMALL_EFFECT_EVIDENCE,
   H2_SUPPORTED and NO_CONFIRMATORY_CLAIM. A policy failure is scored, not treated as missing. q_min = q*(θ̂, ρ̂_e, r̂_e)
   by default, with the grid search as a check.
8. **D2/D3:** `estimate_theta.py` (CPU, FIFO only) and `estimate_emergence_rates.py` (labels only). Both are hashed
   before they are run.

**Finding:** the locked Bài 1 `hodges_lehmann` lower bound is one rank too high (level 0.0523 at n = 16). Bài 1's
verdict is unaffected. H2 uses the corrected index (DEV-H2-02).

**Missing or undefined:** `bai2_cloud/T3/REPORT.md` is absent from the bundle. "Tiers", "ECBRS", "cut order above"
and "ablations above" are not defined in the bundle; the reading I used is in `design_h2.md` §6. The θ, ρ_e and r_e
definitions must match T2's q* [VERIFY].

**NCS decisions**
1. The GPU budget. The cut order cannot be applied without it, and Bài 1's cap of about 6.7 h is below tier A.
2. Whether to add root days.
3. α = 0.025 and the HL index fix.
4. The tier reading and the ECBRS implementation.
5. LightGBM-driven signals for ThreeBin and CURE.
6. F_op, M_op and R_op.
7. Generator v3.
