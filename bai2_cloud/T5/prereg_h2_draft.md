# Preregistration draft: H2 (Bài 2, RQ2) — DRAFT, not locked

Status: draft by T5 for NCS review. Values in ⟨angle brackets⟩ are fixed at the lock. Rationale and numbers are in
`design_h2.md`. Nothing here was computed on data.

## 1. Hypothesis

**H2.** On rare-family and recurring-family situations (outline definitions: rare = < 1 % of the target day's attack
flows; recurring = absent ≥ 3 days before), ThreeBin exceeds **each** tier-A comparator (CURE, CBRS-family, ECBRS) by
at least ε₂ in per-family AUPRC_w (E1). This must hold with TabFM as the model, at 0 % and at 10 % random label noise.
Key secondary (fixed sequence, only if E1 is met): the same for per-family recall at the operating threshold (E2),
with margin ε₂,rec.

- Parameter θ_c,ν: pseudo-median over root days of x_r = ThreeBin − c (comparator c, noise level ν).
- H0: θ_c,ν ≤ ε₂ for at least one (c, ν).
- H1: θ_c,ν > ε₂ for all six (c, ν).
- ε₂ = max(operational criterion, 2 × seed SD), computed by `eps2_criteria.py` ⟨sha256⟩ on the
  development pilot before the lock. The outline's provisional value is 0.03 ⟨locked value⟩.
- LightGBM runs the same contexts. Its results are secondary and descriptive. TabPFN-2.5 is a tier-C robustness check.

## 2. Data, situations and units

- Development data only: NF-ToN-IoT-v3 and NF-CICIDS2018-v3, through the hashed generator and manifest v3 (v2 plus the
  rare-recurring kind B/B′, `design_h2.md` §2) ⟨sha256⟩. UGR'16 and CIDDS-001 are not opened.
- Primary situations: for each of the 10 eligible root days, one natural recurring scenario per family (d = 3), plus
  one injected rare-recurring scenario per non-dominant family with ≥ 10 expected evaluation rows per day at s = 0.009.
  That makes ⟨32⟩ situations. N_PRE for kind B′ is set by the rule "expected labelled rows of f before absence
  ≥ 2 q_min".
- **Unit = root day** (n = 10). Situations and seeds are averaged within the root day:
  x_r,c,ν = mean over situations and 5 seeds of the window mean of ThreeBin − c.
- Window: the 2 reintroduction days.
- Settings as in Bài 1 unless stated: k = 2,000, b = 2,000, η = 0.2, τ = 0, 10,000-row stratified evaluation sample
  with all rare-family rows and 1/π weights, isolation filter with zero tolerated drops. TabFM also scores the D_cal
  rows.
- Model signals of ThreeBin (hard-row margin, BBSE) and of CURE (entropy) come from the locked LightGBM ⟨confirm⟩.

## 3. Primary decision

For each (c, ν): the exact one-sided Wilcoxon signed-rank test of x − ε₂ at **α = 0.025**, equivalently the exact HL
lower bound `walsh[c−1]` ≥ ε₂ (corrected index, DEV-H2-02).

| Label | Rule |
|---|---|
| **INSUFFICIENT_DATA** | Fewer than **8** evaluable root days for any (c, ν), or fewer than 3 per dataset. At n ≤ 4 the rule cannot be met at all, and at n = 6 the FPR under dependence reaches 0.18 (`design_h2.md` §5). This label is neither evidence for nor against H2 |
| **NO_CONFIRMATORY_CLAIM** | Technical failures > 5 % of primary evaluable TabFM days |
| **H2_SUPPORTED** | Lower bound ≥ ε₂ for all six (c, ν). Suffix **_DEPENDENCE_SENSITIVE** if the lag-1 autocorrelation of root-day outcomes exceeds 0.3 in either dataset, or the block-bootstrap bound is < ε₂ |
| **SMALL_EFFECT_EVIDENCE** | Not supported, but the Wilcoxon test against 0 rejects (α = 0.025) for all six (c, ν). This is evidence of an effect below ε₂, with no claim of practical superiority |
| **CRITERION_NOT_MET** | Enough data, and neither of the two rows above applies |

Per (c, ν) the report gives the HL estimate, the exact two-sided 95 % interval, the bounds at α = 0.05 and 0.025, the
exact sign test, n, and the per-dataset values. E2 is tested only after H2_SUPPORTED, with the same rule and ε₂,rec.

## 4. q_min and other tuned parameters

- **q_min = q*(θ̂, ρ̂_e, r̂_e)**, using T2's locked q* function ⟨file, sha256⟩.
  - θ̂ from `estimate_theta.py` (D2) ⟨sha256⟩.
  - ρ̂_e and r̂_e from `estimate_emergence_rates.py` (D3, labels only) ⟨sha256⟩.
  - Both scripts are fixed and hashed before they are run, and their outputs are hashed into this document.
- **Grid search as a check.** The outline's grid search (≤ 12 configurations, criterion "mean AUPRC over controlled
  drift situations", LightGBM, development data) is also run, and its q_min is reported.
- **H2 uses q*.** The grid value is used only for a sensitivity analysis that is reported but supports no claim. If
  q* and the grid value differ, both are reported with the H2 result under each. A rounding rule for q* ⟨to fix⟩ and
  the constraint k_F ≥ G·q_min apply.
- The other ThreeBin parameters (w_R : w_F : w_H, γ, κ_π, κ_X, δ; D = 3; k_H ≤ 0.1k) come from the same grid and are
  locked before any H2 run. Each comparator gets ≤ 12 configurations (τ_CURE is reused from Bài 1).

## 5. Missing data and non-evaluable days

- **not_evaluable (decided before TabFM runs, identical for all policies):** no row of f in the window's evaluation
  sample, < 10 rows of f, or < 10 benign rows (Bài 1 rule). A day that is not evaluable for one policy is excluded
  for all.
- **A policy failure is not missing data.** If a policy's context holds one class, or holds no row of f, the day is
  scored. The model gives a constant score: AUPRC_w = weighted prevalence, recall = 0. This is the actual consequence
  of the policy (Bài 1 PC1 / DEV-PC1-001 lesson). Exception: if every policy's context is one-class, the day is
  not_evaluable.
- **tech_fail:** a TabFM error after 3 attempts. The primary analysis uses the available days. Sensitivity analyses
  S1 (ThreeBin worst case: E1 = 0 on its failed days, comparator unchanged), S2 (complete situations only) and S3 (root
  days with no failure) are reported whenever any tech_fail exists. Above 5 %, the verdict is NO_CONFIRMATORY_CLAIM,
  the cause is fixed, and the run is repeated as a deviation.
- For every situation and policy, the number of target-family rows in the context at the start of each window day is
  recorded, together with whether q_min was reachable.

## 6. Secondary and descriptive analyses (no claim rests on them)

- The outline's mixed model (root day random, situation nested, dataset fixed) with a hierarchical bootstrap.
- Overall AUPRC_w (no-harm check).
- LightGBM results.
- Tiers B and C, and ablations without R, F or H.
- Natural rare (non-recurring) situations of 2018-02-21.
- The moving-block bootstrap (block 3), shown by simulation not to control dependence at n = 10.
- Lag-1 autocorrelations.

## 7. Deviations from the outline (declared before any H2 run)

| ID | Change | Reason |
|---|---|---|
| DEV-H2-01 | Primary test: exact Wilcoxon / HL on root days at one-sided α = 0.025, instead of a mixed model with hierarchical bootstrap. The mixed model becomes secondary | Bài 1 DEV-004 (heavy tail). Simulation: t / bootstrap FPR up to 0.19 / 0.35 at n = 10; α = 0.05 FPR 0.07 at ρ = 0.3 |
| DEV-H2-02 | HL lower bound index corrected to `walsh[c−1]` (the locked Bài 1 function is one rank high; level 0.0527 at n = 10) | Exactness. Bài 1's verdict is unaffected |
| DEV-H2-03 | H2 claim requires all tier-A comparators × both noise levels (intersection-union) | The outline's "higher than CURE and class-balanced buffers … also with 10 % noise" |
| DEV-H2-04 | E1 is the per-family AUPRC_w on the reintroduction window, not the overall AUPRC_w | A < 1 % family barely moves the overall metric |
| DEV-H2-05 | Generator v3 adds the rare-recurring kind | Only 1 root day has a natural rare family |
| ⟨NCS⟩ | Seeds, tiers and cuts per the GPU budget (`design_h2.md` §6) | Budget not yet set |

## 8. Known limitations (stated in advance)

- n = 10 root days. Under the normal between-root-day model, power for the full claim is 0.54–0.56 at θ = 0.10 and
  0.11–0.15 at θ = 0.07.
- Under day-to-day dependence of ρ = 0.6, the FPR at α = 0.025 is 0.06–0.10.
- The variance models are Bài 1 TabFM − LightGBM spreads, not policy contrasts.
- The rare-family evidence comes from injected situations on 10 root days and natural ones on 1 root day.
- Development data only. Confirmation on unused data is separate.
