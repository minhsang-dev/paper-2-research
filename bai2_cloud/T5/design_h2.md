# Design of the H2 test (T5)

H2 (outline v14.3, Mục tiêu 2): on situations with a rare or recurring attack family, the three-compartment policy
(ThreeBin) reaches a higher per-family recall and AUPRC than CURE and class-balanced buffers by at least ε₂, on TabFM
(TabPFN-2.5 as a robustness check), also with 10 % random label noise.

Every number below is printed by a script in this folder. The script is named in brackets: [S] =
`count_situations_h2.py` → `situations_h2.json`; [P] = `power_h2.py` → `power_h2_results.json`, tables in
`power_h2_tables.md`; [C] = `cost_h2.py` → `cost_h2.json`. Bài 1 numbers come from the bundle. No data row was read,
and nothing was computed on data.

**Missing input.** `bai2_cloud/T3/REPORT.md` was not in the bundle or the repository. The ThreeBin interface is
assumed to be the harness interface `update(context, new_rows, unlabelled) → context` with |context| ≤ k. The words
"tier", "ECBRS", "the cut order above" and "the ablations above" are not defined anywhere in the bundle. Section 6
states the reading used here, and the NCS should confirm it.

## 1. Situations in the development data (manifest only) [S]

| Quantity | Count |
|---|---|
| Root days | 16 (7 ToN-IoT, 9 CICIDS2018) |
| Root days with a rare family (< 1 % of attack flows) | **1**: CIC 2018-02-21, DDOS_attack-LOIC-UDP, 0.33 % |
| Root days with recurring scenarios (d ∈ {3, 4, 5}, all ≥ 3) | **10**: ToN 04-24 … 04-29 (6), CIC 02-14, 02-15, 02-16, 02-21 (4) |
| Recurring scenarios / (root day, family) pairs | 63 / 21 (2 families per root day, 3 on ToN 04-25) |
| Rare **and** recurring scenarios | 3, all on 2018-02-21 (LOIC-UDP, d = 3, 4, 5) |
| Rare (not recurring) situations on 2018-02-21 | 6 prior scenarios with temper 0, plus 3 HOIC-recurring scenarios (LOIC-UDP is rare on the pre and reintroduction days); 10 feature scenarios are rare at root but their realised share is not in the manifest [VERIFY at export]; 4 prior scenarios with temper 0.5 lift LOIC-UDP to 5.5 % (not rare) |
| Root days with any H2 situation | **10** |

**Verdict.** The development data contain no more than **10 independent root days** for H2: 6 ToN-IoT and 4
CICIDS2018. For the rare-family part alone they contain **1**. With one root day no test is possible. The exact
signed-rank rule cannot be met below n = 5, because its smallest one-sided p is 2⁻ⁿ [P, table E]. A rare-family claim
therefore needs option B below.

The families of the 6 root days without recurring scenarios are not in the manifest. The locked
`protocol_code/replay/count_days.py` lists them, but its output is not in the bundle.

Labels are drawn uniformly from the policy part (1,600 per day). The expected number of labelled rows per day of each
recurring family is small for the families that matter most [S]: LOIC-UDP 1.57, mitm 4.67, dos (04-25) 17.6,
ransomware 19.3. A family with ~1.6 labelled rows per day enters the context with ~3 rows over the two pre-absence
days. **A per-group floor q_min larger than that cannot be filled, and the situation then tests nothing about
retention.** The analysis must therefore record, for every situation, how many target-family rows each policy held at
the reintroduction day.

## 2. Getting enough situations without changing the outline

| Option | What it adds | How independent units are counted | Outline change? |
|---|---|---|---|
| A. More controlled recurring scenarios per root day (more d, more seeds) | More scenarios on the same 10 root days. The manifest already uses every (family, d) cell (≤ 9 < MAX_PER_KIND = 10) | **No new units.** Still 10 root days; scenarios are nested in root days (outline Nội dung 1) | No |
| **B. Rare-recurring injection by within-pool down-weighting (recommended)** | A new kind built from the existing drift mechanism (b) "resampling by class": weight the rows of a non-dominant family f inside each pool so that f has share s < 1 % of attack flows on every day, then remove it for d = 3 days and bring it back at share s. At s = 0.9 %, 11 (root day, family) pairs on **all 10** root days are evaluable (≥ 10 expected evaluation rows per day, Bài 1 rule) [S] | **Root days, unchanged: 10.** It turns the rare-family part from n = 1 into n = 10 (the same 10 root days) | No: drift kind (b) and the outline's definitions of rare and recurring. It needs a generator v3 with one new kind, hashed with its manifest before any model run (as DEV-001) |
| B′. Longer pre-absence phase for kind B | Expected labelled rows of f at s = 0.9 % are 0.55–12.8 per day [S]. With N_PRE = 2 the context may hold fewer than q_min rows. Set N_PRE per situation so that the expected labelled rows before absence are ≥ 2 q_min (rule fixed before the run) | Unchanged (root days) | No (generator constant, not the outline) |
| C. Lower MIN_FAMILY_FLOWS (generator constant 1,000) | May add recurring families on the 6 other root days. **Cannot be counted from the manifest**; run the locked `count_days.py` first | Up to 16 root days, if those days have ≥ 2 families | No |
| D. Natural recurrence in chronological real-day replay (DEV-011 style) | Among the families visible in the manifest, none reappears after ≥ 3 absent days (CIC families appear on one day each; ToN families recur on consecutive days). Partial count: families < 1,000 flows are not in the manifest | Day transitions, dependent | No. Descriptive only |
| E. Treat (root day, family) pairs as units | 21 pairs | **Not independent** (shared root-day flows). Rejected for the primary analysis | Contradicts the outline's unit |
| **F. Add root days: more datasets or a longer period (NCS decision)** | The outline's own remedy when power < 0.8 at ε: "bổ sung ngày gốc … chứ không chỉ tăng số tình huống, và không nới biên" | New root days count as new units | It adds data. **Not proposed here; listed for the NCS** |

Recommended: **A (existing recurring scenarios, d = 3) + B/B′**, giving **n = 10 root days**, each with natural
recurring and injected rare-recurring situations (32 primary situations in total [C]). Within the outline this is the
maximum. Section 5 shows that n = 10 is enough only for large effects.

## 3. Endpoints and primary analysis

**Window.** Retention matters on the day a family comes back. Both models score day t before day t's labels arrive.
The H2 window is therefore the **N_REINTRO = 2 reintroduction days** of each situation. Averaging over the whole chain
would dilute the effect over ~7 days.

**Endpoints (per situation, target family f).**
- *E1 (primary).* Per-family AUPRC_w on the window: benign rows plus rows of f, label 1 = f, Bài 1 inverse-inclusion
  weights (the DEV-010 per-family computation). The overall AUPRC_w is not used for H2, because a family below 1 % of
  attacks barely moves it. The overall AUPRC_w is reported as a no-harm secondary.
- *E2 (key secondary, fixed sequence after E1).* Recall of f at the Bài 1 operating threshold: the (1 − α_eff)
  quantile of the benign D_cal scores, with α_eff = max(A_FP/N̂, 5/n_benign_cal). TabFM **must score the D_cal rows**
  (Bài 1 DEV-010 lesson), and this is costed in Section 6. The recall margin ε₂,rec is computed by the same two
  criteria as ε₂ (`eps2_criteria.py`, with recall in place of AUPRC).

**Unit and outcome.** For each root day r and comparator c: x_rc = mean over r's situations and 5 seeds of
(ThreeBin − c) in E1. Both models see the same context, as in Bài 1. ThreeBin's model-dependent signals (hard-row
margin, BBSE trigger) and CURE's entropy are computed by the locked LightGBM (Bài 1 DEV-005 precedent) **[NCS to
confirm]**.

**Primary rule (deviation DEV-H2-01 from the outline).** For each tier-A comparator, the exact one-sided Wilcoxon
signed-rank test of x − ε₂ is applied to the 10 root-day outcomes. Equivalently, the exact Hodges–Lehmann lower bound
must be ≥ ε₂, at **one-sided α = 0.025** (Section 5). H2 holds only if the rule holds for **every** tier-A
comparator and at **both** noise levels (0 % and 10 %). This intersection-union test needs no multiplicity
adjustment. The outline's mixed model (root day random, situation nested, dataset fixed) and the hierarchical
bootstrap are kept as **secondary** estimates.

Why the outline's model is not primary: Bài 1 DEV-004 (root-day SD 0.071 by moments vs 0.0024 by MAD) and [P, table
C]. At n = 10 the t rule has a false-positive rate (FPR) up to 0.098 at ρ = 0 (M3) and 0.191 at ρ = 0.6. The
root-level percentile bootstrap has an FPR of 0.08–0.35 at the boundary in every model. Under the heavy-tail model the
t rule has power 0.51 where the exact rule has 1.00.

**Correction to the locked Bài 1 HL function (DEV-H2-02).** `analysis_h1.hodges_lehmann` takes `walsh[c]` with
c = #{t : P(T ≤ t) ≤ α}. The exact bound is `walsh[c − 1]`. The locked index is one rank too high: its exact level is
0.0527 at n = 10 and 0.0523 at n = 16 (Bài 1's case), against 0.0420 and 0.0467 for the corrected index [P, table E].
At n ≤ 4 it returns min(x), with level 2⁻ⁿ (0.0625 at n = 4). The Bài 1 verdict is unaffected: its lower bound
(+0.0038) was far below ε₁. The same bug would give a nominal "bound" at n ≤ 4, where no exact test exists. H2 uses
the corrected index, which `test_power_h2.py` checks against the locked function.

**Dependence.** As in Bài 1, the moving-block bootstrap (block 3) is reported. The simulation shows that it **does not
control** dependence at this n: EXACT+BLOCK has an FPR equal to that of EXACT at every ρ [P, table C]. It is
therefore descriptive. The lag-1 autocorrelation of root-day outcomes within each dataset is reported. If it exceeds
0.3 in either dataset, the verdict carries the suffix DEPENDENCE_SENSITIVE.

## 4. ε₂

`eps2_criteria.py` computes the two outline criteria once the development pilot exists. It was **not run on data**;
only `--selftest` (synthetic) was run.
- (i) Operational, on NF-ToN-IoT-v3 at the operating recall R_op. The script models a uniform downward shift δ of the
  precision–recall curve and gives the δ that adds F_op false alarms per day, or loses M_op attack flows per day at the
  same alarm count (closed forms in the script). The criterion is the median over days of the smaller δ. **The NCS
  must fix F_op, M_op and R_op before running.** They cannot be derived from data.
- (ii) 2 × the seed-to-seed SD of the H2 contrast, pooled within (situation, comparator) cells. The worst comparator is
  used. The computation is blinded, as in Bài 1: only SDs are written.
- Locked value = max(i, ii). If it falls below the outline's 0.03, the output flags it. Lowering the margin is an NCS
  decision, since the outline forbids loosening it.

## 5. Power and false-positive rate [P]

The setup is 4,000 replicates per cell (MC SE 0.0034 at 0.05), seed 20261029, and root-day dependence
corr = ρ^|Δcalendar days| within a dataset. Marginals come only from the bundle: M1 heavy tail and M2 normal (Bài 1
§4.7 models A and B), M3 = the 10 Bài 1 recurring-kind root-day values, and M4 = the 20 Bài 1 per-family median
differences. All four are Bài 1 TabFM − LightGBM spreads, used as stand-ins. **No policy-contrast variance exists in
the bundle.**

**False-positive rate at the boundary (θ = ε₂), n = 10, one comparator, exact rule:**

| α (one-sided) | ρ = 0 | ρ = 0.3 | ρ = 0.6 |
|---|---|---|---|
| 0.05 | 0.035–0.040 | 0.063–0.075 | 0.093–0.134 |
| 0.025 | 0.014–0.025 | 0.035–0.052 | 0.062–0.103 |
| 0.01 | 0.003–0.009 | 0.012–0.025 | 0.035–0.062 |

(range over M1–M4, [P, table F]). **Error control fails under dependence.** At α = 0.05 the FPR exceeds 0.05 already
at ρ = 0.3, and reaches 0.13 at ρ = 0.6. α = 0.025 holds the FPR at ≤ 0.052 up to ρ = 0.3. No α tested holds the FPR
at 0.05 at ρ = 0.6 under the normal model: α = 0.01 gives 0.062. Bài 1 found negative lag-1 autocorrelations
(−0.35 ToN, −0.12 CIC), which gives some support for ρ ≤ 0.3, not proof. Proposal: **α = 0.025**, with the residual
risk stated.

**Power, n = 10, α = 0.025:** Under M1, M3 and M4 the between-root-day spread is tiny (MAD SD 0.0024, or empirical
values mostly < 0.03). Power is then ≈ 1.0 for θ ≥ ε₂ + 0.02 under M3 and M4. Under M1 it is only 0.49–0.51 at every
θ in Table F, because a single negative outlying root day pulls the α = 0.025 bound below ε₂ about half of the time.
The plausible bad case is M2 (SD 0.071):

| θ | 0.05 | 0.07 | 0.10 |
|---|---|---|---|
| single comparator, ρ = 0 / 0.3 | 0.118 / 0.172 | 0.349 / 0.389 | 0.773 / 0.750 |
| all 3 tier-A comparators (IUT), ρ = 0 / 0.3 | — | 0.105 / 0.150 | 0.559 / 0.536 |

Under M2, power does not reach 0.8 at any θ ≤ 0.10 for the full H2 claim. By the outline's own rule, that calls for
more root days (option F, NCS), not more situations and not a looser margin. Smaller n is worse:

| Subset | Result [P, table A] |
|---|---|
| n = 4 (CIC only) | The rule **can never be met**: P = 0.000 at every θ |
| n = 6 (ToN only) | Under M1, power stalls at ~0.5 for θ up to 0.10 (one outlier decides). The FPR at ρ = 0.6 reaches 0.10–0.18 |

## 6. Run design, tiers and TabFM cost [C]

The tier reading is an assumption for the NCS to confirm:
- **Tier A (confirmatory):** ThreeBin vs CURE, CBRS-family, ECBRS. ECBRS is not in the harness or in T4. Its source
  is assumed to be Amalapuram et al. (extended CBRS for NIDS continual learning) [VERIFY]; someone must implement and
  check it before the lock.
- **Tier B (descriptive):** FIFO, DualFIFO, CBRS (binary), herding, LoCalPFN-style kNN. LoCalPFN builds a context per
  query, so its TabFM cost is a multiple of the per-day figure [VERIFY with T4].
- **Tier C (descriptive):** TabPFN-2.5 robustness (ThreeBin + best tier-A comparator, one dataset, 3 seeds, outline
  Table 2.2), binary-label ThreeBin (clusters instead of families, outline Nội dung 2), and ARF (CPU only).
- **Ablations (descriptive):** ThreeBin without R, without F, and without H (outline: "ablation bỏ từng ngăn").

The cost is based on 65.4 GPU-s per scored day: the Bài 1 median of 5.84 s per 1,000 rows × 11,200 rows (10,000
evaluation + ~1,200 D_cal; the linear fit gives 63.2 s). TabFM scores only the 2 window days of each of the 32
situations; LightGBM scores everything on CPU.

| Step | Tier A | B | C | Ablations | **Total GPU-h** |
|---|---|---|---|---|---|
| Base (outline: 5 seeds; tier A also 2 noise levels) | 46.5 + 5.8 | 58.2 | 15.8 | 34.9 | **161.2** |
| Cut 1 (outline): secondary 5 → 3 seeds | 52.3 | 34.9 | 11.1 | 20.9 | **119.3** |
| Cut 2 (proposed): secondary TabFM at 0 % noise only | 52.3 | 17.5 | 7.6 | 10.5 | **87.9** |
| Cut 3 (proposed): secondary on LightGBM only | 52.3 | 0 | 0 | 0 | **52.4** |
| Cut 4 (NCS only; breaks "primary unchanged"): primary 5 → 3 seeds | 31.4 | 0 | 0 | 0 | **31.4** |

**No Bài 2 GPU budget is in the bundle, so the cut order cannot be applied to a real budget.** For reference, Bài 1's
cap of 12 compute units is about 6.7 Colab hours. Even cut 4 exceeds it about 4.7-fold. A budget of 50 GPU-h needs cut
4, and a budget of 100 GPU-h needs cut 2. Tier-B LoCalPFN and tier-C TabPFN-2.5 costs are lower bounds. ThreeBin's
q_min grid search runs on LightGBM (CPU), as τ_CURE did in Bài 1, so it adds no GPU time.

## 7. q_min: q* versus grid search

Two CPU scripts, fixed and hashed before they are run, give the inputs of T2's q* (Proposition 1, Poisson
recurrence):
- `estimate_theta.py` (D2): θ̂ = AUPRC_w lost per 1,000 rows removed from the recent compartment. It uses FIFO with
  k_R ∈ {2,000 … 1,200}, the locked LightGBM, and feature and prior scenario 0 of all 16 root days. θ̂ is the HL
  estimate of the root-day slopes with an exact 95 % interval. It never runs ThreeBin or a comparator.
- `estimate_emergence_rates.py` (D3, labels only, from the output of the locked `count_days.py`): ρ̂_e = (re)emergence
  events per family-day at risk (absent ≥ 3 previous days; Poisson MLE, Garwood interval), and r̂_e = the family's
  share of all flows on its emergence day (= r of Proposition 1, since labels are uniform). Rare events are reported
  separately.

Both definitions are this task's reading and **must match T2's derivation of q*** [VERIFY against T2 REPORT].
**Default: q_min = q*(θ̂, ρ̂_e, r̂_e)** with T2's locked q* function. The 12-configuration grid search (outline) is run
as a check, and both values are reported (prereg §4).

## 8. Decisions for the NCS

1. The GPU budget for Bài 2. Without it the cut order cannot be applied (161.2 GPU-h base, 52.4 GPU-h with tier A
   only).
2. Option F (more root days or datasets). The simulation says n = 10 is underpowered unless θ ≥ ~0.10 under the
   normal model.
3. α = 0.025 (DEV-H2-01) and the HL index correction (DEV-H2-02).
4. Tier reading, the ECBRS source and implementation, and LightGBM-driven model signals for ThreeBin and CURE.
5. F_op, M_op and R_op for ε₂ (i). Whether to keep 0.03 if the computed value is lower.
6. Generator v3 (kinds B and B′) and running `count_days.py` (option C).
