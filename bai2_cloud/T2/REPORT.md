# T2 REPORT: Proposition 1 and the choice of q_min (revised: D1 + D3 applied)

## What was done
- `prop1.md` contains the assumptions and proofs of (a), (b) and (c), followed by the q_min rule. All numbers come from
  `prop1_sim.py` (numpy, seed 20261029, synthetic only, about 1.5 min of CPU) through `prop1_sim_results.json`.
- (a) and (b) are proven. (c) is proven as an invariant, N_c ≥ min(q_min, M_c), under policy axioms P0–P3.
- **D1 (hybrid allocation).** In the normal state k_R = k_R0, and the floor is paid for inside F. In a shift state
  the recent bin is k_R0 + min(δk_F0, k_F0 − G·q), so it loses at most G·q_min. Parts (a) and (b), Lemmas 1–2 and the
  invariant are unchanged, because they only need k_F ≥ G·q_min. Only the cost sentence of Theorem 1 and Section 4 changed.
- **D3 (primary rule).** J(q) = missed recurring reappearances + missed emerging families, per day, with no μ.
  The exact rule enumerates q from q_thr to q_rob.
  The closed form is q_rob = round(max(q̃, κ)), with q̃ = [ln(βA/(φρ_e a_e G)) + a_e(k − k_H)]/(β + a_e G).
  q_thr ≤ q* ≤ q_rob is proven (Theorem 3). The μ rule is kept as Appendix A.
- MC: 59 MC-vs-formula comparisons, largest |z| = 2.49, none above 3. The invariant had 0 violations, and the observed
  recent-bin sizes equal the D1 values. In the illustrative base setting, theory gives q* = 74. The MC minimum is at 72,
  which is 0.13 SE away, so the two agree within MC error.

## OPEN
- (c) is unproven for the cluster variant and for trusted rows that are later removed by the poisoning filter.
- The shift state may be correlated with reappearance (R6 assumes independence).
- The smaller F share left for common groups is not priced. When shifts are rare (φ → 0), q* goes to the cap
  ⌊k_F0/G⌋. A cap or the μ appendix is needed there.
- Monotonicity of the exact q* is numerical only, and unimodality of J is unproven.
- θ, φ, ρ, ρ_e and r_e need values. All parameter values used are illustrative.
- Whether T3 adopts P0–P3 as written is still open.

## Decisions still needed from the NCS
1. The value or estimation plan for θ, and estimates of φ, ρ, ρ_e and r_e on dev data. These feed T5.
2. Whether T3 implements P0–P3 literally: rows leaving R/H are offered to F, and a shift never takes k_F below G·q_min.
3. How to bound q when φ is near 0: a fixed cap, or the μ price.
