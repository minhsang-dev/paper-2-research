# T2 REPORT: Proposition 1 and the choice of q_min

## What was done
- `prop1.md`: assumptions and proofs of (a), (b), (c), then the q_min analysis. All numbers come from
  `prop1_sim.py` (numpy only, seed 20261029, synthetic streams, no data, CPU about 1 min) via `prop1_sim_results.json`.
- (a) and (b) are proven. (c) is proven as an invariant, N_c ≥ min(q_min, M_c) at all times, under explicit policy
  axioms P0–P3 (quota floor, eviction only above quota, rows leaving R/H offered to F). The floor uses at most
  G·q_min slots.
- Poisson recurrence model: P_miss(q) = (1−θ)^q · H(K−Gq), with H in closed form (Prop. 2).
- Objective: J(q) = Σ ρ_c P_miss,c(q) + μ G q. The exact rule is to enumerate q ≤ q_rob. The robust closed form is
  q_rob = round(ln(βΣρ/(μG))/β), β = −ln(1−θ). It is minimax over all absence laws, and q* ≤ q_rob is proven.
  A variant that needs no price μ, using an emerging-family term, also has a closed form.
- MC: 43 MC-vs-formula comparisons, largest |z| = 2.49, none above 3. The invariant had 0 violations in an explicit
  three-compartment buffer. The MC argmin of J equals the theoretical q* (20 in the illustrative base setting).
- The Bài 1 CIDDS-001 observation is cited as motivation only, with no new numbers.

## Open (not proven)
- (c) for the cluster variant (binary labels, groups re-estimated), and when the poisoning filter can remove
  already-trusted rows.
- Monotonicity of the exact q* in k, λ, r, ρ, G is seen numerically only. Unimodality of J outside the flushed
  regime is also unproven; the enumeration rule does not need it.
- All parameter values are illustrative and not calibrated.

## Decisions needed from the NCS
1. **Reading of "recent bin shrinks by G·q_min".** With fixed ratios w_R:w_F:w_H, k_R does not depend on q_min, and
   the floor is paid for inside F. Section 4 assumes k_R = k − k_H − G·q_min. Please confirm, or choose the
   fixed-ratio reading. In that case the cost term must be redefined as F-share loss of common groups.
2. **Coverage parameter θ** (probability that one stored row covers a reappearing variant) is not in the outline.
   It needs a dev-data estimate or a fixed value.
3. **Price μ vs. the same-currency variant (ρ_e, r_e).** One of the two must be chosen before registration. This
   affects T5.
4. Should T3 implement P0–P3 literally? In particular: rows leaving R/H are offered to F, and a shift never takes
   k_F below G·q_min. Theorem 1 relies on both.

## Files
prop1.md, prop1_sim.py, prop1_sim_results.json, REPORT.md, SHA256SUMS.
