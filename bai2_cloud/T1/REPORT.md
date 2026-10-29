# T1 REPORT — novelty and positioning audit (RQ2)

**Verdict: REVISE.** Details are in `novelty_audit.md`.

## What was done
- I read the handoff README, the outline excerpt, `cure_spec.md` and the Bài 1 manuscript.
- I searched the literature on TFM stream context, quota and class-balanced memories, retrieval ICL, drift-triggered memory and security CL. Each claim is marked opened, abstract-only, or [VERIFY].
- `check_numbers.py` reproduces the computed numbers. No dataset was used, and `protocol_code/` is untouched.

## Main findings
1. Every component has a close precedent. The tempered allocation in Eq. (2-4) is PRS's Eq. 1. The q_min floor is ECBRS's per-class threshold (NeurIPS 2023, NIDS). The per-family plus anomalous replay is MADAR (malware). The recent bank is the DualFIFO/CURE short bank. The hard bank is a variant of CURE's entropy gate.
2. Prop. 1(c) also holds for CBRS-family, GDumb and ECBRS. It separates the policy only from FIFO, reservoir and binary-class memories.
3. I found no prior work with a shift-triggered transfer of capacity between banks and no derivation of q_min. These, together with the measured benefit under budget, delay and filtered labels on both TFM and LightGBM, are the defensible contributions.
4. Verma et al. (2026) report that FIFO is as good as existing policies and that memory size matters most. A reviewer will cite this.

## Minimal changes (outline unchanged)
- R1: name the base methods (DualFIFO/CURE, PRS, ECBRS/GDumb, k-center, MADAR) and present Prop. 1(a)–(c) as lemmas.
- R2: claim only q_min*, the trigger (if a placebo/oracle ablation shows a benefit), the cluster fallback, and the measured benefit.
- R3: add comparators CBRS-family (already in the harness), PRS-family/ECBRS and a compute-matched FIFO.
- R4: add a memory-size sweep.
- R5: credit DualFIFO to Lourenço et al. [13].

## Decisions needed from the NCS
1. Make **CBRS-family**, not binary CBRS, the "class-balanced buffer" in H2. This feeds T5's preregistration.
2. Decide which added comparators to fund (PRS-family/ECBRS, MADAR-style, reservoir, FIFO size sweep). Bảng 2.2 assumes 7 policies, so each added policy raises the RQ2 run count. One option is to run the extras with 3 seeds or on LightGBM only.
3. Confirm the [VERIFY] items before citing them (listed in the audit).

## Open questions for T2/T3
- Prop. 1(c) needs k_F ≥ G·q_min at all times while G grows.
- The capacity shift needs δ·k_F ≤ k_F − G·q_min.
- T3's 5-day-absence demo should also run against CBRS-family.
