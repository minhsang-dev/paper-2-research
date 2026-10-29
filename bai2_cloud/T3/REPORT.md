# T3 report: ThreeBin policy for the Bài 1 replay harness

## What was done
- `threebin.py`: `ThreeBin`, a `Policy` subclass driven by `run_replay` unchanged. R = FIFO; F = quotas (2-4) with
  largest-remainder rounding and k-center per group (families, or online k-means clusters in binary mode); H = rows
  with small stored query-time margin (via the harness hook `query_entropy`, as for CURE), k_H ≤ 0.1k, evicted by age.
  Trigger: BBSE L1 or KS test on P(X) moves capacity from F to R for D = 3 updates.
- NCS decisions implemented: **D1** k_F(t) = max(w_F·k, G·q_min), k_R takes the difference; **D4** axioms P0–P3 of T2
  (`bai2_cloud/T2/prop1.md`) literally, with one test each, and the δ cap (moved = min(⌊δ·k_F⌋, k_F − G·q_min)).
  **q_min** is a required parameter with no default.
- P0 makes bin sizes hard caps, so R no longer borrows unused F/H capacity. The context is below k until F fills (DC-11).
- `tests/test_threebin.py`: 32 pytest tests, synthetic only, all pass (`test_output.txt`), covering every listed
  property, including I1–I3 through `run_replay`.
- `run_absence_demo.py` → `absence_demo_results.json`: family 5 absent on days 5–9, k = 1,000, 3 seeds, q_min = 20
  (illustrative).

## Demo result (from `absence_demo_results.json`)
Rows of family 5 in the context after day 9: FIFO 0, DualFIFO 0, CBRS-family 11 / 7 / 9, ThreeBin 11 / 7 / 9 (seeds 0/1/2).
With fewer than q_min rows, ThreeBin kept all of them, matching CBRS-family. Return-day recall of family 5 was 1.0
for all four policies: family 5 shares feature signatures with other families, so the demo shows retention but **no
recall effect**. The stream was not changed after seeing results.

## Open questions / decisions for the NCS
1. Approve DC-01…DC-17 in `DESIGN_CHOICES.md`, mainly DC-02 (π̂_c from labels), DC-03 (KS + Bonferroni, very
   sensitive at 5,000 rows), DC-04 (binary-mode clustering; Theorem 1 is open there per T2), DC-06 (BBSE confusion
   matrix from the policy's own labelled rows), DC-10 (an offered row may swap into a full F_c by k-center, allowed by
   P2(ii); or reject it?).
2. All other parameter values are placeholders for the grid search.
3. Source flags do not exist in the harness yet, so the rule "H accepts only unflagged sources" waits for RQ4 (DC-09).

## Reproduce
Unzip `bai2_handoff.zip` at the repository root (or set `BAI2_PROTOCOL_CODE`), then in `bai2_cloud/T3`:
`python -m pytest -q tests` and `python run_absence_demo.py`. Tested with Python 3.13.17, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1, lightgbm 4.7.0, pytest 9.2.0. `protocol_code/` was not modified; its SHA256SUMS still verify.
