# T3 report: ThreeBin policy for the Bài 1 replay harness

## What was done
- `threebin.py`: `ThreeBin`, a `Policy` subclass driven by `run_replay` unchanged. R = FIFO; F = quotas (2-4) with
  largest-remainder rounding and k-center per group (families, or online k-means clusters in binary mode); H = rows
  with small stored query-time margin (via the harness hook `query_entropy`, as for CURE), k_H ≤ 0.1k, evicted by age.
  Trigger: BBSE L1 or KS test on P(X) moves ⌊δ·k_F⌋ from F to R for D = 3 updates.
- Your added requirements: a group admission rule keeps k_F ≥ G·q_min and δ·k_F ≤ k_F − G·q_min at all times (DC-12).
- `tests/test_threebin.py`: 25 pytest tests, synthetic only, all pass (`test_output.txt`), covering every listed
  property, including I1–I3 through `run_replay`.
- `run_absence_demo.py` → `absence_demo_results.json`: family 5 absent on days 5–9, k = 1,000, 3 seeds.

## Demo result (from `absence_demo_results.json`)
Rows of family 5 in the context after day 9: FIFO 0, DualFIFO 0, CBRS-family 11 / 7 / 9, ThreeBin 11 / 7 / 9 (seeds 0/1/2).
With fewer than q_min = 20 rows, ThreeBin kept all of them, matching CBRS-family. Return-day recall of family 5 was 1.0
for all four policies: family 5 shares feature signatures with other families, so the demo shows retention but **no
recall effect**. The stream was not changed after seeing results.

## Open questions / decisions for the NCS
1. Approve DC-01…DC-17 in `DESIGN_CHOICES.md`, mainly DC-02 (π̂_c from labels; a binary model cannot give family BBSE),
   DC-03 (KS + Bonferroni, very sensitive at 5,000 rows), DC-04 (binary-mode clustering), DC-06 (BBSE confusion matrix
   from the policy's own labelled rows; D_cal is off-limits by I2), DC-12 (refuse groups beyond G_max), DC-13 (reading
   of "R shrinks by at most G·q_min").
2. All parameter values are placeholders for the grid search; q_min comes from T2.
3. Source flags do not exist in the harness yet, so the rule "H accepts only unflagged sources" waits for RQ4 (DC-09).

## Reproduce
Unzip `bai2_handoff.zip` at the repository root (or set `BAI2_PROTOCOL_CODE`), then in `bai2_cloud/T3`:
`python -m pytest -q tests` and `python run_absence_demo.py`. Tested with Python 3.13.17, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1, lightgbm 4.7.0, pytest 9.2.0. `protocol_code/` was not modified; its SHA256SUMS still verify.
