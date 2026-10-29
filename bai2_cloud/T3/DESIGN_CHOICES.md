# ThreeBin: design choices to approve (T3)

The outline (v14.3, Nội dung 2, in `context/outline_v14_3_RQ_excerpt.txt`) fixes the three bins, formula (2-4), the
k_H ≤ 0.1k cap, the eviction order, the two shift signals and D = 3. Everything below is left open by the outline (or
added at the NCS's request) and is implemented as a constructor argument or a documented rule in `threebin.py`.
**Every default is a placeholder.** None was tuned; the parameter values are to be set by the pre-registered grid search
on development data (outline: "Mọi tham số được chọn bằng tìm kiếm lưới …"), and q_min by T2.

Each item: the decision, the default implemented, and the alternatives. Mark ✔ / ✘ / change.

## A. Parameters (placeholders, to be set by grid search / T2)

| Id | Parameter | Default | Note |
|---|---|---|---|
| DC-15a | w_R : w_F : w_H | 0.5 : 0.4 : 0.1 | w_H > 0.1 is rejected |
| DC-15b | q_min | 20 | T2 (Mệnh đề 1) chooses the optimum |
| DC-15c | γ | 0.5 | γ ∈ (0, 1] |
| DC-15d | δ | 0.5 | must be in [0, 1); see DC-12 for the bound δ·k_F ≤ k_F − G·q_min |
| DC-15e | κ_π (BBSE L1) | 0.1 | |
| DC-15f | κ_X (p-value) | 0.01 | on the Bonferroni-corrected p of DC-03 |
| DC-15g | margin threshold for H | 0.2 | see DC-09 |
| DC-15h | D | 3 | fixed by the outline |
| DC-15i | windows for π̂_c and for BBSE's confusion matrix | 7 updates each | DC-02, DC-06 |
| DC-15j | K (clusters, binary mode) | 8 | DC-04 |

## B. Open points of the specification

**DC-01 Bin rounding.** k_H = ⌊w_H·k⌋, k_F = round(w_F·k), k_R = k − k_F − k_H. With k = 2,000 and the defaults:
1,000 / 800 / 200.

**DC-02 Source of π̂_c in (2-4).** The model is binary, so BBSE cannot estimate family shares. Default: π̂_c = share of
group c among the grouped labelled rows that arrived in the last 7 updates (labels are a uniform sample of the day, so
this is unbiased up to the label delay τ). A group absent from the window gets π̂_c = 0, hence quota q_min. If no
grouped row is in the window, the shares are uniform. *Alternatives:* cumulative shares since the start; exponential
decay.

**DC-03 Two-sample test on P(X).** Default: per-feature two-sample Kolmogorov–Smirnov (asymptotic p) between the
unlabelled sample the harness passes on day t and the one of day t − 1, Bonferroni over features:
p = min(1, f · min_j p_j); fires if p < κ_X. Deterministic and cheap. *Warning:* with 5,000 rows per day KS is very
powerful, so small daily differences fire; κ_X must come from the grid. *Alternatives:* MMD with a permutation test
(seeded); classifier two-sample test; a fixed sub-sample size.

**DC-04 Groups in binary-label mode.** Default: online k-means (MacQueen) on attack rows only, K = 8, in the space of
DC-05; centroids seeded by k-means++ with `random_state = seed` on the first K attack rows; each row's cluster is fixed
when it is assigned (so a group's identity never changes and Mệnh đề 1(c) applies to it). Attack rows that arrive
before K attack rows exist wait unassigned in R/H; any that leave R before seeding are dropped. The clustering time
is logged per update (`cluster_seconds`), so the cost of the two modes can be reported separately as the outline asks.
*Alternatives:* batch k-means refit daily with Hungarian matching of cluster ids; DP-means / leader clustering with a
radius (new clusters for new families); HDBSCAN.

**DC-05 Distance space** for k-center and clustering: sign(x)·log1p(|x|), no fitted scaling (identical across days and
runs). *Alternatives:* z-score with statistics frozen on day 0; z-score within each group's pool.

**DC-06 BBSE details.** π̂_t = M⁻¹ μ̂_t, clipped at 0 and renormalised, where μ̂_t is the distribution of predicted
classes (argmax) of the rows queried on day t, scored at query time by the model conditioned on that day's context
(the same call that gives the margin, so no extra model call), and M[i, j] = P(ŷ = i | y = j) is estimated from the
labelled rows that arrived in the last 7 updates, using each row's stored query-time prediction. The policy never sees
D_cal (isolation I2), so D_cal cannot be the confusion-matrix source. The predictions in M come from different daily
contexts, so M is an average over models. Signal = ‖π̂_t − π̂_{t−1}‖₁; NaN (no firing) when either estimate is missing
or M is singular. Relies on the replay querying labels uniformly at random (true in L5). *Alternatives:* M from the
most recent day only; regularised BBSE (RLLS).

**DC-07 Re-firing during a hold.** A new firing restarts the D-update hold; the moved capacity does not stack.
D is counted in `update` calls (the replay calls `update` once per day, also on days with no arrivals). *Alternative:*
ignore firings while a hold is active.

**DC-08 Margin.** Margin = p_top1 − p_top2 of `model.predict_proba(context, rows)` at query time, returned through the
harness hook `query_entropy` (the locked replay calls this hook by name; it stores the value in `Rows.h` and passes it
back at arrival, exactly as for CURE). NaN when there is no prediction (context with < 2 rows or < 2 classes). Nothing
is recomputed at arrival.

**DC-09 Hard-bin admission.** A row enters H if its stored margin is finite and ≤ 0.2 (DC-15g); H keeps the newest k_H
such rows (evicted by age). Rows without a margin never enter H. No pseudo-labels: only rows given to `update` (expert
labels) are stored. The harness has no source-flag field yet, so "từ nguồn không bị gắn cờ" is not enforceable in this
phase; RQ4 must add a flag and H must then reject flagged rows. *Alternatives:* each day admit the m lowest-margin rows;
a quantile threshold.

**DC-10 Row flow.** Bins are disjoint. A new row goes to H if hard (DC-09), otherwise R. A row leaving R (FIFO) or H (age)
becomes a candidate of its group's F pool; F keeps q_c rows per group by k-center. Benign rows (and rows of unprotected
groups, DC-12) leaving R/H are dropped. *Alternative:* F selects from all rows ever seen of the group (rows may then sit
in R and F at once, deduplicated by rid).

**DC-11 Work-conserving budget.** Capacity unused by F or H is used by R, so the context reaches k as soon as there are
k rows. *Alternative:* fixed bin sizes (context smaller than k while F/H are not full).

**DC-12 Group admission rule (NCS requirement 1 and 2).** G_max = ⌊(k_F − ⌊δ·k_F⌋) / q_min⌋. Groups are admitted in order
of first arrival until G = G_max; this keeps k_F ≥ G·q_min **and** ⌊δ·k_F⌋ ≤ k_F − G·q_min at all times, so a trigger
never pushes a group below q_min. A later group is *unprotected*: no quota, its rows live in R/H only and are dropped
when they leave; it is listed in the log (`unprotected`). The constructor rejects settings with G_max < 1. With k = 2,000
and defaults, G_max = (800 − 400) / 20 = 20 (NF-CICIDS2018-v3 has 14 attack families per `cache_meta/…/meta.json`).
*Alternatives:* (a) grow k_F into k_R when a new group arrives; (b) reduce the moved capacity instead of refusing groups;
(c) evict the group with the oldest last arrival to make room.

**DC-13 Reading of "the recent bin shrinks by at most G·q_min".** Default reading: the recency lost to the
representative bin is (k − |H|) − |R| = |F| ≤ k_F_eff; it reaches its minimum G·q_min when all movable capacity is
moved (δ·k_F = k_F − G·q_min). Tested in that configuration. Please confirm this is the intended statement of
Mệnh đề 1(c).

**DC-14 Benign as a group.** Default: no (groups are attack families / attack clusters, as the outline says); benign
lives in R and H. `benign_group=True` makes benign group 0 with its own quota.

**DC-16 k-center details.** Greedy farthest-first (Gonzalez); first centre = row nearest the pool mean; ties to the older
row; re-run on the group's pool whenever the pool exceeds the quota (also when a quota shrinks).

**DC-17 Repeated rows.** A rid delivered twice (possible only with resampled synthetic days) is stored twice, as FIFO does.
