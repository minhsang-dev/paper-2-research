"""ThreeBin: the three-compartment context policy of outline v14.3, Nội dung 2 (RQ2), for the Bài 1 replay harness.

Spec source: context/outline_v14_3_RQ_excerpt.txt (bundle bai2_handoff), "Nội dung 2", formula (2-4), Mệnh đề 1 and
"Quy tắc điều chỉnh tỷ lệ ba ngăn". Every point the outline leaves open is a named constructor argument and is listed in
DESIGN_CHOICES.md (decision ids DC-xx in the comments below). All defaults are PLACEHOLDERS awaiting approval and the
pre-registered grid search; nothing here was tuned.

Interface: a `policies.Policy` subclass, driven by `replay.run_replay` unchanged. The locked harness calls the hook
`query_entropy(model, context, rows)` for any policy that defines it (replay.py, "score first, learn after") and stores
the returned value in `Rows.h`. ThreeBin uses that hook under the same name to return the PREDICTION MARGIN (top-1 minus
top-2 class probability of the model conditioned on the context of the query day); the harness stores it with the row
and hands it back when the label arrives, exactly as for CURE's entropy. Nothing is recomputed at arrival (DC-08).

Bins (disjoint; every context row is in exactly one):
  R  recent:          FIFO of the newest labelled rows;
  F  representative:  per-group quotas q_c (formula 2-4, largest-remainder rounding), k-center selection per group;
  H  hard:            rows whose stored margin is <= margin_max, k_H <= 0.1 k, evicted by age; never a row without a
                      stored margin; never a pseudo-label (only rows handed to `update`, i.e. expert labels).
Capacities (NCS decision D1): k_F(t) = max(round(w_F k), G q_min) and k_R(t) = k - k_H - k_F(t), so k_F >= G q_min as
G grows. During a shift, k_F(t) loses min(floor(delta k_F), k_F - G q_min) to k_R(t) (the delta cap).
The bins follow the axioms P0-P3 of T2 (bai2_cloud/T2/prop1.md) literally (NCS decision D4):
  P0  |R| <= k_R(t), |F| <= k_F(t), |H| <= k_H <= 0.1 k, k_F(t) >= G q_min, k_R(t) + k_F(t) + k_H = k;
  P1  quotas q_c(t) by (2-4) with largest-remainder rounding, sum = k_F(t);
  P2  a row of c leaves F only when |F_c| > q_c(t) or when a k-center swap replaces it with another row of c;
  P3  every new row enters R or H; a row leaving R (FIFO) or H (age) is offered to F_c and accepted when |F_c| < q_c(t)
      (otherwise it competes in the k-center selection of c, i.e. a P2 swap). Ungrouped rows (benign unless
      benign_group=True, unprotected groups) leaving R/H are dropped. Bins are disjoint.
Bin sizes are hard caps (no lending), so the context is below k until F fills (DC-11).

Per update (outline steps 1-5): (1) receive the labelled rows; (2) shift signals: BBSE prior shift (L1 between pi_hat_t
and pi_hat_{t-1}) and a two-sample test on P(X) between the unlabelled samples of days t and t-1; (3) if either fires,
move capacity from F to R for D updates (a new firing restarts the D-update hold, DC-07); (4) quotas by (2-4);
(5) eviction: R by FIFO, F by k-center inside each group, H by age.

Determinism: the only random draw is k-means++ seeding of the binary-mode clustering, from `seed`.
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

_PROTOCOL = Path(os.environ.get("BAI2_PROTOCOL_CODE",
                                Path(__file__).resolve().parents[2] / "bai2_handoff" / "protocol_code"))
if str(_PROTOCOL / "replay") not in sys.path:
    sys.path.insert(0, str(_PROTOCOL / "replay"))

from policies import Policy  # noqa: E402
from rows import Rows  # noqa: E402

HARD_CAP = 0.1   # k_H <= 0.1 k (outline)


def feature_map(X):
    """Space in which k-center and clustering distances are measured (DC-05): sign(x) * log1p(|x|), no fitted
    parameters, so it is identical on every day and every run."""
    X = np.asarray(X, np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def largest_remainder_quotas(groups, pi, k_F, q_min, gamma):
    """Formula (2-4): q_c = q_min + floor((k_F - G q_min) pi_c^gamma / sum pi^gamma), then the units lost to the floors
    go one each to the largest fractional parts (ties: smaller group id), so the quotas sum exactly to k_F.
    If every pi_c is 0 the shares are uniform. If k_F < G q_min every group gets q_min (floor overflow; the sum is then
    G q_min > k_F and the caller records it)."""
    groups = sorted(groups)
    G = len(groups)
    if G == 0:
        return {}
    base = int(k_F) - G * int(q_min)
    if base < 0:
        return {c: int(q_min) for c in groups}
    w = np.array([float(pi.get(c, 0.0)) for c in groups]) ** float(gamma)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(G)
    exact = base * w / w.sum()
    fl = np.floor(exact).astype(int)
    short = base - int(fl.sum())
    for i in sorted(range(G), key=lambda i: (-(exact[i] - fl[i]), groups[i]))[:short]:
        fl[i] += 1
    return {c: int(q_min) + int(f) for c, f in zip(groups, fl)}


def k_center(Z, q):
    """Greedy farthest-first k-center (Gonzalez 1985) selection of q of the rows of Z (rows in arrival order).
    Start: the row nearest to the mean; then repeatedly the row farthest from the selected set. Ties go to the earlier
    (older) row (np.argmin/argmax return the first). Returns sorted positions."""
    n = Z.shape[0]
    if q >= n:
        return np.arange(n)
    if q <= 0:
        return np.zeros(0, int)
    start = int(np.argmin(((Z - Z.mean(0)) ** 2).sum(1)))
    sel = [start]
    mind = ((Z - Z[start]) ** 2).sum(1)
    mind[start] = -1.0
    for _ in range(q - 1):
        j = int(np.argmax(mind))
        sel.append(j)
        mind = np.minimum(mind, ((Z - Z[j]) ** 2).sum(1))
        mind[sel] = -1.0
    return np.sort(np.asarray(sel))


def bbse_prior(y_true, y_pred, mu, n_classes=2):
    """Black-box shift estimation (Lipton et al. 2018): pi_hat = M^{-1} mu, M[i, j] = P(y_hat = i | y = j) from labelled
    rows with their stored query-time prediction, mu = distribution of predictions on today's queried (unlabelled at
    query time) rows. Clipped at 0 and renormalised. None if a class is missing or M is singular."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    M = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        m = y_true == j
        if not m.any():
            return None
        M[:, j] = np.bincount(y_pred[m], minlength=n_classes)[:n_classes] / m.sum()
    try:
        if abs(np.linalg.det(M)) < 1e-9:
            return None
        pi = np.linalg.solve(M, np.asarray(mu, float))
    except np.linalg.LinAlgError:
        return None
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum() if pi.sum() > 0 else None


def ks_pvalue(A, B):
    """Two-sample test on P(X) (DC-03): per-feature two-sample Kolmogorov-Smirnov (asymptotic p), Bonferroni over the
    features: p = min(1, f * min_j p_j)."""
    if A is None or B is None or len(A) < 2 or len(B) < 2:
        return np.nan
    f = A.shape[1]
    ps = [ks_2samp(A[:, j], B[:, j], method="asymp").pvalue for j in range(f)]
    return float(min(1.0, f * np.nanmin(ps))) if ps else np.nan


class ThreeBin(Policy):
    name = "ThreeBin"

    def __init__(self, k, seed=0, *, q_min, w=(0.5, 0.4, 0.1), gamma=0.5, margin_max=0.2,
                 kappa_pi=0.1, kappa_x=0.01, delta=0.5, D=3, group_by="family", n_clusters=8, benign_group=False,
                 prior_window=7, bbse_window=7):
        super().__init__(k, seed)
        self.seed = int(seed)
        wR, wF, wH = (float(v) for v in w)
        if min(wR, wF, wH) < 0 or abs(wR + wF + wH - 1.0) > 1e-9:
            raise ValueError("w must be non-negative and sum to 1")
        if wH > HARD_CAP + 1e-12:
            raise ValueError("the hard bin is capped at 0.1 k")
        if group_by not in ("family", "cluster"):
            raise ValueError("group_by is 'family' (attack-family labels) or 'cluster' (binary labels)")
        if not 0.0 <= float(delta) < 1.0:
            raise ValueError("delta must be in [0, 1)")
        if q_min is None or int(q_min) < 0:
            raise ValueError("q_min must be given (a non-negative integer); it is set by T2 / the grid")
        self.k_H = int(np.floor(wH * self.k))                  # DC-01 rounding
        self.k_F_base = int(round(wF * self.k))                # w_F k, before D1
        self.q_min, self.gamma, self.margin_max = int(q_min), float(gamma), float(margin_max)
        self.kappa_pi, self.kappa_x, self.delta, self.D = float(kappa_pi), float(kappa_x), float(delta), int(D)
        # Admission limit (DC-12): the floors must fit next to H, G q_min <= k - k_H. A group beyond it is
        # 'unprotected': no quota, its rows live in R / H only. Not reached with realistic settings.
        self.G_max = (self.k - self.k_H) // self.q_min if self.q_min > 0 else 10 ** 9
        if self.G_max < 1:
            raise ValueError("k - k_H < q_min: not even one group can be protected")
        self.k_F, self.k_R, self.moved = self.capacities(0, False)
        self.group_by, self.n_clusters, self.benign_group = group_by, int(n_clusters), bool(benign_group)
        self.prior_window, self.bbse_window = int(prior_window), int(bbse_window)
        # row store: seq -> (X, y, fam, rid, h); seq = arrival order (unique even if a rid is delivered twice)
        self._X, self._y, self._fam, self._rid, self._h, self._grp = {}, {}, {}, {}, {}, {}
        self._seq = 0
        self.R = deque()          # seqs, oldest first
        self.H = deque()          # seqs, oldest first
        self.F = {}               # group -> sorted list of seqs
        self.groups = set()       # protected groups (have an F quota), admitted in arrival order (at most G_max)
        self.unprotected = set()  # groups refused by the admission rule (DC-12)
        self.reached_qmin = set() # groups that have had >= q_min rows delivered (Proposition 1(c) premise)
        self._delivered = {}      # group -> rows delivered
        self.day = 0              # number of update calls so far
        self._trigger_left = 0
        self._arrivals = deque()  # per update: (y array, pred array (-1 = none), group list)
        self._qpred = {}          # rid -> class predicted at query time (for the BBSE confusion matrix)
        self._mu_today = None     # prediction distribution of today's queried rows (set by the hook)
        self._pi_prev = None
        self._unlab_prev = None
        # binary mode clustering state (DC-04)
        self._cent, self._cnt = None, None
        self.log = []

    def capacities(self, G, active):
        """(k_F(t), k_R(t), moved). D1: k_F = max(round(w_F k), G q_min), k_R takes the difference. Shift (step 3):
        moved = min(floor(delta k_F), k_F - G q_min) (the delta cap), so k_F(t) >= G q_min also during a shift."""
        k_F = max(self.k_F_base, G * self.q_min)
        moved = min(int(np.floor(self.delta * k_F)), k_F - G * self.q_min) if active else 0
        k_F -= moved
        return k_F, self.k - self.k_H - k_F, moved

    # ------------------------------------------------------------------ harness hook
    def query_entropy(self, model, context, rows):
        """Hook called by replay.run_replay on the query day (name fixed by the locked harness). Returns the prediction
        margin top1 - top2 of `model` conditioned on `context`, which the harness stores in Rows.h; NaN = no prediction
        (context with < 2 rows or < 2 classes). Also records the predicted classes for BBSE."""
        n = len(rows)
        out = np.full(n, np.nan)
        self._mu_today = None
        if n == 0 or len(context) < 2 or len(np.unique(context.y)) < 2:
            return out
        P = np.asarray(model.predict_proba(context, rows.X), float)
        if P.ndim != 2 or P.shape[1] < 2:
            return out
        s = np.sort(P, axis=1)
        out = s[:, -1] - s[:, -2]
        pred = P.argmax(1)
        self._qpred.update(zip(rows.rid.tolist(), pred.tolist()))
        self._mu_today = np.bincount(pred, minlength=2)[:2] / n
        return out

    # ------------------------------------------------------------------ helpers
    def _rows(self, seqs, nf):
        seqs = sorted(seqs)
        if not seqs:
            r = Rows.empty(nf)
            r.h = np.zeros(0)
            return r
        return Rows(np.stack([self._X[s] for s in seqs]).astype(np.float32),
                    np.asarray([self._y[s] for s in seqs], np.uint8), np.asarray([self._fam[s] for s in seqs], np.int16),
                    np.asarray([self._rid[s] for s in seqs], np.int64), np.asarray([self._h[s] for s in seqs], float))

    def all_seqs(self):
        return list(self.R) + list(self.H) + [s for v in self.F.values() for s in v]

    def counts_by_group(self):
        """N_c over the whole context (all three bins)."""
        out = {}
        for s in self.all_seqs():
            g = self._grp.get(s)
            if g is not None:
                out[g] = out.get(g, 0) + 1
        return out

    def _assign_clusters(self, seqs):
        """Binary mode (DC-04): attack rows are grouped by online k-means (MacQueen 1967) in feature_map space,
        K = n_clusters, centroids seeded by k-means++ (random_state = seed) on the first K attack rows; a row's group is
        fixed when it is assigned. Before seeding, attack rows wait unassigned (group None). Returns the seqs assigned
        now (new rows and, at seeding, waiting rows already held in R / H)."""
        from sklearn.cluster import kmeans_plusplus
        att = [s for s in seqs if self._y[s] == 1 and self._grp.get(s) is None]
        if self._cent is None:
            waiting = sorted([s for s in self.all_seqs() if self._y[s] == 1 and self._grp.get(s) is None] + att)
            waiting = list(dict.fromkeys(waiting))
            if len(waiting) < self.n_clusters:
                return []
            Z = feature_map(np.stack([self._X[s] for s in waiting[:self.n_clusters]]))
            self._cent, _ = kmeans_plusplus(Z, self.n_clusters, random_state=self.seed)
            self._cnt = np.zeros(self.n_clusters)
            att = waiting
        for s in att:
            z = feature_map(self._X[s][None, :])[0]
            j = int(np.argmin(((self._cent - z) ** 2).sum(1)))
            self._cnt[j] += 1
            self._cent[j] += (z - self._cent[j]) / self._cnt[j]
            self._grp[s] = j + 1
        return att

    def _priors(self):
        """pi_hat_c for (2-4) (DC-02): share of group c among the grouped labelled rows that arrived in the last
        `prior_window` updates (groups with no such row: 0)."""
        cnt = {}
        for _, _, grp in list(self._arrivals)[-self.prior_window:]:
            for g in grp:
                if g is not None:
                    cnt[g] = cnt.get(g, 0) + 1
        tot = sum(cnt.values())
        return {g: c / tot for g, c in cnt.items()} if tot else {}

    def _bbse_signal(self):
        """L1(pi_hat_t, pi_hat_{t-1}) (DC-06), or NaN when either estimate is unavailable."""
        pi = None
        if self._mu_today is not None:
            ys, ps = [], []
            for y, p, _ in list(self._arrivals)[-self.bbse_window:]:
                m = p >= 0
                ys.append(y[m])
                ps.append(p[m])
            if ys:
                pi = bbse_prior(np.concatenate(ys), np.concatenate(ps), self._mu_today)
        prev, self._pi_prev = self._pi_prev, pi
        if pi is None or prev is None:
            return pi, np.nan
        return pi, float(np.abs(pi - prev).sum())

    # ------------------------------------------------------------------ update
    def update(self, context, new_labelled_rows, unlabelled_stream):
        t0 = time.perf_counter()
        nf = new_labelled_rows.X.shape[1]
        mine = sorted(self.all_seqs())
        if len(context) != len(mine) or not np.array_equal(np.sort(context.rid),
                                                           np.sort([self._rid[s] for s in mine]).astype(np.int64)):
            raise ValueError("the context handed to ThreeBin differs from ThreeBin's own state")
        # (1) receive rows
        n = len(new_labelled_rows)
        h = np.full(n, np.nan) if new_labelled_rows.h is None else np.asarray(new_labelled_rows.h, float)
        new = []
        for j in range(n):
            s = self._seq
            self._seq += 1
            self._X[s] = np.asarray(new_labelled_rows.X[j], np.float32).copy()
            self._y[s], self._fam[s] = int(new_labelled_rows.y[j]), int(new_labelled_rows.fam[j])
            self._rid[s], self._h[s] = int(new_labelled_rows.rid[j]), float(h[j])
            if self._y[s] == 0:
                self._grp[s] = 0 if self.benign_group else None
            elif self.group_by == "family":
                self._grp[s] = self._fam[s]
            else:
                self._grp[s] = None
            new.append(s)
        tc = time.perf_counter()
        grouped_now = list(new)
        if self.group_by == "cluster":
            grouped_now = sorted(set(new) | set(self._assign_clusters(new)))
        cluster_seconds = time.perf_counter() - tc
        for s in grouped_now:
            g = self._grp.get(s)
            if g is not None and g not in self.groups:
                if g in self.unprotected or len(self.groups) >= self.G_max:
                    self.unprotected.add(g)
                    self._grp[s] = None      # unprotected group: no F candidacy
                    g = None
                else:
                    self.groups.add(g)
            if g is not None:
                self._delivered[g] = self._delivered.get(g, 0) + 1
                if self._delivered[g] >= self.q_min:
                    self.reached_qmin.add(g)
        preds = np.asarray([self._qpred.get(self._rid[s], -1) for s in new], int)
        self._arrivals.append((np.asarray([self._y[s] for s in new], int), preds, [self._grp.get(s) for s in new]))
        while len(self._arrivals) > max(self.prior_window, self.bbse_window):
            self._arrivals.popleft()
        # (2) shift signals
        pi_hat, l1 = self._bbse_signal()
        self._mu_today = None
        U = None if unlabelled_stream is None else np.asarray(unlabelled_stream, np.float64)
        p_x = ks_pvalue(U, self._unlab_prev)
        if U is not None:
            self._unlab_prev = U
        fired = bool((np.isfinite(l1) and l1 > self.kappa_pi) or (np.isfinite(p_x) and p_x < self.kappa_x))
        # (3) capacity move
        if fired:
            self._trigger_left = self.D
        active = self._trigger_left > 0
        G = len(self.groups)
        self.k_F, self.k_R, self.moved = self.capacities(G, active)
        k_F_eff = self.k_F
        assert k_F_eff >= G * self.q_min and self.k_R >= 0
        # (4) quotas
        quotas = largest_remainder_quotas(self.groups, self._priors(), k_F_eff, self.q_min, self.gamma)
        # (5) admission and eviction
        for s in new:
            hs = self._h[s]
            if self.k_H > 0 and np.isfinite(hs) and hs <= self.margin_max:
                self.H.append(s)
            else:
                self.R.append(s)
        offered = []
        while len(self.H) > self.k_H:                     # hard bin: by age
            offered.append(self.H.popleft())
        while len(self.R) > self.k_R:                     # recent bin: FIFO
            offered.append(self.R.popleft())
        pool = {g: list(v) for g, v in self.F.items()}
        offered_log = [(s, self._grp.get(s)) for s in offered]
        for s in offered:                                 # P3: offer to F
            g = self._grp.get(s)
            if g is not None:
                pool.setdefault(g, []).append(s)
        self.F = {}
        for g, v in pool.items():                         # representative bin: k-center inside each group
            v = sorted(v)
            q = quotas.get(g, 0)
            if len(v) > q:
                keep = k_center(feature_map(np.stack([self._X[s] for s in v])), q)
                v = [v[i] for i in keep]
            if v:
                self.F[g] = v
        keep_set = set(self.all_seqs())
        for s in [s for s in self._X if s not in keep_set]:
            for d in (self._X, self._y, self._fam, self._rid, self._h):
                del d[s]
            self._grp.pop(s, None)
        out = self._rows(self.all_seqs(), nf)
        self.log.append({
            "day": self.day, "n_new": n, "n_R": len(self.R), "n_F": sum(len(v) for v in self.F.values()),
            "n_H": len(self.H), "k_R_eff": self.k_R, "k_F_eff": k_F_eff, "moved": self.moved, "k_H": self.k_H, "G": G,
            "offered": offered_log,
            "quotas": dict(quotas), "quota_sum": sum(quotas.values()), "unprotected": sorted(self.unprotected),
            "pi_hat_bbse": None if pi_hat is None else pi_hat.tolist(), "bbse_l1": l1, "p_x": p_x, "fired": fired,
            "trigger_active": active, "N_c": self.counts_by_group(), "cluster_seconds": cluster_seconds,
            "update_seconds": time.perf_counter() - t0})
        if active:
            self._trigger_left -= 1
        self.day += 1
        return out
