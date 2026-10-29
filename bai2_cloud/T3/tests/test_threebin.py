"""Unit tests for ThreeBin, synthetic data only. Run from bai2_cloud/T3: python -m pytest -q tests"""
import numpy as np
import pytest

from models import LightGBMModel
from policies import FIFO
from replay import run_replay
from rows import Rows
from synth import N_FEATURES, SynthStream, absence_probs, recording
from threebin import ThreeBin, bbse_prior, k_center, largest_remainder_quotas

FAST = dict(n_estimators=5, verbosity=-1)


def batch(rng, n, fam_p, start, h=None, shift=0.0):
    """n labelled rows with families drawn from fam_p (index 0 benign); rid = start + i."""
    p = np.asarray(fam_p, float)
    fam = rng.choice(len(p), size=n, p=p / p.sum()).astype(np.int16)
    X = rng.normal(0, 1, (n, N_FEATURES)) + shift
    for f in np.unique(fam[fam > 0]):
        X[fam == f, f % N_FEATURES] += 3.0
    r = Rows(X.astype(np.float32), (fam > 0).astype(np.uint8), fam, np.arange(start, start + n, dtype=np.int64))
    r.h = rng.uniform(0, 1, n) if h is None else h
    return r


def drive(policy, days, rng, fam_p_of, n_per_day=300, shift_of=lambda d: 0.0, h_nan_share=0.0):
    """Direct update loop (no replay); yields (day, context) after every update."""
    ctx = Rows.empty(N_FEATURES)
    start = 0
    for d in range(days):
        n = n_per_day(d) if callable(n_per_day) else n_per_day
        b = batch(rng, n, fam_p_of(d), start, shift=shift_of(d))
        if h_nan_share:
            b.h[rng.uniform(size=n) < h_nan_share] = np.nan
        start += n
        U = rng.normal(0, 1, (500, N_FEATURES)) + shift_of(d)
        ctx = policy.update(ctx, b, U)
        yield d, ctx


def bins_of(p):
    return set(p.R), set(p.H), {s for v in p.F.values() for s in v}


# ---------------------------------------------------------------- size and quotas
@pytest.mark.parametrize("group_by", ["family", "cluster"])
def test_context_never_exceeds_k_and_bins_are_disjoint(group_by):
    rng = np.random.default_rng(0)
    p = ThreeBin(k=500, q_min=10, group_by=group_by, kappa_x=0.05)
    for d, ctx in drive(p, 25, rng, lambda d: [0.7, 0.1, 0.1, 0.05, 0.05], n_per_day=lambda d: 50 + 40 * (d % 7)):
        assert len(ctx) <= p.k
        R, H, F = bins_of(p)
        assert not (R & H) and not (R & F) and not (H & F)
        assert len(R) + len(H) + len(F) == len(ctx)


def test_quota_formula_sums_exactly_and_respects_floor():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        G = int(rng.integers(1, 15))
        q_min = int(rng.integers(0, 30))
        k_F = G * q_min + int(rng.integers(0, 3000))
        pi = {c: float(v) for c, v in enumerate(rng.dirichlet(np.ones(G)) * (rng.uniform(size=G) > 0.3))}
        q = largest_remainder_quotas(list(range(G)), pi, k_F, q_min, float(rng.uniform(0.05, 1.0)))
        assert sum(q.values()) == k_F
        assert min(q.values()) >= q_min


def test_quota_formula_hand_example():
    # k_F = 100, q_min = 10, G = 3, pi = (0.5, 0.3, 0.2), gamma = 1: base 70 -> 35, 21, 14 (exact), sum 100.
    assert largest_remainder_quotas([1, 2, 3], {1: 0.5, 2: 0.3, 3: 0.2}, 100, 10, 1.0) == {1: 45, 2: 31, 3: 24}
    # remainder: base 10, pi = 1/3 each -> 3.33 each, one unit left -> smallest group id
    assert largest_remainder_quotas([7, 4, 9], {7: 1, 4: 1, 9: 1}, 10 + 30, 10, 1.0) == {4: 14, 7: 13, 9: 13}


def test_quotas_sum_to_kF_every_update():
    rng = np.random.default_rng(2)
    p = ThreeBin(k=600, q_min=15, kappa_x=0.05)
    for d, _ in drive(p, 20, rng, lambda d: [0.6, 0.2, 0.1, 0.05 if d < 8 else 0.0, 0.05],
                      shift_of=lambda d: 2.0 if d in (6, 13) else 0.0):
        log = p.log[-1]
        assert log["quota_sum"] == log["k_F_eff"]
        assert log["k_F_eff"] == (p.k_F - p.moved if log["trigger_active"] else p.k_F)
    assert any(l["trigger_active"] for l in p.log) and not all(l["trigger_active"] for l in p.log)


# ---------------------------------------------------------------- Proposition 1(c)
@pytest.mark.parametrize("group_by", ["family", "cluster"])
def test_groups_that_reached_qmin_keep_qmin_after_long_absence(group_by):
    rng = np.random.default_rng(3)
    p = ThreeBin(k=400, q_min=12, group_by=group_by, n_clusters=4, kappa_x=0.2, delta=0.5)
    # family 4 present only on days 0-2, then absent 60 days; heavy traffic of others; X drift every 5 days fires
    for d, ctx in drive(p, 63, rng, lambda d: [0.5, 0.2, 0.2, 0.0, 0.1] if d >= 3 else [0.4, 0.2, 0.1, 0.1, 0.2],
                        n_per_day=400, shift_of=lambda d: 1.5 * ((d // 5) % 2)):
        N = p.counts_by_group()
        for g in p.reached_qmin:
            assert N.get(g, 0) >= p.q_min, (d, g, N)
        if group_by == "family" and d >= 3:
            assert 4 in p.reached_qmin
            assert int((ctx.fam == 4).sum()) >= p.q_min
    assert any(l["fired"] for l in p.log)


def test_group_below_qmin_keeps_every_row_it_had():
    rng = np.random.default_rng(4)
    p = ThreeBin(k=300, q_min=20)
    ctx = Rows.empty(N_FEATURES)
    b = batch(rng, 200, [1, 0, 0], 0)
    b.fam[:5], b.y[:5] = 2, 1       # 5 rows of family 2, below q_min
    ctx = p.update(ctx, b, None)
    for d in range(1, 30):
        ctx = p.update(ctx, batch(rng, 300, [0.7, 0.3, 0.0], 1000 * d), None)
        assert int((ctx.fam == 2).sum()) == 5


def test_recent_bin_shrinks_by_at_most_G_qmin():
    # Prop. 1(c) cost, as read in DESIGN_CHOICES DC-13: the recency lost to the representative bin, (k - |H|) - |R|
    # = |F|, is at most k_F_eff, and with all movable capacity moved (delta k_F = k_F - G q_min) at most G q_min.
    G, q_min, k = 4, 20, 1000
    k_F = 400
    delta = 1 - G * q_min / k_F                                   # 0.8 -> moved = 320 = k_F - G q_min
    p = ThreeBin(k=k, w=(0.5, 0.4, 0.1), q_min=q_min, delta=delta, kappa_x=0.5)
    assert p.k_F == k_F and p.moved == k_F - G * q_min and p.G_max == G
    rng = np.random.default_rng(5)
    seen = 0
    for d, ctx in drive(p, 30, rng, lambda d: [0.5, 0.15, 0.15, 0.1, 0.1], n_per_day=500,
                        shift_of=lambda d: 3.0 if d in (10, 20) else 0.0):
        log = p.log[-1]
        assert len(ctx) - len(p.H) - len(p.R) == log["n_F"] <= log["k_F_eff"]
        if len(ctx) < k:
            continue                                              # context not yet full: nothing was evicted
        loss = (k - len(p.H)) - len(p.R)
        if log["trigger_active"]:
            assert loss <= len(p.groups) * q_min
            seen += 1
    assert seen >= 3


# ---------------------------------------------------------------- hard bin
def test_hard_bin_capped_and_only_rows_with_stored_margin():
    rng = np.random.default_rng(6)
    p = ThreeBin(k=500, q_min=10, margin_max=0.5)
    for d, ctx in drive(p, 20, rng, lambda d: [0.7, 0.2, 0.1], h_nan_share=0.4):
        assert len(p.H) <= 0.1 * p.k
        hs = np.array([p._h[s] for s in p.H])
        assert np.isfinite(hs).all() and (hs <= p.margin_max).all()
    assert len(p.H) == p.k_H


def test_hard_bin_above_cap_is_rejected():
    with pytest.raises(ValueError):
        ThreeBin(k=100, w=(0.5, 0.35, 0.15))


def test_hard_bin_via_replay_uses_query_time_margin():
    s = SynthStream(6, lambda d: [0.75, 0.15, 0.1], per_day=2000)
    p = recording(ThreeBin)(k=600, q_min=10, margin_max=0.3)
    run_replay(s, p, LightGBMModel(params=FAST), b=400, tau=1, a_fp=50, n_days=6, n_eval=300)
    assert len(p.H) <= 60
    for seq in p.H:
        assert np.isfinite(p._h[seq]) and p._h[seq] <= 0.3


# ---------------------------------------------------------------- trigger
def test_x_shift_trigger_moves_capacity_and_reverts_after_D_days():
    rng = np.random.default_rng(7)
    p = ThreeBin(k=800, q_min=10, kappa_pi=10.0, kappa_x=1e-6, delta=0.5, D=3)
    shift_day = 12
    for d, _ in drive(p, 22, rng, lambda d: [0.6, 0.2, 0.2], n_per_day=400,
                      shift_of=lambda d: 2.0 if d >= shift_day else 0.0):
        pass
    act = [l["trigger_active"] for l in p.log]
    fired = [l["fired"] for l in p.log]
    assert fired[shift_day] and sum(fired) == 1
    assert act == [shift_day <= d < shift_day + 3 for d in range(22)]
    for d in (shift_day, shift_day + 1, shift_day + 2):
        assert p.log[d]["k_F_eff"] == p.k_F - int(np.floor(0.5 * p.k_F))
        assert p.log[d]["n_F"] <= p.log[d]["k_F_eff"]
    assert p.log[shift_day + 3]["k_F_eff"] == p.k_F
    assert p.log[shift_day]["n_R"] > p.log[shift_day - 1]["n_R"]          # capacity went to R
    assert p.log[shift_day + 5]["n_F"] > p.log[shift_day + 1]["n_F"]       # and came back to F


def test_bbse_prior_hand_example():
    # M = [[0.9, 0.2], [0.1, 0.8]] (columns: true class); true target prior (0.6, 0.4) -> mu = M pi = (0.62, 0.38)
    y = np.array([0] * 10 + [1] * 10)
    yp = np.array([0] * 9 + [1] + [0] * 2 + [1] * 8)
    assert np.allclose(bbse_prior(y, yp, [0.62, 0.38]), [0.6, 0.4])
    assert bbse_prior(np.zeros(5), np.zeros(5), [1, 0]) is None


class FixedModel:
    """predict_proba depends only on feature 0 (no context use), for controlled BBSE signals."""
    def predict_proba(self, context, X):
        p1 = 1 / (1 + np.exp(-3 * X[:, 0]))
        return np.column_stack([1 - p1, p1])


def test_bbse_trigger_fires_on_prior_shift_and_not_without():
    rng = np.random.default_rng(8)
    p = ThreeBin(k=800, q_min=10, kappa_pi=0.15, kappa_x=0.0, D=3)
    ctx, start, m = Rows.empty(N_FEATURES), 0, FixedModel()
    for d in range(12):
        share = 0.1 if d < 8 else 0.5                    # attack prior jumps on day 8
        b = batch(rng, 400, [1 - share, share], start)
        b.X[:, 0] = np.where(b.y == 1, 1.0, -1.0) + rng.normal(0, 1, 400)
        b.h = p.query_entropy(m, Rows(b.X[:2], np.array([0, 1], np.uint8), b.fam[:2], b.rid[:2]), b)
        start += 400
        ctx = p.update(ctx, b, None)
    l1 = [l["bbse_l1"] for l in p.log]
    assert p.log[8]["fired"] and p.log[8]["trigger_active"]
    assert not any(p.log[d]["fired"] for d in range(2, 8))
    assert abs(p.log[9]["pi_hat_bbse"][1] - 0.5) < 0.1 and l1[8] > 0.5


# ---------------------------------------------------------------- constraints on G (user requirements 1 and 2)
def test_kF_ge_G_qmin_and_delta_bound_as_G_grows():
    # k_F = 200, q_min = 25, delta = 0.5 -> moved = 100, G_max = (200 - 100) // 25 = 4. Families arrive one per day.
    p = ThreeBin(k=500, w=(0.5, 0.4, 0.1), q_min=25, delta=0.5, kappa_x=0.3)
    assert (p.k_F, p.moved, p.G_max) == (200, 100, 4)
    rng = np.random.default_rng(9)
    n_fam = 9

    def probs(d):
        q = np.zeros(n_fam + 1)
        q[0] = 0.5
        q[1:min(d + 2, n_fam + 1)] = 1.0
        q[1:] *= 0.5 / q[1:].sum()
        return q
    for d, ctx in drive(p, 30, rng, probs, n_per_day=300, shift_of=lambda d: 1.5 * ((d // 4) % 2)):
        G = len(p.groups)
        assert p.k_F >= G * p.q_min
        assert p.moved <= p.k_F - G * p.q_min
        assert p.log[-1]["k_F_eff"] >= G * p.q_min
        assert set(p.F) <= p.groups                                  # unprotected groups never hold F rows
    assert p.groups == {1, 2, 3, 4} and p.unprotected == {5, 6, 7, 8, 9}
    assert any(l["trigger_active"] for l in p.log)


def test_group_admission_rule_raises_when_no_group_fits():
    with pytest.raises(ValueError):
        ThreeBin(k=100, w=(0.6, 0.3, 0.1), q_min=20, delta=0.5)      # k_F = 30, moved = 15 -> 15 < q_min


# ---------------------------------------------------------------- determinism and isolation
def _run(group_by, seed):
    s = SynthStream(6, lambda d: [0.75, 0.1, 0.1, 0.05], per_day=2000, x_shift=lambda d: 0.5 * (d >= 3))
    p = recording(ThreeBin)(k=600, q_min=10, group_by=group_by, seed=seed, n_clusters=3)
    recs = run_replay(s, p, LightGBMModel(params=FAST, seed=seed), b=400, tau=0, a_fp=50, n_days=6, n_eval=300,
                      seed=seed)
    strip = [{k: v for k, v in l.items() if not k.endswith("seconds")} for l in p.log]
    return [t["rid"].tolist() for t in p.trace], strip, [r["auprc_w"] for r in recs]


@pytest.mark.parametrize("group_by", ["family", "cluster"])
def test_deterministic_across_runs(group_by):
    a, b = _run(group_by, 0), _run(group_by, 0)
    assert a[0] == b[0]
    assert repr(a[1]) == repr(b[1])
    assert a[2] == b[2]


@pytest.mark.parametrize("tau", [0, 1])
def test_isolation_I1_I3_assert_mode(tau):
    s = SynthStream(6, lambda d: [0.75, 0.15, 0.1], per_day=2000)
    recs = run_replay(s, ThreeBin(k=500, q_min=10), LightGBMModel(params=FAST), b=400, tau=tau, a_fp=50, n_days=6,
                      n_eval=300, isolation="assert")
    assert len(recs) == 5
    assert all(r["iso_dropped_ctx"] == 0 and r["iso_dropped_cal"] == 0 for r in recs)


def test_isolation_roles_stream_filter_mode_drops_nothing():
    s = SynthStream(6, lambda d: [0.75, 0.15, 0.1], per_day=3000, roles=True)
    recs = run_replay(s, ThreeBin(k=500, q_min=10, group_by="cluster", n_clusters=3), LightGBMModel(params=FAST),
                      b=400, tau=1, a_fp=50, n_days=6, n_eval=300, isolation="filter", max_filter_drops=0)
    assert len(recs) == 5


def test_context_mismatch_is_rejected():
    p = ThreeBin(k=100, q_min=5)
    rng = np.random.default_rng(10)
    ctx = p.update(Rows.empty(N_FEATURES), batch(rng, 50, [0.5, 0.5], 0), None)
    with pytest.raises(ValueError):
        p.update(ctx.take(np.arange(10)), batch(rng, 10, [0.5, 0.5], 100), None)


def test_k_center_spreads_points():
    Z = np.array([[0.0, 0], [0.1, 0], [0.2, 0], [10, 0], [10.1, 0], [-10, 0]])
    sel = k_center(Z, 3)
    assert len(sel) == 3 and 5 in sel and len(set(sel.tolist()) & {3, 4}) == 1 and len(set(sel.tolist()) & {0, 1, 2}) == 1


# ---------------------------------------------------------------- smoke: 5-day absence
def test_smoke_family_absent_5_days_threebin_keeps_fifo_loses():
    s = SynthStream(11, absence_probs(), per_day=4000, seed=0)
    out = {}
    for cls, kw in ((ThreeBin, dict(q_min=20)), (FIFO, {})):
        p = recording(cls)(k=1000, **kw)
        run_replay(s, p, LightGBMModel(params=FAST), b=600, tau=0, a_fp=50, n_days=11, n_eval=1000)
        out[cls.name] = [t["fam"].get(5, 0) for t in p.trace]
    # trace index d = context after the update at the end of day d; family 5 absent on days 5-9
    assert out["ThreeBin"][4] > 0 and out["FIFO"][4] > 0
    assert out["FIFO"][9] == 0
    assert out["ThreeBin"][9] >= min(20, out["ThreeBin"][4])
