"""Synthetic day streams for the ThreeBin tests and the absence demo (no real data). Duck-types the replay's stream
interface (dataset, n_features, names, rows, labels, families, get; optional roles)."""
import numpy as np

import threebin  # noqa: F401  (puts protocol_code/replay on sys.path)
from rows import Rows

N_FEATURES = 6


class SynthStream:
    """Day d holds `per_day` rows; family of each row drawn from fam_probs(d) (index 0 = benign). Features: benign
    N(0, 1); family f shifts features (f mod 6) and ((f + 2) mod 6) by +3 (separable but overlapping), plus a count-like
    positive column. `x_shift(d)` adds a constant to every feature of day d (P(X) drift). rid = global row index.
    roles=True exposes disjoint eval / policy / cal pools per day (generator-v2 style)."""
    dataset = "NF-ToN-IoT-v3"   # used only as a seed label by the replay; no data is read
    n_features = N_FEATURES

    def __init__(self, n_days, fam_probs, per_day=3000, seed=0, x_shift=None, roles=False):
        self.days = []
        for d in range(n_days):
            rng = np.random.default_rng([seed, d])
            p = np.asarray(fam_probs(d), float)
            fam = rng.choice(len(p), size=per_day, p=p / p.sum()).astype(np.int16)
            X = rng.normal(0, 1, (per_day, N_FEATURES))
            for f in np.unique(fam[fam > 0]):
                m = fam == f
                X[m, f % N_FEATURES] += 3.0
                X[m, (f + 2) % N_FEATURES] += 3.0
            X[:, -1] = rng.poisson(5 + 20 * (fam > 0))
            if x_shift is not None:
                X += x_shift(d)
            rid = np.arange(d * per_day, (d + 1) * per_day, dtype=np.int64)
            self.days.append(Rows(X.astype(np.float32), (fam > 0).astype(np.uint8), fam, rid))
        self.names = [f"s{d}" for d in range(n_days)]
        self.rows = [np.arange(per_day) for _ in range(n_days)]
        self._roles = None
        if roles:
            third = per_day // 3
            self._roles = {"eval": np.arange(0, third), "policy": np.arange(third, 2 * third),
                           "cal": np.arange(2 * third, per_day)}
            self.roles = lambda t: self._roles

    def labels(self, i):
        return self.days[i].y

    def families(self, i):
        return self.days[i].fam

    def get(self, i, pos):
        return self.days[i].take(np.asarray(pos))


def absence_probs(rare=5, n_fam=5, present=range(0, 5), returns=10, p_rare=0.004):
    """Benign 0.8; families 1..n_fam share 0.2; family `rare` has p_rare on days in `present` or >= `returns`,
    0 on the absent days in between."""
    def probs(d):
        p = np.zeros(n_fam + 1)
        p[0] = 0.8
        on = d in present or d >= returns
        p[rare] = p_rare if on else 0.0
        others = [f for f in range(1, n_fam + 1) if f != rare]
        p[others] = (0.2 - p[rare]) / len(others)
        return p
    return probs


def recording(cls):
    """Subclass of a policy that records, after every update, the per-family row counts of the returned context."""
    class Recording(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.trace = []

        def update(self, context, new_labelled_rows, unlabelled_stream):
            ctx = super().update(context, new_labelled_rows, unlabelled_stream)
            f, c = np.unique(ctx.fam, return_counts=True)
            self.trace.append({"n": len(ctx), "fam": {int(a): int(b) for a, b in zip(f, c)},
                               "rid": ctx.rid.copy()})
            return ctx
    Recording.__name__ = f"Recording{cls.__name__}"
    Recording.name = cls.name
    return Recording
