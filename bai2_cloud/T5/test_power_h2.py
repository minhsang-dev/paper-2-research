"""Synthetic checks for power_h2.py. Set BAI2_BUNDLE to the extracted bai2_handoff folder to also compare with the
locked Bai 1 analysis (read-only import)."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_h2 as P  # noqa: E402


def test_hl_matches_locked_analysis():
    b = os.environ.get("BAI2_BUNDLE")
    if not b:
        return
    sys.path.insert(0, str(Path(b) / "protocol_code" / "main_h1"))
    import analysis_h1 as A
    rng = np.random.default_rng(1)
    for n in (5, 6, 10, 16):
        x = rng.standard_normal(n)
        hl, lb, _ = A.hodges_lehmann(x)
        hl2, lb2 = P.hodges_lehmann(x)
        assert np.isclose(hl, hl2) and np.isclose(lb, lb2)
        assert np.isclose(P.locked_lb(x[None])[0], lb)


def test_exact_rule_level_independent_normal():
    rng = np.random.default_rng(2)
    x = P.EPS2 + rng.standard_normal((20000, 10))
    assert np.mean(P.exact_lb(x) >= P.EPS2) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 20000)


def test_small_n_cannot_reject():
    for n in range(1, 5):
        assert 2.0 ** -n > P.ALPHA and P.lb_index(n) is None
        assert np.all(P.exact_lb(np.ones((3, n))) == -np.inf)


def test_locked_index_is_one_rank_high():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40000, 10))
    assert np.mean(P.locked_lb(x) >= 0) > 0.05 > np.mean(P.exact_lb(x) >= 0)


def test_time_correlation_structure():
    lag, same, groups, n = P.corr_time(P.CONFIGS["n10_H2_design"])
    assert n == 10 and [len(g) for g in groups] == [6, 4]
    assert lag[6, 9] == 7 and not same[0, 6]   # CIC 02-14 vs 02-21; ToN vs CIC independent
