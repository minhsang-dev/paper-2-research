"""T5 item 5: false-positive rate and power of the proposed H2 decision rule, by simulation.

    python power_h2.py <path/to/bai2_handoff> [power_h2_results.json]

Every input number comes from the handoff bundle (read from context/numbers.json where it is there) or from the Bai 1
manuscript text in the bundle (section named next to the constant). Nothing here reads data rows.

Unit and outcome: one value x_r per ROOT DAY r = (ThreeBin - comparator) in the H2 endpoint, averaged over the root
day's situations and seeds. Proposed primary rule (design_h2.md section 3): exact one-sided Wilcoxon signed-rank test of
x - EPS2, i.e. the exact one-sided 95 % Hodges-Lehmann lower bound >= EPS2 (same computation as the locked Bai 1
analysis_h1.hodges_lehmann). H2 needs the rule to hold for EVERY tier-A comparator (and both noise levels):
an intersection-union test (IUT), simulated for m = 1, 3, 6 correlated contrasts.

Between-root-day models of x - theta (marginals; the only between-root-day spreads in the bundle are for the Bai 1
TabFM - LightGBM contrast, so they are used as stand-ins for the policy contrast; design_h2.md section 5):
  M1 heavy tail  : N(0, 0.0024^2) on all but one root day, one root day at +-0.27411 (Bai 1 Section 4.7 model A)
  M2 normal      : N(0, 0.071^2) (Bai 1 Section 4.7 model B)
  M3 recurring   : empirical distribution of the 10 Bai 1 recurring-kind root-day values (numbers.json t3.*.recurring)
  M4 per family  : empirical distribution of the 20 Bai 1 per-family median differences (numbers_addenda.json per_family_FIFO)
  M3 and M4 are centred at their own pseudo-median, so theta is the pseudo-median (the Wilcoxon estimand).
  Seed and scenario noise (Bai 1 Section 4.6: SD 0.0021 and 0.0008) are added, divided by sqrt(seeds) / sqrt(S).
Dependence: Gaussian copula, corr(r, r') = rho ** |calendar days between r and r'| within a dataset (AR(1) in
calendar time), 0 across datasets; rho in {0, 0.3, 0.6}. Contrasts of one root day: exchangeable correlation 0.5
(the shared ThreeBin term of two contrasts with equal variances).
Rules compared (n = 10 design): EXACT (primary); EXACT+BLOCK (EXACT and the Bai 1 moving-block bootstrap bound);
T (one-sample t bound, = the outline's mixed model collapsed to balanced root-day means); HBOOT (percentile bootstrap of
the mean over root days, = the outline's hierarchical bootstrap at its top level); SIGN (exact sign test).
"""
import hashlib
import json
import sys
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

EPS2 = 0.03                    # outline v14.3, Noi dung 1 (provisional until eps2_criteria.py is run)
ALPHA = 0.05
REPS = 4000
REPS_BOOT = 1000               # replicates for the cells with an inner bootstrap
B_INNER = 1000
BLOCK = 3                      # Bai 1 locked analysis
SEED = 20261029
SD_SEED, SD_SCEN = 0.0021, 0.0008      # Bai 1 Section 4.6 (blinded pilot)
SD_ROOT_MAD, SD_ROOT_MOM, OUTLIER = 0.0024, 0.071, 0.27411   # Bai 1 Section 4.6/4.7 (power_v3 model A outlier)
N_SEEDS, S_PER_ROOT = 5, 4     # proposed design (design_h2.md section 6)
RHOS = (0.0, 0.3, 0.6)
THETAS = (0.0, 0.03, 0.04, 0.05, 0.07, 0.10, 0.20)
RHO_CONTRAST = 0.5

DATES = {   # root days by dataset, as day numbers (calendar order; manifest root days)
    "ToN": {"2019-04-23": 23, "2019-04-24": 24, "2019-04-25": 25, "2019-04-26": 26, "2019-04-27": 27,
            "2019-04-28": 28, "2019-04-29": 29},
    "CIC": {"2018-02-14": 14, "2018-02-15": 15, "2018-02-16": 16, "2018-02-20": 20, "2018-02-21": 21,
            "2018-02-22": 22, "2018-02-28": 28, "2018-03-01": 29, "2018-03-02": 30},
}
H2_ROOTS = {"ToN": ["2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29"],
            "CIC": ["2018-02-14", "2018-02-15", "2018-02-16", "2018-02-21"]}   # situations_h2.json
CONFIGS = {
    "n10_H2_design": H2_ROOTS,
    "n6_ToN_only": {"ToN": H2_ROOTS["ToN"]},
    "n4_CIC_only": {"CIC": H2_ROOTS["CIC"]},
    "n16_all_root_days": {k: list(v) for k, v in DATES.items()},
}


# ------------------------------------------------------------------ statistics (as the locked Bai 1 analysis)

def signed_rank_null(n):
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1
    for r in range(1, n + 1):
        new = dist.copy()
        new[r:] += dist[:-r]
        dist = new
    return dist / dist.sum()


def hl_index(n, alpha=ALPHA):
    """Index used by the LOCKED Bai 1 analysis_h1.hodges_lehmann: c = #{t : P(T <= t) <= alpha}. The bound walsh[c]
    is exceeded by theta with probability P(T <= c) > alpha, i.e. it is one rank too high (see lb_index)."""
    return int(np.searchsorted(np.cumsum(signed_rank_null(n)), alpha, side="right"))


def lb_index(n, alpha=ALPHA):
    """Index of the exact one-sided (1 - alpha) lower bound: walsh[c - 1], since P(walsh[c-1] > theta) =
    P(T <= c - 1) <= alpha. None when c = 0 (n <= 4 at alpha = 0.05): no bound, the rule cannot be met."""
    c = hl_index(n, alpha)
    return c - 1 if c > 0 else None


def walsh_sorted(x):
    """x: (..., n) -> sorted Walsh averages (..., n(n+1)/2)."""
    i, j = np.triu_indices(x.shape[-1])
    return np.sort((x[..., i] + x[..., j]) / 2, axis=-1)


def hodges_lehmann(x, alpha=ALPHA):
    """(HL estimate, lower bound with the LOCKED Bai 1 index) - used to check equality with analysis_h1."""
    w = walsh_sorted(np.asarray(x, float)[None])[0]
    c = hl_index(len(x), alpha)
    return float(np.median(w)), float(w[c])


def exact_lb(x, alpha=ALPHA):
    """Exact one-sided (1 - alpha) HL lower bound (corrected index); -inf if no bound exists."""
    i = lb_index(x.shape[-1], alpha)
    return np.full(x.shape[:-1], -np.inf) if i is None else walsh_sorted(x)[..., i]


def locked_lb(x):
    return walsh_sorted(x)[..., hl_index(x.shape[-1])]


def sign_rejects(x, eps):
    n = x.shape[-1]
    k = (x > eps).sum(-1)
    pvals = np.array([sum(comb(n, j) for j in range(kk, n + 1)) / 2 ** n for kk in range(n + 1)])
    return pvals[k] < ALPHA


def t_lb(x):
    n = x.shape[-1]
    return x.mean(-1) - stats.t.ppf(1 - ALPHA, n - 1) * x.std(-1, ddof=1) / np.sqrt(n)


def hboot_lb(x, rng):
    n = x.shape[-1]
    idx = rng.integers(0, n, size=(x.shape[0], B_INNER, n))
    means = np.take_along_axis(np.repeat(x[:, None, :], B_INNER, 1), idx, -1).mean(-1)
    return np.quantile(means, ALPHA, axis=1)


def block_lb(x, groups, rng):
    """Bai 1 moving-block bootstrap (blocks of BLOCK consecutive root days within each dataset), HL of the pooled
    resample, lower ALPHA quantile. x: (reps, n); groups: list of index arrays in calendar order."""
    out = np.empty(x.shape[0])
    for r in range(x.shape[0]):
        parts = []
        for g in groups:
            s = x[r, g]
            n, L = len(s), min(BLOCK, len(s))
            nb = -(-n // L)
            st = rng.integers(0, n - L + 1, size=(B_INNER, nb))
            res = s[(st[:, :, None] + np.arange(L)).reshape(B_INNER, -1)[:, :n]]
            parts.append(res)
        pooled = np.concatenate(parts, axis=1)
        out[r] = np.quantile(np.median(walsh_sorted(pooled), axis=-1), ALPHA)
    return out


# ------------------------------------------------------------------ data-generating models

def load_empirical(bundle):
    num = json.loads((Path(bundle) / "context" / "numbers.json").read_text(encoding="utf-8"))
    rec = [n["value"] for n in num["numbers"] if n["id"].startswith("t3.") and n["id"].endswith(".recurring")]
    add = json.loads((Path(bundle) / "context" / "numbers_addenda.json").read_text(encoding="utf-8"))
    fam = [f["median_diff"] for f in add["per_family_FIFO"]]
    assert len(rec) == 10 and len(fam) == 20
    return np.array(rec), np.array(fam)


def corr_time(cfg):
    days, ds = [], []
    for d, names in cfg.items():
        days += [DATES[d][x] for x in names]
        ds += [d] * len(names)
    n = len(days)
    lag = np.abs(np.subtract.outer(days, days))
    same = np.equal.outer(ds, ds)
    groups = [np.flatnonzero(np.array(ds) == d) for d in cfg]
    return lag, same, groups, n


def gen(model, emp, theta, rho, cfg, m, rng, reps):
    """(reps, m, n) root-day outcomes."""
    lag, same, _, n = corr_time(cfg)
    Ct = np.where(same, rho ** lag.astype(float), 0.0)
    Cc = np.full((m, m), RHO_CONTRAST) + (1 - RHO_CONTRAST) * np.eye(m)
    L = np.linalg.cholesky(np.kron(Cc, Ct) + 1e-12 * np.eye(m * n))
    z = (rng.standard_normal((reps, m * n)) @ L.T).reshape(reps, m, n)
    if model == "M1_heavy_tail":
        e = SD_ROOT_MAD * z
        pos = rng.integers(0, n, reps)
        sign = rng.choice([-1.0, 1.0], reps)
        e[np.arange(reps), :, pos] += (sign * OUTLIER)[:, None]
    elif model == "M2_normal":
        e = SD_ROOT_MOM * z
    else:
        v = np.sort(emp[model])
        v = v - hodges_lehmann(v)[0]                  # centre at the pseudo-median
        u = stats.norm.cdf(z)
        e = np.interp(u, (np.arange(len(v)) + 0.5) / len(v), v)
    noise = rng.standard_normal((reps, m, n)) * np.sqrt(SD_SEED ** 2 / N_SEEDS + SD_SCEN ** 2 / S_PER_ROOT)
    return theta + e + noise


# ------------------------------------------------------------------ main

def main(bundle, out_path):
    rec, fam = load_empirical(bundle)
    emp = {"M3_recurring_empirical": rec, "M4_per_family_empirical": fam}
    models = ["M1_heavy_tail", "M2_normal", "M3_recurring_empirical", "M4_per_family_empirical"]
    rng = np.random.default_rng(SEED)
    res = {"single_contrast": [], "iut": [], "rules_n10": [], "small_n_exact_rule": {}}
    # exact rule cannot reject below n = 5; the locked HL index then returns min(x) with coverage 1 - 2^-n
    for n in range(1, 17):
        c = hl_index(n)
        cdf = np.cumsum(signed_rank_null(n))
        res["small_n_exact_rule"][n] = {"locked_index": c, "corrected_index": lb_index(n),
                                        "min_one_sided_p": 2.0 ** -n, "test_can_reject_at_alpha": 2.0 ** -n <= ALPHA,
                                        "exact_level_locked_index": float(cdf[c]),
                                        "exact_level_corrected_index": float(cdf[c - 1]) if c else 0.0}
    for cname, cfg in CONFIGS.items():
        for model in models:
            for rho in RHOS:
                for th in THETAS:
                    x = gen(model, emp, th, rho, cfg, 1, rng, REPS)[:, 0, :]
                    p = float(np.mean(exact_lb(x) >= EPS2))
                    p0 = float(np.mean(exact_lb(x) > 0))
                    pl = float(np.mean(locked_lb(x) >= EPS2))
                    res["single_contrast"].append({"config": cname, "n": x.shape[1], "model": model, "rho": rho,
                                                   "theta": th, "p_exact_rule": p, "p_exact_rule_locked_index": pl,
                                                   "p_lb_above_0": p0})
    cfg = CONFIGS["n10_H2_design"]
    _, _, groups, _ = corr_time(cfg)
    for model in models:
        for rho in RHOS:
            for th in THETAS:
                for m in (3, 6):
                    x = gen(model, emp, th, rho, cfg, m, rng, REPS)
                    p = float(np.mean((exact_lb(x) >= EPS2).all(axis=1)))
                    res["iut"].append({"config": "n10_H2_design", "model": model, "rho": rho, "theta": th, "m": m,
                                       "p_all_m_contrasts": p})
                x = gen(model, emp, th, rho, cfg, 1, rng, REPS)[:, 0, :]
                row = {"model": model, "rho": rho, "theta": th,
                       "EXACT": float(np.mean(exact_lb(x) >= EPS2)),
                       "T": float(np.mean(t_lb(x) >= EPS2)),
                       "HBOOT": float(np.mean(hboot_lb(x[:REPS_BOOT], rng) >= EPS2)),
                       "SIGN": float(np.mean(sign_rejects(x, EPS2)))}
                if th in (0.03, 0.07):
                    xb = x[:REPS_BOOT]
                    row["EXACT+BLOCK"] = float(np.mean((exact_lb(xb) >= EPS2) & (block_lb(xb, groups, rng) >= EPS2)))
                res["rules_n10"].append(row)
    # alpha calibration under dependence: which one-sided alpha keeps the boundary FPR <= 0.05 at rho = 0.3 / 0.6?
    res["alpha_calibration_n10"] = []
    for model in models:
        for rho in RHOS:
            for th in (0.03, 0.05, 0.07, 0.10):
                x = gen(model, emp, th, rho, cfg, 1, rng, REPS)[:, 0, :]
                x3 = gen(model, emp, th, rho, cfg, 3, rng, REPS)
                for a in (0.05, 0.025, 0.01):
                    res["alpha_calibration_n10"].append({
                        "model": model, "rho": rho, "theta": th, "alpha": a,
                        "p_single": float(np.mean(exact_lb(x, a) >= EPS2)),
                        "p_iut_m3": float(np.mean((exact_lb(x3, a) >= EPS2).all(axis=1)))})
    meta = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numbers_json_sha256": hashlib.sha256((Path(bundle) / "context" / "numbers.json").read_bytes()).hexdigest(),
            "numbers_addenda_json_sha256": hashlib.sha256(
                (Path(bundle) / "context" / "numbers_addenda.json").read_bytes()).hexdigest(),
            "eps2": EPS2, "alpha": ALPHA, "reps": REPS, "reps_boot": REPS_BOOT, "b_inner": B_INNER, "seed": SEED,
            "n_seeds": N_SEEDS, "situations_per_root": S_PER_ROOT, "rho_contrast": RHO_CONTRAST,
            "mc_se_at_0.05": float(np.sqrt(0.05 * 0.95 / REPS)),
            "empirical_M3_values": rec.tolist(), "empirical_M4_values": fam.tolist(),
            "M3_pseudo_median": hodges_lehmann(rec)[0], "M4_pseudo_median": hodges_lehmann(fam)[0],
            "configs": {k: {d: v for d, v in c.items()} for k, c in CONFIGS.items()}}
    Path(out_path).write_text(json.dumps({"meta": meta, **res}, indent=1), encoding="utf-8")
    # short console table: false-positive rate (theta = EPS2) and power at theta = 0.07, n = 10, single contrast
    for r in res["single_contrast"]:
        if r["config"] == "n10_H2_design" and r["theta"] in (0.03, 0.07):
            print(r["model"], r["rho"], r["theta"], round(r["p_exact_rule"], 3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "power_h2_results.json")
