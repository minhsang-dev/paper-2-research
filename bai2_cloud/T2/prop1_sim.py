"""Seeded Monte Carlo checks for Proposition 1 (Mệnh đề 1) and the q_min* rule. Synthetic streams only; no data.

Usage: python3 prop1_sim.py [--quick]   -> writes prop1_sim_results.json next to this file.
Every number quoted in prop1.md / REPORT.md is read from that JSON.

Sections (keys of the JSON):
  a_fifo        (a) P(N_c=0) = (1-r)^kR <= exp(-r kR) for a full FIFO bin; certain loss when d*lambda >= kR.
  b_uniform     (b) uniform reservoir (Vitter Algorithm R): E[N_c] = k n_c / n -> 0.
  c_policy      (c) explicit three-compartment buffer: N_c >= min(q_min, M_c) at every step, |context| <= k,
                    k_H <= 0.1k, k_F >= G q_min; FIFO of the same size loses the absent group.
  pmiss         Poisson-recurrence miss probability P_miss(q) = (1-theta)^q * H(K - G q): MC vs closed form.
  qstar         q_min* (exact enumeration, robust closed form, emerging-family variant) and MC argmin check.
  sensitivity   q_min* (theory) as each of k, G, lambda, r, rho, theta, mu varies.
  theorem_bound q*_exact <= q_rob on random parameter sets; objective gap of using q_rob instead of q*_exact.
  (pmiss, qstar, sensitivity, theorem_bound use the full-recency reading k_R = K - G q and the price mu: Appendix A.)
  hybrid        D1 + D3 (primary rule): hybrid capacities, same-currency objective, MC of both miss terms + argmin.
  hybrid_sensitivity, hybrid_theorem   q* of the primary rule vs each parameter; q_thr <= q* <= q_rob on random sets.
"""
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

SEED = 20261029
QUICK = "--quick" in sys.argv
OUT = Path(__file__).with_name("prop1_sim_results.json")


def ss(*key):
    """Independent, reproducible stream per check."""
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


def z(mc, th, se):
    return None if se == 0 else (mc - th) / se


# ----------------------------------------------------------------------------- theory
def H(m, r, theta, s):
    """E[(1 - r theta)^{(m - J)^+}], J ~ Geometric on {0,1,..} with P(J >= j) = exp(-s j).
    = P(no covering c-row survives in a FIFO window of m rows after an absence of J rows)."""
    if m <= 0:
        return 1.0
    u, b = 1.0 - r * theta, math.exp(-s)
    if abs(u - b) < 1e-15:
        return b ** m + (1 - b) * m * b ** m
    return b ** m + (1 - b) * u * (u ** m - b ** m) / (u - b)


def pmiss(q, K, G, r, theta, rho, lam):
    return (1 - theta) ** q * H(K - G * q, r, theta, rho / lam)


def J_obj(q, P):
    """Primary objective: expected missed reappearances per day + mu * (recency slots given to the floor)."""
    return P["n_rec"] * P["rho"] * pmiss(q, P["K"], P["G"], P["r"], P["theta"], P["rho"], P["lam"]) + P["mu"] * P["G"] * q


def qstar_exact(P, obj=J_obj):
    qmax = P["K"] // P["G"]
    vals = [obj(q, P) for q in range(qmax + 1)]
    q = int(np.argmin(vals))
    return q, vals


def qstar_robust(P):
    """Minimiser of the worst case over absence laws, sum rho (1-theta)^q + mu G q (convex)."""
    beta = -math.log(1 - P["theta"])
    A = P["n_rec"] * P["rho"]
    qmax = P["K"] // P["G"]
    qc = math.log(beta * A / (P["mu"] * P["G"])) / beta
    f = lambda q: A * (1 - P["theta"]) ** q + P["mu"] * P["G"] * q
    cands = [min(max(c, 0), qmax) for c in (math.floor(qc), math.ceil(qc))]
    return min(cands, key=f), qc


def J2_obj(q, P):
    """Same-currency variant: missed reappearances + missed emerging families in the recency bin, per day."""
    kR = P["K"] - P["G"] * q
    return (P["n_rec"] * P["rho"] * pmiss(q, P["K"], P["G"], P["r"], P["theta"], P["rho"], P["lam"])
            + P["rho_e"] * (1 - P["theta"] * P["r_e"]) ** kR)


def qstar2_flushed(P):
    beta = -math.log(1 - P["theta"])
    ae = -math.log(1 - P["theta"] * P["r_e"])
    A = P["n_rec"] * P["rho"]
    qc = (math.log(beta * A / (ae * P["G"] * P["rho_e"])) + ae * P["K"]) / (beta + ae * P["G"])
    f = lambda q: A * (1 - P["theta"]) ** q + P["rho_e"] * math.exp(-ae * (P["K"] - P["G"] * q))
    qmax = P["K"] // P["G"]
    cands = [min(max(c, 0), qmax) for c in (math.floor(qc), math.ceil(qc))]
    return min(cands, key=f), qc


# ----------------------------------------------------------------------------- (a)
def check_a():
    out = []
    reps = 4000 if QUICK else 20000
    for i, (r, kR) in enumerate([(1e-4, 5000), (1e-3, 1000), (5e-3, 500)]):
        rng = ss(1, i)
        zero, ncs, done = 0, [], 0
        while done < reps:
            m = min(2000, reps - done)
            rows = rng.random((m, kR)) < r          # last kR rows of an i.i.d. stream (bin full)
            n = rows.sum(1)
            zero += int((n == 0).sum()); ncs.append(n); done += m
        p = zero / reps
        th = (1 - r) ** kR
        se = math.sqrt(th * (1 - th) / reps)          # z uses the SE implied by the theory value
        out.append(dict(r=r, kR=kR, reps=reps, mc_P0=p, se=se, theory_P0=th, bound_exp=math.exp(-r * kR),
                        z=z(p, th, se), bound_holds=th <= math.exp(-r * kR),
                        mc_mean_N=float(np.concatenate(ncs).mean()), theory_mean_N=r * kR))
    # absence: c absent for d days at lambda rows/day, then FIFO window of kR rows inspected.
    absent = []
    reps = 1000 if QUICK else 5000
    for i, (r, kR, lam, d) in enumerate([(0.01, 1000, 200, 5), (0.01, 1000, 200, 4), (0.01, 1000, 300, 3)]):
        rng = ss(2, i)
        J = d * lam
        pre = rng.random((reps, max(kR, 1))) < r       # rows before the absence (c present w.p. r)
        stream = np.concatenate([pre, np.zeros((reps, J), bool)], axis=1)  # absence: no c rows
        n = stream[:, -kR:].sum(1)
        p0 = 1.0 if J >= kR else (1 - r) ** (kR - J)
        absent.append(dict(r=r, kR=kR, lam=lam, d=d, d_lam_ge_kR=J >= kR, reps=reps,
                           z_P0=z(float((n == 0).mean()), p0, math.sqrt(p0 * (1 - p0) / reps)),
                           mc_P0=float((n == 0).mean()), mc_max_N=int(n.max()), mc_mean_N=float(n.mean()),
                           theory_mean_N=r * max(0, kR - J),
                           theory_P0=1.0 if J >= kR else (1 - r) ** (kR - J)))
    return dict(full_bin=out, absence=absent,
                outline_example=dict(r=1e-4, kR=5000, exact=(1 - 1e-4) ** 5000, exp_bound=math.exp(-0.5)))


# ----------------------------------------------------------------------------- (b)
def check_b():
    out = []
    k, n_c = 50, 20
    reps = 500 if QUICK else 2000
    for i, n in enumerate([200, 1000, 5000, 20000]):
        rng = ss(3, i)
        # history: n rows, the n_c rows of group c sit at random positions among the first n_c*5 rows (old, rare)
        is_c = np.zeros((reps, n), bool)
        for t in range(reps):
            is_c[t, rng.choice(min(n, 5 * n_c), n_c, replace=False)] = True
        res = np.tile(np.arange(k), (reps, 1))            # Vitter (1985) Algorithm R, vectorised over reps
        for idx in range(k, n):
            j = rng.integers(0, idx + 1, size=reps)
            hit = j < k
            res[hit, j[hit]] = idx
        N = is_c[np.arange(reps)[:, None], res].sum(1)
        mc, se = float(N.mean()), float(N.std(ddof=1) / math.sqrt(reps))
        th = k * n_c / n
        p0_th = math.comb(n - n_c, k) / math.comb(n, k)
        out.append(dict(n=n, k=k, n_c=n_c, reps=reps, mc_mean_N=mc, se=se, theory_mean_N=th, z=z(mc, th, se),
                        mc_P0=float((N == 0).mean()), theory_P0=p0_th,
                        z_P0=z(float((N == 0).mean()), p0_th, math.sqrt(p0_th * (1 - p0_th) / reps))))
    return out


# ----------------------------------------------------------------------------- (c)
class ThreeCompartment:
    """Minimal model of the policy in outline v14.3 (Nội dung 2), written only to test the invariant of (c).
    R: FIFO; rows leaving R or H are offered to F. F: per-group quotas from (2-4) (largest remainder), an
    over-quota group is trimmed; within-group choice is random (k-center is irrelevant to the counts).
    H: rows flagged hard, cap k_H <= 0.1k, evicted by age. Shift: delta*k_F moves to R for D cycles, never
    pushing k_F below G*q_min."""

    def __init__(self, k, G, q_min, wR=0.5, wF=0.4, wH=0.1, gamma=0.5, delta=0.5, D=3, rng=None):
        self.k, self.G, self.q = k, G, q_min
        self.kH = int(math.floor(wH * k))
        self.kF0 = int(round(wF * k))
        assert self.kF0 >= G * q_min and self.kH <= 0.1 * k
        self.kF = self.kF0
        self.kR = k - self.kH - self.kF
        self.gamma, self.delta, self.D = gamma, delta, D
        self.rng = rng
        self.R, self.H = deque(), deque()
        self.F = {c: [] for c in range(G)}
        self.seen = np.zeros(G, int)       # M_c: trusted c-rows ever received
        self.shift_left = 0

    def quotas(self):
        pi = self.seen / max(1, self.seen.sum())
        w = pi ** self.gamma
        w = w / w.sum() if w.sum() > 0 else np.full(self.G, 1 / self.G)
        spare = self.kF - self.G * self.q
        raw = spare * w
        base = np.floor(raw).astype(int)
        rem = spare - base.sum()
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rem]] += 1                    # largest remainder
        return self.q + base

    def _offer_F(self, row):
        c = row[0]
        qc = self.quotas()
        if len(self.F[c]) < qc[c]:
            if sum(len(v) for v in self.F.values()) >= self.kF:   # room: some group must be over quota
                over = [g for g in range(self.G) if len(self.F[g]) > qc[g]]
                g = over[self.rng.integers(len(over))]
                self.F[g].pop(self.rng.integers(len(self.F[g])))
            self.F[c].append(row)
        else:                                      # at quota: k-center may swap within the group; count unchanged
            if self.F[c] and self.rng.random() < 0.5:
                self.F[c][self.rng.integers(len(self.F[c]))] = row

    def _trim(self):
        qc = self.quotas()
        for g in range(self.G):
            while len(self.F[g]) > qc[g] and sum(len(v) for v in self.F.values()) > self.kF:
                self.F[g].pop(self.rng.integers(len(self.F[g])))
        while sum(len(v) for v in self.F.values()) > self.kF:   # global over-capacity: trim over-quota groups
            over = [g for g in range(self.G) if len(self.F[g]) > qc[g]]
            g = over[self.rng.integers(len(over))]
            self.F[g].pop(self.rng.integers(len(self.F[g])))
        while len(self.R) > self.kR:
            self._offer_F(self.R.popleft())
        while len(self.H) > self.kH:
            self._offer_F(self.H.popleft())

    def cycle(self, rows, shift_signal):
        if shift_signal and self.shift_left == 0:
            self.kF = max(self.G * self.q, self.kF0 - int(self.delta * self.kF0))
            self.shift_left = self.D
        elif self.shift_left > 0:
            self.shift_left -= 1
            if self.shift_left == 0:
                self.kF = self.kF0
        self.kR = self.k - self.kH - self.kF
        self._trim()
        for row in rows:
            self.seen[row[0]] += 1
            (self.H if row[1] else self.R).append(row)
            self._trim()

    def counts(self):
        n = np.zeros(self.G, int)
        for row in list(self.R) + list(self.H):
            n[row[0]] += 1
        for g in range(self.G):
            n[g] += len(self.F[g])
        return n

    def size(self):
        return len(self.R) + len(self.H) + sum(len(v) for v in self.F.values())


def check_c():
    out = []
    configs = [dict(k=400, G=6, q=10), dict(k=1000, G=8, q=20), dict(k=1000, G=8, q=50)]
    days = 25 if QUICK else 60
    lam = 150
    for i, cf in enumerate(configs):
        rng = ss(4, i)
        pol = ThreeCompartment(cf["k"], cf["G"], cf["q"], rng=rng)
        fifo = deque(maxlen=cf["k"])
        G = cf["G"]
        share = np.array([0.6] + [0.4 / (G - 1)] * (G - 1))
        # groups 1..3 recur: active with prob 0.3 each day (long random absences); group G-1 appears once (day 2)
        viol, max_size, min_slack, kH_max, kF_min, max_abs = 0, 0, 10 ** 9, 0, 10 ** 9, np.zeros(G, int)
        last_seen = np.zeros(G, int)
        fifo_lost, tc_lost = 0, 0
        kR_seen = set()
        for day in range(days):
            active = np.ones(G, bool)
            active[1:4] = rng.random(3) < 0.3
            active[G - 1] = day == 2
            p = share * active
            p = p / p.sum()
            y = rng.choice(G, size=lam, p=p)
            hard = rng.random(lam) < 0.1
            rows = [(int(c), bool(h)) for c, h in zip(y, hard)]
            pol.cycle(rows, shift_signal=rng.random() < 0.15)
            fifo.extend(rows)
            for c in range(G):
                if active[c]:
                    last_seen[c] = day
                max_abs[c] = max(max_abs[c], day - last_seen[c])
            n = pol.counts()
            need = np.minimum(pol.q, pol.seen)
            viol += int((n < need).sum())
            min_slack = min(min_slack, int((n - need).min()))
            max_size = max(max_size, pol.size())
            kH_max = max(kH_max, len(pol.H))
            kF_min = min(kF_min, pol.kF)
            kR_seen.add(pol.kR)
            nf = np.bincount([r[0] for r in fifo], minlength=G)
            fifo_lost += int(((nf == 0) & (pol.seen >= pol.q)).sum())
            tc_lost += int(((n == 0) & (pol.seen >= pol.q)).sum())
        out.append(dict(**cf, days=days, lam=lam, invariant_violations=viol, min_slack_N_minus_min_q_M=min_slack,
                        max_context_size=max_size, max_H_size=kH_max, H_cap=0.1 * cf["k"],
                        min_kF=kF_min, G_qmin=cf["G"] * cf["q"],
                        max_kR_possible=cf["k"] - pol.kH - cf["G"] * cf["q"], kR_without_floor=cf["k"] - pol.kH,
                        kR_values_observed=sorted(kR_seen),
                        kR_hybrid_formula=[pol.k - pol.kH - pol.kF0, pol.k - pol.kH - pol.kF0
                                           + min(int(pol.delta * pol.kF0), pol.kF0 - cf["G"] * cf["q"])],
                        max_absence_days_per_group=max_abs.tolist(),
                        group_days_admitted_but_absent_from_FIFO=fifo_lost,
                        group_days_admitted_but_absent_from_3C=tc_lost))
    return out


# ----------------------------------------------------------------------------- P_miss and q*
BASE = dict(K=4500, G=10, n_rec=3, lam=2000.0, r=0.002, theta=0.1, rho=1 / 14, mu=2.5e-4, r_e=0.002, rho_e=0.05)


def mc_pmiss(P, qs, reps, key):
    """Row-level simulation of one reappearance: an active stretch of K rows (c w.p. r), then an absence of
    J = floor(lam*A) rows, A ~ Exp(rho) days, then the reappearance. Context = FIFO window of K-Gq rows + q floor
    rows of c. Each c-row covers the reappearing variant independently w.p. theta; miss = no covering row.
    Common random numbers across q."""
    rng = ss(5, key)
    K, G = P["K"], P["G"]
    miss = np.zeros((len(qs),), float)
    miss_sq = np.zeros_like(miss)
    done = 0
    while done < reps:
        m = min(5000, reps - done)
        rows_c = rng.random((m, K)) < P["r"]
        cover = rng.random((m, K)) < P["theta"]
        cov = rows_c & cover                                # covering c-rows, index K-1 = last active row
        has = cov.any(1)
        d0 = np.where(has, np.argmax(cov[:, ::-1], 1), 10 ** 9)   # rows back from the last active row
        A = rng.exponential(1 / P["rho"], m)
        Jrows = np.floor(P["lam"] * A).astype(np.int64)
        f0 = rng.geometric(P["theta"], m) - 1               # first covering floor slot (0-based)
        for i, q in enumerate(qs):
            kR = K - G * q
            ms = (Jrows + d0 >= kR) & (f0 >= q)
            miss[i] += ms.sum()
        done += m
    p = miss / reps
    se = np.sqrt(p * (1 - p) / reps)
    return p, se


def mc_emerging(P, qs, reps, key):
    rng = ss(6, key)
    out = []
    for q in qs:
        kR = P["K"] - P["G"] * q
        hits = 0
        done = 0
        while done < reps:
            m = min(5000, reps - done)
            cov = (rng.random((m, kR)) < P["r_e"]) & (rng.random((m, kR)) < P["theta"])
            hits += int((~cov.any(1)).sum())
            done += m
        out.append(hits / reps)
    return np.array(out)


def check_pmiss_qstar():
    P = BASE
    reps = 40000 if QUICK else 200000
    qmax = P["K"] // P["G"]
    qs = sorted(set([0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 60, 100, 200, qmax]))
    p_mc, se = mc_pmiss(P, qs, reps, 0)
    p_th = np.array([pmiss(q, P["K"], P["G"], P["r"], P["theta"], P["rho"], P["lam"]) for q in qs])
    se_th = np.sqrt(p_th * (1 - p_th) / reps)
    zz = (p_mc - p_th) / np.where(se_th > 0, se_th, np.inf)
    pm = dict(params=P, reps=reps, q=qs, mc=p_mc.tolist(), se=se.tolist(), theory=p_th.tolist(), z=zz.tolist(),
              max_abs_z=float(np.max(np.abs(zz))),
              robust_bound_holds=bool(np.all(p_th <= (1 - P["theta"]) ** np.array(qs) + 1e-15)))

    # extra MC points away from the flushed regime (short absences): lam small so lam/rho ~ K
    extra = []
    for j, ch in enumerate([dict(lam=300.0), dict(lam=300.0, r=0.01), dict(rho=1.0, lam=500.0)]):
        Q = {**P, **ch}
        qq = [0, 5, 20, 60]
        pmc, sse = mc_pmiss(Q, qq, reps // 2, 10 + j)
        th = np.array([pmiss(q, Q["K"], Q["G"], Q["r"], Q["theta"], Q["rho"], Q["lam"]) for q in qq])
        sth = np.sqrt(th * (1 - th) / (reps // 2))
        extra.append(dict(change=ch, q=qq, mc=pmc.tolist(), se=sse.tolist(), theory=th.tolist(),
                          z=((pmc - th) / np.where(sth > 0, sth, np.inf)).tolist()))
    pm["extra_points"] = extra

    # q*: exact enumeration vs robust closed form; MC argmin on a dense grid around q*
    q_ex, vals = qstar_exact(P)
    q_rb, q_rb_cont = qstar_robust(P)
    grid = list(range(max(0, q_ex - 12), min(qmax, q_ex + 12) + 1))
    p_g, se_g = mc_pmiss(P, grid, reps, 1)
    J_mc = P["n_rec"] * P["rho"] * p_g + P["mu"] * P["G"] * np.array(grid)
    J_se = P["n_rec"] * P["rho"] * se_g
    i_min = int(np.argmin(J_mc))
    i_th = grid.index(q_ex)
    # conservative (unpaired) test of J_mc(q*_theory) - min J_mc
    gap = float(J_mc[i_th] - J_mc[i_min])
    gap_se = float(math.sqrt(J_se[i_th] ** 2 + J_se[i_min] ** 2))
    q2, q2c = qstar2_flushed(P)
    q2_ex, _ = qstar_exact(P, J2_obj)
    em_q = [0, q_ex, q2_ex, 100]
    em_mc = mc_emerging(P, em_q, reps // 4, 0)
    em_th = [(1 - P["theta"] * P["r_e"]) ** (P["K"] - P["G"] * q) for q in em_q]
    qs_out = dict(params=P, q_exact=q_ex, J_at_q_exact=vals[q_ex], J_at_0=vals[0],
                  q_robust_closed_form=q_rb, q_robust_continuous=q_rb_cont,
                  J_at_q_robust=vals[q_rb],
                  mc_grid=grid, mc_J=J_mc.tolist(), mc_J_se=J_se.tolist(), mc_argmin=grid[i_min],
                  gap_J_theory_vs_mc_argmin=gap, gap_se=gap_se, gap_over_se=gap / gap_se if gap_se else None,
                  pmiss_fifo_only_q0=float(p_th[0]), pmiss_at_q_exact=pmiss(q_ex, P["K"], P["G"], P["r"],
                                                                            P["theta"], P["rho"], P["lam"]),
                  variant_emerging=dict(q_exact=q2_ex, q_flushed_closed_form=q2, q_flushed_continuous=q2c,
                                        emerging_q=em_q, emerging_mc=em_mc.tolist(), emerging_theory=em_th,
                                        emerging_reps=reps // 4))
    return pm, qs_out


def sensitivity():
    sweeps = dict(K=[1500, 3000, 4500, 9000, 18000], G=[3, 5, 10, 20, 40], lam=[100, 300, 1000, 2000, 10000],
                  r=[1e-4, 1e-3, 2e-3, 1e-2, 5e-2], rho=[1 / 60, 1 / 30, 1 / 14, 1 / 7, 1 / 2, 2.0],
                  theta=[0.02, 0.05, 0.1, 0.3, 0.6], mu=[2.5e-5, 2.5e-4, 2.5e-3])
    out = {}
    for name, vals in sweeps.items():
        rows = []
        for v in vals:
            Q = {**BASE, name: v}
            qe, _ = qstar_exact(Q)
            qr, qc = qstar_robust(Q)
            q2, _ = qstar_exact(Q, J2_obj)
            rows.append(dict(value=v, q_exact=qe, q_robust=qr, q_robust_cont=qc, q_emerging_variant=q2,
                             H_at_q_exact=H(Q["K"] - Q["G"] * qe, Q["r"], Q["theta"], Q["rho"] / Q["lam"])))
        out[name] = rows
    return out


def check_theorem_bound(n_sets=300):
    """Theorem 2 (q*_exact <= q_rob, and q_rob minimax) on random parameter sets."""
    rng = ss(7)
    bad, gaps = 0, []
    for _ in range(n_sets):
        Q = dict(K=int(rng.integers(500, 20000)), G=int(rng.integers(2, 40)), n_rec=int(rng.integers(1, 6)),
                 lam=float(10 ** rng.uniform(1.5, 4.5)), r=float(10 ** rng.uniform(-4, -1)),
                 theta=float(rng.uniform(0.01, 0.8)), rho=float(10 ** rng.uniform(-2, 0.5)),
                 mu=float(10 ** rng.uniform(-6, -2)))
        qe, vals = qstar_exact(Q)
        qr, _ = qstar_robust(Q)
        bad += int(qe > qr)
        gaps.append(vals[qr] - vals[qe])
    return dict(n_sets=n_sets, violations_q_exact_gt_q_robust=bad, max_J_gap_using_q_robust=float(max(gaps)),
                median_J_gap_using_q_robust=float(np.median(gaps)))


# ----------------------------------------------------------------------------- D1 + D3: hybrid allocation, primary rule
# D1 (hybrid): default ratios fix k_R0 = k - k_H - k_F0 and k_F0; the floor costs nothing in R in the normal state.
# In a shift state (fraction phi of the time) R grows to k_R0 + min(delta*k_F0, k_F0 - G q): the floor caps the shift.
# D3: primary objective = missed recurring reappearances + missed emerging families, per day (no price mu).
BASE_H = dict(k=5000, wR=0.5, wF=0.4, wH=0.1, delta=0.8, phi=0.2, G=10, n_rec=3, lam=2000.0, r=0.002, theta=0.1,
              rho=1 / 14, r_e=0.002, rho_e=0.05)


def hyb_caps(P):
    kH = int(math.floor(P["wH"] * P["k"]))
    kF0 = int(round(P["wF"] * P["k"]))
    return P["k"] - kH - kF0, kF0, kH


def kR_shift(q, P):
    kR0, kF0, _ = hyb_caps(P)
    return kR0 + min(int(P["delta"] * kF0), kF0 - P["G"] * q)


def q_thr(P):
    _, kF0, _ = hyb_caps(P)
    return max(0, math.floor((kF0 - int(P["delta"] * kF0)) / P["G"]))


def q_cap(P):
    return hyb_caps(P)[1] // P["G"]


def pmiss_h(q, P):
    kR0 = hyb_caps(P)[0]
    s = P["rho"] / P["lam"]
    h = (1 - P["phi"]) * H(kR0, P["r"], P["theta"], s) + P["phi"] * H(kR_shift(q, P), P["r"], P["theta"], s)
    return (1 - P["theta"]) ** q * h


def emiss_h(q, P):
    kR0 = hyb_caps(P)[0]
    u = 1 - P["theta"] * P["r_e"]
    return (1 - P["phi"]) * u ** kR0 + P["phi"] * u ** kR_shift(q, P)


def Jh(q, P):
    return P["n_rec"] * P["rho"] * pmiss_h(q, P) + P["rho_e"] * emiss_h(q, P)


def qstar_h(P):
    vals = [Jh(q, P) for q in range(q_cap(P) + 1)]
    return int(np.argmin(vals)), vals


def qrob_h(P):
    """Closed form with H = 1 (worst case over absence laws), clipped to [q_thr, q_cap]."""
    kR0, kF0, _ = hyb_caps(P)
    beta = -math.log(1 - P["theta"])
    ae = -math.log(1 - P["theta"] * P["r_e"])
    A = P["n_rec"] * P["rho"]
    lo, hi = q_thr(P), q_cap(P)
    if P["phi"] == 0:
        return hi, float("inf")
    qc = (math.log(beta * A / (P["phi"] * P["rho_e"] * ae * P["G"])) + ae * (kR0 + kF0)) / (beta + ae * P["G"])
    f = lambda q: A * (1 - P["theta"]) ** q + P["rho_e"] * emiss_h(q, P)
    kink = (kF0 - int(P["delta"] * kF0)) / P["G"]      # real q where the floor starts to cap the shift
    qe = max(qc, kink)                                     # real minimiser of the convex worst-case objective
    cands = sorted(set(min(max(c, lo), hi) for c in (math.floor(qe), math.ceil(qe))))
    return min(cands, key=f), qc


def mc_hybrid(P, qs, reps, key):
    """Row-level MC of one reappearance and one emerging family under the hybrid capacities.
    State: shift with prob. phi (independent). Recurring family as in mc_pmiss; emerging family: FIFO window of k_R
    rows, each an emerging-family row w.p. r_e that covers w.p. theta."""
    rng = ss(8, key)
    kR0, kF0, _ = hyb_caps(P)
    Kmax = kR0 + int(P["delta"] * kF0)
    miss = np.zeros(len(qs))
    emiss = np.zeros(len(qs))
    done = 0
    while done < reps:
        m = min(4000, reps - done)
        cov = (rng.random((m, Kmax)) < P["r"]) & (rng.random((m, Kmax)) < P["theta"])
        has = cov.any(1)
        d0 = np.where(has, np.argmax(cov[:, ::-1], 1), 10 ** 9)
        Jrows = np.floor(P["lam"] * rng.exponential(1 / P["rho"], m)).astype(np.int64)
        f0 = rng.geometric(P["theta"], m) - 1
        ecov = (rng.random((m, Kmax)) < P["r_e"]) & (rng.random((m, Kmax)) < P["theta"])
        ehas = ecov.any(1)
        e0 = np.where(ehas, np.argmax(ecov[:, ::-1], 1), 10 ** 9)   # newest covering emerging row, rows back
        shift = rng.random(m) < P["phi"]
        for i, q in enumerate(qs):
            kR = np.where(shift, kR_shift(q, P), kR0)
            miss[i] += ((Jrows + d0 >= kR) & (f0 >= q)).sum()
            emiss[i] += (e0 >= kR).sum()
        done += m
    return miss / reps, emiss / reps


def check_hybrid():
    P = BASE_H
    reps = 40000 if QUICK else 200000
    kR0, kF0, kH = hyb_caps(P)
    lo, hi = q_thr(P), q_cap(P)
    qe, vals = qstar_h(P)
    qr, qc = qrob_h(P)
    qs = sorted(set([0, 10, lo, qe, 60, 100, 150, hi]))
    pm, em = mc_hybrid(P, qs, reps, 0)
    pth = np.array([pmiss_h(q, P) for q in qs])
    eth = np.array([emiss_h(q, P) for q in qs])
    zp = ((pm - pth) / np.sqrt(np.maximum(pth * (1 - pth), 1e-300) / reps)).tolist()
    ze = ((em - eth) / np.sqrt(np.maximum(eth * (1 - eth), 1e-300) / reps)).tolist()
    grid = list(range(max(0, qe - 12), min(hi, qe + 12) + 1))
    pg, eg = mc_hybrid(P, grid, reps, 1)
    Jmc = P["n_rec"] * P["rho"] * pg + P["rho_e"] * eg
    Jse = np.sqrt((P["n_rec"] * P["rho"]) ** 2 * pg * (1 - pg) / reps + P["rho_e"] ** 2 * eg * (1 - eg) / reps)
    i_min = int(np.argmin(Jmc))
    gap = float(Jmc[grid.index(qe)] - Jmc[i_min])
    gap_se = float(math.sqrt(Jse[grid.index(qe)] ** 2 + Jse[i_min] ** 2))
    return dict(params=P, kR0=kR0, kF0=kF0, kH=kH, kR_shift_no_floor=kR0 + int(P["delta"] * kF0),
                q_thr=lo, q_cap=hi, q_exact=qe, q_robust=qr, q_robust_continuous=qc,
                J_at_q_exact=vals[qe], J_at_q_thr=vals[lo], J_at_0=vals[0], J_at_q_robust=vals[qr],
                kR_shift_at_q_exact=kR_shift(qe, P), pmiss_at_0=pmiss_h(0, P), pmiss_at_q_exact=pmiss_h(qe, P),
                F_share_left_for_proportional_at_q_exact=kF0 - P["G"] * qe,
                mc_q=qs, mc_reps=reps, mc_pmiss=pm.tolist(), th_pmiss=pth.tolist(), z_pmiss=zp,
                mc_emiss=em.tolist(), th_emiss=eth.tolist(), z_emiss=ze,
                mc_grid=grid, mc_J=Jmc.tolist(), mc_J_se=Jse.tolist(), mc_argmin=grid[i_min],
                gap_J_theory_vs_mc_argmin=gap, gap_over_se=gap / gap_se if gap_se else None)


def sensitivity_h():
    sweeps = dict(k=[2000, 5000, 10000, 20000], G=[3, 5, 10, 20, 40], lam=[100, 300, 1000, 2000, 10000],
                  r=[1e-4, 1e-3, 2e-3, 1e-2, 5e-2], rho=[1 / 60, 1 / 30, 1 / 14, 1 / 7, 1 / 2, 2.0],
                  theta=[0.02, 0.05, 0.1, 0.3, 0.6], phi=[0.0, 0.05, 0.2, 0.5, 1.0], delta=[0.2, 0.5, 0.8, 1.0],
                  rho_e=[0.005, 0.05, 0.5, 5.0])
    out = {}
    for name, vals in sweeps.items():
        rows = []
        for v in vals:
            Q = {**BASE_H, name: v}
            qe, _ = qstar_h(Q)
            qr, qc = qrob_h(Q)
            rows.append(dict(value=v, q_thr=q_thr(Q), q_cap=q_cap(Q), q_exact=qe, q_robust=qr,
                             q_robust_cont=qc))
        out[name] = rows
    return out


def check_theorem_h(n_sets=300):
    """Theorem 3: q_thr <= q*_exact <= q_rob (hybrid, same-currency objective) on random parameter sets."""
    rng = ss(9)
    bad_lo, bad_hi, gaps = 0, 0, []
    for _ in range(n_sets):
        wH = float(rng.uniform(0.0, 0.1))
        wF = float(rng.uniform(0.1, 0.6))
        Q = dict(k=int(rng.integers(1000, 20000)), wH=wH, wF=wF, wR=1 - wH - wF, delta=float(rng.uniform(0, 1)),
                 phi=float(rng.uniform(0.01, 1)), G=int(rng.integers(2, 40)), n_rec=int(rng.integers(1, 6)),
                 lam=float(10 ** rng.uniform(1.5, 4.5)), r=float(10 ** rng.uniform(-4, -1)),
                 theta=float(rng.uniform(0.01, 0.8)), rho=float(10 ** rng.uniform(-2, 0.5)),
                 r_e=float(10 ** rng.uniform(-4, -1)), rho_e=float(10 ** rng.uniform(-3, 1)))
        qe, vals = qstar_h(Q)
        qr, _ = qrob_h(Q)
        tol = 1e-12 * max(1.0, abs(vals[qe]))           # ties below float resolution are not violations
        lo = min(q_thr(Q), q_cap(Q))
        bad_lo += int(qe < lo and vals[lo] > vals[qe] + tol)
        bad_hi += int(qe > qr and vals[qr] > vals[qe] + tol)
        gaps.append(vals[qr] - vals[qe])
    return dict(n_sets=n_sets, violations_q_exact_lt_q_thr=bad_lo, violations_q_exact_gt_q_robust=bad_hi,
                max_J_gap_using_q_robust=float(max(gaps)), median_J_gap_using_q_robust=float(np.median(gaps)))


def main():
    res = dict(seed=SEED, quick=QUICK, numpy=np.__version__)
    res["a_fifo"] = check_a()
    res["b_uniform"] = check_b()
    res["c_policy"] = check_c()
    res["pmiss"], res["qstar"] = check_pmiss_qstar()
    res["sensitivity"] = sensitivity()
    res["theorem_bound"] = check_theorem_bound()
    res["hybrid"] = check_hybrid()
    res["hybrid_sensitivity"] = sensitivity_h()
    res["hybrid_theorem"] = check_theorem_h()
    zs = [x["z"] for x in res["a_fifo"]["full_bin"]] + [x["z_P0"] for x in res["a_fifo"]["absence"]]
    zs += [x["z"] for x in res["b_uniform"]] + [x["z_P0"] for x in res["b_uniform"]]
    zs += res["pmiss"]["z"] + [v for e in res["pmiss"]["extra_points"] for v in e["z"]]
    em = res["qstar"]["variant_emerging"]
    n_e = em["emerging_reps"]
    zs += [(m - t) / math.sqrt(t * (1 - t) / n_e) for m, t in zip(em["emerging_mc"], em["emerging_theory"])]
    zs += res["hybrid"]["z_pmiss"] + res["hybrid"]["z_emiss"]
    zs = [v for v in zs if v is not None]
    res["summary"] = dict(n_z=len(zs), max_abs_z=max(abs(v) for v in zs),
                          n_abs_z_gt_2=sum(abs(v) > 2 for v in zs), n_abs_z_gt_3=sum(abs(v) > 3 for v in zs),
                          expected_n_abs_z_gt_2_if_theory_true=round(0.0455 * len(zs), 2),
                          c_invariant_violations=sum(x["invariant_violations"] for x in res["c_policy"]),
                          qstar_mc_argmin_equals_theory=res["qstar"]["mc_argmin"] == res["qstar"]["q_exact"],
                          hybrid_mc_argmin_equals_theory=res["hybrid"]["mc_argmin"] == res["hybrid"]["q_exact"],
                          hybrid_theory_q_within_2se_of_mc_min=res["hybrid"]["gap_over_se"] < 2)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(OUT)


if __name__ == "__main__":
    main()
