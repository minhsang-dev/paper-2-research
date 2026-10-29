# Proposition 1 (Mệnh đề 1): retention of rare attack families, and the choice of $q_{\min}$

Source of the statement: `context/outline_v14_3_RQ_excerpt.txt`, Nội dung 2, formulas (2-4) and (2-5).
All numbers below come from `prop1_sim.py` (seed 20261029) and are read from `prop1_sim_results.json`; the JSON key is
given in brackets. Status tags: **PROVEN** (full proof below), **VERIFIED-MC** (seeded simulation agrees with the
formula within MC error), **OPEN** (not proven; numerical evidence only, or outside the model).

Motivation only. In Bài 1 the chronological (FIFO) context on CIDDS-001 became one-class after attack-free days
(`context/cidds_untrainable_table.md`). That is an empirical instance of part (a) with $c$ = the whole attack class.
No new numbers are derived from it here.

---

## 0. Setting and assumptions

Trusted rows (labelled by an analyst, passed the filter) arrive one at a time. Each row carries a group label
$Y_i \in \{1,\dots,G\}$ (attack family, or benign). $M_c(t)$ is the number of trusted rows of group $c$ received up to
time $t$; $N_c(t)$ the number of rows of $c$ held in the context; superscripts $R,F,H$ denote the compartments.

* **A1 (i.i.d. labels).** While the system is stationary, $Y_i$ are i.i.d. with $P(Y_i=c)=r$ (for the group of
  interest). Used in (a) first part and in Section 4.
* **A2 (arrival rate).** $\lambda$ trusted rows arrive per day. For the "certain loss" claim only the count matters:
  the number of rows received during the absence is $\ge d\lambda$.
* **A3 (full bin).** At the time of inspection at least $k_R$ rows have been received, so the FIFO bin is full.
* **A4 (trust is permanent).** A row accepted as trusted is never later removed for being untrusted. (If the
  poisoning filter can retro-actively remove rows, (c) holds only with $M_c$ replaced by the number of rows still
  trusted.)
* **A5 (fixed groups).** The set of groups and the group of each stored row do not change. This holds for family labels.
  It does **not** hold for the binary-label variant where groups are clusters that are re-estimated; (c) for that
  variant is **OPEN**.

The three-compartment policy is modelled by the following axioms, which is how we read the outline's specification
(Nội dung 2, steps (1)-(5)); T3 must implement them for (c) to apply.

* **P0 (capacity).** $k = k_R(t)+k_F(t)+k_H$, $k_H \le 0.1k$, $|R|\le k_R(t)$, $|F|\le k_F(t)$, $|H|\le k_H$, and
  $k_F(t) \ge G\,q_{\min}$ at all times (including during a shift of $\delta k_F$ to $R$: step (3) "không giảm nhóm nào
  dưới $q_{\min}$").
* **P1 (quotas).** At every time the quotas $q_c(t)$ are given by (2-4) with largest-remainder rounding, so
  $\sum_c q_c(t)=k_F(t)$.
* **P2 (F eviction).** A row of $c$ leaves $F$ only (i) when $|F_c| > q_c(t)$, or (ii) when it is replaced by
  another row of $c$ (k-center swap).
* **P3 (offer).** Every trusted row enters $R$ or $H$ on arrival, and a row leaving $R$ or $H$ is offered to $F$;
  it is accepted when $|F_c| < q_c(t)$ (making room, if $F$ is full, by evicting a row of an over-quota group).
  Compartments are disjoint.

---

## 1. Part (a): FIFO bin of size $k_R$  — **PROVEN**, **VERIFIED-MC**

**Claim.** Under A1, A3: $N_c^{\rm FIFO}\sim{\rm Bin}(k_R,r)$, so
$$P(N_c^{\rm FIFO}=0)=(1-r)^{k_R}\le e^{-rk_R}. \tag{2-5}$$
Under A2: if $c$ is absent for $d$ days with $d\lambda\ge k_R$, then $N_c^{\rm FIFO}=0$ with probability 1.

*Proof.* The FIFO bin holds exactly the last $k_R$ rows (A3). Under A1 their labels are i.i.d., each equal to $c$ with
probability $r$, so $N_c$ is a sum of $k_R$ i.i.d. Bernoulli($r$) variables and $P(N_c=0)=(1-r)^{k_R}$. Since
$\ln(1-r)\le -r$ for $r\in[0,1)$, $(1-r)^{k_R}=e^{k_R\ln(1-r)}\le e^{-rk_R}$. For the second statement: during the
absence at least $d\lambda\ge k_R$ rows arrive, none of group $c$. The last $k_R$ rows are therefore all from the
absence window, and $N_c=0$ deterministically (no probabilistic assumption is needed). More generally, after an
absence of $J<k_R$ rows the bin holds $k_R-J$ pre-absence rows, so $N_c\sim{\rm Bin}(k_R-J,r)$. ∎

The outline's example ($r=10^{-4}$, $k_R=5000$): exact $(1-r)^{k_R}=0.6065$, bound $e^{-0.5}=0.6065$
[`a_fifo.outline_example`].

| $r$ | $k_R$ | MC $P(N_c=0)$ (20 000 reps) | $(1-r)^{k_R}$ | $e^{-rk_R}$ | z |
|---|---|---|---|---|---|
| 1e-4 | 5000 | 0.6133 | 0.6065 | 0.6065 | 1.98 |
| 1e-3 | 1000 | 0.3666 | 0.3677 | 0.3679 | −0.32 |
| 5e-3 | 500 | 0.0836 | 0.0816 | 0.0821 | 1.05 |

[`a_fifo.full_bin`]. Absence ($r=0.01$, $k_R=1000$, 5 000 reps) [`a_fifo.absence`]: $d=5,\lambda=200$
($d\lambda=k_R$): $N_c=0$ in every replicate (max $N_c=0$). $d=4,\lambda=200$: MC 0.122 vs $(1-r)^{200}=0.134$
(z = −2.49). $d=3,\lambda=300$: MC 0.356 vs 0.366 (z = −1.50).

## 2. Part (b): uniform sampling over the whole history — **PROVEN**, **VERIFIED-MC**

**Claim.** If the context is a uniformly random $k$-subset of the $n\ge k$ history rows, of which $n_c$ are of group
$c$, then $E[N_c]=k\,n_c/n$, and for fixed $n_c$, $E[N_c]\to0$ and $P(N_c=0)\to1$ as $n\to\infty$.

*Proof.* $N_c=\sum_{i:\,Y_i=c}\mathbf 1\{i\in S\}$ and each row is in $S$ with probability $k/n$ (by symmetry of
the uniform $k$-subset), so $E[N_c]=n_c k/n$ by linearity. ($N_c$ is hypergeometric; $P(N_c=0)=\binom{n-n_c}{k}/
\binom{n}{k}$.) By Markov's inequality $P(N_c\ge1)\le E[N_c]=kn_c/n\to0$. Reservoir sampling (Algorithm R) produces
exactly a uniform $k$-subset of the stream seen so far (Vitter, "Random sampling with a reservoir", *ACM TOMS*
11(1):37–57, 1985), so the claim applies to it. ∎

MC uses Algorithm R itself, $k=50$, $n_c=20$, 2 000 reps [`b_uniform`]:

| $n$ | MC $E[N_c]$ (SE) | $kn_c/n$ | MC $P(N_c=0)$ | hypergeometric |
|---|---|---|---|---|
| 200 | 5.056 (0.042) | 5.0 | 0.0025 | 0.0023 |
| 1 000 | 1.006 (0.021) | 1.0 | 0.3415 | 0.3549 |
| 5 000 | 0.202 (0.010) | 0.2 | 0.813 | 0.8176 |
| 20 000 | 0.0515 (0.005) | 0.05 | 0.9495 | 0.9511 |

## 3. Part (c): the three-compartment policy — **PROVEN** (under A4, A5, P0–P3), **VERIFIED-MC**

**Lemma 1 (quota floor).** Under P0–P1, $q_c(t)\ge q_{\min}$ for every $c$ and $t$.

*Proof.* In (2-4) the term $\lfloor (k_F-Gq_{\min})\hat\pi_c^\gamma/\sum_{c'}\hat\pi_{c'}^\gamma\rfloor$ is
$\ge0$ because $k_F(t)\ge Gq_{\min}$ (P0) and the weights are non-negative. Largest-remainder rounding only adds
units. (If all $\hat\pi_c=0$ the ratio is undefined; any convention with non-negative weights, e.g. uniform, keeps the
claim.) ∎

**Lemma 2 (room in F).** If $|F|=k_F(t)$ and $|F_c|<q_c(t)$, some group $c'$ has $|F_{c'}|>q_{c'}(t)$.

*Proof.* $\sum_{c'}|F_{c'}|=k_F=\sum_{c'}q_{c'}$ (P1); if $|F_c|<q_c$ some other term must exceed its quota. ∎
So the eviction in P3 is always possible and, by P2(i), leaves that group at $\ge q_{c'}\ge q_{\min}$.

**Theorem 1.** Under A4, A5, P0–P3, for every group $c$ and every time $t$,
$$N_c(t)\ \ge\ \min\{q_{\min},\,M_c(t)\}.$$
In particular, once $M_c\ge q_{\min}$, $N_c\ge q_{\min}$ for ever, whatever the length of any later absence.
Moreover the floor occupies at most $Gq_{\min}$ slots, and the recent bin is never smaller than it would be without a
floor minus $Gq_{\min}$: its largest admissible size is $k-k_H-Gq_{\min}$ instead of $k-k_H$.

*Proof.* Induction over elementary events. Initially $N_c=M_c=0$. Assume the bound before an event.
* Arrival of a trusted $c$-row: it enters $R$ or $H$ (P3), so $N_c$ and $M_c$ both grow by 1.
* A $c$-row leaves $R$ or $H$: it is offered to $F$ (P3). If $|F_c|<q_c$ it is accepted and $N_c$ is unchanged
  (Lemma 2 guarantees room). Otherwise $|F_c|\ge q_c\ge q_{\min}$ (Lemma 1) and afterwards $N_c\ge|F_c|\ge q_{\min}$.
* A $c$-row leaves $F$: by P2 either it is replaced by a $c$-row ($N_c$ unchanged) or $|F_c|>q_c$ before the
  removal, so afterwards $|F_c|\ge q_c\ge q_{\min}$.
* Quota updates and capacity shifts change no row by themselves; the evictions they trigger are of the two kinds
  above. A4 and A5 exclude any other way for a row to leave group $c$ or the context.

In every case the bound holds after the event; the events are the only ways $N_c$ or $M_c$ change, and none refers to
the time since $c$ was last seen. For the cost: guaranteed rows live in $F$, at most $q_{\min}$ per group are guaranteed,
so the guarantee uses at most $Gq_{\min}$ slots; P0 forces $k_F\ge Gq_{\min}$, so
$k_R(t)=k-k_H-k_F(t)\le k-k_H-Gq_{\min}$, and the shift rule can raise $k_R$ to this value but not beyond. ∎

*Reading of "the recent bin shrinks by at most $Gq_{\min}$".* With the default ratios $w_R:w_F:w_H$ fixed,
$k_R=w_Rk$ does not depend on $q_{\min}$ at all, and the price of the floor is paid inside $F$ (less room for
the proportional share of common groups). The statement "shrinks by at most $Gq_{\min}$" is the comparison with
a design that gives all non-hard capacity to recency, $k_R=k-k_H$. Section 4 uses that comparison:
$k_R(q)=K-Gq$ with $K=k-k_H$. Which reading the thesis adopts is a decision for the supervisor (see REPORT.md).

**MC check of Theorem 1** [`c_policy`]: an explicit implementation of P0–P3 (random within-group choice in place of
k-center, which does not affect counts; random shift signals; 10% of rows routed to $H$), 60 days at $\lambda=150$,
groups recurring with random absences and one group seen only on day 2. Three configurations
($k,G,q_{\min}$) = (400, 6, 10), (1000, 8, 20), (1000, 8, 50): **0 violations** of $N_c\ge\min(q_{\min},M_c)$,
maximal context size $\le k$, $|H|\le0.1k$, $k_F\ge Gq_{\min}$ throughout. The group seen once was absent for 57 days
and kept its floor. A FIFO of the same $k$ on the same stream lost an admitted group on 124, 1 and 11 group-days
respectively; the three-compartment context lost none.

---

## 4. Choosing $q_{\min}$

### 4.1 Recurrence model (assumptions R1–R5)

* **R1 (Poisson recurrence).** Reappearances of a recurring family $c$ form a Poisson process of rate $\rho_c$ per day.
  At a reappearance the time $A$ since $c$ was last active is ${\rm Exp}(\rho_c)$ (memorylessness).
* **R2.** During the absence $J=\lfloor\lambda A\rfloor$ rows arrive, none of $c$. Then
  $P(J\ge j)=e^{-sj}$ with $s=\rho_c/\lambda$ (geometric).
* **R3.** The active stretch before the absence is at least $k_R$ rows long and satisfies A1 with share $r_c$.
  (If it is shorter, the recent bin holds fewer $c$-rows and the miss probability below is a lower bound.)
* **R4 (coverage).** The floor holds exactly $q$ rows of $c$ (the guaranteed minimum; conservative for rare families
  whose proportional share in (2-4) rounds to 0). Each stored $c$-row "covers" the reappearing variant independently
  with probability $\theta\in(0,1]$; the family is **missed** if no stored row covers it.
  $\theta$ is a modelling parameter not in the outline (θ = 1: one row suffices). It has to be estimated or fixed by
  decision.
* **R5 (capacity).** $k_R(q)=K-Gq$, $q\in\{0,\dots,\lfloor K/G\rfloor\}$ (Section 3 reading). $q=0$ means no floor.

### 4.2 Miss probability — **PROVEN**, **VERIFIED-MC**

**Proposition 2.** Under R1–R5, with $u=1-r_c\theta$, $b=e^{-s}$, $m=K-Gq$,
$$P_{\rm miss}(q)=(1-\theta)^q\,H(m),\qquad H(m)=E\!\left[u^{(m-J)^+}\right]=b^m+(1-b)\,u\,\frac{u^m-b^m}{u-b}\quad(u\neq b),$$
and $H(m)=b^m+(1-b)\,m\,b^m$ if $u=b$; $H(0)=1$. Also $H\le1$, $H$ is non-increasing in $m$, and $H(m)=1$ whenever
$J\ge m$ a.s.

*Proof.* Coverage events are independent across rows (R4). The floor contributes no cover with probability
$(1-\theta)^q$. Given $J$, the recent bin holds $(m-J)^+$ pre-absence rows (Section 1), each a covering $c$-row
independently with probability $r_c\theta$, so no cover there has conditional probability $u^{(m-J)^+}$.
Independence of floor and recent coverage gives the product. With $P(J=j)=(1-b)b^j$:
$H(m)=\sum_{j<m}(1-b)b^ju^{m-j}+b^m=(1-b)u^m\sum_{j<m}(b/u)^j+b^m$, which sums to the stated form. $(m-J)^+$ is
non-decreasing in $m$ and $u\le1$, so $H$ is non-increasing; $u^{x}\le1$ for $x\ge0$. ∎

MC [`pmiss`]: row-level simulation (actual Bernoulli rows, actual exponential absences, actual geometric floor
coverage), 200 000 reappearances, base parameters $K=4500$, $G=10$, 3 recurring families, $\lambda=2000$/day,
$r=0.002$, $\theta=0.1$, $\rho=1/14$/day. These are **illustrative values, not calibrated** to any dataset.
14 values of $q$ from 0 to 450: largest $|z|=1.82$ (at $q=100$, where theory is 2.6e-5). Selected points: $q=0$: MC
0.94835 vs 0.94825; $q=10$: 0.33131 vs 0.33131; $q=20$: 0.11561 vs 0.11575. Three extra settings with shorter
absences in rows or larger $r$ ($\lambda=300$; $\lambda=300,r=0.01$; $\rho=1,\lambda=500$), where $H$ is far from 1:
12 points, largest $|z|=2.43$
[`pmiss.extra_points`].

### 4.3 Objective

Per day, the expected number of missed reappearances plus a price $\mu$ per recent slot given to the floor:
$$\boxed{\;\mathcal J(q)=\sum_{c\in\mathcal R}\rho_c\,(1-\theta)^q\,H_c(K-Gq)\;+\;\mu\,G\,q\;},\qquad
q^*=\arg\min_{0\le q\le\lfloor K/G\rfloor}\mathcal J(q),$$
$\mathcal R$ = recurring families. $\mu$ converts one recency slot into "missed reappearances per day". It is the
Lagrange multiplier of a budget "the floor may take at most $B$ recency slots". It is a value judgement and must be
fixed by decision (or replaced by the variant in 4.6).

**Computable rule (exact).** Enumerate $q=0,\dots,q_{\rm rob}$ (see Theorem 2: the search can stop at $q_{\rm rob}$)
and take the smallest minimiser. Cost $O(q_{\rm rob})$ closed-form evaluations.

### 4.4 Closed form: the robust (minimax) floor — **PROVEN**

Let $\beta=-\ln(1-\theta)$, $A=\sum_{c\in\mathcal R}\rho_c$, and
$$\tilde q=\frac1\beta\ln\frac{\beta A}{\mu G},\qquad q_{\rm rob}=\arg\min_{q\in\{\lfloor\tilde q\rfloor,\lceil\tilde q\rceil\}\cap[0,\lfloor K/G\rfloor]}\ \bar{\mathcal J}(q),\qquad \bar{\mathcal J}(q)=A(1-\theta)^q+\mu Gq.$$

**Theorem 2.**
(i) $q_{\rm rob}$ minimises $\sup\mathcal J(q)$ over all absence laws (Poisson or not), and equals $q^*$ when absences
always exceed the recent window ($J\ge K$ a.s.).
(ii) Under R1–R5 the smallest minimiser of $\mathcal J$ satisfies $q^*\le q_{\rm rob}$.

*Proof.* (i) $H\le1$ with equality when $J\ge m$ a.s. (Prop. 2), so $\sup_{\rm law}\mathcal J=\bar{\mathcal J}$.
$\bar{\mathcal J}$ is convex in real $q$ (exponential plus linear), its stationary point solves
$A\beta e^{-\beta q}=\mu G$, i.e. $q=\tilde q$. For a convex function an integer minimiser is the floor or the ceiling
of the real minimiser, clipped to the feasible range.
(ii) Write $h(q)=H(K-Gq)$, non-decreasing in $q$ and $\le1$ (Prop. 2). The decrease of the miss term from $q$ to
$q+1$ is $A(1-\theta)^q[h(q)-(1-\theta)h(q+1)]\le A(1-\theta)^q[h(q)-(1-\theta)h(q)]\le A(1-\theta)^q\theta$
(using $h(q+1)\ge h(q)$, then $h\le1$), which is the decrease of $\bar{\mathcal J}$'s miss term. (With heterogeneous
families apply this term by term.) For $q\ge q_{\rm rob}$ convexity of $\bar{\mathcal J}$ gives $A(1-\theta)^q\theta\le\mu G$,
hence $\mathcal J(q+1)-\mathcal J(q)\ge\mu G-A(1-\theta)^q\theta\ge0$. So $\mathcal J$ is non-decreasing on
$[q_{\rm rob},\infty)$ and its smallest minimiser is $\le q_{\rm rob}$. ∎

**Dependence of $q_{\rm rob}$ (from the formula):** grows like $\ln$ of the total recurrence rate $A$; falls like
$\ln G$ and $\ln\mu$; does not depend on $k$, $\lambda$ or $r$ except through the cap $\lfloor K/G\rfloor$. That is
the price of robustness: in the worst case the recent bin has been flushed. It is non-monotone in $\theta$:
$\tilde q$ is small both when one row suffices ($\theta\to1$) and when rows are nearly useless ($\theta\to0$).

### 4.5 Results and dependence of the exact $q^*$

Base setting [`qstar`] ($\mu=2.5\times10^{-4}$): $q^*=20$ (exact enumeration), $\tilde q=20.89$, $q_{\rm rob}=21$;
$\mathcal J(q^*)=0.0748$ vs $\mathcal J(21)=0.0748$ vs $\mathcal J(0)$ (no floor) $=0.2032$ missed reappearances/day.
$P_{\rm miss}$ falls from 0.948 (FIFO only, $q=0$) to 0.116 at $q^*$, at the cost of $Gq^*=200$ of $K=4500$
recency slots. **MC argmin**: on the grid $q=8,\dots,32$ (200 000 reps, common random numbers) the MC objective is
minimised at $q=20$, the same as theory [`qstar.mc_argmin`, `summary.qstar_mc_argmin_equals_theory`].

Theorem 2(ii) on 300 random parameter sets: 0 cases with $q^*>q_{\rm rob}$; using $q_{\rm rob}$ instead of $q^*$
costs a median 5.7e-5 and at most 1.97 in $\mathcal J$ [`theorem_bound`]. The large cases are short-absence settings,
where the recent bin often still holds the family and $q_{\rm rob}$ over-reserves.

Exact $q^*$ when one parameter moves from the base [`sensitivity`] (numerical, base setting only):

| parameter | values | $q^*$ |
|---|---|---|
| $K$ | 1500, 3000, 4500, 9000, 18000 | 21, 21, 20, 19, 17 |
| $G$ | 3, 5, 10, 20, 40 | 32, 27, 20, 14, 7 |
| $\lambda$ | 100, 300, 1000, 2000, 10000 | 15, 18, 20, 20, 21 |
| $r$ | 1e-4, 1e-3, 2e-3, 1e-2, 5e-2 | 21, 21, 20, 20, 19 |
| $\rho$ (/day) | 1/60, 1/30, 1/14, 1/7, 1/2, 2 | 7, 13, 20, 27, 37, 47 |
| $\theta$ | 0.02, 0.05, 0.1, 0.3, 0.6 | 26, 28, 20, 9, 5 |
| $\mu$ | 2.5e-5, 2.5e-4, 2.5e-3 | 42, 20, 0 |

Reading: more recurring traffic ($\rho$) → larger floor; more groups ($G$) → each floor slot is $G$ times dearer →
smaller floor. Larger $k$, larger $r$ or slower arrival $\lambda$ let the recent bin itself keep the family through
short absences → smaller floor. As $\lambda\to\infty$ (absences flush the bin) $q^*\to q_{\rm rob}$.
**OPEN:** these monotone trends are observed numerically, not proven for the exact $q^*$; only $q^*\le q_{\rm rob}$ and
the dependence of $q_{\rm rob}$ are proven.

### 4.6 Variant without a price $\mu$ (same currency) — **PROVEN** closed form in the flushed regime

Replace $\mu Gq$ by the miss rate of *emerging* families that are not yet floored. They appear at rate $\rho_e$/day and
can only be caught by the recent bin, where they have share $r_e$:
$\mathcal J_2(q)=\sum_c\rho_c(1-\theta)^qH_c(K-Gq)+\rho_e(1-\theta r_e)^{K-Gq}$. With $H\equiv1$ and
$a_e=-\ln(1-\theta r_e)$, $\mathcal J_2$ is a sum of two convex exponentials and
$$\tilde q_2=\frac{\ln\!\big(\beta A/(a_eG\rho_e)\big)+a_eK}{\beta+a_eG},$$
rounded as in 4.4. Unlike $q_{\rm rob}$ this grows with $K$. Base, with $\rho_e=0.05$, $r_e=0.002$: exact $q^*=58$,
closed form 59 ($\tilde q_2=58.86$) [`qstar.variant_emerging`]. The emerging-family term $(1-\theta r_e)^{k_R}$ was
checked by row-level MC at 4 values of $q$ (50 000 reps; e.g. $q=58$: MC 0.4558 vs 0.4565). The emerging-family
model (i.i.d. share $r_e$ over the whole window) is crude; it is offered only as an alternative to choosing $\mu$.

### 4.7 Remark: threshold learnability

If instead a family is missed iff $N_c<m_0$ (a fixed number of rows needed), then any $q\ge m_0$ removes all misses and
$\mathcal J$ is increasing beyond $m_0$, so $q^*\le m_0$ trivially. The coverage model R4 is used because it gives a
non-trivial trade-off and contains the threshold case $m_0=1$ as $\theta=1$.

---

## 5. Summary of what is proven and what is open

| Statement | Status |
|---|---|
| (a) $P(N_c=0)=(1-r)^{k_R}\le e^{-rk_R}$; certain loss if $d\lambda\ge k_R$ | PROVEN, VERIFIED-MC |
| (b) $E[N_c]=kn_c/n\to0$ (uniform / reservoir) | PROVEN, VERIFIED-MC |
| (c) $N_c\ge\min(q_{\min},M_c)$ for all $t$; floor uses ≤ $Gq_{\min}$ recency slots | PROVEN under A4, A5, P0–P3; VERIFIED-MC |
| (c) for the cluster variant (binary labels, re-clustering) | OPEN |
| (c) if trusted rows can be removed later by the poisoning filter | OPEN (bound holds with $M_c$ = rows still trusted only if P3 also applies to them) |
| Prop. 2: $P_{\rm miss}(q)=(1-\theta)^qH(K-Gq)$ under R1–R5 | PROVEN, VERIFIED-MC |
| Theorem 2: $q_{\rm rob}$ minimax, closed form; $q^*\le q_{\rm rob}$ | PROVEN; checked on 300 random sets |
| Monotonicity of exact $q^*$ in $k,\lambda,r,\rho,G$ | OPEN (numerical only) |
| Unimodality of $\mathcal J$ in the general (non-flushed) regime | OPEN (the enumeration rule does not need it) |
| Values of $\theta$, $\mu$ (or $\rho_e,r_e$), $\rho$ for real data | OPEN: need dev-data estimates or a decision |

MC agreement overall [`summary`]: 43 comparisons of MC vs closed form; largest $|z|=2.49$; 2 with $|z|>2$ (1.96
expected by chance if all formulas are right); none with $|z|>3$. Theorem-1 invariant: 0 violations.
