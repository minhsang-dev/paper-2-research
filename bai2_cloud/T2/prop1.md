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
Moreover the floor occupies at most $Gq_{\min}$ slots. Under the hybrid allocation (D1, below) the recent bin
equals its default size $k_{R0}$ in the normal state and $k_R^{\rm s}(q_{\min})=k_{R0}+\min(\lfloor\delta k_{F0}\rfloor,\,
k_{F0}-Gq_{\min})$ in a shift state. That is never more than $Gq_{\min}$ below the same shift without a floor,
$k_{R0}+\lfloor\delta k_{F0}\rfloor$.

*Proof.* Induction over elementary events. Initially $N_c=M_c=0$. Assume the bound before an event.
* Arrival of a trusted $c$-row: it enters $R$ or $H$ (P3), so $N_c$ and $M_c$ both grow by 1.
* A $c$-row leaves $R$ or $H$: it is offered to $F$ (P3). If $|F_c|<q_c$ it is accepted and $N_c$ is unchanged
  (Lemma 2 guarantees room). Otherwise $|F_c|\ge q_c\ge q_{\min}$ (Lemma 1) and afterwards $N_c\ge|F_c|\ge q_{\min}$.
* A $c$-row leaves $F$: by P2 either it is replaced by a $c$-row ($N_c$ unchanged) or $|F_c|>q_c$ before the
  removal, so afterwards $|F_c|\ge q_c\ge q_{\min}$.
* Quota updates and capacity shifts change no row by themselves; the evictions they trigger are of the two kinds
  above. A4 and A5 exclude any other way for a row to leave group $c$ or the context.

In every case the bound holds after the event. These events are the only ways $N_c$ or $M_c$ change, and none of them
depends on the time since $c$ was last seen. For the cost: guaranteed rows live in $F$, and at most $q_{\min}$ per
group are guaranteed, so the guarantee uses at most $Gq_{\min}$ slots. In the shift state P0 caps the transfer at
$k_{F0}-Gq_{\min}$, so the loss against the unfloored shift is
$\lfloor\delta k_{F0}\rfloor-\min(\lfloor\delta k_{F0}\rfloor,k_{F0}-Gq_{\min})=(Gq_{\min}-(k_{F0}-\lfloor\delta k_{F0}\rfloor))^+\le Gq_{\min}$. ∎

### Decision D1: hybrid allocation (where the floor is paid for)

The default ratios $w_R:w_F:w_H$ fix $k_{R0}=k-k_H-k_{F0}$ and $k_{F0}$. The floor $q_{\min}$ is paid for in two places:

1. **Normal state.** $k_R=k_{R0}$ does not depend on $q_{\min}$. The floor uses $Gq_{\min}$ of the $k_{F0}$ slots of $F$, and
   $k_{F0}-Gq_{\min}$ slots remain for the proportional share of (2-4). Common groups pay this part.
2. **Shift state** (step (3): $\delta k_F$ moves to $R$ for $D$ days, without taking any group below $q_{\min}$). Here the
   recent bin is $k_R^{\rm s}(q)=k_{R0}+\min(\lfloor\delta k_{F0}\rfloor,k_{F0}-Gq)$. The floor costs recency only when
   $q>q_{\rm thr}:=\lfloor(k_{F0}-\lfloor\delta k_{F0}\rfloor)/G\rfloor$, and then by $(Gq-(k_{F0}-\lfloor\delta k_{F0}\rfloor))^+\le Gq$
   slots.

So "the recent bin shrinks by at most $G\,q_{\min}$" holds in both states: by 0 in the normal state and by at most
$Gq_{\min}$ in a shift. The objective in Section 4 prices the recency loss in the shift state, in the same currency
as the miss term (D3). It does **not** price the smaller proportional share left to common groups inside $F$ (OPEN, see
Section 5).

**What D1 changes, and what it does not.** Parts (a), (b), Lemmas 1–2 and the invariant of Theorem 1 do not use the
allocation of capacity between $R$ and $F$. Only P0 ($k_F\ge Gq_{\min}$ at all times) matters, and D1 satisfies it.
They are unchanged. Only the cost sentence of Theorem 1 and Section 4 change. The MC implementation of the policy
already used the hybrid rule ($\delta=0.5$).

**MC check of Theorem 1** [`c_policy`]: the check uses an explicit implementation of P0–P3. It makes a random within-group choice in place of
k-center, which does not affect counts. It also uses random shift signals and routes 10% of rows to $H$. The run lasts 60 days at $\lambda=150$,
with groups recurring after random absences and one group seen only on day 2. Three configurations
($k,G,q_{\min}$) = (400, 6, 10), (1000, 8, 20), (1000, 8, 50) gave **0 violations** of $N_c\ge\min(q_{\min},M_c)$.
The context size stayed $\le k$, $|H|\le0.1k$ and $k_F\ge Gq_{\min}$ throughout. The recent-bin sizes observed
were exactly the two D1 values $\{k_{R0},k_R^{\rm s}\}$: {200, 280}, {500, 700}, and {500} in the third configuration,
where $Gq_{\min}=k_{F0}$ blocks any shift. The group seen once was absent for 57 days and kept its floor. A FIFO of the same
$k$ on the same stream lost an admitted group on 124, 1 and 11 group-days respectively. The three-compartment context
lost none.

---

## 4. Choosing $q_{\min}$ (primary rule: D1 hybrid allocation, D3 same-currency objective)

### 4.1 Recurrence model (assumptions R1–R6)

* **R1 (Poisson recurrence).** Reappearances of a recurring family $c$ form a Poisson process of rate $\rho_c$ per day.
  At a reappearance the time $A$ since $c$ was last active is ${\rm Exp}(\rho_c)$ (memorylessness).
* **R2.** During the absence $J=\lfloor\lambda A\rfloor$ rows arrive, none of $c$. Then
  $P(J\ge j)=e^{-sj}$ with $s=\rho_c/\lambda$ (geometric).
* **R3.** The active stretch before the absence is at least as long as the recent bin and satisfies A1 with share
  $r_c$. If it is shorter, the recent bin holds fewer $c$-rows and the miss probability below is a lower bound.
* **R4 (coverage).** The floor holds exactly $q$ rows of $c$. This is the guaranteed minimum, and it is conservative for rare families
  whose proportional share in (2-4) rounds to 0. Each stored $c$-row "covers" the reappearing variant independently
  with probability $\theta\in(0,1]$. The family is **missed** if no stored row covers it.
  $\theta$ is a modelling parameter that is not in the outline (θ = 1 means one row suffices). Its value is OPEN.
* **R5 (D1 capacities).** $k_R=k_{R0}$ in the normal state and $k_R=k_R^{\rm s}(q)$ in a shift state, as in Section 3.
  $q\in\{0,\dots,\lfloor k_{F0}/G\rfloor\}$ (P0). $q=0$ means no floor.
* **R6 (state).** At a reappearance the policy is in a shift state with probability $\phi$ (long-run fraction of time
  in shift), independently of the absence and of the stored rows. In reality a shift is triggered by drift, and a
  reappearance is a drift, so the two may be correlated. That correlated case is OPEN.

**Emerging families (the cost side, D3).** Families that are not yet floored appear at rate $\rho_e$ per day. Such a family can only
be caught by the recent bin, where each of its $k_R$ rows is a row of the new family with probability $r_e$ and covers it with
probability $\theta$. It is missed with probability $u_e^{k_R}$, $u_e=1-\theta r_e$, by the computation of Section 1.

### 4.2 Miss probabilities — **PROVEN**, **VERIFIED-MC**

**Proposition 2.** Under R1–R6, with $u=1-r_c\theta$, $b=e^{-s}$,
$$P_{\rm miss}(q)=(1-\theta)^q\big[(1-\phi)H(k_{R0})+\phi\,H(k_R^{\rm s}(q))\big],\qquad
H(m)=E\!\left[u^{(m-J)^+}\right]=b^m+(1-b)\,u\,\frac{u^m-b^m}{u-b}\quad(u\neq b),$$
with $H(m)=b^m+(1-b)\,m\,b^m$ if $u=b$, and $H(0)=1$. The emerging-family miss probability is
$E_{\rm miss}(q)=(1-\phi)u_e^{k_{R0}}+\phi\,u_e^{k_R^{\rm s}(q)}$. Also $H\le1$, $H$ is non-increasing in $m$, and
$H(m)=1$ whenever $J\ge m$ a.s.

*Proof.* Condition on the state (R6), which fixes the recent-bin size $m$. Coverage events are independent across rows (R4).
The floor contributes no cover with probability $(1-\theta)^q$. Given $J$, the recent bin holds $(m-J)^+$
pre-absence rows (Section 1), and each is a covering $c$-row independently with probability $r_c\theta$. So no cover there has
conditional probability $u^{(m-J)^+}$. Independence of floor and recent coverage gives the product. With
$P(J=j)=(1-b)b^j$:
$H(m)=\sum_{j<m}(1-b)b^ju^{m-j}+b^m=(1-b)u^m\sum_{j<m}(b/u)^j+b^m$, which sums to the stated form. $(m-J)^+$ is
non-decreasing in $m$ and $u\le1$, so $H$ is non-increasing, and $u^{x}\le1$ for $x\ge0$. Averaging over the state
gives both formulas. ∎

MC [`hybrid`] is a row-level simulation with actual Bernoulli rows, exponential absences, geometric floor coverage and a
Bernoulli($\phi$) state. It used 200 000 reappearances. Base parameters: $k=5000$, $w_R:w_F:w_H=0.5:0.4:0.1$ (so $k_{R0}=2500$,
$k_{F0}=2000$, $k_H=500$), $\delta=0.8$, $\phi=0.2$, $G=10$, 3 recurring families, $\lambda=2000$/day, $r=0.002$,
$\theta=0.1$, $\rho=1/14$/day, $\rho_e=0.05$/day, $r_e=0.002$. These are **illustrative values, not calibrated** to any
dataset. Over 8 values of $q$ from 0 to 200, the largest $|z|$ is 1.97 for $P_{\rm miss}$ and 0.53 for $E_{\rm miss}$. Selected points:
$P_{\rm miss}(0)$ MC 0.97648 vs 0.97640; $P_{\rm miss}(40)$ 0.01442 vs 0.01443; $E_{\rm miss}(200)$ 0.60637 vs 0.60650.

### 4.3 Objective (D3: same currency, primary)

The objective is the expected number of missed families per day, counting recurring families that reappear and emerging families:
$$\boxed{\;\mathcal J(q)=\sum_{c\in\mathcal R}\rho_c\,P_{{\rm miss},c}(q)\;+\;\rho_e\,E_{\rm miss}(q)\;},\qquad
q^*=\min\arg\min_{0\le q\le\lfloor k_{F0}/G\rfloor}\mathcal J(q).$$
Raising $q$ protects recurring families. Above $q_{\rm thr}$ it also shrinks the recent bin during shifts, which makes
emerging families easier to miss. No price parameter is needed. The parameters $\rho_e$ and $r_e$ describe the threat
model and have to be estimated (OPEN).

**Computable rule (exact).** Evaluate $\mathcal J$ in closed form for $q=q_{\rm thr},\dots,q_{\rm rob}$ and take the
smallest minimiser (Theorem 3 shows the minimiser lies in this range). This takes $O(q_{\rm rob}-q_{\rm thr})$ evaluations.

### 4.4 Closed form and bounds — **PROVEN**

Let $\beta=-\ln(1-\theta)$, $a_e=-\ln(1-\theta r_e)$, $A=\sum_{c\in\mathcal R}\rho_c$, and let
$\kappa=(k_{F0}-\lfloor\delta k_{F0}\rfloor)/G$ be the real kink where the floor starts to cap the shift. Define
$$\tilde q=\frac{\ln\!\big(\beta A/(\phi\rho_e a_e G)\big)+a_e\,(k_{R0}+k_{F0})}{\beta+a_eG},\qquad
\bar{\mathcal J}(q)=A(1-\theta)^q+\rho_eE_{\rm miss}(q),$$
and let $q_{\rm rob}$ be whichever of $\lfloor\max(\tilde q,\kappa)\rfloor$ and $\lceil\max(\tilde q,\kappa)\rceil$, clipped to
$[q_{\rm thr},\lfloor k_{F0}/G\rfloor]$, has the smaller $\bar{\mathcal J}$. If $\phi=0$, set $q_{\rm rob}=\lfloor k_{F0}/G\rfloor$.

**Theorem 3.**
(i) $q_{\rm rob}$ minimises $\sup\mathcal J(q)$ over all absence laws, and it equals $q^*$ when absences always exceed the
recent bin.
(ii) Under R1–R6, $q_{\rm thr}\le q^*\le q_{\rm rob}$ (whenever $q_{\rm thr}\le\lfloor k_{F0}/G\rfloor$).

*Proof.* (i) $H\le1$, with equality when $J\ge m$ a.s. (Prop. 2), so $\sup_{\rm law}\mathcal J=\bar{\mathcal J}$. The
function $q\mapsto\min(\lfloor\delta k_{F0}\rfloor,k_{F0}-Gq)$ is concave, and $x\mapsto u_e^{x}$ is convex and decreasing. Their composition
is therefore convex, so $E_{\rm miss}$ and $\bar{\mathcal J}$ are convex in real $q$. On $q\le\kappa$, $E_{\rm miss}$ is
constant and $\bar{\mathcal J}$ is decreasing. On $q\ge\kappa$,
$\bar{\mathcal J}(q)=Ae^{-\beta q}+\rho_e(1-\phi)u_e^{k_{R0}}+\phi\rho_ee^{-a_e(k_{R0}+k_{F0}-Gq)}$, and the
stationary point solves $A\beta e^{-\beta q}=\phi\rho_ea_eGe^{-a_e(k_{R0}+k_{F0})}e^{a_eGq}$, which is $q=\tilde q$. The real
minimiser is therefore $\max(\tilde q,\kappa)$ (clipped to the feasible range). For a convex function the integer minimiser is its floor or its ceiling.
(ii) Lower bound: for $q\le q_{\rm thr}$, $k_R^{\rm s}(q)=k_{R0}+\lfloor\delta k_{F0}\rfloor$ does not depend on $q$, so
$\mathcal J(q)=c_1(1-\theta)^q+c_0$ with $c_1\ge0$, which is non-increasing in $q$. Upper bound: write
$h(q)=(1-\phi)H(k_{R0})+\phi H(k_R^{\rm s}(q))$. It is non-decreasing in $q$ (because $k_R^{\rm s}$ is non-increasing and $H$ is non-increasing) and $\le1$.
The drop of the recurring term from $q$ to $q+1$ is $A(1-\theta)^q[h(q)-(1-\theta)h(q+1)]\le A(1-\theta)^q\theta$,
which is the drop of $\bar{\mathcal J}$'s first term. The cost term $\rho_eE_{\rm miss}$ is the same in $\mathcal J$
and $\bar{\mathcal J}$. For $q\ge q_{\rm rob}$, convexity gives $\bar{\mathcal J}(q+1)-\bar{\mathcal J}(q)\ge0$, hence
$\mathcal J(q+1)-\mathcal J(q)\ge\bar{\mathcal J}(q+1)-\bar{\mathcal J}(q)\ge0$. So $\mathcal J$ is non-decreasing beyond
$q_{\rm rob}$. With heterogeneous families the same argument applies term by term. ∎

**Dependence of $q_{\rm rob}$ (from the formula).** It grows like $\ln$ of the total recurrence rate $A$ and falls like $\ln$ of the
emerging-family pressure $\phi\rho_e$. It grows with the capacity $k_{R0}+k_{F0}=k-k_H$ through $a_e(k-k_H)$. It falls
with $G$ through both the logarithm and the denominator. It does not depend on $\lambda$ or $r$, because in the worst case the recent bin has
been flushed. It is bounded below by $q_{\rm thr}$, which grows with $k_{F0}$ and falls with $\delta$ and $G$.

### 4.5 Results and dependence of the exact $q^*$

Base setting [`hybrid`]: $q_{\rm thr}=40$, $q^*=74$ (exact enumeration), $\tilde q=73.85$, $q_{\rm rob}=74$. The objective is
$\mathcal J(q^*)=0.0291$, against $\mathcal J(q_{\rm thr})=0.0318$ and $\mathcal J(0)=0.2379$ (no floor) missed families per day.
$P_{\rm miss}$ falls from 0.976 at $q=0$ to 4.0e-4 at $q^*$. At $q^*$ the recent bin in a shift is 3760 rows instead of 4100,
and $k_{F0}-Gq^*=1260$ of the 2000 $F$ slots remain for the proportional share.
**MC argmin**: on the grid $q=62,\dots,86$ (200 000 reps, common random numbers) the MC objective is lowest at
$q=72$. The difference between $\mathcal J_{\rm MC}(74)$ and that minimum is 0.13 SE, so theory and MC agree within MC
error. $\mathcal J$ is flat near its optimum [`hybrid.mc_argmin`, `hybrid.gap_over_se`].

Theorem 3(ii) was checked on 300 random parameter sets [`hybrid_theorem`]. There were 0 cases with $q^*<q_{\rm thr}$ and 0 with
$q^*>q_{\rm rob}$, counting ties below float resolution as non-violations. Using $q_{\rm rob}$ instead of $q^*$ costs a median of 0
and at most 0.049 in $\mathcal J$.

Exact $q^*$ when one parameter moves from the base [`hybrid_sensitivity`] (numerical, base setting only;
$q_{\rm thr}$ in brackets where it changes):

| parameter | values | $q^*$ |
|---|---|---|
| $k$ | 2000, 5000, 10000, 20000 | 69, 74, 82, 160 ($q_{\rm thr}$ = 16, 40, 80, 160) |
| $G$ | 3, 5, 10, 20, 40 | 133, 81, 74, 66, 50 ($q_{\rm thr}$ = 133, 80, 40, 20, 10) |
| $\lambda$ | 100, 300, 1000, 2000, 10000 | 71, 73, 73, 74, 74 |
| $r$ | 1e-4, 1e-3, 2e-3, 1e-2, 5e-2 | 74, 74, 74, 73, 73 |
| $\rho$ (/day) | 1/60, 1/30, 1/14, 1/7, 1/2, 2 | 60, 67, 74, 80, 91, 102 |
| $\theta$ | 0.02, 0.05, 0.1, 0.3, 0.6 | 200 (cap), 142, 74, 40, 40 |
| $\phi$ | 0, 0.05, 0.2, 0.5, 1 | 200 (cap), 87, 74, 65, 58 |
| $\delta$ | 0.2, 0.5, 0.8, 1.0 | 160, 100, 74, 74 ($q_{\rm thr}$ = 160, 100, 40, 0) |
| $\rho_e$ (/day) | 0.005, 0.05, 0.5, 5 | 95, 74, 52, 40 |

How to read the table:
- More recurring traffic ($\rho$) gives a larger floor. More emerging-family pressure ($\phi\rho_e$) and more groups give a
  smaller floor.
- When shifts are rare ($\phi\to0$) or rows are weak ($\theta$ small), $q^*$ goes to the cap $\lfloor k_{F0}/G\rfloor$.
  The objective then sees no cost, because the $F$-share cost of item 1 in D1 is not priced. This is the main limitation of
  the same-currency rule. In that regime a cap on $q$, or the μ price of Appendix A, is needed.
- Slower arrival $\lambda$ and larger $r$ let the recent bin keep the family through short absences, so they lower $q^*$
  slightly.

### 4.6 Remark: threshold learnability

Suppose instead that a family is missed iff $N_c<m_0$, for a fixed number $m_0$ of rows needed. Then any $q\ge m_0$
removes all recurring misses, and $q^*\le\max(m_0,q_{\rm thr})$ trivially. The coverage model R4 is used because it
gives a non-trivial trade-off, and it contains the threshold case $m_0=1$ as $\theta=1$.

---

## 5. Summary of what is proven and what is open

| Statement | Status |
|---|---|
| (a) $P(N_c=0)=(1-r)^{k_R}\le e^{-rk_R}$; certain loss if $d\lambda\ge k_R$ | PROVEN, VERIFIED-MC |
| (b) $E[N_c]=kn_c/n\to0$ (uniform / reservoir) | PROVEN, VERIFIED-MC |
| (c) $N_c\ge\min(q_{\min},M_c)$ for all $t$; floor uses ≤ $Gq_{\min}$ slots; under D1 the recent bin loses 0 (normal) and ≤ $Gq_{\min}$ (shift) | PROVEN under A4, A5, P0–P3; VERIFIED-MC |
| Prop. 2: $P_{\rm miss}$ and $E_{\rm miss}$ under D1 (R1–R6) | PROVEN, VERIFIED-MC |
| Theorem 3: closed form $q_{\rm rob}$ (minimax), $q_{\rm thr}\le q^*\le q_{\rm rob}$ | PROVEN; checked on 300 random sets; MC argmin within 0.13 SE |
| (c) for the cluster variant (binary labels, groups re-estimated) | OPEN |
| (c) when the poisoning filter can remove already-trusted rows | OPEN |
| Shift state correlated with reappearance (R6 relaxed) | OPEN |
| Price of the smaller $F$ proportional share for common groups (D1 item 1) | OPEN: not in $\mathcal J$; it drives $q^*$ to the cap when $\phi\to0$ |
| Monotonicity of the exact $q^*$ in $k,G,\lambda,r,\rho$ | OPEN (numerical only) |
| Unimodality of $\mathcal J$ outside the flushed regime | OPEN (the enumeration rule does not need it) |
| Values of $\theta$, $\phi$, $\rho$, $\rho_e$, $r_e$ for real data | OPEN: dev-data estimates or a decision |
| Whether T3 adopts P0–P3 as written | OPEN: decision |

MC agreement overall [`summary`]: 59 comparisons of MC against closed form, including the Appendix A checks. The largest $|z|$ is 2.49.
Two comparisons have $|z|>2$, against 2.68 expected by chance if all formulas are right, and none has $|z|>3$. The Theorem-1 invariant had 0 violations.

---

## Appendix A. Price-based rule (μ), full-recency reading — kept for reference

This appendix keeps the first version of the rule. It uses a different reading of capacity: $k_R(q)=K-Gq$ with
$K=k-k_H$, so all non-hard capacity not used by the floor is recency. The cost is a price $\mu$ per recent slot.
D1 and D3 replace it as the primary rule. Its proofs are unchanged, and its numbers come from the JSON keys `pmiss`,
`qstar`, `sensitivity` and `theorem_bound`. The miss probability is Proposition 2 with $\phi=1$ and
$k_R^{\rm s}(q)$ replaced by $K-Gq$. Base parameters: $K=4500$, $G=10$, 3 recurring families, $\lambda=2000$/day,
$r=0.002$, $\theta=0.1$, $\rho=1/14$/day, $\mu=2.5\times10^{-4}$. MC [`pmiss`]: 14 values of $q$ from 0 to 450, largest $|z|=1.82$. Three
extra settings ($\lambda=300$; $\lambda=300,r=0.01$; $\rho=1,\lambda=500$) give 12 points, with largest $|z|=2.43$.

### A.1 Objective

Per day, the expected number of missed reappearances plus a price $\mu$ per recent slot given to the floor:
$$\boxed{\;\mathcal J(q)=\sum_{c\in\mathcal R}\rho_c\,(1-\theta)^q\,H_c(K-Gq)\;+\;\mu\,G\,q\;},\qquad
q^*=\arg\min_{0\le q\le\lfloor K/G\rfloor}\mathcal J(q),$$
$\mathcal R$ = recurring families. $\mu$ converts one recency slot into "missed reappearances per day". It is the
Lagrange multiplier of a budget "the floor may take at most $B$ recency slots". It is a value judgement and must be
fixed by decision (superseded by Section 4.3).

**Computable rule (exact).** Enumerate $q=0,\dots,q_{\rm rob}$ (see Theorem A: the search can stop at $q_{\rm rob}$)
and take the smallest minimiser. Cost $O(q_{\rm rob})$ closed-form evaluations.

### A.2 Closed form: the robust (minimax) floor — **PROVEN**

Let $\beta=-\ln(1-\theta)$, $A=\sum_{c\in\mathcal R}\rho_c$, and
$$\tilde q=\frac1\beta\ln\frac{\beta A}{\mu G},\qquad q_{\rm rob}=\arg\min_{q\in\{\lfloor\tilde q\rfloor,\lceil\tilde q\rceil\}\cap[0,\lfloor K/G\rfloor]}\ \bar{\mathcal J}(q),\qquad \bar{\mathcal J}(q)=A(1-\theta)^q+\mu Gq.$$

**Theorem A.**
(i) $q_{\rm rob}$ minimises $\sup\mathcal J(q)$ over all absence laws (Poisson or not), and equals $q^*$ when absences
always exceed the recent window ($J\ge K$ a.s.).
(ii) Under R1–R4 with $k_R=K-Gq$, the smallest minimiser of $\mathcal J$ satisfies $q^*\le q_{\rm rob}$.

*Proof.* (i) $H\le1$ with equality when $J\ge m$ a.s. (Prop. 2 with $\phi=1$), so $\sup_{\rm law}\mathcal J=\bar{\mathcal J}$.
$\bar{\mathcal J}$ is convex in real $q$ (exponential plus linear), its stationary point solves
$A\beta e^{-\beta q}=\mu G$, i.e. $q=\tilde q$. For a convex function an integer minimiser is the floor or the ceiling
of the real minimiser, clipped to the feasible range.
(ii) Write $h(q)=H(K-Gq)$, non-decreasing in $q$ and $\le1$ (Prop. 2 with $\phi=1$). The decrease of the miss term from $q$ to
$q+1$ is $A(1-\theta)^q[h(q)-(1-\theta)h(q+1)]\le A(1-\theta)^q[h(q)-(1-\theta)h(q)]\le A(1-\theta)^q\theta$
(using $h(q+1)\ge h(q)$, then $h\le1$), which is the decrease of $\bar{\mathcal J}$'s miss term. (With heterogeneous
families apply this term by term.) For $q\ge q_{\rm rob}$ convexity of $\bar{\mathcal J}$ gives $A(1-\theta)^q\theta\le\mu G$,
hence $\mathcal J(q+1)-\mathcal J(q)\ge\mu G-A(1-\theta)^q\theta\ge0$. So $\mathcal J$ is non-decreasing on
$[q_{\rm rob},\infty)$ and its smallest minimiser is $\le q_{\rm rob}$. ∎

**Dependence of $q_{\rm rob}$ (from the formula):** grows like $\ln$ of the total recurrence rate $A$; falls like
$\ln G$ and $\ln\mu$; does not depend on $k$, $\lambda$ or $r$ except through the cap $\lfloor K/G\rfloor$. That is
the price of robustness: in the worst case the recent bin has been flushed. It is non-monotone in $\theta$:
$\tilde q$ is small both when one row suffices ($\theta\to1$) and when rows are nearly useless ($\theta\to0$).

### A.3 Results

Base setting [`qstar`] ($\mu=2.5\times10^{-4}$): $q^*=20$ (exact enumeration), $\tilde q=20.89$, $q_{\rm rob}=21$;
$\mathcal J(q^*)=0.0748$ vs $\mathcal J(21)=0.0748$ vs $\mathcal J(0)$ (no floor) $=0.2032$ missed reappearances/day.
$P_{\rm miss}$ falls from 0.948 (FIFO only, $q=0$) to 0.116 at $q^*$, at the cost of $Gq^*=200$ of $K=4500$
recency slots. **MC argmin**: on the grid $q=8,\dots,32$ (200 000 reps, common random numbers) the MC objective is
minimised at $q=20$, the same as theory [`qstar.mc_argmin`, `summary.qstar_mc_argmin_equals_theory`].

Theorem A(ii) on 300 random parameter sets: 0 cases with $q^*>q_{\rm rob}$; using $q_{\rm rob}$ instead of $q^*$
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
