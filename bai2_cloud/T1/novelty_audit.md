# T1 — Novelty and positioning audit: three-compartment context policy by attack family (RQ2, H2, Nội dung 2, Mệnh đề 1)

Date: 2026-10-29. Inputs: `README_HANDOFF.md`, `context/outline_v14_3_RQ_excerpt.txt`, `context/bai1_manuscript_v3.md`, `protocol_code/cure_ref/cure_spec.md` (all from `bai2_handoff.zip`), and a web literature search.
Numbers: every number below is from a cited source (link), from a named bundle file, or from `check_numbers.py` (output in `check_numbers_out.json`).

**Verification key.** *Opened* = I opened the page or PDF and checked the stated mechanism there. *Abstract* = only the abstract or landing page was opened, so the mechanism detail rests on that page or on my prior knowledge. **[VERIFY]** = I could not open a page that states the specific claim. Check it before you cite it.

**Verdict: REVISE.** No published work I found combines a recency bank, a per-family quota bank and a capped hard-example bank for a tabular-FM context, and none uses a label-shift or two-sample trigger to move capacity between banks. But each component, taken alone, has a close published antecedent. The tempered allocation in Eq. (2-4) is in substance PRS's allocation rule. The per-class floor is in ECBRS, a NeurIPS 2023 method for NIDS. Per-family replay with a "representative + anomalous" split already exists for malware (MADAR). Proposition 1(c) also holds for CBRS-family, GDumb or ECBRS. The framing is defensible only as an extension of DualFIFO/CURE with a PRS/ECBRS-style family-quota long bank. The defensible novelty is the q_min* result, the trigger (if an ablation shows it helps), and the measured benefit in the budgeted/delayed/filtered NetFlow setting. Section 4 lists the minimal changes.

---

## 1. Closest works

### 1.1 Bounded context management for tabular FMs on streams

| Work | What it does | What it lacks for our setting | Link / status |
|---|---|---|---|
| **DualFIFO** — Lourenço, Gama, Xing, Marreiros, *In-context learning of evolving data streams with tabular foundational models*, KDD '26 (arXiv:2502.16840) | Frozen TabPFN with context = short FIFO memory + long memory. Short-memory overflow moves to the long memory. When the long memory overflows, the oldest row of the most over-represented class is removed. The paper says the design "maintains inter-class balance and captures intra-class variability". Prequential accuracy on 7 general streams; ARF, SRP and others as baselines. | Balances only by class label, with no family/sub-class groups and no per-group floor. Labels are immediate and free. No drift trigger, no hard-example bank, no security data. | Opened: https://arxiv.org/html/2502.16840v2 |
| **CURE** — Lee, Choi, Choi, Yoo, *Bounded context management for tabular foundation models on stream learning*, ICML 2026 FMSD workshop (arXiv:2606.18677) | Same two banks (defaults B=1000, ρ=0.75 → 750/250, from `check_numbers.py` and `cure_spec.md` §1). Admission to the long bank is gated by normalised predictive entropy ≥ τ, after a warm-fill. Eviction takes the closest same-class pair inside the most represented class and drops the endpoint farther from the recent-class centroid. Comes with a theoretical analysis under local-coherence assumptions. | No explicit per-class quota: the paper's only balance mechanism is the "most represented class" rule. No drift detector, and capacity is fixed. Labels are revealed right after prediction, with no delay or budget. General-purpose streams, not security. | Opened: https://arxiv.org/html/2606.18677; code reference in `cure_spec.md` (commit f7c6b2f) |
| **Verma, Nowak-Assis, Lourenço, Bifet, Pfahringer, Bahri, Lim**, *Rethinking tabular foundation models on data streams* (arXiv:2610.05352, 2026 [VERIFY date]) | An empirical study, with no new policy. It finds that "simply retaining the most recent examples is as effective as existing memory management techniques" and that "memory size has a substantially larger effect … than the choice of memory management policy". CURE helps only under label delay on one stream. | Rare classes are not studied (the rarest posture classes are dropped), and class balance is not isolated. **This is the strongest counter-evidence a reviewer will cite against any new context policy.** | Opened: https://arxiv.org/html/2610.05352 |
| **SAM-kNN** — Losing, Hammer, Wersing, ICDM 2016 / IJCAI 2017 | A kNN stream learner with short-term memory (STM) and long-term memory (LTM). STM size is chosen by bisection to minimise interleaved test-train error. Discarded STM rows are cleaned against the STM and moved to the LTM. When the LTM is full, it is compressed per class with k-means++. | Not a foundation model and no label budget, but it is the classical precedent for an *adaptive recent-memory size* and for *per-class representative compression*. | Opened: https://core.ac.uk/download/211826333.pdf ; abstract: https://www.ijcai.org/proceedings/2017/690 |

### 1.2 Class-balanced and quota-based rehearsal memories

| Work | What it does | What it lacks | Link / status |
|---|---|---|---|
| **CBRS** — Chrysakis & Moens, ICML 2020 | Class-balancing reservoir. A row of a non-full class overwrites a random row of the largest class. A row of a full class replaces a random row of its own class with probability m_c/n_c. Already in the harness, with binary label and family-label variants (`policies.py`, `bai1_manuscript_v3.md` §3.4). | No floor guarantee as a parameter, no selection inside a class (random), no recency bank, no drift trigger. | Bundle reproduction (`cbrs_ref/`); https://proceedings.mlr.press/v119/chrysakis20a.html (landing page found, not re-opened) |
| **ECBRS** — Amalapuram, Channappayya, Tamma, *Augmented memory replay-based continual learning approaches for network intrusion detection*, NeurIPS 2023 | Extends CBRS for NIDS. Replacement priority uses each class's *global running frequency*, and each class gets a threshold γ(i) = m·w(i), with w a softmax over negative running frequencies. A class is never "undermined beyond a certain threshold γ", so this is in effect **a per-class minimum quota that protects minority attack classes**. Evaluated on CICIDS-2017/2018, UNSW-NB15, CTU-13, AnoShift and others. | Works on attack *classes* as labelled (no clustering fallback), random selection within a class, no recency bank, no TFM, no label budget or delay, no trigger. **It is the NIDS precedent for q_min, and a NIDS reviewer will know it.** | Opened (PDF §3.1): https://papers.neurips.cc/paper_files/paper/2023/file/3755a02b1035fbadd5f93a022170e46f-Paper-Conference.pdf |
| **PRS** — Kim, Jeong, Kim, *Imbalanced continual learning with partitioning reservoir sampling*, ECCV 2020 | Target share per class p_i = n_i^ρ / Σ_j n_j^ρ (§3.4, Eq. 1), with n_i the running class frequency and ρ the "power of allocation": ρ=0 gives equal shares, ρ=1 gives proportional shares. Sample-in and sample-out rules push the memory toward that target. | Multi-label vision. No floor, no recency bank, no TFM. **Eq. (2-4)'s tempered part π̂_c^γ / Σ π̂^γ is this rule with γ in place of ρ.** | Opened (PDF): https://arxiv.org/pdf/2009.03632 ; ECCV page: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1912_ECCV_2020_paper.php |
| **GDumb** — Prabhu, Torr, Dokania, ECCV 2020 | A greedy class-balanced sampler. A row is admitted if its class is new or holds fewer than k/|Y| rows. When the memory is full, a random row of the largest class is removed. | Equal quota only, random selection. It is the simplest "quota memory" baseline. | Opened (PDF §3, Alg. 1): https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf |
| **Balanced reservoir (BRS)** — Buzzega et al., ICPR 2020 ("bag of tricks") | Reservoir sampling that removes a row of the most represented class instead of a random row. | Same as GDumb. | Opened: https://arxiv.org/pdf/2010.05595 |
| **Rainbow Memory** — Bang et al., CVPR 2021 | Equal per-class slots k_c = ⌊K/N_t⌋ over seen classes ("following GDumb"). Inside each class, rows are sorted by perturbation uncertainty and taken at regular intervals for diversity (§4.1, Alg. 1). | Image CL, no recency bank, no trigger. A precedent for *class quota + diversity-based selection inside the class*. | Opened (PDF): https://openaccess.thecvf.com/content/CVPR2021/papers/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.pdf |
| **MADAR** — Rahman, Coull, Yu, Wright, *Efficient continual learning for malware analysis with distribution-aware replay* (arXiv:2502.05760; venue [VERIFY]) | Per-**family** replay budget, either "Ratio" (proportional) or "Uniform" (equal). Inside each family, Isolation Forest (contamination 0.1) splits rows into "representative samples that are close to each other and anomalous samples that are separated". | Malware rather than NetFlow, retrained models only, no TFM context, no recency bank, no trigger, no label budget. **It is the security precedent for family-level quotas plus a "hard/anomalous" share, so it is closer in spirit to our k_F + k_H split than CBRS is.** | Opened: https://arxiv.org/html/2502.05760v2 |
| **iCaRL herding** — Rebuffi et al., CVPR 2017 | Per-class exemplars chosen by herding toward the class mean, with equal per-class budget. | It is a within-class selector, not a memory-allocation policy. Already an outline comparator [25]. | Abstract only: https://arxiv.org/abs/1611.07725 [VERIFY herding detail] |
| **GSS** — Aljundi et al., NeurIPS 2019 | Buffer selection that maximises gradient diversity, with no task boundaries. | Gradient-based, so it does not apply directly to a frozen TFM (no gradients). It does apply to LightGBM only weakly. Cite it, but no comparator is needed. | Abstract: https://arxiv.org/abs/1903.08671 |
| **OCS** — Yoon et al., ICLR 2022 | Online coreset selection of "representative and informative" samples per iteration. | Gradient-based, the same issue as GSS. Cite it as related work. | Abstract: https://arxiv.org/abs/2106.01085 |
| **k-center coreset** — Gonzalez 1985 [27]; Sener & Savarese, ICLR 2018 | Greedy farthest-point k-center (a 2-approximation) and its use as a core-set selector. | It is a selector, not a policy. Our use, k-center inside each family, is a direct application. Note that CURE's "drop an endpoint of the closest pair" is already a max-min-dispersion heuristic of the same kind. | Abstract: https://arxiv.org/abs/1708.00489 [VERIFY]; Gonzalez not opened [VERIFY] |

### 1.3 Retrieval-based in-context learning

| Work | What it does | What it lacks | Link / status |
|---|---|---|---|
| **LoCalPFN** — Thomas et al., NeurIPS 2024 | Builds TabPFN's context from the nearest neighbours of the query, then fine-tunes end to end. | Static datasets with the full training set available for retrieval, so no bounded memory and no stream. In our setting retrieval must run over the *bounded* store, which makes it a complement (how to use C_t) rather than a competitor (what to keep). | Abstract: https://arxiv.org/abs/2406.05207 [VERIFY per-query retrieval detail] |
| **TabR** — Gorishniy et al., ICLR 2024 | A retrieval-augmented deep tabular model (kNN-attention over the training set). | Trained, not in-context, and needs the full training set. Relevant only as related work. | Landing page: https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html [VERIFY] |
| **TabDPT** (arXiv:2410.18164) | A tabular FM that uses kNN-retrieved context at inference. | Same as LoCalPFN. | Search result only: https://arxiv.org/abs/2410.18164 [VERIFY] |

### 1.4 Drift-triggered memory adaptation and shift tests

| Work | What it does | What it lacks | Link / status |
|---|---|---|---|
| **AMR** — Ashrafee, Kozal, Woźniak, Krawczyk, *Holistic continual learning under concept drift with adaptive memory realignment* (arXiv:2507.02310) | A per-class KS test on predictive entropy (or MMD) detects drifted classes. AMR then flushes those classes' buffer slots and refills them with new-distribution rows. | It changes *which rows* a class keeps, not the *capacity split* between recent and representative banks. Image CL. **It is the closest precedent for drift-triggered memory changes.** | Opened: https://arxiv.org/pdf/2507.02310 |
| SAM-kNN (above) | STM size is adapted by error, not by a shift test. | — | — |
| **BBSE** — Lipton, Wang, Smola, ICML 2018 [47] | Black-box label-shift estimation. | Valid only under label shift, which the outline already states (Định nghĩa 3). | Search result: https://arxiv.org/abs/1802.03916 [VERIFY] |
| **C2ST** — Lopez-Paz & Oquab, ICLR 2017; **Failing Loudly** — Rabanser, Günnemann, Lipton, NeurIPS 2019 | Classifier two-sample tests. Rabanser et al. find that BBSD with univariate tests performs best, and that domain classifiers "perform poorly with few samples (100 or fewer)". | This bears on the κ_X trigger: the P(X) test runs on the unlabelled stream U_t, so n is large there, but a per-family test would be small-n. | Opened: https://arxiv.org/pdf/1810.11953 ; C2ST: https://arxiv.org/abs/1610.06545 [VERIFY] |

### 1.5 Security-specific context

- **Chen, Ding, Wagner**, *Continuous learning for Android malware detection* (arXiv:2302.04332): the analysts supply *family labels*, which the paper uses for contrastive pairs and active learning. It has no replay quota, and retiring old samples is out of scope. Opened: https://arxiv.org/html/2302.04332v2. This supports the assumption that SOC labels can carry family names.
- **TFMs for IDS** use random or stratified splits, for example Al-Dahmani et al. 2026 (TabPFN-2.5, TON_IoT, stratified 80/20, no drift handling). Opened: https://arxiv.org/pdf/2604.11394. This matches the gap stated in Bài 1 (`bai1_manuscript_v3.md` §1).
- **Not found in this search**: any TFM-context policy evaluated on NetFlow with a family-level floor, label budget, label delay, or label filtering. This is a negative search result, not a proof of absence.

---

## 2. Which framing is defensible

**Thesis standard:** "an extension of a published method to a new setting, with proof and measured benefit".

**Base methods to name explicitly**
1. **DualFIFO [13] / CURE [14]**: the two-bank bounded context (a FIFO recent bank plus a balanced long bank) and the evaluation lineage. The three-compartment policy keeps the recent bank and replaces the long bank.
2. **PRS (Kim et al. 2020)**: the tempered allocation π̂^γ/Σπ̂^γ.
3. **ECBRS (Amalapuram et al. 2023) and GDumb**: the per-class floor and quota.
4. **k-center (Gonzalez [27]; Sener & Savarese)**: the within-group selector.
5. **MADAR**: the per-family representative and anomalous split in security replay. Cite it as the nearest security analogue of k_F + k_H.

**Recommended one-sentence positioning** (for Bài 2 and Chuyên đề 1):
> We extend the two-bank bounded context of DualFIFO/CURE to attack families. The long bank becomes a family-level quota memory that combines a PRS-style tempered allocation with a hard floor q_min, as in ECBRS, using k-center selection within each family. A small, filtered hard-example bank and a shift-triggered transfer of capacity between banks complete the policy. We derive the floor q_min* that is optimal under Poisson recurrence, and we measure the benefit on rare and recurring NetFlow families for both a TFM context and a LightGBM buffer under daily label budgets and label delay.

### 2.1 Substantive technical changes, which can be claimed

| # | Change | Why it is substantive | Condition for claiming it |
|---|---|---|---|
| S1 | **Optimal floor q_min* under a Poisson recurrence model** (the Chuyên đề 1 goal in Mệnh đề 1). It trades the probability of missing a recurring family against the loss from shrinking the recent bank by G·q_min. | None of PRS, ECBRS, GDumb or CBRS derives a floor from a recurrence model; they fix it or let it come from frequencies. This is the main "proof" item. | The proof (T2) and a simulation check. Parts (a)–(c) should be presented as lemmas, not as the contribution (see R1). |
| S2 | **Capacity transfer between recent and representative banks, triggered by BBSE plus a two-sample test**, with a hold time D and a floor-preserving constraint. | I found no precedent that changes the *split* between banks on a shift signal. SAM-kNN adapts STM size by error, and AMR realigns rows within a class. | An ablation against no trigger, an oracle trigger and a **placebo trigger** (random firing at the same rate) must show a benefit. Otherwise, report it as a negative result. |
| S3 | **Grouping fallback**: when only binary labels exist, clusters inside the attack class replace families, and the cost of the two variants is reported. | ECBRS, PRS and MADAR all need group labels. The outline already includes this variant (Nội dung 2). | The guarantee in Prop. 1(c) is weaker when clusters are recomputed and their identities change. This must be stated, or cluster identity must be made stable (for example by matching clusters across days). |
| S4 | **Setting**: a TFM context *and* a LightGBM buffer of the same k, a daily label budget b with an η reserve, label delay τ, labels filtered against poisoning (with k_H fed only by filtered expert labels), and NetFlow family recurrence. | None of the works in §1 has this combination. The protocol itself is Bài 1's contribution and must not be claimed again. Bài 2 claims the *policy evaluated under it*. | Reuse the Bài 1 harness by citation. |
| S5 | **Measured benefit** on rare and recurring families (H2). | It is required by the thesis standard. | The comparator must include **CBRS-family** (see §3), not only binary CBRS. |

### 2.2 Changes that are only renamings or direct reuse, which must not be claimed as new

| Element in the outline | What it is | Required wording |
|---|---|---|
| Recent bank k_R (FIFO) | DualFIFO/CURE short bank; SAM-kNN STM | "the short bank of DualFIFO/CURE" |
| "Per-family" group | The class of CBRS, PRS or ECBRS set to the family label. The harness already has CBRS-family. Chen et al. and MADAR already use family labels in security. | "class-balanced memory at the family level" |
| π̂_c^γ / Σ π̂^γ allocation in Eq. (2-4) | PRS Eq. 1, with γ ≡ ρ | Cite PRS at Eq. (2-4) |
| Floor q_min | ECBRS threshold γ(i); GDumb and RM equal quota as the special case | Cite ECBRS and GDumb. The novelty is only *choosing* q_min (S1). |
| Largest-remainder rounding | A standard apportionment method | No claim |
| k-center within a group | Gonzalez / Sener & Savarese. A diversity selector like RM's interval sampling, SAM's k-means++ and CURE's closest-pair eviction. | "we use k-center [27] as the within-family selector", plus an ablation against random and herding |
| Hard-example bank (small margin at labelling time, ≤ 0.1k) | A variant of CURE's entropy gate (margin instead of entropy) and of MADAR's anomalous share (contamination 0.1). The cap and the filtered-labels-only rule are motivated by poisoning; that part ties into RQ4 but is a constraint, not a method. | "an uncertainty-admitted bank in the spirit of CURE's gate, capped and restricted to filtered expert labels" |
| Prop. 1 (a), (b) | Elementary facts about FIFO (geometric survival) and uniform sampling. The outline already calls (a)–(c) "direct properties". | Lemmas |
| Prop. 1 (c) | **It holds for any per-group floor memory.** CBRS-family keeps every family that is not the largest. GDumb and ECBRS keep their per-class share. CURE and DualFIFO evict only from the *most represented binary class*, so attack families are protected only while "attack" is not the majority class. | State it as "the guarantee FIFO, reservoir and binary-class-balanced memories lack". Do not state it as a guarantee no memory has. |

---

## 3. Comparators and ablations a reviewer will demand

The outline already lists FIFO, DualFIFO, CURE, CBRS, herding, LoCalPFN-style kNN and ARF (Nội dung 2).

### 3.1 Additional comparators, by priority

| Priority | Comparator | Why a reviewer will ask | Cost note |
|---|---|---|---|
| **Must** | **CBRS-family** (already in the harness: `policies.py`, `CBRSFamily`) | Without it, H2's "class-balanced buffer" is a straw man: binary CBRS cannot protect families inside the attack class. It is the natural "same idea, no compartments" baseline. | No new code. Recommend that it, not binary CBRS, be the H2 comparator (T5 decision). |
| **Must** | **PRS at the family level** (and/or ECBRS) | Eq. (2-4) is PRS plus a floor, so the reviewer will ask what the floor, k-center, recent bank and trigger add over PRS. ECBRS is the NIDS-venue method that protects minority attack classes. | One new policy each. ECBRS ≈ CBRS + thresholds, so it is cheap to wrap. |
| **Must** | **FIFO with a memory-size sweep / compute-matched FIFO** | Verma et al. 2026 report that memory size dominates the choice of policy. The reviewer will ask whether a bigger FIFO at the same GPU cost wins. | Uses the k(B) measurements that already exist. |
| Should | **Reservoir (uniform over history)** | It is the empirical counterpart of Prop. 1(b), and it costs nothing. | Trivial |
| Should | **MADAR-style per-family replay** (Ratio/Uniform + Isolation Forest 0.1) | The closest security analogue of k_F + k_H. | One new policy |
| Should | **GDumb-family** (equal quota, random selection) | The minimal quota baseline. It can be run as the ablation "γ=0, random selection, no k_R/k_H" instead of as a separate policy. | Covered by the ablations |
| Optional | SAM-kNN (`river`/MOA) | A stream learner with adaptive STM and LTM. It belongs next to ARF. | Off-the-shelf |
| Optional | Rainbow Memory (per-family, uncertainty-interval selection) | Quota plus an uncertainty-diverse selector. | One new policy |
| Optional | Oracle upper bounds: LightGBM on all labelled history; a family-quota policy with true future-recurrence knowledge | They bound the attainable gain. | LightGBM only |

Gradient-based selectors (GSS, OCS) do not apply to a frozen TFM. Cite them and explain why they are excluded.

### 3.2 Ablations

1. **Remove each bank** (−R, −F, −H). This is in the outline.
2. **Floor**: q_min ∈ {0 (pure PRS-like), q_min*, 2·q_min*}. This is the empirical check of S1 and links to T2.
3. **Tempering**: γ ∈ {0 (equal, GDumb-like), intermediate, 1 (proportional)}.
4. **Within-family selector**: k-center, random, herding, and CURE closest-pair eviction. This isolates the selector from the allocation.
5. **Trigger**: off; BBSE only; two-sample test only; both; **oracle** (true drift days from the generator manifest); **placebo** (random firing at the same rate). Also sweep δ and D, with D=3 fixed in the main grid as the outline says.
6. **Grouping**: true family, clusters inside the attack class, binary class. This reports the value of family labels.
7. **Hard bank**: k_H ∈ {0, 0.05k, 0.1k} (0.1k = 200 rows at k = 2000, from `check_numbers.py`), with and without poisoning, and fed by filtered or unfiltered labels. The unfiltered case is the RQ4 interaction from Nội dung 5.
8. **Setting sensitivity**: b, τ, and 10 % label noise (already in H2). Every ablation runs on both TFM and LightGBM, so policy × model interaction is visible (H1 link).
9. **Tuning parity**: every policy, including the added ones, gets ≤ 12 configurations (Nội dung 1). This is a reviewer fairness check.

---

## 4. Verdict and minimal changes

**REVISE.** The framing "proposed three-compartment policy" is not defensible as written if it is presented as a new memory design. Each component has a close precedent: PRS for the allocation, ECBRS and GDumb for the floor, k-center for selection, CURE and MADAR for the hard/anomalous share, and DualFIFO and CURE for the recent bank. Prop. 1(c) also holds for existing per-class floor memories. The framing **is** defensible as an extension, and none of these changes requires editing the outline:

- **R1. Name the base methods** in the method section and at Eq. (2-4): DualFIFO/CURE, PRS, ECBRS/GDumb, k-center and MADAR. Present Prop. 1(a)–(c) as lemmas. State that (c) holds for any per-group floor memory and distinguishes the policy only from FIFO, reservoir and binary-class-balanced memories.
- **R2. Claim S1–S5 only.** The headline contributions are q_min* under Poisson recurrence, the trigger (contingent on the placebo/oracle ablation), and the measured benefit in the budgeted, delayed and filtered NetFlow setting on both models. Do not claim the quota formula, the family grouping, k-center or the hard bank as new.
- **R3. Strengthen the comparator set.** Use CBRS-family as the "class-balanced buffer" of H2. The outline's wording, "bộ đệm cân bằng lớp CBRS [26]", allows it. Add PRS-family (or ECBRS) and a compute-matched FIFO. This changes the run count in Bảng 2.2 (currently 7 policies), so the NCS must decide on cost (see REPORT.md).
- **R4. Address Verma et al. 2026 directly.** State the regime in which policy should matter (rare and recurring families, absences longer than k_R/λ days, using Prop. 1(a)'s condition dλ ≥ k_R), and show the memory-size sweep.
- **R5. Credit DualFIFO correctly.** The outline writes "DualFIFO (như mô tả trong [14])". The method originates in Lourenço et al. [13], and [14] App. B.1 is the description the harness follows. Cite both.

**What would have made it NO-GO:** a published TFM-context or replay policy with a family floor *and* a shift-triggered capacity transfer, or a published q_min derivation. None was found in this search (§1). A residual risk remains: the empirical margin over CBRS-family and PRS-family may fall below ε₂ = 0.03 (outline). This is a power and H2 question for T5, not a novelty question. Under R2, the thesis contribution would still stand on S1 and a reported negative result.

## 5. Small consistency notes for T2/T3 (not part of the verdict)

- Prop. 1(c) needs k_F ≥ G·q_min to hold *at all times*. G grows as new families appear, so the policy needs a rule for when G·q_min > k_F. Step (3) of the capacity rule also needs δ·k_F ≤ k_F − G·q_min.
- Outline example check: (1 − 10⁻⁴)^5000 = 0.6065 and e^(−0.5) = 0.6065 (`check_numbers.py`), consistent with the outline's "≈ 0,61".
- In the harness, CURE's binary eviction acts on the most represented *binary* class. On days when attacks are the majority, CURE and DualFIFO evict attack rows first, regardless of family. This is the mechanism behind the gap in §2.2, and T3's demo (a family absent for 5 days) should also run against CBRS-family, not only FIFO.

## Sources consulted
https://arxiv.org/html/2502.16840v2 · https://arxiv.org/html/2606.18677 · https://arxiv.org/html/2610.05352 · https://core.ac.uk/download/211826333.pdf · https://www.ijcai.org/proceedings/2017/690 · https://proceedings.mlr.press/v119/chrysakis20a.html · https://papers.neurips.cc/paper_files/paper/2023/file/3755a02b1035fbadd5f93a022170e46f-Paper-Conference.pdf · https://arxiv.org/pdf/2009.03632 · https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf · https://arxiv.org/pdf/2010.05595 · https://openaccess.thecvf.com/content/CVPR2021/papers/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.pdf · https://arxiv.org/html/2502.05760v2 · https://arxiv.org/abs/1611.07725 · https://arxiv.org/abs/1903.08671 · https://arxiv.org/abs/2106.01085 · https://arxiv.org/abs/1708.00489 · https://arxiv.org/abs/2406.05207 · https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html · https://arxiv.org/abs/2410.18164 · https://arxiv.org/pdf/2507.02310 · https://arxiv.org/abs/1802.03916 · https://arxiv.org/abs/1610.06545 · https://arxiv.org/pdf/1810.11953 · https://arxiv.org/html/2302.04332v2 · https://arxiv.org/pdf/2604.11394 · https://link.springer.com/article/10.1007/s10994-017-5642-8 (ARF [48], not opened [VERIFY])
