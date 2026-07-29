# Scope Boundary — what is in, what is out, what needs confirming

Derived from Ricardo Codinhoto's email of 2026-07-28 ("MSc Progress – Dataset Review and Next
Steps"), addressed to **Fabio Nemetz, Paul Shepherd and Srinidhi**. This is the reference for what
the dissertation formally claims.

> **Note on supervision.** KB §17 lists "primary supervisor unconfirmed (Fabio vs Ricardo)" as an
> open discrepancy. Ricardo issued this directive with Fabio and Paul Shepherd copied — evidence
> worth weighing, though still not a formal confirmation. Paul Shepherd does not currently appear
> in the KB.

He also confirms the previously agreed EDA/preprocessing approach still stands — *"which aligns with
the objectives of your dissertation"* — so the method is unchanged; only the dataset is now final.

---

## 1. The formal research objective, in his words

> *"the experiment should investigate whether patterns within the data can be learned through
> active learning and used as proxies for AI-mediated interactions with humans."*

The operative word is **proxy**. The question is not merely "does active learning work on this
dataset" — it is whether the loop's *query behaviour* meaningfully approximates what would
otherwise require live human feedback.

> *"A very in-depth analysis of the results obtained from the active learning experiment should
> give you an excellent outcome for the dissertation."*

Depth of analysis on the experiment is what earns the grade — not breadth of features.

---

## 2. Explicitly IN scope

| # | Item | Why it is in scope |
|---|---|---|
| 1 | **EDA and preprocessing** of the Bath dataset | Directly instructed: *"proceed with the EDA and preprocessing approach we previously discussed"* |
| 2 | **Training the active-learning system** on the comfort data | The core experiment |
| 3 | **Evaluating whether learned patterns are a valid proxy** for human feedback | The formal research question |
| 4 | **In-depth analysis of experiment results** — convergence, query behaviour, label efficiency | Explicitly named as the path to an excellent outcome |
| 5 | **Synthetic labelling** (PMV + calibrated noise) | The mechanism that stands in for live human response |
| 6 | **Explainability analysis (SHAP)** | Needed to show *why* the loop queries what it queries — this is analysis, not interface |
| 7 | **Audit log as evidence base** | The record the in-depth analysis draws on |
| 8 | **Comfort-model calibration** (e.g. the clothing assumption) | Required for the experiment to produce non-degenerate labels |

---

## 3. Explicitly OUT of scope

| # | Item | His wording |
|---|---|---|
| 1 | **Building or exploring human-facing interfaces** | *"You will not explore interfaces for that"* |
| 2 | **Human-in-the-loop as a primary research objective** | *"falls outside the formal scope of your dissertation"* |
| 3 | **Speculative claims** about human-AI interaction | *"must not be speculative in nature"* |
| 4 | **Claims about how real humans would respond** | Follows from 1–3; no human subjects are involved |

> **Tone matters here.** He calls HITL *"an interesting extension of the work"* — it is **not**
> dismissed as irrelevant, only placed outside *this* dissertation's formal scope. That is the
> natural framing for the **future-work** section: a legitimate and interesting direction, deferred.

### The permitted middle ground

> *"any findings related to this aspect should be presented as exploratory observations (which must
> not be speculative in nature) rather than as a primary research objective."*

Human-in-the-loop findings are **not banned** — they are **demoted**. They may appear as
exploratory observations provided they are grounded strictly in what the results show.

**The test to apply:** can the statement be traced to a number in the results?

| ✅ Permitted (grounded) | ❌ Not permitted (speculative) |
|---|---|
| "The loop converged after N labels, ~X% of the pool." | "This would reduce occupant burden in practice." |
| "Queries clustered near the 75% RH boundary." | "Facilities managers would find this intuitive." |
| "Feedback weighted 10× shifted the prediction after 5 labels." | "Occupants would trust these recommendations." |

---

## 4. ⚠️ Items needing confirmation — the ambiguity in the current code

**The tension:** the existing Streamlit dashboard has a confirm/override/defer flow with a
mandatory justification box. That **is** a human-in-the-loop interface — the thing his email places
outside formal scope.

**How this has been handled so far:** the code was kept, but `ARCHITECTURE.md` and `justify.md` were
updated to reposition it as a *demonstration artefact, not a research object*, with any findings
from it marked exploratory. The experiment itself (Layers 1–4, 7) runs headlessly via
`evaluation/run.py` and does not depend on the dashboard.

**This interpretation should be confirmed rather than assumed.** Four specific questions:

### Q1 — Does the existing dashboard need to be removed, or is demotion sufficient?
It already exists and works. Current interpretation: **keep as a demonstration artefact**, exclude
from research claims, present no usability findings. *Needs confirmation.*

### Q2 — Is the override/confirm/defer mechanism in scope as an ML mechanism?
There are two separable things here:
- The **learning mechanism** (a label enters the model with 10× weight and changes predictions) —
  arguably in scope as active-learning machinery.
- The **interface** for collecting it (buttons, justification box) — out of scope.

Current interpretation: the mechanism is in scope, the interface is not. *Needs confirmation.*

### Q3 — How should the dashboard be presented in the written dissertation?
Options: (a) omit entirely; (b) appendix only; (c) brief demonstration section with explicit
non-claim. Current plan: **(c)**. *Needs confirmation.*

### Q4 — Is weather-station data acceptable for deriving the clothing assumption?
It would **not** become a model feature (T+RH-only holds), but would inform a comfort-science
parameter. This matters because without it the PMV labels are degenerate — see
`docs/bath_dataset_review.md` §4. *Recommend raising, as it affects experiment validity.*

---

## 5. His direct question — "Is there any other data you would need?"

This requires a reply. Based on the dataset review, the honest answer is:

**No additional data is needed.** The dataset is sufficient: 8 months, 514,030 readings, 7 named
rooms plus 2 external references, natively paired T+RH, zero missing values.

Three things worth raising alongside that answer:

1. **Confirm "Room2" on the floorplan corresponds to the `spareroom` sensor.**
2. **Heating schedule or occupancy periods**, if any record exists — not required, but it would let
   the analysis distinguish "cold because unoccupied" from "cold because underheated", which
   affects how query behaviour is interpreted.
3. **Flag the 16-day January gap** (2024-01-10 → 2024-01-26) as a known limitation.

**Also worth surfacing in the same reply:** the PMV cold-saturation finding (§4 of the dataset
review). Every reading currently classifies as "too cold" under the inherited 0.5 clo assumption.
This is a substantive methodological point, the fix is defensible and literature-grounded, and it 
is better raised now than discovered at write-up.

---

## 6. "But how does the loop run with no interface and no human?"

The obvious objection, and the answer is central to the whole design: **active learning never
required a live human — it requires an _oracle_**, i.e. something that returns a label when queried.
A human is one implementation of that interface; the synthetic label generator is another.

Here the oracle is `PMV(T, RH) + Gaussian noise` — it answers deterministically, in microseconds,
with no interface involved. This is standard AL research practice: most AL papers evaluate on
fully-labelled datasets by *hiding* the labels and revealing them only as the model requests them.
That is exactly what happens here — all ~514k readings have a hidden synthetic label, and the model
only sees the ones it earns by asking.

**The reframe that matters: the experiment is about the _questions_, not the _answers_.** The oracle
will answer anything. What varies — and carries all the research value — is *which* readings the loop
chooses to ask about and *when it stops*.

> Random selection: label 500 readings → accuracy X
> Active learning: label ~27 readings → accuracy ≈ X

If AL reaches comparable accuracy from a small, deliberately chosen subset, its query strategy has
captured something real about where the informative regions are. **That is the proxy claim.**

### How it runs (headless, `evaluation/run.py`)

1. Build the pool (~514k Bath readings, each with a hidden synthetic label).
2. Seed: reveal ~20 labels at random, train the initial RF.
3. Score: `predict_proba` over the unlabelled pool → entropy per reading.
4. Select: highest entropy, with domain-triggered rows (RH > 75%, PMV outside ±0.5) promoted.
5. "Query": reveal those rows' synthetic labels — **this is the oracle call**.
6. Retrain; record accuracy on a held-out test set.
7. Repeat until the most-uncertain remaining row falls below the stop threshold.

### Where the in-depth analysis lives (all human-free)

- **Query composition** — which rooms did it ask about? Kitchen should dominate early (66.2% of its
  readings exceed 75% RH → both domain-triggered and genuinely ambiguous). Concentration there is
  evidence of sensible targeting rather than arbitrary picking.
- **Boundary behaviour** — do queried readings cluster near comfort-class boundaries or scatter
  uniformly? Clustering is the signature of a strategy that found the decision surface.
- **Trigger contribution** — how many queries came from domain rules vs pure entropy? Tests whether
  encoding comfort science into the selector added anything over statistics alone.
- **SHAP on queried instances** — is the model uncertain for *physically sensible* reasons?
- **Stopping behaviour** — where does it go quiet, and does it stop before or after the seasonal
  transitions (8 months of data includes the March–May shift)?
- **Baselines** — AL vs random sampling vs full labelling. The AL-vs-random gap **is** the result.

### The honest limitation

The loop demonstrates that **AL query strategies are label-efficient on real heritage-building data
and that their selections are physically meaningful.** It does not demonstrate that a real occupant
would have given those answers — the oracle is PMV-derived, and PMV is right about individuals only
~33% of the time. The σ=1.0 noise calibration is what makes this a *harder, more honest* test: the
loop learns against a realistically unreliable oracle rather than a perfect one.

> **Prerequisite.** None of this analysis is possible until the clothing-assumption blocker is fixed
> (`docs/bath_dataset_review.md` §4). With 100% of readings in one class there is no decision
> boundary, entropy is ~0 everywhere, and the loop has nothing to select between. **The PMV fix is a
> prerequisite for the experiment, not a refinement of it.**

---

## 7. Practical consequences for the work

| Area | Consequence |
|---|---|
| **Effort allocation** | Depth on the AL experiment and its analysis; no further interface development |
| **Dashboard** | Frozen as-is — maintained, not extended |
| **Dissertation framing** | Research question is the proxy question; interface appears only as demonstration |
| **Results chapter** | Convergence, query behaviour, SHAP analysis, label efficiency — the in-depth analysis |
| **Any HITL observation** | Must trace to a number in the results, or it is cut |
