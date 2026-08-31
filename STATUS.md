# STATUS.md — Project Journal

> Append a dated entry whenever the code, data, or model changes. Newest on top. Terse by design:
> this is a journal to jog memory, not a report anyone reads end to end.

---

## 2026-08-10/11 — Three-stage comparison separates "code works" from "works on Bath"
Ran a three-stage comparison: (1) a controlled deterministic 3-class PMV benchmark on the production
model, (2) the committed LaSDPC run, (3) the committed Bath run (neither rerun, just loaded from CSV).
**Controlled:** entropy beats random, balanced acc 0.9925 vs 0.9826
(p=1.09e-05), matches random's final score with 52% fewer labels → the AL implementation itself is
correct. **LaSDPC:** 0.5782 vs 0.5801, p=0.850 → no measurable AL advantage. **Bath:** 0.5043 vs
0.5087, p=0.0567, macro-F1 significantly worse under entropy (p=0.00049) → no advantage, and LaSDPC's
earlier raw-accuracy win (+0.0189) doesn't survive seed-pairing (CI crosses zero, p=0.0779).
**Balanced accuracy is the metric to trust from here on; raw accuracy is misleading on this data.**

Also ran a signal-variation analysis to decompose *why*: Bath indoor temp std is
1.97°C vs LaSDPC's 3.38°C (narrower, not uniform — RH std 5.5pp, 20% of readings >75%). The ~4x
PMV-std gap between the two buildings splits roughly 40% building variance / 60% our own adaptive-
clothing model — i.e. "the building doesn't vary enough" is not the honest story; the clothing rule's
negative feedback compresses variation by design. No Bath row exceeds the `too_warm` PMV boundary
(max PMV 0.077) — refines an earlier "5.8% warm" claim that was based on the hottest single reading
rather than the max-PMV reading.

## 2026-08-09/10 — Both experiments complete: same code, opposite outcomes on the two datasets
LaSDPC signal-to-noise ratio 1.07 → entropy beats random (+0.019, p=0.040, 81% fewer labels to match
random's final score). Bath signal-to-noise ratio 0.27 → no effect (-0.003, p=0.845). The pre-flight
diagnostic (`validation.label_signal_report`) predicted both outcomes *before either ran* — that
diagnostic is the real contribution, not the headline number. Bath fails because label noise (σ=1.0)
is ~4x the entire PMV signal range; at the median reading P(too_cool)=0.455, a coin flip, so more
data can't fix it.

Two rescue attempts were tried on Bath and **both rejected by measurement**: (1) single-occupant
noise simulation — raw accuracy improves in 4/6 configs but balanced accuracy shows this is a
class-imbalance artefact, with a significant *loss* in two configs — this is why balanced accuracy is
used everywhere in this project; (2) the de Dear directional-noise correction implied by the
dissertation text — makes the null *stronger*, not applied (post-hoc label changes are out regardless
of which direction they push the result). Added a selective-prediction check: even after isotonic
calibration, max P(too_cool) across all Bath readings is 0.478 — no reading in the 8-month dataset
supports a confident intervention under a correctly specified noise model.

## 2026-07-29 — Bath dataset ingested; clothing model fixed; full AL matrix built
Built the Bath adapter (`data/bath.py`), aliasing 11 drifting sheet names across 4 quarterly workbooks
onto 9 canonical sensors — **514,030 readings, zero nulls, gaps preserved**. Implemented
outdoor-driven clothing (DD-017): per-row clo from a 7-day running mean of the dataset's own external
sensors, replacing the fixed-0.5-clo assumption that had put **100% of readings in `too_cool`**
(mean PMV -2.5 to -3.1, inherited from warm-climate LaSDPC tuning). The fix brought this to
34.8–47.7% too_cool per month, PMV-vote agreement 0.363 (matches the ~1/3 figure from literature).
A simpler month→clo lookup schedule was tried first and **rejected by measurement** — it made June
colder (-1.40) than doing nothing (-0.73). Added the random-arm baseline (needed to make the
"AL is a valid proxy" claim testable at all), temporal + by-room splits, and a precomputed-trigger
pool builder (the naive version was costing ~7 min/run in redundant recomputation). Bayes ceiling on
Bath measured at 0.595 — a model scoring 0.563 is already ~95% of the achievable maximum, so raw
accuracy on synthetic labels is close to meaningless without this reference point. 81 tests pass.

## 2026-07-28 — Scope re-anchored by supervisor; Bath dataset reviewed; PMV blocker found
Supervisor confirmed the EDA/preprocessing approach as correct and put human-in-the-loop interfaces
explicitly **out of formal scope** — demoted to future work / non-speculative observations only.
Formal objective narrowed to the proxy question: does active-learning query behaviour on this data
serve as a valid proxy for AI-mediated human interaction? `DigitalTwinData/` confirmed as the final
dataset. Dataset review: 514,030 readings across 8 months, 7 rooms + 2 external sensors, T+RH
natively paired at 5-min cadence (no LaSDPC-style resampling workaround needed); a 16-day gap
(Jan 10–26) is the largest of three known gaps; kitchen is the humidity hotspot (66.2% of readings
>75% RH). Dashboard demoted to demonstration-only in the architecture docs; no code changed.

**Blocker found:** running the existing PMV code on real Bath data put 100% of rows in `too_cool` —
the inherited 0.5-clo assumption (tuned for warm-climate LaSDPC) puts thermal neutrality at ~24.5°C
against Bath's ~15°C average. Proposed fix: seasonally-varying clo from outdoor temperature — became
DD-017, implemented the following session. Added `openpyxl` (dev dep) to read the workbooks.

## 2026-06-28 — Active-learning loop closed live (feedback → retrain → persist)
Built the label store (append-only JSONL, distinct from the audit log), feedback→label mapping
(confirm/override/defer), `update_with_labels` (retrains on synthetic-PMV prior + human labels, human
rows weighted 10x), `next_query` (most-uncertain pool instance + convergence stop), and model
persistence (joblib artifact + run manifest). Rewrote the Streamlit dashboard into two tabs
(check-a-condition, answer-a-query) plus a multi-zone trade-off header. Verified end-to-end on real
LaSDPC data: 24°C/55% predicts comfortable → 5 "too warm" overrides flip it to too_warm → persists
across a fresh reload. 49 tests pass. Closes the two biggest gaps up to that point: live feedback not
updating the model, and the multi-zone trade-off module being built but unused.

## 2026-06-28 — Active learning core, SHAP, recommendation layer, dashboard, evaluation harness
Built the interpretable RF comfort model (T+RH, 3 classes), entropy/margin active learning with PMV +
sustained-RH uncertainty triggers, SHAP explanations (raw values in the audit log, plain-language
sentence to the user), the recommend/decide/trade-off layer, and the audit + evaluation harness. First
Streamlit dashboard. Verified: 37 tests pass, scenario testing 6/6 correct; the convergence study
showed AL reaching within 5% of the full-label baseline (0.708) using only ~2% of labels — later
understood to mostly reflect the low synthetic-noise ceiling (~0.71) rather than strong AL signal.
Caught and fixed two real bugs: a read-only numpy array from `.to_numpy()` breaking in-place query
selection, and the audit-log path being resolved at import time instead of call time (config changes
silently didn't take effect).

## 2026-06-28 — Phase 1 backend: data layer, hand-rolled PMV, synthetic labels
Built the canonical T+RH data schema, the LaSDPC loader, a hand-rolled Fanger PMV implementation
(chosen over `pythermalcomfort` for academic-integrity "own understanding" — DD-003), and the
synthetic-label generator (PMV + calibrated Gaussian noise, σ=1.0). Verified PMV against the ISO 7730
worked example (22°C/60% → -0.753 vs reference -0.75). Caught a real pairing bug: exact-timestamp
join only matched 5 of 109,081 readings because T and RH come from separate devices logging seconds
apart — fixed with a config-driven 10-min resample grid (DD-006), recovering 1,355 paired
observations across 4 zones. Synthetic-label PMV-vote agreement measured at 0.365, matching the <34%
real-occupant PMV accuracy figure from literature — this became the calibration anchor used
throughout the project. 18 tests pass. Bath adapter stubbed (raw files not yet available).

## 2026-06-25 — 7-layer architecture designed; EDA moved into a notebook
Wrote `ARCHITECTURE.md`: data/ingestion → synthetic labelling → active learning → explanation →
recommendation/decision → Streamlit interface → audit/evaluation (cross-cutting). Fixed the
uncertainty trigger design early and never revisited it: outside ±0.5 PMV OR sustained RH >75% for
>30min OR low model margin. **Hard scope constraint set here:** CO₂/air quality is dashboard context
only, never a model feature. Converted the EDA scripts into `eda.ipynb`.

## 2026-06-21 — Sprint 1 EDA: LaSDPC temperature + RH characterisation
First real data pass: 182,809 raw rows → 109,081 after restricting to T+RH (CO₂/light excluded per
scope). Data span only ~2.5 days (16–19 Jan) — a known limitation, not hidden. Zones 2 and 4 are the
prime candidates for humidity-based AL triggers (>75% RH); zone 3 would almost never trigger. Caught
two silent bugs: output dir renamed from `code/` to `dissertation_code/` (collides with stdlib
`code`) without updating `DEFAULT_OUTPUT_DIR`, so the EDA silently wrote nothing; and time-series
plots were connecting readings across different physical devices within the same zone, producing a
fake zig-zag — fixed by grouping on `(zone, device)`. Bath dataset not yet ingested at this point.
