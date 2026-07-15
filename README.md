# Human–AI Decision Support for Digital Twins of Heritage Buildings

A **proof-of-concept (PoC)** human-in-the-loop machine-learning system for thermal comfort in a
heritage-building digital twin (MSc AI dissertation, University of Bath).

It ingests **temperature + relative-humidity** sensor data, runs an **active-learning loop** over an
interpretable model, generates **synthetic comfort labels** (PMV + calibrated noise), and surfaces
**interpretable, overridable** recommendations to a non-technical decision-maker (e.g. a facilities
manager), logging every decision for audit.

> **What it is / isn't.** It demonstrates that the mechanics work end-to-end, cleanly and
> reproducibly. It is **not** a deployment-ready system, and the synthetic labels validate the
> *mechanics*, **not** accuracy for a real occupant (a real-occupant trial is future work).

**Research question.** *How can a digital twin, using active learning on temperature and humidity
data, learn when and how to query occupant comfort preferences to provide interpretable, personalised
recommendations that support informed human decision-making in a heritage context?*

Further reading: [`ARCHITECTURE.md`](ARCHITECTURE.md) (system design + diagrams),
[`STATUS.md`](STATUS.md) (activity log), [`docs/design_decisions.md`](docs/design_decisions.md)
(design-decision audit trail).

---

## 1. Setup from scratch (macOS)

Starting from a clean Mac, top to bottom. (Linux/Windows differ only in how you install `uv` — see the
[uv docs](https://docs.astral.sh/uv/).)

**Step 1 — Install Homebrew** (the macOS package manager), if you don't already have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2 — Install Python 3.12+** (optional — `uv` can fetch Python for you in Step 3, but installing
it explicitly is fine):

```bash
brew install python@3.12
python3 --version          # expect Python 3.12.x or newer
```

**Step 3 — Install `uv`** (the environment + dependency manager). Pick whichever you prefer:

```bash
# a) Homebrew — recommended on macOS
brew install uv

# b) official install script
curl -LsSf https://astral.sh/uv/install.sh | sh

# c) with pip or pipx
pip3 install uv            # or:  pipx install uv
```

Verify it, and let `uv` provide Python if you skipped Step 2:

```bash
uv --version
uv python install 3.12     # downloads a matching Python only if one isn't already present
```

## 2. Get the project running

```bash
git clone <repo-url>
cd srinidhi_project
uv sync                    # creates .venv and installs ALL pinned dependencies from uv.lock
```

That one `uv sync` installs everything (pandas, numpy, scikit-learn, SHAP, Streamlit, matplotlib,
pytest, jupyter) and, if a suitable Python isn't found, fetches it automatically. You do **not** need
to create a virtualenv or `pip install` anything by hand. The LaSDPC dataset slice is already bundled
at `datasets/iot-dataset/Dataset_slice.csv` — nothing to download.

**First run — confirm everything works (run these in order):**

```bash
uv run pytest -q                                          # 1) 49 tests should pass
uv run main.py                                            # 2) data + synthetic-label pipeline
uv run python -m dissertation_code.evaluation.run         # 3) evaluation studies (scenario + convergence)
uv run streamlit run dissertation_code/dashboard/app.py   # 4) opens the dashboard at localhost:8501
```

Step 4 opens the interactive app in your browser. See **§3** for what each command does and **§4** for
the full dashboard guide.

## 3. How to use it — four entry points

Every command is run with `uv run …` (which uses the project's environment automatically).

### 3a. Run the data pipeline (Sprint 1)

```bash
uv run main.py
```

Loads the LaSDPC data (temperature + RH only), resamples it onto a common 10-minute grid, pairs T+RH,
and generates synthetic comfort labels. Prints a log like:

```
Loaded 109081 T+RH readings (long format)
Resampled to a 10min grid: 2711 readings
Paired into 1355 co-located T+RH observations across 4 zones
Generated synthetic labels: 904/1355 comfortable (66.7%)
PMV-vote agreement rate: 0.365 (expect ~0.33, per Cheung et al. 2019)
```

### 3b. Launch the interactive dashboard (the main artefact)

```bash
uv run streamlit run dissertation_code/dashboard/app.py
```

This opens the decision-support app in your browser (usually `http://localhost:8501`). See the
**[walkthrough in §4](#4-dashboard-walkthrough)** below.

### 3c. Run the evaluation studies (Sprint 4)

```bash
uv run python -m dissertation_code.evaluation.run
```

Runs **scenario-based testing** (six representative comfort episodes) and the **active-learning
convergence study** (accuracy vs label budget), and logs the results.

### 3d. Run the test suite

```bash
uv run pytest -q
```

49 unit tests covering PMV correctness, the schema/validation rules, the synthetic-label calibration,
the active-learning loop, explanations, recommendations, the audit log, and the closed feedback loop.

### (optional) View the EDA notebook

```bash
uv run jupyter lab    # then open dissertation_code/eda.ipynb
```

---

## 4. Streamlit usage — the dashboard, option by option

Launch it with `uv run streamlit run dissertation_code/dashboard/app.py` and open
`http://localhost:8501`. The dashboard is a **genuine closed learning loop**, not just a predictor:
the model starts from a synthetic PMV prior and updates itself from your answers. The screen has a
**sidebar** (left) and a **main area** with a header and **two tabs**.

### 4.1 Sidebar — the inputs (always visible)

| Control | What it is | How it works |
|---|---|---|
| **Human labels collected** | A counter (e.g. `1`). | The number of answers you've given that became training labels — read live from the label store (`model/label_store.py`). It goes up each time you confirm/override or answer a query. |
| **Zone** | A dropdown (zones 1–4). | The building zone the recommendation is attributed to in the audit log. The list comes from the actual zones in the dataset. |
| **Temperature (°C)** | A slider, 10–35 °C. | A **model input**. Moving it re-runs the prediction instantly. |
| **Relative humidity (%)** | A slider, 20–100 %. | The other **model input**. |

> Only temperature + humidity are model inputs — there is deliberately no CO₂/light control, because
> those are out of modelling scope (interface-only). The sliders are the *whole* feature space.

### 4.2 Header (top of the main area)

- **Title + caption** — names the tool and states the proof-of-concept / synthetic-labels caveat.
- **Multi-zone status line** — e.g. *"3 of 4 zones need attention (2 too warm, 1 too cool); 1
  comfortable."* It's computed by taking the **latest reading in each zone**, predicting each one, and
  summarising (`recommend/tradeoff.py`). It gives a building-wide glance before you drill into a zone.

### 4.3 Tab 1 — "Check a condition" (you ask the model)

Use this when *you* have a condition in mind and want the recommendation. For the zone/T/RH set in the
sidebar, the app shows:

| Item | Meaning | Behind it |
|---|---|---|
| **Predicted comfort** | 🔵 too cool / 🟢 comfortable / 🔴 too warm. | The Random-Forest model's predicted class. |
| **Confidence** | e.g. `66%`. | The model's probability for that class (`predict_proba` max). |
| **Model uncertainty** (bar) | e.g. `34%`. | Normalised entropy across the classes — high means the model is genuinely unsure, low means confident. |
| **Trigger warning** (yellow box) | Appears only sometimes. | Fires when a *domain* rule trips: PMV outside ±0.5 of neutral, or RH > 75 %. This is the "ask a human" signal, independent of model confidence. |
| **Action** (blue box) | e.g. *"Consider lowering the temperature by ~1.5 °C."* | Derived from the predicted class — the actionable suggestion. |
| **Why** | A plain-language sentence. | Generated from the SHAP attribution (`explain/narrate.py`) — what drove the prediction, in words. |
| **Supporting evidence** (expander) | A small SHAP table. | The raw per-feature SHAP contributions kept for audit; opened only if you want the numbers. |

Then you make a **decision** (this is what updates the model):

1. **Action radio — confirm / override / defer:**
   - **confirm** — you agree with the prediction; that class becomes a stored label.
   - **override** — you disagree; a **"Actual comfort"** dropdown appears so you pick the real class,
     which becomes the label.
   - **defer** — you postpone; nothing is labelled.
2. **Justification (required)** — a free-text reason. The app refuses to record without it
   (auditability requirement — every action must say *why*).
3. **"Record decision & update model"** button:
   - writes a **recommendation** record and a **human-decision** record to the audit log;
   - for confirm/override, **stores the label, retrains the model on the prior + your weighted labels,
     and re-saves it** — so your next prediction reflects the answer;
   - for defer, it only logs (model unchanged).

### 4.4 Tab 2 — "Answer a query (active learning)" (the model asks you)

This is the "active" half — the system decides *what it most needs to know*:

1. It shows the **single most-uncertain unlabelled case** in the pool, e.g. *"Zone 4 — temperature
   23.7 °C, humidity 81 % (model uncertainty 91%, current guess: comfortable)."* (`next_query` picks
   the highest-entropy instance.)
2. **"How would this feel?"** dropdown — pick too cool / comfortable / too warm.
3. **"Submit answer"** — stores your label (tagged `human_query`), retrains the model, and advances to
   the next most-uncertain case.
4. When the most-uncertain remaining case falls **below the stop threshold**, the app says the model
   has **converged** and stops asking — minimum intervention: it goes quiet once it's confident.

### 4.5 Recent audit log (expander at the bottom)

Shows the last 10 audit records — each recommendation and human decision with its timestamp, inputs,
prediction, action, your decision, justification, and SHAP values. This append-only log is the
heritage "explicable, auditable intervention" trail and the dissertation's evidence base.

### 4.6 What persists

The trained model, the run manifest, and your labels are saved under
`dissertation_code/model/artifacts/` (gitignored), and the audit log under `dissertation_code/audit/`.
So everything you teach the model **survives a restart** — re-launch and the human-label counter and
learned behaviour are still there. (Delete those files to reset to a fresh synthetic-only model.)

### 4.7 A 2-minute demo flow (e.g. to show a supervisor)

1. Tab 1: set **29 °C / 78 %** → prediction **too warm**, trigger warning fires, action "lower
   temperature". Open *Supporting evidence* to show temperature dominates.
2. Set **24 °C / 55 %** → likely **comfortable**. Choose **override → too warm**, justify
   ("south-facing room runs hot"), record. Watch the **Human labels collected** counter increment.
3. Re-check **24 °C / 55 %** → the prediction has now shifted toward **too warm** — the model learned
   from you.
4. Tab 2: answer two active queries, then point out the **audit log** capturing every step.

---

## 5. How it works (pipeline overview)

```
LaSDPC CSV ──▶ load + filter to T+RH ──▶ resample to 10-min grid ──▶ pair T+RH
                                                                        │
                              synthetic comfort labels (PMV + noise) ◀──┘
                                                                        │
                       interpretable Random Forest comfort model ◀──────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
   uncertainty + triggers        SHAP attribution              recommendation
   (when to ask)                 → plain-language reason        + action
        │                                                            │
   active query ──▶ human answer ──▶ store label ──▶ retrain ◀── confirm/override/defer
                                                                     │
                                                            append-only audit log
```

- **Model inputs are temperature + relative humidity only.** This is enforced in code — the data
  schema *rejects* any other channel (CO₂, light) before it can reach the model.
- **Synthetic labels:** occupant sensation = PMV + Gaussian noise calibrated to Cheung et al. (2019),
  so the synthetic feedback disagrees with PMV at roughly the real ~⅓ rate.
- **Active learning:** the model queries only when uncertain (entropy) or when a domain trigger fires
  (outside ±0.5 PMV from neutral, or RH > 75 % sustained > 30 min) — the algorithmic embodiment of the
  heritage *minimum-intervention* principle.
- **Explanations:** SHAP values go to the audit log; the user sees a plain-language sentence.
- **Reproducible:** all randomness is seeded from `config.RANDOM_SEED`; same input ⇒ same output.

---

## 6. What's been done so far (progress)

Mapped to the dissertation objectives (O1–O6) and sprints:

| Objective | Sprint | Status | What exists |
|---|---|---|---|
| **O1** Literature review | 1 | ✅ Done (document) | In `detailed_dissertation.md` — not code |
| **O2** Dataset characterisation + synthetic labels | 1 | 🟡 Partial | LaSDPC EDA ✅, synthetic-label generator ✅. **Bath dataset not yet ingested** (adapter stubbed; raw files pending) |
| **O3** System architecture | 2 | ✅ Done | `ARCHITECTURE.md` (detailed + simplified diagrams, triggers, override/defer/confirm) |
| **O4** PoC implementation | 3 | ✅ Done | Active-learning loop, synthetic generator, SHAP + plain language, Streamlit dashboard, **live closed feedback loop**, audit log |
| **O5** Evaluation | 4 | 🟡 Partial | Scenario testing (100% on 6 episodes) ✅, convergence study ✅. **Uses hand-specified scenarios, not real held-out LaSDPC episodes** |
| **O6** Critical reflection | 4 | ⬜ Pending | Dissertation prose, not code |

**Verified results:** 49 tests pass; PMV matches the ISO 7730 reference (22 °C/60 % → −0.75);
synthetic-label PMV-agreement 0.365 on real data; closed loop demonstrably flips a prediction after
feedback and persists across restart; scenario testing 100%; the AL loop reaches within 5 % of the
full-labelled baseline (~0.71) using ~2 % of the labels.

**Known gaps / honest limitations:**
- Bath/Connaught Mansions dataset not yet ingested (waiting on raw Tinytag export).
- Evaluation episodes are hand-specified, not sampled held-out LaSDPC windows.
- "Incremental" update is a fast full *retrain* (Random Forest has no `partial_fit`) — chosen to keep
  the model interpretable for SHAP (see `docs/design_decisions.md` DD-013).
- Synthetic labels cannot validate real-occupant accuracy.

---

## 7. Project structure

```
main.py                       # Phase 1 pipeline entry point
dissertation_code/
  config.py                   # single source of truth: seeds, thresholds, paths (no magic numbers)
  pipeline.py                 # shared load→label→train + live retrain/persist lifecycle
  data/                       # Layer 1 — ingestion
    schema.py                 #   canonical T+RH schema, validation, resampling, long→wide
    sources.py                #   dataset adapters (LaSDPC; Bath stub)
  comfort/                    # Layer 2 — comfort science
    pmv.py                    #   Fanger PMV (ASHRAE 55 / ISO 7730)
    synthetic_labels.py       #   synthetic comfort-label generator (O2)
  model/                      # Layer 3 — active learning
    base.py                   #   interpretable RandomForest comfort model
    active_learning.py        #   uncertainty, triggers, query selection, retrain-on-feedback
    label_store.py            #   append-only store of human labels
    store.py                  #   model persistence + run manifest
  explain/                    # Layer 4 — explanation
    shap_explain.py           #   SHAP attributions
    narrate.py                #   plain-language sentence
  recommend/                  # Layer 5 — recommendation + decision
    recommender.py            #   assemble recommendation
    decision.py               #   override/confirm/defer + feedback→label
    tradeoff.py               #   multi-zone trade-off summary
  audit/log.py                # Layer 7 — append-only audit log
  evaluation/                 # Layer 7 — scenario testing + convergence study
  dashboard/app.py            # Layer 6 — Streamlit UI (no business logic)
  eda/                        # Sprint 1 EDA (+ eda.ipynb for the supervisor)
  utils/                      # seeding + logging configuration
tests/                        # pytest, mirrors dissertation_code/
docs/design_decisions.md      # design-science audit trail
datasets/iot-dataset/         # LaSDPC slice (nested clone; read-only)
ARCHITECTURE.md  STATUS.md
```

## 8. Configuration

All tunable values live in [`dissertation_code/config.py`](dissertation_code/config.py) — never
hard-coded inline. Key knobs: `RANDOM_SEED`, `COMFORT_VARS` (the T+RH allow-list), `PMV_NEUTRAL_BAND`
(±0.5), `RH_SUSTAINED_THRESHOLD`/`_MINUTES` (75 % / 30 min), `NOISE_STD` (label noise),
`RESAMPLE_FREQUENCY` (10 min), `HUMAN_LABEL_WEIGHT` (feedback weighting), and the artifact paths.

## 9. Data ethics (enforced in code)

- **Only temperature + relative humidity** are model inputs. CO₂ / light are interface-only context
  and are *rejected by the schema* if they reach the modelling pipeline.
- **No PII** — no occupant identifiers, no fine-grained location beyond city.
- **Comfort labels are synthetic** (PMV + calibrated Gaussian noise) and never presented as real votes.
