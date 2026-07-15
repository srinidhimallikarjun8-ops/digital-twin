# System Architecture (O3)

Proof-of-concept human-in-the-loop decision-support system for thermal comfort in a heritage
building. This document is the Sprint 2 / Objective **O3** deliverable: it defines the components,
the data flow, the active-learning uncertainty trigger, and the override / defer / confirm logic.

> **Design principle.** The architecture is the algorithmic embodiment of the Burra Charter's
> *minimum-intervention* principle: the system stays silent unless it is uncertain, asks only then,
> acts only with human confirmation, and logs everything for audit. Active learning *is* minimum
> intervention expressed in code.

---

## 1. Simplified architecture (for non-technical readers)

```mermaid
flowchart LR
    A["Sensor data<br/>Temperature + Humidity"] --> B["Active-learning<br/>comfort model"]
    B --> C{"Confident<br/>enough?"}
    C -->|"No - uncertain"| D["Ask the occupant<br/>(synthetic in PoC)"]
    D --> B
    C -->|"Yes"| E["Recommendation<br/>+ plain-language reason<br/>+ confidence"]
    E --> F["Dashboard:<br/>manager confirms /<br/>overrides / defers"]
    F --> G[("Audit log")]
```

**Read it as one sentence:** temperature and humidity go in; a model decides whether it is confident;
if not, it asks (in the proof of concept a synthetic occupant answers); when confident it produces a
plain-language recommendation that a manager confirms, overrides, or defers — and every step is logged.

---

## 2. Detailed architecture

```mermaid
flowchart TB
    subgraph L1["1 - Data and ingestion"]
        RAW1["LaSDPC IoT CSV"]
        RAW2["Bath Connaught Mansions<br/>Tinytag logs"]
        LOAD["loader: filter to T+RH,<br/>clean, resample to common grid"]
        SCHEMA[("Unified time series:<br/>timestamp, zone, T, RH")]
        RAW1 --> LOAD
        RAW2 --> LOAD
        LOAD --> SCHEMA
    end

    subgraph L2["2 - Synthetic labelling (stands in for the human)"]
        PMV["PMV calculator<br/>Fanger / ASHRAE 55"]
        SYN["Synthetic feedback generator:<br/>PMV + Gaussian noise<br/>calibrated to Cheung 2019"]
        PMV --> SYN
    end

    subgraph L3["3 - Active-learning core (T + RH only)"]
        MODEL["Interpretable base model<br/>RandomForest / DecisionTree"]
        TRIG{"Uncertainty trigger:<br/>outside 0.5 PMV from neutral<br/>OR RH above 75pct over 30 min<br/>OR low model margin"}
        QUERY["Query selector"]
        UPDATE["Incremental update / retrain"]
        CONV["Convergence tracker:<br/>label budget vs accuracy"]
    end

    subgraph L4["4 - Explanation"]
        SHAP["SHAP explainer"]
        PLAIN["Plain-language translator"]
    end

    subgraph L5["5 - Recommendation and decision logic"]
        RECGEN["Per-zone recommendation"]
        TRADE["Multi-zone trade-off reasoner"]
        ACT["Override / Confirm / Defer handler"]
    end

    subgraph L6["6 - Interface (Streamlit)"]
        DASH["Per-zone dashboard:<br/>conditions, recommendation,<br/>confidence, evidence"]
        CO2["CO2 / air-quality<br/>context indicator ONLY"]
    end

    subgraph L7["7 - Audit and evidence (cross-cutting)"]
        LOG[("Audit log:<br/>action + SHAP + justification")]
        DDL[("Design-decision log")]
        EVAL["Evaluation harness:<br/>scenario tests + convergence study"]
    end

    SCHEMA --> MODEL
    SCHEMA --> TRIG
    SCHEMA --> PMV
    MODEL --> TRIG
    TRIG -->|"uncertain"| QUERY
    QUERY -->|"ask"| SYN
    SYN -->|"simulated vote"| UPDATE
    UPDATE --> MODEL
    UPDATE --> CONV

    MODEL --> SHAP
    SHAP --> PLAIN
    TRIG -->|"confident"| RECGEN
    MODEL --> RECGEN
    PLAIN --> RECGEN
    RECGEN --> TRADE
    TRADE --> DASH
    CO2 -. "context only, never a model input" .-> DASH
    DASH --> ACT
    ACT --> LOG
    PLAIN --> LOG
    CONV --> EVAL
    SCHEMA --> EVAL
```

### The active-learning loop, isolated

```mermaid
flowchart LR
    S["New T+RH reading"] --> P["Predict comfort<br/>+ uncertainty"]
    P --> Q{"Uncertain?<br/>(triggers in layer 3)"}
    Q -->|"No"| R["Recommend + explain"]
    Q -->|"Yes"| A["Query oracle<br/>(synthetic occupant)"]
    A --> U["Add label,<br/>incrementally update model"]
    U --> P
```

---

## 3. Why these choices (design-decision log seed)

| Decision | Chosen | Alternatives considered | Criterion |
|---|---|---|---|
| Base model | Interpretable RandomForest / DecisionTree | RL; deep NN; Gaussian Process | Interpretability + no large labelled set needed (D1 Table 2.1) |
| Sample efficiency | Active learning (query on uncertainty) | Exhaustive labelling; static RF | ~60% fewer queries (Tekler 2023); mirrors minimum intervention |
| Labels | Synthetic PMV + calibrated Gaussian noise | Real occupant votes; reuse ASHRAE DB II | Ethics/scope; avoids distribution mismatch (D1 §1.1.2) |
| Uncertainty trigger | Outside ±0.5 PMV **OR** RH >75% >30 min **OR** low margin | Pure model-entropy threshold | Combines comfort science with model confidence; auditable |
| Explanation | SHAP in audit log, plain language to user | Raw SHAP to user; no explanation | Calibrated trust for non-technical user (Amershi 2019 G11) |
| CO₂ / air quality | Interface context indicator only | Add as model feature | Hard scope constraint — T+RH only model |

---

## 4. Map to the codebase

Each architecture layer becomes a package under `dissertation_code/` (built incrementally per sprint):

| Layer | Package | Status |
|---|---|---|
| 1 Data & ingestion | `dissertation_code/data/` (extends current `eda/loader.py`) | EDA done; unified loader next |
| 1b EDA | `dissertation_code/eda/` | **Done (Sprint 1)** |
| 2 Synthetic labelling | `dissertation_code/comfort/` (`pmv.py`, `synthetic_labels.py`) | Sprint 1 close / Sprint 2 |
| 3 Active-learning core | `dissertation_code/active_learning/` (`model.py`, `query_strategy.py`, `loop.py`, `convergence.py`) | Sprint 3 |
| 4 Explanation | `dissertation_code/explain/` (`shap_explainer.py`, `plain_language.py`) | Sprint 3 |
| 5 Recommendation & decision | `dissertation_code/recommend/` (`recommender.py`, `tradeoff.py`, `decision.py`) | Sprint 3 |
| 6 Interface | `dissertation_code/dashboard/app.py` (Streamlit) | Sprint 3 |
| 7 Audit & evaluation | `dissertation_code/audit/` + `dissertation_code/evaluation/` | Sprint 3–4 |

---

## 5. Cross-cutting invariants (must hold everywhere)

- **Model inputs are temperature + RH only.** CO₂/light enter the dashboard as context, never the model.
- **Every recommendation is auditable.** SHAP values + justification land in the audit log; the user
  sees plain language.
- **Every action is reversible by a human.** Override / confirm / defer, each with logged justification.
- **Synthetic labels validate mechanics, not real-occupant accuracy.** State this honestly anywhere
  results are reported.
