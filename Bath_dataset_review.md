# Bath Dataset Review (Connaught Mansions) — structure, quality, and implications

Review of `DigitalTwinData/` following supervisor confirmation (2026-07-28) that this is the
final dataset. Covers all four quarterly workbooks, the weather station, and the floorplan.

> **Confidentiality.** Building-level figures in this document are internal project material.
> Do not paste into public tools or external outputs.

---

## 1. What is actually in the folder

The brief mentioned two files; the folder contains **seven**, and the extra ones matter.

| File | Content | Status |
|---|---|---|
| `oct_jan_anonymised.xlsx` | 7 sensors, 2023-11-14 → 2024-01-10 | Core data |
| `jan_march_anonymised.xlsx` | 8 sensors, 2024-01-26 → 2024-03-14 | Core data |
| `march_may_anonymised.xlsx` | 9 sensors, 2024-03-19 → 2024-05-15 | Core data |
| `may_july_anonymised.xlsx` | 9 sensors, 2024-05-16 → 2024-07-10 | Core data |
| `weather_station.csv` | 35,057 rows, 5-min external weather | **Newly available context** |
| `Floorplan.docx` | Single-floor plan, 7 labelled rooms | Spatial context |
| `FloorPlan-Level0.3dm` | Rhino 3D model | Digital-twin geometry (unused so far) |

**Total: 514,030 sensor readings across ~8 months** (2023-11-14 → 2024-07-10).

---

## 2. Sheet structure

Each workbook has two chart sheets plus, per sensor, a data sheet and a `_Daily` summary sheet.

**Data sheet columns** (identical across all sensors):

| Column | Content | Note |
|---|---|---|
| `Day Time` | Timestamp, 5-minute cadence | |
| `Temperature (°C)` | Air temperature | |
| `Humidity (%RH)` | Relative humidity | |
| `Dew Point (°C)` | Logger-computed dew point | |
| `Dew Point Calculated (°C)` | **Excel formula string** | Must be ignored on read |

**This is a major improvement over LaSDPC.** Temperature and humidity are on the *same row at the
same timestamp*, so the 10-minute resampling workaround (DD-006) — needed because LaSDPC's T and RH
came from separate asynchronous devices — is **not required here**. Pairing is native.

`_Daily` sheets contain only per-day min/avg/max of temperature. They are derived, not source, and
can be ignored.

---

## 3. Sensor inventory

Nine sensors: seven internal rooms, two external references.

| Canonical room | Readings | T min/mean/max (°C) | RH mean | % readings RH > 75% |
|---|---|---|---|---|
| livingroom | 62,270 | 10.6 / 15.0 / 21.5 | 71.1% | 17.6% |
| bathroom | 62,271 | 11.1 / 15.4 / 21.7 | 69.7% | 11.2% |
| study | 62,272 | 10.5 / 15.1 / 21.1 | 70.8% | 13.9% |
| kitchen | 62,272 | 10.5 / 14.3 / 22.9 | 77.8% | **66.2%** |
| spareroom | 62,274 | 10.4 / 15.0 / 20.7 | 71.7% | 20.2% |
| bedroom | 62,277 | 10.9 / 16.4 / 20.4 | 68.1% | 5.2% |
| hall | 45,966 | 13.2 / 16.3 / 20.7 | 67.4% | 1.7% |
| external_front | 62,272 | −1.5 / 11.4 / 27.1 | 87.5% | 80.8% |
| external_rear | 32,156 | 4.7 / 13.5 / 23.7 | 78.6% | 64.1% |

**Data quality is excellent:** zero malformed or missing T/RH values across all 514,030 rows.

### 3.1 Naming is inconsistent across workbooks — needs normalisation

The same physical sensor is named differently in each file. An adapter must alias these:

| Workbook | External front | Hall | Living room |
|---|---|---|---|
| oct_jan | `erht2_externalfront` | *(absent)* | `rht11_livingroom` |
| jan_march | `erht2_outsidefront` | `rht19_hall` | `rht11_livingroom` |
| march_may | `erht2 external front` | `rht19 hallway` | `rht11 livingroom` |
| may_july | `erht2_ExternalFront` | `rht19_Hall` | `rht11_LivingRoom` |

Underscores vs spaces, case changes, `outsidefront`/`externalfront`, `hall`/`hallway`. Naive loading
would treat these as **11 distinct sensors instead of 9**.

### 3.2 Coverage gaps

| Gap | Duration | Affects |
|---|---|---|
| 2024-01-10 → 2024-01-26 | **16 days** | All sensors (between oct_jan and jan_march) |
| 2024-03-14 → 2024-03-19 | 5 days | All sensors |
| 2024-05-15 → 2024-05-16 | ~1 day | All sensors |

Two sensors were installed later: **hall** from 2024-01-26, **external_rear** from 2024-03-19.
Gaps must be left as gaps — not interpolated — and the AL pool built only from real readings.

---

## 4. The critical finding — PMV saturates on this building

Running the existing PMV implementation with current assumptions (1.2 met, **0.5 clo**) over the
real Bath data:

| Room | Mean PMV | Classified "too cold" | Neutral | "too warm" | Trigger fires |
|---|---|---|---|---|---|
| kitchen | −3.09 | 100% | 0% | 0% | 100% |
| livingroom | −2.92 | 100% | 0% | 0% | 100% |
| spareroom | −2.91 | 100% | 0% | 0% | 100% |
| study | −2.87 | 100% | 0% | 0% | 100% |
| bathroom | −2.77 | 100% | 0% | 0% | 100% |
| hall | −2.52 | 100% | 0% | 0% | 100% |
| bedroom | −2.49 | 100% | 0% | 0% | 100% |

**Every reading in the building classifies as "too cold", and the uncertainty trigger fires on
100% of readings.**

### Why this happens

The building runs cold (mean ~15 °C) — plausible for an unheated or intermittently heated Georgian
flat in a British winter. But the modelling assumption is the real problem. At 0.5 clo (light
summer indoor clothing), PMV requires ~24.5 °C for neutrality:

| Clothing (clo) | PMV at 15 °C / 71% RH | Neutral temperature |
|---|---|---|
| 0.5 (light shirt) | −2.91 | ~24.5 °C |
| 0.75 | −1.98 | — |
| 1.0 (business suit) | −1.34 | ~21.5 °C |
| 1.25 (suit + jumper) | −0.87 | ~19.5 °C |
| 1.5 (heavy winter indoor) | −0.52 | — |

**Occupants of a cold flat do not dress in 0.5 clo.** They wear jumpers. The assumption is
imported from LaSDPC (a warm-climate Brazilian dataset) and does not transfer to a British
heritage building in winter.

### Why this breaks the experiment if left unfixed

1. **Synthetic labels collapse to one class.** A classifier trained on ~100% `too_cool` learns
   nothing — it predicts the majority class and scores high accuracy while being useless.
2. **The uncertainty trigger becomes meaningless.** Firing on 100% of readings is the opposite of
   minimum intervention; it cannot demonstrate selective querying.
3. **The proxy question becomes untestable.** With no class variation there is no decision boundary
   for active learning to find, so query selection cannot be shown to be meaningful.

### Recommended fix — seasonally varying clothing

This is a defensible, literature-grounded change rather than parameter-fitting:

- ASHRAE 55 and ISO 7730 both specify clothing as a **seasonal/contextual input**, not a constant.
- Adaptive comfort research (de Dear & Brager) establishes that occupants adjust clothing in
  response to indoor and outdoor conditions.
- Implement clo as a function of outdoor temperature — which the **weather station data now makes
  possible** — e.g. ~1.0–1.25 clo in winter, ~0.5–0.7 clo in summer.

This should be recorded as a new design decision (**DD-017**) with the alternatives considered
(fixed 0.5 clo — rejected as producing degenerate labels; fixed 1.0 clo — better but ignores the
8-month seasonal span; **seasonally varying clo — recommended**).

**This finding is itself a dissertation contribution:** it demonstrates empirically that PMV's
fixed-assumption form fails on real heritage-building data, which is precisely the motivation for
a personalising active-learning approach.

---

## 5. The weather station — newly usable context

`weather_station.csv` is **headerless with units embedded in values** and encoding artefacts
(`°C` mis-encoded). 35,057 rows at 5-minute cadence covering 2024.

Observed column order (positional — no header row):

| # | Field | Example |
|---|---|---|
| 0 | Date | `01/01/2024` |
| 1 | Time | `12:04 AM` |
| 2 | Outdoor temperature | `7.5 °C` |
| 3 | Dew point | `4.6 °C` |
| 4 | Outdoor humidity | `82 %` |
| 5 | Wind direction | `West` |
| 6 | Wind speed | `19.8 km/h` |
| 7 | Gust speed | `31.9 km/h` |
| 8 | Pressure | `997.97 hPa` |
| 9–10 | Rainfall | `0.00 mm` |
| 11 | *(empty)* | |
| 12 | Solar radiation | `0 w/m²` |

**Scope caution.** Weather variables must **not** become model features — the T+RH-only constraint
holds. Legitimate uses:
- **Deriving the seasonal clothing assumption** (§4) — this is a comfort-science input, not a feature.
- **Contextualising the analysis** (e.g. does the AL loop query more during cold snaps?).
- **Dashboard context display**, like CO₂ in the current design.

---

## 6. Floorplan

`Floorplan.docx` is a single-floor plan image with seven text labels: Livingroom, Bedroom, Room2,
Hall, Kitchen, Study, Bathroom. "Room2" most likely corresponds to the `spareroom` sensor — **worth
confirming with the supervisor.**

`FloorPlan-Level0.3dm` is a Rhino 3D model. Not needed for the active-learning experiment, but it
is the geometric basis of an actual digital twin and could support spatial visualisation if time
allows. Out of scope for the core experiment.

---

## 7. Do we need additional data?

**No — the dataset is sufficient for the formal research objective.** It is a substantial upgrade
on LaSDPC: 8 months vs 2.5 days, 7 named rooms vs 4 anonymous zones, natively paired T+RH, and
zero missing values.

Points worth raising with the supervisor:

1. **Confirm "Room2" = spareroom** on the floorplan.
2. **Heating schedule / occupancy periods**, if any record exists. Not required, but it would let
   the analysis distinguish "cold because unoccupied" from "cold because underheated" — which
   directly affects how query behaviour should be interpreted.
3. **Note the 16-day January gap** as a known limitation.

No new data is needed. The clothing-assumption issue (§4) is a modelling fix, not a data gap.

---

## 8. Implications for the code

| Area | Change needed |
|---|---|
| `data/sources.py` | Implement the Bath adapter: read 4 workbooks, skip `_Daily`/chart sheets, alias sensor names, drop the formula column |
| `data/schema.py` | Resampling **not** required (native pairing); keep validation and the T+RH constraint |
| `comfort/pmv.py` | Add seasonal/contextual clothing insulation (DD-017) |
| `config.py` | Add Bath paths, room aliases, clo-by-season parameters |
| `eda/` | Re-run characterisation on Bath; the LaSDPC EDA remains as prior work |
| `evaluation/` | Rebuild scenarios from real Bath conditions rather than hand-specified points |

The layered architecture holds — this is a new adapter behind the existing `data/` boundary, and
Layers 3–7 are unaffected apart from the PMV calibration.
