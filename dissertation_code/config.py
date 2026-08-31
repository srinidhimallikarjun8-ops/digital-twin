"""Single source of truth for all tunable values (backend guidelines §6).

Every threshold, seed, path, and model assumption lives here — never inline in logic. When a new
constant is introduced, add it here AND record its rationale in docs/design_decisions.md.

Units are stated explicitly everywhere: temperature in degrees Celsius, relative humidity in
percent, durations in minutes.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility -----------------------------------------------------------------------
#: Fixed seed threaded through numpy, random, and every sklearn estimator (guidelines §7).
RANDOM_SEED = 42

# --- Modelling scope (data-ethics, enforced in code) ---------------------------------------
#: The ONLY columns permitted as model inputs. CO2/light are interface-only, never features.
COMFORT_VARS = ("temperature", "relative_humidity")

# --- Thermal-comfort / uncertainty triggers (architecture Layer 3) -------------------------
#: PMV neutral point on the ASHRAE -3..+3 scale.
PMV_NEUTRAL = 0.0
#: |PMV - neutral| <= this half-width is "comfortable"; outside it is an uncertainty trigger.
PMV_NEUTRAL_BAND = 0.5
#: Sustained relative humidity above this percent is the humidity uncertainty trigger.
RH_SUSTAINED_THRESHOLD = 75.0
#: ...when sustained for at least this many minutes.
RH_SUSTAINED_MINUTES = 30
#: Active-learning uncertainty sampling strategy: "entropy" or "margin".
UNCERTAINTY_STRATEGY = "entropy"

# --- Synthetic-label generation (Sprint 1 / O2) --------------------------------------------
#: Gaussian sigma for synthetic comfort labels, calibrated to Cheung et al. (2019) variance so
#: the synthetic vote agrees with the PMV category at ~1/3 rate (their <34% individual accuracy).
NOISE_STD = 1.0
#: Discrete ASHRAE vote magnitude treated as "comfortable" (neutral/slightly = |vote| <= this).
COMFORT_VOTE_THRESHOLD = 1

# --- PMV fixed assumptions (cannot be measured from T+RH-only data; exposed for sensitivity) -
DEFAULT_METABOLIC_RATE = 1.2  # met (seated, light activity)
DEFAULT_CLOTHING_INSULATION = 0.5  # clo (light indoor clothing)
DEFAULT_AIR_VELOCITY = 0.1  # m/s (still indoor air)

# --- Time-series alignment -----------------------------------------------------------------
#: Raw T and RH come from separate devices logging at slightly different sub-minute timestamps.
#: Each (zone, channel) is resampled (mean) onto this common grid so T and RH can be paired.
#: 10 min comfortably resolves the 30-min sustained-RH trigger and matches the Bath cadence.
RESAMPLE_FREQUENCY = "10min"

# --- Data validation bounds ----------------------------------------------------------------
#: Hard physical bounds; values outside raise a SchemaError (no silent clipping).
PHYSICAL_TEMPERATURE_BOUNDS = (-20.0, 60.0)  # deg C
PHYSICAL_HUMIDITY_BOUNDS = (0.0, 100.0)  # percent
#: Dataset sanity range (LaSDPC baseline RH); values outside are warned, not rejected.
LASDPC_HUMIDITY_RANGE = (57.0, 85.0)  # percent

# --- Comfort classes (model target) --------------------------------------------------------
#: The model predicts one of three directional comfort classes, derived from the sensation vote.
#: too_cool = vote < -COMFORT_VOTE_THRESHOLD; too_warm = vote > +COMFORT_VOTE_THRESHOLD; else comfortable.
COMFORT_CLASS_TOO_COOL = "too_cool"
COMFORT_CLASS_COMFORTABLE = "comfortable"
COMFORT_CLASS_TOO_WARM = "too_warm"
COMFORT_CLASSES = (
    COMFORT_CLASS_TOO_COOL,
    COMFORT_CLASS_COMFORTABLE,
    COMFORT_CLASS_TOO_WARM,
)
#: Two-class target used for the Bath experiment (DD-019, supersedes DD-009 for this building).
#: Connaught Mansions never exceeds 22.9 degC in 8 months; at that reading PMV = -0.08, so
#: P(vote > +1) = 0.058 even at the hottest point. Every "too_warm" label would therefore be a
#: sigma=1.0 noise draw with no physical signal, and class_weight="balanced" would up-weight pure
#: noise ~150x. The warm class is merged into "comfortable" and the collapse reported as a finding.
COMFORT_CLASSES_2 = (COMFORT_CLASS_TOO_COOL, COMFORT_CLASS_COMFORTABLE)
#: When True, vote > +threshold maps to "comfortable" rather than "too_warm".
MERGE_WARM_CLASS = True

# --- Active-learning model + loop ----------------------------------------------------------
#: RandomForest size (kept small: interpretable, fast, sufficient for a PoC on T+RH).
N_ESTIMATORS = 200
#: Number of instances queried per active-learning iteration.
QUERY_BATCH_SIZE = 10
#: Size of the initial randomly-labelled seed set before active learning begins.
SEED_LABEL_COUNT = 20
#: Recommended temperature adjustment per discomfort recommendation (deg C).
TEMPERATURE_ADJUSTMENT_STEP = 1.5

# --- Live closed-loop learning -------------------------------------------------------------
#: Human feedback is weighted higher than the synthetic PMV prior when retraining, so an
#: occupant's answers visibly personalise the model (synthetic rows have weight 1.0).
HUMAN_LABEL_WEIGHT = 10.0
#: Active-query mode stops asking once the most-uncertain pool instance falls below this.
UNCERTAINTY_STOP_THRESHOLD = 0.35

# --- Active-learning convergence reference values (DD-022) ---------------------------------
#: Descriptive legacy target: within five percentage points of the full-label baseline. The final
#: study reports complete learning curves and balanced metrics rather than declaring success from
#: this threshold alone.
CONVERGENCE_ACCURACY_TOLERANCE = 0.05
#: Descriptive budget cap retained for backwards-compatible convergence tests. The formal strategy
#: comparison uses equal absolute budgets and labels-to-random-target.
LABEL_BUDGET_FRACTION = 0.40

# --- Clothing insulation from outdoor conditions (DD-017) ----------------------------------
#: A FIXED clo is untenable on the Bath data: at 0.5 clo (inherited from the warm-climate LaSDPC
#: work) PMV neutrality sits at ~24.5 degC while the flat averages ~15 degC, so 100% of readings
#: label "too_cool" and the uncertainty trigger fires on 100% of rows — no decision boundary, so
#: the active-learning experiment cannot run at all.
#: A month->clo schedule was tried and REJECTED by measurement: this building's summer is ~17.8
#: degC, so lightening clothing in June drove PMV DOWN (-1.40 vs -0.73 at a flat 1.0 clo) — it
#: anti-correlated the correction with the thing it was meant to correct.
#: Instead clo responds to a running mean of OUTDOOR temperature (ASHRAE 55 / ISO 7730 treat clo
#: as contextual; de Dear & Brager on adaptive clothing adjustment). Verified: Jan 1.17 -> Jul
#: 0.72, giving a stable 35-49% "too_cool" share in every month with no seasonal inversion.
CLO_MIN = 0.6  # clo (lightest indoor clothing assumed)
CLO_MAX = 1.25  # clo (heaviest: indoor winter layers in an unheated flat)
CLO_SLOPE = 0.045  # clo lost per degC of outdoor warming
CLO_REF_TEMP = 5.0  # deg C outdoor running mean at which clo = CLO_MAX
CLO_RUNNING_MEAN_DAYS = 7  # days in the outdoor running mean

# --- Bath experiment: pool sampling and splits ---------------------------------------------
#: Active-learning pool size. The full 514k readings are 5-min cadence, so adjacent rows are
#: near-duplicates that add no information; 20k keeps predict_proba ~0.5s so the full experiment
#: matrix stays tractable. Stratified by (zone x month) to preserve the real class mix.
POOL_SIZE = 20_000
#: Held-out rows used for scoring. The test set is evaluated after *every* AL iteration (~200
#: per run), so the full ~129k held-out period would dominate total runtime. At 25k the standard
#: error on accuracy is below 0.003 — an order of magnitude finer than the strategy differences
#: being measured.
TEST_SET_SIZE = 25_000
#: Temporal split boundary (DD-020). Train on everything before this date, test on/after it.
#: Random splitting leaks: at 5-min cadence adjacent readings are near-identical, so a random
#: split puts near-duplicates on both sides and inflates accuracy. A temporal split also matches
#: deployment (train on history, predict forward) and tests the seasonal transition.
TEMPORAL_SPLIT_CUTOFF = "2024-05-01"
#: Rooms held out for the secondary cross-room generalisation arm.
HOLDOUT_ROOMS = ("kitchen",)

# --- Experiment matrix (Sprint 5) ----------------------------------------------------------
#: Query strategies compared. "random" is the null hypothesis the proxy claim must beat.
EXPERIMENT_STRATEGIES = ("random", "entropy", "margin")
#: Seeds for repeated runs; 10 repeats give usable confidence intervals on the learning curves.
EXPERIMENT_SEEDS = tuple(range(42, 52))
#: Label budget per run (upper bound on oracle queries).
EXPERIMENT_LABEL_BUDGET = 2_000

# --- Paths (resolved relative to the repo root) --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LASDPC_DATASET_PATH = PROJECT_ROOT / "datasets" / "iot-dataset" / "Dataset_slice.csv"
BATH_DATASET_DIR = PROJECT_ROOT / "DigitalTwinData"
#: Bath quarterly workbooks, in chronological order.
BATH_WORKBOOKS = (
    "oct_jan_anonymised.xlsx",
    "jan_march_anonymised.xlsx",
    "march_may_anonymised.xlsx",
    "may_july_anonymised.xlsx",
)
PROCESSED_DIR = PROJECT_ROOT / "dissertation_code" / "data" / "processed"
AUDIT_LOG_PATH = PROJECT_ROOT / "dissertation_code" / "audit" / "audit_log.jsonl"
#: Experiment results (tidy CSV, one row per AL iteration) and figures.
RESULTS_DIR = PROJECT_ROOT / "dissertation_code" / "evaluation" / "results"
EXPERIMENT_CSV_PATH = RESULTS_DIR / "experiment_runs.csv"
FIGURES_DIR = RESULTS_DIR / "figures"

# Model + label artifacts (regenerated; gitignored).
ARTIFACTS_DIR = PROJECT_ROOT / "dissertation_code" / "model" / "artifacts"
LABEL_STORE_PATH = ARTIFACTS_DIR / "human_labels.jsonl"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "comfort_model.joblib"
RUN_MANIFEST_PATH = ARTIFACTS_DIR / "run_manifest.json"
