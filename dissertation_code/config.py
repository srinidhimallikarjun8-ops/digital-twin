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

# --- Active-learning convergence targets (Sprint 4; placeholders) --------------------------
#: Target: reach within this fraction of the full-labelled baseline accuracy...
CONVERGENCE_ACCURACY_TOLERANCE = 0.05  # TODO: confirm in methodology chapter
#: ...using no more than this fraction of the available labels.
LABEL_BUDGET_FRACTION = 0.40  # TODO: confirm in methodology chapter

# --- Paths (resolved relative to the repo root) --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LASDPC_DATASET_PATH = PROJECT_ROOT / "datasets" / "iot-dataset" / "Dataset_slice.csv"
PROCESSED_DIR = PROJECT_ROOT / "dissertation_code" / "data" / "processed"
AUDIT_LOG_PATH = PROJECT_ROOT / "dissertation_code" / "audit" / "audit_log.jsonl"

# Model + label artifacts (regenerated; gitignored).
ARTIFACTS_DIR = PROJECT_ROOT / "dissertation_code" / "model" / "artifacts"
LABEL_STORE_PATH = ARTIFACTS_DIR / "human_labels.jsonl"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "comfort_model.joblib"
RUN_MANIFEST_PATH = ARTIFACTS_DIR / "run_manifest.json"
