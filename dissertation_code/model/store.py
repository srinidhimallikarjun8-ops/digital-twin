"""Model persistence + run manifest (active-learning items 4 and 9).

Persists the fitted ComfortModel so learning survives restarts, and writes a run manifest (seed,
config snapshot, label counts, accuracy) next to it for reproducibility (backend guidelines §7).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from dissertation_code import config
from dissertation_code.model.base import ComfortModel

logger = logging.getLogger(__name__)


def save_model(
    model: ComfortModel,
    manifest: dict[str, Any] | None = None,
    model_path: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Persist the model and write a run manifest beside it."""
    model_path = model_path or config.MODEL_ARTIFACT_PATH
    manifest_path = manifest_path or config.RUN_MANIFEST_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    full_manifest = {
        "saved_at": datetime.now(UTC).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "model_features": list(config.COMFORT_VARS),
        "comfort_classes": list(config.COMFORT_CLASSES),
        "human_label_weight": config.HUMAN_LABEL_WEIGHT,
        **(manifest or {}),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2, default=str)
    logger.info("saved model -> %s", model_path)


def load_model(model_path: Path | None = None) -> ComfortModel | None:
    """Load the persisted model, or None if there is no artifact yet."""
    model_path = model_path or config.MODEL_ARTIFACT_PATH
    if not model_path.exists():
        return None
    return joblib.load(model_path)
