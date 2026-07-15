"""Plain-language narration of a comfort prediction (architecture Layer 4).

Turns a SHAP Attribution into a sentence a non-technical building manager understands in ~30s,
e.g. "Likely too warm: the temperature (26.5 C) is the main driver, with humidity (78%) adding to
the discomfort." The raw SHAP numbers stay in the audit log; this sentence is the user-facing layer.
"""

from __future__ import annotations

from dissertation_code import config
from dissertation_code.data import schema
from dissertation_code.explain.shap_explain import Attribution

_CLASS_PHRASE = {
    config.COMFORT_CLASS_TOO_COOL: "likely too cool",
    config.COMFORT_CLASS_COMFORTABLE: "likely comfortable",
    config.COMFORT_CLASS_TOO_WARM: "likely too warm",
}

_FEATURE_NOUN = {
    schema.TEMPERATURE: "temperature",
    schema.RELATIVE_HUMIDITY: "humidity",
}


def _format_value(feature: str, value: float) -> str:
    if feature == schema.TEMPERATURE:
        return f"{value:.1f} C"
    if feature == schema.RELATIVE_HUMIDITY:
        return f"{value:.0f}%"
    return f"{value:.1f}"


def narrate(attribution: Attribution) -> str:
    """Generate a one-sentence, plain-language explanation from a SHAP attribution."""
    phrase = _CLASS_PHRASE.get(attribution.predicted_class, attribution.predicted_class)
    ranked = attribution.ranked()

    if attribution.predicted_class == config.COMFORT_CLASS_COMFORTABLE:
        return f"Conditions are {phrase}; no change recommended."

    primary, primary_val = ranked[0]
    primary_noun = _FEATURE_NOUN.get(primary, primary)
    primary_str = _format_value(primary, attribution.feature_values[primary])
    sentence = f"Conditions are {phrase}: the {primary_noun} ({primary_str}) is the main driver"

    if len(ranked) > 1:
        secondary, _ = ranked[1]
        secondary_noun = _FEATURE_NOUN.get(secondary, secondary)
        secondary_str = _format_value(secondary, attribution.feature_values[secondary])
        sentence += f", with {secondary_noun} ({secondary_str}) also contributing"
    return sentence + "."
