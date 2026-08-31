"""Scenario-based testing (architecture Layer 7; Sprint 4 / O5).

Evaluates the trained model on a small set of representative, hand-specified comfort episodes,
comparing the predicted comfort class against the expected ("correct") class. Mirrors the
methodology's scenario testing on held-out LaSDPC episodes; the expected labels here are derived
from the comfort science (clearly cold / neutral / hot conditions) so the test is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from dissertation_code import config
from dissertation_code.model.base import ComfortModel, make_instance


@dataclass(frozen=True)
class Scenario:
    """One evaluation episode: input conditions and the expected comfort class."""

    name: str
    temperature: float
    relative_humidity: float
    expected_class: str


# Representative episodes spanning the comfort range (deterministic expected outcomes).
# Calibrated for the warm-climate LaSDPC slice; see BATH_SCENARIOS for the Bath building.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("cold dry morning", 15.0, 50.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("cool basement", 17.0, 70.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("neutral office", 23.0, 50.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("comfortable evening", 24.5, 55.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("warm humid afternoon", 28.0, 75.0, config.COMFORT_CLASS_TOO_WARM),
    Scenario("hot top-floor flat", 30.0, 60.0, config.COMFORT_CLASS_TOO_WARM),
)

# Scenarios for the Bath building, derived from its *observed* conditions rather than from
# generic comfort science.
#
# DEFAULT_SCENARIOS cannot be used here: four of its six episodes have **zero** nearby readings
# in this building (23 degC/50%, 24.5/55, 28/75, 30/60 simply never occur — the all-time maximum
# is 22.9 degC). Scoring a model on conditions it can never encounter measures nothing.
#
# Each episode below sits on a real percentile of the Bath distribution (T 11.3-19.7 degC,
# RH 59-87%) and carries thousands of nearby readings. The expected class is the *dominant*
# observed class in that neighbourhood.
#
# **Interpretation caveat.** No condition in this building has an unambiguous label: with
# sigma=1.0 label noise, neighbourhood purity is only 0.52-0.65. A perfect model would therefore
# score well below 100% on this suite, and the match rate must be read against `purity`, not
# against 1.0.
BATH_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("cold winter night", 11.5, 65.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("typical winter day", 13.0, 72.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("damp cool kitchen", 14.0, 82.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("spring shoulder", 15.0, 70.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("mild spring day", 16.5, 68.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("warm summer day", 18.0, 66.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("humid summer kitchen", 17.0, 80.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("warmest observed", 19.5, 64.0, config.COMFORT_CLASS_COMFORTABLE),
)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    predicted_class: str

    @property
    def correct(self) -> bool:
        return self.predicted_class == self.scenario.expected_class


def run_scenarios(
    model: ComfortModel, scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS
) -> list[ScenarioResult]:
    """Run each scenario through the model and record predicted vs expected."""
    results = []
    for s in scenarios:
        predicted = str(
            model.predict(make_instance(s.temperature, s.relative_humidity))[0]
        )
        results.append(ScenarioResult(s, predicted))
    return results


def match_rate(results: list[ScenarioResult]) -> float:
    """Fraction of scenarios whose prediction matched the expected class."""
    return sum(r.correct for r in results) / len(results) if results else 0.0
