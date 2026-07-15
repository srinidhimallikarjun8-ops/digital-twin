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
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("cold dry morning", 15.0, 50.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("cool basement", 17.0, 70.0, config.COMFORT_CLASS_TOO_COOL),
    Scenario("neutral office", 23.0, 50.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("comfortable evening", 24.5, 55.0, config.COMFORT_CLASS_COMFORTABLE),
    Scenario("warm humid afternoon", 28.0, 75.0, config.COMFORT_CLASS_TOO_WARM),
    Scenario("hot top-floor flat", 30.0, 60.0, config.COMFORT_CLASS_TOO_WARM),
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
