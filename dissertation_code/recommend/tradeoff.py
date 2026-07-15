"""Multi-zone trade-off summary (architecture Layer 5).

Aggregates per-zone recommendations into a one-line situational summary for the dashboard, e.g.
"2 of 4 zones need attention (1 too warm, 1 too cool); 2 comfortable." Kept deliberately simple:
the energy/CO2 cost dimension is an interface-only indicator and is not modelled here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dissertation_code import config
from dissertation_code.recommend.recommender import Recommendation


@dataclass(frozen=True)
class TradeoffSummary:
    """Counts of zones by predicted comfort class."""

    n_zones: int
    n_comfortable: int
    n_too_warm: int
    n_too_cool: int

    @property
    def n_need_attention(self) -> int:
        return self.n_too_warm + self.n_too_cool

    def headline(self) -> str:
        if self.n_need_attention == 0:
            return f"All {self.n_zones} zones comfortable."
        return (
            f"{self.n_need_attention} of {self.n_zones} zones need attention "
            f"({self.n_too_warm} too warm, {self.n_too_cool} too cool); "
            f"{self.n_comfortable} comfortable."
        )


def summarise(recommendations: list[Recommendation]) -> TradeoffSummary:
    """Summarise a list of per-zone recommendations into class counts."""
    counts = Counter(r.predicted_class for r in recommendations)
    return TradeoffSummary(
        n_zones=len(recommendations),
        n_comfortable=counts.get(config.COMFORT_CLASS_COMFORTABLE, 0),
        n_too_warm=counts.get(config.COMFORT_CLASS_TOO_WARM, 0),
        n_too_cool=counts.get(config.COMFORT_CLASS_TOO_COOL, 0),
    )
