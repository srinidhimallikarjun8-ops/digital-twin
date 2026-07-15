"""Predicted Mean Vote (PMV) — Fanger's thermal comfort model (ASHRAE 55-2020).

Architecture Layer 2. PMV is the *prior* the system is built around: useful at population scale,
unreliable per-individual (Cheung et al. 2019 -> <34% individual accuracy). The active-learning
loop exists precisely to correct this prior with (synthetic, here) occupant feedback.

We only have temperature + relative humidity. The remaining Fanger inputs (metabolic rate,
clothing insulation, air velocity, mean radiant temperature) are fixed to documented, standard
sedentary-indoor assumptions and exposed as parameters so a sensitivity analysis is possible.

Implementation follows the iterative clothing-surface-temperature solution in ASHRAE 55-2020
Normative Appendix B / ISO 7730.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dissertation_code import config


@dataclass(frozen=True)
class ComfortAssumptions:
    """Fixed Fanger inputs we cannot measure from a T+RH-only dataset (defaults from config)."""

    metabolic_rate: float = config.DEFAULT_METABOLIC_RATE  # met
    clothing_insulation: float = config.DEFAULT_CLOTHING_INSULATION  # clo
    air_velocity: float = config.DEFAULT_AIR_VELOCITY  # m/s


def pmv(
    air_temperature: float,
    relative_humidity: float,
    assumptions: ComfortAssumptions | None = None,
    mean_radiant_temperature: float | None = None,
) -> float:
    """Compute Fanger's PMV for one (temperature, RH) observation.

    Args:
        air_temperature: dry-bulb air temperature, deg C.
        relative_humidity: relative humidity, percent (0-100).
        assumptions: fixed met/clo/velocity; defaults to sedentary indoor.
        mean_radiant_temperature: deg C; defaults to air temperature (no radiant data available).

    Returns:
        PMV on the ASHRAE -3 (cold) .. +3 (hot) scale; 0 is neutral.
    """
    a = assumptions or ComfortAssumptions()
    ta = air_temperature
    tr = mean_radiant_temperature if mean_radiant_temperature is not None else ta
    vel = a.air_velocity

    # Metabolic rate W/m^2 (1 met = 58.15 W/m^2); no external work for sedentary occupants.
    m = a.metabolic_rate * 58.15
    w = 0.0
    mw = m - w

    # Clothing insulation m^2K/W (1 clo = 0.155).
    icl = a.clothing_insulation * 0.155
    fcl = 1.05 + 0.645 * icl if icl > 0.078 else 1.00 + 1.290 * icl

    # Water-vapour partial pressure (Pa) from RH and the Antoine-style saturation expression.
    pa = relative_humidity * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))

    hcf = 12.1 * math.sqrt(vel)  # forced convection coefficient

    taa = ta + 273.0
    tra = tr + 273.0

    # Iterate for clothing surface temperature (ASHRAE 55 Appendix B).
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = xn
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25  # natural convection
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= 0.00015:
            break
    tcl = 100.0 * xn - 273.0

    # Heat-loss components.
    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)  # skin diffusion
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0  # sweating
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)  # latent respiration
    hl4 = 0.0014 * m * (34.0 - ta)  # dry respiration
    hl5 = 3.96 * fcl * (xn**4 - (tra / 100.0) ** 4)  # radiation
    hl6 = fcl * hc * (tcl - ta)  # convection

    thermal_sensation = 0.303 * math.exp(-0.036 * m) + 0.028
    return thermal_sensation * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)


def is_comfortable(
    pmv_value: float, half_width: float = config.PMV_NEUTRAL_BAND
) -> bool:
    """True when PMV is within the ASHRAE comfort band |PMV - neutral| <= half_width."""
    return abs(pmv_value - config.PMV_NEUTRAL) <= half_width
