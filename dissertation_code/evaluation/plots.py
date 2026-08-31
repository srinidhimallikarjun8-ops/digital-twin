"""Figures for the active-learning experiment (architecture Layer 7).

Rendering only — every number comes from ``analysis.py``, so a figure can always be traced back
to a table and the table back to the run that produced it.

Figures target the dissertation PDF: a light print surface, one measure per axis, confidence
bands rather than bare means, and direct labels so series identity never depends on colour alone.
The categorical palette below was validated for colour-vision deficiency (worst adjacent pair
ΔE 8.5 protan / 21.7 normal, all slots inside the lightness band and above 3:1 contrast).
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed
import matplotlib.pyplot as plt

from dissertation_code import config
from dissertation_code.evaluation import analysis

logger = logging.getLogger(__name__)

#: Validated categorical palette, assigned in fixed order (never cycled).
STRATEGY_COLOURS = {
    "random": "#3b6fd4",
    "entropy": "#d4761f",
    "margin": "#2f9e6b",
}
#: Fallback for any strategy not in the fixed order.
DEFAULT_COLOUR = "#6b6b6b"

SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED_INK = "#6b6b6b"
GRID = "#e2e2df"


def _style_axes(ax) -> None:
    """Recessive grid and axes so the data carries the visual weight."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED_INK, labelsize=9)


def _colour(strategy: str) -> str:
    return STRATEGY_COLOURS.get(strategy, DEFAULT_COLOUR)


def plot_learning_curves(
    runs, metric: str = "accuracy", bayes_ceiling: float | None = None, path=None
):
    """The headline figure: accuracy against label count, with 95% CI bands per strategy.

    The confidence bands are the point — a gap between strategies only means something if it
    exceeds seed-to-seed variation. The Bayes-ceiling line, when supplied, is what makes the
    absolute numbers interpretable: on synthetic labels the achievable maximum is well below 1.0.
    """
    curve = analysis.learning_curve(runs, metric)

    figure, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style_axes(ax)

    for strategy, group in curve.groupby("strategy"):
        group = group.sort_values("n_labels")
        colour = _colour(strategy)
        ax.plot(
            group["n_labels"],
            group["mean"],
            color=colour,
            linewidth=2.0,
            label=strategy,
        )
        ax.fill_between(
            group["n_labels"],
            group["ci_lower"],
            group["ci_upper"],
            color=colour,
            alpha=0.18,
            linewidth=0,
        )
        # Direct label at the line end so identity is not colour-alone.
        last = group.iloc[-1]
        ax.annotate(
            strategy,
            xy=(last["n_labels"], last["mean"]),
            xytext=(6, 0),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            fontweight="bold",
            va="center",
        )

    if bayes_ceiling is not None:
        ax.axhline(bayes_ceiling, color=MUTED_INK, linewidth=1.2, linestyle="--")
        ax.annotate(
            f"Bayes ceiling ({bayes_ceiling:.3f})",
            xy=(curve["n_labels"].min(), bayes_ceiling),
            xytext=(0, 5),
            textcoords="offset points",
            color=MUTED_INK,
            fontsize=8.5,
        )

    ax.set_xlabel("Labels queried", color=INK, fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").capitalize(), color=INK, fontsize=10)
    ax.set_title(
        f"Active learning vs random selection ({metric.replace('_', ' ')})",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED_INK, loc="lower right")
    figure.tight_layout()

    return _save(figure, path or _figure_path(f"learning_curve_{metric}.png"))


def plot_query_composition(queries, by: str = "zone", path=None):
    """Share of queries per room (or month), grouped by strategy.

    Reveals whether the strategy concentrates its budget on informative regions or spreads it
    like random selection would.
    """
    composition = analysis.query_composition(queries, by=by)
    pivot = composition.pivot(index=by, columns="strategy", values="share").fillna(0.0)

    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style_axes(ax)

    positions = range(len(pivot))
    strategies = list(pivot.columns)
    width = 0.8 / max(len(strategies), 1)

    for offset, strategy in enumerate(strategies):
        ax.bar(
            [p + offset * width for p in positions],
            pivot[strategy].to_numpy(),
            width=width * 0.92,  # 2px-equivalent gap between adjacent bars
            color=_colour(strategy),
            label=strategy,
        )

    ax.set_xticks([p + (len(strategies) - 1) * width / 2 for p in positions])
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_xlabel(by.capitalize(), color=INK, fontsize=10)
    ax.set_ylabel("Share of queries", color=INK, fontsize=10)
    ax.set_title(
        f"Where each strategy spends its query budget (by {by})",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED_INK)
    figure.tight_layout()

    return _save(figure, path or _figure_path(f"query_composition_{by}.png"))


def plot_query_scatter(queries, pool=None, path=None):
    """Queried instances in temperature-humidity space, one panel per strategy.

    Direct evidence for or against the proxy claim: uncertainty sampling should trace the
    decision boundary, while random selection should look like the pool.
    """
    from dissertation_code.data import schema

    strategies = sorted(queries["strategy"].unique())
    figure, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(5.0 * len(strategies), 4.4),
        facecolor=SURFACE,
        squeeze=False,
    )

    for ax, strategy in zip(axes[0], strategies):
        _style_axes(ax)
        if pool is not None:
            ax.scatter(
                pool[schema.TEMPERATURE],
                pool[schema.RELATIVE_HUMIDITY],
                s=4,
                color=GRID,
                alpha=0.5,
                linewidths=0,
                label="pool",
            )
        subset = queries[queries["strategy"] == strategy]
        ax.scatter(
            subset["temperature"],
            subset["relative_humidity"],
            s=14,
            color=_colour(strategy),
            alpha=0.75,
            linewidths=0.5,
            edgecolors=SURFACE,  # surface ring on overlapping marks
            label="queried",
        )
        ax.set_title(strategy, color=INK, fontsize=10.5, fontweight="bold", loc="left")
        ax.set_xlabel("Temperature (deg C)", color=INK, fontsize=9.5)
        ax.set_ylabel("Relative humidity (%)", color=INK, fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED_INK, loc="upper right")

    figure.suptitle(
        "Queried conditions vs the available pool",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    figure.tight_layout()

    return _save(figure, path or _figure_path("query_scatter.png"))


def plot_stopping_behaviour(runs, path=None):
    """Peak pool uncertainty against label count, against the configured stop threshold.

    Minimum intervention requires the loop to go quiet. If the curve never crosses the threshold,
    the system would keep querying indefinitely — a result to report, not to hide.
    """
    stopping = analysis.stopping_behaviour(runs)
    if stopping.empty:
        logger.warning("no uncertainty recorded; skipping stopping-behaviour figure")
        return None

    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style_axes(ax)

    for strategy, group in stopping.groupby("strategy"):
        group = group.sort_values("n_labels")
        ax.plot(
            group["n_labels"],
            group["mean"],
            color=_colour(strategy),
            linewidth=2.0,
            label=strategy,
        )

    ax.axhline(
        config.UNCERTAINTY_STOP_THRESHOLD,
        color=MUTED_INK,
        linewidth=1.2,
        linestyle="--",
    )
    ax.annotate(
        f"stop threshold ({config.UNCERTAINTY_STOP_THRESHOLD})",
        xy=(stopping["n_labels"].min(), config.UNCERTAINTY_STOP_THRESHOLD),
        xytext=(0, 5),
        textcoords="offset points",
        color=MUTED_INK,
        fontsize=8.5,
    )

    ax.set_xlabel("Labels queried", color=INK, fontsize=10)
    ax.set_ylabel("Peak pool uncertainty", color=INK, fontsize=10)
    ax.set_title(
        "Does the loop ever go quiet?",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED_INK)
    figure.tight_layout()

    return _save(figure, path or _figure_path("stopping_behaviour.png"))


def _figure_path(filename: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return config.FIGURES_DIR / filename


def _save(figure, path):
    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("wrote figure %s", path)
    return path
