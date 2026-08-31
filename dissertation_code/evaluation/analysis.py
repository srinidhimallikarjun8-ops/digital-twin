"""In-depth analysis of the active-learning experiment (architecture Layer 7).

Every table here is derived from the two tidy CSVs written by ``experiment.py`` — the
per-iteration results and the per-query records. Keeping computation here and rendering in
``plots.py`` means the numbers behind every figure are independently testable, and any figure can
be traced back to the run that produced it.

The tables answer, in order:

1. **Does active learning beat random?** (``strategy_comparison``, ``label_efficiency``) — the
   proxy claim itself. Random is the null hypothesis.
2. **Does it ask about sensible things?** (``query_composition``, ``query_conditions``) — a
   strategy that reached good accuracy by querying arbitrary rows would not support the claim.
3. **Does the domain trigger contribute?** (``trigger_contribution``) — on this building the
   trigger fires on ~97% of rows, so it is expected to add nothing over pure entropy. Measuring
   that is a finding, not a failure (DD-018).
4. **Does it know when to stop?** (``stopping_behaviour``) — minimum intervention requires the
   loop to go quiet, not query forever.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dissertation_code import config

logger = logging.getLogger(__name__)


def load_results(
    runs_path=None, queries_path=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the experiment CSVs written by ``experiment.save_results``."""
    runs_path = runs_path or config.EXPERIMENT_CSV_PATH
    queries_path = queries_path or (config.RESULTS_DIR / "experiment_queries.csv")
    return pd.read_csv(runs_path), pd.read_csv(queries_path)


def learning_curve(runs: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Mean and 95% CI of a metric against label count, per strategy.

    The confidence interval is what makes the headline figure honest: a single-seed gap between
    two strategies is indistinguishable from run-to-run noise.
    """
    grouped = runs.groupby(["strategy", "n_labels"])[metric]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()

    standard_error = summary["std"] / np.sqrt(summary["count"])
    summary["ci_lower"] = summary["mean"] - 1.96 * standard_error
    summary["ci_upper"] = summary["mean"] + 1.96 * standard_error
    return summary


def strategy_comparison(
    runs: pd.DataFrame, metric: str = "accuracy", at_labels: int | None = None
) -> pd.DataFrame:
    """Compare strategies at a fixed label budget (defaults to the largest reached).

    Includes a Welch t-test of each strategy against ``random``: the proxy claim needs the gap
    to be distinguishable from seed noise, not merely positive.
    """
    from scipy import stats

    budget = at_labels if at_labels is not None else int(runs["n_labels"].max())
    at_budget = runs[runs["n_labels"] == budget]

    summary = (
        at_budget.groupby("strategy")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": metric})
    )
    summary["n_labels"] = budget

    if "random" in set(summary["strategy"]):
        baseline = at_budget[at_budget["strategy"] == "random"][metric]
        deltas, p_values = [], []
        for strategy in summary["strategy"]:
            values = at_budget[at_budget["strategy"] == strategy][metric]
            deltas.append(float(values.mean() - baseline.mean()))
            if strategy == "random" or len(values) < 2 or len(baseline) < 2:
                p_values.append(float("nan"))
            else:
                p_values.append(
                    float(stats.ttest_ind(values, baseline, equal_var=False).pvalue)
                )
        summary["delta_vs_random"] = deltas
        summary["p_value_vs_random"] = p_values

    return summary.sort_values(metric, ascending=False).reset_index(drop=True)


def label_efficiency(runs: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Labels each strategy needs to match random's final score — the efficiency claim.

    This is the number the proxy argument turns on: if entropy reaches random's end-state score
    using materially fewer labels, its query selection carries real information.
    """
    curve = learning_curve(runs, metric)
    if "random" not in set(curve["strategy"]):
        raise ValueError(
            "no 'random' arm present; label efficiency needs the null baseline"
        )

    random_curve = curve[curve["strategy"] == "random"]
    target = float(random_curve.loc[random_curve["n_labels"].idxmax(), "mean"])

    rows = []
    for strategy, group in curve.groupby("strategy"):
        reached = group[group["mean"] >= target].sort_values("n_labels")
        labels_needed = int(reached.iloc[0]["n_labels"]) if not reached.empty else None
        rows.append(
            {
                "strategy": strategy,
                "target_metric": target,
                "labels_to_target": labels_needed,
                "reached_target": labels_needed is not None,
            }
        )

    efficiency = pd.DataFrame(rows)
    random_labels = efficiency.loc[
        efficiency["strategy"] == "random", "labels_to_target"
    ].iloc[0]
    if random_labels:
        efficiency["label_saving"] = efficiency["labels_to_target"].apply(
            lambda n: None if n is None else 1.0 - (n / random_labels)
        )
    return efficiency


def query_composition(queries: pd.DataFrame, by: str = "zone") -> pd.DataFrame:
    """Share of queries directed at each room (or month), per strategy.

    Tests whether the strategy concentrates on informative regions. The kitchen is the natural
    candidate on this building: 66% of its readings exceed 75% RH, against 20% or less elsewhere.
    """
    counts = queries.groupby(["strategy", by]).size().reset_index(name="n_queries")
    totals = counts.groupby("strategy")["n_queries"].transform("sum")
    counts["share"] = counts["n_queries"] / totals
    return counts.sort_values(["strategy", "share"], ascending=[True, False])


def query_conditions(queries: pd.DataFrame) -> pd.DataFrame:
    """Temperature/humidity distribution of queried instances, per strategy.

    Uncertainty sampling should cluster near the decision boundary, giving a *narrower*
    temperature spread than random selection. A comparable spread would suggest the strategy is
    not finding the boundary at all.
    """
    return (
        queries.groupby("strategy")
        .agg(
            n_queries=("temperature", "size"),
            temperature_mean=("temperature", "mean"),
            temperature_std=("temperature", "std"),
            humidity_mean=("relative_humidity", "mean"),
            humidity_std=("relative_humidity", "std"),
            triggered_share=("triggered", "mean"),
        )
        .reset_index()
    )


def trigger_contribution(
    queries: pd.DataFrame, pool_base_rate: float | None = None
) -> pd.DataFrame:
    """Triggered share among queries versus the pool's base rate (DD-018).

    "Lift" is the excess over base rate. A lift near zero means the trigger carried no
    information — expected here, because it fires on ~97% of pool rows, so prioritising
    triggered rows degenerates to pure uncertainty ranking.
    """
    summary = (
        queries.groupby("strategy")["triggered"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "triggered_share", "count": "n_queries"})
    )
    if pool_base_rate is not None:
        summary["pool_base_rate"] = pool_base_rate
        summary["lift"] = summary["triggered_share"] - pool_base_rate
    return summary


def stopping_behaviour(runs: pd.DataFrame) -> pd.DataFrame:
    """Peak pool uncertainty against label count — does the loop ever go quiet?

    Minimum intervention requires the system to stop asking once it is confident. If peak
    uncertainty never falls below ``UNCERTAINTY_STOP_THRESHOLD``, the loop would query
    indefinitely, and that must be reported rather than glossed over.
    """
    uncertainty = runs.dropna(subset=["max_uncertainty"])
    if uncertainty.empty:
        return pd.DataFrame()

    summary = (
        uncertainty.groupby(["strategy", "n_labels"])["max_uncertainty"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["below_stop_threshold"] = (
        summary["mean"] < config.UNCERTAINTY_STOP_THRESHOLD
    )
    return summary


def build_all_tables(
    runs: pd.DataFrame, queries: pd.DataFrame, pool_base_rate: float | None = None
) -> dict[str, pd.DataFrame]:
    """Compute every analysis table in one call, keyed by name."""
    tables = {
        "learning_curve": learning_curve(runs),
        "strategy_comparison": strategy_comparison(runs),
        "stopping_behaviour": stopping_behaviour(runs),
    }
    try:
        tables["label_efficiency"] = label_efficiency(runs)
    except ValueError as exc:  # pragma: no cover - only without a random arm
        logger.warning("skipping label efficiency: %s", exc)

    if not queries.empty:
        tables["query_composition_by_zone"] = query_composition(queries, by="zone")
        tables["query_composition_by_month"] = query_composition(queries, by="month")
        tables["query_conditions"] = query_conditions(queries)
        tables["trigger_contribution"] = trigger_contribution(queries, pool_base_rate)

    return tables


def save_tables(tables: dict[str, pd.DataFrame], directory=None) -> None:
    """Write every analysis table to CSV for the write-up."""
    directory = directory or (config.RESULTS_DIR / "tables")
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    logger.info("wrote %d analysis tables to %s", len(tables), directory)
