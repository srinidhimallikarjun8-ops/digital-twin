"""Reproducible query-behaviour and retrospective SHAP evidence.

This module analyses the saved per-query CSVs; it never reruns active learning. SHAP values are
computed from a retrospective reference model fitted to the exported queried labels for one matched
run. Initial seed rows, historical model snapshots and selection-time probabilities were not
exported, so these are illustrative audit examples rather than reconstructions of historical query
decisions.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dissertation_code import config
from dissertation_code.comfort.synthetic_labels import COMFORT_CLASS
from dissertation_code.data import schema
from dissertation_code.explain.shap_explain import explain_instance
from dissertation_code.model.base import ComfortModel

RESULTS = config.RESULTS_DIR
DATASETS = {
    "Bath": RESULTS / "experiment_queries.csv",
    "LaSDPC": RESULTS / "lasdpc_queries.csv",
}


def _hhi(series: pd.Series) -> float:
    shares = series.value_counts(normalize=True)
    return float(np.square(shares).sum())


def _mean_pairwise_tvd(frame: pd.DataFrame, column: str) -> float:
    levels = sorted(frame[column].dropna().unique())
    distributions = []
    for _, group in frame.groupby("seed"):
        distributions.append(
            group[column].value_counts(normalize=True).reindex(levels, fill_value=0.0)
        )
    if len(distributions) < 2:
        return float("nan")
    distances = [
        0.5 * np.abs(left.to_numpy() - right.to_numpy()).sum()
        for left, right in combinations(distributions, 2)
    ]
    return float(np.mean(distances))


def summarise_queries(queries: pd.DataFrame) -> pd.DataFrame:
    """One evidence row per dataset and strategy."""
    rows = []
    for (dataset, strategy), group in queries.groupby(["dataset", "strategy"]):
        class_shares = group[COMFORT_CLASS].value_counts(normalize=True)
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "queries": len(group),
                "runs": group["run_id"].nunique(),
                "mean_queries_per_run": len(group) / group["run_id"].nunique(),
                "temperature_mean": group[schema.TEMPERATURE].mean(),
                "temperature_std": group[schema.TEMPERATURE].std(),
                "rh_mean": group[schema.RELATIVE_HUMIDITY].mean(),
                "rh_std": group[schema.RELATIVE_HUMIDITY].std(),
                "triggered_share": group["triggered"].astype(bool).mean(),
                "too_cool_share": class_shares.get("too_cool", 0.0),
                "comfortable_share": class_shares.get("comfortable", 0.0),
                "too_warm_share": class_shares.get("too_warm", 0.0),
                "zone_hhi": _hhi(group["zone"]),
                "month_hhi": _hhi(group["month"]),
                "zone_seed_tvd": _mean_pairwise_tvd(group, "zone"),
                "month_seed_tvd": _mean_pairwise_tvd(group, "month"),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "strategy"]).reset_index(drop=True)


def query_composition(queries: pd.DataFrame) -> pd.DataFrame:
    """Long-form zone, month, and class shares for audit-ready tables."""
    parts = []
    for dimension in ("zone", "month", COMFORT_CLASS):
        counts = (
            queries.groupby(["dataset", "strategy", dimension], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        counts["share"] = counts["count"] / counts.groupby(
            ["dataset", "strategy"]
        )["count"].transform("sum")
        counts = counts.rename(columns={dimension: "value"})
        counts["dimension"] = dimension
        parts.append(counts)
    return pd.concat(parts, ignore_index=True)[
        ["dataset", "strategy", "dimension", "value", "count", "share"]
    ]


def build_shap_examples(queries: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Explain early/middle/late rows using a model fitted to exported queried labels."""
    rows = []
    selected = queries[queries["seed"] == seed]
    for (dataset, strategy), group in selected.groupby(["dataset", "strategy"]):
        group = group.sort_values(["iteration", "n_labels_before"]).reset_index(drop=True)
        model = ComfortModel(random_state=seed).fit(group, group[COMFORT_CLASS])
        positions = {"early": 0, "middle": len(group) // 2, "late": len(group) - 1}
        for phase, position in positions.items():
            instance = group.iloc[[position]]
            attribution = explain_instance(model, instance)
            ranked = attribution.ranked()
            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "seed": seed,
                    "phase": phase,
                    "iteration": int(instance.iloc[0]["iteration"]),
                    "temperature": float(instance.iloc[0][schema.TEMPERATURE]),
                    "relative_humidity": float(
                        instance.iloc[0][schema.RELATIVE_HUMIDITY]
                    ),
                    "observed_label": str(instance.iloc[0][COMFORT_CLASS]),
                    "final_model_prediction": attribution.predicted_class,
                    "temperature_shap": attribution.contributions[schema.TEMPERATURE],
                    "rh_shap": attribution.contributions[schema.RELATIVE_HUMIDITY],
                    "dominant_feature": ranked[0][0],
                }
            )
    return pd.DataFrame(rows)


def _load_queries() -> pd.DataFrame:
    frames = []
    for dataset, path in DATASETS.items():
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _save_figures(queries: pd.DataFrame, shap_examples: pd.DataFrame) -> None:
    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    zones = query_composition(queries)
    zones = zones[zones["dimension"] == "zone"]
    for dataset, group in zones.groupby("dataset"):
        pivot = group.pivot(index="value", columns="strategy", values="share").fillna(0)
        ax = pivot.plot.bar(figsize=(8, 4), color=["#d97706", "#2563eb"])
        ax.set(title=f"{dataset}: query share by zone", xlabel="Zone", ylabel="Share")
        ax.legend(title="Strategy", frameon=False)
        ax.figure.tight_layout()
        ax.figure.savefig(
            figure_dir / f"{dataset.lower()}_query_zone_share_balanced_review.png", dpi=160
        )
        plt.close(ax.figure)

    for dataset, group in queries.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for strategy, subset in group.groupby("strategy"):
            axes[0].hist(subset[schema.TEMPERATURE], bins=35, alpha=0.45, label=strategy)
            axes[1].hist(
                subset[schema.RELATIVE_HUMIDITY], bins=35, alpha=0.45, label=strategy
            )
        axes[0].set(title="Queried temperature", xlabel="Temperature (deg C)", ylabel="Count")
        axes[1].set(title="Queried relative humidity", xlabel="RH (%)", ylabel="Count")
        for axis in axes:
            axis.legend(frameon=False)
        fig.suptitle(f"{dataset}: conditions selected by each strategy")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"{dataset.lower()}_query_conditions_balanced_review.png", dpi=160
        )
        plt.close(fig)

    plot = shap_examples.copy()
    plot["temperature_abs"] = plot["temperature_shap"].abs()
    plot["rh_abs"] = plot["rh_shap"].abs()
    aggregate = plot.groupby(["dataset", "strategy"])[["temperature_abs", "rh_abs"]].mean()
    ax = aggregate.plot.bar(figsize=(8, 4), color=["#b91c1c", "#0f766e"])
    ax.set(
        title="Retrospective final-model SHAP magnitude for representative queries",
        xlabel="Dataset and strategy",
        ylabel="Mean absolute SHAP value",
    )
    ax.legend(["Temperature", "Relative humidity"], frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "query_shap_representatives.png", dpi=160)
    plt.close(ax.figure)


def _write_report(summary: pd.DataFrame, shap_examples: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        rendered = frame.round(4).fillna("N/A").astype(str)
        headers = list(rendered.columns)
        rows = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        rows.extend("| " + " | ".join(row) + " |" for row in rendered.to_numpy())
        return "\n".join(rows)

    report = Path("docs/query_shap_results.md")
    lines = [
        "# Query Behaviour and SHAP Evidence\n",
        "This report is generated from the saved Bath and LaSDPC query CSVs. It does not rerun "
        "active learning. The SHAP examples use a retrospective reference model fitted to the "
        "exported queried labels for seed 42. Initial seed rows, historical model snapshots and "
        "selection-time probabilities were not exported, so these are illustrative audit examples "
        "rather than reconstructions of historical query decisions.\n",
        "## Query summary\n",
        markdown_table(summary) + "\n",
        "## Retrospective SHAP examples\n",
        markdown_table(shap_examples) + "\n",
        "## Interpretation boundary\n",
        "The tables support comparisons of where each strategy spent its label budget, how "
        "stable those distributions were across seeds, and which permitted feature dominated "
        "representative reference-model explanations. They do not establish real-occupant validity, "
        "causal feature effects, or the uncertainty that existed at the instant of selection.\n",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    queries = _load_queries()
    summary = summarise_queries(queries)
    composition = query_composition(queries)
    shap_examples = build_shap_examples(queries)

    summary.to_csv(RESULTS / "query_behaviour_summary.csv", index=False)
    composition.to_csv(RESULTS / "query_composition_detailed.csv", index=False)
    shap_examples.to_csv(RESULTS / "query_shap_examples.csv", index=False)
    _save_figures(queries, shap_examples)
    _write_report(summary, shap_examples)

    print(summary.round(4).to_string(index=False))
    print(f"\nWrote query/SHAP evidence to {RESULTS} and docs/query_shap_results.md")


if __name__ == "__main__":
    main()
