"""The active-learning experiment: does query strategy beat random selection?

Architecture Layer 7. This module *is* the dissertation's formal contribution — the supervisor's
research objective is whether patterns in the data can be learned through active learning and used
as a proxy for AI-mediated interaction with humans.

The design point that makes the question answerable is the **random arm**. Uncertainty sampling
reaching some accuracy means nothing on its own; what matters is whether it reaches that accuracy
with *fewer labels than choosing at random*. Random is the null hypothesis, and if entropy cannot
beat it there is no proxy claim to make.

Output is a single tidy CSV, one row per active-learning iteration, plus a companion CSV of every
individual query. Every table and figure in the analysis is a groupby over those two files, so the
numbers behind a figure can always be traced back to the run that produced them.
"""

from __future__ import annotations

import logging

import pandas as pd

from dissertation_code import config, pipeline
from dissertation_code.comfort import synthetic_labels as sl
from dissertation_code.data import sampling
from dissertation_code.model import active_learning as al
from dissertation_code.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

#: Column order of the per-iteration results CSV.
RUN_COLUMNS = (
    "run_id",
    "strategy",
    "use_trigger",
    "seed",
    "split",
    "iteration",
    "n_labels",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "max_uncertainty",
)

#: Column order of the per-query CSV.
QUERY_COLUMNS = (
    "run_id",
    "strategy",
    "seed",
    "split",
    "iteration",
    "n_labels_before",
    "zone",
    "month",
    "temperature",
    "relative_humidity",
    "comfort_class",
    "triggered",
)


def run_single(
    pool: pd.DataFrame,
    test_set: pd.DataFrame,
    strategy: str,
    seed: int,
    split_name: str,
    *,
    use_trigger: bool = False,
    label_budget: int = config.EXPERIMENT_LABEL_BUDGET,
    batch_size: int = config.QUERY_BATCH_SIZE,
    seed_count: int = config.SEED_LABEL_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one active-learning configuration and return (iteration rows, query rows)."""
    run_id = f"{split_name}-{strategy}{'+trig' if use_trigger else ''}-s{seed}"
    logger.info("running %s", run_id)

    result = al.run_active_learning(
        pool,
        test_set,
        seed_count=seed_count,
        batch_size=batch_size,
        max_labels=label_budget,
        random_state=seed,
        strategy=strategy,
        use_trigger=use_trigger,
        record_queries=True,
    )

    iterations = pd.DataFrame(
        {
            "run_id": run_id,
            "strategy": strategy,
            "use_trigger": use_trigger,
            "seed": seed,
            "split": split_name,
            "iteration": range(len(result.label_counts)),
            "n_labels": result.label_counts,
            "accuracy": result.accuracies,
            "balanced_accuracy": result.balanced_accuracies,
            "macro_f1": result.macro_f1s,
            "max_uncertainty": result.max_uncertainties,
        }
    )[list(RUN_COLUMNS)]

    queries = pd.DataFrame(result.queried_rows)
    if not queries.empty:
        queries = queries.assign(
            run_id=run_id, strategy=strategy, seed=seed, split=split_name
        )[list(QUERY_COLUMNS)]

    return iterations, queries


def build_experiment_data(
    source: str = "bath",
    pool_size: int = config.POOL_SIZE,
    test_size: int = config.TEST_SET_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the labelled dataset and derive the (pool, test) pair for the temporal split.

    The pool is drawn from the *training* period only; the test set is the held-out later period.
    Drawing the pool before splitting would let the loop query instances from the test period.

    The test set is subsampled because it is scored after *every* active-learning iteration —
    roughly 200 times per run, times 20+ runs. The full held-out period is ~129k rows, where the
    scoring cost dominates the entire experiment; a stratified 25k sample gives a standard error
    below 0.003 on accuracy, far finer than the differences being measured.
    """
    labelled = pipeline.build_labelled_dataset(source=source)
    train, test = pipeline.split_temporal(labelled)

    pool = sampling.build_pool(train, n=pool_size)
    test_set = sampling.attach_month(test)
    if test_size and len(test_set) > test_size:
        test_set = test_set.sample(
            n=test_size, random_state=config.RANDOM_SEED
        ).reset_index(drop=True)

    logger.info(
        "experiment data: pool=%d test=%d | pool balance %s | test balance %s",
        len(pool),
        len(test_set),
        pool[sl.COMFORT_CLASS].value_counts(normalize=True).round(3).to_dict(),
        test_set[sl.COMFORT_CLASS].value_counts(normalize=True).round(3).to_dict(),
    )
    return pool, test_set


def run_gate(
    strategies: tuple[str, ...] = ("random", "entropy"),
    seeds: tuple[int, ...] = config.EXPERIMENT_SEEDS,
    source: str = "bath",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the go/no-go gate: random vs entropy over repeated seeds, temporal split.

    Deliberately the smallest experiment that can answer the research question. If entropy does
    not separate from random here, no amount of downstream analysis will rescue the proxy claim,
    and the framing has to change before more is built on top of it.
    """
    pool, test_set = build_experiment_data(source=source)

    iteration_frames, query_frames = [], []
    for strategy in strategies:
        for seed in seeds:
            iterations, queries = run_single(
                pool, test_set, strategy, seed, split_name="temporal"
            )
            iteration_frames.append(iterations)
            if not queries.empty:
                query_frames.append(queries)

    runs = pd.concat(iteration_frames, ignore_index=True)
    all_queries = (
        pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame()
    )
    return runs, all_queries


def save_results(runs: pd.DataFrame, queries: pd.DataFrame) -> None:
    """Write the tidy results to the configured paths."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.EXPERIMENT_CSV_PATH, index=False)
    queries_path = config.RESULTS_DIR / "experiment_queries.csv"
    queries.to_csv(queries_path, index=False)
    logger.info(
        "wrote %d iteration rows to %s and %d query rows to %s",
        len(runs),
        config.EXPERIMENT_CSV_PATH,
        len(queries),
        queries_path,
    )


def summarise_gate(runs: pd.DataFrame) -> pd.DataFrame:
    """Compare strategies at matched label counts — the gate's decision table."""
    return (
        runs.groupby(["strategy", "n_labels"])["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def main() -> None:
    configure_logging()
    runs, queries = run_gate()
    save_results(runs, queries)

    summary = summarise_gate(runs)
    final_labels = summary["n_labels"].max()
    final = summary[summary["n_labels"] == final_labels]

    logger.info("=== GATE RESULT at %d labels ===", final_labels)
    for _, row in final.iterrows():
        logger.info(
            "  %-8s accuracy %.4f +/- %.4f (n=%d)",
            row["strategy"],
            row["mean"],
            row["std"],
            row["count"],
        )

    by_strategy = final.set_index("strategy")["mean"]
    if {"entropy", "random"}.issubset(by_strategy.index):
        delta = by_strategy["entropy"] - by_strategy["random"]
        logger.info("  entropy - random = %+.4f", delta)


if __name__ == "__main__":
    main()
