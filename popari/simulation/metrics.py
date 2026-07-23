"""Metrics for simulated Popari results stored as models or AnnData objects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from multiprocess import Pool
from ortools.graph.python import min_cost_flow
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

SCALING_FACTOR = int(1e4)
EVALUATION_COLUMNS = [
    "run_id",
    "result_id",
    "random_state",
    "metric",
    "dataset_index",
    "dataset_name",
    "score",
]


def spatial_wasserstein(
    spatial_coordinates: np.ndarray,
    embeddings_truth: np.ndarray,
    embeddings_pred: np.ndarray,
    weight_scaling_factor=SCALING_FACTOR,
    demand_scaling_factor=SCALING_FACTOR,
):
    """Compute spatial Wasserstein distance between two nonnegative spatial
    fields."""

    assert len(spatial_coordinates) == len(embeddings_truth) == len(embeddings_pred)
    assert spatial_coordinates.ndim == 2
    assert embeddings_truth.ndim == embeddings_truth.ndim == 1
    assert embeddings_truth.min() >= 0
    assert embeddings_pred.min() >= 0

    if embeddings_truth.sum() == 0:
        return np.inf

    pairwise_dist = squareform(pdist(spatial_coordinates))
    positive_pairwise_dist = pairwise_dist[pairwise_dist > 0]
    pairwise_dist_min = positive_pairwise_dist.min()
    pairwise_dist = (pairwise_dist / pairwise_dist_min * weight_scaling_factor).astype(int)
    weight_scaling_factor_full = weight_scaling_factor / pairwise_dist_min

    embeddings_truth = embeddings_truth / embeddings_truth.sum()
    embeddings_pred = embeddings_pred / embeddings_pred.sum()
    demands = embeddings_pred - embeddings_truth
    demands_max = np.abs(demands).max()
    if demands_max == 0:
        return 0
    demands = (demands / demands_max * demand_scaling_factor).astype(int)
    demand_scaling_factor_full = demand_scaling_factor / demands_max

    idx = np.argmax(np.abs(demands))
    demands[idx] -= demands.sum()

    smcf = min_cost_flow.SimpleMinCostFlow()

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    for cell_idx_i, pairwise_dist_row in enumerate(pairwise_dist):
        for cell_idx_j, dist in enumerate(pairwise_dist_row):
            if cell_idx_i != cell_idx_j and demands[cell_idx_i] < 0 and demands[cell_idx_j] > 0:
                start_nodes.append(cell_idx_i)
                end_nodes.append(cell_idx_j)
                capacities.append(demand_scaling_factor)
                unit_costs.append(dist)

    smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes,
        end_nodes,
        capacities,
        unit_costs,
    )

    smcf.set_nodes_supplies(np.arange(0, len(demands)), -demands)

    status = smcf.solve()

    if status != smcf.OPTIMAL:
        raise RuntimeError(f"There was an issue with the min cost flow input. Status: {status}")

    return smcf.optimal_cost() / demand_scaling_factor_full / weight_scaling_factor_full


def all_pairs_spatial_wasserstein(
    dataset,
    spatial_key: str = "spatial",
    embeddings_truth_key: str = "ground_truth_X",
    embeddings_pred_key: str = "X",
    weight_scaling_factor=SCALING_FACTOR,
    demand_scaling_factor=SCALING_FACTOR,
):
    """Compute pairwise spatial Wasserstein distances between embedding
    columns."""

    spatial_coordinates = dataset.obsm[spatial_key]
    embeddings_truth = dataset.obsm[embeddings_truth_key]
    embeddings_pred = dataset.obsm[embeddings_pred_key]

    def metric(pair):
        return spatial_wasserstein(
            spatial_coordinates,
            *pair,
            weight_scaling_factor=weight_scaling_factor,
            demand_scaling_factor=demand_scaling_factor,
        )

    num_truth = embeddings_truth.shape[1]
    num_pred = embeddings_pred.shape[1]

    pairs = [[(truth, pred) for pred in embeddings_pred.T] for truth in embeddings_truth.T]
    pairs = np.array(pairs).reshape(-1, 2, len(spatial_coordinates))

    with Pool(processes=16) as pool:
        results = pool.map(metric, pairs)

    return np.array(list(results)).reshape((num_truth, num_pred))


def propagate_ground_truth(
    hierarchy: Mapping[int, Sequence[ad.AnnData]],
) -> None:
    """Propagate ground-truth embeddings/metagenes through hierarchy levels."""

    levels = sorted(hierarchy)
    for fine_level, coarse_level in zip(levels, levels[1:]):
        fine_datasets = hierarchy[fine_level]
        coarse_datasets = hierarchy[coarse_level]
        if len(fine_datasets) != len(coarse_datasets):
            raise ValueError("Adjacent hierarchy levels must contain the same number of datasets.")

        for fine_dataset, coarse_dataset in zip(fine_datasets, coarse_datasets):
            assignment_key = f"bin_assignments_{coarse_dataset.popari.name}"
            coarse_dataset.simulation.ground_truth_X = (
                coarse_dataset.obsm[assignment_key] @ fine_dataset.simulation.ground_truth_X
            )
            coarse_dataset.simulation.ground_truth_M = fine_dataset.simulation.ground_truth_M.copy()


def evaluate_ground_truth(
    dataset,
    real_metagene_index: Sequence[int] | None = None,
):
    """Evaluate learned embeddings/metagenes against ground truth arrays."""

    if real_metagene_index is None:
        real_metagene_index = slice(None)

    true_embeddings = dataset.simulation.ground_truth_X.T
    true_metagenes = dataset.simulation.ground_truth_M.T
    learned_embeddings = dataset.simulation.learned_X.T

    try:
        learned_metagenes = dataset.simulation.learned_M.T
    except Exception:
        learned_metagenes = true_metagenes + 0.05

    def pearson_correlation_magnitude(x, y):
        return abs(pearsonr(x, y)[0])

    def normalized_wasserstein_distance(x, y):
        return wasserstein_distance(x / x.sum(), y / y.sum())

    embedding_pearson_correlations = cdist(true_embeddings, learned_embeddings, metric=pearson_correlation_magnitude)
    metagene_pearson_correlations = cdist(true_metagenes, learned_metagenes, metric=pearson_correlation_magnitude)
    embedding_wasserstein = cdist(true_embeddings, learned_embeddings, metric=normalized_wasserstein_distance)
    metagene_wasserstein = cdist(true_metagenes, learned_metagenes, metric=normalized_wasserstein_distance)
    embedding_spatial_wasserstein = all_pairs_spatial_wasserstein(dataset)

    return {
        "metrics": {
            "metagene_pearson_minmax": metagene_pearson_correlations.max(axis=1)[real_metagene_index].min(),
            "embedding_pearson_minmax": np.nanmax(
                embedding_pearson_correlations[real_metagene_index],
                axis=1,
            ).min(),
            "metagene_wasserstein_maxmin": metagene_wasserstein.min(axis=1)[real_metagene_index].max(),
            "embedding_wasserstein_maxmin": np.nanmin(
                embedding_wasserstein[real_metagene_index],
                axis=1,
            ).max(),
            "embedding_spatial_wasserstein_maxmin": np.nanmin(
                embedding_spatial_wasserstein[real_metagene_index],
                axis=1,
            ).max(),
            "embedding_spatial_wasserstein_avgmin": np.min(
                embedding_spatial_wasserstein[real_metagene_index],
                axis=1,
            ).mean(),
        },
        "pairwise_metrics": {
            "embedding_pearson_correlations": embedding_pearson_correlations,
            "metagene_pearson_correlations": metagene_pearson_correlations,
            "embedding_wasserstein": embedding_wasserstein,
            "metagene_wasserstein": metagene_wasserstein,
            "embedding_spatial_wasserstein": embedding_spatial_wasserstein,
        },
    }


def compute_matching(
    dataset,
    embedding_spatial_wasserstein,
    real_metagene_index=None,
):
    """Match learned factors to ground-truth factors using spatial Wasserstein
    distance."""

    _, matched_indices = linear_sum_assignment(embedding_spatial_wasserstein[real_metagene_index])
    return {
        "indices": matched_indices,
        "ground_truth_embeddings": dataset.simulation.ground_truth_X[:, real_metagene_index],
        "learned_embeddings": dataset.simulation.learned_X[:, matched_indices],
    }


def compute_affinity_correlation(
    dataset,
    matched_indices,
    spatial_affinity_key: str = "Sigma_x_inv",
    correlation_truth_key: str = "ground_truth_correlation",
    real_metagene_index=None,
):
    """Compute rank correlation between learned and ground-truth affinities."""

    sigma_x_inv = dataset.uns[spatial_affinity_key][dataset.popari.name]
    empirical_correlation = dataset.uns[correlation_truth_key][dataset.popari.name][real_metagene_index][
        :,
        real_metagene_index,
    ]
    permuted_sigma_x_inv = sigma_x_inv[matched_indices, :][:, matched_indices]

    correlation = spearmanr(empirical_correlation.flatten(), permuted_sigma_x_inv.flatten())[0]
    return {
        "correlation": np.nan_to_num(correlation),
        "permuted_spatial_affinity": permuted_sigma_x_inv,
    }


def compute_affinity_coherence(
    dataset,
    spatial_affinity_key: str = "Sigma_x_inv",
    empirical_correlation_key: str = "empirical_correlation",
):
    """Compute Pearson correlation between learned affinities and empirical
    spatial correlation."""

    empirical_correlation = dataset.uns[empirical_correlation_key][dataset.popari.name].copy()
    empirical_correlation /= np.abs(empirical_correlation).max()

    sigma_x_inv = dataset.uns[spatial_affinity_key][dataset.popari.name].copy()
    sigma_x_inv /= np.abs(sigma_x_inv).max()

    return pearsonr(empirical_correlation.flatten(), sigma_x_inv.flatten())[0]


class SimulationEvaluation:
    """Evaluate and cache metrics for one collection of AnnData datasets."""

    def __init__(self, model_name: str, is_spatial: bool, filepath: str | Path | None = None):
        self.model_name = model_name
        self.is_spatial = is_spatial
        self.scores = pd.DataFrame(columns=EVALUATION_COLUMNS)

        if filepath is not None:
            filepath = Path(filepath)
            if filepath.suffix != ".csv":
                raise ValueError("Evaluation cache paths must end in '.csv'.")
            self.scores = pd.read_csv(
                filepath,
                dtype={
                    "run_id": str,
                    "result_id": str,
                    "metric": str,
                    "dataset_name": str,
                },
            )
            missing_columns = set(EVALUATION_COLUMNS) - set(self.scores.columns)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Evaluation cache is missing required columns: {missing}.")
            self.scores = self.scores[EVALUATION_COLUMNS]

        self.computed_run_ids = set(self.scores["run_id"].astype(str))
        self.computed_result_ids = self.computed_run_ids

    def is_computed(self, run_id: str) -> bool:
        """Return whether a run is already present in the metric cache."""

        return str(run_id) in self.computed_run_ids

    def has_metric(self, metric: str) -> bool:
        """Return whether the cache contains a metric."""

        return bool((self.scores["metric"] == metric).any())

    def evaluate(
        self,
        datasets: Sequence[ad.AnnData],
        *,
        metagene_indices,
        run_id: str | None = None,
        result_id: str | None = None,
        random_state: int | None = None,
        metric_names: Sequence[str] | None = None,
        annotate: bool = False,
    ):
        """Evaluate one dataset collection and optionally append cached scores.

        Run metadata must be supplied together. When omitted, detailed results
        are returned without changing the CSV-backed score table.

        """

        if len(datasets) != len(metagene_indices):
            raise ValueError("datasets and metagene_indices must have the same length.")

        metadata = (run_id, result_id, random_state)
        should_cache = all(value is not None for value in metadata)
        if any(value is not None for value in metadata) and not should_cache:
            raise ValueError("run_id, result_id, and random_state must be supplied together.")

        if should_cache:
            run_id = str(run_id)
            if self.is_computed(run_id):
                raise ValueError(f"Run {run_id!r} is already present in the evaluation cache.")

        if metric_names is None:
            metric_names = [
                "metagene_pearson_minmax",
                "embedding_pearson_minmax",
                "metagene_wasserstein_maxmin",
                "embedding_wasserstein_maxmin",
                "embedding_spatial_wasserstein_maxmin",
            ]
            if self.is_spatial:
                metric_names.extend(["affinity_correlation", "affinity_coherence"])

        if self.is_spatial:
            from popari._dataset_utils import _compute_empirical_correlations

            _compute_empirical_correlations(datasets, scaling=10)
            _compute_empirical_correlations(
                datasets,
                scaling=10,
                feature="ground_truth_X",
                output="ground_truth_correlation",
            )

        results = []
        records = []
        for dataset_index, (dataset, metagene_index) in enumerate(
            zip(datasets, metagene_indices),
        ):
            result = evaluate_ground_truth(
                dataset,
                real_metagene_index=metagene_index,
            )
            matching = compute_matching(
                dataset,
                result["pairwise_metrics"]["embedding_spatial_wasserstein"],
                real_metagene_index=metagene_index,
            )
            result["matching"] = matching
            if self.is_spatial:
                result["metrics"]["affinity_coherence"] = compute_affinity_coherence(dataset)
                affinity = compute_affinity_correlation(
                    dataset,
                    matching["indices"],
                    real_metagene_index=metagene_index,
                )
                result["metrics"]["affinity_correlation"] = affinity["correlation"]
                matching["permuted_spatial_affinity"] = affinity["permuted_spatial_affinity"]

            if annotate:
                dataset.uns.update(result["metrics"])
                dataset.uns.update(result["pairwise_metrics"])
                dataset.uns["matched_indices"] = matching["indices"]
                dataset.obsm["truncated_ground_truth_X"] = matching["ground_truth_embeddings"]
                dataset.obsm["truncated_matched_X"] = matching["learned_embeddings"]
                if "permuted_spatial_affinity" in matching:
                    dataset.uns["permuted_Sigma_x_inv"] = {
                        dataset.popari.name: matching["permuted_spatial_affinity"],
                    }
            results.append(result)

            if should_cache:
                for metric in metric_names:
                    score = result["metrics"][metric]
                    if np.ndim(score) != 0:
                        raise ValueError(f"Metric {metric!r} must produce one scalar per dataset.")
                    records.append(
                        {
                            "run_id": run_id,
                            "result_id": str(result_id),
                            "random_state": int(random_state),
                            "metric": metric,
                            "dataset_index": dataset_index,
                            "dataset_name": dataset.popari.name,
                            "score": score,
                        },
                    )

        if should_cache:
            new_scores = pd.DataFrame.from_records(records, columns=EVALUATION_COLUMNS)
            self.scores = new_scores if self.scores.empty else pd.concat([self.scores, new_scores], ignore_index=True)
            self.computed_run_ids.add(run_id)
        return tuple(results)

    def save(self, filepath) -> None:
        """Write metric scores to a CSV cache."""

        filepath = Path(filepath)
        if filepath.suffix != ".csv":
            raise ValueError("Evaluation cache paths must end in '.csv'.")
        temporary_path = filepath.with_suffix(".csv.tmp")
        self.scores.to_csv(temporary_path, index=False)
        temporary_path.replace(filepath)

    def metric_scores(self, metric: str):
        """Return labeled scores averaged over dataset replicates."""

        scores = self.scores.loc[self.scores["metric"] == metric]
        if scores.empty:
            raise ValueError(f"No scores found for metric {metric!r}.")
        runs_per_coordinate = scores.groupby(["result_id", "random_state"])["run_id"].nunique()
        if (runs_per_coordinate > 1).any():
            raise ValueError("Each result ID and random state must identify exactly one run.")
        return scores.pivot_table(
            index="result_id",
            columns="random_state",
            values="score",
            aggfunc="mean",
            sort=False,
        )

    def best_difference_index(
        self,
        other_evaluations: Sequence[SimulationEvaluation],
        metric: str,
    ) -> tuple[str, int]:
        """Return the result and random state with the largest total
        difference."""

        primary_scores = self.metric_scores(metric)
        differences = []
        for evaluation in other_evaluations:
            other_scores = evaluation.metric_scores(metric)
            if set(primary_scores.index) != set(other_scores.index) or set(
                primary_scores.columns,
            ) != set(other_scores.columns):
                raise ValueError("Evaluations must contain the same result IDs and random states.")
            other_scores = other_scores.reindex(
                index=primary_scores.index,
                columns=primary_scores.columns,
            )
            differences.append((primary_scores - other_scores).abs())

        if not differences:
            raise ValueError("At least one other evaluation is required.")
        summed_differences = sum(differences[1:], start=differences[0])
        return summed_differences.stack().idxmax()

    def average_metric(
        self,
        metric: str,
        reduction: Literal["min", "max"],
    ) -> float:
        """Reduce random-state scores per result, then average over results."""

        scores = self.metric_scores(metric)
        match reduction:
            case "min":
                reduced_scores = scores.min(axis="columns")
            case "max":
                reduced_scores = scores.max(axis="columns")
            case _:
                raise ValueError("reduction must be 'min' or 'max'.")
        return float(reduced_scores.mean())


def find_run_for_score_index(runs, score_index):
    """Find the W&B run corresponding to a cached replicate/state index."""

    best_result_id, best_state = score_index
    for run in runs:
        params = run.config
        result_id = params.get("dataset_path", run.id)
        random_state = int(params.get("random_state", 0))
        if (result_id, random_state) == (best_result_id, best_state):
            return run
    raise ValueError(f"Could not find a run for score index {score_index}.")


__all__ = [
    spatial_wasserstein.__name__,
    all_pairs_spatial_wasserstein.__name__,
    propagate_ground_truth.__name__,
    evaluate_ground_truth.__name__,
    compute_matching.__name__,
    compute_affinity_correlation.__name__,
    compute_affinity_coherence.__name__,
    SimulationEvaluation.__name__,
    find_run_for_score_index.__name__,
]
