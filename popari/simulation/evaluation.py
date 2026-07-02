"""Evaluation cache and score utilities for simulation experiments."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Optional, Sequence

import numpy as np

from popari.simulation import metrics as simulation_metrics


class NestedDefaultDict(defaultdict):
    """Nested extension of ``defaultdict`` for structured metric storage."""

    def __init__(self, type_constructor, levels: int = 1):
        if levels == 1:
            super().__init__(type_constructor)
        elif levels > 1:
            super().__init__(lambda: NestedDefaultDict(type_constructor, levels=levels - 1))
        else:
            raise ValueError("levels must be at least 1")


class CachedEvaluation:
    """Shared cache bookkeeping for evaluation helpers."""

    def _initialize_cache(
        self,
        results,
        filepath: str | None = None,
        cache_id: Callable | None = None,
    ):
        self.results = list(results)
        self.runs = self.results
        self._cache_id = cache_id or (lambda result: str(result))
        self.computed_result_ids = set()
        self.computed_run_ids = self.computed_result_ids
        self.is_computed = np.full(len(self.results), False)

        if filepath is None:
            self.scores = NestedDefaultDict(list, levels=3)
            return

        with open(filepath) as f:
            cache = json.load(f)

        if "scores" in cache:
            scores = cache["scores"]
            self.computed_result_ids = set(cache.get("computed_result_ids", cache.get("computed_run_ids", [])))
            self.computed_run_ids = self.computed_result_ids
        else:
            scores = cache

        self.scores = {int(key): value for key, value in scores.items()}
        self.is_computed = np.array([self._cache_id(result) in self.computed_result_ids for result in self.results])

    def _mark_computed(self, index: int, result) -> None:
        self.is_computed[index] = True
        self.computed_result_ids.add(self._cache_id(result))

    def save(self, filepath):
        """Serialize metric scores to JSON."""

        completed_ids = sorted(self.computed_result_ids)
        with open(filepath, "w") as f:
            json.dump(
                {
                    "scores": self.scores,
                    "computed_result_ids": completed_ids,
                    "computed_run_ids": completed_ids,
                },
                f,
            )


class SimulationEvaluation(CachedEvaluation):
    """Evaluate simulation results from models, W&B runs, or AnnData
    hierarchies."""

    def __init__(
        self,
        model_name,
        results,
        load_result,
        is_spatial: bool,
        filepath: str | None = None,
        cache_id: Callable | None = None,
    ):
        self.model_name = model_name
        self.load_result = load_result
        self.is_spatial = is_spatial
        self._initialize_cache(results, filepath, cache_id=cache_id)

    @classmethod
    def from_wandb_runs(
        cls,
        model_name,
        runs,
        is_spatial: bool,
        filepath: str | None = None,
    ):
        """Create an evaluator for migrated W&B AnnData artifacts."""

        def load_result(run, metagene_indices, metric_names):
            from popari.wandb_util import load_popari_anndata_from_wandb

            hierarchy = load_popari_anndata_from_wandb(run_id(run), artifact_stem=None)
            simulation_metrics.evaluate_anndata_hierarchy(
                hierarchy,
                metagene_indices,
                is_spatial=is_spatial,
                metric_names=metric_names,
            )
            return run_config(run).get("dataset_path", run_id(run)), hierarchy

        return cls(
            model_name,
            runs,
            load_result,
            is_spatial=is_spatial,
            filepath=filepath,
            cache_id=run_id,
        )

    @classmethod
    def from_models(
        cls,
        model_name,
        models,
        is_spatial: bool,
        result_ids: Sequence[str] | None = None,
        filepath: str | None = None,
    ):
        """Create an evaluator for already-instantiated Popari-like models."""

        if isinstance(models, Mapping):
            result_ids = tuple(models.keys())
            model_values = tuple(models.values())
        else:
            model_values = tuple(models)
            result_ids = tuple(result_ids or [str(index) for index in range(len(model_values))])

        if len(result_ids) != len(model_values):
            raise ValueError("`result_ids` must have one value per model.")

        model_records = list(zip(result_ids, model_values))

        def load_result(record, metagene_indices, metric_names):
            result_id, model = record
            simulation_metrics.evaluate_model(
                model,
                metagene_indices,
                is_spatial=is_spatial,
            )
            hierarchy = {level: tuple(model.hierarchy[level].datasets) for level in range(model.hierarchical_levels)}
            return str(result_id), hierarchy

        return cls(
            model_name,
            model_records,
            load_result,
            is_spatial=is_spatial,
            filepath=filepath,
            cache_id=lambda record: str(record[0]),
        )

    @classmethod
    def from_anndata(
        cls,
        model_name,
        hierarchies,
        is_spatial: bool,
        result_ids: Sequence[str] | None = None,
        filepath: str | None = None,
    ):
        """Create an evaluator for already-loaded AnnData hierarchies."""

        if isinstance(hierarchies, Mapping):
            result_ids = tuple(hierarchies.keys())
            hierarchy_values = tuple(hierarchies.values())
        else:
            hierarchy_values = tuple(hierarchies)
            result_ids = tuple(result_ids or [str(index) for index in range(len(hierarchy_values))])

        if len(result_ids) != len(hierarchy_values):
            raise ValueError("`result_ids` must have one value per hierarchy.")

        hierarchy_records = list(zip(result_ids, hierarchy_values))

        def load_result(record, metagene_indices, metric_names):
            result_id, hierarchy = record
            simulation_metrics.evaluate_anndata_hierarchy(
                hierarchy,
                metagene_indices,
                is_spatial=is_spatial,
                metric_names=metric_names,
            )
            return str(result_id), hierarchy

        return cls(
            model_name,
            hierarchy_records,
            load_result,
            is_spatial=is_spatial,
            filepath=filepath,
            cache_id=lambda record: str(record[0]),
        )

    def compute_metrics(self, metagene_indices, metric_names: Sequence[str] | None = None):
        """Load/evaluate each result and store metric arrays."""

        from tqdm.notebook import tqdm

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

        for index, result in enumerate(tqdm(self.results)):
            if self.is_computed[index]:
                continue

            result_id, hierarchy = self.load_result(result, metagene_indices, metric_names)

            for level in sorted(hierarchy):
                datasets = hierarchy[level]
                print(f"Computing level {level} metrics...")
                for metric in metric_names:
                    self.scores[level][metric][result_id].append([dataset.uns[metric] for dataset in datasets])

            self._mark_computed(index, result)


def metric_score_array(evaluation, metric: str, level: int = 0):
    """Return cached metric scores as an array averaged over dataset
    replicates."""

    scores = evaluation.scores[level][metric]
    scores = np.array([replicate_scores for replicate_scores in scores.values()])
    return scores.mean(axis=-1)


def best_metric_difference_index(primary_evaluation, other_evaluations, metric: str, level: int = 0):
    """Find the score index where the primary evaluation differs most from
    comparisons."""

    primary_scores = metric_score_array(primary_evaluation, metric, level=level)
    differences = [
        np.abs(primary_scores - metric_score_array(other_evaluation, metric, level=level))
        for other_evaluation in other_evaluations
    ]
    summed_differences = np.sum(differences, axis=0)
    return np.unravel_index(np.argmax(summed_differences), summed_differences.shape)


def find_run_for_score_index(evaluation, score_index):
    """Find the W&B run matching a cached metric score index."""

    best_replicate, best_state = score_index
    pattern = re.compile("random_state_[0-9]+")
    for run in evaluation.runs:
        params = run_config(run)
        matches = pattern.findall(params.get("dataset_path", ""))
        if not matches:
            continue

        replicate = int(matches[0].split("_")[-1])
        random_state = int(params.get("random_state", 0))
        if (replicate, random_state) == (best_replicate, best_state):
            return run

    raise ValueError(f"Could not find a run for score index {score_index}.")


def get_average_metric(evaluation, metric, direction, level: int = 0):
    """Average metric over replicates and reduce over random seeds."""

    scores = metric_score_array(evaluation, metric, level=level)
    reduction = np.max if direction == "larger" else np.min
    return reduction(scores, axis=-1).mean()


def run_id(run):
    """Return a W&B-like run ID."""

    return run.id


def run_config(run):
    """Return a W&B-like run config."""

    return run.config


__all__ = [
    NestedDefaultDict.__name__,
    CachedEvaluation.__name__,
    SimulationEvaluation.__name__,
    metric_score_array.__name__,
    best_metric_difference_index.__name__,
    find_run_for_score_index.__name__,
    get_average_metric.__name__,
    run_id.__name__,
    run_config.__name__,
]
