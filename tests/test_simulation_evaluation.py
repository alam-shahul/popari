import json

import anndata as ad
import numpy as np
import pytest

from popari.simulation import metrics as simulation_metrics
from popari.simulation.evaluation import (
    SimulationEvaluation,
    best_metric_difference_index,
    find_run_for_score_index,
    metric_score_array,
)


class FakeRun:
    def __init__(self, run_id, config=None):
        self.id = run_id
        self.config = config or {}


class FakeView:
    def __init__(self, datasets):
        self.datasets = datasets


class FakeModel:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        self.hierarchical_levels = len(hierarchy)


def test_evaluation_cache_marks_matching_runs_computed(tmp_path):
    cache_path = tmp_path / "evaluation.json"
    cache_path.write_text(
        json.dumps(
            {
                "scores": {"0": {"metric": {"replicate": [[1.0]]}}},
                "computed_run_ids": ["run-a"],
            },
        ),
    )

    evaluation = SimulationEvaluation.from_wandb_runs(
        "Popari",
        [FakeRun("run-a"), FakeRun("run-b")],
        is_spatial=False,
        filepath=cache_path,
    )

    assert evaluation.is_computed.tolist() == [True, False]
    assert evaluation.computed_run_ids == {"run-a"}
    assert evaluation.scores[0]["metric"]["replicate"] == [[1.0]]


def test_evaluation_cache_loads_old_raw_score_json(tmp_path):
    cache_path = tmp_path / "old_evaluation.json"
    cache_path.write_text(json.dumps({"0": {"metric": {"replicate": [[1.0]]}}}))

    evaluation = SimulationEvaluation.from_wandb_runs(
        "Popari",
        [FakeRun("run-a")],
        is_spatial=False,
        filepath=cache_path,
    )

    assert evaluation.is_computed.tolist() == [False]
    assert evaluation.computed_run_ids == set()
    assert evaluation.scores[0]["metric"]["replicate"] == [[1.0]]


def test_evaluation_cache_save_writes_scores_and_completed_run_ids(tmp_path):
    cache_path = tmp_path / "evaluation.json"
    evaluation = SimulationEvaluation.from_wandb_runs("Popari", [FakeRun("run-a")], is_spatial=False)
    evaluation.scores[0]["metric"]["replicate"].append([1.0])
    evaluation._mark_computed(0, FakeRun("run-a"))

    evaluation.save(cache_path)

    cache = json.loads(cache_path.read_text())
    assert cache["computed_run_ids"] == ["run-a"]
    assert cache["scores"]["0"]["metric"]["replicate"] == [[1.0]]


def test_anndata_evaluation_records_scores_for_all_levels(monkeypatch):
    def make_dataset(name):
        dataset = ad.AnnData(X=np.ones((2, 2)))
        dataset.popari.name = name
        dataset.obs["batch"] = name
        dataset.obsm["X"] = np.ones((2, 1))
        dataset.obsm["ground_truth_X"] = np.ones((2, 1))
        dataset.uns["M"] = {name: np.ones((2, 1))}
        return dataset

    hierarchy = {
        0: (make_dataset("replicate_level_0"),),
        1: (make_dataset("replicate_level_1"),),
    }

    monkeypatch.setattr(
        "popari.wandb_util.load_popari_anndata_from_wandb",
        lambda *args, **kwargs: hierarchy,
    )
    monkeypatch.setattr(
        simulation_metrics,
        "evaluate_ground_truth",
        lambda dataset, real_metagene_index=None: dataset.uns.update({"metric": len(dataset.obs)}),
    )
    monkeypatch.setattr(simulation_metrics, "compute_matching", lambda dataset, real_metagene_index=None: None)

    evaluation = SimulationEvaluation.from_wandb_runs(
        "Popari",
        [FakeRun("run-a", config={"dataset_path": "replicate"})],
        is_spatial=False,
    )
    evaluation.compute_metrics([slice(None)], metric_names=["metric"])

    assert evaluation.scores[0]["metric"]["replicate"] == [[2]]
    assert evaluation.scores[1]["metric"]["replicate"] == [[2]]
    assert evaluation.is_computed.tolist() == [True]


def test_metric_score_array_averages_over_dataset_replicates():
    evaluation = SimulationEvaluation.from_wandb_runs("Popari", [], is_spatial=False)
    evaluation.scores[0]["metric"]["replicate_0"].append([1.0, 3.0])
    evaluation.scores[0]["metric"]["replicate_0"].append([2.0, 4.0])
    evaluation.scores[0]["metric"]["replicate_1"].append([5.0, 7.0])
    evaluation.scores[0]["metric"]["replicate_1"].append([6.0, 8.0])

    scores = metric_score_array(evaluation, "metric", level=0)

    np.testing.assert_allclose(scores, [[2.0, 3.0], [6.0, 7.0]])


def test_best_metric_difference_index_finds_largest_total_difference():
    primary = SimulationEvaluation.from_wandb_runs("Popari", [], is_spatial=False)
    other_a = SimulationEvaluation.from_wandb_runs("A", [], is_spatial=False)
    other_b = SimulationEvaluation.from_wandb_runs("B", [], is_spatial=False)

    primary.scores[0]["metric"]["replicate_0"].append([0.0])
    primary.scores[0]["metric"]["replicate_0"].append([10.0])
    primary.scores[0]["metric"]["replicate_1"].append([3.0])
    primary.scores[0]["metric"]["replicate_1"].append([4.0])
    other_a.scores[0]["metric"]["replicate_0"].append([0.0])
    other_a.scores[0]["metric"]["replicate_0"].append([0.0])
    other_a.scores[0]["metric"]["replicate_1"].append([3.0])
    other_a.scores[0]["metric"]["replicate_1"].append([4.0])
    other_b.scores[0]["metric"]["replicate_0"].append([0.0])
    other_b.scores[0]["metric"]["replicate_0"].append([1.0])
    other_b.scores[0]["metric"]["replicate_1"].append([3.0])
    other_b.scores[0]["metric"]["replicate_1"].append([4.0])

    assert best_metric_difference_index(primary, [other_a, other_b], "metric", level=0) == (0, 1)


def test_find_run_for_score_index_matches_dataset_replicate_and_random_state():
    runs = [
        FakeRun("run-a", {"dataset_path": "path/random_state_0", "random_state": 0}),
        FakeRun("run-b", {"dataset_path": "path/random_state_3", "random_state": 2}),
    ]
    evaluation = SimulationEvaluation.from_wandb_runs("Popari", runs, is_spatial=False)

    assert find_run_for_score_index(evaluation, (3, 2)).id == "run-b"


def test_find_run_for_score_index_raises_for_missing_match():
    evaluation = SimulationEvaluation.from_wandb_runs(
        "Popari",
        [FakeRun("run-a", {"dataset_path": "path/random_state_0", "random_state": 0})],
        is_spatial=False,
    )

    with pytest.raises(ValueError, match="Could not find a run"):
        find_run_for_score_index(evaluation, (1, 0))


def test_simulation_evaluation_from_anndata_records_scores(monkeypatch):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.popari.name = "nsf"
    dataset.obs["batch"] = "nsf"
    dataset.obsm["X"] = np.ones((2, 1))
    dataset.obsm["ground_truth_X"] = np.ones((2, 1))
    dataset.uns["M"] = {"nsf": np.ones((2, 1))}

    monkeypatch.setattr(
        simulation_metrics,
        "evaluate_ground_truth",
        lambda dataset, real_metagene_index=None: dataset.uns.update({"metric": len(dataset.obs)}),
    )
    monkeypatch.setattr(simulation_metrics, "compute_matching", lambda dataset, real_metagene_index=None: None)

    evaluation = SimulationEvaluation.from_anndata("NSF", [{0: (dataset,)}], result_ids=["replicate"], is_spatial=False)
    evaluation.compute_metrics([slice(None)], metric_names=["metric"])

    assert evaluation.scores[0]["metric"]["replicate"] == [[2]]
    assert evaluation.is_computed.tolist() == [True]
    assert evaluation.computed_result_ids == {"replicate"}


def test_simulation_evaluation_from_models_records_scores(monkeypatch):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.popari.name = "model_dataset"
    model = FakeModel({0: FakeView((dataset,))})

    def evaluate_model(model, metagene_indices, is_spatial):
        model.hierarchy[0].datasets[0].uns["metric"] = 3.0

    monkeypatch.setattr(simulation_metrics, "evaluate_model", evaluate_model)

    evaluation = SimulationEvaluation.from_models(
        "Popari",
        [model],
        result_ids=["model-run"],
        is_spatial=False,
    )
    evaluation.compute_metrics([slice(None)], metric_names=["metric"])

    assert evaluation.scores[0]["metric"]["model-run"] == [[3.0]]
    assert evaluation.is_computed.tolist() == [True]
    assert evaluation.computed_result_ids == {"model-run"}
