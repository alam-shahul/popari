import anndata as ad
import numpy as np
import pandas as pd
import pytest

import popari.simulation.metrics as simulation_metrics
from popari.simulation.metrics import SimulationEvaluation, find_run_for_score_index, propagate_ground_truth


class FakeRun:
    def __init__(self, run_id, config=None):
        self.id = run_id
        self.config = config or {}


def score_frame(records):
    rows = [
        {
            "run_id": run_id,
            "result_id": result_id,
            "random_state": random_state,
            "metric": metric,
            "dataset_index": dataset_index,
            "dataset_name": f"dataset_{dataset_index}",
            "score": score,
        }
        for run_id, result_id, random_state, metric, dataset_index, score in records
    ]
    return pd.DataFrame(rows, columns=simulation_metrics.EVALUATION_COLUMNS)


def simulation_dataset(name, ground_truth_x, ground_truth_m):
    dataset = ad.AnnData(X=np.ones((len(ground_truth_x), len(ground_truth_m))))
    dataset.popari.name = name
    dataset.obs["batch"] = name
    dataset.simulation.ground_truth_X = np.asarray(ground_truth_x)
    dataset.simulation.ground_truth_M = np.asarray(ground_truth_m)
    return dataset


def test_propagate_ground_truth_across_multiple_levels():
    fine = simulation_dataset("replicate", [[1.0], [3.0], [5.0]], [[2.0], [4.0]])
    middle = simulation_dataset("replicate_level_1", [[-1.0], [-1.0]], [[-1.0], [-1.0]])
    coarse = simulation_dataset("replicate_level_2", [[-1.0]], [[-1.0], [-1.0]])
    middle.obsm["bin_assignments_replicate_level_1"] = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    coarse.obsm["bin_assignments_replicate_level_2"] = np.array([[0.25, 0.75]])

    propagate_ground_truth({0: (fine,), 1: (middle,), 2: (coarse,)})

    np.testing.assert_allclose(middle.simulation.ground_truth_X, [[2.0], [5.0]])
    np.testing.assert_allclose(coarse.simulation.ground_truth_X, [[4.25]])
    np.testing.assert_allclose(middle.simulation.ground_truth_M, fine.simulation.ground_truth_M)
    np.testing.assert_allclose(coarse.simulation.ground_truth_M, fine.simulation.ground_truth_M)


def test_propagate_ground_truth_is_noop_for_flat_results():
    dataset = ad.AnnData(X=np.ones((2, 2)))

    propagate_ground_truth({0: (dataset,)})

    assert "ground_truth_X" not in dataset.obsm
    assert "ground_truth_M" not in dataset.uns


def test_propagate_ground_truth_rejects_mismatched_levels():
    fine = simulation_dataset("replicate", [[1.0]], [[1.0]])
    coarse = simulation_dataset("replicate_level_1", [[1.0]], [[1.0]])

    with pytest.raises(ValueError, match="same number of datasets"):
        propagate_ground_truth({0: (fine, fine.copy()), 1: (coarse,)})


def test_propagate_ground_truth_requires_canonical_assignment_key():
    fine = simulation_dataset("replicate", [[1.0]], [[1.0]])
    coarse = simulation_dataset("replicate_level_1", [[1.0]], [[1.0]])

    with pytest.raises(KeyError, match="bin_assignments_replicate_level_1"):
        propagate_ground_truth({0: (fine,), 1: (coarse,)})


def test_evaluation_cache_marks_matching_runs_computed(tmp_path):
    cache_path = tmp_path / "evaluation.csv"
    score_frame([("run-a", "replicate", 0, "metric", 0, 1.0)]).to_csv(cache_path, index=False)

    evaluation = SimulationEvaluation("Popari", is_spatial=False, filepath=cache_path)

    assert evaluation.is_computed("run-a")
    assert not evaluation.is_computed("run-b")
    assert evaluation.computed_run_ids == {"run-a"}
    assert evaluation.scores.loc[0, "score"] == 1.0


def test_evaluation_cache_rejects_json_paths(tmp_path):
    with pytest.raises(ValueError, match="must end in '.csv'"):
        SimulationEvaluation("Popari", is_spatial=False, filepath=tmp_path / "evaluation.json")


def test_evaluation_cache_save_writes_scores_atomically(tmp_path):
    cache_path = tmp_path / "evaluation.csv"
    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    evaluation.scores = score_frame([("run-a", "replicate", 0, "metric", 0, 1.0)])
    evaluation.computed_run_ids.add("run-a")

    evaluation.save(cache_path)

    cache = pd.read_csv(cache_path)
    assert cache.loc[0, "run_id"] == "run-a"
    assert cache.loc[0, "score"] == 1.0
    assert not (tmp_path / "evaluation.csv.tmp").exists()


def test_evaluation_cache_rejects_missing_columns(tmp_path):
    cache_path = tmp_path / "evaluation.csv"
    pd.DataFrame({"run_id": ["run-a"]}).to_csv(cache_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        SimulationEvaluation("Popari", is_spatial=False, filepath=cache_path)


def test_simulation_evaluation_records_scores_for_one_dataset_collection(monkeypatch):
    dataset = simulation_dataset("replicate", [[1.0], [1.0]], [[1.0], [1.0]])
    dataset.obsm["X"] = np.ones((2, 1))
    monkeypatch.setattr(
        simulation_metrics,
        "evaluate_ground_truth",
        lambda dataset, real_metagene_index=None: {
            "metrics": {"metric": len(dataset.obs)},
            "pairwise_metrics": {"embedding_spatial_wasserstein": np.zeros((1, 1))},
        },
    )

    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    evaluation.evaluate(
        (dataset,),
        run_id="run-a",
        result_id="replicate",
        random_state=3,
        metagene_indices=[slice(None)],
        metric_names=["metric"],
    )

    assert evaluation.scores[["dataset_name", "score"]].to_dict("records") == [
        {"dataset_name": "replicate", "score": 2},
    ]
    assert evaluation.is_computed("run-a")
    assert evaluation.scores["random_state"].unique().tolist() == [3]


def test_simulation_evaluation_rejects_duplicate_run(monkeypatch):
    dataset = simulation_dataset("replicate", [[1.0]], [[1.0]])
    dataset.obsm["X"] = np.ones((1, 1))
    monkeypatch.setattr(
        simulation_metrics,
        "evaluate_ground_truth",
        lambda dataset, real_metagene_index=None: {
            "metrics": {"metric": 1.0},
            "pairwise_metrics": {"embedding_spatial_wasserstein": np.zeros((1, 1))},
        },
    )
    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    arguments = {
        "run_id": "run-a",
        "result_id": "replicate",
        "random_state": 0,
        "metagene_indices": [slice(None)],
        "metric_names": ["metric"],
    }

    evaluation.evaluate((dataset,), **arguments)

    with pytest.raises(ValueError, match="already present"):
        evaluation.evaluate((dataset,), **arguments)


def test_simulation_evaluation_requires_complete_run_metadata():
    dataset = simulation_dataset("replicate", [[1.0]], [[1.0]])
    evaluation = SimulationEvaluation("Popari", is_spatial=False)

    with pytest.raises(ValueError, match="must be supplied together"):
        evaluation.evaluate(
            (dataset,),
            metagene_indices=[slice(None)],
            run_id="run-a",
        )


def test_simulation_evaluation_rejects_mismatched_metagene_indices():
    dataset = simulation_dataset("replicate", [[1.0]], [[1.0]])
    evaluation = SimulationEvaluation("Popari", is_spatial=False)

    with pytest.raises(ValueError, match="must have the same length"):
        evaluation.evaluate((dataset,), metagene_indices=[])


def test_evaluate_ground_truth_returns_results_without_annotating(monkeypatch):
    dataset = simulation_dataset("replicate", [[1.0], [2.0]], [[1.0], [2.0]])
    dataset.obsm["X"] = np.array([[1.0], [2.0]])
    dataset.uns["M"] = np.array([[1.0], [2.0]])
    monkeypatch.setattr(simulation_metrics, "all_pairs_spatial_wasserstein", lambda dataset: np.zeros((1, 1)))

    result = simulation_metrics.evaluate_ground_truth(dataset)

    assert "embedding_pearson_minmax" in result["metrics"]
    assert "embedding_spatial_wasserstein" in result["pairwise_metrics"]
    assert "embedding_pearson_minmax" not in dataset.uns
    assert "embedding_spatial_wasserstein" not in dataset.uns


def test_evaluate_annotates_only_when_requested(monkeypatch):
    def evaluation_result(dataset, real_metagene_index=None):
        return {
            "metrics": {"metric": 1.0},
            "pairwise_metrics": {"embedding_spatial_wasserstein": np.zeros((1, 1))},
        }

    dataset = simulation_dataset("replicate", [[1.0]], [[1.0]])
    dataset.obsm["X"] = np.ones((1, 1))
    monkeypatch.setattr(simulation_metrics, "evaluate_ground_truth", evaluation_result)

    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    results = evaluation.evaluate(
        (dataset,),
        metagene_indices=[slice(None)],
    )

    assert results[0]["metrics"] == {"metric": 1.0}
    assert evaluation.scores.empty
    assert "metric" not in dataset.uns
    assert "truncated_matched_X" not in dataset.obsm

    evaluation.evaluate(
        (dataset,),
        metagene_indices=[slice(None)],
        annotate=True,
    )

    assert dataset.uns["metric"] == 1.0
    assert "truncated_matched_X" in dataset.obsm


def test_metric_scores_averages_over_dataset_replicates():
    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    evaluation.scores = score_frame(
        [
            ("run-a", "replicate_0", 9, "metric", 0, 1.0),
            ("run-a", "replicate_0", 9, "metric", 1, 3.0),
            ("run-b", "replicate_0", 2, "metric", 0, 2.0),
            ("run-b", "replicate_0", 2, "metric", 1, 4.0),
            ("run-c", "replicate_1", 9, "metric", 0, 5.0),
            ("run-c", "replicate_1", 9, "metric", 1, 7.0),
            ("run-d", "replicate_1", 2, "metric", 0, 6.0),
            ("run-d", "replicate_1", 2, "metric", 1, 8.0),
        ],
    )

    scores = evaluation.metric_scores("metric")

    assert scores.index.tolist() == ["replicate_0", "replicate_1"]
    assert scores.columns.tolist() == [9, 2]
    np.testing.assert_allclose(scores.to_numpy(), [[2.0, 3.0], [6.0, 7.0]])


def test_has_metric_checks_metric_name():
    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    evaluation.scores = score_frame(
        [("run-a", "replicate_0", 0, "metric", 0, 1.0)],
    )

    assert evaluation.has_metric("metric")
    assert not evaluation.has_metric("other")


def test_best_difference_index_finds_largest_total_difference():
    primary = SimulationEvaluation("Popari", is_spatial=False)
    other_a = SimulationEvaluation("A", is_spatial=False)
    other_b = SimulationEvaluation("B", is_spatial=False)

    primary.scores = score_frame(
        [
            ("p0", "replicate_0", 2, "metric", 0, 0.0),
            ("p1", "replicate_0", 9, "metric", 0, 10.0),
            ("p2", "replicate_1", 2, "metric", 0, 3.0),
            ("p3", "replicate_1", 9, "metric", 0, 4.0),
        ],
    )
    other_a.scores = score_frame(
        [
            ("a3", "replicate_1", 9, "metric", 0, 4.0),
            ("a2", "replicate_1", 2, "metric", 0, 3.0),
            ("a1", "replicate_0", 9, "metric", 0, 0.0),
            ("a0", "replicate_0", 2, "metric", 0, 0.0),
        ],
    )
    other_b.scores = score_frame(
        [
            ("b0", "replicate_0", 2, "metric", 0, 0.0),
            ("b1", "replicate_0", 9, "metric", 0, 1.0),
            ("b2", "replicate_1", 2, "metric", 0, 3.0),
            ("b3", "replicate_1", 9, "metric", 0, 4.0),
        ],
    )

    assert primary.best_difference_index([other_a, other_b], "metric") == (
        "replicate_0",
        9,
    )


def test_average_metric_reduces_states_then_averages_results():
    evaluation = SimulationEvaluation("Popari", is_spatial=False)
    evaluation.scores = score_frame(
        [
            ("run-a", "replicate_0", 1, "metric", 0, 1.0),
            ("run-b", "replicate_0", 2, "metric", 0, 3.0),
            ("run-c", "replicate_1", 1, "metric", 0, 5.0),
            ("run-d", "replicate_1", 2, "metric", 0, 9.0),
        ],
    )

    assert evaluation.average_metric("metric", "min") == 3.0
    assert evaluation.average_metric("metric", "max") == 6.0


def test_find_run_for_score_index_matches_dataset_replicate_and_random_state():
    runs = [
        FakeRun("run-a", {"dataset_path": "path/random_state_0", "random_state": 0}),
        FakeRun("run-b", {"dataset_path": "path/random_state_3", "random_state": 2}),
    ]
    assert find_run_for_score_index(runs, ("path/random_state_3", 2)).id == "run-b"


def test_find_run_for_score_index_raises_for_missing_match():
    runs = [FakeRun("run-a", {"dataset_path": "path/random_state_0", "random_state": 0})]

    with pytest.raises(ValueError, match="Could not find a run"):
        find_run_for_score_index(runs, ("missing", 0))
