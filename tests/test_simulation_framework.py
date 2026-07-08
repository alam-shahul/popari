import numpy as np
import pytest

from popari.simulation_framework import (
    MultiReplicateSyntheticDataset,
    SimulationParameters,
    SyntheticDataset,
    sample_2D_points,
    sample_gaussian,
)


def _simulation_parameters() -> SimulationParameters:
    return SimulationParameters(
        num_genes=50,
        num_cells=64,
        num_real_metagenes=4,
        num_noise_metagenes=2,
        sig_y_scale=1.0,
        sig_x_scale=1.0,
        real_metagene_parameter=3.0,
        noise_metagene_parameter=2.0,
        lambda_s=1.0,
        metagene_variation_probabilities=[0, 0.2, 0.2, 0.0],
        cell_type_definitions={
            "type_a": [1, 0, 0, 0],
            "type_b": [0, 1, 0, 0],
            "type_c": [0, 0, 1, 0],
            "type_d": [0, 0, 0, 1],
        },
        spatial_distributions={
            "L0": {"type_a": 0.5, "type_b": 0.5},
            "L1": {"type_c": 0.5, "type_d": 0.5},
        },
    )


def _assign_domains(dataset):
    points = dataset.obsm["spatial"]
    midpoint = np.median(points[:, 0])
    domains = {
        "L0": points[points[:, 0] <= midpoint],
        "L1": points[points[:, 0] > midpoint],
    }
    dataset.domain_canvas.load_domains(domains)
    dataset.assign_domain_labels()


@pytest.mark.baseline
def test_sample_gaussian_is_deterministic_with_seed():
    rng = np.random.default_rng(0)
    covariance = np.diag(rng.random(4) + 0.5)
    means = rng.random(4)

    samples_1 = sample_gaussian(covariance, means, N=10, random_state=0)
    samples_2 = sample_gaussian(covariance, means, N=10, random_state=0)
    samples_3 = sample_gaussian(covariance, means, N=10, random_state=1)

    assert np.allclose(samples_1, samples_2)
    assert not np.allclose(samples_1, samples_3)


@pytest.mark.baseline
def test_sample_2d_points_respects_minimum_distance():
    minimum_distance = 0.1
    points = sample_2D_points(12, minimum_distance=minimum_distance, random_state=0)

    for index, point in enumerate(points):
        distances = np.linalg.norm(points[index + 1 :] - point, axis=1)
        assert np.all(distances > minimum_distance)


@pytest.mark.expensive
def test_synthetic_dataset_simulation_is_self_consistent():
    dataset = SyntheticDataset("rep0", _simulation_parameters(), random_state=0, verbose=0)
    _assign_domains(dataset)
    dataset.simulate_expression()

    assert dataset.X.shape == (dataset.params.num_cells, dataset.params.num_genes)
    assert np.all(dataset.X >= 0)
    assert dataset.obsm["ground_truth_X"].shape[1] == (
        dataset.params.num_real_metagenes + dataset.params.num_noise_metagenes
    )
    assert dataset.uns["ground_truth_M"][dataset.popari.name].shape == (
        dataset.params.num_genes,
        dataset.params.num_real_metagenes + dataset.params.num_noise_metagenes,
    )


@pytest.mark.expensive
def test_synthetic_dataset_is_deterministic_for_fixed_seed():
    dataset_0 = SyntheticDataset("rep0", _simulation_parameters(), random_state=0, verbose=0)
    dataset_1 = SyntheticDataset("rep0", _simulation_parameters(), random_state=0, verbose=0)
    _assign_domains(dataset_0)
    _assign_domains(dataset_1)

    dataset_0.simulate_expression()
    dataset_1.simulate_expression()

    assert np.allclose(dataset_0.X, dataset_1.X)
    assert np.allclose(dataset_0.obsm["ground_truth_X"], dataset_1.obsm["ground_truth_X"])
    assert np.allclose(
        dataset_0.uns["ground_truth_M"][dataset_0.popari.name],
        dataset_1.uns["ground_truth_M"][dataset_1.popari.name],
    )


@pytest.mark.expensive
def test_multireplicate_synthetic_dataset_shares_metagenes_and_varies_embeddings():
    params = _simulation_parameters()
    multireplicate = MultiReplicateSyntheticDataset(
        replicate_parameters={"original": params, "shifted": params},
        dataset_constructor=SyntheticDataset,
        random_state=0,
        verbose=0,
    )

    for dataset in multireplicate:
        _assign_domains(dataset)

    multireplicate.simulate_expression()

    original, shifted = list(multireplicate)
    assert np.allclose(
        original.uns["ground_truth_M"][original.popari.name],
        shifted.uns["ground_truth_M"][shifted.popari.name],
    )
    assert not np.allclose(original.obsm["ground_truth_X"], shifted.obsm["ground_truth_X"])
