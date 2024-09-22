from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gamma

from popari.simulation_framework import (
    SimulationParameters,
    SyntheticDataset,
    sample_2D_points,
    sample_gaussian,
    sample_normalized_embeddings,
)


@pytest.fixture(scope="module")
def parameters() -> SimulationParameters:

    parameters = {
        "num_genes": 500,
        "num_cells": 500,
        "num_real_metagenes": 10,
        "num_noise_metagenes": 3,
        "sig_y_scale": 3.0,
        "sig_x_scale": 3.0,
        "real_metagene_parameter": 4.0,
        "noise_metagene_parameter": 4.0,
        "lambda_s": 1.0,
        "metagene_variation_probabilities": [0, 0.1, 0.1, 0.1, 0, 0.25, 0, 0.25, 0, 0.25],
        "cell_type_definitions": {
            "Excitatory L1": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "Excitatory L2": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "Excitatory L3": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "Excitatory L4": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "Inhibitory 1": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "Inhibitory 2": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            "Non-Neuron L1": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "Non-Neuron Ubiquitous": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        },
        "spatial_distributions": {
            "L1": {
                "Excitatory L1": 0.33,
                "Inhibitory 1": 0.1,
                "Non-Neuron L1": 0.5,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L2": {
                "Excitatory L2": 0.93,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L3": {
                "Excitatory L3": 0.53,
                "Inhibitory 1": 0.1,
                "Inhibitory 2": 0.3,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L4": {
                "Excitatory L4": 0.73,
                "Inhibitory 1": 0.1,
                "Inhibitory 2": 0.1,
                "Non-Neuron Ubiquitous": 0.07,
            },
        },
    }

    simulation_parameters = SimulationParameters(**parameters)

    return simulation_parameters


def test_sample_gaussian():
    test_size = 10

    rng = np.random.default_rng(0)
    covariance = rng.random(size=test_size) * np.eye(test_size)
    means = rng.random(size=test_size)

    rng = np.random.default_rng(0)
    samples_1 = sample_gaussian(covariance, means, N=5, random_state=rng)
    samples_2 = sample_gaussian(covariance, means, N=5, random_state=rng)

    rng = np.random.default_rng(0)
    samples_3 = sample_gaussian(covariance, means, N=5, random_state=rng)

    assert np.allclose(samples_1, samples_3)  # Check that random seeds return same values
    assert not np.allclose(samples_1, samples_2)  # Check that random samples are different

    samples_4 = sample_gaussian(covariance, means, N=int(1e5), random_state=rng)

    assert np.allclose(samples_4.mean(axis=1), means, atol=1e-2)


def test_sample_2D_points():
    test_size = 10
    rng = np.random.default_rng(0)

    minimum_distance = 0.75 / np.sqrt(test_size)
    sampled_points = sample_2D_points(test_size, minimum_distance, random_state=rng)

    for point_1 in sampled_points:
        for point_2 in sampled_points:
            assert (np.linalg.norm(point_1 - point_2) > minimum_distance) or (np.allclose(point_1, point_2))

    different_points = sample_2D_points(test_size, minimum_distance, random_state=rng)

    assert not np.allclose(different_points, sampled_points)

    rng = np.random.default_rng(0)
    resampled_points = sample_2D_points(test_size, minimum_distance, random_state=rng)

    assert np.allclose(resampled_points, sampled_points)


@pytest.fixture(scope="module")
def synthetic_dataset(parameters):

    synthetic_dataset = SyntheticDataset(
        replicate_name="test",
        parameters=parameters,
        random_state=0,
    )

    return synthetic_dataset


def test_synthesize_dataset(synthetic_dataset): ...


@pytest.fixture
def domain_assigned_dataset(synthetic_dataset):
    # Basically test domain assignment with domain landmarks exactly equal to layered points
    sampled_points = synthetic_dataset.obsm["spatial"]

    def get_layer_mask(points, lower_threshold, upper_threshold, dim=0):
        return (lower_threshold <= points[:, dim]) & (points[:, dim] < upper_threshold)

    layer_masks = [
        get_layer_mask(sampled_points, 0, 0.35),
        get_layer_mask(sampled_points, 0.35, 0.5),
        get_layer_mask(sampled_points, 0.5, 0.75),
        get_layer_mask(sampled_points, 0.75, 1),
    ]

    domains = {f"L{index}": sampled_points[layer_mask] for index, layer_mask in enumerate(layer_masks)}

    synthetic_dataset.domain_canvas.load_domains(domains)
    synthetic_dataset.assign_domain_labels()

    for domain_name, layer_mask in zip(domains, layer_masks):
        assert np.all(synthetic_dataset[layer_mask].obs["domain"] == domain_name)

    return synthetic_dataset


def test_domain_assignment(domain_assigned_dataset): ...


def test_synthesize_metagenes(domain_assigned_dataset, parameters):
    raw_metagenes = domain_assigned_dataset.synthesize_metagenes()

    dataset_path = Path("tests/test_data/simulation_dataset")
    if not (dataset_path / "raw_metagenes.npy").exists():
        dataset_path.mkdir(exist_ok=True)
        np.save(dataset_path / "raw_metagenes.npy", raw_metagenes)

    saved_metagenes = np.load(dataset_path / "raw_metagenes.npy")

    assert np.allclose(raw_metagenes, saved_metagenes)


def test_synthesize_embeddings(domain_assigned_dataset, parameters):
    raw_embeddings, raw_cell_types = domain_assigned_dataset.synthesize_cell_embeddings()

    dataset_path = Path("tests/test_data/simulation_dataset")
    dataset_path.mkdir(exist_ok=True)
    if not (dataset_path / "cell_embeddings.npy").exists():
        np.save(dataset_path / "cell_embeddings.npy", raw_embeddings)

    if not (dataset_path / "cell_types.npy").exists():
        np.save(dataset_path / "cell_types.npy", raw_cell_types)

    saved_embeddings = np.load(dataset_path / "cell_embeddings.npy")
    saved_cell_types = np.load(dataset_path / "cell_types.npy")

    assert np.allclose(raw_embeddings, saved_embeddings)
    assert np.allclose(raw_cell_types, saved_cell_types)


def test_synthesize_expression(domain_assigned_dataset, parameters):
    domain_assigned_dataset.simulate_metagene_based_expression()
