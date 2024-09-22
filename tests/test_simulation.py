from pathlib import Path

import numpy as np
import pytest
import scanpy as sc
from scipy.stats import gamma

from popari.simulation_framework import (
    MultiReplicateSyntheticDataset,
    SimulationParameters,
    SyntheticDataset,
    sample_2D_points,
    sample_gaussian,
    sample_normalized_embeddings,
)


def walk_dict(d, depth=0):
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        if isinstance(v, dict):
            print("  " * depth + (f"{k}"))
            walk_dict(v, depth + 1)
        else:
            print("  " * depth + f"{k} {type(k)}[{type(v)}]: {v}")


@pytest.fixture(scope="module")
def test_datapath():
    return Path("tests/test_data/simulation_dataset")


@pytest.fixture(scope="module")
def original_parameters() -> SimulationParameters:
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
                "Excitatory L1": 0.53,
                "Inhibitory L1": 0.2,
                "Inhibitory L2": 0.2,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L2": {
                "Excitatory L2": 0.53,
                "Inhibitory L2": 0.2,
                "Inhibitory L3": 0.2,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L3": {
                "Excitatory L3": 0.53,
                "Inhibitory L3": 0.2,
                "Inhibitory L4": 0.2,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L4": {
                "Excitatory L4": 0.53,
                "Inhibitory L3": 0.2,
                "Inhibitory L4": 0.2,
                "Non-Neuron Ubiquitous": 0.07,
            },
        },
    }

    simulation_parameters = SimulationParameters(**parameters)

    return simulation_parameters


@pytest.fixture(scope="module")
def shifted_parameters() -> SimulationParameters:

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
                "Excitatory L1": 0.2,
                "Excitatory L2": 0.2,
                "Inhibitory L1": 0.53,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L2": {
                "Excitatory L2": 0.2,
                "Excitatory L3": 0.2,
                "Inhibitory L2": 0.53,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L3": {
                "Excitatory L3": 0.2,
                "Excitatory L4": 0.2,
                "Inhibitory L3": 0.53,
                "Non-Neuron Ubiquitous": 0.07,
            },
            "L4": {
                "Excitatory L3": 0.2,
                "Excitatory L4": 0.2,
                "Inhibitory L4": 0.53,
                "Non-Neuron Ubiquitous": 0.07,
            },
        },
    }

    simulation_parameters = SimulationParameters(**parameters)

    return simulation_parameters


def test_sample_gaussian(test_datapath):
    test_size = 10

    rng = np.random.default_rng(0)
    covariance = rng.random(size=test_size) * np.eye(test_size)
    means = rng.random(size=test_size)

    rng = np.random.default_rng(0)
    samples_1 = sample_gaussian(covariance, means, N=5, random_state=rng)
    samples_2 = sample_gaussian(covariance, means, N=5, random_state=rng)

    if not (test_datapath / "gaussian_samples.npy").exists():
        np.save(test_datapath / "gaussian_samples.npy", samples_1)

    saved_gaussian_samples = np.load(test_datapath / "gaussian_samples.npy")

    assert np.allclose(samples_1, saved_gaussian_samples)

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
def synthetic_dataset(original_parameters):
    synthetic_dataset = SyntheticDataset(
        replicate_name="test",
        parameters=original_parameters,
        random_state=0,
        verbose=1,
    )

    return synthetic_dataset


def test_synthesize_dataset(synthetic_dataset): ...


@pytest.fixture(scope="module")
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


def test_synthesize_metagenes(test_datapath, domain_assigned_dataset):
    raw_metagenes = domain_assigned_dataset.synthesize_metagenes()

    if not (test_datapath / "raw_metagenes.npy").exists():
        test_datapath.mkdir(exist_ok=True)
        np.save(test_datapath / "raw_metagenes.npy", raw_metagenes)

    saved_metagenes = np.load(test_datapath / "raw_metagenes.npy")

    assert np.allclose(raw_metagenes, saved_metagenes)


def test_synthesize_embeddings(test_datapath, domain_assigned_dataset):
    raw_embeddings, raw_cell_types = domain_assigned_dataset.synthesize_cell_embeddings()

    test_datapath.mkdir(exist_ok=True)
    if not (test_datapath / "cell_embeddings.npy").exists():
        np.save(test_datapath / "cell_embeddings.npy", raw_embeddings)

    if not (test_datapath / "cell_types.npy").exists():
        np.save(test_datapath / "cell_types.npy", raw_cell_types)

    saved_embeddings = np.load(test_datapath / "cell_embeddings.npy")
    saved_cell_types = np.load(test_datapath / "cell_types.npy")

    assert np.allclose(raw_embeddings, saved_embeddings)
    assert np.allclose(raw_cell_types, saved_cell_types)


def test_synthesize_expression(test_datapath, domain_assigned_dataset):
    domain_assigned_dataset.simulate_metagene_based_expression()

    if not (test_datapath / "simulated_dataset.h5ad").exists():
        walk_dict(domain_assigned_dataset.uns)
        domain_assigned_dataset.write_h5ad(test_datapath / "simulated_dataset.h5ad")

    saved_dataset = sc.read_h5ad(test_datapath / "simulated_dataset.h5ad")

    assert np.allclose(domain_assigned_dataset.X, saved_dataset.X)


@pytest.fixture(scope="module")
def multireplicate_dataset(original_parameters, shifted_parameters):
    replicate_parameters = {
        "original": original_parameters,
        "shifted": shifted_parameters,
    }

    multireplicate_synthetic_dataset = MultiReplicateSyntheticDataset(
        replicate_parameters=replicate_parameters,
        dataset_constructor=SyntheticDataset,
        random_state=0,
        verbose=1,
    )

    return multireplicate_synthetic_dataset


def test_multireplicate_dataset(multireplicate_dataset):
    for dataset, expected_name in zip(multireplicate_dataset, ("original", "shifted")):
        assert dataset.name == expected_name


@pytest.fixture(scope="module")
def domain_assigned_multireplicate_dataset(multireplicate_dataset):
    # Basically test domain assignment with domain landmarks exactly equal to layered points
    def get_layer_mask(points, lower_threshold, upper_threshold, dim=0):
        return (lower_threshold <= points[:, dim]) & (points[:, dim] < upper_threshold)

    for synthetic_dataset in multireplicate_dataset:
        sampled_points = synthetic_dataset.obsm["spatial"]

        layer_masks = [
            get_layer_mask(sampled_points, 0, 0.35),
            get_layer_mask(sampled_points, 0.35, 0.5),
            get_layer_mask(sampled_points, 0.5, 0.75),
            get_layer_mask(sampled_points, 0.75, 1),
        ]

        domains = {f"L{index}": sampled_points[layer_mask] for index, layer_mask in enumerate(layer_masks)}

        synthetic_dataset.domain_canvas.load_domains(domains)

    multireplicate_dataset.assign_domain_labels()
    return multireplicate_dataset


def test_synthesize_expression_multireplicate(test_datapath, domain_assigned_multireplicate_dataset):
    domain_assigned_multireplicate_dataset.simulate_expression()

    assert np.allclose(
        *[dataset.uns["ground_truth_M"][dataset.name] for dataset in domain_assigned_multireplicate_dataset],
    )
    assert not np.allclose(*[dataset.obsm["ground_truth_X"] for dataset in domain_assigned_multireplicate_dataset])
    for synthetic_dataset in domain_assigned_multireplicate_dataset:
        if not (test_datapath / f"simulated_dataset_replicate_{synthetic_dataset.name}.h5ad").exists():
            synthetic_dataset.write_h5ad(test_datapath / f"simulated_dataset_replicate_{synthetic_dataset.name}.h5ad")

        saved_dataset = sc.read_h5ad(test_datapath / f"simulated_dataset_replicate_{synthetic_dataset.name}.h5ad")

        assert np.allclose(synthetic_dataset.X, saved_dataset.X)
