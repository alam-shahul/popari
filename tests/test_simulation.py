import pytest
import numpy as np
from scipy.stats import gamma

from popari.simulation_framework import sample_gaussian, sample_2D_points, SyntheticDataset, synthesize_metagenes, synthesize_cell_embeddings, sample_normalized_embeddings

@pytest.fixture(scope="module")
def hyperparameter_setup():
    cell_type_definitions = {
        "Excitatory L1":         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Excitatory L2":         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Excitatory L3":         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Excitatory L4":         [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "Inhibitory 1":          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "Inhibitory 2":          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "Non-Neuron L1":         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Non-Neuron Ubiquitous": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    layer_distributions = {
        "L1": {
            "Excitatory L1": 0.33,
            "Inhibitory 1": 0.1,
            "Non-Neuron L1": 0.5,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L2": {
            "Excitatory L2": 0.93,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L3": {
            "Excitatory L3": 0.53,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.3,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L4": {
            "Excitatory L4": 0.73,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.1,
            "Non-Neuron Ubiquitous": 0.07
        },
    }
    
    shifted_distributions = {
        "L1": {
            "Excitatory L1": 0.83,
            "Inhibitory 1": 0.1,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L2": {
            "Excitatory L2": 0.43,
            "Non-Neuron L1": 0.5,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L3": {
            "Excitatory L3": 0.53,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.3,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L4": {
            "Excitatory L4": 0.73,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.1,
            "Non-Neuron Ubiquitous": 0.07
        },
    }
    
    simulation_parameters = {
        'num_real_metagenes': 10,
        'num_noise_metagenes': 2,
        'sigY_scale': 2.0,
        'sigX_scale': 2.0,
        "real_metagene_parameter": 4.0,
        "noise_metagene_parameter": 8.0,
        'lambda_s': 1.0,
    }
    
    metagene_variation_probabilities = [0, 0.25, 0, 0.25, 0, 0.25, 0, 0.1, 0.1, 0.1]

    np.random.seed(0)


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

    assert np.allclose(samples_1, samples_3) # Check that random seeds return same values
    assert not np.allclose(samples_1, samples_2) # Check that random samples are different
    
    samples_4 = sample_gaussian(covariance, means, N=int(1e5), random_state=rng)

    assert np.allclose(samples_4.mean(axis=1), means, atol=1e-2)

def test_sample_2D_points():
    test_size = 10
    rng = np.random.default_rng(0)

    minimum_distance = 0.75 / np.sqrt(test_size)
    sampled_points = sample_2D_points(test_size, minimum_distance, random_state=rng)

    for point_1 in sampled_points:
        for point_2 in sampled_points:
            assert (np.linalg.norm(point_1 - point_2) > minimum_distance) or \
                (np.allclose(point_1, point_2))

def test_synthesize_dataset():
    num_genes = G = 500
    n_noise_metagenes = 7
    num_real_metagenes = 13
    real_metagene_parameter = 4.0
    noise_metagene_parameter = 4.0
    metagene_variation_probabilities = [0, 0.25, 0, 0.25, 0.25, 0, 0.25, 0, 0.25, 0, 0.1, 0.1, 0.1]

    cell_type_definitions = {
        "Excitatory L1":         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Excitatory L2":         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Excitatory L3":         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Excitatory L4":         [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "Inhibitory 1":          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "Inhibitory 2":          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "Non-Neuron L1":         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Non-Neuron Ubiquitous": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    layer_distributions = {
        "L1": {
            "Excitatory L1": 0.33,
            "Inhibitory 1": 0.1,
            "Non-Neuron L1": 0.5,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L2": {
            "Excitatory L2": 0.93,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L3": {
            "Excitatory L3": 0.53,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.3,
            "Non-Neuron Ubiquitous": 0.07
        },
        "L4": {
            "Excitatory L4": 0.73,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.1,
            "Non-Neuron Ubiquitous": 0.07
        },
    }
    metagene_variation_probabilities = [0, 0.25, 0, 0.25, 0.25, 0, 0.25, 0, 0.25, 0, 0.1, 0.1, 0.1]
    
    test_size = N = 500
    sigma_x_scale = 3.0
    sigma_y_scale = 3.0

    simulation_parameters = {
        'num_real_metagenes': 10,
        'num_noise_metagenes': 3,
        'sigY_scale': sigma_y_scale,
        'sigX_scale': sigma_x_scale,
        "real_metagene_parameter": 4.0,
        "noise_metagene_parameter": noise_metagene_parameter,
        'lambda_s': 1.0,
    }

    synthesized_dataset = SyntheticDataset(test_size, num_genes, "test", spatial_distributions=layer_distributions, cell_type_definitions=cell_type_definitions, metagene_variation_probabilities=metagene_variation_probabilities)

    sampled_points = synthesized_dataset.obsm["spatial"]
    domains = {}
    domains["L1"] = sampled_points[sampled_points[:, 0] < 0.35]
    domains["L2"] = sampled_points[(0.35 <= sampled_points[:, 0]) & (sampled_points[:, 0] < 0.5)]
    domains["L3"] = sampled_points[(0.5 <= sampled_points[:, 0]) & (sampled_points[:, 0] < 0.75)]
    domains["L4"] = sampled_points[(0.75 <= sampled_points[:, 0]) & (sampled_points[:, 0]< 1)]
    
    synthesized_dataset.domain_canvas.load_domains(domains)
    synthesized_dataset.assign_layer_labels()

    synthesized_dataset.simulate_metagene_based_expression(**simulation_parameters)
