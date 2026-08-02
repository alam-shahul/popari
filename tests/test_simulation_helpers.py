from dataclasses import is_dataclass

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import issparse

from popari.simulation.recipes import (
    SimulationConfig,
    SimulationRecipe,
    SpatialDropoutConfig,
    default_cortex_layer_recipe,
)
from popari.simulation.synthetic import (
    SPATIAL_AFFINITY_DEMO_SCENARIOS,
    _assign_domains,
    _create_dataset,
    _grid_coordinates,
    create_spatial_affinity_demo_datasets,
    four_neighbor_grid_adjacency,
    generate_simulation,
    make_disjoint_metagenes,
    spatial_affinity_demo_label_grid,
)


def _small_config(random_state=0):
    return SimulationConfig(
        num_genes=20,
        grid_size=6,
        num_noise_metagenes=0,
        sig_y_scale=1.0,
        sig_x_scale=1.0,
        random_state=random_state,
    )


def test_simulation_configs_are_dataclasses():
    assert is_dataclass(SimulationRecipe)
    assert is_dataclass(SimulationConfig)
    assert is_dataclass(SpatialDropoutConfig)


def test_grid_coordinates_respect_recipe_dimensions():
    recipe = SimulationRecipe(
        cell_type_definitions={"cell": [1]},
        spatial_distributions={"domain": {"cell": 1}},
        metagene_variation_probabilities=[0],
        width=2,
        height=3,
    )

    coordinates = _grid_coordinates(recipe, grid_size=3)

    np.testing.assert_allclose(
        coordinates,
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1.5],
            [1, 1.5],
            [2, 1.5],
            [0, 3],
            [1, 3],
            [2, 3],
        ],
    )


def test_assign_domains_uses_nearest_landmarks_and_preserves_metadata():
    recipe = SimulationRecipe(
        cell_type_definitions={"cell": [1]},
        spatial_distributions={"left": {"cell": 1}, "right": {"cell": 1}},
        metagene_variation_probabilities=[0],
        width=4,
        height=4,
    )
    dataset = _create_dataset("sample", recipe, SimulationConfig(num_genes=2, grid_size=3))
    landmarks = {
        "left": np.array([[0.0, 0.0], [0.0, 4.0]]),
        "right": np.array([[3.0, 0.0], [3.0, 4.0]]),
    }

    _assign_domains(dataset, recipe, landmarks)

    assert dataset.obs[recipe.domain_key].tolist() == [
        "left",
        "right",
        "right",
        "left",
        "right",
        "right",
        "left",
        "right",
        "right",
    ]
    for domain_name, coordinates in landmarks.items():
        np.testing.assert_array_equal(dataset.uns["domain_landmarks"][domain_name], coordinates)


def test_generation_returns_plain_anndata_with_expected_schema():
    result = generate_simulation(
        recipes={"layer": default_cortex_layer_recipe()},
        config=_small_config(),
        replicates={"layer_0": "layer"},
        dropout=SpatialDropoutConfig(sparsity=0.05, random_state=0),
    )
    dataset = result.adata

    assert type(dataset) is ad.AnnData
    assert dataset.popari.sample_names == ("layer_0",)
    assert dataset.obs_names.is_unique
    assert dataset.X.shape == (36, 20)
    assert issparse(dataset.X)
    assert dataset.obsm["spatial"].shape == (36, 2)
    assert dataset.simulation.ground_truth_X.shape == (36, 9)
    assert dataset.simulation.ground_truth_M.shape == (20, 9)
    assert dataset.obs["layer"].dtype.name == "category"
    assert dataset.obs["cell_type"].dtype.name == "category"
    assert dataset.obs["batch"].dtype.name == "category"
    assert "adjacency_matrix" in dataset.obsp
    assert "adjacency_list" in dataset.obsm


def test_generation_is_deterministic_and_shares_metagenes():
    kwargs = {
        "recipes": {"layer": default_cortex_layer_recipe()},
        "config": _small_config(random_state=3),
        "replicates": {"layer_0": "layer", "layer_1": "layer"},
        "calculate_neighbors": False,
    }
    first = generate_simulation(**kwargs)
    second = generate_simulation(**kwargs)

    np.testing.assert_allclose(first.adata.X.toarray(), second.adata.X.toarray())
    np.testing.assert_allclose(first.adata.simulation.ground_truth_X, second.adata.simulation.ground_truth_X)
    np.testing.assert_allclose(first.adata.simulation.ground_truth_M, second.adata.simulation.ground_truth_M)


def test_spatial_dropout_requires_neighbor_calculation():
    with pytest.raises(ValueError, match="Spatial dropout requires calculate_neighbors=True"):
        generate_simulation(
            recipes={"layer": default_cortex_layer_recipe()},
            config=_small_config(),
            replicates={"layer_0": "layer"},
            dropout=SpatialDropoutConfig(sparsity=0.05, random_state=0),
            calculate_neighbors=False,
        )


def test_disjoint_metagenes_have_nonoverlapping_gene_support():
    metagenes, magnitudes = make_disjoint_metagenes(num_genes=12, num_metagenes=2)

    assert metagenes.shape == (12, 2)
    assert np.all(metagenes[:6, 0] > 0)
    assert np.all(metagenes[:6, 1] == 0)
    assert np.all(metagenes[6:, 0] == 0)
    assert np.all(metagenes[6:, 1] > 0)
    assert np.allclose(metagenes.sum(axis=0), 1)
    assert np.allclose(magnitudes, 1)


def test_spatial_affinity_demo_label_grids_match_requested_patterns():
    alternating, checker, only_a, thirds = (
        spatial_affinity_demo_label_grid(scenario_name, grid_size=6)
        for scenario_name in SPATIAL_AFFINITY_DEMO_SCENARIOS
    )

    assert np.all(alternating[:, 0::2] == "Type A")
    assert np.all(alternating[:, 1::2] == "Type B")
    assert np.all(checker[:-1, :] != checker[1:, :])
    assert np.all(checker[:, :-1] != checker[:, 1:])
    assert set(only_a.ravel()) == {"Type A"}
    assert np.all(thirds[:, :2] == "Type A")
    assert np.all(thirds[:, 2:4] == "Type B")
    assert np.all(thirds[:, 4:] == "Type C")


def test_four_neighbor_grid_adjacency_has_expected_degrees():
    adjacency = four_neighbor_grid_adjacency(grid_size=4)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()

    assert set(degrees[[0, 3, 12, 15]]) == {2}
    assert set(degrees[[1, 2, 4, 7, 8, 11, 13, 14]]) == {3}
    assert set(degrees[[5, 6, 9, 10]]) == {4}


def test_spatial_affinity_demo_datasets_use_simulation_schema():
    datasets = create_spatial_affinity_demo_datasets(_small_config())

    for dataset in datasets:
        clean_expression = dataset.simulation.ground_truth_expression
        metagene_support = np.where(dataset.simulation.ground_truth_M > 0, 1, 0)
        assert type(dataset) is ad.AnnData
        assert dataset.X.shape == (36, 20)
        assert issparse(dataset.X)
        assert dataset.simulation.ground_truth_M.shape == (20, 3)
        assert dataset.simulation.ground_truth_X.shape == (36, 3)
        assert np.all(metagene_support.sum(axis=1) == 1)
        assert np.all(np.count_nonzero(dataset.simulation.ground_truth_X, axis=1) == 1)
        assert np.all(dataset.X.toarray()[clean_expression == 0] == 0)
        assert "adjacency_matrix" in dataset.obsp
        assert "adjacency_list" in dataset.obsm
