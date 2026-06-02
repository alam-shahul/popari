from dataclasses import is_dataclass

import numpy as np
from scipy.sparse import issparse

from popari.simulation import (
    LayerRecipe,
    ReplicateConfig,
    SpatialDropoutConfig,
    SyntheticDataConfig,
    calculate_grid_neighbors,
    create_multireplicate_dataset,
    create_spatial_affinity_demo_datasets,
    default_cortex_layer_recipe,
    four_neighbor_grid_adjacency,
    generate_simulation,
    load_or_assign_domains,
    make_disjoint_metagenes,
    simulate_expression_with_metagenes,
    spatial_affinity_demo_label_grid,
    spatial_affinity_demo_scenario_names,
    two_cell_type_ratio_recipes,
)


def _small_config(random_state=0):
    return SyntheticDataConfig(
        num_genes=20,
        grid_size=6,
        num_noise_metagenes=0,
        sig_y_scale=1.0,
        sig_x_scale=1.0,
        random_state=random_state,
    )


def test_notebook_configs_are_dataclasses():
    assert is_dataclass(LayerRecipe)
    assert is_dataclass(SyntheticDataConfig)
    assert is_dataclass(ReplicateConfig)
    assert is_dataclass(SpatialDropoutConfig)


def test_hierarchical_notebook_generation_path_populates_expected_fields():
    result = generate_simulation(
        recipe=default_cortex_layer_recipe(),
        config=_small_config(),
        replicates=ReplicateConfig(names=("layer_0",)),
        dropout=SpatialDropoutConfig(sparsity=0.05, random_state=0),
        calculate_neighbors=True,
    )
    (dataset,) = result.datasets

    assert result.replicate_names == ("layer_0",)
    assert dataset.obs_names.is_unique
    assert dataset.X.shape == (36, 20)
    assert issparse(dataset.X)
    assert dataset.obsm["spatial"].shape == (36, 2)
    assert dataset.obsm["ground_truth_X"].shape == (36, 9)
    assert dataset.uns["ground_truth_M"][dataset.name].shape == (20, 9)
    assert "layer" in dataset.obs
    assert dataset.obs["layer"].dtype.name == "category"
    assert "cell_type" in dataset.obs
    assert dataset.obs["cell_type"].dtype.name == "category"
    assert dataset.obs["batch"].dtype.name == "category"
    assert "adjacency_matrix" in dataset.obsp
    assert "adjacency_list" in dataset.obsm


def test_hierarchical_notebook_generation_is_deterministic_for_fixed_seed():
    kwargs = {
        "recipe": default_cortex_layer_recipe(),
        "config": _small_config(random_state=3),
        "replicates": ReplicateConfig(names=("layer_0",)),
        "calculate_neighbors": False,
    }

    result_0 = generate_simulation(**kwargs)
    result_1 = generate_simulation(**kwargs)
    dataset_0 = result_0.datasets[0]
    dataset_1 = result_1.datasets[0]

    assert np.allclose(dataset_0.X.toarray(), dataset_1.X.toarray())
    assert np.allclose(dataset_0.obsm["ground_truth_X"], dataset_1.obsm["ground_truth_X"])
    assert np.allclose(dataset_0.uns["ground_truth_M"][dataset_0.name], dataset_1.uns["ground_truth_M"][dataset_1.name])


def test_minimal_notebook_recipe_has_one_dataset_per_ratio():
    recipes = two_cell_type_ratio_recipes()

    assert set(recipes) == {"A_to_B_1_to_9", "A_to_B_1_to_1", "A_to_B_9_to_1"}
    for ratio_name, recipe in recipes.items():
        assert tuple(recipe.layer_distributions) == (ratio_name,)
        assert set(recipe.cell_type_definitions) == {"Type A", "Type B"}


def test_disjoint_metagenes_have_nonoverlapping_gene_support():
    metagenes, magnitudes = make_disjoint_metagenes(num_genes=12, num_metagenes=2)

    assert metagenes.shape == (12, 2)
    assert np.all(metagenes[:6, 0] > 0)
    assert np.all(metagenes[:6, 1] == 0)
    assert np.all(metagenes[6:, 0] == 0)
    assert np.all(metagenes[6:, 1] > 0)
    assert np.allclose(metagenes.sum(axis=0), 1)
    assert np.allclose(magnitudes, 1)


def test_minimal_notebook_generation_keeps_exact_metagenes_and_support_restricted_noise():
    config = _small_config()
    metagenes, magnitudes = make_disjoint_metagenes(config.num_genes, num_metagenes=2)
    counts_by_ratio = {}

    for ratio_name, recipe in two_cell_type_ratio_recipes().items():
        multireplicate = create_multireplicate_dataset(recipe, config, ReplicateConfig(names=(ratio_name,)))
        load_or_assign_domains(multireplicate, recipe)
        simulate_expression_with_metagenes(
            multireplicate,
            metagenes,
            magnitudes,
            noiseless=False,
            exact_cell_type_embeddings=True,
            noise_on_expressed_genes_only=True,
        )
        calculate_grid_neighbors(multireplicate)
        dataset = next(iter(multireplicate))
        ground_truth_X = dataset.obsm["ground_truth_X"]
        clean_expression = ground_truth_X @ dataset.uns["ground_truth_M"][dataset.name].T

        counts_by_ratio[ratio_name] = dataset.obs["cell_type"].value_counts().to_dict()
        assert set(dataset.obs["region"]) == {ratio_name}
        assert np.allclose(dataset.uns["ground_truth_M"][dataset.name], metagenes)
        assert np.allclose(ground_truth_X[dataset.obs["cell_type"] == "Type A"], [1, 0])
        assert np.allclose(ground_truth_X[dataset.obs["cell_type"] == "Type B"], [0, 1])
        assert np.all(dataset.X[dataset.obs["cell_type"] == "Type A", config.num_genes // 2 :] == 0)
        assert np.all(dataset.X[dataset.obs["cell_type"] == "Type B", : config.num_genes // 2] == 0)
        assert not np.allclose(dataset.X[clean_expression > 0], clean_expression[clean_expression > 0])

    for ratio_name, recipe in two_cell_type_ratio_recipes().items():
        proportions = list(recipe.layer_distributions[ratio_name].values())
        labels = list(recipe.layer_distributions[ratio_name])
        partition_indices = (np.cumsum(proportions) * config.grid_size**2).astype(int)
        expected_counts = dict(zip(labels, np.diff(np.concatenate([[0], partition_indices]))))
        assert counts_by_ratio[ratio_name] == expected_counts


def test_spatial_affinity_demo_label_grids_match_requested_patterns():
    alternating = spatial_affinity_demo_label_grid("alternating_columns", grid_size=6)
    checker = spatial_affinity_demo_label_grid("checker_diagonals", grid_size=6)
    only_a = spatial_affinity_demo_label_grid("only_type_a", grid_size=6)
    thirds = spatial_affinity_demo_label_grid("vertical_thirds", grid_size=6)

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


def test_spatial_affinity_demo_datasets_have_three_disjoint_metagenes_and_grid_graphs():
    config = _small_config()
    datasets = create_spatial_affinity_demo_datasets(config)

    assert tuple(dataset.name for dataset in datasets) == spatial_affinity_demo_scenario_names()
    for dataset in datasets:
        ground_truth_M = dataset.uns["ground_truth_M"][dataset.name]
        ground_truth_X = dataset.obsm["ground_truth_X"]
        clean_expression = ground_truth_X @ ground_truth_M.T
        metagene_support = np.where(ground_truth_M > 0, 1, 0)

        assert dataset.X.shape == (36, 20)
        assert dataset.obs_names.is_unique
        assert issparse(dataset.X)
        assert dataset.obs["scenario"].dtype.name == "category"
        assert dataset.obs["cell_type"].dtype.name == "category"
        assert dataset.obs["batch"].dtype.name == "category"
        assert ground_truth_M.shape == (20, 3)
        assert ground_truth_X.shape == (36, 3)
        assert np.all(metagene_support.sum(axis=1) == 1)
        assert np.allclose(ground_truth_M.sum(axis=0), 1)
        assert np.all(np.count_nonzero(ground_truth_X, axis=1) == 1)
        assert np.all(dataset.X.toarray()[clean_expression == 0] == 0)
        assert "adjacency_matrix" in dataset.obsp
        assert "adjacency_list" in dataset.obsm
