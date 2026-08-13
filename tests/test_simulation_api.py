import numpy as np

from popari.simulation.recipes import (
    SimulationConfig,
    SpatialDropoutConfig,
    alternating_replicates,
    default_differential_cortex_recipes,
    default_hierarchical_cortex_recipes,
    default_joint_improvement_recipes,
    named_replicates,
    paired_replicates,
    simulation_sweep_output_path,
)
from popari.simulation.synthetic import generate_simulation


def test_hierarchical_cortex_preset_matches_expected_shape():
    recipes = default_hierarchical_cortex_recipes()
    recipe = recipes["layer"]

    assert tuple(recipes) == ("layer",)
    assert recipe.num_real_metagenes == 9
    assert recipe.cell_type_definitions["Excitatory L1"] == [0.5, 0, 0, 0, 0, 1, 0, 0, 0]
    assert tuple(recipe.spatial_distributions) == ("L1", "L2", "L3", "L4")


def test_differential_cortex_preset_matches_expected_shape():
    recipes = default_differential_cortex_recipes()

    assert tuple(recipes) == ("progenitor", "layer")
    assert recipes["progenitor"].num_real_metagenes == 10
    assert recipes["layer"].num_real_metagenes == 10
    assert tuple(recipes["progenitor"].spatial_distributions) == ("progenitor_L1_2", "progenitor_L3_4")
    assert tuple(recipes["layer"].spatial_distributions) == ("L1", "L2", "L3", "L4")


def test_joint_improvement_preset_matches_expected_shape():
    recipes = default_joint_improvement_recipes()

    assert tuple(recipes) == ("progenitor", "layer")
    assert recipes["progenitor"].num_real_metagenes == 11
    assert recipes["layer"].num_real_metagenes == 11
    assert recipes["progenitor"].domain_key == "domain"
    assert recipes["layer"].domain_key == "domain"
    assert tuple(recipes["progenitor"].spatial_distributions) == ("L1", "L2", "L3", "L4")
    assert tuple(recipes["layer"].spatial_distributions) == ("L1", "L2", "L3", "L4")


def test_replicate_mappings_are_ordered_and_named():
    assert named_replicates("layer", count=2) == {
        "layer_0": "layer",
        "layer_1": "layer",
    }
    assert alternating_replicates(("progenitor", "layer"), count=4) == {
        "progenitor_0": "progenitor",
        "layer_1": "layer",
        "progenitor_2": "progenitor",
        "layer_3": "layer",
    }
    assert paired_replicates(("progenitor", "layer"), count=2) == {
        "progenitor_0": "progenitor",
        "layer_0": "layer",
        "progenitor_1": "progenitor",
        "layer_1": "layer",
    }


def test_generate_simulation_supports_two_recipe_tiny_grid():
    config = SimulationConfig(
        num_genes=8,
        grid_size=4,
        num_noise_metagenes=0,
        sig_y_scale=0.5,
        sig_x_scale=0.5,
        real_metagene_parameter=4.0,
        random_state=0,
    )
    result = generate_simulation(
        config=config,
        recipes=default_differential_cortex_recipes(),
        replicates=alternating_replicates(("progenitor", "layer"), count=2),
        dropout=SpatialDropoutConfig(sparsity=0.1, random_state=0),
    )

    assert result.adata.popari.sample_names == (
        "progenitor_0",
        "layer_1",
    )
    for sample in result.adata.popari.sample_names:
        dataset = result.adata[result.adata.popari.sample_indices(sample)]
        assert dataset.shape == (16, 8)
        assert dataset.obs_names.is_unique
        assert "cell_type" in dataset.obs
        assert "layer" in dataset.obs
        assert "batch" in dataset.obs
        assert "ground_truth_X" in dataset.obsm
        assert "ground_truth_M" in dataset.uns
        assert "adjacency_matrix" in dataset.obsp
        assert np.isfinite(dataset.X.toarray()).all()


def test_generate_joint_improvement_supports_paired_replicates():
    config = SimulationConfig(
        num_genes=11,
        grid_size=4,
        num_noise_metagenes=0,
        sig_y_scale=0.5,
        sig_x_scale=0.5,
        real_metagene_parameter=4.0,
        random_state=0,
    )
    result = generate_simulation(
        config=config,
        recipes=default_joint_improvement_recipes(),
        replicates=paired_replicates(("progenitor", "layer"), count=1),
    )

    assert result.adata.popari.sample_names == (
        "progenitor_0",
        "layer_0",
    )
    for sample in result.adata.popari.sample_names:
        dataset = result.adata[result.adata.popari.sample_indices(sample)]
        assert dataset.shape == (16, 11)
        assert "domain" in dataset.obs
        assert "ground_truth_X" in dataset.obsm
        assert "ground_truth_M" in dataset.uns
        assert "adjacency_matrix" in dataset.obsp
        assert np.isfinite(dataset.X.toarray()).all()


def test_simulation_sweep_output_path_preserves_old_layout(tmp_path):
    config = SimulationConfig(
        num_genes=100,
        grid_size=15,
        sig_y_scale=2.0,
        sig_x_scale=3.0,
        noise_metagene_parameter=4.0,
        random_state=7,
    )

    assert simulation_sweep_output_path(tmp_path, config, num_replicates=5) == (
        tmp_path
        / "num_replicates_5"
        / "synthetic_225_100_sigY-scale-2.0_sigX-scale-3.0_noise-metagene-parameter_4.0"
        / "random_state_7"
        / "processed_dataset.h5ad"
    )
