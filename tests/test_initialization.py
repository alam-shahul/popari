import numpy as np
import pytest
from scipy.sparse import issparse

from popari.model import Popari
from popari.simulation import SyntheticDataConfig, create_spatial_affinity_demo_datasets


@pytest.mark.baseline
def test_random_state_controls_initialization(shared_model_factory):
    model_0 = shared_model_factory(random_state=0, initialization_method="dummy")
    model_1 = shared_model_factory(random_state=0, initialization_method="dummy")
    model_2 = shared_model_factory(random_state=1, initialization_method="dummy")

    for dataset_0, dataset_1, dataset_2 in zip(model_0.datasets, model_1.datasets, model_2.datasets):
        assert np.allclose(dataset_0.uns["M"][dataset_0.popari.name], dataset_1.uns["M"][dataset_1.popari.name])
        assert np.allclose(dataset_0.obsm["X"], dataset_1.obsm["X"])
        assert np.allclose(
            dataset_0.uns["Sigma_x_inv"][dataset_0.popari.name],
            dataset_1.uns["Sigma_x_inv"][dataset_1.popari.name],
        )

        assert not np.allclose(dataset_0.uns["M"][dataset_0.popari.name], dataset_2.uns["M"][dataset_2.popari.name])
        assert not np.allclose(dataset_0.obsm["X"], dataset_2.obsm["X"])


@pytest.mark.baseline
def test_ground_truth_initialization_uses_cell_type_labels(shared_model_factory):
    model = shared_model_factory(initialization_method="ground_truth")

    for dataset in model.datasets:
        label_indices = dataset.obs["cell_type"].str.removeprefix("type_").astype(int).to_numpy()
        assert np.array_equal(dataset.obsm["X"].argmax(axis=1), label_indices)
        active_values = dataset.obsm["X"][np.arange(dataset.n_obs), label_indices]
        inactive_values = dataset.obsm["X"].copy()
        inactive_values[np.arange(dataset.n_obs), label_indices] = -np.inf
        assert np.all(active_values > inactive_values.max(axis=1))


@pytest.mark.baseline
def test_ground_truth_initialization_requires_k_to_match_labels(shared_model_factory):
    with pytest.raises(ValueError, match="found 3 labels"):
        shared_model_factory(initialization_method="ground_truth", K=2)


@pytest.mark.baseline
def test_ground_truth_initialization_handles_absent_classes_with_random_vectors(context):
    config = SyntheticDataConfig(num_genes=12, grid_size=4, sig_y_scale=0.5, random_state=0)
    (dataset,) = create_spatial_affinity_demo_datasets(config, scenario_names=("Monotype",))
    assert issparse(dataset.X)

    model = Popari(
        K=3,
        datasets=(dataset,),
        replicate_names=(dataset.popari.name,),
        lambda_Sigma_x_inv=1e-4,
        initialization_method="ground_truth",
        torch_context=context,
        initial_context=context,
        random_state=0,
        verbose=0,
    )

    initialized_dataset = model.datasets[0]
    assert np.all(initialized_dataset.obsm["X"].argmax(axis=1) == 0)
    assert np.all(np.isfinite(initialized_dataset.uns["M"][initialized_dataset.popari.name]))
    assert np.all(np.isfinite(initialized_dataset.uns["Sigma_x_inv"][initialized_dataset.popari.name]))


@pytest.mark.baseline
def test_shared_mode_reuses_group_parameters(shared_model_factory):
    model = shared_model_factory()
    first_name, second_name = (dataset.popari.name for dataset in model.datasets)

    assert model.parameter_optimizer.metagene_state[first_name].data_ptr() == (
        model.parameter_optimizer.metagene_state[second_name].data_ptr()
    )
    assert model.parameter_optimizer.spatial_affinity[first_name].data_ptr() == (
        model.parameter_optimizer.spatial_affinity[second_name].data_ptr()
    )


@pytest.mark.expensive
def test_differential_initialization_creates_group_averages(differential_model_factory):
    model = differential_model_factory()

    assert model.metagene_mode == "differential"
    assert model.spatial_affinity_mode == "differential lookup"
    assert model.parameter_optimizer.metagene_state.M_bar
    assert model.parameter_optimizer.spatial_affinity_bar.spatial_affinity_bar

    for group_name, group_replicates in model.metagene_groups.items():
        averaged = sum(
            model.parameter_optimizer.metagene_state[dataset_name].detach().cpu().numpy()
            for dataset_name in group_replicates
        ) / len(group_replicates)
        assert np.allclose(
            averaged,
            model.parameter_optimizer.metagene_state.M_bar[group_name].detach().cpu().numpy(),
            atol=1e-4,
        )


@pytest.mark.expensive
def test_hierarchical_initialization_builds_resolution_stack(hierarchical_model_factory):
    model = hierarchical_model_factory(hierarchical_levels=3, binning_downsample_rate=0.4)

    assert model.hierarchical_levels == 3
    assert len(model.hierarchy.view_container) == 3
    assert model.base_view.level == 2

    for level in range(model.hierarchical_levels):
        view = model.hierarchy[level]
        assert view.level == level
        assert len(view.datasets) == len(model.replicate_names)
        assert all("X" in dataset.obsm for dataset in view.datasets)
