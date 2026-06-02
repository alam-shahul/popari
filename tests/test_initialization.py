import numpy as np
import pytest


@pytest.mark.baseline
def test_random_state_controls_initialization(shared_model_factory):
    model_0 = shared_model_factory(random_state=0, initialization_method="dummy")
    model_1 = shared_model_factory(random_state=0, initialization_method="dummy")
    model_2 = shared_model_factory(random_state=1, initialization_method="dummy")

    for dataset_0, dataset_1, dataset_2 in zip(model_0.datasets, model_1.datasets, model_2.datasets):
        assert np.allclose(dataset_0.uns["M"][dataset_0.name], dataset_1.uns["M"][dataset_1.name])
        assert np.allclose(dataset_0.obsm["X"], dataset_1.obsm["X"])
        assert np.allclose(
            dataset_0.uns["Sigma_x_inv"][dataset_0.name],
            dataset_1.uns["Sigma_x_inv"][dataset_1.name],
        )

        assert not np.allclose(dataset_0.uns["M"][dataset_0.name], dataset_2.uns["M"][dataset_2.name])
        assert not np.allclose(dataset_0.obsm["X"], dataset_2.obsm["X"])


@pytest.mark.baseline
def test_shared_mode_reuses_group_parameters(shared_model_factory):
    model = shared_model_factory()
    first_name, second_name = (dataset.name for dataset in model.datasets)

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
