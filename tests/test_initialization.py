import numpy as np
import pytest
from scipy.sparse import csr_array, issparse

from popari.model import Popari
from popari.schema import BIN_ASSIGNMENTS_KEY
from popari.simulation.recipes import SimulationConfig
from popari.simulation.synthetic import create_spatial_affinity_demo_datasets


@pytest.mark.baseline
def test_random_state_controls_initialization(shared_model_factory):
    model_0 = shared_model_factory(random_state=0, initialization_method="dummy")
    model_1 = shared_model_factory(random_state=0, initialization_method="dummy")
    model_2 = shared_model_factory(random_state=1, initialization_method="dummy")

    assert np.allclose(model_0.adata.obsm["X"], model_1.adata.obsm["X"])
    assert not np.allclose(model_0.adata.obsm["X"], model_2.adata.obsm["X"])
    assert np.allclose(model_0.adata.uns["M"], model_1.adata.uns["M"])
    assert not np.allclose(model_0.adata.uns["M"], model_2.adata.uns["M"])
    for sample in model_0.replicate_names:
        assert np.allclose(
            model_0.adata.uns["Sigma_x_inv"][sample],
            model_1.adata.uns["Sigma_x_inv"][sample],
        )


@pytest.mark.baseline
def test_ground_truth_initialization_uses_cell_type_labels(shared_model_factory):
    model = shared_model_factory(initialization_method="ground_truth")

    label_indices = model.adata.obs["cell_type"].str.removeprefix("type_").astype(int).to_numpy()
    assert np.array_equal(model.adata.obsm["X"].argmax(axis=1), label_indices)
    active_values = model.adata.obsm["X"][np.arange(model.adata.n_obs), label_indices]
    inactive_values = model.adata.obsm["X"].copy()
    inactive_values[np.arange(model.adata.n_obs), label_indices] = -np.inf
    assert np.all(active_values > inactive_values.max(axis=1))


@pytest.mark.baseline
def test_ground_truth_initialization_requires_k_to_match_labels(shared_model_factory):
    with pytest.raises(ValueError, match="found 3 labels"):
        shared_model_factory(initialization_method="ground_truth", K=2)


@pytest.mark.baseline
def test_ground_truth_initialization_handles_absent_classes_with_random_vectors(context):
    config = SimulationConfig(num_genes=12, grid_size=4, sig_y_scale=0.5, random_state=0)
    (dataset,) = create_spatial_affinity_demo_datasets(config, scenario_names=("Monotype",))
    assert issparse(dataset.X)

    model = Popari(
        K=3,
        adata=dataset,
        lambda_Sigma_x_inv=1e-4,
        initialization_method="ground_truth",
        torch_context=context,
        initial_context=context,
        random_state=0,
        verbose=0,
    )

    sample = model.replicate_names[0]
    assert np.all(model.adata.obsm["X"].argmax(axis=1) == 0)
    assert np.all(np.isfinite(model.adata.uns["M"]))
    assert np.all(np.isfinite(model.adata.uns["Sigma_x_inv"][sample]))


@pytest.mark.baseline
def test_model_has_one_shared_metagene_parameter(shared_model_factory):
    model = shared_model_factory()
    first_name, second_name = model.replicate_names

    assert model.parameter_optimizer.metagenes.shape == (model.adata.n_vars, model.K)
    assert model.parameter_optimizer.spatial_affinity[first_name].data_ptr() == (
        model.parameter_optimizer.spatial_affinity[second_name].data_ptr()
    )


@pytest.mark.expensive
def test_differential_affinity_initialization_creates_group_averages(differential_model_factory):
    model = differential_model_factory()

    assert model.spatial_affinity_mode == "differential lookup"
    assert model.parameter_optimizer.spatial_affinity_bar.spatial_affinity_bar


@pytest.mark.expensive
def test_hierarchical_initialization_builds_resolution_stack(hierarchical_model_factory):
    model = hierarchical_model_factory(hierarchical_levels=3, binning_downsample_rate=0.4)

    assert model.hierarchical_levels == 3
    assert len(model.hierarchy.view_container) == 3
    assert model.base_view.level == 2
    assert model.adata is model.hierarchy[0].adata

    for level in range(model.hierarchical_levels):
        view = model.hierarchy[level]
        assert view.level == level
        assert view.sample_axis.names == tuple(model.replicate_names)
        assert view.adata.obsm["X"].shape == (view.adata.n_obs, model.K)

        graph = csr_array(view.adata.obsp["adjacency_matrix"]).tocoo()
        assert np.all(view.sample_axis.codes[graph.row] == view.sample_axis.codes[graph.col])

        if level > 0:
            previous_view = model.hierarchy[level - 1]
            assignments = csr_array(view.adata.obsm[BIN_ASSIGNMENTS_KEY])
            assert assignments.shape == (view.adata.n_obs, previous_view.adata.n_obs)
            rows, columns = assignments.nonzero()
            assert np.all(
                view.sample_axis.codes[rows] == previous_view.sample_axis.codes[columns],
            )


@pytest.mark.expensive
def test_hierarchical_assignments_support_interleaved_samples(hierarchical_model_factory, adata_factory):
    adata = adata_factory(num_cells=36)
    first, second = adata.popari.sample_names
    order = np.column_stack(
        [adata.popari.sample_indices(first), adata.popari.sample_indices(second)],
    ).ravel()
    adata = adata[order].copy()

    model = hierarchical_model_factory(adata=adata, hierarchical_levels=2)
    coarse = model.hierarchy[1]
    assignments = csr_array(coarse.adata.obsm[BIN_ASSIGNMENTS_KEY]).tocoo()

    assert np.all(
        coarse.sample_axis.codes[assignments.row] == model.hierarchy[0].sample_axis.codes[assignments.col],
    )
