import anndata as ad
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.sparse import csr_matrix

from popari import pl, tl


def _close_figures(*figures):
    for figure in figures:
        if isinstance(figure, list):
            _close_figures(*figure)
            continue
        if isinstance(figure, Figure):
            plt.close(figure)


def test_affinity_difference_plot_returns_figure():
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.uns["Sigma_x_inv"] = {
        "dataset_1": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "dataset_2": np.array([[5.0, 7.0], [11.0, 13.0]]),
    }

    figure = pl.affinity_difference(dataset, "dataset_2", "dataset_1")

    try:
        assert isinstance(figure, Figure)
        assert figure.axes
        plotted_values = figure.axes[0].images[0].get_array()
        assert plotted_values[0, 0] == 4.0
        assert plotted_values[1, 0] == 8.0
        assert np.ma.is_masked(plotted_values[0, 1])
    finally:
        _close_figures(figure)


def test_matrix_heatmap_uses_dataframe_labels_and_centered_scale():
    matrix = pd.DataFrame(
        [[-1.0, 2.0], [3.0, -4.0]],
        index=["row_0", "row_1"],
        columns=["col_0", "col_1"],
    )

    figure = pl.matrix_heatmap(matrix, center_zero=True, colorbar=False)

    try:
        assert isinstance(figure, Figure)
        ax = figure.axes[0]
        image = ax.images[0]
        assert image.norm.vmin == pytest.approx(-4.0)
        assert image.norm.vmax == pytest.approx(4.0)
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["col_0", "col_1"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["row_0", "row_1"]
    finally:
        _close_figures(figure)


def test_matrix_heatmap_panel_uses_shared_scale_and_colorbar():
    matrices = {
        "first": pd.DataFrame([[1.0, 2.0], [3.0, 4.0]]),
        "second": pd.DataFrame([[-10.0, 0.0], [0.0, 5.0]]),
    }

    figure = pl.matrix_heatmap_panel(matrices, center_zero=True, shared_scale=True, colorbar="shared")

    try:
        assert isinstance(figure, Figure)
        image_axes = [ax for ax in figure.axes if ax.images]
        assert len(image_axes) == 2
        for ax in image_axes:
            image = ax.images[0]
            assert image.norm.vmin == pytest.approx(-10.0)
            assert image.norm.vmax == pytest.approx(10.0)
        assert len(figure.axes) == 3
    finally:
        _close_figures(figure)


def test_matrix_heatmap_panel_center_zero_without_shared_scale():
    matrices = {
        "first": pd.DataFrame([[-1.0, 2.0]]),
        "second": pd.DataFrame([[-3.0, 1.0]]),
    }

    figure = pl.matrix_heatmap_panel(matrices, center_zero=True, shared_scale=False, colorbar="each")

    try:
        assert isinstance(figure, Figure)
        image_axes = [ax for ax in figure.axes if ax.images]
        assert len(image_axes) == 2
        expected_limits = [(-2.0, 2.0), (-3.0, 3.0)]
        for ax, (expected_vmin, expected_vmax) in zip(image_axes, expected_limits):
            image = ax.images[0]
            assert image.norm.vmin == pytest.approx(expected_vmin)
            assert image.norm.vmax == pytest.approx(expected_vmax)
    finally:
        _close_figures(figure)


def _edge_interaction_dataset():
    dataset = ad.AnnData(X=np.ones((3, 2)))
    dataset.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )
    dataset.obsm["X"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
    )
    dataset.obs["cell_type"] = pd.Categorical(["A", "B", "A"])
    dataset.obsp["adjacency_matrix"] = csr_matrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
    )
    dataset.uns["Sigma_x_inv"] = {"replicate_0": np.diag([2.0, 3.0])}
    return dataset


def test_edge_interactions_plot_returns_figure():
    dataset = _edge_interaction_dataset()

    affinity_figure = pl.edge_interactions(dataset, 1, 1, mode="affinity", size=20)
    cooccurrence_figure = pl.edge_interactions(dataset, 1, 1, mode="cooccurrence", color=None, size=20)

    try:
        for figure in [affinity_figure, cooccurrence_figure]:
            assert isinstance(figure, Figure)
            assert figure.axes
    finally:
        _close_figures(affinity_figure, cooccurrence_figure)


@pytest.mark.expensive
def test_plotting_wrappers_return_figures(analyzed_shared_model):
    model = analyzed_shared_model
    marker_genes = {
        "type_0": [model.datasets[0].var_names[0], model.datasets[0].var_names[1]],
        "type_1": [model.datasets[0].var_names[2], model.datasets[0].var_names[3]],
    }

    metagene_figure = pl.metagene_embedding(model, metagene_index=0)
    heatmap_figure = pl.multireplicate_heatmap(model, uns="M")
    affinity_figure = pl.spatial_affinity_heatmap(model)
    embeddings_figure = pl.all_embeddings(model)
    medians, cell_type_figure = pl.cell_type_to_metagene(model, marker_genes)
    difference_medians, difference_figure = pl.cell_type_to_metagene_difference(
        model,
        marker_genes,
        first_metagene=0,
        second_metagene=1,
    )

    try:
        assert isinstance(embeddings_figure, list)
        assert len(embeddings_figure) == len(model.datasets)
        for figure in embeddings_figure:
            assert isinstance(figure, Figure)
            assert figure.axes

        for figure in [
            metagene_figure,
            heatmap_figure,
            affinity_figure,
            cell_type_figure,
            difference_figure,
        ]:
            assert isinstance(figure, Figure)
            assert figure.axes

        assert set(medians) == set(marker_genes)
        assert difference_medians
        assert set(difference_medians).issubset(marker_genes)
    finally:
        _close_figures(
            metagene_figure,
            heatmap_figure,
            affinity_figure,
            embeddings_figure,
            cell_type_figure,
            difference_figure,
        )


@pytest.mark.expensive
def test_embedding_category_and_umap_plots(clustered_shared_model):
    model = clustered_shared_model
    tl.umap(model, use_rep="normalized_X", joint=False)
    tl.compute_confusion_matrix(model, labels="cell_type", predictions="cell_type", joint=False)
    marker_genes = {
        "type_0": [model.datasets[0].var_names[0], model.datasets[0].var_names[1]],
        "type_1": [model.datasets[0].var_names[2], model.datasets[0].var_names[3]],
    }

    in_situ_figure = pl.in_situ(model, color="leiden")
    umap_figure, _ = pl.umap(model, color="cell_type")
    confusion_figure, _ = pl.confusion_matrix(model, labels="cell_type")
    categories_figure = pl.clusters_to_categories(model, marker_genes)

    try:
        assert isinstance(in_situ_figure, list)
        assert len(in_situ_figure) == len(model.datasets)
        for figure in in_situ_figure:
            assert isinstance(figure, Figure)
            assert figure.axes

        for figure in [umap_figure, confusion_figure, categories_figure]:
            assert isinstance(figure, Figure)
            assert figure.axes

        for dataset in model.datasets:
            assert "X_umap" in dataset.obsm
            assert "confusion_matrix" in dataset.uns
    finally:
        _close_figures(in_situ_figure, umap_figure, confusion_figure, categories_figure)


@pytest.mark.gpu
@pytest.mark.expensive
def test_affinity_magnitude_plot(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    figure, top_pairs = pl.affinity_magnitude_vs_difference(model, n_best=2)

    try:
        assert isinstance(figure, Figure)
        assert figure.axes
        assert len(top_pairs) == len(model.datasets)
        assert all(len(dataset_pairs) == 2 for dataset_pairs in top_pairs)
    finally:
        _close_figures(figure)


@pytest.mark.expensive
def test_affinity_trend_and_signature_plots(analyzed_shared_model):
    model = analyzed_shared_model
    timepoints = list(range(len(model.datasets)))
    tl.normalized_affinity_trends(model, timepoint_values=timepoints, n_best=2)

    trend_figure = pl.normalized_affinity_trends(model, timepoint_values=timepoints, n_best=2)
    enrichment_figure, signature = pl.metagene_signature_enrichment(
        model,
        metagene_index=0,
        categories=["left", "right"],
        category_key="domain",
    )

    try:
        for figure in [trend_figure, enrichment_figure]:
            assert isinstance(figure, Figure)
            assert figure.axes

        assert signature
    finally:
        _close_figures(trend_figure, enrichment_figure)


@pytest.mark.gpu
@pytest.mark.expensive
def test_multigroup_heatmap_with_differential_model(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    figure = pl.multigroup_heatmap(model, key="M_bar")

    try:
        assert isinstance(figure, Figure)
        assert figure.axes
    finally:
        _close_figures(figure)
