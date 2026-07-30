import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
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


def test_gene_set_upset_filters_contents_and_forwards_options(monkeypatch):
    captured = {}

    def fake_from_contents(contents):
        captured["contents"] = contents
        return "membership"

    def fake_plot(membership, *, fig, **kwargs):
        captured["membership"] = membership
        captured["fig"] = fig
        captured["kwargs"] = kwargs

    monkeypatch.setattr("popari.plotting.gene_sets.from_contents", fake_from_contents)
    monkeypatch.setattr("popari.plotting.gene_sets.plot", fake_plot)

    figure = pl.gene_set_upset(
        {
            "signature": ["g1", "g1", "outside"],
            "module": ["g1", "g2"],
        },
        background=["g1", "g2"],
        dpi=150,
        sort_by="cardinality",
    )

    try:
        assert isinstance(figure, Figure)
        assert captured["contents"] == {
            "signature": {"g1"},
            "module": {"g1", "g2"},
        }
        assert captured["membership"] == "membership"
        assert captured["fig"] is figure
        assert captured["kwargs"] == {"sort_by": "cardinality"}
        assert figure.dpi == pytest.approx(150)
    finally:
        _close_figures(figure)


def test_enrichment_barplot_facets_libraries_and_encodes_significance():
    results = pd.DataFrame(
        {
            "Gene_set": ["GO", "GO", "KEGG"],
            "Term": ["cell differentiation", "immune response", "signaling"],
            "Overlap": ["3/10", "2/20", "4/10"],
            "Adjusted P-value": [0.001, 0.01, 0.02],
        },
    )

    figure = pl.enrichment_barplot(results, title="Enrichment", top_term=1)

    try:
        plot_axes = figure.axes[:2]
        assert [axis.get_title() for axis in plot_axes] == ["GO", "KEGG"]
        assert all(len(axis.patches) == 1 for axis in plot_axes)
        assert figure._suptitle.get_text() == "Enrichment"
        assert figure.axes[-1].get_ylabel() == "Gene overlap ratio"
    finally:
        _close_figures(figure)


def test_enrichment_barplot_uses_odds_ratio_when_overlap_is_absent():
    results = pd.DataFrame(
        {
            "Gene_set": ["GO", "GO"],
            "Term": ["cell differentiation", "immune response"],
            "Odds Ratio": [2.0, 4.0],
            "Adjusted P-value": [0.001, 0.01],
        },
    )

    figure = pl.enrichment_barplot(results)

    try:
        assert len(figure.axes[0].patches) == 2
        assert figure.axes[-1].get_ylabel() == "Odds ratio"
    finally:
        _close_figures(figure)


def test_enrichment_dotplot_consumes_result_table(monkeypatch):
    captured = {}
    figure, ax = plt.subplots()

    def fake_plot(results, **kwargs):
        captured["results"] = results
        captured["kwargs"] = kwargs
        return kwargs["ax"]

    monkeypatch.setattr("gseapy.dotplot", fake_plot)
    results = pd.DataFrame(
        {
            "Gene_set": ["GO"],
            "Term": ["cell differentiation"],
            "Adjusted P-value": [0.01],
        },
    )

    try:
        returned = pl.enrichment_dotplot(results, ax=ax, title="Enrichment")

        assert returned is figure
        assert captured["results"] is results
        assert captured["kwargs"]["ax"] is ax
        assert captured["kwargs"]["title"] == "Enrichment"
    finally:
        _close_figures(figure)


@pytest.mark.parametrize("function", [pl.enrichment_barplot, pl.enrichment_dotplot])
def test_enrichment_plots_reject_empty_results(function):
    with pytest.raises(ValueError, match="at least one"):
        function(pd.DataFrame())


def test_gene_set_upset_accepts_existing_figure():
    figure = plt.figure()

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = pl.gene_set_upset(
                {"first": {"g1", "g2"}, "second": {"g2", "g3"}},
                fig=figure,
            )

        assert result is figure
        assert figure.axes
        assert not [warning for warning in captured if issubclass(warning.category, FutureWarning)]
    finally:
        _close_figures(figure)


@pytest.mark.parametrize(
    ("gene_sets", "background", "message"),
    [
        ({"only": {"g1"}}, None, "at least two"),
        ({"first": set(), "second": set()}, None, "at least one gene"),
        ({"first": {"g1"}, "second": {"g2"}}, {"other"}, "at least one gene"),
    ],
)
def test_gene_set_upset_validates_inputs(gene_sets, background, message):
    with pytest.raises(ValueError, match=message):
        pl.gene_set_upset(gene_sets, background=background)


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
        plotted_values = figure.axes[0].collections[0].get_array()
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
        image = ax.collections[0]
        assert image.norm.vmin == pytest.approx(-4.0)
        assert image.norm.vmax == pytest.approx(4.0)
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["col_0", "col_1"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["row_0", "row_1"]
    finally:
        _close_figures(figure)


def test_matrix_heatmap_draws_default_cell_grid():
    matrix = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["row_0", "row_1"],
        columns=["col_0", "col_1"],
    )

    figure = pl.matrix_heatmap(
        matrix,
        colorbar=False,
    )

    try:
        ax = figure.axes[0]
        mesh = ax.collections[0]
        assert mesh.get_linewidths()[0] == pytest.approx(0.5)
        np.testing.assert_allclose(mesh.get_edgecolors()[0], [0.5, 0.5, 0.5, 1], atol=0.01)
        assert ax.get_ylim() == pytest.approx((1.5, -0.5))
    finally:
        _close_figures(figure)


def test_category_marker_heatmap_uses_grouped_marker_matrix():
    index = pd.MultiIndex.from_tuples(
        [("A", "gene_0"), ("A", "gene_1"), ("B", "gene_2")],
        names=["marker_category", "gene"],
    )
    scores = pd.DataFrame(
        [[2.0, -1.0], [1.0, -0.5], [-2.0, 3.0]],
        index=index,
        columns=pd.Index(["A", "B"], name="category"),
    )

    figure = pl.category_marker_heatmap(scores, colorbar=False)

    try:
        ax = figure.axes[0]
        image = ax.collections[0]
        assert image.norm.vmin == pytest.approx(-3.0)
        assert image.norm.vmax == pytest.approx(3.0)
        assert image.get_linewidths()[0] == pytest.approx(0)
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "A - gene_0, gene_1",
            "B - gene_2",
        ]
    finally:
        _close_figures(figure)


def test_embedding_label_dotplot_returns_scanpy_plot():
    dataset = ad.AnnData(X=np.ones((4, 1)))
    dataset.obs["domain"] = pd.Categorical(["A", "A", "B", "B"])
    dataset.obsm["normalized_X"] = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
    )

    dotplot = pl.embedding_label_dotplot(dataset, label_key="domain")

    try:
        assert list(dotplot.var_names) == ["m0", "m1"]
        assert dotplot.fig is not None
    finally:
        _close_figures(dotplot.fig)


def test_matrix_heatmap_panel_uses_shared_scale_and_colorbar():
    matrices = {
        "first": pd.DataFrame([[1.0, 2.0], [3.0, 4.0]]),
        "second": pd.DataFrame([[-10.0, 0.0], [0.0, 5.0]]),
    }

    figure = pl.matrix_heatmap_panel(matrices, center_zero=True, shared_scale=True, colorbar="shared")

    try:
        assert isinstance(figure, Figure)
        heatmap_axes = figure.axes[:2]
        for ax in heatmap_axes:
            image = ax.collections[0]
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
        heatmap_axes = figure.axes[:2]
        expected_limits = [(-2.0, 2.0), (-3.0, 3.0)]
        for ax, (expected_vmin, expected_vmax) in zip(heatmap_axes, expected_limits):
            image = ax.collections[0]
            assert image.norm.vmin == pytest.approx(expected_vmin)
            assert image.norm.vmax == pytest.approx(expected_vmax)
    finally:
        _close_figures(figure)


def _edge_interaction_dataset():
    dataset = ad.AnnData(X=np.ones((3, 2)))
    dataset.popari.name = "replicate_0"
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
    interactions = tl.compute_edge_interactions(dataset)

    category_figure = pl.edge_interactions(
        dataset,
        interactions,
        category_key="cell_type",
        category_pair=("A", "B"),
        size=20,
    )
    affinity_figure = pl.edge_interactions(
        dataset,
        interactions,
        score="affinity",
        metagene_pair=(1, 1),
        size=20,
    )
    cooccurrence_figure = pl.edge_interactions(
        dataset,
        interactions,
        score="cooccurrence",
        metagene_pair=(1, 1),
        color=None,
        size=20,
    )

    try:
        for figure in [category_figure, affinity_figure, cooccurrence_figure]:
            assert isinstance(figure, Figure)
            assert figure.axes
    finally:
        _close_figures(category_figure, affinity_figure, cooccurrence_figure)


def test_edge_interactions_panel_uses_shared_category_pair_scale():
    datasets = [_edge_interaction_dataset(), _edge_interaction_dataset()]
    datasets[0].popari.name = "first"
    datasets[1].popari.name = "second"
    datasets[0].popari.spatial_affinity = np.diag([2.0, 3.0])
    datasets[1].popari.spatial_affinity = np.diag([2.0, 3.0])
    interactions = [tl.compute_edge_interactions(dataset) for dataset in datasets]

    figure = pl.edge_interactions_panel(
        datasets,
        interactions,
        category_key="cell_type",
        category_pair=("A", "B"),
        color=None,
        size=20,
    )

    try:
        assert isinstance(figure, Figure)
        assert [axis.get_title() for axis in figure.axes[:2]] == ["first", "second"]
        assert figure.axes[-1].get_ylabel() == "Edge interaction score"
    finally:
        _close_figures(figure)


def test_all_embeddings_without_adjacency_returns_figure():
    dataset = ad.AnnData(X=np.ones((4, 3)))
    dataset.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    )
    dataset.obsm["X"] = np.arange(8, dtype=float).reshape(4, 2)

    figure = pl.all_embeddings(dataset)

    try:
        assert isinstance(figure, Figure)
        assert [axis.get_title() for axis in figure.axes[:2]] == ["X_0", "X_1"]
    finally:
        _close_figures(figure)


def test_in_situ_supports_continuous_observation_values():
    dataset = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(["replicate"] * 4),
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
    dataset.popari.name = "replicate"
    dataset.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    )

    figure = pl.in_situ(
        dataset,
        color="distance",
        edges_width=0,
        figsize=(7, 3),
        shape=None,
        library_key="batch",
    )

    try:
        assert isinstance(figure, Figure)
        np.testing.assert_allclose(figure.get_size_inches(), (7, 3))
        assert figure.axes[0].collections
    finally:
        _close_figures(figure)


def test_in_situ_supports_categorical_observation_values():
    dataset = ad.AnnData(
        X=np.ones((4, 1)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(["replicate"] * 4),
                "domain": pd.Categorical(["A", "A", "B", "B"]),
            },
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
    dataset.popari.name = "replicate"
    dataset.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
    )

    figure = pl.in_situ(dataset, color="domain", edges_width=0, shape=None)

    try:
        assert isinstance(figure, Figure)
        assert figure.axes[0].collections
    finally:
        _close_figures(figure)


@pytest.mark.expensive
def test_plotting_wrappers_return_figures(analyzed_shared_model):
    model = analyzed_shared_model
    marker_genes = {
        "type_0": [model.datasets[0].var_names[0], model.datasets[0].var_names[1]],
        "type_1": [model.datasets[0].var_names[2], model.datasets[0].var_names[3]],
    }

    metagene_figure = pl.metagene_embedding(model.datasets, metagene_index=0)
    heatmap_figure = pl.multireplicate_heatmap(model.datasets, uns="M")
    affinity_figure = pl.spatial_affinity_heatmap(model.datasets)
    embeddings_figure = pl.all_embeddings(model.datasets)
    cell_type_figure, medians = pl.cell_type_to_metagene(model.datasets[0], marker_genes)
    difference_figure, difference_medians = pl.cell_type_to_metagene_difference(
        model.datasets[0],
        marker_genes,
        first_metagene=0,
        second_metagene=1,
    )

    try:
        for figure in [
            metagene_figure,
            heatmap_figure,
            affinity_figure,
            embeddings_figure,
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
    tl.umap(model.datasets, use_rep="normalized_X", joint=False)
    tl.compute_confusion_matrix(model.datasets, labels="cell_type", predictions="cell_type", joint=False)
    marker_genes = {
        "type_0": [model.datasets[0].var_names[0], model.datasets[0].var_names[1]],
        "type_1": [model.datasets[0].var_names[2], model.datasets[0].var_names[3]],
    }

    in_situ_figure = pl.in_situ(model.datasets, color="leiden")
    umap_figure, _ = pl.umap(model.datasets, color="cell_type")
    confusion_figure, _ = pl.confusion_matrix(model.datasets, labels="cell_type")
    categories_figure = pl.clusters_to_categories(model.datasets, marker_genes)
    label_heatmap_figure, label_heatmap = pl.embedding_label_heatmap(
        model.datasets,
        embedding_key="normalized_X",
        label_key="cell_type",
    )

    try:
        for figure in [
            in_situ_figure,
            umap_figure,
            confusion_figure,
            categories_figure,
            label_heatmap_figure,
        ]:
            assert isinstance(figure, Figure)
            assert figure.axes

        assert label_heatmap.shape[1] == model.K
        for dataset in model.datasets:
            assert "X_umap" in dataset.obsm
            assert "confusion_matrix" in dataset.uns
    finally:
        _close_figures(
            in_situ_figure,
            umap_figure,
            confusion_figure,
            categories_figure,
            label_heatmap_figure,
        )


@pytest.mark.gpu
@pytest.mark.expensive
def test_affinity_magnitude_plot(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    figure, top_pairs = pl.affinity_magnitude_vs_difference(model.datasets, n_best=2)

    try:
        assert isinstance(figure, Figure)
        assert figure.axes
        assert len(top_pairs) == len(model.datasets)
        assert all(len(dataset_pairs) == 2 for dataset_pairs in top_pairs)
    finally:
        _close_figures(figure)


@pytest.mark.expensive
def test_affinity_trend_plot(analyzed_shared_model):
    model = analyzed_shared_model
    timepoints = list(range(len(model.datasets)))
    tl.normalized_affinity_trends(model.datasets, timepoint_values=timepoints, n_best=2)

    trend_figure = pl.normalized_affinity_trends(model.datasets, timepoint_values=timepoints, n_best=2)

    try:
        assert isinstance(trend_figure, Figure)
        assert trend_figure.axes
    finally:
        _close_figures(trend_figure)


@pytest.mark.gpu
@pytest.mark.expensive
def test_multigroup_heatmap_with_differential_model(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    figure = pl.multigroup_heatmap(
        model.datasets,
        groups=model.metagene_groups,
        key="M_bar",
    )

    try:
        assert isinstance(figure, Figure)
        assert figure.axes
    finally:
        _close_figures(figure)


def test_set_notebook_mode_configures_plotting_and_warnings():
    original_fonttype = matplotlib.rcParams["pdf.fonttype"]
    original_axes_grid = matplotlib.rcParams["axes.grid"]
    original_vector_friendly = sc.settings._vector_friendly
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            pl.set_notebook_mode("publication")
            warnings.warn("hidden", UserWarning)
            assert not caught_warnings
            assert matplotlib.rcParams["pdf.fonttype"] == 42
            assert not matplotlib.rcParams["axes.grid"]
            assert sc.settings._vector_friendly

            pl.set_notebook_mode("exploration")
            warnings.warn("visible", UserWarning)
            assert len(caught_warnings) == 1
            assert matplotlib.rcParams["pdf.fonttype"] == matplotlib.rcParamsDefault["pdf.fonttype"]
            assert not matplotlib.rcParams["axes.grid"]
            assert not sc.settings._vector_friendly
    finally:
        matplotlib.rcParams["pdf.fonttype"] = original_fonttype
        matplotlib.rcParams["axes.grid"] = original_axes_grid
        sc.set_figure_params(vector_friendly=original_vector_friendly)
