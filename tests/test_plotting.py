import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

from popari import pl, tl
from tests._training import train_model


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


def test_metagene_gene_set_aurocs_uses_single_cell_grid():
    dataset = ad.AnnData(X=np.ones((2, 4)), var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]))
    dataset.uns["M"] = np.array(
        [
            [4.0, 0.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [0.0, 4.0],
        ],
    )
    gene_sets = pd.DataFrame({"first": ["g0", "g1"], "second": ["g2", "g3"]})

    figure = pl.plot_metagene_gene_set_aurocs(dataset, gene_sets)

    try:
        axis = figure.axes[0]
        mesh = axis.collections[0]
        assert axis.get_xlim() == pytest.approx((-0.5, 1.5))
        assert axis.get_ylim() == pytest.approx((1.5, -0.5))
        assert [label.get_text() for label in axis.get_yticklabels()] == ["first", "second"]
        assert all(not gridline.get_visible() for gridline in axis.get_xgridlines() + axis.get_ygridlines())
        assert np.allclose(mesh.get_linewidths(), 0.5)
        assert np.allclose(mesh.get_edgecolors()[0], matplotlib.colors.to_rgba("black"))
    finally:
        _close_figures(figure)


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


def test_sample_to_sample_matrix_distance_heatmap_matches_existing_clustering():
    matrices = {
        "sample_0": np.array([[0.0, 1.0], [2.0, 4.0]]),
        "sample_1": np.array([[0.0, 2.0], [1.0, 3.0]]),
        "sample_2": np.array([[4.0, 1.0], [0.0, 2.0]]),
        "sample_3": np.array([[3.0, 0.0], [2.0, 1.0]]),
    }
    features = np.stack([matrix.flatten() for matrix in matrices.values()])
    distances = pdist(features, metric="correlation")
    expected_linkage = linkage(distances, method="ward")
    expected_order = leaves_list(expected_linkage)
    expected_distances = squareform(distances)[expected_order][:, expected_order]
    expected_names = [tuple(matrices)[index] for index in expected_order]

    figure, dendrogram_result = pl.sample_to_sample_matrix_distance_heatmap(matrices)

    try:
        heatmap_ax = figure.axes[0]
        plotted_distances = heatmap_ax.images[0].get_array()
        np.testing.assert_allclose(plotted_distances.data, expected_distances)
        np.testing.assert_array_equal(np.ma.getmaskarray(plotted_distances), np.triu(np.ones((4, 4)), k=1))
        assert [tick.get_text() for tick in heatmap_ax.get_yticklabels()] == expected_names
        assert all(not spine.get_visible() for spine in heatmap_ax.spines.values())
        assert dendrogram_result["ivl"] == expected_names
        assert len(dendrogram_result["leaves_color_list"]) == len(matrices)
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


def test_metagene_proportion_difference_plots_categories():
    dataset = ad.AnnData(X=np.ones((6, 1)))
    dataset.obsm["metagene_proportions"] = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
        ],
    )
    dataset.obs["cluster"] = pd.Categorical(["C1"] * 3 + ["C2"] * 3)

    figure = pl.metagene_proportion_difference(
        dataset,
        0,
        1,
        category_key="cluster",
        category_order=["C1", "C2"],
    )

    try:
        assert isinstance(figure, Figure)
        assert figure.axes[0].get_xlabel() == "Metagene proportion difference (m0 - m1)"
        assert figure.axes[0].get_legend() is not None
    finally:
        _close_figures(figure)


def test_metagene_proportion_difference_requires_computed_proportions():
    dataset = ad.AnnData(X=np.ones((2, 1)))
    dataset.obs["cluster"] = ["C1", "C2"]

    with pytest.raises(KeyError, match="compute_metagene_proportions"):
        pl.metagene_proportion_difference(dataset, 0, 1, category_key="cluster")


def test_embedding_label_dotplot_returns_scanpy_plot():
    dataset = ad.AnnData(X=np.ones((4, 1)))
    dataset.obs["domain"] = pd.Categorical(["A", "A", "B", "B"])
    dataset.obs["m1"] = np.arange(dataset.n_obs)
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


def test_embedding_label_heatmap_has_single_cell_grid():
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

    figure, _ = pl.embedding_label_heatmap(dataset, label_key="domain")

    try:
        mesh = figure.axes[0].collections[0]
        assert mesh.get_linewidths()[0] == pytest.approx(0.5)
        np.testing.assert_allclose(mesh.get_edgecolors()[0], [0, 0, 0, 1])
        assert figure.axes[0].get_xlim() == pytest.approx((-0.5, 1.5))
        assert figure.axes[0].get_ylim() == pytest.approx((1.5, -0.5))
        assert not any(line.get_visible() for line in figure.axes[0].get_xgridlines())
        assert not any(line.get_visible() for line in figure.axes[0].get_ygridlines())
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


def _edge_interaction_dataset(sample="replicate_0"):
    dataset = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {"batch": pd.Categorical([sample] * 3, categories=[sample])},
            index=[f"{sample}_cell_{index}" for index in range(3)],
        ),
    )
    dataset.popari.name = sample
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
    dataset.uns["Sigma_x_inv"] = {sample: np.diag([2.0, 3.0])}
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
    first = _edge_interaction_dataset("first")
    second = _edge_interaction_dataset("second")
    second.obsm["spatial"] = second.obsm["spatial"] + 100
    dataset = ad.concat(
        {"first": first, "second": second},
        label="batch",
        index_unique=None,
        merge="same",
        pairwise=True,
    )
    dataset.obs["batch"] = pd.Categorical(
        dataset.obs["batch"],
        categories=["first", "second"],
        ordered=True,
    )
    dataset.uns["Sigma_x_inv"] = {
        "first": np.diag([2.0, 3.0]),
        "second": np.diag([2.0, 3.0]),
    }
    interactions = {
        sample: tl.compute_edge_interactions(dataset, sample=sample) for sample in dataset.obs["batch"].cat.categories
    }

    figure = pl.edge_interactions_panel(
        dataset,
        interactions,
        category_key="cell_type",
        category_pair=("A", "B"),
        color=None,
        size=20,
    )

    try:
        assert isinstance(figure, Figure)
        assert [axis.get_title() for axis in figure.axes[:2]] == ["first", "second"]
        first_axis, second_axis = figure.axes[:2]
        assert not first_axis.get_shared_x_axes().joined(first_axis, second_axis)
        assert not first_axis.get_shared_y_axes().joined(first_axis, second_axis)
        assert first_axis.get_xlim() != second_axis.get_xlim()
        assert first_axis.get_ylim() != second_axis.get_ylim()
        assert figure.axes[-1].get_ylabel() == "Edge accordance score"
    finally:
        _close_figures(figure)


def test_all_embeddings_without_adjacency_returns_figure():
    dataset = ad.AnnData(
        X=np.ones((4, 3)),
        obs=pd.DataFrame(
            {"batch": pd.Categorical(["replicate"] * 4)},
            index=[f"cell_{index}" for index in range(4)],
        ),
    )
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
        assert [axis.get_title() for axis in figure.axes[:2]] == ["m_0", "m_1"]
    finally:
        _close_figures(figure)


def test_all_embeddings_restores_publication_size_scaling(monkeypatch):
    dataset = _multisample_spatial_dataset()
    dataset.obsm["X"] = np.arange(16, dtype=float).reshape(8, 2)
    sizes = []
    linewidths = []

    def fake_spatial_scatter(adata, **kwargs):
        sizes.append(kwargs["size"])
        linewidths.append(kwargs["linewidths"])

    monkeypatch.setattr("popari.plotting.spatial.sq.pl.spatial_scatter", fake_spatial_scatter)

    figure = pl.all_embeddings(dataset, size=2)

    try:
        assert sizes == [pytest.approx(2 * dataset.n_obs / 100)] * 4
        assert linewidths == [0] * 4
    finally:
        _close_figures(figure)


def test_all_embeddings_uses_column_titles_and_sample_row_labels(monkeypatch):
    dataset = _multisample_spatial_dataset()
    dataset.obsm["X"] = np.arange(16, dtype=float).reshape(8, 2)
    monkeypatch.setattr("popari.plotting.spatial.sq.pl.spatial_scatter", lambda *args, **kwargs: None)

    figure = pl.all_embeddings(dataset)

    try:
        axes = np.asarray(figure.axes).reshape(2, 2)
        assert [axis.get_title() for axis in axes[0]] == ["m_0", "m_1"]
        assert [axis.get_title() for axis in axes[1]] == ["", ""]
        assert [axis.get_ylabel() for axis in axes[:, 0]] == ["sample_a", "sample_b"]
        assert [axis.get_ylabel() for axis in axes[:, 1]] == ["", ""]
        assert all(not axis.get_xticks().size and not axis.get_yticks().size for axis in axes.flat)
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
        point_collections = [collection for collection in figure.axes[0].collections if len(collection.get_offsets())]
        assert point_collections
        assert all(len(collection.get_edgecolors()) == 0 for collection in point_collections)
        assert all(np.all(collection.get_linewidths() == 0) for collection in point_collections)
    finally:
        _close_figures(figure)


def _multisample_spatial_dataset():
    dataset = ad.AnnData(
        X=np.ones((8, 1)),
        obs=pd.DataFrame(
            {
                "batch": pd.Categorical(["sample_a"] * 4 + ["sample_b"] * 4),
                "distance": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
                "domain": pd.Categorical(["A", "B", "A", "B"] * 2),
            },
            index=[f"cell_{index}" for index in range(8)],
        ),
    )
    dataset.obsm["spatial"] = np.tile(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
        ),
        (2, 1),
    )
    dataset.obsp["adjacency_matrix"] = csr_matrix(
        (
            np.ones(8),
            (
                np.arange(8),
                [1, 0, 3, 2, 5, 4, 7, 6],
            ),
        ),
        shape=(8, 8),
    )
    return dataset


def test_in_situ_restores_publication_plotting_defaults(monkeypatch):
    dataset = _multisample_spatial_dataset()
    captured = {}

    def fake_spatial_scatter(adata, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("popari.plotting.spatial.sq.pl.spatial_scatter", fake_spatial_scatter)

    figure = pl.in_situ(
        dataset,
        samples="sample_a",
        color="domain",
        edges_width=0,
        size=2,
        shape=None,
    )

    try:
        assert captured["size"] == pytest.approx(2 * 5000 / 4)
        assert captured["legend_fontsize"] == "xx-small"
        expected_colors = np.asarray(sc.pl.palettes.godsnot_102)[[0, -1]]
        np.testing.assert_array_equal(captured["palette"].colors, expected_colors)
    finally:
        _close_figures(figure)


def test_in_situ_preserves_global_categories_while_squidpy_facets_samples(monkeypatch):
    dataset = _multisample_spatial_dataset()
    dataset.obs["domain"] = pd.Categorical(
        ["A", "B", "C", "A", "A", "B", "A", "B"],
        categories=["A", "B", "C"],
    )
    captured = {}

    def fake_spatial_scatter(adata, **kwargs):
        captured["remove_unused_categories"] = ad.settings.remove_unused_categories
        captured["categories"] = adata[adata.obs["batch"] == "sample_b"].obs["domain"].cat.categories.tolist()

    monkeypatch.setattr("popari.plotting.spatial.sq.pl.spatial_scatter", fake_spatial_scatter)

    figure = pl.in_situ(dataset, color="domain", edges_width=0, shape=None)

    try:
        assert captured["remove_unused_categories"] is False
        assert captured["categories"] == ["A", "B", "C"]
    finally:
        _close_figures(figure)

    assert ad.settings.remove_unused_categories is True


def test_in_situ_facets_selected_samples_with_total_figure_size():
    dataset = _multisample_spatial_dataset()

    figure = pl.in_situ(
        dataset,
        samples=["sample_b", "sample_a"],
        color="distance",
        edges_width=0,
        colorbar=False,
        figsize=(8, 3),
        shape=None,
    )

    try:
        np.testing.assert_allclose(figure.get_size_inches(), (8, 3))
        assert [axis.get_title() for axis in figure.axes[:2]] == ["sample_b", "sample_a"]
        for axis in figure.axes[:2]:
            norms = [collection.norm for collection in axis.collections if collection.norm is not None]
            assert norms
            assert all(norm.vmin == pytest.approx(0.0) for norm in norms)
            assert all(norm.vmax == pytest.approx(13.0) for norm in norms)
    finally:
        _close_figures(figure)


def test_umap_uses_total_figure_size_and_shared_categorical_legend(monkeypatch):
    dataset = _multisample_spatial_dataset()
    dataset.obs["domain"] = pd.Categorical(
        ["A", "B", "A", "B", "A", "B", "A", "B"],
        categories=["A", "B"],
    )
    calls = []

    def fake_umap(adata, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("popari.plotting.spatial.sc.pl.umap", fake_umap)

    figure, axes = pl.umap(dataset, color="domain", figsize=(8, 4))

    try:
        np.testing.assert_allclose(figure.get_size_inches(), (8, 4))
        assert len(calls) == len(dataset.popari.sample_names)
        assert all(call["legend_loc"] == "none" for call in calls)
        assert all(call["palette"] == calls[0]["palette"] for call in calls)
        assert [text.get_text() for text in figure.legends[0].get_texts()] == ["A", "B"]
        assert axes.size >= len(dataset.popari.sample_names)
    finally:
        _close_figures(figure)


def test_in_situ_uses_shared_categorical_palette_and_spatial_edges():
    dataset = _multisample_spatial_dataset()
    dataset.obs["domain"] = dataset.obs["domain"].astype(str)

    figure = pl.in_situ(
        dataset,
        color="domain",
        edges_width=1,
        colorbar=False,
        figsize=(8, 3),
        shape=None,
    )

    try:
        axes = figure.axes[:2]
        assert all(any(isinstance(collection, LineCollection) for collection in axis.collections) for axis in axes)
        palettes = []
        for axis in axes:
            colors = np.concatenate(
                [
                    collection.get_facecolors()
                    for collection in axis.collections
                    if isinstance(collection, PathCollection) and len(collection.get_facecolors())
                ],
            )
            palettes.append(np.unique(colors, axis=0))
        np.testing.assert_allclose(palettes[0], palettes[1])
    finally:
        _close_figures(figure)


@pytest.mark.expensive
def test_plotting_wrappers_return_figures(analyzed_shared_model):
    model = analyzed_shared_model
    marker_genes = {
        "type_0": [model.adata.var_names[0], model.adata.var_names[1]],
        "type_1": [model.adata.var_names[2], model.adata.var_names[3]],
    }

    metagene_figure = pl.metagene_embedding(model.adata, metagene_index=0, figsize=(8, 5))
    heatmap_figure = pl.multireplicate_heatmap(model.adata, uns="M")
    affinity_figure = pl.spatial_affinity_heatmap(model.adata)
    embeddings_figure = pl.all_embeddings(model.adata)
    cell_type_figure, medians = pl.cell_type_to_metagene(model.adata, marker_genes)
    difference_figure, difference_medians = pl.cell_type_to_metagene_difference(
        model.adata,
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

        assert tuple(metagene_figure.get_size_inches()) == pytest.approx((8, 5))
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


def test_spatial_affinity_factor_heatmaps(differential_model_factory):
    model = differential_model_factory(
        spatial_affinity_parameterization="factorized",
        spatial_affinity_rank=2,
    )
    model.materialize_results()

    transform_figure = pl.spatial_affinity_factor_heatmap(
        model.adata,
        component="A",
        figsize=(4, 5),
    )
    interaction_figure = pl.spatial_affinity_factor_heatmap(
        model.adata,
        component="B",
        figsize=(4, 3),
    )

    try:
        assert isinstance(transform_figure, Figure)
        assert tuple(transform_figure.get_size_inches()) == pytest.approx((4, 5))
        transform_axes = [axis for axis in transform_figure.axes if axis.get_title()]
        assert [axis.get_title() for axis in transform_axes] == ["Shared spatial-factor composition A"]
        transform_axis = transform_axes[0]
        assert [tick.get_text() for tick in transform_axis.get_xticklabels()] == ["m0", "m1", "m2"]
        assert [tick.get_text() for tick in transform_axis.get_yticklabels()] == ["s0", "s1"]
        transform_image = transform_axis.collections[0]
        assert transform_image.get_cmap().name == "Reds"
        assert transform_image.get_clim() == (0, 1)

        assert isinstance(interaction_figure, Figure)
        interaction_axes = [axis for axis in interaction_figure.axes if axis.get_title()]
        assert [axis.get_title() for axis in interaction_axes] == model.replicate_names
    finally:
        _close_figures(transform_figure, interaction_figure)


@pytest.mark.expensive
def test_embedding_category_and_umap_plots(clustered_shared_model):
    model = clustered_shared_model
    tl.umap(model.adata, use_rep="normalized_X")
    tl.compute_confusion_matrix(model.adata, labels="cell_type", predictions="cell_type")
    marker_genes = {
        "type_0": [model.adata.var_names[0], model.adata.var_names[1]],
        "type_1": [model.adata.var_names[2], model.adata.var_names[3]],
    }

    in_situ_figure = pl.in_situ(model.adata, color="leiden")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ad.ImplicitModificationWarning)
        umap_figure, _ = pl.umap(model.adata, color="cell_type")
    assert not any(isinstance(warning.message, ad.ImplicitModificationWarning) for warning in caught_warnings)
    confusion_figure, _ = pl.confusion_matrix(model.adata, labels="cell_type")
    categories_figure = pl.clusters_to_categories(model.adata, marker_genes)
    label_heatmap_figure, label_heatmap = pl.embedding_label_heatmap(
        model.adata,
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
        assert "X_umap" in model.adata.obsm
        assert "confusion_matrix" in model.adata.uns
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
    train_model(model, iterations=2)
    model.materialize_results()

    figure, top_pairs = pl.affinity_magnitude_vs_difference(model.adata, n_best=2)

    try:
        assert isinstance(figure, Figure)
        assert figure.axes
        assert len(top_pairs) == len(model.adata.popari.sample_names)
        assert all(len(dataset_pairs) == 2 for dataset_pairs in top_pairs)
    finally:
        _close_figures(figure)


@pytest.mark.expensive
def test_affinity_trend_plot(analyzed_shared_model):
    model = analyzed_shared_model
    timepoints = list(range(len(model.adata.popari.sample_names)))
    tl.normalized_affinity_trends(model.adata, timepoint_values=timepoints, n_best=2)

    trend_figure = pl.normalized_affinity_trends(model.adata, timepoint_values=timepoints, n_best=2)

    try:
        assert isinstance(trend_figure, Figure)
        assert trend_figure.axes
    finally:
        _close_figures(trend_figure)


def test_matrix_trend_dotplot_encodes_lower_triangle():
    trends = tl.MatrixTrends(
        correlations=np.array(
            [
                [0.0, 0.25, 0.5],
                [-0.5, 0.75, 0.9],
                [-1.0, 0.5, 1.0],
            ],
        ),
        slopes=np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.5, 1.0],
                [0.25, 0.75, 1.0],
            ],
        ),
    )

    figure = pl.matrix_trend_dotplot(trends, ["A", "B", "C"])

    try:
        axis = figure.axes[0]
        points = axis.collections[0]
        np.testing.assert_array_equal(
            points.get_offsets(),
            [[0, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 2]],
        )
        assert points.get_cmap().name == "seismic"
        assert points.get_clim() == (-1, 1)
        assert points.get_sizes().min() == pytest.approx(5)
        assert points.get_sizes().max() == pytest.approx(150)
        assert len(axis.patches) == 6
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["A", "B", "C"]
        assert all(not spine.get_visible() for spine in axis.spines.values())
    finally:
        _close_figures(figure)


def test_matrix_trend_dotplot_validates_shapes():
    with pytest.raises(ValueError, match="square"):
        pl.matrix_trend_dotplot(
            tl.MatrixTrends(np.ones((2, 3)), np.ones((2, 3))),
            ["A", "B"],
        )


@pytest.mark.gpu
@pytest.mark.expensive
def test_multigroup_heatmap_with_differential_affinities(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    train_model(model, iterations=2)
    model.materialize_results()

    figure = pl.multigroup_heatmap(
        model.adata,
        groups=model.spatial_affinity_groups,
        key="spatial_affinity_bar",
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
