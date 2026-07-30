from types import SimpleNamespace

import anndata as ad
import gseapy
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array, csr_matrix
from scipy.stats import false_discovery_control, fisher_exact, zscore

from popari import pl, tl
from popari._sample_axis import SampleAxis


def test_run_enrichr_returns_sorted_table(monkeypatch):
    captured = {}

    def fake_enrichr(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            results=pd.DataFrame(
                {
                    "Term": ["less significant", "more significant (GO:0000001)"],
                    "Adjusted P-value": [0.04, 0.01],
                },
            ),
        )

    monkeypatch.setattr(gseapy, "enrichr", fake_enrichr)
    monkeypatch.setattr(
        gseapy,
        "get_library",
        lambda **kwargs: {
            "less significant": ["gene_2", "gene_3"],
            "More Significant": ["gene_1", "gene_3", "gene_4"],
        },
    )

    results = tl.run_enrichr(
        ["gene_1", "gene_1", "gene_2"],
        gene_sets=["GO_Biological_Process_2023"],
        organism="Mouse",
        background=["gene_1", "gene_2", "gene_3"],
    )

    assert results["Term"].tolist() == ["more significant (GO:0000001)", "less significant"]
    assert results["Overlap Count"].tolist() == [1, 1]
    assert results["Gene Set Size"].tolist() == [3, 2]
    assert results["Overlap Ratio"].tolist() == pytest.approx([1 / 3, 1 / 2])
    assert captured == {
        "gene_list": ["gene_1", "gene_2"],
        "gene_sets": ["GO_Biological_Process_2023"],
        "organism": "Mouse",
        "background": ["gene_1", "gene_2", "gene_3"],
        "outdir": None,
        "no_plot": True,
    }


def test_run_enrichr_rejects_empty_gene_list(monkeypatch):
    monkeypatch.setattr(
        gseapy,
        "enrichr",
        lambda **kwargs: pytest.fail("Enrichr should not run for an empty gene list."),
    )

    with pytest.raises(ValueError, match="gene_list"):
        tl.run_enrichr(
            [],
            gene_sets=["GO_Biological_Process_2023"],
            organism="Mouse",
            background=["gene_1"],
        )


def test_compute_metagene_enrichment_combines_selected_metagenes(monkeypatch):
    dataset = ad.AnnData(X=np.ones((2, 3)))
    dataset.var_names = ["gene_0", "gene_1", "gene_2"]
    dataset.popari.name = "sample"
    dataset.popari.metagenes = np.arange(9).reshape(3, 3)
    calls = []

    monkeypatch.setattr(
        "popari.analysis.gene_sets.get_metagene_signature",
        lambda metagene, gene_names, **kwargs: [gene_names[np.argmax(metagene)]],
    )

    def fake_run_enrichr(gene_list, **kwargs):
        calls.append((gene_list, kwargs))
        return pd.DataFrame(
            {
                "Term": [f"term_{gene_list[0]}"],
                "Adjusted P-value": [0.01],
            },
        )

    monkeypatch.setattr("popari.analysis.gene_sets.run_enrichr", fake_run_enrichr)

    results = tl.compute_metagene_enrichment(
        dataset,
        gene_sets=["GO"],
        organism="Mouse",
        metagene_indices=[2, 0],
        sensitivity=0.75,
    )

    assert results["metagene"].tolist() == ["m2", "m0"]
    assert [call[0] for call in calls] == [["gene_2"], ["gene_2"]]
    assert all(call[1]["background"] == dataset.var_names.tolist() for call in calls)


@pytest.fixture
def signature_expression_datasets():
    comparison = ad.AnnData(
        X=np.array(
            [
                [1.0, 10.0, 3.0],
                [3.0, 20.0, 5.0],
                [5.0, 30.0, 7.0],
                [7.0, 40.0, 9.0],
            ],
        ),
        obs=pd.DataFrame({"domain": ["A", "A", "B", "B"]}),
    )
    comparison.var_names = ["gene_0", "gene_1", "gene_2"]
    comparison.uns["M"] = np.array([[1.0], [2.0], [3.0]])

    reference = comparison.copy()
    reference.X = comparison.X - 1
    return comparison, reference


def test_metagene_signature_expression_returns_labeled_means_and_differences(
    signature_expression_datasets,
    monkeypatch,
):
    comparison, reference = signature_expression_datasets
    monkeypatch.setattr(
        "popari.analysis.differential.get_metagene_signature",
        lambda *args, **kwargs: ["gene_0", "gene_2"],
    )

    means = tl.compute_metagene_signature_expression(
        comparison,
        0,
        categories=["B", "A"],
    )
    differences = tl.compute_metagene_signature_expression(
        comparison,
        0,
        reference_dataset=reference,
        categories=["A", "B"],
    )

    assert means.index.equals(pd.Index(["gene_0", "gene_2"], name="gene"))
    assert means.columns.equals(pd.Index(["B", "A"], name="category"))
    np.testing.assert_array_equal(means.to_numpy(), [[6.0, 2.0], [8.0, 4.0]])
    np.testing.assert_array_equal(differences.to_numpy(), np.ones((2, 2)))


def test_metagene_signature_expression_requires_metagenes(signature_expression_datasets):
    comparison, _ = signature_expression_datasets
    del comparison.uns["M"]

    with pytest.raises(KeyError, match="is missing"):
        tl.compute_metagene_signature_expression(comparison, 0)


def test_metagene_signature_expression_validates_reference_and_categories(
    signature_expression_datasets,
    monkeypatch,
):
    comparison, reference = signature_expression_datasets
    monkeypatch.setattr(
        "popari.analysis.differential.get_metagene_signature",
        lambda *args, **kwargs: ["gene_0"],
    )

    with pytest.raises(ValueError, match="'missing'"):
        tl.compute_metagene_signature_expression(comparison, 0, categories=["missing"])

    reference.var_names = ["gene_0", "gene_2", "gene_1"]
    with pytest.raises(ValueError, match="same genes"):
        tl.compute_metagene_signature_expression(
            comparison,
            0,
            reference_dataset=reference,
        )


@pytest.fixture
def differential_expression_dataset():
    dataset = ad.AnnData(
        X=csr_array(
            [
                [8, 0, 1],
                [7, 0, 1],
                [9, 1, 0],
                [0, 8, 1],
                [0, 7, 1],
                [1, 9, 0],
            ],
        ),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["A", "A", "A", "B", "B", "B"])},
            index=[f"cell_{index}" for index in range(6)],
        ),
    )
    dataset.var_names = ["gene_a", "gene_b", "shared"]
    return dataset


@pytest.mark.parametrize("filter_genes", [False, True])
def test_call_de_genes_handles_sparse_arrays(
    differential_expression_dataset,
    filter_genes,
):
    genes_by_group, all_genes = tl.call_de_genes(
        differential_expression_dataset,
        "group",
        "de",
        filter=filter_genes,
        min_fold_change=0,
        max_genes=1,
        p_value_threshold=1,
    )

    assert isinstance(differential_expression_dataset.X, csr_matrix)
    assert set(genes_by_group) == {"A", "B"}
    assert all(len(genes) <= 1 for genes in genes_by_group.values())
    assert all_genes == set().union(*map(set, genes_by_group.values()))


def test_compile_de_genes_filters_without_mutating_rankings():
    dataset = ad.AnnData(
        X=np.ones((4, 3)),
        obs=pd.DataFrame({"group": pd.Categorical(["A", "A", "B", "B"])}),
    )
    names = np.rec.fromarrays(
        [["gene_a", "nan", "gene_c"], ["gene_b", "gene_c", "nan"]],
        names=["A", "B"],
    )
    pvals = np.rec.fromarrays(
        [[0.001, 0.002, 0.2], [0.01, 0.03, 0.04]],
        names=["A", "B"],
    )
    dataset.uns["de"] = {"names": names.copy(), "pvals_adj": pvals}

    genes_by_group, all_genes = tl.compile_de_genes(
        dataset,
        de_category="group",
        filtered_deg_key="de",
        p_value_threshold=0.05,
        max_genes=1,
    )

    np.testing.assert_array_equal(dataset.uns["de"]["names"], names)
    np.testing.assert_array_equal(genes_by_group["A"], ["gene_a"])
    np.testing.assert_array_equal(genes_by_group["B"], ["gene_b"])
    assert all_genes == {"gene_a", "gene_b"}


def test_gene_set_enrichment_uses_background_and_corrects_pvalues():
    query_sets = {
        "query_1": {"g1", "g2", "g3", "g4", "not_measured"},
        "query_2": {"g8", "g9"},
    }
    reference_sets = {
        "reference": {"g1", "g2", "g5", "also_not_measured"},
    }
    background = {f"g{index}" for index in range(1, 11)}

    results = tl.compute_gene_set_enrichment(
        query_sets,
        reference_sets,
        background=background,
    )

    expected_first = fisher_exact([[2, 2], [1, 5]], alternative="greater")
    assert results.columns.tolist() == [
        "query",
        "reference",
        "query_size",
        "reference_size",
        "overlap_size",
        "overlap_genes",
        "odds_ratio",
        "pvalue",
        "adjusted_pvalue",
    ]
    assert results.loc[0, "query_size"] == 4
    assert results.loc[0, "reference_size"] == 3
    assert results.loc[0, "overlap_genes"] == ("g1", "g2")
    assert results.loc[0, "odds_ratio"] == pytest.approx(expected_first.statistic)
    assert results.loc[0, "pvalue"] == pytest.approx(expected_first.pvalue)
    np.testing.assert_allclose(
        results["adjusted_pvalue"],
        false_discovery_control(results["pvalue"], method="bh"),
    )


def test_gene_set_enrichment_validates_inputs():
    with pytest.raises(ValueError, match="background"):
        tl.compute_gene_set_enrichment({"query": {"g1"}}, {"reference": {"g1"}}, background=[])
    with pytest.raises(ValueError, match="query_sets"):
        tl.compute_gene_set_enrichment({}, {"reference": {"g1"}}, background=["g1"])
    with pytest.raises(ValueError, match="reference_sets"):
        tl.compute_gene_set_enrichment({"query": {"g1"}}, {}, background=["g1"])


@pytest.mark.baseline
def test_category_marker_scores_are_cell_weighted_and_handle_missing_categories():
    first = ad.AnnData(
        X=csr_array([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        obs=pd.DataFrame(
            {"domain": pd.Categorical(["A", "A", "B"])},
            index=["first_0", "first_1", "first_2"],
        ),
    )
    second = ad.AnnData(
        X=np.array([[0.0, 10.0]]),
        obs=pd.DataFrame({"domain": pd.Categorical(["B"])}, index=["second_0"]),
    )
    first.var_names = second.var_names = ["gene_0", "gene_1"]
    dataset = ad.concat(
        {"first": first, "second": second},
        label="batch",
        index_unique=None,
    )
    dataset.obs["batch"] = pd.Categorical(
        dataset.obs["batch"],
        categories=["first", "second"],
    )

    pooled = tl.compute_category_marker_scores(
        dataset,
        groupby="domain",
        n_genes=1,
        categories=["A", "B"],
    )
    per_dataset = tl.compute_category_marker_scores(
        dataset,
        groupby="domain",
        n_genes=1,
        categories=["A", "B"],
        per_dataset=True,
    )

    expected_index = pd.MultiIndex.from_tuples(
        [("A", "gene_0"), ("B", "gene_1")],
        names=["marker_category", "gene"],
    )
    assert pooled.index.equals(expected_index)
    assert pooled.columns.tolist() == ["A", "B"]
    assert np.allclose(pooled.to_numpy(), [[1.0, -1.0], [-1.0, 1.0]])
    assert per_dataset.index.equals(expected_index)
    assert per_dataset.columns.equals(
        pd.MultiIndex.from_product(
            [["A", "B"], ["first", "second"]],
            names=["category", "dataset"],
        ),
    )
    assert np.isfinite(per_dataset.to_numpy()).all()


@pytest.mark.baseline
def test_propagate_labels_mutates_hierarchy_and_returns_none():
    fine = ad.AnnData(X=np.ones((3, 1)))
    coarse = ad.AnnData(X=np.ones((2, 1)))
    coarse.obs["domain"] = ["A", "B"]
    coarse.obsm["bin_assignments"] = csr_array(
        [
            [1, 1, 0],
            [0, 0, 1],
        ],
    )

    result = tl.propagate_labels({0: fine, 1: coarse}, "domain")

    assert result is None
    assert fine.obs["domain"].tolist() == ["A", "A", "B"]


def test_postprocess_embeddings_is_sample_local_and_preserves_observation_order(monkeypatch):
    sample_a = np.column_stack(
        [
            np.arange(6, dtype=float),
            np.arange(6, dtype=float) ** 2,
        ],
    )
    sample_b = sample_a * np.array([10.0, 0.1]) + np.array([100.0, -20.0])
    embeddings = np.empty((12, 2))
    embeddings[::2] = sample_a
    embeddings[1::2] = sample_b

    dataset = ad.AnnData(
        X=np.ones((12, 1)),
        obs=pd.DataFrame(
            {"sample": pd.Categorical(["a", "b"] * 6, categories=["b", "a"])},
            index=[f"cell_{index}" for index in range(12)],
        ),
    )
    dataset.uns["popari_sample_key"] = "sample"
    dataset.obsm["X"] = embeddings

    def fake_neighbors(sample_dataset, use_rep):
        num_observations = sample_dataset.n_obs
        rows = np.arange(num_observations - 1)
        columns = rows + 1
        graph = csr_matrix(
            (np.ones(num_observations - 1), (rows, columns)),
            shape=(num_observations, num_observations),
        )
        sample_dataset.obsp["distances"] = graph
        sample_dataset.obsp["connectivities"] = graph + graph.T
        sample_dataset.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "distances_key": "distances",
            "params": {"use_rep": use_rep},
        }

    monkeypatch.setattr("popari.analysis.embeddings.sc.pp.neighbors", fake_neighbors)

    tl.postprocess_embeddings(dataset)

    expected = np.empty_like(embeddings)
    expected[::2] = zscore(sample_a, axis=0)
    expected[1::2] = zscore(sample_b, axis=0)
    np.testing.assert_allclose(dataset.obsm["normalized_X"], expected)

    sample_axis = SampleAxis.from_anndata(dataset, sample_key=dataset.popari.sample_key)
    rows, columns = dataset.obsp["connectivities"].nonzero()
    assert np.all(sample_axis.codes[rows] == sample_axis.codes[columns])
    assert dataset.uns["neighbors"]["params"]["use_rep"] == "normalized_X"


def test_postprocess_embeddings_joint_mode_treats_all_observations_as_one_population(monkeypatch):
    dataset = ad.AnnData(
        X=np.ones((6, 1)),
        obs=pd.DataFrame(
            {"batch": pd.Categorical(["a", "b"] * 3)},
            index=[f"cell_{index}" for index in range(6)],
        ),
    )
    dataset.obsm["X"] = np.array(
        [
            [0.0, 0.0],
            [100.0, 20.0],
            [1.0, 1.0],
            [110.0, 21.0],
            [2.0, 4.0],
            [120.0, 24.0],
        ],
    )
    neighbor_calls = []

    def fake_neighbors(processed_dataset, use_rep):
        neighbor_calls.append((processed_dataset.n_obs, use_rep))

    monkeypatch.setattr("popari.analysis.embeddings.sc.pp.neighbors", fake_neighbors)

    tl.postprocess_embeddings(dataset, mode="joint")

    np.testing.assert_allclose(dataset.obsm["normalized_X"], zscore(dataset.obsm["X"], axis=0))
    assert neighbor_calls == [(dataset.n_obs, "normalized_X")]


def test_cluster_domains_thresholds_each_sample(monkeypatch):
    sample_a = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ],
    )
    sample_b = sample_a * np.array([10.0, 0.1]) + np.array([100.0, 20.0])
    normalized_embeddings = np.empty((8, 2))
    normalized_embeddings[::2] = sample_a
    normalized_embeddings[1::2] = sample_b

    dataset = ad.AnnData(
        X=np.ones((8, 1)),
        obs=pd.DataFrame(
            {"sample": pd.Categorical(["a", "b"] * 4)},
            index=[f"cell_{index}" for index in range(8)],
        ),
    )
    dataset.uns["popari_sample_key"] = "sample"
    dataset.obsm["normalized_X"] = normalized_embeddings
    dataset.obsp["adjacency_matrix"] = csr_matrix(np.eye(dataset.n_obs))

    def fake_cluster(clustered_dataset, **kwargs):
        clustered_dataset.obs["leiden"] = pd.Categorical(["0"] * clustered_dataset.n_obs)

    monkeypatch.setattr("popari.analysis.clustering.cluster", fake_cluster)

    tl.cluster_domains(dataset, skip_thresholding=False)

    def threshold_normalize(values):
        thresholds = np.percentile(values, 99, axis=0)
        total_expression = (thresholds * (values > thresholds)).sum(axis=0)
        return values / total_expression

    expected = np.empty_like(normalized_embeddings)
    expected[::2] = threshold_normalize(sample_a)
    expected[1::2] = threshold_normalize(sample_b)
    np.testing.assert_allclose(dataset.obsm["normalized_thresholded_expression"], expected)
    assert dataset.obs["smoothed_domain"].tolist() == ["0"] * dataset.n_obs


def _fit_model(model, n_steps: int = 2):
    for _ in range(n_steps):
        model.estimate_parameters()
        model.estimate_weights()
    return model


@pytest.mark.baseline
def test_preprocess_and_pca(preprocessed_shared_model):
    model = preprocessed_shared_model

    assert "normalized_X" in model.adata.obsm
    assert "X_pca" in model.adata.obsm
    assert model.adata.obsm["normalized_X"].shape == (model.adata.n_obs, model.K)
    assert model.adata.obsm["X_pca"].shape == (model.adata.n_obs, model.K)
    assert np.isfinite(model.adata.obsm["normalized_X"]).all()
    assert np.isfinite(model.adata.obsm["X_pca"]).all()


@pytest.mark.expensive
def test_clustering_metrics_and_classification(clustered_shared_model):
    model = clustered_shared_model
    try:
        tl.compute_confusion_matrix(model.adata, labels="cell_type", predictions="leiden")
    except ValueError:
        pass

    assert "leiden" in model.adata.obs
    assert np.isfinite(model.adata.uns["ari"])
    assert np.isfinite(model.adata.uns["silhouette"])
    assert 0 <= model.adata.uns["microprecision_validation"] <= 1
    assert 0 <= model.adata.uns["macroprecision_validation"] <= 1


@pytest.mark.expensive
def test_embedding_and_spatial_summaries(analyzed_shared_model):
    model = analyzed_shared_model

    assert "empirical_correlation" in model.adata.uns
    assert "spatial_gene_correlation" in model.adata.uns
    assert "neighbor_interactions" in model.adata.uns
    assert "domain" in model.adata.obs
    assert set(model.adata.uns["empirical_correlation"]) == set(model.adata.popari.sample_names)
    for empirical in model.adata.uns["empirical_correlation"].values():
        assert empirical.shape[0] == empirical.shape[1] == model.K
        assert np.allclose(empirical, empirical.T)


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_affinity_analysis_helpers(differential_model_factory, gpu_context):
    model = _fit_model(
        differential_model_factory(torch_context=gpu_context, initial_context=gpu_context),
        n_steps=1,
    )

    top_pairs, correlations, variances = tl.normalized_affinity_trends(
        model.adata,
        timepoint_values=list(range(len(model.adata.popari.sample_names))),
    )

    assert top_pairs
    assert correlations
    assert variances
