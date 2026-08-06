"""Differential-expression helpers."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from scipy.stats import zscore

from popari._sample_axis import SampleAxis
from popari.analysis.gene_sets import get_metagene_signature


def compute_metagene_signature_expression(
    dataset: ad.AnnData,
    metagene_index: int,
    *,
    reference_dataset: ad.AnnData | None = None,
    categories: Sequence | None = None,
    category_key: str = "domain",
    sensitivity: float = 1,
) -> pd.DataFrame:
    """Compute category-level expression of a metagene's signature genes.

    Args:
        dataset: Dataset that defines both the metagene signature and the
            comparison expression values.
        metagene_index: Column of ``dataset.uns["M"]`` used to select signature
            genes.
        reference_dataset: Optional reference whose category means are
            subtracted from those in ``dataset``.
        categories: Optional category order. By default, preserve the order
            observed in ``dataset.obs[category_key]``.
        category_key: Observation column containing category labels.
        sensitivity: Knee-detection sensitivity used to select signature genes.

    Returns:
        Signature-gene-by-category expression matrix. If ``reference_dataset``
        is supplied, values are ``dataset - reference_dataset``.

    Raises:
        KeyError: If required annotations or metagenes are missing.
        ValueError: If genes do not match, a requested category is absent, or
            ``metagene_index`` is invalid.

    """

    if category_key not in dataset.obs:
        raise KeyError(f"Observation key {category_key!r} is missing from dataset.")
    if reference_dataset is not None:
        if category_key not in reference_dataset.obs:
            raise KeyError(f"Observation key {category_key!r} is missing from reference_dataset.")
        if not dataset.var_names.equals(reference_dataset.var_names):
            raise ValueError("dataset and reference_dataset must contain the same genes in the same order.")

    if "M" not in dataset.uns:
        raise KeyError('dataset.uns["M"] is missing.')
    metagenes = np.asarray(dataset.uns["M"])
    if not 0 <= metagene_index < metagenes.shape[1]:
        raise ValueError(f"metagene_index must be between 0 and {metagenes.shape[1] - 1}.")

    signature = get_metagene_signature(
        metagenes[:, metagene_index],
        dataset.var_names,
        sensitivity=sensitivity,
        type="upregulated",
        show_plot=False,
    )
    signature_indices = dataset.var_names.get_indexer(signature)

    if categories is None:
        categories = dataset.obs[category_key].dropna().unique().tolist()
    else:
        categories = list(categories)
    if not categories:
        raise ValueError("categories must contain at least one category.")

    def category_means(source: ad.AnnData, source_name: str) -> np.ndarray:
        means = []
        for category in categories:
            mask = (source.obs[category_key] == category).to_numpy()
            if not mask.any():
                raise ValueError(f"Category {category!r} is absent from {source_name}.")
            mean = source.X[mask][:, signature_indices].mean(axis=0)
            means.append(np.asarray(mean).ravel())
        return np.column_stack(means)

    values = category_means(dataset, "dataset")
    if reference_dataset is not None:
        values -= category_means(reference_dataset, "reference_dataset")

    return pd.DataFrame(
        values,
        index=pd.Index(signature, name="gene"),
        columns=pd.Index(categories, name="category"),
    )


def compute_category_marker_scores(
    dataset: ad.AnnData,
    groupby: str,
    n_genes: int = 10,
    categories: Sequence | None = None,
    layer: str | None = None,
    per_dataset: bool = False,
) -> pd.DataFrame:
    """Compute z-scored expression of category-specific marker genes.

    Marker genes are selected independently for each category from pooled,
    cell-weighted mean expression. The returned rows identify both the
    category that selected each marker and the gene itself.

    Args:
        dataset: Unified multisample AnnData.
        groupby: Observation column containing category labels.
        n_genes: Number of marker genes to select per category.
        categories: Optional category order. Must contain every observed category.
        layer: Expression layer to aggregate. By default, use ``X``.
        per_dataset: Whether columns should represent category-replicate pairs
            instead of pooled categories.
    Returns:
        Marker-by-category z-score matrix. Rows have ``marker_category`` and
        ``gene`` index levels. In per-dataset mode, columns have ``category``
        and ``dataset`` levels.

    """

    if n_genes < 1 or n_genes > dataset.n_vars:
        raise ValueError("n_genes must be between 1 and the number of genes.")
    if groupby not in dataset.obs:
        raise KeyError(f"Observation key {groupby!r} must be present in dataset.")

    observed_categories = set(dataset.obs[groupby].dropna().unique())
    if categories is None:
        categories = sorted(observed_categories)
    else:
        categories = list(categories)
        if set(categories) != observed_categories:
            raise ValueError("categories must contain exactly the observed categories.")

    genes = dataset.var_names

    def aggregate(source: ad.AnnData):
        aggregated = sc.get.aggregate(source, by=groupby, func=["mean", "sum"], layer=layer)
        aggregate_index = pd.Index(aggregated.obs[groupby], name="category")
        means = pd.DataFrame(
            np.asarray(aggregated.layers["mean"]),
            index=aggregate_index,
            columns=genes,
        ).reindex(categories, fill_value=0.0)
        sums = pd.DataFrame(
            np.asarray(aggregated.layers["sum"]),
            index=aggregate_index,
            columns=genes,
        ).reindex(categories, fill_value=0.0)
        counts = pd.Series(
            np.asarray(aggregated.obs["n_obs_aggregated"]),
            index=aggregate_index,
        ).reindex(categories, fill_value=0)
        return means, sums, counts

    _, pooled_sums, pooled_counts = aggregate(dataset)
    pooled_means = pooled_sums.div(pooled_counts, axis="index")
    pooled_zscores = pd.DataFrame(
        np.nan_to_num(zscore(pooled_means.to_numpy(), axis=0)),
        index=pd.Index(categories, name="category"),
        columns=genes,
    )

    marker_rows = []
    for category in categories:
        marker_indices = np.argsort(-pooled_zscores.loc[category].to_numpy(), kind="stable")[:n_genes]
        marker_rows.extend((category, genes[index]) for index in marker_indices)
    marker_index = pd.MultiIndex.from_tuples(marker_rows, names=["marker_category", "gene"])

    if per_dataset:
        sample_axis = SampleAxis.from_anndata(
            dataset,
            sample_key=dataset.popari.sample_key,
        )
        category_means = {sample: aggregate(dataset[sample_axis.indices(sample)])[0] for sample in sample_axis.names}
        column_index = pd.MultiIndex.from_product(
            [categories, sample_axis.names],
            names=["category", "dataset"],
        )
        means = np.vstack(
            [category_means[dataset_name].loc[category].to_numpy() for category, dataset_name in column_index],
        )
        category_zscores = pd.DataFrame(
            np.nan_to_num(zscore(means, axis=0)),
            index=column_index,
            columns=genes,
        )
    else:
        category_zscores = pooled_zscores

    return pd.DataFrame(
        np.vstack([category_zscores[gene].to_numpy() for _, gene in marker_index]),
        index=marker_index,
        columns=category_zscores.index,
    )


def compile_de_genes(
    dataset: ad.AnnData,
    de_category: str = "cell_type",
    filtered_deg_key: str = "t-test filtered",
    p_value_threshold: float = 1e-5,
    max_genes: int = 500,
):
    """Collect significant genes from stored Scanpy ranking results.

    Args:
        dataset: Dataset containing Scanpy differential-expression results.
        de_category: Observation key used to define the ranked groups.
        filtered_deg_key: Key in ``dataset.uns`` containing ranking results.
        p_value_threshold: Maximum adjusted p-value for retained genes.
        max_genes: Maximum number of genes retained per category.

    Returns:
        A mapping from category to ranked genes and the union of retained genes.

    Raises:
        KeyError: If the category annotation or ranking results are missing.

    """

    if de_category not in dataset.obs:
        raise KeyError(f"Observation key {de_category!r} is missing from dataset.")
    if filtered_deg_key not in dataset.uns:
        raise KeyError(f"Differential-expression results {filtered_deg_key!r} are missing from dataset.uns.")

    all_de_genes = set()
    ranked_genes = dataset.uns[filtered_deg_key]["names"]
    adjusted_pvals = dataset.uns[filtered_deg_key]["pvals_adj"]
    cell_type_de_genes = {}
    for category in dataset.obs[de_category].dropna().unique():
        field = str(category)
        if field not in ranked_genes.dtype.names:
            raise KeyError(f"Category {category!r} is missing from {filtered_deg_key!r} results.")

        category_genes = ranked_genes[field]
        category_pvals = adjusted_pvals[field]
        valid_names = np.array(
            [not pd.isna(gene) and str(gene) != "nan" for gene in category_genes],
            dtype=bool,
        )
        keep = valid_names & np.isfinite(category_pvals) & (category_pvals < p_value_threshold)
        filtered_genes = category_genes[keep][:max_genes]
        cell_type_de_genes[category] = filtered_genes
        all_de_genes.update(filtered_genes)

    return cell_type_de_genes, all_de_genes


def call_de_genes(
    dataset: ad.AnnData,
    label_key: str,
    output_key: str,
    filter: bool = True,
    min_fold_change: float = 1,
    max_genes: int = 500,
    p_value_threshold: float = 1e-5,
):
    """Rank, filter, and collect differentially expressed genes.

    Args:
        dataset: Dataset whose observations will be compared.
        label_key: Observation key defining comparison groups.
        output_key: Key under which Scanpy stores ranking results.
        filter: Whether to apply Scanpy's expression and fold-change filters.
        min_fold_change: Minimum fold change used when filtering rankings.
        max_genes: Maximum number of genes retained per category.
        p_value_threshold: Maximum adjusted p-value for retained genes.

    Returns:
        A mapping from category to ranked genes and the union of retained genes.

    """

    if issparse(dataset.X):
        dataset.X = csr_matrix(dataset.X)

    filtered_deg_key = output_key

    sc.tl.rank_genes_groups(dataset, label_key, method="t-test", key_added=output_key)
    if filter:
        filtered_deg_key = f"{filtered_deg_key}_filtered"
        sc.tl.filter_rank_genes_groups(
            dataset,
            key=output_key,
            key_added=f"{output_key}_filtered",
            min_fold_change=min_fold_change,
        )

    cell_type_de_genes, all_de_genes = compile_de_genes(
        dataset,
        label_key,
        max_genes=max_genes,
        filtered_deg_key=filtered_deg_key,
        p_value_threshold=p_value_threshold,
    )

    return cell_type_de_genes, all_de_genes
