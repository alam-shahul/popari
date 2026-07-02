"""Gene-set analysis helpers."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def run_gsea_analysis(
    gene_list,
    name,
    background,
    organism: str = "mouse",
    output_name=None,
    get_dotplot: bool = True,
):
    """Run Enrichr through GSEApy and return the result plus plot axis."""

    import gseapy as gp
    from gseapy import barplot, dotplot

    enrichment_result = gp.enrichr(
        gene_list=list(gene_list),
        gene_sets=["GO_Biological_Process_2023", "KEGG_2019_Mouse"],
        organism=organism,
        background=background,
        outdir=None,
    )
    enrichment_result.results.sort_values(by="Adjusted P-value")

    if get_dotplot:
        ax = dotplot(
            enrichment_result.results,
            column="Adjusted P-value",
            x="Gene_set",
            size=2,
            top_term=5,
            figsize=(3, 5),
            title=f"{name} GSEA Enrichment",
            xticklabels_rot=45,
            show_ring=True,
            ofname=output_name,
            marker="o",
        )
    else:
        ax = barplot(
            enrichment_result.results,
            column="Adjusted P-value",
            group="Gene_set",
            size=10,
            top_term=5,
            figsize=(3, 5),
            color=["red", "green", "blue"],
            title=f"{name} GSEA Enrichment",
            ofname=output_name,
        )

    return enrichment_result, ax


def order_columns_by_best_row(scores):
    """Order matrix columns by strongest row association, strongest columns
    first."""

    return np.argsort(np.argmax(scores, axis=0) - np.max(scores, axis=0) / (np.max(scores) + 1))


def compute_gene_set_auroc(dataset, gene_sets, metagene_key: str = "M"):
    """Compute metagene correspondence to gene-set columns for one dataset."""

    from scipy.stats import mannwhitneyu
    from sklearn.metrics import auc, roc_curve

    metagenes = dataset.uns[metagene_key][dataset.name]
    _, num_metagenes = metagenes.shape
    num_gene_sets = len(gene_sets.columns)

    aurocs = np.zeros((num_gene_sets, num_metagenes))
    pvalues = np.zeros(aurocs.shape)
    for gene_set_index, gene_set_name in enumerate(gene_sets):
        gene_set = gene_sets[gene_set_name]
        gene_set_membership = dataset.var_names.isin(gene_set)

        for metagene_index, metagene in enumerate(metagenes.T):
            metagene_subset = metagene[gene_set_membership]
            metagene_subset_complement = metagene[~gene_set_membership]

            fpr, tpr, _ = roc_curve(gene_set_membership, metagene)
            aurocs[gene_set_index, metagene_index] = auc(fpr, tpr)
            pvalues[gene_set_index, metagene_index] = mannwhitneyu(
                metagene_subset,
                metagene_subset_complement,
            ).pvalue

    return aurocs, pvalues


def plot_metagene_gene_set_aurocs(dataset, gene_sets, metagene_key: str = "M"):
    """Plot metagene-by-gene-set AUROC scores for one dataset."""

    from matplotlib import gridspec

    metagenes = dataset.uns[metagene_key][dataset.name]
    _, num_metagenes = metagenes.shape
    num_gene_sets = len(gene_sets.columns)

    aurocs, pvalues = compute_gene_set_auroc(dataset, gene_sets, metagene_key=metagene_key)
    sorted_indices = order_columns_by_best_row(aurocs)
    aurocs = aurocs[:, sorted_indices]

    fig = plt.figure(figsize=(num_metagenes * 0.3, num_gene_sets * 0.5))
    grid_spec = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax = plt.subplot(grid_spec[0])
    cax = plt.subplot(grid_spec[1])

    im = ax.pcolormesh(aurocs, vmin=1 - aurocs.max(), vmax=aurocs.max(), cmap="bwr", edgecolor="k")
    ax.grid(color="k", linewidth=0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_yticks(np.arange(num_gene_sets), gene_sets.columns.values)
    ax.set_xticks(np.arange(num_metagenes), [f"m{k}" for k in sorted_indices], rotation=90)

    for label in ax.get_yticklabels():
        label.set_verticalalignment("top")
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("left")

    plt.colorbar(im, cax=cax, orientation="horizontal")
    return fig


__all__ = [
    run_gsea_analysis.__name__,
    order_columns_by_best_row.__name__,
    compute_gene_set_auroc.__name__,
    plot_metagene_gene_set_aurocs.__name__,
]
