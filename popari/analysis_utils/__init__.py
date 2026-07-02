"""General analysis helpers used by Popari result notebooks."""

from popari.analysis_utils.gene_sets import (
    compute_gene_set_auroc,
    order_columns_by_best_row,
    plot_metagene_gene_set_aurocs,
    run_gsea_analysis,
)
from popari.analysis_utils.interactions import average_category_interaction, compute_category_interaction
from popari.analysis_utils.similarity import calculate_dataset_similarity_matrix

__all__ = [
    run_gsea_analysis.__name__,
    order_columns_by_best_row.__name__,
    compute_gene_set_auroc.__name__,
    plot_metagene_gene_set_aurocs.__name__,
    average_category_interaction.__name__,
    compute_category_interaction.__name__,
    calculate_dataset_similarity_matrix.__name__,
]
