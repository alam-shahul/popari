"""General analysis helpers used by Popari result notebooks."""

from popari.analysis_utils.gene_sets import (
    compute_gene_set_auroc,
    order_columns_by_best_row,
    plot_metagene_gene_set_aurocs,
    run_gsea_analysis,
)
from popari.analysis_utils.interactions import (
    EdgeInteractions,
    average_category_interaction,
    compute_category_edge_rates,
    compute_category_interaction,
    compute_cell_average_interaction,
    compute_cell_type_pair_interaction,
    compute_edge_interactions,
    compute_metagene_pair_interaction,
    compute_pair_edge_classification_scores,
    compute_posthoc_colocalization,
    compute_spatial_colocalization,
    frequency_weighted_interaction_matrix,
    match_categories_to_factors,
    metagene_pair_edge_values,
    summarize_matrix_correlations,
)
from popari.analysis_utils.similarity import calculate_dataset_similarity_matrix

__all__ = [
    run_gsea_analysis.__name__,
    order_columns_by_best_row.__name__,
    compute_gene_set_auroc.__name__,
    plot_metagene_gene_set_aurocs.__name__,
    EdgeInteractions.__name__,
    compute_edge_interactions.__name__,
    average_category_interaction.__name__,
    compute_category_interaction.__name__,
    compute_category_edge_rates.__name__,
    match_categories_to_factors.__name__,
    compute_pair_edge_classification_scores.__name__,
    compute_posthoc_colocalization.__name__,
    compute_spatial_colocalization.__name__,
    frequency_weighted_interaction_matrix.__name__,
    summarize_matrix_correlations.__name__,
    compute_cell_average_interaction.__name__,
    compute_cell_type_pair_interaction.__name__,
    compute_metagene_pair_interaction.__name__,
    metagene_pair_edge_values.__name__,
    calculate_dataset_similarity_matrix.__name__,
]
