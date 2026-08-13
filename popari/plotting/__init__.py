"""Plotting tools for Popari AnnData results and analysis matrices."""

from popari.plotting.diagnostics import (
    cell_type_to_metagene,
    cell_type_to_metagene_difference,
    metagene_proportion_difference,
    pretty_spatial_affinities,
    sparsity,
)
from popari.plotting.gene_sets import (
    enrichment_barplot,
    enrichment_dotplot,
    gene_set_upset,
    plot_metagene_gene_set_aurocs,
)
from popari.plotting.heatmaps import (
    category_marker_heatmap,
    confusion_matrix,
    matrix_heatmap,
    matrix_heatmap_panel,
    multigroup_heatmap,
    multireplicate_heatmap,
    sample_to_sample_matrix_distance_heatmap,
    spatial_affinity_factor_heatmap,
    spatial_affinity_heatmap,
)
from popari.plotting.interactions import affinity_difference, edge_interactions, edge_interactions_panel
from popari.plotting.settings import set_notebook_mode
from popari.plotting.spatial import (
    all_embeddings,
    clusters_to_categories,
    embedding_label_dotplot,
    embedding_label_heatmap,
    in_situ,
    metagene_embedding,
    umap,
)
from popari.plotting.trends import affinity_magnitude_vs_difference, matrix_trend_dotplot, normalized_affinity_trends
from popari.plotting.utils import setup_squarish_axes

__all__ = [
    cell_type_to_metagene.__name__,
    cell_type_to_metagene_difference.__name__,
    metagene_proportion_difference.__name__,
    pretty_spatial_affinities.__name__,
    sparsity.__name__,
    enrichment_barplot.__name__,
    enrichment_dotplot.__name__,
    gene_set_upset.__name__,
    plot_metagene_gene_set_aurocs.__name__,
    category_marker_heatmap.__name__,
    confusion_matrix.__name__,
    matrix_heatmap.__name__,
    matrix_heatmap_panel.__name__,
    multigroup_heatmap.__name__,
    multireplicate_heatmap.__name__,
    sample_to_sample_matrix_distance_heatmap.__name__,
    spatial_affinity_factor_heatmap.__name__,
    spatial_affinity_heatmap.__name__,
    affinity_difference.__name__,
    edge_interactions.__name__,
    edge_interactions_panel.__name__,
    set_notebook_mode.__name__,
    all_embeddings.__name__,
    clusters_to_categories.__name__,
    embedding_label_dotplot.__name__,
    embedding_label_heatmap.__name__,
    in_situ.__name__,
    metagene_embedding.__name__,
    umap.__name__,
    affinity_magnitude_vs_difference.__name__,
    matrix_trend_dotplot.__name__,
    normalized_affinity_trends.__name__,
    setup_squarish_axes.__name__,
]
