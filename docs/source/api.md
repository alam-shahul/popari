```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

```

# API

## Model and training

Core entry points for constructing and training Popari models.

```{eval-rst}
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    Popari
    Trainer
    from_pretrained
```

## Input and output: `io`

Read and write canonical unified Popari results.

```{eval-rst}
.. currentmodule:: popari.io

.. autosummary::
    :toctree: api/
    :recursive:

    load_anndata
    save_anndata
    load_anndata_hierarchy
    save_anndata_hierarchy
```

## Preprocessing: `pp`

Prepare unified expression data, annotations, sample subsets, and spatial
graphs.

```{eval-rst}
.. currentmodule:: popari.pp

.. autosummary::
    :toctree: api/
    :recursive:

    pca
    compute_spatial_neighbors
    remove_connectivity_artifacts
    relabel_categories
    subset_samples
```

## Analysis: `tl`

Postprocess embeddings and quantify clusters, metagenes, spatial interactions,
sample differences, and trajectories.

```{eval-rst}
.. currentmodule:: popari.tl

.. autosummary::
    :toctree: api/
    :recursive:

    postprocess_embeddings
    compute_metagene_proportions
    cluster
    cluster_domains
    leiden
    umap
    compute_ari_scores
    compute_silhouette_scores
    compute_confusion_matrix
    evaluate_classification_task
    compute_empirical_correlations
    compute_columnwise_autocorrelation
    propagate_labels
    call_de_genes
    compile_de_genes
    compute_category_marker_scores
    compute_metagene_signature_expression
    compute_gene_set_auroc
    compute_gene_set_enrichment
    compute_metagene_enrichment
    compute_edge_interactions
    compute_differential_edge_interactions
    compute_posthoc_colocalization
    match_categories_to_factors
    aggregate_sample_matrices
    calculate_dataset_similarity_matrix
    matrix_trends
    normalized_affinity_trends
```

## Plotting: `pl`

Visualize embeddings, spatial annotations, metagenes, affinities, interactions,
gene sets, and sample trajectories.

```{eval-rst}
.. currentmodule:: popari.pl

.. autosummary::
    :toctree: api/
    :recursive:

    in_situ
    umap
    all_embeddings
    metagene_embedding
    embedding_label_dotplot
    embedding_label_heatmap
    spatial_affinity_heatmap
    spatial_affinity_factor_heatmap
    matrix_heatmap
    matrix_heatmap_panel
    sample_to_sample_matrix_distance_heatmap
    multireplicate_heatmap
    multigroup_heatmap
    confusion_matrix
    category_marker_heatmap
    edge_interactions
    edge_interactions_panel
    affinity_difference
    metagene_proportion_difference
    enrichment_barplot
    enrichment_dotplot
    gene_set_upset
    matrix_trend_dotplot
    normalized_affinity_trends
```

## Simulation

Generate and evaluate synthetic multisample spatial transcriptomics data.

```{eval-rst}
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    simulation
```
