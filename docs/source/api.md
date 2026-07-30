```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

```

# API

```{eval-rst}
.. module:: popari

.. automodule:: popari
   :noindex:
```

## IO: `io`

Tools for loading and saving Popari data and parameters.

```{eval-rst}
.. module:: popari.io
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    io.save_anndata
    io.load_anndata
```

## Model

Entry points for implementations of the Popari algorithm.

```{eval-rst}
.. module:: popari.model
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    model.Popari
    model.load_trained_model
```

## Analysis: `tl`

Functions for postprocessing and evaluating synchronized Popari results.

```{eval-rst}
.. module:: popari.tl
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    tl.postprocess_embeddings
    tl.cluster
    tl.leiden
    tl.umap
    tl.compute_ari_scores
    tl.compute_silhouette_scores
    tl.compute_empirical_correlations
    tl.propagate_labels
    tl.find_differential_genes
    tl.compute_metagene_signature_expression
    tl.evaluate_classification_task
    tl.compute_confusion_matrix
    tl.compute_columnwise_autocorrelation
    tl.compute_posthoc_colocalization
    tl.compute_edge_interactions
```

## Preprocessing: `pp`

Functions that prepare expression data and spatial graphs.

```{eval-rst}
.. module:: popari.pp
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    pp.pca
    pp.compute_spatial_neighbors
    pp.remove_connectivity_artifacts
```

## Plotting: `pl`

Functions for visualizing and evaluating Popari results.

```{eval-rst}
.. module:: popari.pl
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    pl.in_situ
    pl.all_embeddings
    pl.metagene_embedding
    pl.multireplicate_heatmap
    pl.multigroup_heatmap
    pl.confusion_matrix
    pl.spatial_affinity_heatmap
    pl.matrix_heatmap
    pl.metagene_gsea
    pl.clusters_to_categories
    pl.gene_activations
    pl.gene_trajectories
```

## Simulation

Tools to generate simulated (multisample) spatially-resolved transcriptomics, or (m)SRT.

```{eval-rst}

.. module:: popari.simulation
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    simulation
```
