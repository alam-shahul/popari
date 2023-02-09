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

## IO

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

## Components

Objects that are helpful for working with Popari.

```{eval-rst}
.. module:: popari.components
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:
    
    components.PopariDataset
```

## Analysis

Functions for visualizing and evaluating Popari results.

```{eval-rst}
.. module:: popari.analysis
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:
 
    analysis.preprocess_embeddings
    analysis.leiden
    analysis.plot_in_situ
    analysis.compute_ari_scores
    analysis.multireplicate_heatmap
    analysis.plot_all_embeddings
    analysis.plot_metagene_embedding
    analysis.multigroup_heatmap
    analysis.compute_ari_scores
    analysis.compute_silhouette_scores
    analysis.compute_empirical_correlations
    analysis.find_differential_genes
    analysis.plot_gene_activations
    analysis.plot_gene_trajectories
    analysis.evaluate_classification_task
    analysis.compute_confusion_matrix
    analysis.plot_confusion_matrix
    analysis.compute_columnwise_autocorrelation
```
