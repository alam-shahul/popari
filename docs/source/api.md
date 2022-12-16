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

```{eval-rst}
.. module:: popari.model
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:
    
    model.SpiceMixPlus
    model.load_trained_model
```

## Components

```{eval-rst}
.. module:: popari.components
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:
    
    components.SpiceMixDataset
```

## Analysis

```{eval-rst}
.. module:: popari.analysis
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:
 
    analysis.leiden
    analysis.plot_in_situ
    analysis.compute_ari_scores
    analysis.multireplicate_heatmap
    analysis.plot_all_metagene_embeddings
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
