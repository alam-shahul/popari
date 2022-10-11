```{eval-rst}
.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

```

# API

```{eval-rst}
.. module:: spicemix

.. automodule:: spicemix
   :noindex:
```

## IO

```{eval-rst}
.. module:: spicemix.io
.. currentmodule:: spicemix

.. autosummary::
    :toctree: api/
    :recursive:
    
    io.save_anndata
    io.load_anndata
```

## Model

```{eval-rst}
.. module:: spicemix.model
.. currentmodule:: spicemix

.. autosummary::
    :toctree: api/
    :recursive:
    
    model.SpiceMixPlus
    model.load_trained_model
```

## Components

```{eval-rst}
.. module:: spicemix.components
.. currentmodule:: spicemix

.. autosummary::
    :toctree: api/
    :recursive:
    
    components.SpiceMixDataset
```

## Analysis

```{eval-rst}
.. module:: spicemix.analysis
.. currentmodule:: spicemix

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
