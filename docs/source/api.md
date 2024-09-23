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

## Analysis: `tl`

Functions for visualizing and evaluating Popari results.

```{eval-rst}
.. module:: popari.tl
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    tl.preprocess_embeddings
    tl.leiden
    tl.compute_ari_scores
    tl.compute_silhouette_scores
    tl.compute_empirical_correlations
    tl.find_differential_genes
    tl.plot_gene_activations
    tl.plot_gene_trajectories
    tl.evaluate_classification_task
    tl.compute_confusion_matrix
    tl.compute_columnwise_autocorrelation
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
```

## Simulation

Tools to generate simulated (multisample) spatially-resolved transcriptomics, or (m)SRT.

```{eval-rst}
.. module:: popari.simulation_framework
.. currentmodule:: popari

.. autosummary::
    :toctree: api/
    :recursive:

    simulation_framework.SimulationParameters
    simulation_framework.SyntheticDataset
    simulation_framework.MultiReplicateSyntheticDataset
```
