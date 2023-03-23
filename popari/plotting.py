from typing import Optional, Sequence
from functools import partial

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

import numpy as np
import scanpy as sc

from popari.model import Popari
from popari._dataset_utils import _preprocess_embeddings, _plot_metagene_embedding, _cluster, _pca, _plot_in_situ, \
                                  _plot_umap, _multireplicate_heatmap, _multigroup_heatmap, _compute_empirical_correlations, \
                                  _broadcast_operator, _compute_ari_score, _compute_silhouette_score, _plot_all_embeddings, \
                                  _evaluate_classification_task, _compute_confusion_matrix, _compute_columnwise_autocorrelation, \
                                  _plot_confusion_matrix, _compute_spatial_correlation

def in_situ(trained_model: Popari, color="leiden", axes = None, **spatial_kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained Popari model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """
    datasets = trained_model.datasets
    _plot_in_situ(datasets, color=color, axes=axes, **spatial_kwargs)

def metagene_embedding(trained_model: Popari, metagene_index: int, axes: Optional[Sequence[Axes]] = None, **scatterplot_kwargs):
    r"""Plot a single metagene in-situ across all datasets.

    Args:
        trained_model: the trained Popari model.
        metagene_index: the index of the metagene to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """

    datasets = trained_model.datasets
    _plot_metagene_embedding(datasets, metagene_index=metagene_index, axes=axes, **scatterplot_kwargs)

def confusion_matrix(trained_model: Popari, labels: str, confusion_matrix_key: str = "confusion_matrix"):
    datasets = trained_model.datasets

    _broadcast_operator(datasets, partial(_plot_confusion_matrix, labels=labels, confusion_matrix_key=confusion_matrix_key))

def multigroup_heatmap(trained_model: Popari,
    title_font_size: Optional[int] = None,
    group_type: str = "metagene",
    axes: Optional[Sequence[Axes]] = None,
    key: Optional[str] = None,
    **heatmap_kwargs
  ):
    r"""Plot 2D heatmap data across all datasets.

    Wrapper function to enable plotting of continuous 2D data across multiple replicates. Only
    one of ``obsm``, ``obsp`` or ``uns`` should be used.

    Args:
        trained_model: the trained Popari model.
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset
    """
    datasets = trained_model.datasets
    groups = trained_model.metagene_groups if group_type == "metagene" else trained_model.spatial_affinity_groups
    _multigroup_heatmap(datasets, title_font_size=title_font_size, groups=groups, axes=axes, key=key, **heatmap_kwargs)

def umap(trained_model: Popari, color="leiden", axes = None, **kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained Popari model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """
    datasets = trained_model.datasets

    _plot_umap(datasets, color=color, axes=axes, **kwargs)

def multireplicate_heatmap(trained_model: Popari,
    title_font_size: Optional[int] = None,
    axes: Optional[Sequence[Axes]] = None,
    obsm: Optional[str] = None,
    obsp: Optional[str] = None,
    uns: Optional[str] = None,
    nested: bool = True,
    **heatmap_kwargs
  ):
    r"""Plot 2D heatmap data across all datasets.

    Wrapper function to enable plotting of continuous 2D data across multiple replicates. Only
    one of ``obsm``, ``obsp`` or ``uns`` should be used.

    Args:
        trained_model: the trained Popari model.
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset
    """
    datasets = trained_model.datasets
    
    _multireplicate_heatmap(datasets, title_font_size=title_font_size, axes=axes,
                            obsm=obsm, obsp=obsp, uns=uns, nested=nested, **heatmap_kwargs)

def all_embeddings(trained_model: Popari, embedding_key: str = "X", column_names: Optional[str] = None, **spatial_kwargs):
    r"""Plot all laerned metagenes in-situ across all replicates.

    Each replicate's metagenes are contained in a separate plot.

    Args:
        trained_model: the trained Popari model.
        embedding_key: the key in the ``.obsm`` dataframe for the cell/spot embeddings.
        column_names: a list of the suffixes for each latent feature. If ``None``, it is assumed
            that these suffixes are just the indices of the latent features.
    """


    datasets = trained_model.datasets
    first_dataset = datasets[0]
    _, K = first_dataset.obsm[f"{embedding_key}"].shape

    if column_names == None:
        column_names = [f"{embedding_key}_{index}" for index in range(K)]

    _broadcast_operator(datasets, partial(_plot_all_embeddings, embedding_key=embedding_key, column_names=column_names, **spatial_kwargs))
