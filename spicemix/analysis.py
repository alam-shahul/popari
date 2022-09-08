from typing import Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import numpy as np
import scanpy as sc
import squidpy as sq

from sklearn.metrics import adjusted_rand_score

import pandas as pd
import seaborn as sns

from spicemix.model import SpiceMixPlus

def plot_metagene_embedding(trained_model: SpiceMixPlus, metagene_index: int, axes: Optional[Sequence[Axes]] = None):
    r"""Plot a single metagene in-situ across all datasets.

    Args:
        trained_model: the trained SpiceMixPlus model.
        metagene_index: the index of the metagene to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """
    datasets = trained_model.datasets

    if axes == None:
        height = int(np.sqrt(len(datasets)))
        width = len(datasets) // width
        height += (width * height != len(datasets))
        fig, axes = plt.subplots(height, width)

    for dataset, ax in zip(datasets, axes.flat):
        dataset.plot_metagene_embedding(metagene_index, ax=ax)

    return fig

def leiden(trained_model: SpiceMixPlus, use_rep="normalized_X", resolution: float = 1.0):
    r"""Compute Leiden clustering for all datasets.

    Uses ``use_rep`` as the basis for 

    Args:
        trained_model: the trained SpiceMixPlus model.
        use_rep: the key in the ``.obsm`` dataframe to use as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters..
    """
    
    datasets = trained_model.datasets

    for dataset in datasets:
        sc.pp.neighbors(dataset, use_rep=use_rep)
        sc.tl.leiden(dataset, resolution=resolution)

def plot_in_situ(trained_model: SpiceMixPlus, color="leiden", axes: Optional[Sequence[Axes]] = None):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained SpiceMixPlus model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """
    datasets = trained_model.datasets
    
    fig = None
    if axes == None:
        height = int(np.sqrt(len(datasets)))
        width = len(datasets) // height
        height += (width * height != len(datasets))
        fig, axes = plt.subplots(height, width, dpi=300)

    for dataset, ax in zip(datasets, axes.flat):
        sc.pl.spatial(dataset, spot_size=0.02, neighbors_key="spatial_neighbors",
            color=color, edges=True,  edges_width=0.5, ax=ax)

def multireplicate_heatmap(trained_model: SpiceMixPlus,
    axes: Optional[Sequence[Axes]] = None,
    obsm: Optional[str] = None,
    obsp: Optional[str] = None,
    uns: Optional[str] = None,
    **heatmap_kwargs
  ):
    r"""Plot 2D heatmap data across all datasets.

    Wrapper function to enable plotting of continuous 2D data across multiple replicates. Only
    one of ``obsm``, ``obsp`` or ``uns`` should be used.

    Args:
        trained_model: the trained SpiceMixPlus model.
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the 
    """
    datasets = trained_model.datasets
    

    fig = None
    if axes == None:
        height = int(np.sqrt(len(datasets)))
        width = len(datasets) // height
        height += (width * height != len(datasets))
        fig, axes = plt.subplots(height, width)

    for dataset_index, ax in enumerate(axes.flat):
        if dataset_index > len(datasets):
            ax.set_visible(False)
            continue
        
        dataset = datasets[dataset_index]
        key = None
        if obsm:
            image = dataset.obsm[obsm][dataset.name]
        if obsp:
            image = dataset.obsp[obsp][dataset.name]
        if uns:
            image = dataset.uns[uns][dataset.name]

        aspect = 0.05 if "aspect" not in heatmap_kwargs else heatmap_kwargs.pop("aspect")
       
        ax.imshow(image, cmap='hot', interpolation='nearest', aspect=aspect, **heatmap_kwargs)

    return fig

def compute_ari_scores(trained_model: SpiceMixPlus, labels: str, predictions: str, ari_key: str = "ari"):
    r"""Compute adjusted Rand index (ARI) score  between a set of ground truth labels and an unsupervised clustering.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        trained_model: the trained SpiceMixPlus model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.
    """
    datasets = trained_model.datasets

    for dataset in datasets:
        ari = adjusted_rand_score(dataset.obs[labels], dataset.obs[predictions])
        dataset.uns[ari_key] = ari

def plot_all_metagene_embeddings(trained_model: SpiceMixPlus, embedding_key: str = "X", column_names: Optional[str] = None):
    r"""Plot all laerned metagenes in-situ across all replicates.

    Each replicate's metagenes are contained in a separate plot.

    Args:
        trained_model: the trained SpiceMixPlus model.
        embedding_key: the key in the ``.obsm`` dataframe for the cell/spot embeddings.
        column_names: a list of the suffixes for each latent feature. If ``None``, it is assumed
            that these suffixes are just the indices of the latent features.
    """

    if column_names == None:
        column_names = [f"{embedding_key}_{index}" for index in range(trained_model.K)]

    datasets = trained_model.datasets
    for dataset in datasets:
        axes = sc.pl.spatial(
            sq.pl.extract(dataset, embedding_key, prefix=f"{embedding_key}"),
            color=column_names,
            spot_size=2,
            wspace=0.2,
            ncols=2,
        )
