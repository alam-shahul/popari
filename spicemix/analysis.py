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
    datasets = trained_model.datasets

    for dataset in datasets:
        sc.pp.neighbors(dataset, use_rep=use_rep)
        sc.tl.leiden(dataset, resolution=resolution)

def plot_in_situ(trained_model: SpiceMixPlus, color="leiden", axes: Optional[Sequence[Axes]] = None):
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
        image = dataset.uns[uns][dataset.name]
        aspect = 0.05 if "aspect" not in heatmap_kwargs else hetmap_kwargs["aspect"]
        ax.imshow(image, cmap='hot', interpolation='nearest', aspect=aspect)

    return fig

def compute_ari_scores(trained_model: SpiceMixPlus, labels: str, predictions: str, ari_key: str = "ari"):
    datasets = trained_model.datasets

    for dataset in datasets:
        ari = adjusted_rand_score(dataset.obs[labels], dataset.obs[predictions])
        dataset.uns[ari_key] = ari

def plot_ari_scores(trained_model: SpiceMixPlus, ax: Optional[Axes] = None):
    fig = None
    if ax == None:
        fig, ax = plt.subplots()

    # ari_scores = pd.DataFrame(data={
    #     "method": ["SpiceMixPlus"] * len(spicemixplus_ari_scores) + ["NMF"] * len(nmf_ari_scores),
    #     "ari": spicemixplus_ari_scores + nmf_ari_scores
    # })

def plot_all_metagene_embeddings(trained_model: SpiceMixPlus, embedding_key: str = "X", column_names: Optional[str] = None):
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
