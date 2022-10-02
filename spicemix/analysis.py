from typing import Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import anndata as ad
import scanpy as sc
import squidpy as sq

from sklearn.metrics import adjusted_rand_score

import pandas as pd
import seaborn as sns

from spicemix.model import SpiceMixPlus

def setup_squarish_axes(num_axes, **subplots_kwargs):
    """Create matplotlib subplots as squarely as possible."""

    height = int(np.sqrt(num_axes))
    width = num_axes // height
    height += (width * height != num_axes)

    constrained_layout = True if "constrained_layout" not in subplots_kwargs else subplots_kwargs.pop("constrained_layout")
    dpi = 300 if "dpi" not in subplots_kwargs else subplots_kwargs.pop("dpi")

    fig, axes = plt.subplots(height, width, squeeze=False, constrained_layout=constrained_layout, dpi=dpi, sharex=True, sharey=True, **subplots_kwargs)

    return fig, axes
    
def plot_metagene_embedding(trained_model: SpiceMixPlus, metagene_index: int, axes: Optional[Sequence[Axes]] = None):
    r"""Plot a single metagene in-situ across all datasets.

    Args:
        trained_model: the trained SpiceMixPlus model.
        metagene_index: the index of the metagene to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """
    datasets = trained_model.datasets

    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets))

    for dataset, ax in zip(datasets, axes.flat):
        dataset.plot_metagene_embedding(metagene_index, ax=ax)

    return fig

def leiden(trained_model: SpiceMixPlus, use_rep="normalized_X", joint: bool = False, resolution: float = 1.0, target_clusters: Optional[int] = None, tolerance: float = 0.05):
    r"""Compute Leiden clustering for all datasets.

    Args:
        trained_model: the trained SpiceMixPlus model.
        joint: if `True`, jointly cluster the spots
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters..
    """
    # TODO: implement joint clustering
   
    cluster(trained_model, use_rep=use_rep, joint=joint, resolution=resolution, target_clusters=target_clusters, tolerance=tolerance)

def cluster(trained_model: SpiceMixPlus, use_rep="normalized_X", joint: bool = False, method: str = "leiden", resolution: float = 1.0, target_clusters: Optional[int] = None, tolerance: float = 0.05):
    r"""Compute clustering for all datasets.

    Args:
        trained_model: the trained SpiceMixPlus model.
        joint: if `True`, jointly cluster the spots
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters..
    """
    # TODO: implement joint clustering
    
    datasets = trained_model.datasets
    if joint:
        dataset_names = [dataset.name for dataset in datasets]
        merged_dataset = ad.concat(datasets, label="batch", keys=dataset_names, merge="unique", uns_merge="unique", pairwise=True)
        datasets = [merged_dataset]

        
    clustering_function = getattr(sc.tl, method)
    for dataset in datasets:
        sc.pp.neighbors(dataset, use_rep=use_rep)
        clustering_function(dataset, resolution=resolution)
    
        num_clusters = len(dataset.obs[method].unique())
       
        lower_bound = 0.25 * resolution 
        upper_bound = 1.75 * resolution 
        while target_clusters and num_clusters != target_clusters and np.abs(lower_bound - upper_bound) > tolerance:
            effective_resolution = (lower_bound * upper_bound) ** 0.5
            clustering_function(dataset, resolution=effective_resolution)
            num_clusters = len(dataset.obs[method].unique())
            if num_clusters < target_clusters:
                lower_bound = effective_resolution
            elif num_clusters >= target_clusters:
                upper_bound = effective_resolution
            print(num_clusters)

    if joint:
        indices = merged_dataset.obs.groupby("batch").indices.values()
        unmerged_datasets = [merged_dataset[index] for index in indices]
        for unmerged_dataset, original_dataset in zip(unmerged_datasets, trained_model.datasets):
            original_dataset.obs[method] = unmerged_dataset.obs[method]


def plot_in_situ(trained_model: SpiceMixPlus, color="leiden", axes = None, **spatial_kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained SpiceMixPlus model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """
    datasets = trained_model.datasets
        
    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets))

    edges_width = 0.2 if "edges_width" not in spatial_kwargs else spatial_kwargs.pop("edges_width")
    spot_size = 0.04 if "spot_size" not in spatial_kwargs else spatial_kwargs.pop("spot_size")
    edges = True if "edges" not in spatial_kwargs else spatial_kwargs.pop("edges")
    palette = sc.pl.palettes.godsnot_102 if "palette" not in spatial_kwargs else spatial_kwargs.pop("palette")
    legend_fontsize = "xx-small" if "legend_fontsize" not in spatial_kwargs else spatial_kwargs.pop("legend_fontsize")

    neighbors_key = "spatial_neighbors" if "spatial_neighbors" not in spatial_kwargs else spatial_kwargs.pop("neighbors_key")
    for dataset, ax in zip(datasets, axes.flat):
        sc.pl.spatial(dataset, spot_size=spot_size, neighbors_key=neighbors_key,
            color=color, edges=edges,  edges_width=edges_width, legend_fontsize=legend_fontsize,
            ax=ax, show=False, palette=palette, **spatial_kwargs)

def plot_umap(trained_model: SpiceMixPlus, color="leiden", axes = None, **_kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained SpiceMixPlus model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """
    datasets = trained_model.datasets
        
    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets))

    edges_width = 0.2 if "edges_width" not in spatial_kwargs else spatial_kwargs.pop("edges_width")
    spot_size = 0.04 if "spot_size" not in spatial_kwargs else spatial_kwargs.pop("spot_size")
    edges = True if "edges" not in spatial_kwargs else spatial_kwargs.pop("edges")
    palette = sc.pl.palettes.godsnot_102 if "palette" not in spatial_kwargs else spatial_kwargs.pop("palette")
    legend_fontsize = "xx-small" if "legend_fontsize" not in spatial_kwargs else spatial_kwargs.pop("legend_fontsize")
    for dataset, ax in zip(datasets, axes.flat):
        sc.pl.spatial(dataset, spot_size=spot_size, neighbors_key="spatial_neighbors",
            color=color, edges=True,  edges_width=edges_width, legend_fontsize=legend_fontsize,
            ax=ax, show=False, palette=palette, **spatial_kwargs)

def multireplicate_heatmap(trained_model: SpiceMixPlus,
    title_font_size: Optional[int] = None,
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
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset
    """
    datasets = trained_model.datasets
    

    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets))

    aspect = 0.05 if "aspect" not in heatmap_kwargs else heatmap_kwargs.pop("aspect")    
    cmap = "hot" if "cmap" not in heatmap_kwargs else heatmap_kwargs.pop("cmap")    

    for dataset_index, ax in enumerate(axes.flat):
        if dataset_index >= len(datasets):
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
       
        im = ax.imshow(image, cmap=cmap, interpolation='nearest', aspect=aspect, **heatmap_kwargs)
        if title_font_size is not None:
            ax.set_title(dataset.name, fontsize= title_font_size)

        fig.colorbar(im, ax=ax, orientation='vertical')


def multigroup_heatmap(trained_model: SpiceMixPlus,
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
        trained_model: the trained SpiceMixPlus model.
        axes: A predefined set of matplotlib axes to plot on.
        obsm: the key in the ``.obsm`` dataframe to plot.
        obsp: the key in the ``.obsp`` dataframe to plot.
        uns: the key in the ``.uns`` dataframe to plot. Unstructured data must be 2D in shape.
        **heatmap_kwargs: arguments to pass to the `ax.imshow` call for each dataset
    """
    datasets = trained_model.datasets
    groups = trained_model.metagene_groups if group_type == "metagene" else trained_model.spatial_affinity_groups
    

    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(groups))

    aspect = 0.05 if "aspect" not in heatmap_kwargs else heatmap_kwargs.pop("aspect")    
    cmap = "hot" if "cmap" not in heatmap_kwargs else heatmap_kwargs.pop("cmap")    

    for group_index, (ax, group_name) in enumerate(zip(axes.flat, groups)):
        first_dataset_name = groups[group_name][0]
        first_dataset = next(filter(lambda dataset: dataset.name == first_dataset_name, datasets))

        if group_index > len(groups):
            ax.set_visible(False)
            continue
        
        image = first_dataset.uns[key][group_name]
       
        im = ax.imshow(image, cmap=cmap, interpolation='nearest', aspect=aspect, **heatmap_kwargs)
        if title_font_size is not None:
            ax.set_title(group_name, fontsize= title_font_size)
        fig.colorbar(im, ax=ax, orientation='vertical')

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

def plot_all_metagene_embeddings(trained_model: SpiceMixPlus, embedding_key: str = "X", column_names: Optional[str] = None, **spatial_kwargs):
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
    spot_size = 0.1 if "spot_size" not in spatial_kwargs else spatial_kwargs.pop("spot_size")
    palette = sc.pl.palettes.godsnot_102 if "palette" not in spatial_kwargs else spatial_kwargs.pop("palette")
    for dataset in datasets:
        axes = sc.pl.spatial(
            sq.pl.extract(dataset, embedding_key, prefix=f"{embedding_key}"),
            color=column_names,
            spot_size=spot_size,
            wspace=0.2,
            ncols=2,
        )

def compute_empirical_correlations(trained_model: SpiceMixPlus, feature: str = "X", output: str = "empirical_correlation"):
    """Compute the empirical spatial correlation for a feature set across all datasets.

    Args:
        trained_model: the trained SpiceMixPlus model.
        feature: key in `.obsm` of feature set for which spatial correlation should be computed.
        output: key in `.uns` where output correlation matrices should be stored.
    """

    datasets = trained_model.datasets
    num_replicates = len(datasets)
    K = trained_model.K
    scaling = trained_model.parameter_optimizer.spatial_affinity_state.scaling
    empirical_correlations = np.zeros([num_replicates, K, K])
    for replicate, dataset in enumerate(datasets):
        adjacency_list = dataset.obs["adjacency_list"]
        X = dataset.obsm[feature]
        Z = X / np.linalg.norm(X, axis=1, keepdims=True, ord=1)
        edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])

        x = Z[edges[:, 0]]
        y = Z[edges[:, 1]]
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        y_std = y.std(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True)
        corr = (y / y_std).T @ (x / x_std) / len(x)
        empirical_correlations[replicate] = - corr

    # Symmetrizing and zero-centering empirical_correlation
    empirical_correlations = (empirical_correlations + np.transpose(empirical_correlations, (0, 2, 1))) / 2
    empirical_correlations -= empirical_correlations.mean(axis=(1, 2), keepdims=True)
    empirical_correlations *= scaling

    for dataset, empirical_correlation in zip(datasets, empirical_correlations):
        all_correlations = {dataset.name: empirical_correlation}
        dataset.uns[output] = all_correlations

def find_differential_genes(trained_model: SpiceMixPlus, top_gene_limit: int = 1):
    """Identify genes/features that distinguish differential metagenes within a group.

    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained SpiceMixPlus model.
        top_gene_limit: the number of top genes to mark as differential for each metagene in a dataset.

    Returns:
        The names of genes that are differentially expressed with respect to their group.

    """

    datasets = trained_model.datasets
    for dataset in datasets:
        if "M_bar" not in dataset.uns:
            raise ValueError("This model was not trained in differential metagene mode.")

    genes_of_interest = set()
    for dataset in datasets:
        for group_name in trained_model.metagene_tags[dataset.name]:
            image = dataset.uns["M"][dataset.name] - dataset.uns["M_bar"][group_name]
            top_genes_per_metagene = np.argpartition(np.abs(image), -top_gene_limit, axis=0)[-top_gene_limit:]
            dataset_top_genes = dataset.var_names[top_genes_per_metagene.flatten()] 
            genes_of_interest.update(dataset_top_genes)
        
    return genes_of_interest

def plot_gene_activations(trained_model: SpiceMixPlus, gene_subset: Sequence[str]):
    """Plot metagene activation heatmaps for target genes across all groups.
    
    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained SpiceMixPlus model.
        gene_subset: names of genes to plot for.

    Returns:
        The names of genes that are differentially expressed with respect to their group.
    """
    gene_indices = trained_model.datasets[0].var_names.get_indexer(gene_subset)
    images = np.zeros((len(gene_indices), trained_model.K, len(trained_model.metagene_groups)))
    for group_index, group_name in enumerate(trained_model.metagene_groups):
        M_bar_subset = trained_model.parameter_optimizer.metagene_state.M_bar[group_name][gene_indices]
        images[:, :, group_index] = M_bar_subset
    
    fig, axes = setup_squarish_axes(len(gene_indices), figsize=(10, 10))
    for ax, image, gene in zip(axes.flat, images, gene_subset):
        aspect = 0.1
        im = ax.imshow(image, interpolation='nearest', aspect=aspect)
        ax.set_title(gene)
        colorbar = fig.colorbar(im, ax=ax, orientation='vertical')

def plot_gene_trajectories(trained_model: SpiceMixPlus, gene_subset: Sequence[str], covariate_values: Sequence[float], **subplots_kwargs):
    """Plot metagene activation lineplots for target genes across all groups.

    
    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.
    
    Args:
        trained_model: the trained SpiceMixPlus model.
        gene_subset: names of genes to plot for.
        covariate_values: dependent vairable values against which gene trajectories will be plotted.

    Returns:
        The names of genes that are differentially expressed with respect to their group.

    """
    gene_indices = trained_model.datasets[0].var_names.get_indexer(gene_subset)
    images = np.zeros((len(gene_indices), trained_model.K, len(trained_model.metagene_groups)))
    for group_index, group_name in enumerate(trained_model.metagene_groups):
        M_bar_subset = trained_model.parameter_optimizer.metagene_state.M_bar[group_name][gene_indices]
        images[:, :, group_index] = M_bar_subset
    
    summed_weights = images.sum(axis=1)
    fig, axes = setup_squarish_axes(len(gene_indices), figsize=(10, 10))
    for ax, trend, gene in zip(axes.flat, summed_weights, gene_subset):
        aspect = 0.1
        r = np.corrcoef(covariate_values, y=trend)[0, 1]
        im = ax.plot(covariate_values, trend)
        ax.set_title(f"{gene}, R = {r:.2f}")
