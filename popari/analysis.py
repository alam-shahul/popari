from typing import Optional, Sequence
from functools import partial

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

import numpy as np
import scanpy as sc

from popari.model import Popari
from popari._dataset_utils import _preprocess_embeddings, _plot_metagene_embedding, _cluster, _pca, _umap, \
                                  _plot_umap, _multireplicate_heatmap, _multigroup_heatmap, _compute_empirical_correlations, \
                                  _broadcast_operator, _compute_ari_score, _compute_silhouette_score, _plot_all_embeddings, \
                                  _evaluate_classification_task, _compute_confusion_matrix, _compute_columnwise_autocorrelation, \
                                  _plot_confusion_matrix, _compute_spatial_correlation

def setup_squarish_axes(num_axes, **subplots_kwargs):
    """Create matplotlib subplots as squarely as possible."""

    height = int(np.sqrt(num_axes))
    width = num_axes // height
    height += (width * height != num_axes)

    constrained_layout = True if "constrained_layout" not in subplots_kwargs else subplots_kwargs.pop("constrained_layout")
    dpi = 300 if "dpi" not in subplots_kwargs else subplots_kwargs.pop("dpi")
    sharex = True if "sharex" not in subplots_kwargs else subplots_kwargs.pop("sharex")
    sharey = True if "sharey" not in subplots_kwargs else subplots_kwargs.pop("sharey")

    fig, axes = plt.subplots(height, width, squeeze=False, constrained_layout=constrained_layout, dpi=dpi, sharex=sharex, sharey=sharey, **subplots_kwargs)

    return fig, axes
    
def preprocess_embeddings(trained_model: Popari, normalized_key="normalized_X"):
    """Normalize embeddings per each cell.
    
    This step helps to make cell embeddings comparable, and facilitates downstream tasks like clustering.

    """

    datasets = trained_model.datasets
    _preprocess_embeddings(datasets, normalized_key=normalized_key)

def leiden(trained_model: Popari, use_rep="normalized_X", joint: bool = False, resolution: float = 1.0, target_clusters: Optional[int] = None, tolerance: float = 0.05):
    r"""Compute Leiden clustering for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly cluster the spots
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters..
    """
   
    cluster(trained_model, use_rep=use_rep, joint=joint, resolution=resolution, target_clusters=target_clusters, tolerance=tolerance)

def cluster(trained_model: Popari, use_rep="normalized_X", joint: bool = False, method: str = "leiden", resolution: float = 1.0, target_clusters: Optional[int] = None, tolerance: float = 0.01):
    r"""Compute clustering for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly cluster the spots
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters.
    """
    
    datasets = trained_model.datasets
    _cluster(datasets, use_rep=use_rep, joint=joint, method=method, resolution=resolution, target_clusters=target_clusters, tolerance=tolerance)

def pca(trained_model: Popari, joint: bool = False, n_comps: int = 50):
    r"""Compute PCA for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly reduce dimensionality.
    """
    
    datasets = trained_model.datasets
    _pca(datasets, joint=joint, n_comps=n_comps)

def umap(trained_model: Popari, joint: bool = False, n_neighbors: int = 20):
    r"""Compute PCA for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly reduce dimensionality.
    """
    
    datasets = trained_model.datasets
    _umap(datasets, joint=joint, n_neighbors=n_neighbors)

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

def compute_ari_scores(trained_model: Popari, labels: str, predictions: str, ari_key: str = "ari"):
    r"""Compute adjusted Rand index (ARI) score  between a set of ground truth labels and an unsupervised clustering.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.
    """
    datasets = trained_model.datasets

    _broadcast_operator(datasets, partial(_compute_ari_score, labels=labels, predictions=predictions, ari_key=ari_key))

def compute_silhouette_scores(trained_model: Popari, labels: str, embeddings: str, silhouette_key: str = "silhouette"):
    r"""Compute silhouette score for a clustering based on Popari embeddings.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.
    """
    datasets = trained_model.datasets
    
    _broadcast_operator(datasets, partial(_compute_silhouette_score, labels=labels, embeddings=embeddings, silhouette_key=silhouette_key))

def compute_empirical_correlations(trained_model: Popari, feature: str = "X", output: str = "empirical_correlation"):
    """Compute the empirical spatial correlation for a feature set across all datasets.

    Args:
        trained_model: the trained Popari model.
        feature: key in `.obsm` of feature set for which spatial correlation should be computed.
        output: key in `.uns` where output correlation matrices should be stored.
    """

    datasets = trained_model.datasets
    scaling = trained_model.parameter_optimizer.spatial_affinity_state.scaling

    _compute_empirical_correlations(datasets, scaling, feature=feature, output=output)

def find_differential_genes(trained_model: Popari, top_gene_limit: int = 1):
    """Identify genes/features that distinguish differential metagenes within a group.

    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained Popari model.
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

def plot_gene_activations(trained_model: Popari, gene_subset: Sequence[str]):
    """Plot metagene activation heatmaps for target genes across all groups.
    
    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.

    Args:
        trained_model: the trained Popari model.
        gene_subset: names of genes to plot for.

    Returns:
        The names of genes that are differentially expressed with respect to their group.
    """
    gene_indices = trained_model.datasets[0].var_names.get_indexer(gene_subset)
    images = np.zeros((len(gene_indices), trained_model.K, len(trained_model.metagene_groups)))
    for group_index, group_name in enumerate(trained_model.metagene_groups):
        M_bar_subset = trained_model.datasets[0].uns["M_bar"][group_name][gene_indices]
        images[:, :, group_index] = M_bar_subset
   
    
    fig, axes = setup_squarish_axes(len(gene_indices), figsize=(10, 10))
    for ax, image, gene in zip(axes.flat, images, gene_subset):
        aspect = 0.1
        im = ax.imshow(image, interpolation='nearest', aspect=aspect)
        ax.set_title(gene)
        colorbar = fig.colorbar(im, ax=ax, orientation='vertical')

def plot_gene_trajectories(trained_model: Popari, gene_subset: Sequence[str], covariate_values: Sequence[float], **subplots_kwargs):
    """Plot metagene activation lineplots for target genes across all groups.

    
    This type of analysis is only valid for runs of Popari in which ``metagene_mode="differential"``
    was used.
    
    Args:
        trained_model: the trained Popari model.
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

def evaluate_classification_task(trained_model: Popari, embeddings: str, labels: str, joint: bool):
    r"""Use cell labels to train classifier on Popari embeddings, and evaluate train/test accuracy.

    Args:
        trained_model: the trained Popari model.
        embeddings: the key in the ``.obsm`` dataframe where the embeddings are stored.
        labels: the key in the ``.obs`` dataframe for the label data.
        joint: if `True`, jointly train the classifier across all datasets.
    """

    datasets = trained_model.datasets
    _evaluate_classification_task(datasets, embeddings=embeddings, labels=labels, joint=joint)

def compute_confusion_matrix(trained_model: Popari, labels: str, predictions: str, result_key: str = "confusion_matrix"):
    r"""Compute confusion matrix for labels and predictions.

    Useful for visualizing clustering validity.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        result_key: the key in the ``.uns`` dictionary where the reordered confusion matrix will be stored.
    """
    datasets = trained_model.datasets
    _broadcast_operator(datasets, partial(_compute_confusion_matrix, labels=labels, predictions=predictions, result_key=result_key))

def compute_columnwise_autocorrelation(trained_model: Popari, uns:str = "ground_truth_M", result_key: str = "ground_truth_M_correlation"):
    datasets = trained_model.datasets

    _broadcast_operator(datasets, partial(_compute_columnwise_autocorrelation, uns=uns, result_key=result_key))

def compute_spatial_correlation(trained_model: Popari, spatial_key: str = "Sigma_x_inv", metagene_key: str = "M", spatial_correlation_key: str = "spatial_correlation", neighbor_interactions_key: str = "neighbor_interactions"):
    """Computes spatial gene correlation according to learned metagenes.

    """
    datasets = trained_model.datasets

    _broadcast_operator(datasets, partial(_compute_spatial_correlation, spatial_key=spatial_key, metagene_key=metagene_key, spatial_correlation_key=spatial_correlation_key, neighbor_interactions_key=neighbor_interactions_key))
