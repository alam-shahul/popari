from typing import Optional, Sequence, Callable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

import numpy as np
import anndata as ad
import scanpy as sc
import squidpy as sq
import networkx as nx

from scipy.stats import zscore

from sklearn.metrics import adjusted_rand_score, silhouette_score, precision_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns

from popari.components import PopariDataset

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

def _preprocess_embeddings(datasets: Sequence[PopariDataset], normalized_key="normalized_X"):
    """Normalize embeddings per each cell.
    
    This step helps to make cell embeddings comparable, and facilitates downstream tasks like clustering.

    """
    # TODO: implement

    for dataset in datasets:
        if "X" not in dataset.obsm:
            raise ValueError("Must initialize embeddings before normalizing them.")

        dataset.obsm[normalized_key] = zscore(dataset.obsm["X"])
        sc.pp.neighbors(dataset, use_rep=normalized_key)

    return datasets

def _plot_metagene_embedding(datasets: Sequence[PopariDataset], metagene_index: int, axes: Optional[Sequence[Axes]] = None, **scatterplot_kwargs):
    r"""Plot a single metagene in-situ across all datasets.

    Args:
        trained_model: the trained Popari model.
        metagene_index: the index of the metagene to plot.
        axes: A predefined set of matplotlib axes to plot on.

    """

    legend = False if "legend" not in scatterplot_kwargs else scatterplot_kwargs.pop("legend")
    s = 0.5 if "legend" not in scatterplot_kwargs else scatterplot_kwargs.pop("s")
    linewidth= 0 if "linewidth" not in scatterplot_kwargs else scatterplot_kwargs.pop("linewidth")
    palette = "viridis" if "palette" not in scatterplot_kwargs else scatterplot_kwargs.pop("palette")
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=False, sharey=False)

    for dataset, ax in zip(datasets, axes.flat):
        ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        ax.set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        ax.set_yticks([], [])  # same for y ticks
        dataset.plot_metagene_embedding(metagene_index, legend=legend, s=s, linewidth=linewidth, palette=palette, ax=ax, **scatterplot_kwargs)

def _cluster(datasets: Sequence[PopariDataset], use_rep="normalized_X", joint: bool = False, method: str = "leiden",
             n_neighbors:int = 20, resolution: float = 1.0, target_clusters: Optional[int] = None, random_state: int = 0, tolerance: float = 0.01, **kwargs):
    r"""Compute clustering for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly cluster the spots
        use_rep: the key in the ``.obsm`` dataframe to ue as input to the Leiden clustering algorithm.
        resolution: the resolution to use for Leiden clustering. Higher values yield finer clusters.
    """
    if joint:
        original_datasets = datasets
        dataset_names = [dataset.name for dataset in datasets]
        merged_dataset = ad.concat(datasets, label="batch", keys=dataset_names, merge="unique", uns_merge="unique", pairwise=True)
        datasets = [merged_dataset]

    clustering_function = getattr(sc.tl, method)
    for dataset in datasets:
        sc.pp.neighbors(dataset, use_rep=use_rep, n_neighbors=n_neighbors)
        clustering_function(dataset, resolution=resolution, random_state=random_state, **kwargs)
    
        num_clusters = len(dataset.obs[method].unique())
       
        lower_bound = 0.1 * resolution 
        upper_bound = 10 * resolution 
        while target_clusters and num_clusters != target_clusters and np.abs(lower_bound - upper_bound) > tolerance:
            effective_resolution = (lower_bound * upper_bound) ** 0.5
            clustering_function(dataset, resolution=effective_resolution)
            num_clusters = len(dataset.obs[method].unique())
            if num_clusters < target_clusters:
                lower_bound = effective_resolution
            elif num_clusters >= target_clusters:
                upper_bound = effective_resolution
            print(f"Current number of clusters: {num_clusters}")
            print(f"Resolution: {effective_resolution}")

    if joint:
        indices = merged_dataset.obs.groupby("batch").indices.values()
        unmerged_datasets = [merged_dataset[index] for index in indices]
        for unmerged_dataset, original_dataset in zip(unmerged_datasets, original_datasets):
            original_dataset.obs[method] = unmerged_dataset.obs[method]

        return original_datasets, merged_dataset

    return datasets

def _pca(datasets: Sequence[PopariDataset], joint: bool = False, n_comps: int = 50):
    r"""Compute PCA for all datasets.

    Args:
        trained_model: the trained Popari model.
        joint: if `True`, jointly reduce dimensionality.
    """
    
    if joint:
        original_datasets = datasets
        dataset_names = [dataset.name for dataset in datasets]
        merged_dataset = ad.concat(datasets, label="batch", keys=dataset_names, merge="unique", uns_merge="unique", pairwise=True)
        datasets = [merged_dataset]
        
    for dataset in datasets:
        sc.pp.pca(dataset, n_comps=n_comps)

    if joint:
        indices = merged_dataset.obs.groupby("batch").indices.values()
        unmerged_datasets = [merged_dataset[index] for index in indices]
        for unmerged_dataset, original_dataset in zip(unmerged_datasets, original_datasets):
            original_dataset.obsm["X_pca"] = unmerged_dataset.obsm["X_pca"]
            original_dataset.varm["PCs"] = unmerged_dataset.varm["PCs"]
            original_dataset.uns["pca"] = {
                "variance_ratio": unmerged_dataset.uns["pca"]["variance_ratio"],
                "variance": unmerged_dataset.uns["pca"]["variance"],
            }

        return original_datasets, merged_dataset

    return datasets

def _plot_in_situ(datasets: Sequence[PopariDataset], color="leiden", axes = None, **spatial_kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained Popari model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """

    sharex = False if "sharex" not in spatial_kwargs else spatial_kwargs.pop("sharex")
    sharey = False if "sharey" not in spatial_kwargs else spatial_kwargs.pop("sharey")
        
    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

    edges_width = 0.2 if "edges_width" not in spatial_kwargs else spatial_kwargs.pop("edges_width")
    size = 0.04 if "size" not in spatial_kwargs else spatial_kwargs.pop("size")
    edges = True if "edges" not in spatial_kwargs else spatial_kwargs.pop("edges")
    palette = ListedColormap(sc.pl.palettes.godsnot_102) if "palette" not in spatial_kwargs else spatial_kwargs.pop("palette")
    legend_fontsize = "xx-small" if "legend_fontsize" not in spatial_kwargs else spatial_kwargs.pop("legend_fontsize")

    neighbors_key = "spatial_neighbors" if "spatial_neighbors" not in spatial_kwargs else spatial_kwargs.pop("neighbors_key")
    for dataset, ax in zip(datasets, axes.flat):
        ax.set_aspect('equal', 'box')
        sq.pl.spatial_scatter(dataset, shape=None, size=size, connectivity_key="adjacency_matrix",
            color=color, edges_width=edges_width, legend_fontsize=legend_fontsize,
            ax=ax, palette=palette, **spatial_kwargs)

    return fig

def _plot_umap(datasets: Sequence[PopariDataset], color="leiden", axes = None, **kwargs):
    r"""Plot a categorical label across all datasets in-situ.

    Extends AnnData's ``sc.pl.spatial`` function to plot labels/values across multiple replicates.

    Args:
        trained_model: the trained Popari model.
        color: the key in the ``.obs`` dataframe to plot.
        axes: A predefined set of matplotlib axes to plot on.
    """

    sharex = False if "sharex" not in kwargs else kwargs.pop("sharex")
    sharey = False if "sharey" not in kwargs else kwargs.pop("sharey")
        
    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

    edges_width = 0.2 if "edges_width" not in kwargs else kwargs.pop("edges_width")
    spot_size = 0.04 if "spot_size" not in kwargs else kwargs.pop("spot_size")
    edges = True if "edges" not in kwargs else kwargs.pop("edges")
    palette = ListedColormap(sc.pl.palettes.godsnot_102) if "palette" not in kwargs else kwargs.pop("palette")
    legend_fontsize = "xx-small" if "legend_fontsize" not in kwargs else kwargs.pop("legend_fontsize")
    for dataset, ax in zip(datasets, axes.flat):
        sc.pl.umap(dataset, spot_size=spot_size, neighbors_key="spatial_neighbors",
            color=color, edges=True,  edges_width=edges_width, legend_fontsize=legend_fontsize,
            ax=ax, show=False, palette=palette, **kwargs)

    return fig

def _multireplicate_heatmap(datasets: Sequence[PopariDataset],
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
    
    sharex = True if "sharex" not in heatmap_kwargs else heatmap_kwargs.pop("sharex")
    sharey = True if "sharey" not in heatmap_kwargs else heatmap_kwargs.pop("sharey")
        
    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

    aspect = 1 if "aspect" not in heatmap_kwargs else heatmap_kwargs.pop("aspect")    
    cmap = "hot" if "cmap" not in heatmap_kwargs else heatmap_kwargs.pop("cmap")    

    for dataset_index, ax in enumerate(axes.flat):
        if dataset_index >= len(datasets):
            ax.set_visible(False)
            continue
        
        dataset = datasets[dataset_index]
        key = None
        if obsm:
            image = dataset.obsm[obsm]
        if obsp:
            image = dataset.obsp[obsp]
        if uns:
            image = dataset.uns[uns]

        if nested:
            image = image[dataset.name]
       
        im = ax.imshow(image, cmap=cmap, interpolation='nearest', aspect=aspect, **heatmap_kwargs)
        if title_font_size is not None:
            ax.set_title(dataset.name, fontsize= title_font_size)

        plt.colorbar(im, ax=ax, orientation='vertical')


    return fig

def _multigroup_heatmap(datasets: Sequence[PopariDataset],
    groups: dict,
    title_font_size: Optional[int] = None,
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
    
    sharex = True if "sharex" not in heatmap_kwargs else heatmap_kwargs.pop("sharex")
    sharey = True if "sharey" not in heatmap_kwargs else heatmap_kwargs.pop("sharey")
        
    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

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

    return fig

def _compute_empirical_correlations(datasets: Sequence[PopariDataset], scaling: float, feature: str = "X", output: str = "empirical_correlation"):
    """Compute the empirical spatial correlation for a feature set across all datasets.

    Args:
        trained_model: the trained Popari model.
        feature: key in `.obsm` of feature set for which spatial correlation should be computed.
        output: key in `.uns` where output correlation matrices should be stored.
    """

    num_replicates = len(datasets)

    first_dataset = datasets[0]
    _, K = first_dataset.obsm[feature].shape
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

    return datasets

def _broadcast_operator(datasets: Sequence[PopariDataset], operator: Callable):
    r"""Broadcast a dataset operator to a list of datasets.

    Args:
        datasets: list of datasets to broadcast to
        dataset_function: function that takes in a single dataset

    """

    for dataset in datasets:
        operator(dataset)

    return datasets

def _compute_ari_score(dataset: PopariDataset, labels: str, predictions: str, ari_key: str = "ari"):
    r"""Compute adjusted Rand index (ARI) score  between a set of ground truth labels and an unsupervised clustering.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.
    """

    ari = adjusted_rand_score(dataset.obs[labels], dataset.obs[predictions])
    dataset.uns[ari_key] = ari

def _compute_silhouette_score(dataset: PopariDataset, labels: str, embeddings: str, silhouette_key: str = "silhouette"):
    r"""Compute silhouette score for a clustering based on Popari embeddings.

    Useful for assessing clustering validity. ARI score is computed per dataset.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        ari_key: the key in the ``.uns`` dictionary where the ARI score will be stored.
    """
        
    silhouette = silhouette_score(dataset.obsm[embeddings], dataset.obs[labels])
    dataset.uns[silhouette_key] = silhouette

def _plot_all_embeddings(dataset: PopariDataset, embedding_key: str = "X", column_names: Optional[str] = None, **spatial_kwargs):
    r"""Plot all laerned metagenes in-situ across all replicates.

    Each replicate's metagenes are contained in a separate plot.

    Args:
        trained_model: the trained Popari model.
        embedding_key: the key in the ``.obsm`` dataframe for the cell/spot embeddings.
        column_names: a list of the suffixes for each latent feature. If ``None``, it is assumed
            that these suffixes are just the indices of the latent features.
    """


    if column_names == None:
        column_names = [f"{embedding_key}_{index}" for index in range(trained_model.K)]
    size = 0.1 if "size" not in spatial_kwargs else spatial_kwargs.pop("size")
    palette = ListedColormap(sc.pl.palettes.godsnot_102) if "palette" not in spatial_kwargs else spatial_kwargs.pop("palette")
        
    axes = sq.pl.spatial_scatter(
        sq.pl.extract(dataset, embedding_key, prefix=f"{embedding_key}"),
        shape=None,
        color=column_names,
        size=size,
        wspace=0.2,
        ncols=2,
        **spatial_kwargs
    )

def _evaluate_classification_task(datasets: Sequence[PopariDataset], embeddings: str, labels: str, joint: bool):
    """

    """

    if joint:
        original_datasets = datasets
        dataset_names = [dataset.name for dataset in datasets]
        merged_dataset = ad.concat(datasets, label="batch", keys=dataset_names, merge="unique", uns_merge="unique", pairwise=True)
        datasets = [merged_dataset]

    for dataset in datasets:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(dataset.obs[labels].astype(str))
        dataset_embeddings = dataset.obsm[embeddings]

        X_train, X_valid, y_train, y_valid = train_test_split(dataset_embeddings, encoded_labels, train_size=0.25, random_state=42, stratify=encoded_labels)
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(X_train, y_train)

        df = []
        for split, X, y in [('train', X_train, y_train), ('validation', X_valid, y_valid)]:
            y_soft = model.predict_proba(X)
            y_hat = np.argmax(y_soft, 1)
            dataset.uns[f'microprecision_{split}'] = precision_score(y, y_hat, average='micro')
            dataset.uns[f'macroprecision_{split}'] = precision_score(y, y_hat, average='macro')
        
    if joint:
        indices = merged_dataset.obs.groupby("batch").indices.values()
        unmerged_datasets = [merged_dataset[index] for index in indices]
        for unmerged_dataset, original_dataset in zip(unmerged_datasets, original_datasets):
            for split in ("train", "validation"):
                original_dataset.uns[f'microprecision_{split}'] = unmerged_dataset.uns[f'microprecision_{split}']
                original_dataset.uns[f'macroprecision_{split}'] = unmerged_dataset.uns[f'macroprecision_{split}']

        return original_datasets, merged_dataset

    return datasets

def _compute_confusion_matrix(dataset: PopariDataset, labels: str, predictions: str, result_key: str = "confusion_matrix"):
    r"""Compute confusion matrix for labels and predictions.

    Useful for visualizing clustering validity.

    Args:
        trained_model: the trained Popari model.
        labels: the key in the ``.obs`` dataframe for the label data.
        predictions: the key in the ``.obs`` dataframe for the predictions data.
        result_key: the key in the ``.uns`` dictionary where the reordered confusion matrix will be stored.
    """

    unique_labels = sorted(dataset.obs[labels].unique())
    unique_predictions = sorted(dataset.obs[predictions].unique())
    if len(unique_labels) != len(unique_predictions):
        raise ValueError("Number of unique labels and unique predictions must be equal.")

    encoded_labels = [unique_labels.index(label) for label in dataset.obs[labels].values]
    encoded_predictions = [unique_predictions.index(prediction) for prediction in dataset.obs[predictions].values]

    confusion_output = confusion_matrix(encoded_labels, encoded_predictions)

    permutation, index = calcPermutation(confusion_output)
    dataset.obs[f'{labels}_inferred'] = [unique_labels[permutation[prediction]] for prediction in encoded_predictions]

    reordered_confusion = confusion_matrix(dataset.obs[labels], dataset.obs[f"{labels}_inferred"])[:len(unique_labels)]

    dataset.uns[result_key] = reordered_confusion

def calcPermutation(confusion_output):
    """
    TODO: document
    maximum weight bipartite matching
    :param confusion_output:
    :return: confusion_output[perm, index], where index is sorted
    """

    num_label_classes, num_prediction_classes = confusion_output.shape

    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from([("label", i) for i in range(num_label_classes)], bipartite=0)
    bipartite_graph.add_nodes_from([("prediction", i) for i in range(num_prediction_classes)], bipartite=1)

    bipartite_graph.add_edges_from([
        (("label", i), ("prediction", j), {'weight': confusion_output[i, j]})
        for i in range(num_label_classes) for j in range(num_prediction_classes)
    ])

    assert nx.is_bipartite(bipartite_graph)
    matching = nx.max_weight_matching(bipartite_graph, maxcardinality=True)
    assert len(set(__ for _ in matching for __ in _)) == num_label_classes * 2

    matching = [sorted(match, key=lambda node_attributes: node_attributes[0]) for match in matching]

    matching = [tuple(index for (_, index) in match) for match in matching]
    matching = sorted(matching, key=lambda pair: pair[1])

    perm, index = tuple(map(np.array, zip(*matching)))

    return perm, index

def _compute_columnwise_autocorrelation(dataset: PopariDataset, uns:str = "ground_truth_M", result_key: str = "ground_truth_M_correlation"):

    matrix  = dataset.uns[uns][f"{dataset.name}"].T

    num_columns, _= matrix.shape
    correlation_coefficient_matrix = np.corrcoef(matrix, matrix)[:num_columns, :num_columns]
    dataset.uns[result_key] = correlation_coefficient_matrix

def _plot_confusion_matrix(dataset: PopariDataset, labels: str, confusion_matrix_key: str = "confusion_matrix"):

    ordered_labels = sorted(dataset.obs[labels].unique())
    sns.heatmap(dataset.uns[confusion_matrix_key], xticklabels=ordered_labels, yticklabels=ordered_labels, annot=True)
    plt.show()

def _compute_spatial_correlation(dataset: PopariDataset, spatial_key: str = "Sigma_x_inv", metagene_key: str = "M", spatial_correlation_key: str = "spatial_correlation", neighbor_interactions_key: str = "neighbor_interactions"):
    """Computes spatial gene correlation according to learned metagenes.

    """

    spatial_affinity_matrix = dataset.uns[spatial_key][f"{dataset.name}"]
    metagenes = dataset.uns[metagene_key][f"{dataset.name}"]

    neighbor_interactions = metagenes @ spatial_affinity_matrix
    spatial_correlation = neighbor_interactions @ metagenes.T

    dataset.uns[spatial_correlation_key] = spatial_correlation
    dataset.uns[neighbor_interactions_key] = neighbor_interactions
