from typing import Sequence, Tuple

from popari.util import print_datetime
import torch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA

import itertools
from popari.components import PopariDataset
from popari._dataset_utils import _cluster, _pca

def initialize_kmeans(datasets: Sequence[PopariDataset], K: int, context: dict, kwargs_kmeans:dict) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
    """Initialize metagenes and hidden states using k-means clustering.

    Args:
        datasets: input ST replicates to use for initialization
        K: dimension of latent states for cell embeddings
        context: context to use for creating PyTorch tensors
        kwargs_kmeans: parameters to pass to KMeans classifier

    Returns:
        A tuple (M, Xs), where M is the initial estimate of the metagene
        values and Xs is the list of initial estimates of the hidden states
        of each replicate.

    """
    assert 'random_state' in kwargs_kmeans
    Ns, Gs = zip(*[dataset.X.shape for dataset in datasets])
    Ys = [dataset.X for dataset in datasets]
    Y_cat = np.concatenate(Ys, axis=0)
    pca = PCA(n_components=20)
    # pca = None
    Y_cat_reduced = Y_cat if pca is None else pca.fit_transform(Y_cat)
    kmeans = KMeans(n_clusters=K, **kwargs_kmeans)
    label = kmeans.fit_predict(Y_cat_reduced)
    M = np.stack([Y_cat[label == l].mean(0) for l in np.unique(label)]).T
    # M = kmeans.cluster_centers_.T
    Xs = []
    for N, Y in zip(Ns, Ys):
        Y_reduced = Y if pca is None else pca.transform(Y)
        label = kmeans.predict(Y_reduced)
        X = np.full([N, K], 1e-10)
        X[(range(N), label)] = 1

        Xs.append(X)
    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]

    return M, Xs

def initialize_louvain(datasets: Sequence[PopariDataset], K: int, context: dict, kwargs_louvain:dict, n_neighbors: int = 20, n_components: int = 50, eps: float = 1e-10) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
    """Initialize metagenes and hidden states using k-means clustering.

    Args:
        datasets: input ST replicates to use for initialization
        K: dimension of latent states for cell embeddings
        context: context to use for creating PyTorch tensors
        kwargs_kmeans: parameters to pass to KMeans classifier

    Returns:
        A tuple (M, Xs), where M is the initial estimate of the metagene
        values and Xs is the list of initial estimates of the hidden states
        of each replicate.

    """
    
    assert 'random_state' in kwargs_louvain

    _, merged_dataset = _pca(datasets, n_comps=n_components, joint=True)

    # Y_cat_reduced = Y_cat if pca is None else pca.fit_transform(Y_cat)

    while True:
        _cluster([merged_dataset], method="louvain", use_rep="X_pca", target_clusters=K, n_neighbors=n_neighbors)

        labels = merged_dataset.obs["louvain"].astype(int)
        num_clusters = len(labels.unique())
        if num_clusters == K:
            break

        n_neighbors = int(n_neighbors * 1.5)

    # Initialize based on clustering
    indices = merged_dataset.obs.groupby("batch").indices.values()
    unmerged_datasets = [merged_dataset[index] for index in indices]
    unmerged_labels = [unmerged_dataset.obs["louvain"].astype(int).values for unmerged_dataset in unmerged_datasets]
    
    M = np.stack([merged_dataset[labels == cluster].X.mean(axis=0) for cluster in labels.unique()]).T

    Xs = []
    for unmerged_label in unmerged_labels:
        N = len(unmerged_label)
        X = np.full([N, K], eps)
        X[np.arange(N), unmerged_label] = 1
        Xs.append(X)

    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]

    return M, Xs
    
def initialize_svd(datasets: Sequence[PopariDataset], K: int, context: dict, M_nonneg: bool = True, X_nonneg: bool = True) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
    """Initialize metagenes and hidden states using SVD.

    Args:
        datasets: input ST replicates to use for initialization
        K: dimension of latent states for cell embeddings
        context: context to use for creating PyTorch tensors
        M_nonneg: if specified, initial M estimate will contain only non-negative values
        X_nonneg: if specified, initial X estimate will contain only non-negative values

    Returns:
        A tuple (M, Xs), where M is the initial estimate of the metagene
        values and Xs is the list of initial estimates of the hidden states
        of each replicate.
    """

    # TODO: add check that number of genes is the same for all datasets
    Y_cat = np.concatenate([dataset.X for dataset in datasets], axis=0)

    svd = TruncatedSVD(K)
    X_cat = svd.fit_transform(Y_cat)
    M = svd.components_.T
    norm_p = np.ones([1, K])
    norm_n = np.ones([1, K])

    if M_nonneg:
        M_positive = np.clip(M, a_min=0, a_max=None)
        M_negative = np.clip(M, a_min=None, a_max=0)
        norm_p *= np.linalg.norm(M_positive, axis=0, ord=1, keepdims=True)
        norm_n *= np.linalg.norm(M_negative, axis=0, ord=1, keepdims=True)

    if X_nonneg:
        X_cat_positive = np.clip(X_cat, a_min=0, a_max=None)
        X_cat_negative = np.clip(X_cat, a_min=None, a_max=0)
        norm_p *= np.linalg.norm(X_cat_positive, axis=0, ord=1, keepdims=True)
        norm_n *= np.linalg.norm(X_cat_negative, axis=0, ord=1, keepdims=True)

    # Since M must be non-negative, choose the_value that yields greater L1-norm
    sign = np.where(norm_p >= norm_n, 1., -1.)
    M *= sign
    X_cat *= sign
    X_cat_iter = X_cat
    if M_nonneg:
        M = np.clip(M, a_min=1e-10, a_max=None)
        
    Xs = []
    for dataset in datasets:
        Y = dataset.X
        N = len(dataset)

        X = X_cat_iter[:N]
        X_cat_iter = X_cat_iter[N:]
        if X_nonneg:
            # fill negative elements by zero
            # X = np.clip(X, a_min=1e-10, a_max=None)
            # fill negative elements by the average of nonnegative elements
            for x in X.T:
                idx = x < 1e-10
                # Bugfix below: if statement necessary, otherwise nan elements may be introduced...
                if len(x[~idx]) > 0:
                    x[idx] = x[~idx].mean()
        else:
            X = np.full([N, K], 1/K)
        Xs.append(X)

    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]

    return M, Xs
