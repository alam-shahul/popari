import itertools
from typing import Sequence, Tuple

import anndata as ad
import numpy as np
import torch
from scipy import sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD

from popari._dataset_utils import _cluster, _pca
from popari.util import concatenate


def initialize_kmeans(
    datasets: Sequence[ad.AnnData],
    K: int,
    context: dict,
    kwargs_kmeans: dict,
) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
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
    assert "random_state" in kwargs_kmeans
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


def initialize_leiden(
    datasets: Sequence[ad.AnnData],
    K: int,
    context: dict,
    kwargs_leiden: dict,
    n_neighbors: int = 20,
    n_components: int = 50,
    eps: float = 1e-10,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
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

    assert "random_state" in kwargs_leiden

    _pca(datasets, n_comps=n_components, joint=True)

    # Y_cat_reduced = Y_cat if pca is None else pca.fit_transform(Y_cat)

    while True:
        _cluster(
            datasets,
            joint=True,
            method="leiden",
            use_rep="X_pca",
            target_clusters=K,
            n_neighbors=n_neighbors,
            verbose=verbose,
            **kwargs_leiden,
        )

        merged_dataset = concatenate(datasets)
        labels = merged_dataset.obs["leiden"].astype(int)
        num_clusters = len(labels.unique())
        if num_clusters == K:
            break

        n_neighbors = int(n_neighbors * 1.5)

    # Initialize based on clustering
    indices = merged_dataset.obs.groupby("batch").indices.values()
    unmerged_datasets = [merged_dataset[index] for index in indices]
    unmerged_labels = [unmerged_dataset.obs["leiden"].astype(int).values for unmerged_dataset in unmerged_datasets]

    M = np.stack([merged_dataset[labels == cluster].X.mean(axis=0) for cluster in labels.unique()]).T

    Xs = []
    for unmerged_label in unmerged_labels:
        unique_labels = np.unique(unmerged_label)
        if len(unique_labels) == 1:
            raise ValueError(
                "All spots from a replicate are being assigned to a single cluster "
                "during Louvain initialization. This indicates that there is a "
                "noticeable batch effect. This may be due to 1) too few spots, "
                "2) too few metagenes (i.e. clusters), or a truly significant batch "
                "effect. Try addressing these issues, or switch to a different "
                "initialization method.",
            )
        N = len(unmerged_label)
        X = np.full([N, K], eps)
        X[np.arange(N), unmerged_label] = 1
        Xs.append(X)

    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]

    return M, Xs


def initialize_ground_truth(
    datasets: Sequence[ad.AnnData],
    K: int,
    context: dict,
    label_key: str = "cell_type",
    random_state: int = 0,
    eps: float = 1e-10,
    absent_class_embedding_scale: float = 0.05,
) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
    """Initialize metagenes and hidden states from known labels.

    This follows the same clustering-based initialization idea as Leiden, but
    uses an existing ``.obs`` column as the cluster assignment. Factor order is
    determined by categorical order when available, otherwise by sorted label
    values.

    """

    for dataset in datasets:
        if label_key not in dataset.obs:
            raise KeyError(f"Ground-truth initialization requires dataset.obs[{label_key!r}].")

    first_dataset = datasets[0]
    first_labels = first_dataset.obs[label_key]
    if (
        "cell_type_definitions" in first_dataset.uns
        and first_dataset.name in first_dataset.uns["cell_type_definitions"]
    ):
        labels = list(first_dataset.uns["cell_type_definitions"][first_dataset.name])
    elif str(first_labels.dtype) == "category":
        labels = list(first_labels.cat.categories)
    else:
        labels = sorted(set(np.concatenate([dataset.obs[label_key].dropna().unique() for dataset in datasets])))

    if len(labels) != K:
        raise ValueError(
            f"Ground-truth initialization found {len(labels)} labels in obs[{label_key!r}], "
            f"but Popari was configured with K={K}.",
        )

    label_to_index = {label: index for index, label in enumerate(labels)}
    Ys = [dataset.X.toarray() if sp.issparse(dataset.X) else np.asarray(dataset.X) for dataset in datasets]
    labels_cat = np.concatenate([dataset.obs[label_key].to_numpy() for dataset in datasets])
    Y_cat = np.concatenate(Ys, axis=0)
    rng = np.random.default_rng(random_state)
    absent_label_indices = [label_to_index[label] for label in labels if not np.any(labels_cat == label)]

    metagenes = []
    for label in labels:
        label_mask = labels_cat == label
        if np.any(label_mask):
            metagene = Y_cat[label_mask].mean(axis=0)
        else:
            metagene = rng.random(Y_cat.shape[1])
        metagenes.append(metagene)

    M = np.stack(metagenes).T

    Xs = []
    for dataset in datasets:
        dataset_labels = dataset.obs[label_key].to_numpy()
        unknown_labels = set(dataset_labels) - set(label_to_index)
        if unknown_labels:
            raise ValueError(f"Found labels not included in initialization order: {sorted(unknown_labels)}")

        N = len(dataset_labels)
        X = rng.random((N, K)) * eps
        if absent_label_indices:
            X[:, absent_label_indices] = rng.random((N, len(absent_label_indices))) * absent_class_embedding_scale
        X[np.arange(N), [label_to_index[label] for label in dataset_labels]] = 1
        Xs.append(X)

    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]

    return M, Xs


def initialize_svd(
    datasets: Sequence[ad.AnnData],
    K: int,
    context: dict,
    M_nonneg: bool = True,
    X_nonneg: bool = True,
) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
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
    Y_cat = sp.vstack([dataset.X for dataset in datasets])

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
    sign = np.where(norm_p >= norm_n, 1.0, -1.0)
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
            X = np.full([N, K], 1 / K)
        Xs.append(X)

    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]

    return M, Xs


def initialize_dummy(
    datasets: Sequence[ad.AnnData],
    K: int,
    context: dict,
) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
    """Initialize metagenes and hidden states with random values.

    Internal method used simply for code

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

    first_dataset = datasets[0]
    _, num_genes = first_dataset.shape

    M = torch.rand((num_genes, K), **context)
    Xs = [torch.rand((dataset.shape[0], K), **context) for dataset in datasets]

    return M, Xs
