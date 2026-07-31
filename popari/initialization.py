"""Initialization strategies for unified Popari AnnData objects."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD

from popari._sample_axis import SampleAxis
from popari.analysis import cluster
from popari.preprocessing import pca


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def initialize_kmeans(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    K: int,
    context: dict,
    kwargs_kmeans: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize metagenes and global embeddings using k-means."""

    if "random_state" not in kwargs_kmeans:
        raise ValueError("kwargs_kmeans must specify random_state.")

    expression = _dense(adata.X)
    n_components = min(20, expression.shape[0], expression.shape[1])
    reducer = PCA(n_components=n_components)
    reduced = reducer.fit_transform(expression)
    kmeans = KMeans(n_clusters=K, **kwargs_kmeans)
    labels = kmeans.fit_predict(reduced)

    metagenes = np.stack([expression[labels == label].mean(axis=0) for label in range(K)]).T
    embeddings = np.full((adata.n_obs, K), 1e-10)
    embeddings[np.arange(adata.n_obs), labels] = 1
    return torch.tensor(metagenes, **context), torch.tensor(embeddings, **context)


def initialize_leiden(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    K: int,
    context: dict,
    kwargs_leiden: dict,
    n_neighbors: int = 20,
    n_components: int = 50,
    eps: float = 1e-10,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize metagenes and global embeddings using Leiden clusters."""

    if "random_state" not in kwargs_leiden:
        raise ValueError("kwargs_leiden must specify random_state.")

    pca(adata, n_comps=min(n_components, adata.n_obs - 1, adata.n_vars - 1))
    while True:
        cluster(
            adata,
            method="leiden",
            use_rep="X_pca",
            target_clusters=K,
            n_neighbors=n_neighbors,
            verbose=verbose,
            **kwargs_leiden,
        )
        labels = adata.obs["leiden"].astype(int).to_numpy()
        if len(np.unique(labels)) == K:
            break
        n_neighbors = int(n_neighbors * 1.5)

    for sample in sample_axis.names:
        if len(np.unique(labels[sample_axis.indices(sample)])) == 1:
            raise ValueError(
                "All spots from a sample were assigned to one cluster during "
                "Leiden initialization. Try a different K or initialization method.",
            )

    metagenes = np.stack(
        [np.asarray(adata[labels == label].X.mean(axis=0)).ravel() for label in range(K)],
    ).T
    embeddings = np.full((adata.n_obs, K), eps)
    embeddings[np.arange(adata.n_obs), labels] = 1
    return torch.tensor(metagenes, **context), torch.tensor(embeddings, **context)


def initialize_ground_truth(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    K: int,
    context: dict,
    label_key: str = "cell_type",
    random_state: int = 0,
    eps: float = 1e-10,
    absent_class_embedding_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize metagenes and global embeddings from simulation truth or
    labels."""

    rng = np.random.default_rng(random_state)
    if "ground_truth_X" in adata.obsm and "ground_truth_M" in adata.uns:
        metagenes = np.asarray(adata.uns["ground_truth_M"])
        embeddings = np.asarray(adata.obsm["ground_truth_X"]).copy()
        if metagenes.shape[1] != K:
            raise ValueError(
                f"ground_truth_M has {metagenes.shape[1]} factors, but Popari was configured with K={K}.",
            )
        if embeddings.shape != (adata.n_obs, K):
            raise ValueError(
                f"ground_truth_X has shape {embeddings.shape}; expected {(adata.n_obs, K)}.",
            )

        for sample in sample_axis.names:
            indices = sample_axis.indices(sample)
            sample_embeddings = embeddings[indices]
            absent = np.flatnonzero(np.isclose(sample_embeddings.sum(axis=0), 0))
            if len(absent):
                sample_embeddings[:, absent] = rng.random((len(indices), len(absent))) * absent_class_embedding_scale
                embeddings[indices] = sample_embeddings
        return torch.tensor(metagenes, **context), torch.tensor(embeddings, **context)

    if label_key not in adata.obs:
        raise KeyError(f"Ground-truth initialization requires adata.obs[{label_key!r}].")

    observed_labels = adata.obs[label_key]
    definitions = adata.uns.get("cell_type_definitions")
    if isinstance(definitions, dict) and sample_axis.names[0] in definitions:
        labels = list(definitions[sample_axis.names[0]])
    elif isinstance(observed_labels.dtype, pd.CategoricalDtype):
        labels = list(observed_labels.cat.categories)
    else:
        labels = sorted(observed_labels.dropna().unique())

    if len(labels) != K:
        raise ValueError(
            f"Ground-truth initialization found {len(labels)} labels in obs[{label_key!r}], "
            f"but Popari was configured with K={K}.",
        )

    label_to_index = {label: index for index, label in enumerate(labels)}
    label_values = observed_labels.to_numpy()
    unknown = set(label_values) - set(label_to_index)
    if unknown:
        raise ValueError(f"Found labels not included in initialization order: {sorted(unknown)}")

    expression = _dense(adata.X)
    metagenes = []
    for label in labels:
        mask = label_values == label
        metagenes.append(expression[mask].mean(axis=0) if mask.any() else rng.random(adata.n_vars))
    metagenes = np.stack(metagenes).T

    embeddings = rng.random((adata.n_obs, K)) * eps
    embeddings[np.arange(adata.n_obs), [label_to_index[label] for label in label_values]] = 1
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_labels = label_values[indices]
        absent = [label_to_index[label] for label in labels if not np.any(sample_labels == label)]
        if absent:
            embeddings[np.ix_(indices, absent)] = rng.random((len(indices), len(absent))) * absent_class_embedding_scale

    return torch.tensor(metagenes, **context), torch.tensor(embeddings, **context)


def initialize_svd(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    K: int,
    context: dict,
    M_nonneg: bool = True,
    X_nonneg: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize metagenes and global embeddings using truncated SVD."""

    svd = TruncatedSVD(K)
    embeddings = svd.fit_transform(adata.X)
    metagenes = svd.components_.T
    positive_norm = np.ones((1, K))
    negative_norm = np.ones((1, K))

    if M_nonneg:
        positive_norm *= np.linalg.norm(np.clip(metagenes, 0, None), axis=0, ord=1, keepdims=True)
        negative_norm *= np.linalg.norm(np.clip(metagenes, None, 0), axis=0, ord=1, keepdims=True)
    if X_nonneg:
        positive_norm *= np.linalg.norm(np.clip(embeddings, 0, None), axis=0, ord=1, keepdims=True)
        negative_norm *= np.linalg.norm(np.clip(embeddings, None, 0), axis=0, ord=1, keepdims=True)

    sign = np.where(positive_norm >= negative_norm, 1.0, -1.0)
    metagenes *= sign
    embeddings *= sign
    if M_nonneg:
        metagenes = np.clip(metagenes, 1e-10, None)
    if X_nonneg:
        for sample in sample_axis.names:
            sample_embeddings = embeddings[sample_axis.indices(sample)]
            for component in sample_embeddings.T:
                negative = component < 1e-10
                if np.any(~negative):
                    component[negative] = component[~negative].mean()
            embeddings[sample_axis.indices(sample)] = sample_embeddings
    else:
        embeddings = np.full((adata.n_obs, K), 1 / K)

    return torch.tensor(metagenes, **context), torch.tensor(embeddings, **context)


def initialize_dummy(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    K: int,
    context: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize metagenes and global embeddings with random values."""

    return (
        torch.rand((adata.n_vars, K), **context),
        torch.rand((adata.n_obs, K), **context),
    )
