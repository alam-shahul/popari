import sys, logging, time, gc, os, itertools
from multiprocessing import Pool
from util import print_datetime
import torch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD, PCA

def initialize_kmeans(K, Ys, kwargs_kmeans, context):
    assert 'random_state' in kwargs_kmeans
    Ns, Gs = zip(*[Y.shape for Y in Ys])
    GG = max(Gs)
    repli_valid = np.array(Gs) == GG
    Ys = [Y.cpu().numpy() for Y in Ys]
    Y_cat = np.concatenate(list(itertools.compress(Ys, repli_valid)), axis=0)
    pca = PCA(n_components=20)
    # pca = None
    Y_cat_reduced = Y_cat if pca is None else pca.fit_transform(Y_cat)
    kmeans = KMeans(n_clusters=K, **kwargs_kmeans)
    label = kmeans.fit_predict(Y_cat_reduced)
    M = np.stack([Y_cat[label == l].mean(0) for l in np.unique(label)]).T
    # M = kmeans.cluster_centers_.T
    Xs = []
    for is_valid, N, Y in zip(repli_valid, Ns, Ys):
        if is_valid:
            Y_reduced = Y if pca is None else pca.transform(Y)
            label = kmeans.predict(Y_reduced)
            X = np.full([N, K], 1e-10)
            X[(range(N), label)] = 1
        else:
            X = np.full([N, K], 1/K)
        Xs.append(X)
    M = torch.tensor(M, **context)
    Xs = [torch.tensor(X, **context) for X in Xs]
    return M, Xs


def initialize_svd(K, Ys, context, M_nonneg=True, X_nonneg=True, random_state=0):
    """Initialize metagenes and hidden states using SVD.

    Args:
        K (int): number of metagenes
        Ys (list of numpy.ndarray): list of gene expression data for each FOV.
            Note that currently all Ys must have the same number of genes;
            otherwise they are simply absent from the analysis.

    Returns:
        A tuple of (M, Xs), where M is the initial estimate of the metagene
        values and Xs is the list of initial estimates of the hidden states
        of the gene expression data.
    """

    Ns, Gs = zip(*[Y.shape for Y in Ys])
    max_genes = max(Gs)
    repli_valid = (np.array(Gs) == max_genes)
    Ys = [Y.cpu().numpy() for Y in Ys]
    Y_cat = np.concatenate(list(itertools.compress(Ys, repli_valid)), axis=0)

    svd = TruncatedSVD(K, random_state=0)
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
    for is_valid, N, Y in zip(repli_valid, Ns, Ys):
        if is_valid:
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

def initialize_Sigma_x_inv(K, betas, context, datasets, scaling=10):
    """Initialize Sigma_x_inv using the empirical correlation between hidden state metagenes.

    Args:
        K (int): number of metagenes
        datasets (list of SpiceMixDataset): list of datasets for each FOV
        Es (list of lists): list of adjacency lists for each FOV
        betas (list of integers): list of weights, indicating relevance of each FOV
        scaling (int): scale of Sigma_x_inv values

    Returns:
        Initial estimates of Sigma_x_invs
    """

    num_replicates = len(datasets)
    Sigma_x_invs = torch.zeros([num_replicates, K, K], **context)
    for replicate, (beta, dataset) in enumerate(zip(betas, datasets)):
        adjacency_list = dataset.obs["adjacency_list"]
        X = dataset.obsm["X"]
        Z = X / torch.linalg.norm(X, dim=1, keepdim=True, ord=1)
        edges = np.array([(i, j) for i, e in enumerate(adjacency_list) for j in e])

        x = Z[edges[:, 0]]
        y = Z[edges[:, 1]]
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        y_std = y.std(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True)
        corr = (y / y_std).T @ (x / x_std) / len(x)
        Sigma_x_invs[replicate] = -beta * corr

    # Symmetrizing and zero-centering Sigma_x_inv
    Sigma_x_invs = (Sigma_x_invs + torch.transpose(Sigma_x_invs, 1, 2)) / 2
    Sigma_x_invs -= Sigma_x_invs.mean(dim=(1, 2), keepdims=True)
    Sigma_x_invs *= scaling
    
    return Sigma_x_invs
