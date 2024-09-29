import datetime
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Optional

import matplotlib
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from matplotlib import pyplot as plt
from ortools.graph.python import min_cost_flow
from scipy.sparse import csr_array, csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange
from umap import UMAP

from popari.sample_for_integral import integrate_of_exponential_over_simplex


def create_neighbor_groups(replicate_names, covariate_values, window_size=1):
    if window_size == None:
        return None

    unique_covariates = np.unique(covariate_values)
    covariate_to_replicate_name = defaultdict(list)
    for covariate_value, replicate_name in zip(covariate_values, replicate_names):
        covariate_to_replicate_name[covariate_value].append(replicate_name)

    sort_indices = np.argsort(unique_covariates)
    sorted_covariates = unique_covariates[sort_indices]

    groups = {}
    for index in range(window_size, len(sorted_covariates) - window_size):
        group_covariates = sorted_covariates[index - window_size : index + window_size + 1]
        group_replicates = sum(
            [covariate_to_replicate_name[group_covariate] for group_covariate in group_covariates],
            [],
        )
        group_name = f"{sorted_covariates[index]}"

        groups[group_name] = list(group_replicates)

    return groups


def calc_modularity(adjacency_matrix, label, resolution=1):
    adjacency_matrix = adjacency_matrix.tocoo()
    n = adjacency_matrix.shape[0]
    adjacency_matrix_sum = adjacency_matrix.data.sum()
    score = (
        adjacency_matrix.data[label[adjacency_matrix.row] == label[adjacency_matrix.col]].sum() / adjacency_matrix_sum
    )

    idx = np.argsort(label)
    label = label[idx]
    k = np.array(adjacency_matrix.sum(0)).ravel() / adjacency_matrix_sum
    k = k[idx]
    idx = np.concatenate([[0], np.nonzero(label[:-1] != label[1:])[0] + 1, [len(label)]])
    score -= sum(k[i:j].sum() ** 2 for i, j in zip(idx[:-1], idx[1:])) * resolution
    return score


def evaluate_embedding(obj, embedding="X", do_plot=True, do_sil=True):
    if embedding == "X":
        Xs = [X.cpu().numpy() for X in obj.Xs]
    elif embedding in obj.phenotype_predictors:
        Xs = [obj.phenotype_predictors["cell type encoded"][0](X).cpu().numpy() for X in obj.Xs]
    else:
        raise NotImplementedError

    # x = np.concatenate(Xs, axis=0)

    for x, replicate, dataset in zip(Xs, obj.repli_list, obj.datasets):
        x = StandardScaler().fit_transform(x)
        for n_clust in [6, 7, 8, 9]:
            y = AgglomerativeClustering(
                n_clusters=n_clust,
                linkage="ward",
            ).fit_predict(x)
            y = pd.Categorical(y, categories=np.unique(y))
            dataset.obs["spicemixplus_label"] = y

            print(
                f"replicate {replicate}"
                f"hierarchical w/ K={n_clust}."
                f"ARI = {adjusted_rand_score(dataset.obs['cell_type'].values, y):.2f}",
                sep=" ",
            )
        # y = clustering_louvain_nclust(
        #     x.copy(), 8,
        #     kwargs_neighbors=dict(n_neighbors=10),
        #     kwargs_clustering=dict(),
        #     resolution_boundaries=(.1, 1.),
        # )

    # obj.meta['label SpiceMixPlus'] = y
    # print('ari = {:.2f}'.format(adjusted_rand_score(*obj.meta[['cell type', 'label SpiceMixPlus']].values.T)))
    # for repli, df in obj.meta.groupby('repli'):
    #     print('ari {} = {:.8f}'.format(repli, adjusted_rand_score(*df[['cell type', 'label SpiceMixPlus']].values.T)))
    if do_plot:
        ncol = 4
        nrow = (obj.num_repli + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
        for ax, replicate, dataset in zip(axes.flat, obj.repli_list, obj.datasets):
            sns.heatmap(
                dataset.obs.groupby(["cell_type", "spicemixplus_label"]).size().unstack().fillna(0).astype(int),
                ax=ax,
                annot=True,
                fmt="d",
            )
        plt.show()
        plt.close()

    if do_sil:
        for x, replicate, dataset in zip(Xs, obj.repli_list, obj.datasets):
            x = StandardScaler().fit_transform(x)
            x = UMAP(
                random_state=obj.random_state,
                n_neighbors=10,
            ).fit_transform(x)
            print("sil", silhouette_score(x, dataset.obs["cell_type"], random_state=obj.random_state))
            if do_plot:
                keys = ["cell_type", "replicate", "spicemixplus_label"]
                ncol = len(keys)
                nrow = 1 + obj.num_repli
                fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))

                def plot(axes, idx):
                    for ax, key in zip(axes, keys):
                        sns.scatterplot(ax=ax, data=dataset.obs.iloc[idx], x=x[idx, 0], y=x[idx, 1], hue=key, s=5)

                plot(axes[0], slice(None))
                for ax_row, (repli, df) in zip(axes[1:], dataset.obs.reset_index().groupby("replicate")):
                    plot(ax_row, df.index)
                plt.show()
                plt.close()


def evaluate_prediction_wrapper(obj, *, key_truth="cell type encoded", display_fn=print, **kwargs):
    X_all = torch.cat([X for X in obj.Xs], axis=0)
    obj.meta["label SpiceMixPlus predictor"] = obj.phenotype_predictors[key_truth][0](X_all).argmax(1).cpu().numpy()
    display_fn(evaluate_prediction(obj.meta, key_pred="label SpiceMixPlus predictor", key_truth=key_truth, **kwargs))


def evaluate_prediction(df_meta, *, key_pred="label SpiceMixPlus", key_truth="cell type", key_repli="repli"):
    df_score = {}
    for repli, df in df_meta.groupby(key_repli):
        t = tuple(df[[key_truth, key_pred]].values.T)
        r = {
            "acc": accuracy_score(*t),
            "f1 micro": f1_score(*t, average="micro"),
            "f1 macro": f1_score(*t, average="macro"),
            "ari": adjusted_rand_score(*t),
        }
        df_score[repli] = r
    df_score = pd.DataFrame(df_score).T
    return df_score


class NesterovGD:
    """Optimizer that implements Nesterov's Accelerated Gradient Descent.

    See below for implementation details:
    # https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

    Attributes:
        parameters: object to optimize with Nesterov
        step_size: size of gradient update

    """

    def __init__(self, parameters: torch.Tensor, step_size: float):
        """Initialize Nesterov optimization.

        Args:
            parameters: object to optimize with Nesterov
            step_size: size of gradient update

        """
        self.parameters = parameters
        self.step_size = step_size
        # self.y = x.clone()
        self.y = torch.zeros_like(parameters)
        # self.lam = 0
        self.k = 0

    def set_parameters(self, parameters):
        """Reset parameters.

        Args:
            parameters: object to optimize with Nesterov

        """
        self.parameters = parameters

    def step(self, grad: torch.Tensor):
        """Update parameters according to state and step size.

        Args:
            grad: naive gradient for updating parameters

        """

        # method 1
        # lam_new = (1 + np.sqrt(1 + 4 * self.lam ** 2)) / 2
        # gamma = (1 - self.lam) / lam_new
        # method 2
        self.k += 1
        gamma = -(self.k - 1) / (self.k + 2)
        # method 3 - GD
        # gamma = 0
        # y_new = self.x.sub(grad, alpha=self.step_size)
        y_new = self.parameters - grad * self.step_size  # use addcmul
        self.y = (self.y * gamma).add(y_new, alpha=(1 - gamma))
        self.parameters = self.y
        # self.lam = lam_new
        self.y = y_new

        return self.parameters


def print_datetime():
    return datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S]\t")


def get_datetime():
    return datetime.datetime.now().strftime("[%Y/%m/%d %H:%M:%S]\t")


def getRank(m, thr=0):
    rank = np.empty(m.shape, dtype=int)
    for r, a in zip(rank, m):
        r[np.argsort(a)] = np.arange(len(r))
        mask = a < thr
        r[mask] = np.mean(r[mask])
    return rank


@torch.no_grad()
def project_M(M, M_constraint):
    result = M.clone()
    if M_constraint == "simplex":
        result = project2simplex(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == "unit sphere":
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == "nonneg unit sphere":
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


def project_M_(M, M_constraint):
    result = M.clone()
    if M_constraint == "simplex":
        result = project2simplex_(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == "unit sphere":
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == "nonneg unit sphere":
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result


def project2simplex(y, dim: int = 0, zero_threshold: float = 1e-10) -> torch.Tensor:
    """Projects a matrix such that the columns (or rows) lie on the unit
    simplex.

    See https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    for a reference.

    The goal is to find a scalar mu such that || (y-mu)_+ ||_1 = 1

    Currently uses Newton's method to optimize || y - mu ||^2

    TODO: try implementing it this way instead: https://arxiv.org/pdf/1101.6081.pdf

    Args:
        y: list of vectors to be projected to unit simplex
        dim: dimension along which to project
        zero_threshold: threshold to treat as zero for numerical stability purposes

    """

    num_components = y.shape[dim]

    mu = (y.sum(dim=dim, keepdim=True) - 1) / num_components
    previous_derivative = derivative = None
    for _ in range(num_components):
        difference = y - mu
        derivative = -(difference > zero_threshold).sum(dim=dim, keepdim=True).to(y.dtype)
        assert -derivative.min() > 0, difference.clip(min=0).sum(dim).min()
        if previous_derivative is not None and (derivative == previous_derivative).all():
            break
        objective_value = torch.clip(difference, min=zero_threshold).sum(dim=dim, keepdim=True) - 1
        newton_update = objective_value / derivative
        mu -= newton_update

        previous_derivative = derivative
    assert (derivative == previous_derivative).all()

    assert not torch.isnan(y).any(), y

    y = (y - mu).clip(min=zero_threshold)
    assert not torch.isnan(y).any(), (mu, derivative)

    assert y.sum(dim=dim).sub_(1).abs_().max() < 1e-3, y.sum(dim=dim).sub_(1).abs_().max()

    return y


def project2simplex_(y, dim: int = 0, zero_threshold: float = 1e-10) -> torch.Tensor:
    """(In-place) Projects a matrix such that the columns (or rows) lie on the
    unit simplex.

    See https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
    for a reference.

    The goal is to find a scalar mu such that || (y-mu)_+ ||_1 = 1

    Currently uses Newton's method to optimize || y - mu ||^2

    TODO: try implementing it this way instead: https://arxiv.org/pdf/1101.6081.pdf

    Args:
        y: list of vectors to be projected to unit simplex
        dim: dimension along which to project
        zero_threshold: threshold to treat as zero for numerical stability purposes

    """
    y_copy = y.clone()
    num_components = y.shape[dim]

    y_copy.sub_(y_copy.sum(dim=dim, keepdim=True).sub_(1), alpha=1 / num_components)
    mu = y_copy.max(dim=dim, keepdim=True)[0].div_(2)
    derivative_prev, derivative = None, None
    for _ in range(num_components):
        difference = y_copy.sub(mu)
        objective_value = difference.clip_(min=zero_threshold).sum(dim, keepdim=True).sub_(1)
        derivative = difference.gt_(zero_threshold).sum(dim, keepdim=True)

        if derivative_prev is not None and (derivative == derivative_prev).all():
            break

        mu.addcdiv_(objective_value, derivative)
        derivative_prev = derivative

    y_copy.sub_(mu).clip_(min=zero_threshold)
    assert y_copy.sum(dim=dim).sub_(1).abs_().max() < 1e-4, y_copy.sum(dim=dim).sub_(1).abs_().max()
    return y_copy


class IndependentSet:
    """Iterator class that yields a list of batch_size independent nodes from a
    spatial graph.

    For each iteration, no pair of yielded nodes can be neighbors of each other according to the
    adjacency matrix.

    Attributes:
        N: number of nodes in graph
        adjacency_list: graph neighbor information stored in adjacency list format
        batch_size: number of nodes to draw independently every iteration

    """

    def __init__(self, adjacency_list, device, batch_size=50):
        self.N = len(adjacency_list)
        self.adjacency_list = adjacency_list
        self.batch_size = batch_size
        self.indices_remaining = None
        self.device = device

    def __iter__(self):
        """Return iterator over nodes in graph.

        Resets indices_remaining before returning the iterator.

        """
        self.indices_remaining = set(range(self.N))
        return self

    def __next__(self):
        """Returns the indices of batch_size nodes such that none of them are
        neighbors of each other.

        Makes sure selected nodes are not adjacent to each other, i.e., finds an
        independent set of `valid indices` in a greedy manner

        """
        if len(self.indices_remaining) == 0:
            raise StopIteration

        valid_indices = sample_graph_iid(self.adjacency_list, self.indices_remaining, self.batch_size)
        self.indices_remaining -= set(valid_indices)

        return torch.tensor(valid_indices, device=self.device, dtype=torch.long)


def sample_graph_iid(adjacency_list, indices_remaining, sample_size):
    valid_indices = []
    excluded_indices = set()
    effective_batch_size = min(sample_size, len(indices_remaining))
    candidate_indices = np.random.choice(
        list(indices_remaining),
        size=effective_batch_size,
        replace=False,
    )
    for index in candidate_indices:
        if index not in excluded_indices:
            valid_indices.append(index)
            excluded_indices |= set(adjacency_list[index])

    return valid_indices


def convert_numpy_to_pytorch_sparse_coo(numpy_coo, context):
    indices = numpy_coo.nonzero()
    values = numpy_coo.data[numpy_coo.data.nonzero()]

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    size = numpy_coo.shape

    torch_coo = torch.sparse_coo_tensor(i, v, size=size, **context)

    return torch_coo


def compute_neighborhood_enrichment(features: np.ndarray, adjacency_matrix: csr_matrix):
    r"""Compute the normalized enrichment of features in direct neighbors on a
    graph.

    Args:
        features: attributes on the nodes of the graph on which to compute enrichment.
        adjacency_matrix: sparse graph representation.

    """

    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T).astype(bool).astype(adjacency_matrix.dtype)
    edges_per_node = np.squeeze(np.asarray(adjacency_matrix.sum(axis=0)))
    connected_mask = edges_per_node > 0

    features = features[connected_mask]
    adjacency_matrix = adjacency_matrix[connected_mask][:, connected_mask]

    total_counts = features.sum(axis=0)[:, np.newaxis]
    assert np.all(total_counts > 0)

    normalized_enrichment = ((1 / total_counts) * features.T) @ (
        1 / edges_per_node[connected_mask][:, np.newaxis] * adjacency_matrix.toarray() @ features
    )

    return np.asarray(normalized_enrichment)


def bin_expression(
    spot_expression: np.ndarray,
    spot_coordinates: np.ndarray,
    bin_coordinates: np.ndarray,
    num_jobs: int,
):
    """Bin spot expressions into filtered coordinates.

    Args:
        spot_expression: expression of original spots
        spot_coordinates: coordinates of original spots
        bin_coordinates: coordinates of downsampled bin spots

    Returns:
        A tuple of (binned expression for metaspots, assignment matrix of bins to spots)

    """

    num_spots, num_genes = spot_expression.shape
    num_bins, _ = bin_coordinates.shape

    bin_expression = np.zeros((num_bins, num_genes))
    bin_assignments = csr_array((num_bins, num_spots), dtype=np.int32)

    neigh = NearestNeighbors(n_neighbors=1, n_jobs=num_jobs)
    neigh.fit(bin_coordinates)
    indices = neigh.kneighbors(spot_coordinates, return_distance=False)

    bin_to_spots = defaultdict(list)
    for i in range(num_spots):
        bin_to_spots[indices[i].item()].append(i)

    for i in range(num_bins):
        bin_spots = bin_to_spots[i]
        bin_assignments[i, bin_spots] = 1
        bin_expression[i] = np.sum(spot_expression[bin_spots], axis=0)

    return bin_expression, bin_assignments


def compute_nll(model, level: int = 0, use_spatial=False):
    """Compute overall negative log-likelihood for current model parameters."""

    datasets = model.hierarchy[level].datasets
    parameter_optimizer = model.hierarchy[level].parameter_optimizer
    embedding_optimizer = model.hierarchy[level].embedding_optimizer
    Ys = model.hierarchy[level].Ys
    betas = model.hierarchy[level].betas

    with torch.no_grad():
        total_loss = torch.zeros(1, **model.context)
        if use_spatial:
            weighted_total_cells = 0
            for dataset in datasets:
                E_adjacency_list = embedding_optimizer.adjacency_lists[dataset.name]
                weighted_total_cells += sum(map(len, E_adjacency_list))

        for dataset_index, dataset in enumerate(datasets):
            sigma_yx = parameter_optimizer.sigma_yxs[dataset_index]
            Y = Ys[dataset_index].to(model.context["device"])
            X = embedding_optimizer.embedding_state[dataset.name].to(model.context["device"])
            M = parameter_optimizer.metagene_state[dataset.name].to(model.context["device"])
            prior_x_mode = parameter_optimizer.prior_x_modes[dataset_index]
            beta = betas[dataset_index]
            prior_x = parameter_optimizer.prior_xs[dataset_index]

            # Precomputing quantities
            MTM = M.T @ M / (sigma_yx**2)
            YM = Y.to(M.device) @ M / (sigma_yx**2)
            Ynorm = torch.linalg.norm(Y, ord="fro").item() ** 2 / (sigma_yx**2)
            S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)

            Z = X / S
            N, G = Y.shape

            loss = ((X @ MTM) * X).sum() / 2 - (X * YM).sum() + Ynorm / 2

            logZ_i_Y = torch.full((N,), G / 2 * np.log(2 * np.pi * sigma_yx**2), **model.context)
            if not use_spatial:
                logZ_i_X = torch.full((N,), 0, **model.context)
                if (prior_x[0] != 0).all():
                    logZ_i_X += torch.full((N,), model.K * torch.log(prior_x[0]).item(), **model.context)
                log_partition_function = (logZ_i_Y + logZ_i_X).sum()
            else:
                adjacency_matrix = embedding_optimizer.adjacency_matrices[dataset.name].to(model.context["device"])
                Sigma_x_inv = parameter_optimizer.spatial_affinity_state[dataset.name].to(model.context["device"])
                nu = adjacency_matrix @ Z
                eta = nu @ Sigma_x_inv
                logZ_i_s = torch.full((N,), 0, **model.context)
                if (prior_x[0] != 0).all():
                    logZ_i_s = torch.full(
                        (N,),
                        -model.K * torch.log(prior_x[0]).item() + torch.log(factorial(model.K - 1, exact=True)).item(),
                        **model.context,
                    )

                logZ_i_z = integrate_of_exponential_over_simplex(eta)
                log_partition_function = (logZ_i_Y + logZ_i_z + logZ_i_s).sum()

                if prior_x_mode == "exponential shared fixed":
                    loss += prior_x[0][0] * S.sum()
                elif not prior_x_mode:
                    pass
                else:
                    raise NotImplementedError

                if Sigma_x_inv is not None:
                    loss += (eta).mul(Z).sum() / 2

                spatial_affinity_bars = None
                if model.spatial_affinity_mode == "differential lookup":
                    spatial_affinity_bars = [
                        parameter_optimizer.spatial_affinity_state.spatial_affinity_bar[group_name].detach()
                        for group_name in model.hierarchy[level].spatial_affinity_tags[dataset.name]
                    ]

                regularization = torch.zeros(1, **model.context)
                if spatial_affinity_bars is not None:
                    group_weighting = 1 / len(spatial_affinity_bars)
                    for group_Sigma_x_inv_bar in spatial_affinity_bars:
                        regularization += (
                            group_weighting
                            * model.lambda_Sigma_bar
                            * (group_Sigma_x_inv_bar - Sigma_x_inv).pow(2).sum()
                            / 2
                        )

                regularization += model.lambda_Sigma_x_inv * Sigma_x_inv.pow(2).sum() / 2

                regularization *= weighted_total_cells

                loss += regularization.item()

            loss += log_partition_function

            differential_regularization_term = torch.zeros(1, **model.context)
            M_bar = None
            if model.metagene_mode == "differential":
                M_bar = [
                    parameter_optimizer.metagene_state.M_bar[group_name]
                    for group_name in model.hierarchy[level].metagene_tags[dataset.name]
                ]

            if model.lambda_M > 0 and M_bar is not None:
                differential_regularization_quadratic_factor = model.lambda_M * torch.eye(model.K, **model.context)

                differential_regularization_linear_term = torch.zeros_like(M, **model.context)
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_linear_term += group_weighting * model.lambda_M * group_M_bar

                differential_regularization_term = (M @ differential_regularization_quadratic_factor * M).sum() - 2 * (
                    differential_regularization_linear_term * M
                ).sum()
                group_weighting = 1 / len(M_bar)
                for group_M_bar in M_bar:
                    differential_regularization_term += (
                        group_weighting * model.lambda_M * (group_M_bar * group_M_bar).sum()
                    )

            loss += differential_regularization_term.item()

            total_loss += loss

    return total_loss.cpu().numpy()
