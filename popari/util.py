from typing import Optional
import os, time, pickle, sys, datetime, logging
from tqdm.auto import tqdm, trange

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from anndata import AnnData
import scanpy as sc

import torch

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import defaultdict


def create_neighbor_groups(replicate_names, covariate_values, window_size = 1):
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
        group_replicates = sum([covariate_to_replicate_name[group_covariate] for group_covariate in group_covariates], [])
        group_name = f"{sorted_covariates[index]}"

        groups[group_name] = list(group_replicates)

    return groups

def calc_modularity(adjacency_matrix, label, resolution=1):
    adjacency_matrix = adjacency_matrix.tocoo()
    n = adjacency_matrix.shape[0]
    adjacency_matrix_sum = adjacency_matrix.data.sum()
    score = adjacency_matrix.data[label[adjacency_matrix.row] == label[adjacency_matrix.col]].sum() / adjacency_matrix_sum

    idx = np.argsort(label)
    label = label[idx]
    k = np.array(adjacency_matrix.sum(0)).ravel() / adjacency_matrix_sum
    k = k[idx]
    idx = np.concatenate([[0], np.nonzero(label[:-1] != label[1:])[0] + 1, [len(label)]])
    score -= sum(k[i:j].sum() ** 2 for i, j in zip(idx[:-1], idx[1:])) * resolution
    return score


def clustering_louvain(X, *, kwargs_neighbors, kwargs_clustering, num_rs=100, method='louvain'):
    adata = AnnData(X)
    sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
    best = {'score': np.nan}
    resolution = kwargs_clustering.get('resolution', 1)
    pbar = trange(num_rs, desc=f'Louvain clustering: res={resolution:.2e}')
    for rs in pbar:
        getattr(sc.tl, method)(adata, **kwargs_clustering, random_state=rs)
        cluster = np.array(list(adata.obs[method]))
        score = calc_modularity(
            adata.obsp['connectivities'], cluster, resolution=resolution)
        if not best['score'] >= score:
            best.update({'score': score, 'cluster': cluster.copy(), 'rs': rs})
        pbar.set_description(
            f'Louvain clustering: res={resolution:.2e}; '
            f"best: score = {best['score']:.2f} rs = {best['rs']} # of clusters = {len(set(best['cluster']))}"
        )
    y = best['cluster']
    y = pd.Categorical(y, categories=np.unique(y))
    return y


def clustering_louvain_nclust(
    X, n_clust_target, *, kwargs_neighbors, kwargs_clustering,
    resolution_boundaries=None,
    resolution_init=1, resolution_update=2,
    num_rs=100, method='louvain',
):
    """Some sort of wrapper around Louvain clustering.

    """

    adata = AnnData(X)

    # Get nearest neighbors in embedding space
    sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
    kwargs_clustering = kwargs_clustering.copy()
    y = None

    def do_clustering(res):
        y = clustering_louvain(
            X,
            kwargs_neighbors=kwargs_neighbors,
            kwargs_clustering=dict(**kwargs_clustering, **dict(resolution=res)),
            method=method,
            num_rs=num_rs,
        )
        n_clust = len(set(y))
        return y, n_clust

    lb = rb = None
    if resolution_boundaries is not None:
        lb, rb = resolution_boundaries
    else:
        res = resolution_init
        y, n_clust = do_clustering(res)
        if n_clust > n_clust_target:
            while n_clust > n_clust_target and res > 1e-2:
                rb = res
                res /= resolution_update
                y, n_clust = do_clustering(res)
            lb = res
        elif n_clust < n_clust_target:
            while n_clust < n_clust_target:
                lb = res
                res *= resolution_update
                y, n_clust = do_clustering(res)
            rb = res
        if n_clust == n_clust_target:
            lb = rb = res

    while rb - lb > .01:
        mid = (lb * rb) ** .5
        y = clustering_louvain(
            X,
            kwargs_neighbors=kwargs_neighbors,
            kwargs_clustering=dict(**kwargs_clustering, **dict(resolution=mid)),
            method=method,
            num_rs=num_rs,
        )
        n_clust = len(set(y))
        # print(
        #   f'binary search for resolution: lb={lb:.2f}\trb={rb:.2f}\tmid={mid:.2f}\tn_clust={n_clust}',
        #   # '{:.2f}'.format(adjusted_rand_score(obj.data['cell type'], obj.data['cluster'])),
        #   sep='\t',
        # )
        if n_clust == n_clust_target:
            break
        if n_clust > n_clust_target:
            rb = mid
        else:
            lb = mid

    return y

def evaluate_embedding(obj, embedding='X', do_plot=True, do_sil=True):
    if embedding == 'X':
        Xs = [X.cpu().numpy() for X in obj.Xs]
    elif embedding in obj.phenotype_predictors:
        Xs = [
            obj.phenotype_predictors['cell type encoded'][0](X).cpu().numpy()
            for X in obj.Xs
        ]
    else:
        raise NotImplementedError

    # x = np.concatenate(Xs, axis=0)

    for x, replicate, dataset in zip(Xs, obj.repli_list, obj.datasets):
        x = StandardScaler().fit_transform(x)
        for n_clust in [6, 7, 8, 9]:
            y = AgglomerativeClustering(
                n_clusters=n_clust,
                linkage='ward',
            ).fit_predict(x)
            y = pd.Categorical(y, categories=np.unique(y))
            dataset.obs["spicemixplus_label"] = y

            print(
                f"replicate {replicate}"
                f"hierarchical w/ K={n_clust}."
                f"ARI = {adjusted_rand_score(dataset.obs['cell_type'].values, y):.2f}",
                sep=' ',
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
        nrow =(obj.num_repli + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow))
        for ax, replicate, dataset in zip(axes.flat, obj.repli_list, obj.datasets):
            sns.heatmap(
                dataset.obs.groupby(['cell_type', 'spicemixplus_label']).size().unstack().fillna(0).astype(int),
                ax=ax, annot=True, fmt='d',
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
            print('sil', silhouette_score(x, dataset.obs['cell_type'], random_state=obj.random_state))
            if do_plot:
                keys = ['cell_type', 'replicate', 'spicemixplus_label']
                ncol = len(keys)
                nrow = 1 + obj.num_repli
                fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow))
                def plot(axes, idx):
                    for ax, key in zip(axes, keys):
                        sns.scatterplot(ax=ax, data=dataset.obs.iloc[idx], x=x[idx, 0], y=x[idx, 1], hue=key, s=5)
                plot(axes[0], slice(None))
                for ax_row, (repli, df) in zip(axes[1:], dataset.obs.reset_index().groupby('replicate')):
                    plot(ax_row, df.index)
                plt.show()
                plt.close()


def evaluate_prediction_wrapper(obj, *, key_truth='cell type encoded', display_fn=print, **kwargs):
    X_all = torch.cat([X for X in obj.Xs], axis=0)
    obj.meta['label SpiceMixPlus predictor'] = obj.phenotype_predictors[key_truth][0](X_all)\
        .argmax(1).cpu().numpy()
    display_fn(evaluate_prediction(obj.meta, key_pred='label SpiceMixPlus predictor', key_truth=key_truth, **kwargs))


def evaluate_prediction(df_meta, *, key_pred='label SpiceMixPlus', key_truth='cell type', key_repli='repli'):
    df_score = {}
    for repli, df in df_meta.groupby(key_repli):
        t = tuple(df[[key_truth, key_pred]].values.T)
        r = {
            'acc': accuracy_score(*t),
            'f1 micro': f1_score(*t, average='micro'),
            'f1 macro': f1_score(*t, average='macro'),
            'ari': adjusted_rand_score(*t),
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
        gamma = - (self.k - 1) / (self.k + 2)
        # method 3 - GD
        # gamma = 0
        # y_new = self.x.sub(grad, alpha=self.step_size)
        y_new = self.parameters - grad * self.step_size # use addcmul
        self.y = (self.y * gamma).add(y_new, alpha=(1 - gamma))
        self.parameters = self.y
        # self.lam = lam_new
        self.y = y_new

        return self.parameters


def print_datetime():
    return datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]\t')

def get_datetime():
    return datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]\t')

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
    if M_constraint == 'simplex':
        result = project2simplex(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == 'unit sphere':
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == 'nonneg unit sphere':
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result

def project_M_(M, M_constraint):
    result = M.clone()
    if M_constraint == 'simplex':
        result = project2simplex_(result, dim=0, zero_threshold=1e-5)
    elif M_constraint == 'unit sphere':
        result = M.div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    elif M_constraint == 'nonneg unit sphere':
        result = M.clip(1e-10).div(torch.linalg.norm(result, ord=2, dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return result

def project2simplex(y, dim: int = 0, zero_threshold: float = 1e-10) -> torch.Tensor:
    """Projects a matrix such that the columns (or rows) lie on the unit simplex.

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
    """(In-place) Projects a matrix such that the columns (or rows) lie on the unit simplex.

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

    y_copy.sub_(y_copy.sum(dim=dim, keepdim=True).sub_(1), alpha=1/num_components)
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
    """Iterator class that yields a list of batch_size independent nodes from a spatial graph.

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
        """Returns the indices of batch_size nodes such that none of them are neighbors of each other.

        Makes sure selected nodes are not adjacent to each other, i.e., finds an independent set of
        `valid indices` in a greedy manner
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
    candidate_indices = np.random.choice(list(indices_remaining),
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
    r"""Compute the normalized enrichment of features in direct neighbors on a graph.

    Args:
        features: attributes on the nodes of the graph on which to compute enrichment.
        adjacency_matrix: sparse graph representation.
    """

    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T).astype(bool).astype(adjacency_matrix.dtype)
    edges_per_node = np.squeeze(np.asarray(adjacency_matrix.sum(axis=0)))
    connected_mask = (edges_per_node > 0)

    features = features[connected_mask]
    adjacency_matrix = adjacency_matrix[connected_mask][:, connected_mask]

    total_counts = features.sum(axis=0)[:, np.newaxis]
    assert np.all(total_counts > 0)

    normalized_enrichment = ((1 / total_counts) * features.T) @ (1/ edges_per_node[connected_mask][:, np.newaxis] * adjacency_matrix.toarray() @ features)

    return np.asarray(normalized_enrichment)

def chunked_coordinates(coordinates: np.ndarray, chunks: int = None, step_size: float = None):
    """Split a list of 2D coordinates into local chunks.
    
    Args:
        chunks: number of equal-sized chunks to split horizontal axis. Vertical chunks are constructed
            with the same chunk size determined by splitting the horizontal axis.
            
    Yields:
        The next chunk of coordinates, in row-major order.
    """
    
    num_points, _ = coordinates.shape

    horizontal_base, vertical_base = np.min(coordinates, axis=0)
    horizontal_range, vertical_range = np.ptp(coordinates, axis=0)
  
    if step_size is None and chunks is None:
        raise ValueError("One of `chunks` or `step_size` must be specified.")

    if step_size is None:
        horizontal_borders, step_size = np.linspace(horizontal_base, horizontal_base + horizontal_range, chunks + 1, retstep=True)
    elif chunks is None:
        horizontal_borders = np.arange(horizontal_base, horizontal_base + horizontal_range, step_size)
        
        # Adding endpoint
        horizontal_borders = np.append(horizontal_borders, horizontal_borders[-1] + step_size)

    vertical_borders = np.arange(vertical_base, vertical_base + vertical_range, step_size)
    
    # Adding endpoint
    vertical_borders = np.append(vertical_borders, vertical_borders[-1] + step_size)
    
    for i in range(len(horizontal_borders)-1):
        horizontal_low, horizontal_high = horizontal_borders[i:i+2]
        for j in range(len(vertical_borders)-1):
            vertical_low, vertical_high = vertical_borders[j:j+2]
            horizontal_mask = (coordinates[:, 0] > horizontal_low) & (coordinates[:, 0] <= horizontal_high)
            vertical_mask = (coordinates[:, 1] > vertical_low) & (coordinates[:, 1] <= vertical_high)
            chunk_coordinates = coordinates[horizontal_mask & vertical_mask]
            
            chunk_data = {
                'horizontal_low': horizontal_low,
                'horizontal_high': horizontal_high,
                'vertical_low': vertical_low,
                'vertical_high': vertical_high,
                'step_size': step_size,
                'chunk_coordinates': chunk_coordinates
            }
            yield chunk_data

def finetune_chunk_number(coordinates: np.ndarray, chunks: int, downsample_rate: float, max_nudge: Optional[int] = None):
    """Heuristically search for a chunk number that cleanly splits up points.
    
    Using a linear search, searches for a better value of the ``chunks`` value such that
    the average points-per-chunk is closer to the number of points that will be in the chunk.
    
    Args:
        coordinates: original spot coordinates
        chunks: number of equal-sized chunks to split horizontal axis
        downsample_rate: approximate desired ratio of meta-spots to spots after downsampling
    
    Returns:
        finetuned number of chunks
    """
    if max_nudge is None:
        max_nudge = chunks // 2
        
    num_points, num_dimensions = coordinates.shape
    target_points = num_points * downsample_rate

    horizontal_base, vertical_base = np.min(coordinates, axis=0)
    horizontal_range, vertical_range = np.ptp(coordinates, axis=0)

    direction = 0
    for chunk_nudge in range(max_nudge):
        valid_chunks = []
        for chunk_data in chunked_coordinates(coordinates, chunks=chunks + direction * chunk_nudge):
            if len(chunk_data['chunk_coordinates']) > 0:
                valid_chunks.append(chunk_data)

        points_per_chunk = num_points * downsample_rate / len(valid_chunks)
        downsampled_1d_density = int(np.round(np.sqrt(points_per_chunk)))
        new_direction = 1 - 2 * ((downsampled_1d_density**2) > points_per_chunk)
        if direction == 0:
            direction = new_direction
        else:
            if direction != new_direction:
                break
    
    return chunks + direction * chunk_nudge

def chunked_downsample_on_grid(coordinates: np.ndarray, downsample_rate: float, chunks: Optional[int] = None,
        chunk_size: Optional[float] = None, downsampled_1d_density: Optional[int] = None):
    """Downsample spot coordinates to a square grid of meta-spots using chunks.
    
    By chunking the coordinates, we can:
    
    1. Remove unused chunks.
    2. Estimate the density of spots at chunk-sized resolution.
    
    We use this information when downsampling in order to
    
    Args:
        coordinates: original spot coordinates
        chunks: number of equal-sized chunks to split horizontal axis
        downsample_rate: approximate desired ratio of meta-spots to spots after downsampling
        
    Returns:
        coordinates of downsampled meta-spots
    
    """
    
    num_points, num_dimensions = coordinates.shape

    horizontal_base, vertical_base = np.min(coordinates, axis=0)
    horizontal_range, vertical_range = np.ptp(coordinates, axis=0)
    
    chunks = finetune_chunk_number(coordinates, chunks, downsample_rate)
    valid_chunks = []
    for chunk_data in chunked_coordinates(coordinates, chunks=chunks, step_size=chunk_size):
        if len(chunk_data['chunk_coordinates']) > 0:
            valid_chunks.append(chunk_data)

    points_per_chunk = num_points * downsample_rate / len(valid_chunks)

    if downsampled_1d_density is None:
        downsampled_1d_density = int(np.round(np.sqrt(points_per_chunk)))

    if points_per_chunk < 2:
        raise ValueError("Chunk density is < 1")
    
    all_new_coordinates = []
    for index, chunk_data in enumerate(valid_chunks):
        horizontal_low =  chunk_data['horizontal_low']
        horizontal_high =  chunk_data['horizontal_high']
        vertical_low =  chunk_data['vertical_low']
        vertical_high =  chunk_data['vertical_high']
        step_size =  chunk_data['step_size']
        
        x = np.linspace(horizontal_low, horizontal_high, downsampled_1d_density, endpoint=False)
        if np.allclose(horizontal_high, horizontal_base + horizontal_range):
            x_gap = x[-1] - x[-2]
            x = np.append(x, x.max() + x_gap)
            
        y = np.linspace(vertical_low, vertical_high, downsampled_1d_density, endpoint=False)
        if np.allclose(vertical_high, vertical_base + vertical_range):
            y_gap = y[-1] - y[-2]
            y = np.append(y, y.max() + y_gap)
          
        xv, yv = np.meshgrid(x, y)
          
        new_coordinates = np.array(list(zip(xv.flat, yv.flat)))
        all_new_coordinates.append(new_coordinates)
    
    new_coordinates = np.vstack(all_new_coordinates)
    new_coordinates = np.unique(new_coordinates, axis=0)
    
    return new_coordinates, chunk_size, downsampled_1d_density

def filter_gridpoints(spot_coordinates: np.ndarray, grid_coordinates: np.ndarray, num_jobs: int):
    """Use nearest neighbors approach to filter out relevant grid coordinates.
    
    Keeps only the grid coordinates that are mapped to at least a single original spot.
    
    Args:
        spot_coordinates: coordinates of original spots
        grid_coordinates: coordinates of downsampled grid
        
    Returns:
        metaspots that meet the filtering criterion
    """
        
    spot_to_metaspot_mapper = NearestNeighbors(n_neighbors=1, n_jobs=num_jobs)
    spot_to_metaspot_mapper.fit(grid_coordinates)
    
    indices = spot_to_metaspot_mapper.kneighbors(spot_coordinates, return_distance=False)
    
    used_bins = set(indices.flat)
    
    filtered_bin_coordinates = grid_coordinates[list(used_bins)]
    
    return filtered_bin_coordinates

def bin_expression(spot_expression: np.ndarray, spot_coordinates: np.ndarray, bin_coordinates: np.ndarray, num_jobs: int):
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
    bin_assignments = np.zeros((num_bins, num_spots))
    
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
