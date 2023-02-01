import os, time, pickle, sys, datetime, logging
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, f1_score
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


def array2string(x):
    return np.array2string(x, formatter={'all': '{:.2e}'.format})


def parse_suffix(s):
    return '' if s is None or s == '' else '_' + s


def openH5File(filename, mode='a', num_attempts=5, duration=1):
    for i in range(num_attempts):
        try:
            return h5py.File(filename, mode=mode)
        except OSError as e:
            logging.warning(str(e))
            time.sleep(duration)
    return None


def encode4h5(v):
    if isinstance(v, str): return v.encode('utf-8')
    return v


def parseIiter(g, iiter):
    if iiter < 0: iiter += max(map(int, g.keys())) + 1
    return iiter


def a2i(a, order=None, ignores=()):
    if order is None:
        order = np.array(list(set(a) - set(ignores)))
    else:
        order = order[~np.isin(order, list(ignores))]
    d = dict(zip(order, range(len(order))))
    for k in ignores: d[k] = -1
    a = np.fromiter(map(d.get, a), dtype=int)
    return a, d, order


def zipTensors(*tensors):
    return np.concatenate([
        np.array(a).flatten()
        for a in tensors
    ])


def unzipTensors(arr, shapes):
    assert np.all(arr.shape == (np.sum(list(map(np.prod, shapes))),))
    tensors = []
    for shape in shapes:
        size = np.prod(shape)
        tensors.append(arr[:size].reshape(*shape).squeeze())
        arr = arr[size:]
    return tensors


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
