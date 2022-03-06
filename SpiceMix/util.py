import os, time, pickle, sys, datetime, h5py, logging
from collections import Iterable
from tqdm.auto import tqdm, trange

import numpy as np, pandas as pd
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


def calc_modularity(A, label, resolution=1):
    A = A.tocoo()
    n = A.shape[0]
    Asum = A.data.sum()
    score = A.data[label[A.row] == label[A.col]].sum() / Asum

    idx = np.argsort(label)
    label = label[idx]
    k = np.array(A.sum(0)).ravel() / Asum
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

    x = np.concatenate(Xs, axis=0)
    x = StandardScaler().fit_transform(x)

    for n_clust in [6, 7, 8, 9]:
        y = AgglomerativeClustering(
            n_clusters=n_clust,
            linkage='ward',
        ).fit_predict(x)
        y = pd.Categorical(y, categories=np.unique(y))
        print(
            f"hierarchical w/ K={n_clust}."
            f"ARI = {adjusted_rand_score(obj.meta['cell type'].values, y):.2f}",
            sep=' ',
        )
    y = clustering_louvain_nclust(
        x.copy(), 8,
        kwargs_neighbors=dict(n_neighbors=10),
        kwargs_clustering=dict(),
        resolution_boundaries=(.1, 1.),
    )
    obj.meta['label SpiceMixPlus'] = y
    print('ari = {:.2f}'.format(adjusted_rand_score(*obj.meta[['cell type', 'label SpiceMixPlus']].values.T)))
    for repli, df in obj.meta.groupby('repli'):
        print('ari {} = {:.2f}'.format(repli, adjusted_rand_score(*df[['cell type', 'label SpiceMixPlus']].values.T)))
    if do_plot:
        ncol = 4
        nrow =(obj.num_repli + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow))
        for ax, (repli, df) in zip(axes.flat, obj.meta.groupby('repli')):
            sns.heatmap(
                df.groupby(['cell type', 'label SpiceMixPlus']).size().unstack().fillna(0).astype(int),
                ax=ax, annot=True, fmt='d',
            )
        plt.show()
        plt.close()

    if do_sil:
        x = UMAP(
            random_state=obj.random_state,
            n_neighbors=10,
        ).fit_transform(x)
        print('sil', silhouette_score(x, obj.meta['cell type'], random_state=obj.random_state))
        if do_plot:
            keys = ['cell type', 'repli', 'label SpiceMixPlus']
            ncol = len(keys)
            nrow = 1 + obj.num_repli
            fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow))
            def plot(axes, idx):
                for ax, key in zip(axes, keys):
                    sns.scatterplot(ax=ax, data=obj.meta.iloc[idx], x=x[idx, 0], y=x[idx, 1], hue=key, s=5)
            plot(axes[0], slice(None))
            for ax_row, (repli, df) in zip(axes[1:], obj.meta.reset_index().groupby('repli')):
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

    See below for hints on implementation details:
    # https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

    """
    def __init__(self, parameters, step_size):
        self.parameters = parameters
        self.step_size = step_size
        # self.y = x.clone()
        self.y = torch.zeros_like(parameters)
        # self.lam = 0
        self.k = 0

    def set_parameters(self, parameters):
        self.parameters = parameters

    def step(self, grad):
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


# def getRank(m):
#   rank = np.empty(m.shape, dtype=int)
#   for r, a in zip(rank, m):
#       r[np.argsort(a)] = np.arange(len(r))
#   return rank


def getRank(m, thr=0):
    rank = np.empty(m.shape, dtype=int)
    for r, a in zip(rank, m):
        r[np.argsort(a)] = np.arange(len(r))
        mask = a < thr
        r[mask] = np.mean(r[mask])
    return rank
