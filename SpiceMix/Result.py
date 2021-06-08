import copy
import h5py
from pathlib import Path
import pandas as pd
from util import print_datetime, parseIiter, array2string, load_dict_from_hdf5_group, dict_to_list

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import umap

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# sns.set_style("white")
# plt.rcParams['font.family'] = "Liberation Sans"
# plt.rcParams['font.size'] = 16
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['svg.fonttype'] = 'none'

from load_data import load_expression
from model import SpiceMix

def findBest(path2dataset, result_filenames, iiter=-1):
    Q = []
    for r in result_filenames:
        f = h5py.File(path2dataset / 'results' / r, 'r')
        i = parseIiter(f[f'progress/Q'], iiter)
        print(f'Using iteration {i} from {r}')
        Q.append(f[f'progress/Q/{i}'][()])
        f.close()
    i = np.argmax(Q)
    print(f'The best one is model #{i} - result filename = {result_filenames[i]}')
    return result_filenames[i]

class SpiceMixResult:
    """
    
    """
    
    def __init__(self, path2dataset, result_filename, neighbor_suffix=None, expression_suffix=None, showHyperparameters=False):
        self.path2dataset = Path(path2dataset)
        self.result_filename = result_filename
        print(f'Result file = {self.result_filename}')

        with h5py.File(self.result_filename, 'r') as f:
            self.hyperparameters = load_dict_from_hdf5_group(f, 'hyperparameters/')

        self.num_repli = len(self.hyperparameters["replicate_names"])
        self.use_spatial = [True] * self.num_repli
        self.load_dataset()
        
        self.weight_columns = np.array([f'Metagene {metagene}' for metagene in range(self.hyperparameters["K"])])
        self.columns_exprs = np.array([f'{gene}' for gene in self.dataset["gene_sets"][0]])
        self.data = pd.DataFrame(index=range(sum(self.dataset["Ns"])))
        self.data[['x', 'y']] = np.concatenate([load_expression(self.path2dataset / 'files' / f'coordinates_{int(replicate)}.txt') for replicate in self.hyperparameters["replicate_names"]], axis=0)
        # self.data['cell type'] = np.concatenate([
        #     np.loadtxt(self.path2dataset / 'files' / f'celltypes_{repli}.txt', dtype=str)
        #     for repli in self.replicate_names
        # ], axis=0)
        self.data["replicate"] = sum([[replicate] * N for replicate, N in zip(self.hyperparameters["replicate_names"], self.dataset["Ns"])], [])
        self.data[self.columns_exprs] = np.concatenate(self.dataset["unscaled_YTs"], axis=0)
        self.colors = {}
        self.orders = {}

        self.metagene_order = np.arange(self.hyperparameters["K"])

    def load_dataset(self):
        with h5py.File(self.result_filename, 'r') as f:
            self.dataset = load_dict_from_hdf5_group(f, 'dataset/')
       
        self.dataset["Es"] = {int(node): adjacency_list for node, adjacency_list in self.dataset["Es"].items()} 
        self.dataset["unscaled_YTs"] = dict_to_list(self.dataset["unscaled_YTs"])
        self.dataset["YTs"] = dict_to_list(self.dataset["YTs"])
        self.dataset["gene_sets"] = dict_to_list(self.dataset["gene_sets"])
        
        self.dataset["Ns"], self.dataset["Gs"] = zip(*map(np.shape, self.dataset["unscaled_YTs"]))
        self.dataset["max_genes"] = max(self.dataset["Gs"])
        self.dataset["total_edge_counts"] = [sum(map(len, E.values())) for E in self.dataset["Es"].values()]
        
        # self.scaling = [G / self.dataset["max_genes"] * self.hyperparameters["K"] / YT.sum(axis=1).mean() for YT, G in zip(self.dataset["YTs"], self.dataset["Gs"])]
        if "scaling" not in self.dataset:
            self.dataset["scaling"] = [G / self.dataset["max_genes"] * self.hyperparameters["K"] / YT.sum(axis=1).mean() for YT, G in zip(self.dataset["YTs"], self.dataset["Gs"])]

    def plot_convergence(self, ax, **kwargs):
        label = kwargs.pop("label", "")
        with h5py.File(self.result_filename, 'r') as f:
            Q_values = f['progress/Q']
            iterations = np.fromiter(map(int, Q_values.keys()), dtype=int)
            selected_Q_values = np.fromiter((Q_values[step][()] for step in iterations.astype(str)), dtype=float)
        
        Q = np.full(iterations.max() - iterations.min() + 1, np.nan)
        Q[iterations - iterations.min()] = selected_Q_values
        print(f'Found {iterations.max()} iterations from {self.result_filename}')
        for interval, linestyle in zip([1, 5], ['-', ':']):
            dQ = (Q[interval:] - Q[:-interval]) / interval
            ax.plot(np.arange(iterations.min(), iterations.max() + 1 - interval) + interval / 2 + 1, dQ, linestyle=linestyle, label="{}-iteration Interval {}".format(interval, label), **kwargs)

    def load_latent_states(self, iiter=-1):
        with h5py.File(self.result_filename, 'r') as f:
            # iiter = parseIiter(f[f'latent_states/XT/{self.replicate_names[0]}'], iiter)
            print(f'Iteration {iiter}')
            self.weights = load_dict_from_hdf5_group(f, "weights/")
        
        self.weights = dict_to_list(self.weights)
            # XTs = [f[f'latent_states/XT/{repli}/{iiter}'][()] for repli in self.replicate_names]
        
        # XTs = [XT/ YT for XT, YT in zip(XTs, self.dataset["YTs"])]
        self.data[self.weight_columns] = np.concatenate([self.weights[replicate][iiter] / scale for replicate, scale in enumerate(self.dataset["scaling"])])

    def determine_optimal_clusters(self, ax, K_range):
        XTs = self.data[self.weight_columns].values
        XTs = StandardScaler().fit_transform(XTs)
        K_range = np.array(K_range)

        get_cluster_labels = lambda K: AgglomerativeClustering(
            n_clusters=K,
            linkage='ward',
        ).fit_predict(XTs)
        ch_scores = np.fromiter((silhouette_score(XTs, get_cluster_labels(K)) for K in K_range), dtype=float)
        optimal_K = K_range[ch_scores.argmax()]

        print(f'optimal K = {optimal_K}')
        labels = get_cluster_labels(optimal_K)
       
        num_clusters = len(set(labels) - {-1})
        print(f'#clusters = {num_clusters}, #-1 = {(labels == -1).sum()}')

        ax.scatter(K_range, ch_scores, marker='x', color=np.where(K_range == optimal_K, 'C1', 'C0'))

        self.data['cluster_raw'] = labels
        self.data['cluster'] = list(map(str, labels))

    def annotateClusters(self, clusteri2a):
        self.data['cluster'] = [clusteri2a[cluster_name] for cluster_name in self.data['cluster_raw']]

    def assignColors(self, key, mapping):
        assert set(mapping.keys()) >= set(self.data[key])
        self.colors[key] = copy.deepcopy(mapping)

    def assignOrder(self, key, order):
        s = set(self.data[key])
        assert set(order) >= s and len(order) == len(set(order))
        order = list(filter(lambda _: _ in s, order))
        self.orders[key] = np.array(order)

    def UMAP(self, **kwargs):
        XT = self.data[self.weight_columns].values
        XT = StandardScaler().fit_transform(XT)
        XT = umap.UMAP(**kwargs).fit_transform(XT)
        self.data[[f'UMAP {i+1}' for i in range(XT.shape[1])]] = XT

    def plot_feature(self, ax, key, key_x='UMAP 1', key_y='UMAP 2', replicate=None, **kwargs):
        if isinstance(replicate, int):
            replicate = self.dataset["replicate_names"][replicate]

        if replicate:
            data = self.data.groupby('replicate').get_group(replicate)
        else:
            data = self.data

        if data[key].dtype == 'O':
            kwargs.setdefault('hue_order', self.orders.get(key, None))
            kwargs.setdefault('palette', self.colors.get(key, None))
            sns.scatterplot(ax=ax, data=data, x=key_x, y=key_y, hue=key, **kwargs)
        else:
            kwargs.setdefault('cmap', self.colors.get(key, None))
            sca = ax.scatter(data[key_x], data[key_y], c=data[key], **kwargs)
            cbar = plt.colorbar(sca, ax=ax, pad=.01, shrink=1, aspect=40)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=10)

    def plot_metagenes(self, axes, replicate, *args, **kwargs):
        keys = np.array(self.weight_columns)
        keys = keys[self.metagene_order]
        self.plot_multifeature(axes, keys, replicate)

    def plot_multifeature(self, axes, keys, replicate, key_x='x', key_y='y', *args, **kwargs):
        # if len(keys) == 0:
        #     keys = self.weight_columns

        # keys = np.array(keys)
        # keys_old = keys
        # if tuple(keys) == tuple(self.weight_columns) and permute_metagenes:
        #     keys = keys[self.metagene_order]

        for ax, key in zip(axes.flat, keys):
            self.plot_feature(ax, key, key_x, key_y, replicate, *args, **kwargs)
            ax.set_title(key)

    def visualizeLabelEnrichment(
            self, ax,
            key_x='cluster', order_x=None, ignores_x=(),
            # key_y='cell type', order_y=None, ignores_y=(),
            **kwargs,
    ):
        n_x = len(set(self.data[key_x].values) - set(ignores_x))
        n_y = len(set(self.data[key_y].values) - set(ignores_y))
        if order_x is None: order_x = self.orders.get(key_x)
        if order_y is None: order_y = self.orders.get(key_y)
        value_x, _, order_x = a2i(self.data[key_x].values, order_x, ignores_x)
        value_y, _, order_y = a2i(self.data[key_y].values, order_y, ignores_y)
        c = np.stack([value_x, value_y]).T
        c = c[~(c == -1).any(1)]
        c = c[:, 0] + c[:, 1] * n_x
        c = np.bincount(c, minlength=n_x * n_y).reshape(n_y, n_x)
        cp = c / c.sum(0, keepdims=True)

        im = ax.imshow(cp, vmin=0, vmax=1, aspect='auto', extent=(-.5, n_x - .5, -.5, n_y - .5), **kwargs)
        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)
        ax.set_xticks(range(n_x))
        ax.set_yticks(range(n_y)[::-1])
        ax.set_xticklabels(order_x, rotation=-90)
        ax.set_yticklabels(order_y)
        ax.set_ylim([-.5, n_y - .5])
        ax.set(frame_on=False)
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=.3)
        cbar.outline.set_visible(False)

        for i in range(n_y):
            for j in range(n_x):
                if c[i, j] == 0: continue
                text = ax.text(j, c.shape[0]-i-1, f'{c[i, j]:d}', ha="center", va="center", color="w" if cp[i, j] > .4 else 'k')

    def visualizeFeatureEnrichment(
            self, ax,
            keys_x=(), permute_metagenes=True,
            key_y='cluster', order_y=None, ignores_y=(),
            normalizer_raw=None,
            normalizer_avg=None,
            **kwargs,
    ):
        n_y = len(set(self.data[key_y].values) - set(ignores_y))
        if order_y is None: order_y = self.orders.get(key_y)
        value_y, _, order_y = a2i(self.data[key_y].values, order_y, ignores_y)
        if len(keys_x) == 0: keys_x = self.weight_columns
        keys_x = np.array(keys_x)
        keys_x_old = keys_x
        if tuple(keys_x) == tuple(self.weight_columns) and permute_metagenes: keys_x = keys_x[self.metagene_order]
        n_x = len(keys_x)

        df = self.data[[key_y] + list(keys_x)].copy()
        if normalizer_raw is not None: df[keys_x] = normalizer_raw(df[keys_x].values)
        c = df.groupby(key_y)[keys_x].mean().loc[order_y].values
        if normalizer_avg is not None: c = normalizer_avg(c)

        if c.min() >= 0: vmin, vmax = 0, None
        else: vlim = np.abs(c).max(); vmin, vmax = -vlim, vlim
        im = ax.imshow(c, vmin=vmin, vmax=vmax, aspect='auto', extent=(-.5, n_x - .5, -.5, n_y - .5), **kwargs)
        ax.set_ylabel(key_y)
        ax.set_xticks(range(n_x))
        ax.set_yticks(range(n_y)[::-1])
        ax.set_xticklabels(keys_x_old, rotation=-90)
        ax.set_yticklabels(order_y)
        ax.set_ylim([-.5, n_y - .5])
        ax.set(frame_on=False)
        cbar = plt.colorbar(im, ax=ax, shrink=.3)
        cbar.outline.set_visible(False)

    def plot_affinity_metagenes(self, ax, iteration=-1, **kwargs):
        with h5py.File(self.result_filename, 'r') as f:
            iteration = parseIiter(f['parameters/sigma_x_inverse'], iteration)
            print(f'Iteration {iteration}')
            sigma_x_inverse = f[f'parameters/sigma_x_inverse/{iteration}'][()]
        
        sigma_x_inverse = sigma_x_inverse[self.metagene_order, :]
        sigma_x_inverse = sigma_x_inverse[:, self.metagene_order]
        sigma_x_inverse = sigma_x_inverse - sigma_x_inverse.mean()
        vertical_range = np.abs(sigma_x_inverse).max()
        image = ax.imshow(sigma_x_inverse, vmin=-vertical_range, vmax=vertical_range, **kwargs)
        ticks = list(range(0, self.K - 1, 5)) + [self.K - 1]

        if len(ax.get_xticks()):
            ax.set_xticks(ticks)
        if ax.get_yticks:
            ax.set_yticks(ticks)
        
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
        ax.set_xlabel('metagene ID')
        ax.set_ylabel('metagene ID')
        
        cbar = plt.colorbar(image, ax=ax, pad=.01, shrink=.3, aspect=20)
        cbar.outline.set_visible(False)
        ax.set_frame_on(False)

    def plotAffinityClusters(self, ax, key='cluster', ignores=(), **kwargs):
        ignores = list(ignores)
        y, mapping, order = a2i(self.data[key].values, self.orders.get(key, None), ignores)
        y = y[y != -1]
        ncluster = len(set(y))
        n = np.bincount(y)  # number of cells in each cluster
        c = np.zeros([ncluster, ncluster])
        for repli, E in zip(self.replicate_names, self.Es.values()):
            yy = self.data.groupby('repli').get_group(repli)[key].values
            yy = np.fromiter(map(mapping.get, yy), dtype=int)
            c += np.bincount(
                [i * ncluster + j for i, e in zip(yy, E.values()) if i != -1 for j in yy[e] if j != -1],
                minlength=c.size,
            ).reshape(c.shape)
        assert (c == c.T).all(), (c - c.T)
        k = c.sum(0)  # degree of each cluster = sum of node deg
        m = c.sum()
        c -= np.outer(k, k / (m - 1))
        c.ravel()[::ncluster + 1] += k / (m - 1)
        c *= 2
        c.ravel()[::ncluster + 1] /= 2
        n = np.sqrt(n)
        c /= n[:, None]
        c /= n[None, :]

        vlim = np.abs(c).max()
        im = ax.imshow(c, vmax=vlim, vmin=-vlim, **kwargs)
        ax.set_xticks(range(ncluster))
        ax.set_yticks(range(ncluster))
        ax.set_xticklabels(order, rotation='270')
        ax.set_yticklabels(order)
        ax.set_xlabel(f'Cell clusters')
        ax.set_ylabel(f'Cell clusters')
        if key in self.colors:
            for tick_ind, tick in enumerate(ax.get_xticklabels()):
                bbox = dict(boxstyle="round", ec='none', fc=self.colors[key][order[tick_ind]], alpha=0.5, pad=.08)
                plt.setp(tick, bbox=bbox)
            for tick_ind, tick in enumerate(ax.get_yticklabels()):
                bbox = dict(boxstyle="round", ec='none', fc=self.colors[key][order[tick_ind]], alpha=0.5, pad=.08)
                plt.setp(tick, bbox=bbox)
        cbar = plt.colorbar(im, ax=ax, pad=.01, shrink=.3, aspect=20)
        cbar.outline.set_visible(False)
        ax.set_frame_on(False)
