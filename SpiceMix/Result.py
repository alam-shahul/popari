import copy
import h5py
from pathlib import Path
import pandas as pd
from util import print_datetime, parseIiter, array2string, load_dict_from_hdf5_group, dict_to_list

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import umap

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
from scipy.stats import pearsonr

# sns.set_style("white")
# plt.rcParams['font.family'] = "Liberation Sans"
# plt.rcParams['font.size'] = 16
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['svg.fonttype'] = 'none'

from load_data import load_expression
from model import SpiceMix
from pathlib import Path

class SpiceMixResult:
    """Provides methods to interpret a SpiceMix result.
    
    """
    
    def __init__(self, path2dataset, result_filename, neighbor_suffix=None, expression_suffix=None, showHyperparameters=False):
        self.path2dataset = Path(path2dataset)
        self.result_filename = result_filename
        print(f'Result file = {self.result_filename}')

        self.load_progress()
        self.load_hyperparameters()

        self.load_dataset()
        self.num_repli = len(self.dataset["replicate_names"])
        self.use_spatial = [True] * self.num_repli
  
        self.load_parameters()

        self.weight_columns = np.array([f'Metagene {metagene}' for metagene in range(self.hyperparameters["K"])])
        self.columns_exprs = np.array(self.dataset["gene_sets"][self.dataset["replicate_names"][0]])
    
        self.data = pd.DataFrame(index=range(sum(self.dataset["Ns"])))
        self.data[['x', 'y']] = np.concatenate([load_expression(self.path2dataset / 'files' / f'coordinates_{replicate}.txt') for replicate in self.dataset["replicate_names"]], axis=0)
        # self.data['cell type'] = np.concatenate([
        #     np.loadtxt(self.path2dataset / 'files' / f'celltypes_{repli}.txt', dtype=str)
        #     for repli in self.replicate_names
        # ], axis=0)
        self.data["replicate"] = sum([[replicate] * N for replicate, N in zip(self.dataset["replicate_names"], self.dataset["Ns"])], [])
        print(self.columns_exprs)
        self.columns_exprs = [" ".join(symbols) for symbols in self.columns_exprs]
        print(self.columns_exprs)
        self.data[self.columns_exprs] = np.concatenate(self.dataset["unscaled_YTs"], axis=0)
        
        if "labels" in self.dataset:
            self.dataset["labels"] = dict_to_list(self.dataset["labels"])
            self.data["label"] =  np.concatenate(self.dataset["labels"])
            
        self.colors = {}
        self.orders = {}

        self.metagene_order = np.arange(self.hyperparameters["K"])

    def load_hyperparameters(self):
        with h5py.File(self.result_filename, 'r') as f:
            self.hyperparameters = load_dict_from_hdf5_group(f, 'hyperparameters/')
    
    def load_progress(self):
        with h5py.File(self.result_filename, 'r') as f:
            self.progress = load_dict_from_hdf5_group(f, 'progress/')
            
        self.progress["Q"] = dict_to_list(self.progress["Q"])

    def load_parameters(self):
        with h5py.File(self.result_filename, 'r') as f:
            self.parameters = load_dict_from_hdf5_group(f, 'parameters/')

        self.parameters["sigma_x_inverse"] = dict_to_list(self.parameters["sigma_x_inverse"])
        self.parameters["M"] = dict_to_list(self.parameters["M"])
        self.parameters["sigma_yx_inverses"] = dict_to_list(self.parameters["sigma_yx_inverses"])
        self.parameters["prior_x_parameter"] = dict_to_list(self.parameters["prior_x_parameter"])

    def load_dataset(self):
        with h5py.File(self.result_filename, 'r') as f:
            self.dataset = load_dict_from_hdf5_group(f, 'dataset/')
       
        self.dataset["Es"] = {int(node): adjacency_list for node, adjacency_list in self.dataset["Es"].items()} 
        self.dataset["unscaled_YTs"] = dict_to_list(self.dataset["unscaled_YTs"])
        self.dataset["YTs"] = dict_to_list(self.dataset["YTs"])
        for replicate_index, replicate_name in enumerate(self.dataset["gene_sets"]):
            self.dataset["gene_sets"][replicate_name] =  np.loadtxt(Path(self.path2dataset) / "files" / f"genes_{replicate_name}.txt", dtype=str)
            # np.char.decode(self.dataset["gene_sets"][replicate_name], encoding="utf-8")
            replicate_index = str(replicate_index)
            if "labels" in self.dataset:
                self.dataset["labels"][replicate_index] =  np.char.decode(self.dataset["labels"][replicate_index], encoding="utf-8")
        
        self.dataset["Ns"], self.dataset["Gs"] = zip(*map(np.shape, self.dataset["unscaled_YTs"]))
        self.dataset["max_genes"] = max(self.dataset["Gs"])
        self.dataset["total_edge_counts"] = [sum(map(len, E.values())) for E in self.dataset["Es"].values()]
        
        self.dataset["replicate_names"] =  [replicate_name.decode("utf-8")  for replicate_name in self.dataset["replicate_names"]]
        
        # self.scaling = [G / self.dataset["max_genes"] * self.hyperparameters["K"] / YT.sum(axis=1).mean() for YT, G in zip(self.dataset["YTs"], self.dataset["Gs"])]
        if "scaling" not in self.dataset:
            self.dataset["scaling"] = [G / self.dataset["max_genes"] * self.hyperparameters["K"] / YT.sum(axis=1).mean() for YT, G in zip(self.dataset["YTs"], self.dataset["Gs"])]

    def plot_convergence(self, ax, **kwargs):
            
        label = kwargs.pop("label", "")
        with h5py.File(self.result_filename, 'r') as f:
            Q_values = load_dict_from_hdf5_group(f, 'progress/Q')
            iterations = np.fromiter(map(int, Q_values.keys()), dtype=int)
            selected_Q_values = np.fromiter((Q_values[step][()] for step in iterations.astype(str)), dtype=float)
        
        Q = np.full(iterations.max() - iterations.min() + 1, np.nan)
        Q[iterations - iterations.min()] = selected_Q_values
        print(f'Found {iterations.max()} iterations from {self.result_filename}')
        
        ax.set_title('Q Score')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('$\Delta$Q')
        ax.set_yscale('log')
        ax.set_ylim(10**-1, 10**3)
        ax.legend()
    
        for interval, linestyle in zip([1, 5], ['-', ':']):
            dQ = (Q[interval:] - Q[:-interval]) / interval
            ax.plot(np.arange(iterations.min(), iterations.max() + 1 - interval) + interval / 2 + 1, dQ, linestyle=linestyle, label="{}-iteration $\Delta$Q ({})".format(interval, label), **kwargs)

    def load_latent_states(self, iiter=-1):
        with h5py.File(self.result_filename, 'r') as f:
            # iiter = parseIiter(f[f'latent_states/XT/{self.replicate_names[0]}'], iiter)
            print(f'Iteration {iiter}')
            self.weights = load_dict_from_hdf5_group(f, "weights/")
        
        self.weights = dict_to_list(self.weights)
            # XTs = [f[f'latent_states/XT/{repli}/{iiter}'][()] for repli in self.replicate_names]
        
        # XTs = [XT/ YT for XT, YT in zip(XTs, self.dataset["YTs"])]
        self.data[self.weight_columns] = np.concatenate([self.weights[replicate_index][iiter] / scale for replicate_index, scale in zip(range(self.num_repli), self.dataset["scaling"])])

    def determine_optimal_clusters(self, ax, K_range, metric="callinski_harabasz"):
        XTs = self.data[self.weight_columns].values
        XTs = StandardScaler().fit_transform(XTs)
        K_range = np.array(K_range)

       
        if metric == "callinski_harabasz":
            scores = np.fromiter((calinski_harabasz_score(XTs, self.determine_clusters(K)) for K in K_range), dtype=float)
        elif metric == "silhouette":
            scores = np.fromiter((silhouette_score(XTs, self.determine_clusters(K)) for K in K_range), dtype=float)
        
        optimal_K = K_range[scores.argmax()]

        print(f'optimal K = {optimal_K}')
        labels = self.determine_clusters(optimal_K)
       
        num_clusters = len(set(labels) - {-1})
        print(f'#clusters = {num_clusters}, #-1 = {(labels == -1).sum()}')

        ax.scatter(K_range, scores, marker='x', color=np.where(K_range == optimal_K, 'C1', 'C0'))
        
    def determine_clusters(self, K, features=None, replicate=None):
        data = self.data
        if replicate:
            replicate_mask = (data["replicate"] == replicate)
            data = data.loc[replicate_mask]
           
        if not features:
            features = self.weight_columns
            
        XTs = data[features].values
        cluster_labels = AgglomerativeClustering(
            n_clusters=K,
            linkage='ward',
        ).fit_predict(XTs)
       
        if replicate:
            self.data.loc[replicate_mask, 'cluster_raw'] = cluster_labels
            self.data.loc[replicate_mask, 'cluster'] = list(map(str, cluster_labels))
        else:    
            self.data.loc[:, 'cluster_raw'] = cluster_labels
            self.data.loc[:, 'cluster'] = list(map(str, cluster_labels))
        
        return cluster_labels

    def annotateClusters(self, clusteri2a):
        self.data['cluster'] = [clusteri2a[cluster_name] for cluster_name in self.data['cluster_raw']]

    def assignColors(self, key, mapping):
        assert set(mapping.keys()) >= set(self.data[key])
        self.colors[key] = copy.deepcopy(mapping)

    def assignOrder(self, key, order):
        categories = set(self.data[key])
        assert set(order) >= categories and len(order) == len(set(order))
        order = list(filter(lambda category: category in categories, order))
        self.orders[key] = np.array(order)

    def UMAP(self, **kwargs):
        XTs = self.data[self.weight_columns].values
        XTs = StandardScaler().fit_transform(XTs)
        XTs = umap.UMAP(**kwargs).fit_transform(XTs)
        self.data[[f'UMAP {i+1}' for i in range(XTs.shape[1])]] = XTs

    def plot_feature(self, ax, key, key_x='UMAP 1', key_y='UMAP 2', replicate=None, show_colorbar=True, **kwargs):
        # We ovrlap latent states on the spatial space
        # SpiceMix metagenes are expected to show clearer spatial patterns with less background expressions
        segmentdata = copy.deepcopy(plt.get_cmap('Reds')._segmentdata)
        segmentdata['red'  ][0] = (0., 1., 1.)
        segmentdata['green'][0] = (0., 1., 1.)
        segmentdata['blue' ][0] = (0., 1., 1.)
        cmap = LinearSegmentedColormap('', segmentdata=segmentdata, N=256)

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
            kwargs.setdefault('cmap', cmap)
            sca = ax.scatter(data[key_x], data[key_y], c=data[key], **kwargs)
            if show_colorbar:
                cbar = plt.colorbar(sca, ax=ax, pad=.01, shrink=1, aspect=40)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        ax.tick_params(axis='both', labelsize=10)

    def plot_aggregated_feature(self, ax, keys, key_x="x", key_y="y", replicate=None, show_colorbar=True, **kwargs):
        if isinstance(replicate, int):
            replicate = self.dataset["replicate_names"][replicate]

        if replicate:
            data = self.data.groupby('replicate').get_group(replicate)
        else:
            data = self.data

        sca = ax.scatter(data[key_x], data[key_y], c=data[keys].sum(axis="columns"), **kwargs)
        if show_colorbar:
            cbar = plt.colorbar(sca, ax=ax, pad=.01, shrink=1, aspect=40)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=10)

    def plot_metagenes(self, axes, replicate, *args, **kwargs):
        keys = np.array(self.weight_columns)
        keys = keys[self.metagene_order]
        self.plot_multifeature(axes, keys, replicate, **kwargs)

    def plot_multifeature(self, axes, keys, replicate, key_x='x', key_y='y', show_colorbar=True, *args, **kwargs):
        """Plot multiple SpiceMixResult features on the provided axes for a given replicate.
        
        """
        
        for ax, key in zip(axes.flat, keys):
            self.plot_feature(ax, key, key_x, key_y, replicate, show_colorbar=show_colorbar, *args, **kwargs)
            ax.set_title(key)

    def plot_multireplicate(self, axes, key, key_x="x", key_y="y", palette_option="husl", *args, **kwargs):
        categories = self.data[key].unique()
        category_map = {category: index for index, category in enumerate(categories)}
        num_categories = len(categories)
        
        palette = sns.color_palette(palette_option, num_categories)
        sns.set_palette(palette)
        
        colormap = ListedColormap(palette)
        
        bounds = np.linspace(0, num_categories, num_categories + 1)
        norm = BoundaryNorm(bounds, colormap.N)

        
        for ax, replicate in zip(axes.flat, self.dataset["replicate_names"]):
            if replicate not in self.data["replicate"].values:
                ax.axis('off')
                continue
                
            subdata = self.data.groupby("replicate").get_group(replicate).groupby(key)
            for subkey, group in subdata:
                group.plot(ax=ax, kind='scatter', x='x', y='y', label=subkey, color=colormap(category_map[subkey]), **kwargs)
        
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(replicate)
            ax.get_legend().remove()
            ax.set(adjustable='box', aspect='equal')
        
        legend_axis = axes.flat[-1]
        legend_axis.set_title("Legend")
        legend_axis.imshow(np.arange(num_categories)[:, np.newaxis], cmap=colormap, aspect=1)
        legend_axis.set_xticks([])
        legend_axis.set_yticks(np.arange(num_categories))
        legend_axis.set_yticklabels(categories)
        plt.tight_layout()

    def calculate_metagene_correlations(self, replicate, benchmark, comparison_features):
        correlations = pd.DataFrame(index=self.weight_columns, columns=comparison_features)

        replicate_data = self.data.groupby("replicate").get_group(replicate)

        for feature in comparison_features:
            feature_values = benchmark[feature]
            for metagene in self.weight_columns:
                correlation = pearsonr(replicate_data[metagene].values, feature_values.values)[0]
                correlations.loc[metagene, feature] = correlation

        return correlations

    def calculate_ari_score(self, replicate=None):
        data = self.data
        if replicate:
            data = data[data["replicate"] == replicate]
            
        label_values, label_indices, label_encoded = np.unique(data["label"], return_index=True, return_inverse=True)
        cluster_values, cluster_indices, cluster_encoded = np.unique(data["cluster"], return_index=True, return_inverse=True)
        
        ari = adjusted_rand_score(label_encoded, cluster_encoded)
        
        return ari
    
    def plot_ari_versus_clusters(self, ax, K_range):
        """Plot ARI score as a function of the number of clusters used in K-means clustering.
        
        """
        
        XTs = self.data[self.weight_columns].values
        XTs = StandardScaler().fit_transform(XTs)
        K_range = np.array(K_range)
        
        ari_scores = []
       
        for index, K in enumerate(K_range):
            labels = self.determine_clusters(K)
            ari_scores.append(self.calculate_ari_score())
        
        optimal_num_clusters = np.argmax(ari_scores) + K_range[0]
        
        ax.set_ylabel("ARI Score")
        ax.set_xlabel("Clusters")
        ax.plot(K_range, ari_scores)
        
        return optimal_num_clusters
        
    def get_important_features(self):
        self.parameters["M"][-1]

    def plot_categorical_overlap(self, ax,
            key_x='cluster', order_x=None, ignores_x=(),
            key_y='label', order_y=None, ignores_y=(),
            **kwargs,
    ):
        num_x_categories = len(set(self.data[key_x].values) - set(ignores_x))
        num_y_categories = len(set(self.data[key_y].values) - set(ignores_y))
        
        value_x = self.data[key_x].values
        if order_x:
            indices = np.arange(len(order_x))
            remapping = dict(zip(order_x, indices))
            value_x = [remapping[label] for label in value_x]
        else:
            order_x, value_x = np.unique(value_x, return_inverse=True)
        
        value_y = self.data[key_y].values
        if order_y:
            indices = np.arange(len(order_y))
            remapping = dict(zip(order_y, indices))
            value_y = [remapping[label] for label in value_y]
        else:
            order_y, value_y = np.unique(value_y, return_inverse=True)
        
        num_bins = num_x_categories * num_y_categories
        pairs = np.stack([value_x, value_y]).T

        values = pairs[:, 0] + pairs[:, 1] * num_x_categories
        counts = np.bincount(values, minlength=num_bins).reshape(num_y_categories, num_x_categories)
        normalized_counts = counts / counts.sum(axis=0, keepdims=True)

        count_image = ax.imshow(normalized_counts, vmin=0, vmax=1, aspect='auto', extent=(-.5, num_x_categories - .5, -.5, num_y_categories - .5), **kwargs)
        
        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)
        
        ax.set_xticks(range(num_x_categories))
        ax.set_yticks(range(num_y_categories)[::-1])
        ax.set_xticklabels(order_x, rotation=-90)
        ax.set_yticklabels(order_y)
        
        ax.set_ylim([-.5, num_y_categories - .5])
        ax.set(frame_on=False)
        
        cbar = plt.colorbar(count_image, ax=ax, ticks=[0, 1], shrink=.3)
        cbar.outline.set_visible(False)

        for y_category in range(num_y_categories):
            for x_category in range(num_x_categories):
                if counts[y_category, x_category] == 0:
                    continue
                ax.text(x_category, counts.shape[0]-y_category-1, f'{counts[y_category, x_category]:d}', ha="center", va="center", color="w" if normalized_counts[y_category, x_category] > .4 else 'k')

    def visualizeFeatureEnrichment(
            self, ax,
            keys_x=(), permute_metagenes=True,
            key_y='cluster', order_y=None, ignores_y=(),
            normalizer_raw=None,
            normalizer_avg=None,
            **kwargs,
    ):
        n_y = len(set(self.data[key_y].values) - set(ignores_y))
        
        value_y = self.data[key_y].values
        order_y, value_y = np.unique(value_y, return_inverse=True)
        
        if len(keys_x) == 0: keys_x = self.weight_columns
        keys_x = np.array(keys_x)
        keys_x_old = keys_x
        if tuple(keys_x) == tuple(self.weight_columns) and permute_metagenes: keys_x = keys_x[self.metagene_order]
        n_x = len(keys_x)

        df = self.data[[key_y] + list(keys_x)].copy()
        if normalizer_raw is not None:
            df[keys_x] = normalizer_raw(df[keys_x].values)
        c = df.groupby(key_y)[keys_x].mean().loc[order_y].values
        if normalizer_avg is not None:
            c = normalizer_avg(c)

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
        sigma_x_inverse = self.parameters["sigma_x_inverse"][iteration]
        sigma_x_inverse = sigma_x_inverse[self.metagene_order, :]
        sigma_x_inverse = sigma_x_inverse[:, self.metagene_order]
        sigma_x_inverse = sigma_x_inverse - sigma_x_inverse.mean()
        
        self.plot_metagene_heatmap(ax, sigma_x_inverse, iteration=iteration, **kwargs)
        
    def plot_metagene_heatmap(self, ax, data, iteration=-1, **kwargs):
        vertical_range = np.abs(data).max()
        image = ax.imshow(data, vmin=-vertical_range, vmax=vertical_range, **kwargs)
        ticks = list(range(0, self.hyperparameters["K"] - 1, 5)) + [self.hyperparameters["K"] - 1]

        if len(ax.get_xticks()):
            ax.set_xticks(ticks)
        if ax.get_yticks:
            ax.set_yticks(ticks)
        
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
        ax.set_xlabel('metagene ID')
        ax.set_ylabel('metagene ID')
        
        cbar = plt.colorbar(image, ax=ax, shrink=.3, aspect=20, fraction=0.046, pad=0.04)
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
            yy = self.data.groupby('replicate').get_group(repli)[key].values
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
