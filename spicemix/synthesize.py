#!/usr/bin/env python
import random
import numpy as np

import scipy
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde, gamma, truncnorm, truncexpon, expon, bernoulli, dirichlet

import emcee
from numpy.linalg import inv
from numpy.random import multivariate_normal

from sklearn.decomposition import NMF

import umap
import pickle as pkl
import seaborn as sns
import pandas as pd
import networkx as nx
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import ListedColormap
        
import seaborn as sns

def sample_gaussian(sigma, m, N=1):
    """

    TODO: Not exactly sure how this works.
    
    """
    K = len(sigma)
    assert sigma.shape[0] == sigma.shape[1]
    assert len(m) == K
    
    # Box-Muller Method
    L = np.linalg.cholesky(sigma)
    n_z = K + (K % 2)
    x = np.zeros((n_z, N))
    num_samples = 0
    while True:
        n_valid = 0
        while True:
            z = 2*np.random.rand(2) - 1
            if (z[0]**2 + z[1]**2 <= 1):
                r = np.linalg.norm(z)
                x[n_valid, num_samples] = z[0]*np.sqrt(-2*np.log(r**2)/r**2)
                x[n_valid + 1, num_samples] = z[1]*np.sqrt(-2*np.log(r**2)/r**2)
                n_valid += 2
            if n_valid == n_z:
                num_samples += 1
                break
        if num_samples == N:
            break
            
    # if K is odd, there will be one extra sample, so throw it away
    x = x[0:K, :]
    x = np.dot(L, x) + np.expand_dims(m, -1)
    
    return np.squeeze(x)

# def sample_truncated_gaussian(covariance_matrix, means, num_walkers, num_steps=10000):
#     position = multivariate_normal(means, covariance_matrix, size=num_walkers)
    
    
    
def sample_truncated_gaussian(covariance_matrix, means, num_walkers, num_steps=10000):
    """Sample gene expression vectors from a multivariate truncated Gaussian distribution.
    
    See https://stackoverflow.com/a/20188431/13952002 for details
    
    Args:
        covariance_matrix:
        means:
        num_walkers:
    Return:
    
    Todo:
        Finish docstring
    """
    num_genes = len(covariance_matrix)
    bounds = np.tile([0, np.inf], (1, num_genes))
    
    def lnprob_trunc_norm(x, means, bounds, covariance_matrix):
        if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
            return -np.inf
        else:
            return -0.5*(x-means).dot(inv(covariance_matrix)).dot(x-means)
    
    sampler = emcee.EnsembleSampler(num_walkers, num_genes, lnprob_trunc_norm, args = (means, bounds, covariance_matrix), moves=[(emcee.moves.StretchMove(live_dangerously=True), 1)])
    position = multivariate_normal(means, covariance_matrix, size=num_walkers)

#     plt.figure()
#     plt.scatter(position[:, 0], position[:, 1])
    
    position, prob, state = sampler.run_mcmc(position, num_steps, skip_initial_state_check=True)
   
    # Some data may still be negative if the number of steps in the simulation is too low;
    # in this case, simply make them positive
    position = np.abs(position)
#     plt.scatter(position[:, 0], position[:, 1])
#     plt.show()
    
    return position
    
def sample_2D_points(num_points, minimum_distance):
    """Generate 2D samples that are at least minimum_distance apart from each other.
    
    """
    # TODO: Implement Poisson disc sampling for a vectorized operation
    
    points = np.zeros((num_points, 2))
    points[0] = np.random.random_sample(2)
    for index in range(1, num_points):
        while True:
            point = np.random.random_sample((1, 2))
            distances = cdist(points[:index], point)
            if np.min(distances) > minimum_distance:
                points[index] = point
                break
                
    return points


def generate_affinity_matrix(points, tau=1.0, method="delaunay"):
    """Given a set of 2D spatial coordinates, generate an affinity matrix.

    Can optionally use Delaunay triangulation or simple distance thresholding.
    """

    num_cells = len(points)
    if method == "delaunay":
        affinity_matrix = np.zeros((num_cells, num_cells))
        distance_matrix = np.zeros((num_cells, num_cells))
        triangulation = Delaunay(points)
        for triangle in triangulation.simplices:
            affinity_matrix[triangle[0], triangle[1]] = 1
            distance_matrix[triangle[0], triangle[1]] = np.linalg.norm(points[triangle[0]] - points[triangle[1]])
            affinity_matrix[triangle[1], triangle[2]] = 1
            distance_matrix[triangle[1], triangle[2]] = np.linalg.norm(points[triangle[1]] - points[triangle[2]])
            affinity_matrix[triangle[2], triangle[0]] = 1
            distance_matrix[triangle[2], triangle[0]] = np.linalg.norm(points[triangle[2]] - points[triangle[0]])
                  
        threshold = np.percentile(distance_matrix[np.nonzero(distance_matrix)], 95)
        affinity_matrix[distance_matrix > threshold] = 0

    else:
        disjoint_nodes = True
        while(disjoint_nodes):
            N = points.shape[0]
            # Construct graph
            distances = squareform(pdist(p))
            affinity_matrix = distances < tau
            identity_matrix = np.identity(N, dtype='bool')
            affinity_matrix = affinity_matrix * ~identity_matrix
            graph = nx.from_numpy_matrix(affinity_matrix)
            if not nx.is_connected(graph):
                # increase tau by 10% and repeat
                tau = 1.1*tau
                print('Graph is not connected, increasing tau to %s', tau)
            else:
                disjoint_nodes = False
                
    return affinity_matrix


def synthesize_metagenes(num_genes, num_real_metagenes, n_noise_metagenes, real_metagene_parameter, noise_metagene_parameter,  metagene_variation_probabilities, original_metagenes=None, replicate_variability=None, normalize=True):
    """Synthesize related metagenes according to the metagene_variation_probabilities vector.
    
    Creates num_real_metagenes synthetic metagenes using a random Gamma distribution with
    shape parameter real_metagene_parameter. For each metagene i, if dropout_probabilities[i] != 0,
    randomly permutes a metagene_variation_probabilities[i] fraction of metagene i-1 to create metagene i;
    otherwise, creates a new random metagene. In addition, adds n_noise_metagenes parameterized by
    a Gamma distribution with shape parameter noise_metagene_parameter.
    """
    
    num_metagenes = num_real_metagenes + n_noise_metagenes 
    metagenes = np.zeros((num_metagenes, num_genes))
    
    for index in range(num_real_metagenes):
        variation_probability = metagene_variation_probabilities[index]

        if variation_probability == 0 and not replicate_variability:
            metagene = gamma.rvs(real_metagene_parameter, size=num_genes)
            metagenes[index] = metagene
        else:
            if variation_probability == 0:
                metagene = original_metagenes[index].copy()
                variation_probability = replicate_variability
                
            mask = bernoulli.rvs(variation_probability, size=num_genes).astype('bool')
            perturbed_metagene = metagene

            perturbations = gamma.rvs(real_metagene_parameter, size=np.sum(mask))
            if original_metagenes is not None:
                # Use dirichlet distribution
                perturbations *= np.sum(metagene[mask]) / np.sum(perturbations)
            perturbed_metagene[mask] = perturbations
        
            metagenes[index] = perturbed_metagene
            
    for index in range(num_real_metagenes, num_metagenes):
        metagenes[index] = gamma.rvs(noise_metagene_parameter, size=num_genes)
        
    metagenes = metagenes
    
    if normalize:
        metagenes = metagenes / np.sum(metagenes, axis=1, keepdims=True)
       
    return metagenes


def synthesize_cell_embeddings(points, distributions, cell_type_definitions, mask_conditions, num_cells, n_noise_metagenes=3, signal_sigma_x=0.1, background_sigma_x=0.03, sigma_x_scale=1.0):
    """Generate synthetic cell embeddings.
    
    """
    
    num_patterns, num_cell_types, num_real_metagenes = cell_type_definitions.shape
    num_metagenes = num_real_metagenes + n_noise_metagenes
    
    sigma_x = np.concatenate([np.full(num_real_metagenes, signal_sigma_x), np.full(n_noise_metagenes, background_sigma_x)])
    sigma_x = sigma_x * sigma_x_scale

    cell_type = np.zeros((num_cells), dtype='int')
    Z = np.zeros((num_cells, num_metagenes))
    X = np.zeros((num_cells, num_metagenes))
    
    for pattern_index in range(len(mask_conditions)):
        pattern = mask_conditions[pattern_index](points)
        cell_type_definition = cell_type_definitions[pattern_index]
        distribution = distributions[pattern_index]
        
        cell_indices, = np.nonzero(pattern)
        random.shuffle(cell_indices)
        partition_indices = (np.cumsum(distribution) * len(cell_indices)).astype(int)
        partitions = np.split(cell_indices, partition_indices[:-1])
        
        for cell_type_index in range(len(cell_type_definition)):
            cell_type_composition = cell_type_definition[cell_type_index]
            partition = partitions[cell_type_index]
            cell_type[partition] = cell_type_index
            Z[partition, :num_real_metagenes] = cell_type_composition
        
   
    # Extrinsic factors
    Z[:, num_real_metagenes:num_metagenes] = 0.05

    # TODO: vectorize
    for cell in range(num_cells):
        for metagene in range(num_metagenes):
            X[cell, metagene] = sigma_x[metagene]*truncnorm.rvs(-Z[cell, metagene]/sigma_x[metagene], 100) + Z[cell, metagene]
    X = X * (Z > 0)
    X = X.T
    X = X / np.sum(X, axis=0)
    
    return X, cell_type

def perturb_genes(gene_expression, num_metagenes, first_threshold=.2, second_threshold=.2, shape=2.5):
    """Randomly perturb gene expression values.

    """
    genes, num_samples = gene_expression.shape
    random_sample_size = int(first_threshold * genes)
    for sample in range(num_samples):
        random_indices = random.sample(range(genes), random_sample_size)
        gene_expression[random_indices, sample] = gamma.rvs(shape, size=random_sample_size) / float(genes)
    random_sample_size = int(second_threshold * genes)
    indices = random.sample(range(genes), random_sample_size)
    gene_expression[indices, :] = (gamma.rvs(shape, size=(random_sample_size*num_samples)) / float(genes)).reshape((random_sample_size, num_samples))

    return gene_expression

class SyntheticDataset:
    """Synthetic mouse brain cortex dataset.
    
    This class provides methods for initializing a semi-random mouse cortex spatial
    transcriptomics dataset, as well as methods to visualize aspects of the dataset.
    
    """
    
    def __init__(self, distributions, cell_type_definitions, mask_conditions, metagene_variation_probabilities,
                 parameters, parent_directory, shared_metagenes=None, key=''):
        self.num_metagenes = parameters["num_real_metagenes"] + parameters['num_noise_metagenes']
        self.num_cells = parameters['num_cells']
        self.num_genes = parameters["num_genes"]
        
        # TODO: make color work for variable number of colors
        self.colors = {0: 'darkkhaki', 1: 'mediumspringgreen', 2: 'greenyellow', 3: '#95bfa6',
                       4: 'violet', 5: 'firebrick',
                       6: 'deepskyblue', 7: 'darkslateblue'}
        
        print('Synthesizing X, A, and p...')
        self.num_replicates = parameters['num_replicates']
        self.sig_y = parameters['sigY_scale']
        if isinstance(self.sig_y, float):
            self.sig_y *= np.identity(self.num_genes) / self.num_genes
            self.variance_y = (self.sig_y**2)
        elif isinstance(self.sig_y, dict):
            self.sig_y = {cell_type: cell_specific_sig_y / self.num_genes for cell_type, cell_specific_sig_y in self.sig_y.items()}
            random_key = next(iter(self.sig_y))
            if isinstance(self.sig_y[random_key], float):
                self.sig_y = {cell_type: cell_specific_sig_y * np.identity(self.num_genes) for cell_type, cell_specific_sig_y in self.sig_y.items()}
            
            self.variance_y = {cell_type: cell_specific_sig_y ** 2 for cell_type, cell_specific_sig_y in self.sig_y.items()}
        
        # self.sig_y = float(parameters['sigY_scale']) / self.num_genes
        self.metagenes = np.zeros((self.num_replicates, self.num_genes, self.num_metagenes))
        self.Y = np.zeros((self.num_replicates, self.num_genes, self.num_cells))
        self.cell_embeddings = np.zeros((self.num_replicates, self.num_metagenes, self.num_cells))
        self.points = np.zeros((self.num_replicates, self.num_cells, 2))
        self.affinity_matrices = np.zeros((self.num_replicates, self.num_cells, self.num_cells))
        self.cell_types = np.zeros((self.num_replicates, self.num_cells))
        
        minimum_distance = 0.75 / np.sqrt(self.num_cells)
        tau = minimum_distance * 2.2
        for replicate in range(self.num_replicates):
            print('Synthesizing M...')
            if replicate == 0:
                self.metagenes[replicate] = synthesize_metagenes(self.num_genes, parameters["num_real_metagenes"],
                                                  parameters['num_noise_metagenes'],
                                                  parameters["real_metagene_parameter"], parameters["noise_metagene_parameter"],
                                                  metagene_variation_probabilities=metagene_variation_probabilities)
            else:
                self.metagenes[replicate] = synthesize_metagenes(self.num_genes, parameters["num_real_metagenes"],
                                                  parameters['num_noise_metagenes'],
                                                  parameters["real_metagene_parameter"], parameters["noise_metagene_parameter"],
                                                  metagene_variation_probabilities=metagene_variation_probabilities,
                                                  original_metagenes=self.metagenes[replicate-1].T,
                                                  replicate_variability=parameters["replicate_variability"])
        
            p_i = sample_2D_points(self.num_cells, minimum_distance)
            A_i = generate_affinity_matrix(p_i, tau)
            X_i, C_i = synthesize_cell_embeddings(p_i, distributions, cell_type_definitions, mask_conditions, self.num_cells, n_noise_metagenes=parameters["num_noise_metagenes"])

            self.S = gamma.rvs(self.num_metagenes, scale=parameters['lambda_s'], size=self.num_cells)
            self.affinity_matrices[replicate] = A_i
            self.points[replicate] = p_i
            self.cell_embeddings[replicate] = X_i * self.S
            self.cell_types[replicate] = C_i

        print(self.metagenes.sum(axis=1))
        print('Synthesizing Y...')
        for replicate in range(self.num_replicates):
            Y_i = np.matmul(self.metagenes[replicate], self.cell_embeddings[replicate])
            self.Y[replicate] = Y_i
#             variance_y = (self.sig_y**2) * np.identity(self.num_genes)
            
            for cell, cell_type in zip(range(self.num_cells), self.cell_types[replicate]):
                if isinstance(self.variance_y, dict):
                    cell_type_variance_y = self.variance_y[int(cell_type)]
                else:
                    cell_type_variance_y = self.variance_y
                    
                self.Y[replicate][:, cell] = np.abs(sample_gaussian(cell_type_variance_y, Y_i[:, cell]))
                
            if parameters['gene_replace_prob'] > 0 and parameters['element_replace_prob'] > 0:
                self.Y[replicate] = perturb_genes(Y_i, self.num_metagenes, first_threshold=parameters['gene_replace_prob'],
                                        second_threshold=parameters['element_replace_prob'],
                                        shape=self.num_metagenes)
                
        # gene_ind variable is just all genes -- we don't remove any
        # TODO: remove this field
        self.gene_ind = range(self.num_genes)
        
        # create empty Sig_x_inverse, since it is not used in this data generation
        self.sigma_x_inverse = np.zeros((self.num_metagenes, self.num_metagenes))

#         data_subdirectory = 'synthetic_{}_{}_{}_{:.0f}_{}'
#         data_subdirectory = data_subdirectory.format(self.num_cells, self.num_genes, "covariance",
#                                                          parameters['gene_replace_prob']*100, key)
#         
#         self.data_directory = Path(parent_directory) / data_subdirectory
#         self.initialize_data_directory()
        
        print('Finished')

    def initialize_data_directory(self):
        """Initialize data directory structure on user file system.
        
        """
        (self.data_directory / "files").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "logs").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "scripts").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "plots").mkdir(parents=True, exist_ok=True)

    def plot_cells_UMAP(self, replicate=0, latent_space=False, cell_types=None,colors=None,save_figure=False, normalize=True, annotate=False):
        """Plot synthesized cells using UMAP.
        """
        # TODO: fix colors to be more compatible with variable cell types..
        
        if latent_space:
            gene_expression = self.cell_embeddings[replicate].T
        else:
            gene_expression = self.Y[replicate].T
            
        num_cells, num_features = gene_expression.shape
        C_i = self.cell_types[replicate]
        
        if not colors:
            colors = {0: 'darkkhaki', 1: 'mediumspringgreen', 2: 'greenyellow', 3: '#95bfa6',
                      4: 'violet', 5: 'firebrick', 6: 'gold',
                      7: 'deepskyblue', 8: 'darkslateblue', 9: 'gainsboro'}
            
       
        cell_type_names, cell_type_index = np.unique(C_i, return_index=True)
        sort_index = np.argsort(cell_type_names)
        unique_cell_types, cell_type_index = cell_type_names[sort_index], cell_type_index[sort_index]
        palette = sns.color_palette("husl", len(unique_cell_types))
        sns.set_palette(palette)
        
        colormap = ListedColormap(palette)

        if normalize:
            gene_expression = (gene_expression - np.average(gene_expression, axis=0))
            gene_expression_std = gene_expression.std(axis=0)
            for feature in range(num_features):
                if gene_expression_std[feature] != 0:
                    gene_expression[:, feature] = np.divide(gene_expression[:, feature], gene_expression_std[feature])

        # TODO: cleanup unnecessary lines
        gene_expression_reduced = umap.UMAP(
                        n_components=2,
                        #         spread=1,
                        n_neighbors=10,
                        min_dist=0.3,
                        #         learning_rate=100,
                        #         metric='euclidean',
                        #         metric='manhattan',
                        #         metric='canberra',
                        #         metric='braycurtis',
                        #         metric='mahalanobis',
                        #         metric='cosine',
                        #         metric='correlation',
                        ).fit_transform(gene_expression)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_axes([0.1, 0.1, .8, .8])
        for color, cell_type in np.ndenumerate(unique_cell_types):
            index = (C_i == cell_type)
            ax.scatter(gene_expression_reduced[index, 0], gene_expression_reduced[index, 1], alpha=.7, c=colormap(color), label=cell_type)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.yaxis.set_label_position("right")
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        
        if annotate:
            for label, index in zip(unique_cell_types, cell_type_index):
                ax.annotate(label, (gene_expression_reduced[index, 0], gene_expression_reduced[index, 1]))

        plt.show()
        
        if save_figure:
            plt.savefig(self.data_directory / "plots" / 'synthesized_data_umap.png')

    def plot_metagenes(self, order_genes=True):
        """Plot density map of metagenes in terms of constituent genes.
        
        """
        
        if order_genes:
            mclust = scipy.cluster.hierarchy.linkage(self.metagenes, 'ward')
            mdendro = scipy.cluster.hierarchy.dendrogram(mclust, no_plot=True)
            plt.imshow(self.metagenes[mdendro['leaves']], aspect='auto')
        else:
            plt.imshow(self.metagenes, aspect='auto')
            
        plt.xlabel('Metagene ID')
        plt.ylabel('Gene ID')
        plt.show()

    def plot_cell_types(self, replicate=0, save_figure=False, colors=None):
        """Plot cells in situ using cell type labels.
        
        """
        
        points = self.points[replicate]
        affinity_matrix = self.affinity_matrices[replicate]
        cell_types = self.cell_types[replicate]
        if not colors:
            colors = {0: 'sandybrown', 1: 'lightskyblue',
                      2: 'mediumspringgreen', 3: 'palegreen',
                      4: 'greenyellow', 5: 'darkseagreen',
                      6: 'burlywood', 7: 'orangered', 8: 'firebrick',
                      9: 'gold', 10: 'mediumorchid', 11: 'magenta',
                      12: 'palegoldenrod', 13: 'gainsboro', 14: 'teal',
                      15: 'darkslateblue'}
        df = pd.DataFrame({'X': points[:, 0], 'Y': points[:, 1], 'cell_type': cell_types})
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        
        for (source, destination) in zip(*np.where(affinity_matrix == 1)):
            plt.plot([points[source, 0], points[destination, 0]],
                [points[source, 1], points[destination, 1]], color="gray", linewidth=1)
        
        sns.scatterplot(data=df, x='X', y='Y', hue='cell_type', ax=ax, palette=colors,
                        legend=False, hue_order=list(set(cell_types)), size_norm=10.0)
        plt.show()

    def plot_metagenes_in_situ(self, replicate=0, save_figure=False):
        """Plot metagene values per cell in-situ.
        
        """
        
        points = self.points[replicate]
        cell_embeddings = self.cell_embeddings[replicate]
        for metagene in range(self.num_metagenes):
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_axes([.1, .1, .8, .8])
            ax.set_ylabel('Metagene {}'.format(metagene))
            sca = ax.scatter(points[:, 0], points[:, 1], c=cell_embeddings[metagene], s=23, cmap=plt.get_cmap('Blues'), vmin=0)
            fig.colorbar(sca)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
            if save_figure:
                plt.savefig(self.data_directory / "plots" / ('synthetic_metagene_{}.png'.format(metagene)))

    def plot_hidden_states(self, replicate=0, save_figure=False):
        """Plot synthetic cell embeddings.
        
        """
        cell_embeddings = self.cell_embeddings[replicate]
        image = plt.imshow(cell_embeddings, aspect='auto')
        plt.xlabel('Cell ID')
        plt.ylabel('Metagene ID')
        plt.colorbar(image)
        plt.show()
