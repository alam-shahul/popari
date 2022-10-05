#!/usr/bin/env python
import random
import numpy as np

import scipy
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde, gamma, truncnorm, truncexpon, expon, bernoulli, dirichlet

# import emcee
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
    
    
    
# def sample_truncated_gaussian(covariance_matrix, means, num_walkers, num_steps=10000):
#     """Sample gene expression vectors from a multivariate truncated Gaussian distribution.
#     
#     See https://stackoverflow.com/a/20188431/13952002 for details
#     
#     Args:
#         covariance_matrix:
#         means:
#         num_walkers:
#     Return:
#     
#     Todo:
#         Finish docstring
#     """
#     num_genes = len(covariance_matrix)
#     bounds = np.tile([0, np.inf], (1, num_genes))
#     
#     def lnprob_trunc_norm(x, means, bounds, covariance_matrix):
#         if np.any(x < bounds[:,0]) or np.any(x > bounds[:,1]):
#             return -np.inf
#         else:
#             return -0.5*(x-means).dot(inv(covariance_matrix)).dot(x-means)
#     
#     sampler = emcee.EnsembleSampler(num_walkers, num_genes, lnprob_trunc_norm, args = (means, bounds, covariance_matrix), moves=[(emcee.moves.StretchMove(live_dangerously=True), 1)])
#     position = multivariate_normal(means, covariance_matrix, size=num_walkers)
# 
# #     plt.figure()
# #     plt.scatter(position[:, 0], position[:, 1])
#     
#     position, prob, state = sampler.run_mcmc(position, num_steps, skip_initial_state_check=True)
#    
#     # Some data may still be negative if the number of steps in the simulation is too low;
#     # in this case, simply make them positive
#     position = np.abs(position)
# #     plt.scatter(position[:, 0], position[:, 1])
# #     plt.show()
#     
#     return position
    
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

#     last_index = None
    for index in range(num_real_metagenes):
        variation_probability = metagene_variation_probabilities[index]

        if variation_probability == 0 and not replicate_variability:
            metagene = gamma.rvs(real_metagene_parameter, size=num_genes)
            metagenes[index] = metagene
#             last_index = index
        else:
            if variation_probability == 0:
                metagene = original_metagenes[index].copy()
                variation_probability = replicate_variability
                
            mask = bernoulli.rvs(variation_probability, size=num_genes).astype('bool')
            perturbed_metagene = metagene.copy()

            perturbations = gamma.rvs(real_metagene_parameter, size=np.sum(mask))
            if original_metagenes is not None:
                # Use dirichlet distribution
                perturbations *= np.sum(metagene[mask]) / np.sum(perturbations)
            perturbed_metagene[mask] = perturbations
        
            metagenes[index] = perturbed_metagene
            
#         print(f"Difference between last_index and current index: {((metagenes[index] - metagenes[last_index]) == 0).sum() / num_genes}")
            
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