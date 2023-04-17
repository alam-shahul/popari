from typing import Union, Sequence, Optional, Tuple
import random

import json

import numpy as np

import scipy
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde, gamma, truncnorm, truncexpon, expon, bernoulli, dirichlet
from scipy.sparse import csr_matrix

from sklearn.decomposition import NMF

import anndata as ad
import scanpy as sc
import squidpy as sq

import umap
import pickle as pkl
import seaborn as sns
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import ListedColormap
        
import seaborn as sns

from _canvas import DomainCanvas, MetageneCanvas

def sample_gaussian(sigma: np.ndarray, means: np.ndarray, N: int = 1, random_state=0):
    """Sample multivariate Gaussian from covariance matrix.
    
    Args:
        sigma: covariance matrix
        N: number of multivariate samples to take
        
    Returns:
        (K, N) sample matrix
    """

    rng = np.random.default_rng(random_state)
    K = len(sigma)
    assert sigma.shape[0] == sigma.shape[1]
    assert len(means) == K
    
    # Box-Muller Method
    L = np.linalg.cholesky(sigma)
    n_z = K + (K % 2)
    x = np.zeros((n_z, N))
    num_samples = 0
    while True:
        n_valid = 0
        while True:
            z = 2*rng.random(2) - 1
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
    x = np.dot(L, x) + np.expand_dims(means, -1)
    
    return np.squeeze(x)

def sample_2D_points(num_points, minimum_distance: float, width: float = 1.0, height: float = 1.0, random_state=0):
    """Generate 2D samples that are at least minimum_distance apart from each other.
    
    """
    # TODO: Implement Poisson disc sampling for a vectorized operation
    rng = np.random.default_rng(random_state)
    
    points = np.zeros((num_points, 2))
    points[0] = rng.random(2) * np.array([width, height])
    for index in range(1, num_points):
        while True:
            point = rng.random((1, 2)) * np.array([width, height])
            distances = cdist(points[:index], point)
            if np.min(distances) > minimum_distance:
                points[index] = point
                break
                
    return points

def synthesize_metagenes(num_genes, num_real_metagenes, n_noise_metagenes, real_metagene_parameter, noise_metagene_parameter,  metagene_variation_probabilities, original_metagenes=None, replicate_variability=None, normalize=True, random_state=0):
    """Synthesize related metagenes according to the metagene_variation_probabilities vector.
    
    Creates num_real_metagenes synthetic metagenes using a random Gamma distribution with
    shape parameter real_metagene_parameter. For each metagene i, if dropout_probabilities[i] != 0,
    randomly permutes a metagene_variation_probabilities[i] fraction of metagene i-1 to create metagene i;
    otherwise, creates a new random metagene. In addition, adds n_noise_metagenes parameterized by
    a Gamma distribution with shape parameter noise_metagene_parameter.
    """
    
    rng = np.random.default_rng(random_state)
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
                
            # mask = bernoulli.rvs(variation_probability, size=num_genes).astype('bool')
            mask = np.full(num_genes, False)
            masked_genes = rng.choice(num_genes, size=int(variation_probability * num_genes), replace=False)
            mask[masked_genes] = True
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

def synthesize_metagenes_nsf(num_genes, num_spatial_metagenes: int, n_nonspatial_metagenes: int,
        spatial_metagene_parameter: float, nonspatial_metagene_parameter: float,
        original_metagenes: np.ndarray =None, normalize: bool = False, random_state: int = 0,
        nonspatial_nonzero_prob: Optional[float]=None):
    """Test nsf metagene synthesis
    
    """
    # TODO: delete the below lines for generalizability
    rng = np.random.default_rng(random_state)
    
    num_metagenes = num_spatial_metagenes + n_nonspatial_metagenes 
    metagenes = np.zeros((num_metagenes, num_genes))

    # last_index = None
    metagene_indices = rng.choice(num_spatial_metagenes, size=num_genes, replace=True)
    metagenes[metagene_indices, np.arange(num_genes)] = spatial_metagene_parameter
    
    if n_nonspatial_metagenes > 0:
        if nonspatial_nonzero_prob is not None:
            for metagene_index in range(n_nonspatial_metagenes):
                gene_indices = rng.binomial(n=1, p=nonspatial_nonzero_prob, size=num_genes).astype(bool)
                metagenes[num_spatial_metagenes + metagene_index, gene_indices] = nonspatial_metagene_parameter
        else:
            metagene_indices = rng.choice(n_nonspatial_metagenes, size=num_genes, replace=True)
            metagenes[metagene_indices + num_spatial_metagenes, np.arange(num_genes)] = nonspatial_metagene_parameter
    
    return metagenes

def sample_normalized_embeddings(Z, sigma_x):
    """Helper function to project simulated embeddings to simplex.
    
    """
    X = np.zeros_like(Z)
    num_cells, num_metagenes = Z.shape
    # TODO: vectorize
    for cell in range(num_cells):
        for metagene in range(num_metagenes):
            X[cell, metagene] = sigma_x[metagene] * truncnorm.rvs(-Z[cell, metagene]/sigma_x[metagene], 100) + Z[cell, metagene]
 
    X = X * (Z > 0)
    X = X / np.sum(X, axis=1, keepdims=True)
    
    return X

def synthesize_cell_embeddings(layer_labels, distributions, cell_type_definitions, num_cells, num_real_metagenes, n_noise_metagenes=3, signal_sigma_x=0.1, background_sigma_x=0.03, sigma_x_scale=1.0):
    """Generate synthetic cell embeddings.
    
    """
    
    num_metagenes = num_real_metagenes + n_noise_metagenes
    
    cell_type_assignments = np.zeros((num_cells), dtype='int')
    Z = np.zeros((num_cells, num_metagenes))
    
    cell_types = cell_type_definitions.keys()
    
    for layer_index, (layer_name, distribution) in enumerate(distributions.items()):
        layer_cells = layer_labels == layer_name
        distribution = distributions[layer_name]
        layer_cell_types, proportions = zip(*distribution.items())
        
        cell_indices, = np.nonzero(layer_cells)
        random.shuffle(cell_indices)

        partition_indices = (np.cumsum(proportions) * len(cell_indices)).astype(int)
        partitions = np.split(cell_indices, partition_indices[:-1])


        cell_type_to_partition = dict(zip(layer_cell_types, partitions))
        
        for cell_type_index, cell_type in enumerate(cell_types):
            if cell_type not in layer_cell_types:
                continue
                
            partition = cell_type_to_partition[cell_type]
            if len(partition) == 0:
                continue
            
            cell_type_assignments[partition] = cell_type_index
            Z[partition, :num_real_metagenes] = cell_type_definitions[cell_type]
        
   
    # Extrinsic factors
    Z[:, num_real_metagenes:num_metagenes] = 0.05
    
    sigma_x = np.concatenate([np.full(num_real_metagenes, signal_sigma_x), np.full(n_noise_metagenes, background_sigma_x)])
    sigma_x = sigma_x * sigma_x_scale

    X = sample_normalized_embeddings(Z, sigma_x)
    
    return X, cell_type_assignments

def synthesize_cell_embeddings_nsf(cell_type_labels, num_cells, cell_type_definitions, num_spatial_metagenes, n_nonspatial_metagenes=3, signal_sigma_x=0.1, background_sigma_x=0.2, nonspatial_nonzero_prob=0.1, sigma_x_scale=1.0, random_state=None):
    """Generate synthetic cell embeddings.
    
    """
   
    rng = np.random.default_rng(random_state)
    num_metagenes = num_spatial_metagenes + n_nonspatial_metagenes
   
    _, cell_type_encoded_labels = np.unique(cell_type_labels, return_inverse=True)
    cell_type_assignments = np.zeros((num_cells), dtype='int')
    Z = np.zeros((num_cells, num_metagenes))
    
    for cell_index, (cell_type, cell_type_encoded) in enumerate(zip(cell_type_labels, cell_type_encoded_labels)):
        cell_type_assignments[cell_index] = cell_type_encoded
        Z[cell_index, :num_spatial_metagenes] = cell_type_definitions[cell_type]
   
    # Extrinsic factors
    Z[:, num_spatial_metagenes:num_metagenes] = rng.binomial(1, nonspatial_nonzero_prob, size=(num_cells, n_nonspatial_metagenes))
    Z += 1e-6

    return Z, cell_type_assignments

class SyntheticDataset(ad.AnnData):
    """Simulated spatial transcriptomics dataset.
    
    Uses AnnData as a base class, with additional methods for simulation.
    """
    
    def __init__(self, num_cells: int=None, num_genes: int=100, replicate_name: Union[int, str]="default",
            annotation_mode: str = "layer", spatial_distributions: dict =None,
            cell_type_definitions: dict = None, metagene_variation_probabilities: Sequence = None,
            shared_metagenes: np.ndarray = None, width: float =1.0, height: float =1.0,
            minimum_distance: float =None, grid_size: int =None,
            random_state: Union[int, np.random.Generator] = None, verbose: int = 0):
        """Generate random coordinates (as well as expression values) for a single ST FOV.
        
        Args:
            num_cells: number of cells to simulate
            num_genes: number of total genes to simulate
        
        """

        self.verbose = verbose

        if num_cells is not None:
            self.num_cells = num_cells
        elif grid_size is not None:
            self.grid_size = grid_size
            self.num_cells = self.grid_size ** 2

        self.num_genes = num_genes
        self.annotation_mode = annotation_mode

        self.rng = np.random.default_rng(random_state)
        
        dummy_expression = np.zeros((self.num_cells, num_genes))
        
        ad.AnnData.__init__(self, X=dummy_expression)
        
        self.name = f"{replicate_name}"
        
        self.uns["domain_names"] = list(spatial_distributions.keys())
        self.uns["width"] = self.width = width
        self.uns["height"] = self.height = height
            
        self.uns["domain_distributions"] = {
            self.name: spatial_distributions
        }
        self.uns["metagene_variation_probabilities"] = {
            self.name: metagene_variation_probabilities
        }

        if self.annotation_mode == "layer":
            self.uns["cell_type_names"] = list(cell_type_definitions.keys())
            self.uns["cell_type_definitions"] = {
                self.name: cell_type_definitions
            }
      
        if grid_size:
            x = np.linspace(0, self.width, grid_size)
            y = np.linspace(0, self.height, grid_size)
            xv, yv = np.meshgrid(x, y)

            self.obsm["spatial"] = np.vstack([xv.flatten() * self.width, yv.flatten() * self.height]).T
        else:
            if not minimum_distance:
                minimum_distance = 0.75 / np.sqrt(self.num_cells)
            tau = minimum_distance * 2.2
            self.obsm["spatial"] = sample_2D_points(self.num_cells, minimum_distance, width=self.uns["width"], height=self.uns["height"], random_state=self.rng)
           
        canvas_constructor = DomainCanvas if self.annotation_mode == "layer" else MetageneCanvas
        self.domain_canvas = canvas_constructor(self.obsm["spatial"], self.uns["domain_names"], canvas_width=600, density=1)
        
    def simulate_expression(self, predefined_metagenes=None, **simulation_parameters):
        """Simulate expression using parameters.

        """
        if self.verbose:
            print(f"Simulating {self.annotation_mode}-annotated expression...")

        if self.annotation_mode == "layer":
            return self.simulate_metagene_based_expression(predefined_metagenes=predefined_metagenes, **simulation_parameters)
        elif self.annotation_mode == "metagene":
            return self.simulate_nsf_expression(predefined_metagenes=predefined_metagenes, **simulation_parameters)
    
    def simulate_metagene_based_expression(self, num_real_metagenes, num_noise_metagenes, real_metagene_parameter, noise_metagene_parameter,
                                           sigY_scale, sigX_scale, lambda_s, predefined_metagenes=None):
        """Simulate metagenes and embeddings following metagene-based SpiceMix model.

        
        """
        
        # Get num_genes x num_genes covariance matrix
        self.sig_y = sigY_scale
        num_metagenes = num_real_metagenes + num_noise_metagenes
        if isinstance(self.sig_y, float):
            self.sig_y *= np.identity(self.num_genes) / self.num_genes
            self.variance_y = (self.sig_y**2)
        elif isinstance(self.sig_y, dict):
            self.sig_y = {cell_type: cell_specific_sig_y / self.num_genes for cell_type, cell_specific_sig_y in self.sig_y.items()}
            random_key = next(iter(self.sig_y))
            if isinstance(self.sig_y[random_key], float):
                self.sig_y = {cell_type: cell_specific_sig_y * np.identity(self.num_genes) for cell_type, cell_specific_sig_y in self.sig_y.items()}
            
            self.variance_y = {cell_type: cell_specific_sig_y ** 2 for cell_type, cell_specific_sig_y in self.sig_y.items()}
        
        if predefined_metagenes is None:
            metagenes = synthesize_metagenes(self.num_genes, num_real_metagenes,
                num_noise_metagenes, real_metagene_parameter, noise_metagene_parameter, metagene_variation_probabilities=self.uns["metagene_variation_probabilities"][self.name], random_state=self.rng)
            self.uns["ground_truth_M"] = {self.name: metagenes.T}
        else:
            self.uns["ground_truth_M"] = {self.name: predefined_metagenes}
        
        X_i, C_i = synthesize_cell_embeddings(self.obs["layer"].to_numpy(), self.uns["domain_distributions"][self.name], self.uns["cell_type_definitions"][self.name],
                       self.num_cells, num_real_metagenes, sigma_x_scale=sigX_scale, n_noise_metagenes=num_noise_metagenes, random_state=self.rng)

        self.S = gamma.rvs(num_metagenes, scale=lambda_s, size=self.num_cells)
        self.obsm["ground_truth_X"] = (X_i * self.S[:, np.newaxis])
        cell_type_encoded = C_i.astype(int)
        cell_type = [self.uns["cell_type_names"][index] for index in cell_type_encoded]
        self.obs["cell_type"] = cell_type
        self.obs["cell_type_encoded"] = cell_type_encoded

        self.X = np.matmul(self.obsm["ground_truth_X"], self.uns["ground_truth_M"][self.name].T)

        self.sample_noisy_expression()
    
    def simulate_nsf_expression(self, num_spatial_metagenes: int, num_nonspatial_metagenes: int,
            spatial_metagene_parameter: float, nonspatial_metagene_parameter: float,
            lambda_s: float, background_expression: float = 0.2, predefined_metagenes=None,
            metagene_magnitudes=None, rate: float = 10.0, nonspatial_nonzero_prob: Optional[float] = None):
        """Simulate metagenes and embeddings following metagene-based NSF model.

        Args:
            num_spatial_metagenes: Number of spatial metagenes to simulate
            num_nonspatial_metagenes: Number of non-spatial (intrinsic) metagenes to simulate
        
        """
        
        num_metagenes = num_spatial_metagenes + num_nonspatial_metagenes 
        magnitudes = None
        if predefined_metagenes is None:
            metagenes = synthesize_metagenes_nsf(self.num_genes, num_spatial_metagenes,
                    num_nonspatial_metagenes, spatial_metagene_parameter, nonspatial_metagene_parameter,
                    random_state=self.rng, normalize=True, nonspatial_nonzero_prob=nonspatial_nonzero_prob
            )

            self.magnitudes = np.sum(metagenes, axis=1)
            metagenes = metagenes / self.magnitudes[:, np.newaxis]
            metagenes = metagenes.T
        else:
            metagenes = predefined_metagenes
            self.magnitudes = metagene_magnitudes
            
        self.uns["ground_truth_M"] = {self.name: metagenes}
        
        X_i, C_i = synthesize_cell_embeddings_nsf(self.obs["cell_type"].to_numpy(), self.num_cells,
                cell_type_definitions=self.uns["cell_type_definitions"][self.name], num_spatial_metagenes=num_spatial_metagenes,
                n_nonspatial_metagenes=num_nonspatial_metagenes, random_state=self.rng)

        # self.S = gamma.rvs(num_metagenes, scale=lambda_s, size=self.num_cells)
        self.obsm["ground_truth_X"] = (X_i * self.magnitudes)
        
        # self.obsm["ground_truth_X"] = X_i

        cell_type_encoded = C_i.astype(int)
        cell_type = [self.uns["cell_type_names"][index] for index in cell_type_encoded]
        self.obs["cell_type"] = cell_type
        self.obs["cell_type_encoded"] = cell_type_encoded

        self.X = np.matmul(self.obsm["ground_truth_X"], self.uns["ground_truth_M"][self.name].T)
        self.X += background_expression

        self.sample_noisy_expression(rate=rate)
            
    def sample_noisy_expression(self, rate=10.0):
        """Samples gene expression from Negative Binomial distribution according to SyntheticDataset attributes.

        Uses ``self.variance_y``.

        TODO: make it so that mean expression is stored in .obsm, not in .X (so that this method does
        not mutate mean expression).

        """

        if self.annotation_mode == "layer":
            for cell, cell_type in zip(range(self.num_cells), self.obs["cell_type_encoded"].to_numpy()):
                if isinstance(self.variance_y, dict):
                    cell_type_variance_y = self.variance_y[int(cell_type)]
                else:
                    cell_type_variance_y = self.variance_y

                # Ensure that gene expression is positive
                self.X[cell] = np.abs(sample_gaussian(cell_type_variance_y, self.X[cell]))

        elif self.annotation_mode == "metagene": 
            self.X =  self.rng.negative_binomial(rate, rate/(self.X + rate))

            self.raw = self
            sc.pp.log1p(self)
            
    def annotate_layer(self, points=None):
        return self.domain_canvas.annotate_domain(points=points)
        
    def assign_layer_labels(self):
        if self.annotation_mode == "metagene":
            self.domain_canvas.convert_metagenes_to_cell_types()

        domain_kd_tree = self.domain_canvas.generate_domain_kd_tree()
        domain_key = "layer" if self.annotation_mode == "layer" else "cell_type"
        self.obs[domain_key] = domain_kd_tree.query(self.obsm["spatial"])
        if self.annotation_mode == "metagene":
            self.uns["cell_type_names"] = list(set(self.obs[domain_key]))
            cell_type_definitions = {
                    label: np.zeros(len(self.domain_canvas.domain_names)) for label in self.uns["cell_type_names"]
            }

            self.uns["cell_type_definitions"] = {
                self.name: cell_type_definitions
            }
            for cell_type in self.uns["cell_type_names"]:
                definition = json.loads(cell_type)
                if None not in definition:
                    self.uns["cell_type_definitions"][self.name][cell_type][definition] = 1

        self.uns["domain_landmarks"] = self.domain_canvas.domains

class MultiReplicateSyntheticDataset():
    """Synthetic multireplicate dataset to model biological variation and batch effects
    in spatial transcriptomics data.
    
    """
    
    def __init__(self, num_genes, replicate_parameters, dataset_class, random_state=0, verbose=0): 
        self.verbose = verbose
        self.datasets = {}
        self.replicate_parameters = replicate_parameters

        random.seed(random_state)
        self.rng = np.random.default_rng(random_state)

        for replicate_name in self.replicate_parameters:
            synthetic_dataset = dataset_class(num_genes=num_genes, replicate_name=replicate_name, **self.replicate_parameters[replicate_name], random_state=self.rng, verbose=self.verbose)
            self.datasets[replicate_name] = synthetic_dataset
    
    def annotate_replicate_layer(self, replicate_name, points=None):
        print(f"Annotating replicate {replicate_name}")

        return self.datasets[replicate_name].annotate_layer(points=points)
    
    def assign_layer_labels(self):
        for replicate_dataset in self.datasets.values():
            replicate_dataset.assign_layer_labels()
            
    def simulate_expression(self, replicate_simulation_parameters):
        """Convenience method for shared simulation parameters between all replicates.
        
        """
        
        metagenes = None
        metagene_magnitudes= None
        for replicate_name, replicate_dataset in self.datasets.items():
            print(replicate_name)
            simulation_parameters = replicate_simulation_parameters[replicate_name]
            if replicate_dataset.annotation_mode == "layer":
                replicate_dataset.simulate_expression(predefined_metagenes=metagenes, **simulation_parameters)
            elif replicate_dataset.annotation_mode == "metagene":
                replicate_dataset.simulate_expression(predefined_metagenes=metagenes, metagene_magnitudes=metagene_magnitudes, **simulation_parameters)

            metagenes = replicate_dataset.uns["ground_truth_M"][replicate_name]
            metagene_magnitudes = replicate_dataset.magnitudes
            
    def calculate_neighbors(self, **neighbors_kwargs):
        coord_type = neighbors_kwargs.pop("coord_type") if "coord_type" in neighbors_kwargs else "generic"
        delaunay = neighbors_kwargs.pop("delaunay") if "delaunay" in neighbors_kwargs else True
        for replicate, dataset in self.datasets.items():
            sq.gr.spatial_neighbors(dataset, coord_type=coord_type, delaunay=delaunay, radius=[0, 0.1], **neighbors_kwargs)
            dataset.obsp["adjacency_matrix"] = dataset.obsp["spatial_connectivities"]
            
            num_cells, _ = dataset.obsp["adjacency_matrix"].shape

            adjacency_list = [[] for _ in range(num_cells)]
            for x, y in zip(*dataset.obsp["adjacency_matrix"].nonzero()):
                adjacency_list[x].append(y)

            dataset.obs["adjacency_list"] = adjacency_list

def remove_connectivity_artifacts(sparse_distance_matrix):
    dense_distances = sparse_distance_matrix.toarray()
    distances = sparse_distance_matrix.data
    cutoff = np.percentile(distances, 94.5)
    mask = dense_distances < cutoff
    
    return csr_matrix(dense_distances * mask)

from scipy.spatial import Delaunay
def generate_affinity_mat(p, tau=1.0, delaunay=True):
    if delaunay:
        A = np.zeros((p.shape[0], p.shape[0]))
        D = Delaunay(p)
        for tri in D.simplices:
            A[tri[0], tri[1]] = 1
            A[tri[1], tri[2]] = 1
            A[tri[2], tri[0]] = 1
    else:
        disjoint_nodes = True
        while(disjoint_nodes):
            N = p.shape[0]
            # Construct graph
            D = squareform(pdist(p))
            A = D < tau
            Id = np.identity(N, dtype='bool')
            A = A * ~Id
            G = nx.from_numpy_matrix(A)
            if not nx.is_connected(G):
                # increase tau by 10% and repeat
                tau = 1.1*tau
                print('Graph is not connected, increasing tau to %s', tau)
            else:
                disjoint_nodes = False
    return A
