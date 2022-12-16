from typing import Union, Sequence, Optional
from collections import defaultdict
import random

import numpy as np

import scipy
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde, gamma, truncnorm, truncexpon, expon, bernoulli, dirichlet
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix

from sklearn.decomposition import NMF

import anndata as ad
import squidpy as sq

import umap
import pickle as pkl
import seaborn as sns
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import ListedColormap
        
import seaborn as sns
from ipycanvas import Canvas, hold_canvas
from ipywidgets import Output

def sample_gaussian(sigma: np.ndarray, means: np.ndarray, N: int = 1):
    """Sample multivariate Gaussian from covariance matrix.
    
    Args:
        sigma: covariance matrix
        N: number of multivariate samples to take
        
    Returns:
        (K, N) sample matrix
    """
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
    x = np.dot(L, x) + np.expand_dims(means, -1)
    
    return np.squeeze(x)

def sample_2D_points(num_points, minimum_distance, width=1.0, height=1.0):
    """Generate 2D samples that are at least minimum_distance apart from each other.
    
    """
    # TODO: Implement Poisson disc sampling for a vectorized operation
    
    points = np.zeros((num_points, 2))
    points[0] = np.random.random_sample(2) * np.array([width, height])
    for index in range(1, num_points):
        while True:
            point = np.random.random_sample((1, 2)) * np.array([width, height])
            distances = cdist(points[:index], point)
            if np.min(distances) > minimum_distance:
                points[index] = point
                break
                
    return points

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
                
            # mask = bernoulli.rvs(variation_probability, size=num_genes).astype('bool')
            mask = np.full(num_genes, False)
            masked_genes = np.random.choice(num_genes, size=int(variation_probability * num_genes), replace=False)
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

class DomainKDTree:
    """Wrapper of KDTree for spatial domain simulation.
    
    Using these, can query a "landmark" KD tree in order to obtain spatial domain labels for simulation.
    """
    
    def __init__(self, landmarks: np.ndarray, domain_labels: Sequence[Union[int, str]]):
        self.kd_tree = KDTree(data=landmarks, copy_data=True)
        self.domain_labels = np.array(domain_labels)
        
    def query(self, coordinates):
        """Query a list of simulation spatial coordinates for their layer label.
       
        Args:
            coordinates: list of spatial coordinates
        
        Returns:
            list of domain labels
        """
        distances, indices = self.kd_tree.query(coordinates)
        
        return self.domain_labels[indices]

class DomainCanvas():
    """Annotate spatial domains on a reference ST dataset and use to define domains for simulations.
    
    """
    
    def __init__(self, points: np.ndarray, domain_names: Sequence[str], canvas_width: int = 400, density: float =10):
        """Create DomainCanvas.
        
        Args:
            points (nd.array): background points to use as reference for domain annotation
            domain_names (list of str): list of domains that will be annotated
            canvas_width (int): display width of canvas in Jupyter Notebook, in pixels
            density (float): Try 1 for Visium and 4 for Slide-seq V2
        """
        self.points = points
        self.domain_names = domain_names
        self.domain_option_string = "\n".join(f"{index}: {domain}" for index, domain in enumerate(self.domain_names))
        self.domain_deletion_string = \
            "This domain has already been annotated. What would you like to do? (enter 0 or 1) \n" \
            "0: Further annotate it\n" \
            "1: Delete the existing annotation and start from scratch\n"       
        
        dimensions = self.points.ptp(axis=0)
        lower_boundaries = self.points.min(axis=0)
        self.width, self.height = dimensions + lower_boundaries
 
        self.canvas_width = canvas_width
        self.canvas_height = int((self.height / self.width) * self.canvas_width)

        self.scaling_factor = self.canvas_width / self.width
        
        self.canvas = Canvas(width=self.width * self.scaling_factor, height=self.height * self.scaling_factor)

        self.canvas.layout.width = f"{self.canvas_width}px"
        self.canvas.layout.height = f"{self.canvas_height}px"
        
        self.density = density
        self.radius = 6 / self.density       
        self.canvas.line_width = 2.5 / self.density
        
        self.render_reference_points()
        
        self.domains = defaultdict(list)
        self.colors = {}
        self.out = self.bind_canvas()
        
        return
    
    def render_reference_points(self):
        x, y = self.points.T * self.scaling_factor

        self.canvas.stroke_style = "gray"
        self.canvas.stroke_circles(x, y, self.radius)
        
    def redraw(self):
        self.canvas.clear()
        self.render_reference_points()
        self.load_domains(self.domains)
        
    def bind_canvas(self):
        """Bind mouse click to canvas.
        
        Only used during initialization.
        """
        out = Output()

        @out.capture()
        def handle_mouse_down(x, y):
            with hold_canvas():
                self.canvas.sync_image_data=True
                self.canvas.stroke_circle(x, y, self.radius)
                self.canvas.fill_circle(x, y, self.radius)
                self.canvas.stroke()
                self.canvas.fill()
                self.canvas.sync_image_data=False
                
            self.domains[self.current_domain].append((x / self.scaling_factor, y / self.scaling_factor))
            return

        self.canvas.on_mouse_down(handle_mouse_down)
        
        return out

    def display(self):
        """Display editable canvas.
        
        Click to add landmarks for the domain self.current_domain.
        
        """
        display(self.out)
        return self.canvas
    
    def annotate_domain(self):
        """Create a new domain and display the canvas for annotation.
        
        """
        domain_index = int(input(
            "Choose a domain to annotate (enter an integer):\n"
            f"{self.domain_option_string}\n"
        ))
        if not (0 <= domain_index < len(self.domain_names)):
            raise ValueError(f"`{domain_index}` is not a valid index.")
                                              
        self.current_domain = self.domain_names[domain_index]
        
        if self.current_domain in self.colors:
            start_afresh = int(input(self.domain_deletion_string))
            if start_afresh not in (0, 1):
                raise ValueError(f"`{start_afresh}` is not a valid option.")
                
            if start_afresh:
                del self.domains[self.current_domain]
                self.redraw()
                
            color = self.colors[self.current_domain]

        else:
            r, g, b = np.random.randint(0, 255, size=3)
            color = f"rgb({r}, {g}, {b})"
            self.colors[self.current_domain] = color
            
        self.canvas.stroke_style = color
        self.canvas.fill_style = color
        
        return self.display()
    
    def load_domains(self, domains):
        """Load and display a pre-defined set of domains.
        
        """

        for domain_name in domains:
            self.domains[domain_name] = domains[domain_name]
            coordinates = domains[domain_name]
            
            if domain_name in self.colors:
                color = self.colors[domain_name]
            else:
                r, g, b = np.random.randint(0, 255, size=3)
                color = f"rgb({r}, {g}, {b})"
                self.colors[domain_name] = color

            self.canvas.stroke_style = color
            self.canvas.fill_style = color
            
            x, y = np.array(coordinates).T * self.scaling_factor
            self.canvas.stroke_circles(x, y, self.radius)
            self.canvas.fill_circles(x, y, self.radius)
    
    def generate_domain_kd_tree(self):
        """Export annotated dataset to KD-tree.
        
        """
        domains, coordinates = zip(*self.domains.items())

        domain_labels = [[domain] * len(coordinate) for domain, coordinate in zip(domains, coordinates)]

        flattened_domain_labels = np.concatenate(domain_labels)
        flattened_coordinates = np.concatenate(coordinates)
        annotated_domain_kd_tree = DomainKDTree(flattened_coordinates, flattened_domain_labels)
        
        return annotated_domain_kd_tree

class SyntheticDataset(ad.AnnData):
    """Simulated spatial transcriptomics dataset.
    
    Uses AnnData as a base class, with additional methods for simulation.
    """
    
    def __init__(self, num_cells: int=500, num_genes: int=100, replicate_name: Union[int, str]="default",
                 distributions=None, cell_type_definitions=None, metagene_variation_probabilities=None,
                 domain_landmarks=None, shared_metagenes=None, width=1.0, height=1.0, minimum_distance=None):
        """Generate random coordinates (as well as expression values) for a single ST FOV.
        
        Args:
            num_cells: number of cells to simulate
        
        """
        self.num_cells = num_cells
        self.num_genes = num_genes
        
        dummy_expression = np.zeros((num_cells, num_genes))
        
        ad.AnnData.__init__(self, X=dummy_expression)
        
        self.name = f"{replicate_name}"
        
        self.uns["layer_names"] = list(distributions.keys())
        self.uns["cell_type_names"] = list(cell_type_definitions.keys())
        self.uns["width"] = width
        self.uns["height"] = height
            
        self.uns["domain_landmarks"] = {
            self.name: domain_landmarks
        }
        self.uns["domain_distributions"] = {
            self.name: distributions
        }
        self.uns["metagene_variation_probabilities"] = {
            self.name: metagene_variation_probabilities
        }
        self.uns["cell_type_definitions"] = {
            self.name: cell_type_definitions
        }
        
#         simulation_coordinates = sample_2D_points(num_cells, minimum_distance)
       
        if not minimum_distance:
            minimum_distance = 0.75 / np.sqrt(self.num_cells)
        tau = minimum_distance * 2.2
        
        self.obsm["spatial"] = sample_2D_points(self.num_cells, minimum_distance, width=self.uns["width"], height=self.uns["height"])
        self.domain_canvas = DomainCanvas(self.obsm["spatial"], self.uns["layer_names"], canvas_width=600, density=1)
        
    def simulate_expression(self, mode="metagene_based", predefined_metagenes=None, **simulation_parameters):
            return self.simulate_metagene_based_expression(predefined_metagenes=predefined_metagenes, **simulation_parameters)
    
    def simulate_metagene_based_expression(self, num_real_metagenes, num_noise_metagenes, real_metagene_parameter, noise_metagene_parameter,
                                           sigY_scale, sigX_scale, lambda_s, predefined_metagenes=None):
        """TySc
        
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
            self.uns["ground_truth_M"] = {self.name: synthesize_metagenes(self.num_genes, num_real_metagenes,
                    num_noise_metagenes, real_metagene_parameter, noise_metagene_parameter, metagene_variation_probabilities=self.uns["metagene_variation_probabilities"][self.name
                    ])
            }
        else:
            self.uns["ground_truth_M"] = {self.name: predefined_metagenes}
        
        X_i, C_i = synthesize_cell_embeddings(self.obs["layer"].to_numpy(), self.uns["domain_distributions"][self.name], self.uns["cell_type_definitions"][self.name],
                       self.num_cells, num_real_metagenes, sigma_x_scale=sigX_scale, n_noise_metagenes=num_noise_metagenes)

        self.S = gamma.rvs(num_metagenes, scale=lambda_s, size=self.num_cells)
        self.obsm["ground_truth_X"] = (X_i * self.S[:, np.newaxis])
        cell_type_encoded = C_i.astype(int)
        cell_type = [self.uns["cell_type_names"][index] for index in cell_type_encoded]
        self.obs["cell_type"] = cell_type
        self.obs["cell_type_encoded"] = cell_type_encoded

        self.X = np.matmul(self.obsm["ground_truth_X"], self.uns["ground_truth_M"][self.name])

        self.sample_noisy_expression()
            
    def sample_noisy_expression(self):
        for cell, cell_type in zip(range(self.num_cells), self.obs["cell_type_encoded"].to_numpy()):
            if isinstance(self.variance_y, dict):
                cell_type_variance_y = self.variance_y[int(cell_type)]
            else:
                cell_type_variance_y = self.variance_y

            # Ensure that gene expression is positive
            self.X[cell] = np.abs(sample_gaussian(cell_type_variance_y, self.X[cell]))
            
    def annotate_layer(self):
        return self.domain_canvas.annotate_domain()
        
    def assign_layer_labels(self):
        domain_kd_tree = self.domain_canvas.generate_domain_kd_tree()
        self.obs["layer"] = domain_kd_tree.query(self.obsm["spatial"])
        self.uns["domain_landmarks"] = self.domain_canvas.domains

class MultiReplicateSyntheticDataset():
    """Synthetic multireplicate dataset to model biological variation and batch effects
    in spatial transcriptomics data.
    
    """
    
    def __init__(self, total_cells, num_genes, replicate_parameters, dataset_class, random_state=0): 
        self.datasets = {}
        self.replicate_parameters = replicate_parameters
        np.random.seed(random_state)
        random.seed(random_state)

        for replicate_name in self.replicate_parameters:
            synthetic_dataset = dataset_class(num_genes=num_genes, replicate_name=replicate_name, **self.replicate_parameters[replicate_name])
            self.datasets[replicate_name] = synthetic_dataset
            
        self.total_cells = total_cells
    
    def annotate_replicate_layer(self, replicate_name):
        print(f"Annotating replicate {replicate_name}")

        return self.datasets[replicate_name].annotate_layer()
    
    def assign_layer_labels(self):
        for replicate_dataset in self.datasets.values():
            replicate_dataset.assign_layer_labels()
            
    def simulate_expression(self, replicate_simulation_parameters):
        """Convenience method for shared simulation parameters between all replicates.
        
        """
        
        metagenes = None
        for replicate_name, replicate_dataset in self.datasets.items():
            print(replicate_name)
            simulation_parameters = replicate_simulation_parameters[replicate_name]
            replicate_dataset.simulate_expression(predefined_metagenes=metagenes, **simulation_parameters)
            metagenes = replicate_dataset.uns["ground_truth_M"][replicate_name]
            
    def calculate_neighbors(self):
        for replicate, dataset in self.datasets.items():
            sq.gr.spatial_neighbors(dataset, coord_type="generic", delaunay=True, radius=[0, 0.1])
#             dataset.obsp["spatial_connectivities"] = remove_connectivity_artifacts(dataset.obsp["spatial_distances"])
            dataset.obsp["adjacency_matrix"] = dataset.obsp["spatial_connectivities"]

#             points = dataset.obsm["spatial"]
#             minimum_distance = 0.75 / np.sqrt(len(points))
#             tau = minimum_distance * 2.2
#             dataset.obsp["adjacency_matrix"] = generate_affinity_mat(points, tau=tau)

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

class SpatialBatchEffectDataset(SyntheticDataset):
    """Simulation dataset with gradient-like batch effect.
    
    """
    def __init__(self, num_cells, num_genes, replicate_name, distributions=None, cell_type_definitions=None, metagene_variation_probabilities=None,
                 domain_landmarks=None, shared_metagenes=None):
        super().__init__(num_cells, num_genes, replicate_name, distributions, cell_type_definitions, metagene_variation_probabilities,
                 domain_landmarks, shared_metagenes)
        
    def simulate_expression(self, spatial_function, spatial_metagene_index, mode="metagene_based", **simulation_parameters):
        super().simulate_expression(**simulation_parameters)
        
        spatial_effect = spatial_function(self.obsm["spatial"])

        self.obsm["ground_truth_X"][:, spatial_metagene_index] += spatial_effect
        self.obsm["ground_truth_X"] = (project_embeddings(self.obsm["ground_truth_X"] / self.S[:, np.newaxis])) * self.S[:, np.newaxis]
        
        self.X = np.matmul(self.obsm["ground_truth_X"], self.uns["ground_truth_M"][self.name])
        self.sample_noisy_expression()
