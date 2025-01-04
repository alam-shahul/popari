import json
import pickle as pkl
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import scanpy as sc
import scipy
import squidpy as sq
from anndata import AnnData
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.stats import gamma, truncnorm

from popari._canvas import DomainCanvas, MetageneCanvas
from popari.util import convert_adjacency_matrix_to_awkward_array


def sample_gaussian(sigma: NDArray, means: NDArray, N: int = 1, random_state=0) -> NDArray:
    """Sample multivariate Gaussian given a mean vector and covariance matrix.

    Args:
        sigma: covariance matrix
        means: mean vector
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
            z = 2 * rng.random(2) - 1
            if z[0] ** 2 + z[1] ** 2 <= 1:
                r = np.linalg.norm(z)
                x[n_valid, num_samples] = z[0] * np.sqrt(-2 * np.log(r**2) / r**2)
                x[n_valid + 1, num_samples] = z[1] * np.sqrt(-2 * np.log(r**2) / r**2)
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
    """Generate 2D samples that are at least minimum_distance apart from each
    other."""
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


def synthesize_metagenes_nsf(
    num_genes,
    num_spatial_metagenes: int,
    n_nonspatial_metagenes: int,
    spatial_metagene_parameter: float,
    nonspatial_metagene_parameter: float,
    original_metagenes: Optional[NDArray] = None,
    normalize: bool = False,
    random_state: int = 0,
    nonspatial_nonzero_prob: Optional[float] = None,
):
    """Test nsf metagene synthesis."""
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


def sample_normalized_embeddings(Z: NDArray, sigma_x: NDArray, rng: Union[int, np.random.Generator]):
    """Sample embeddings from truncated Gaussian given mean vectors, and project
    to simplex.

    Args:
        Z: mean values for each emebdding dimension
        sigma_x: variance vector for all embedding dimensions

    """

    rng = np.random.default_rng(rng)

    X = np.zeros_like(Z)
    num_cells, num_metagenes = Z.shape
    # TODO: vectorize
    for cell in range(num_cells):
        for metagene in range(num_metagenes):
            X[cell, metagene] = (
                sigma_x[metagene] * truncnorm.rvs(-Z[cell, metagene] / sigma_x[metagene], 100, random_state=rng)
                + Z[cell, metagene]
            )

    X = X * (Z > 0)
    X = X / np.sum(X, axis=1, keepdims=True)

    return X


def synthesize_cell_embeddings_nsf(
    cell_type_labels,
    num_cells,
    cell_type_definitions,
    num_spatial_metagenes,
    n_nonspatial_metagenes=3,
    signal_sigma_x=0.1,
    background_sigma_x=0.2,
    nonspatial_nonzero_prob=0.1,
    sigma_x_scale=1.0,
    random_state=None,
):
    """Generate synthetic cell embeddings."""

    rng = np.random.default_rng(random_state)
    num_metagenes = num_spatial_metagenes + n_nonspatial_metagenes

    _, cell_type_encoded_labels = np.unique(cell_type_labels, return_inverse=True)
    cell_type_assignments = np.zeros((num_cells), dtype="int")
    Z = np.zeros((num_cells, num_metagenes))

    for cell_index, (cell_type, cell_type_encoded) in enumerate(zip(cell_type_labels, cell_type_encoded_labels)):
        cell_type_assignments[cell_index] = cell_type_encoded
        Z[cell_index, :num_spatial_metagenes] = cell_type_definitions[cell_type]

    # Extrinsic factors
    Z[:, num_spatial_metagenes:num_metagenes] = rng.binomial(
        1,
        nonspatial_nonzero_prob,
        size=(num_cells, n_nonspatial_metagenes),
    )
    Z += 1e-6

    return Z, cell_type_assignments


@dataclass
class SimulationParameters:
    """Container for simulation parameters.

    Args:
        num_cells: number of cells to simulate
        num_genes: number of total genes to simulate
        annotation_mode: whether the Canvas is annotating `domain` or `cell_type`
        num_real_metagenes: number of real metagenes
        num_noise_metagenes: number of noise metagenes
        real_metagene_parameter: shape parameter for Gamma distribution from which real metagene
            weights are sampled
        noise_metagene_parameter: shape parameter for Gamma distribution from which noise metagene
            weights are sampled
        spatial_distributions: proportions of cell types in each domain of the simulation
        cell_type_definitions: definitions of simulated cell types by metagene proportion
        metagene_variation_probabilities: variation of metagene weight definitions between metagenes
        domain_key: key in `.obs` where the domain identity of each cell is stored
        width: width of the canvas
        height: height of the canvas
        minimum_distance: minimum distance between simulated datapoints
        grid_size: number of rows/columns for grid-based simulation. Alternative to `num_cells` parameter.
        sig_y_scale: standard deviation of additive Gaussian noise used during sampling of gene expression
        sig_x_scale: standard deviation of additive Gaussian noise used during sampling of latent states
        lambda_s: shape parameter used for sampling cell sizes

    """

    num_cells: Optional[int] = None
    num_genes: int = 100
    annotation_mode: str = "domain"
    num_real_metagenes: int = 10
    num_noise_metagenes: int = 3
    real_metagene_parameter: float = 4.0
    noise_metagene_parameter: float = 4.0
    spatial_distributions: dict = None
    cell_type_definitions: dict = None
    metagene_variation_probabilities: Sequence = None
    domain_key: str = "domain"
    width: float = 1.0
    height: float = 1.0
    minimum_distance: float = None
    grid_size: int = None
    sig_y_scale: float = 3.0
    sig_x_scale: float = 3.0
    lambda_s: float = 1.0


def get_grid_coordinates(width: float, height: float, grid_size: int):
    """Generate grid coordinates for simulation.

    Assumes exactly grid_size rows and grid_size columns, therefore yielding (grid_size)^2 rows.

    Args:
        width: width of grid, in relative units
        height: height of grid, in relative units
        grid_size: number of rows/columns in the grid.

    """

    x = np.linspace(0, width, grid_size)
    y = np.linspace(0, height, grid_size)
    xv, yv = np.meshgrid(x, y)

    coordinates = np.vstack([xv.flatten() * width, yv.flatten() * height]).T

    return coordinates


class SyntheticDataset(AnnData):
    """Simulated spatial transcriptomics dataset.

    Uses AnnData as a base class, with additional methods for simulation.

    """

    def __init__(
        self,
        replicate_name: Union[int, str],
        parameters: SimulationParameters,
        random_state: Union[int, np.random.Generator] = None,
        verbose: int = 0,
    ):
        """Generate random coordinates (as well as expression values) for a
        single ST FOV."""

        self.params = parameters
        self.verbose = verbose

        try:
            self.params.num_cells = self.params.num_cells or self.params.grid_size**2
        except TypeError as e:
            raise ValueError(
                "At least one of `num_cells` or `grid_size` must be defined in the input `SimulationParameters` dataclass object.",
            )

        invalid_definition_lengths = [
            (len(cell_type_definition) != self.params.num_real_metagenes)
            for cell_type_definition in self.params.cell_type_definitions.values()
        ]
        if (len(self.params.metagene_variation_probabilities) != self.params.num_real_metagenes) or np.any(
            invalid_definition_lengths,
        ):
            raise ValueError(
                "The dimensions of simulation parameters must be aligned. Please "
                "check that the `metagene_variation_probabilities`, `cell_type_definitions`"
                " and `num_real_metagenes` values are all compatible.",
            )

        self.rng = np.random.default_rng(random_state)

        dummy_expression = np.zeros((self.params.num_cells, self.params.num_genes))

        super().__init__(X=dummy_expression)

        self.name = f"{replicate_name}"

        self.uns["domain_names"] = list(self.params.spatial_distributions.keys())

        self.uns["simulation_parameters"] = vars(parameters)

        if self.params.annotation_mode == "domain":
            self.uns["simulation_parameters"]["cell_type_names"] = list(self.params.cell_type_definitions.keys())
            self.uns["cell_type_definitions"] = {
                self.name: self.params.cell_type_definitions,
            }

        if self.params.grid_size:
            self.obsm["spatial"] = get_grid_coordinates(self.params.width, self.params.height, self.params.grid_size)
        else:
            if not self.params.minimum_distance:
                minimum_distance = 0.75 / np.sqrt(self.params.num_cells)
            tau = minimum_distance * 2.2
            self.obsm["spatial"] = sample_2D_points(
                self.params.num_cells,
                minimum_distance,
                width=self.params.width,
                height=self.params.height,
                random_state=self.rng,
            )

        canvas_constructor = DomainCanvas if self.params.annotation_mode == "domain" else MetageneCanvas
        self.domain_canvas = canvas_constructor(
            self.obsm["spatial"],
            self.uns["domain_names"],
            canvas_width=600,
            density=1,
        )

    def synthesize_metagenes(
        self,
        original_metagenes: Optional[NDArray] = None,
        replicate_variability: Optional[float] = None,
        normalize: bool = True,
    ) -> NDArray:
        """Synthesize related metagenes according to the
        `metagene_variation_probabilities` vector.

        Creates `num_real_metagenes` synthetic metagenes using a random Gamma distribution with
        shape parameter `real_metagene_parameter`. For each metagene `i`, if
        `dropout_probabilities[i] != 0`, randomly permutes a `metagene_variation_probabilities[i]`
        fraction of metagene `i-1` to create metagene `i`; otherwise, creates a new random metagene.
        In addition, adds `n_noise_metagenes` parameterized by a Gamma distribution with shape
        parameter `noise_metagene_parameter`.

        Args:
            original_metagenes: base metagenes to resample. If `None`, metagenes are generated from
                scratch.
            replicate_variability: variation in metagenes definition between simulation replicates
            normalize: whether metagenes should be normalized such that each metagene sums to 1.
            random_state: random state to use for sampling of metagene weights

        Return:
            An array of ground truth metagene weights.

        """

        num_metagenes = self.params.num_real_metagenes + self.params.num_noise_metagenes
        metagenes = np.zeros((num_metagenes, self.params.num_genes))

        #     last_index = None
        for index in range(self.params.num_real_metagenes):
            variation_probability = self.params.metagene_variation_probabilities[index]

            if variation_probability == 0 and not replicate_variability:
                metagene = gamma.rvs(
                    self.params.real_metagene_parameter,
                    size=self.params.num_genes,
                    random_state=self.rng,
                )
                metagenes[index] = metagene
            #             last_index = index
            else:
                if variation_probability == 0:
                    metagene = original_metagenes[index].copy()
                    variation_probability = replicate_variability

                mask = np.full(self.params.num_genes, False)
                masked_genes = self.rng.choice(
                    self.params.num_genes,
                    size=int(variation_probability * self.params.num_genes),
                    replace=False,
                )
                mask[masked_genes] = True
                perturbed_metagene = metagene.copy()

                perturbations = gamma.rvs(self.params.real_metagene_parameter, size=np.sum(mask), random_state=self.rng)
                if original_metagenes is not None:
                    # Use dirichlet distribution
                    perturbations *= np.sum(metagene[mask]) / np.sum(perturbations)
                perturbed_metagene[mask] = perturbations

                metagenes[index] = perturbed_metagene

        #         print(f"Difference between last_index and current index: {((metagenes[index] - metagenes[last_index]) == 0).sum() / num_genes}")

        for index in range(self.params.num_real_metagenes, num_metagenes):
            metagenes[index] = gamma.rvs(
                self.params.noise_metagene_parameter,
                size=self.params.num_genes,
                random_state=self.rng,
            )

        metagenes = metagenes

        if normalize:
            metagenes = metagenes / np.sum(metagenes, axis=1, keepdims=True)

        return metagenes

    def synthesize_cell_embeddings(self, signal_sigma_x=0.1, background_sigma_x=0.03):
        """Generate synthetic cell embeddings."""

        domain_labels = self.obs[self.params.domain_key].to_numpy()

        num_metagenes = self.params.num_real_metagenes + self.params.num_noise_metagenes

        cell_type_assignments = np.zeros((self.params.num_cells), dtype="int")
        Z = np.zeros((self.params.num_cells, num_metagenes))

        cell_types = self.params.cell_type_definitions.keys()

        for domain_index, (domain_name, distribution) in enumerate(self.params.spatial_distributions.items()):
            domain_cells = domain_labels == domain_name
            domain_cell_types, proportions = zip(*distribution.items())

            (cell_indices,) = np.nonzero(domain_cells)
            self.rng.shuffle(cell_indices)

            partition_indices = (np.cumsum(proportions) * len(cell_indices)).astype(int)
            partitions = np.split(cell_indices, partition_indices[:-1])

            cell_type_to_partition = dict(zip(domain_cell_types, partitions))

            for cell_type_index, cell_type in enumerate(cell_types):
                if cell_type not in domain_cell_types:
                    continue

                partition = cell_type_to_partition[cell_type]
                if len(partition) == 0:
                    continue

                cell_type_assignments[partition] = cell_type_index
                Z[partition, : self.params.num_real_metagenes] = self.params.cell_type_definitions[cell_type]

        # Extrinsic factors
        Z[:, self.params.num_real_metagenes : num_metagenes] = 0.05

        sigma_x = np.concatenate(
            [
                np.full(self.params.num_real_metagenes, signal_sigma_x),
                np.full(self.params.num_noise_metagenes, background_sigma_x),
            ],
        )
        sigma_x = sigma_x * self.params.sig_x_scale

        X = sample_normalized_embeddings(Z, sigma_x, rng=self.rng)

        return X, cell_type_assignments

    def simulate_expression(self, predefined_metagenes=None, metagene_magnitudes=None, **simulation_parameters):
        """Simulate expression using parameters."""
        if self.verbose:
            print(f"Simulating {self.params.annotation_mode}-annotated expression...")

        if self.params.annotation_mode == "domain":
            self.simulate_metagene_based_expression(
                predefined_metagenes=predefined_metagenes,
                metagene_magnitudes=metagene_magnitudes,
            )
        elif self.params.annotation_mode == "metagene":
            return simulate_nsf_expression(predefined_metagenes=predefined_metagenes, **simulation_parameters)

    def simulate_metagene_based_expression(
        self,
        metagene_magnitudes=None,
        predefined_metagenes=None,
    ):
        """Simulate metagenes and embeddings following metagene-based SpiceMix
        model."""

        # Get num_genes x num_genes covariance matrix
        num_metagenes = self.params.num_real_metagenes + self.params.num_noise_metagenes
        if isinstance(self.params.sig_y_scale, float):
            self.variance_y = (
                self.params.sig_y_scale * np.identity(self.params.num_genes) / self.params.num_genes
            ) ** 2
        elif isinstance(self.params.sig_y_scale, dict):
            self.variance_y = {
                cell_type: cell_specific_sig_y_scale / self.params.num_genes
                for cell_type, cell_specific_sig_y_scale in self.params.sig_y_scale.items()
            }
            random_key = next(iter(self.variance_y))
            if isinstance(self.variance_y[random_key], float):
                self.variance_y = {
                    cell_type: cell_specific_sig_y_scale * np.identity(self.params.num_genes)
                    for cell_type, cell_specific_sig_y_scale in self.variance_y.items()
                }

            self.variance_y = {
                cell_type: cell_specific_sig_y_scale**2
                for cell_type, cell_specific_sig_y_scale in self.variance_y.items()
            }

        if self.verbose > 1:
            print(f"Gene covariance set: {self.variance_y is not None}")

        magnitudes = None
        if predefined_metagenes is None:
            metagenes = self.synthesize_metagenes()

            self.magnitudes = np.sum(metagenes, axis=1)
            metagenes = metagenes / self.magnitudes[:, np.newaxis]
            metagenes = metagenes.T
        else:
            metagenes = predefined_metagenes
            self.magnitudes = metagene_magnitudes

        self.uns["ground_truth_M"] = {self.name: metagenes}

        X_i, C_i = self.synthesize_cell_embeddings()

        self.S = gamma.rvs(
            num_metagenes / self.params.lambda_s,
            scale=self.params.lambda_s,
            size=self.params.num_cells,
            random_state=self.rng,
        )
        self.obsm["ground_truth_X"] = X_i * self.S[:, np.newaxis]
        cell_type_encoded = C_i.astype(int)
        cell_type = [self.uns["simulation_parameters"]["cell_type_names"][index] for index in cell_type_encoded]
        self.obs["cell_type"] = cell_type
        self.obs["cell_type_encoded"] = cell_type_encoded

        self.sample_noisy_expression()

    def simulate_nsf_expression(
        self,
        num_spatial_metagenes: int,
        num_nonspatial_metagenes: int,
        spatial_metagene_parameter: float,
        nonspatial_metagene_parameter: float,
        lambda_s: float,
        background_expression: float = 0.2,
        predefined_metagenes=None,
        metagene_magnitudes=None,
        rate: float = 10.0,
        nonspatial_nonzero_prob: Optional[float] = None,
    ):
        """Simulate metagenes and embeddings following metagene-based NSF model.

        Args:
            num_spatial_metagenes: Number of spatial metagenes to simulate
            num_nonspatial_metagenes: Number of non-spatial (intrinsic) metagenes to simulate

        """

        num_metagenes = num_spatial_metagenes + num_nonspatial_metagenes
        magnitudes = None
        if predefined_metagenes is None:
            metagenes = synthesize_metagenes_nsf(
                self.num_genes,
                num_spatial_metagenes,
                num_nonspatial_metagenes,
                spatial_metagene_parameter,
                nonspatial_metagene_parameter,
                random_state=self.rng,
                normalize=True,
                nonspatial_nonzero_prob=nonspatial_nonzero_prob,
            )

            self.magnitudes = np.sum(metagenes, axis=1)
            metagenes = metagenes / self.magnitudes[:, np.newaxis]
            metagenes = metagenes.T
        else:
            metagenes = predefined_metagenes
            self.magnitudes = metagene_magnitudes

        self.uns["ground_truth_M"] = {self.name: metagenes}

        X_i, C_i = synthesize_cell_embeddings_nsf(
            self.obs["cell_type"].to_numpy(),
            self.num_cells,
            cell_type_definitions=self.uns["cell_type_definitions"][self.name],
            num_spatial_metagenes=num_spatial_metagenes,
            n_nonspatial_metagenes=num_nonspatial_metagenes,
            random_state=self.rng,
        )

        # self.S = gamma.rvs(num_metagenes, scale=lambda_s, size=self.num_cells)
        self.obsm["ground_truth_X"] = X_i * self.magnitudes

        # self.obsm["ground_truth_X"] = X_i

        cell_type_encoded = C_i.astype(int)
        cell_type = [self.uns["cell_type_names"][index] for index in cell_type_encoded]
        self.obs["cell_type"] = cell_type
        self.obs["cell_type_encoded"] = cell_type_encoded

        self.sample_noisy_expression(background_expression=background_expression, rate=rate)

    def sample_noisy_expression(self, background_expression: float = 0.2, rate=10.0):
        """Samples gene expression from Negative Binomial distribution according
        to SyntheticDataset attributes.

        Uses ``self.variance_y``.

        TODO: make it so that mean expression is stored in .obsm, not in .X (so that this method does
        not mutate mean expression).

        """

        self.X = np.matmul(self.obsm["ground_truth_X"], self.uns["ground_truth_M"][self.name].T)

        if self.params.annotation_mode == "domain":
            for cell, cell_type in zip(range(self.params.num_cells), self.obs["cell_type_encoded"].to_numpy()):
                if isinstance(self.variance_y, dict):
                    cell_type_variance_y = self.variance_y[int(cell_type)]
                else:
                    cell_type_variance_y = self.variance_y

                # Ensure that gene expression is positive
                self.X[cell] = np.abs(sample_gaussian(cell_type_variance_y, self.X[cell], random_state=self.rng))

        elif self.annotation_mode == "metagene":
            self.X += background_expression
            self.X = self.rng.negative_binomial(rate, rate / (self.X + rate))

            self.raw = self
            sc.pp.log1p(self)

    def annotate_domain(self, points=None):
        return self.domain_canvas.annotate_domain(points=points)

    def assign_domain_labels(self):
        if self.params.annotation_mode == "metagene":
            self.domain_canvas.convert_metagenes_to_cell_types()

        domain_kd_tree = self.domain_canvas.generate_domain_kd_tree()
        self.obs[self.params.domain_key] = domain_kd_tree.query(self.obsm["spatial"])
        if self.params.annotation_mode == "metagene":
            self.uns["cell_type_names"] = list(set(self.obs[self.params.domain_key]))
            cell_type_definitions = {
                label: np.zeros(len(self.domain_canvas.domain_names)) for label in self.uns["cell_type_names"]
            }

            self.uns["cell_type_definitions"] = {
                self.name: cell_type_definitions,
            }
            for cell_type in self.uns["cell_type_names"]:
                definition = json.loads(cell_type)
                if None not in definition:
                    self.uns["cell_type_definitions"][self.name][cell_type][definition] = 1

        self.uns["domain_landmarks"] = dict(self.domain_canvas.domains)


class MultiReplicateSyntheticDataset:
    """Synthetic multireplicate dataset to model biological variation and batch
    effects in spatial transcriptomics data."""

    def __init__(
        self,
        replicate_parameters: dict[SimulationParameters],
        dataset_constructor: SyntheticDataset,
        random_state=0,
        verbose=0,
    ):
        self.verbose = verbose
        self.datasets = {}
        self.replicate_parameters = replicate_parameters

        # random.seed(random_state)
        self.rng = np.random.default_rng(random_state)

        for replicate_name in self.replicate_parameters:
            synthetic_dataset = dataset_constructor(
                replicate_name=replicate_name,
                parameters=replicate_parameters[replicate_name],
                random_state=self.rng,
                verbose=self.verbose,
            )
            self.datasets[replicate_name] = synthetic_dataset

    def annotate_replicate_domain(self, replicate_name, points=None):
        print(f"Annotating replicate {replicate_name}")

        return self.datasets[replicate_name].annotate_domain(points=points)

    def __iter__(self):
        yield from self.datasets.values()

    def assign_domain_labels(self):
        for replicate_dataset in self:
            replicate_dataset.assign_domain_labels()

    def simulate_expression(self):
        metagenes = None
        metagene_magnitudes = None
        for replicate_dataset in self:
            if self.verbose:
                print(f"Simulating expression for {replicate_dataset.name}.")

            replicate_dataset.simulate_expression(
                predefined_metagenes=metagenes,
                metagene_magnitudes=metagene_magnitudes,
            )

            metagenes = replicate_dataset.uns["ground_truth_M"][replicate_dataset.name]
            metagene_magnitudes = replicate_dataset.magnitudes

    def calculate_neighbors(self, **neighbors_kwargs):
        coord_type = neighbors_kwargs.pop("coord_type", "generic")
        delaunay = neighbors_kwargs.pop("delaunay", True)
        for dataset in self:
            sq.gr.spatial_neighbors(
                dataset,
                coord_type=coord_type,
                delaunay=delaunay,
                radius=[0, 0.1],
                **neighbors_kwargs,
            )
            dataset.obsp["adjacency_matrix"] = dataset.obsp["spatial_connectivities"]
            dataset.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(dataset.obsp["adjacency_matrix"])
