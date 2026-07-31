"""AnnData-native generators for Popari synthetic datasets.

The main generator constructs each replicate in five stages:

1. initialize an empty grid-based :class:`~anndata.AnnData`;
2. assign spatial domains and sample cell types within each domain;
3. sample metagenes and per-cell metagene embeddings;
4. generate noisy expression from the metagene factorization; and
5. construct the spatial graph and apply optional spatial dropout.

Metagenes are sampled once and shared across all replicates in a generated
simulation. Cell types, embeddings, expression noise, and graph dropout are
sampled separately for each replicate from one reproducible random stream.

"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import squidpy as sq
from scipy.sparse import csr_array
from scipy.stats import gamma, truncnorm

from popari._canvas import DomainCanvas
from popari._sample_axis import SampleAxis
from popari.io import save_anndata
from popari.schema import DATASET_NAME_KEY, SAMPLE_KEY_KEY, SCHEMA_VERSION, SCHEMA_VERSION_KEY
from popari.simulation.recipes import (
    LayerLandmarks,
    SimulationConfig,
    SimulationRecipe,
    SpatialDropoutConfig,
    default_domain_landmarks,
    named_replicates,
    simulation_sweep_output_path,
)
from popari.util import convert_adjacency_matrix_to_awkward_array

SPATIAL_AFFINITY_DEMO_CELL_TYPES = {
    "Type A": [1, 0, 0],
    "Type B": [0, 1, 0],
    "Type C": [0, 0, 1],
}
SPATIAL_AFFINITY_DEMO_SCENARIOS = ("Alternating", "Checkers", "Monotype", "Layers")


@dataclass(frozen=True)
class SimulationResult:
    """Unified dataset and configuration produced by
    :func:`generate_simulation`."""

    adata: ad.AnnData
    config: SimulationConfig
    recipes: Mapping[str, SimulationRecipe]
    output_path: Path | None = None


def _grid_coordinates(recipe: SimulationRecipe, grid_size: int) -> np.ndarray:
    """Create a regular spatial grid for one recipe.

    The grid contains ``grid_size`` points along each axis and spans
    ``[0, recipe.width]`` by ``[0, recipe.height]``. Coordinates are returned
    in row-major order with shape ``(grid_size**2, 2)``.

    """

    x = np.linspace(0, recipe.width, grid_size)
    y = np.linspace(0, recipe.height, grid_size)
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])


def _create_dataset(name: str, recipe: SimulationRecipe, config: SimulationConfig) -> ad.AnnData:
    """Initialize an empty AnnData replicate and record its simulation schema.

    This creates the expression matrix and spatial coordinates but does not
    yet assign domains, cell types, metagenes, embeddings, or graph edges.
    Recipe and numerical settings are copied into ``.uns`` for provenance.

    """

    num_cells = config.grid_size**2
    dataset = ad.AnnData(X=np.zeros((num_cells, config.num_genes)))
    dataset.popari.name = name
    dataset.obsm["spatial"] = _grid_coordinates(recipe, config.grid_size)
    dataset.uns["domain_names"] = list(recipe.domain_names)
    dataset.uns["simulation_parameters"] = {
        **asdict(config),
        "cell_type_names": list(recipe.cell_type_definitions),
        "domain_key": recipe.domain_key,
        "spatial_distributions": dict(recipe.spatial_distributions),
        "metagene_variation_probabilities": list(recipe.metagene_variation_probabilities),
    }
    dataset.uns["cell_type_definitions"] = {name: dict(recipe.cell_type_definitions)}
    return dataset


def _assign_domains(
    dataset: ad.AnnData,
    recipe: SimulationRecipe,
    landmarks: LayerLandmarks,
) -> None:
    """Assign each cell to its nearest user-supplied domain landmark.

    Domain labels are written to ``dataset.obs[recipe.domain_key]`` and the
    resolved landmark coordinates are retained in ``.uns["domain_landmarks"]``.

    """

    canvas = DomainCanvas(
        dataset.obsm["spatial"],
        list(recipe.domain_names),
        canvas_width=600,
        density=1,
    )
    canvas.load_domains(dict(landmarks))
    dataset.obs[recipe.domain_key] = canvas.generate_domain_kd_tree().query(dataset.obsm["spatial"])
    dataset.uns["domain_landmarks"] = dict(canvas.domains)


def _sample_metagenes(
    recipe: SimulationRecipe,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample gene-loading vectors shared across simulation replicates.

    Each zero entry in ``metagene_variation_probabilities`` starts an
    independent Gamma-distributed metagene. A nonzero entry copies the
    immediately preceding metagene and resamples that fraction of its genes.
    Additional noise metagenes are sampled independently with
    ``noise_metagene_parameter``.

    All metagenes are normalized to sum to one. The returned loading matrix has
    shape ``(num_genes, num_real_metagenes + num_noise_metagenes)``; the second
    return value contains the corresponding post-normalization column sums.

    """

    num_metagenes = recipe.num_real_metagenes + config.num_noise_metagenes
    metagenes = np.zeros((num_metagenes, config.num_genes))

    for index, variation_probability in enumerate(recipe.metagene_variation_probabilities):
        if variation_probability == 0:
            metagene = gamma.rvs(
                config.real_metagene_parameter,
                size=config.num_genes,
                random_state=rng,
            )
        else:
            metagene = metagenes[index - 1].copy()
            varied_genes = rng.choice(
                config.num_genes,
                size=int(variation_probability * config.num_genes),
                replace=False,
            )
            metagene[varied_genes] = gamma.rvs(
                config.real_metagene_parameter,
                size=len(varied_genes),
                random_state=rng,
            )
        metagenes[index] = metagene

    for index in range(recipe.num_real_metagenes, num_metagenes):
        metagenes[index] = gamma.rvs(
            config.noise_metagene_parameter,
            size=config.num_genes,
            random_state=rng,
        )

    metagenes /= metagenes.sum(axis=1, keepdims=True)
    magnitudes = metagenes.sum(axis=1)
    return metagenes.T, magnitudes


def _sample_normalized_embeddings(
    means: np.ndarray,
    standard_deviations: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample nonnegative cell embeddings around prescribed mean mixtures.

    Every active cell-metagene entry is sampled from a lower-truncated normal
    distribution with the supplied mean and metagene-specific standard
    deviation. Entries whose means are zero remain exactly zero, and each
    cell's sampled vector is normalized to sum to one.

    ``means`` and the returned array both have shape
    ``(num_cells, num_metagenes)``.

    """

    embeddings = np.zeros_like(means)
    for cell_index in range(means.shape[0]):
        for metagene_index in range(means.shape[1]):
            mean = means[cell_index, metagene_index]
            sigma = standard_deviations[metagene_index]
            embeddings[cell_index, metagene_index] = sigma * truncnorm.rvs(-mean / sigma, 100, random_state=rng) + mean
    embeddings *= means > 0
    return embeddings / embeddings.sum(axis=1, keepdims=True)


def _sample_embeddings(
    dataset: ad.AnnData,
    recipe: SimulationRecipe,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cell types and metagene embeddings for one replicate.

    Within each spatial domain, cells are shuffled and partitioned according
    to the recipe's cell-type proportions. Cell-type definitions provide the
    mean activity of real metagenes, while every noise metagene receives a
    small background mean. The resulting simplex embeddings are multiplied by
    Gamma-distributed cell-size factors.

    Returns an embedding matrix of shape ``(num_cells, num_metagenes)`` and an
    integer cell-type assignment vector of shape ``(num_cells,)``.

    """

    num_metagenes = recipe.num_real_metagenes + config.num_noise_metagenes
    cell_type_names = list(recipe.cell_type_definitions)
    assignments = np.zeros(dataset.n_obs, dtype=int)
    means = np.zeros((dataset.n_obs, num_metagenes))
    domain_labels = dataset.obs[recipe.domain_key].to_numpy()

    for domain_name, distribution in recipe.spatial_distributions.items():
        cell_indices = np.flatnonzero(domain_labels == domain_name)
        rng.shuffle(cell_indices)
        cell_types, proportions = zip(*distribution.items())
        boundaries = (np.cumsum(proportions) * len(cell_indices)).astype(int)
        partitions = np.split(cell_indices, boundaries[:-1])
        cells_by_type = dict(zip(cell_types, partitions))

        for cell_type_index, cell_type in enumerate(cell_type_names):
            partition = cells_by_type.get(cell_type)
            if partition is None or len(partition) == 0:
                continue
            assignments[partition] = cell_type_index
            means[partition, : recipe.num_real_metagenes] = recipe.cell_type_definitions[cell_type]

    means[:, recipe.num_real_metagenes :] = 0.05
    sigma = np.concatenate(
        [
            np.full(recipe.num_real_metagenes, 0.1),
            np.full(config.num_noise_metagenes, 0.03),
        ],
    )
    embeddings = _sample_normalized_embeddings(means, sigma * config.sig_x_scale, rng)
    sizes = gamma.rvs(
        num_metagenes / config.lambda_s,
        scale=config.lambda_s,
        size=dataset.n_obs,
        random_state=rng,
    )
    return embeddings * sizes[:, np.newaxis], assignments


def _sample_gaussian(covariance: np.ndarray, means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one multivariate-normal sample with a custom Box-Muller sampler.

    ``covariance`` has shape ``(num_genes, num_genes)`` and ``means`` has shape
    ``(num_genes,)``. The custom implementation preserves the random-sampling
    behavior of the original simulation framework.

    """

    num_dimensions = len(covariance)
    cholesky = np.linalg.cholesky(covariance)
    num_normals = num_dimensions + (num_dimensions % 2)
    standard_normal = np.zeros(num_normals)
    num_valid = 0
    while num_valid < num_normals:
        values = 2 * rng.random(2) - 1
        radius_squared = values[0] ** 2 + values[1] ** 2
        if radius_squared <= 1:
            scale = np.sqrt(-2 * np.log(radius_squared) / radius_squared)
            standard_normal[num_valid : num_valid + 2] = values * scale
            num_valid += 2
    return cholesky @ standard_normal[:num_dimensions] + means


def _expression_covariances(config: SimulationConfig) -> np.ndarray | dict[int, np.ndarray]:
    """Construct isotropic gene-expression noise covariance matrices.

    A scalar ``sig_y_scale`` produces one covariance shared by all cells. A
    mapping produces one covariance per encoded cell type, keyed by integer
    cell-type identifier.

    """

    if isinstance(config.sig_y_scale, Mapping):
        return {
            int(cell_type): (float(scale) * np.identity(config.num_genes) / config.num_genes) ** 2
            for cell_type, scale in config.sig_y_scale.items()
        }
    return (float(config.sig_y_scale) * np.identity(config.num_genes) / config.num_genes) ** 2


def _simulate_expression(
    dataset: ad.AnnData,
    recipe: SimulationRecipe,
    config: SimulationConfig,
    rng: np.random.Generator,
    metagenes: np.ndarray | None = None,
    metagene_magnitudes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Populate one replicate with latent states and noisy gene expression.

    Supplied metagenes are reused verbatim, allowing the first replicate's
    sampled programs to be shared by later replicates. Otherwise, metagenes
    are sampled for this dataset. Ground-truth loadings and embeddings are
    stored through ``dataset.simulation``. Expected expression is computed as
    ``ground_truth_X @ ground_truth_M.T`` and each cell receives additive
    multivariate Gaussian noise followed by an absolute-value transform.

    Returns the metagene matrix and magnitudes that should be reused for the
    next replicate.

    """

    if metagenes is None:
        metagenes, metagene_magnitudes = _sample_metagenes(recipe, config, rng)

    embeddings, assignments = _sample_embeddings(dataset, recipe, config, rng)
    dataset.simulation.ground_truth_M = metagenes
    dataset.simulation.ground_truth_X = embeddings
    dataset.uns["ground_truth_metagene_magnitudes"] = metagene_magnitudes
    dataset.obs["cell_type_encoded"] = assignments
    cell_type_names = list(recipe.cell_type_definitions)
    dataset.obs["cell_type"] = [cell_type_names[index] for index in assignments]

    expression = dataset.simulation.ground_truth_expression
    covariances = _expression_covariances(config)
    for cell_index, cell_type in enumerate(assignments):
        covariance = covariances[int(cell_type)] if isinstance(covariances, dict) else covariances
        expression[cell_index] = np.abs(_sample_gaussian(covariance, expression[cell_index], rng))
    dataset.X = expression
    return metagenes, metagene_magnitudes


def _calculate_grid_neighbors(adata: ad.AnnData, sample_axis: SampleAxis, n_neighs: int = 4) -> None:
    """Construct one block-diagonal Squidpy grid graph.

    The sparse connectivity matrix is exposed as
    ``.obsp["adjacency_matrix"]`` and an Awkward neighbor-list representation
    is stored in ``.obsm["adjacency_list"]``.

    """

    graph_blocks = []
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        sample_adata = adata[indices].copy()
        sq.gr.spatial_neighbors(sample_adata, coord_type="grid", n_neighs=n_neighs, delaunay=False)
        graph = sample_adata.obsp["spatial_connectivities"].tocoo()
        graph_blocks.append(
            csr_array(
                (graph.data, (indices[graph.row], indices[graph.col])),
                shape=(adata.n_obs, adata.n_obs),
            ),
        )

    adjacency = sum(graph_blocks[1:], start=graph_blocks[0])
    adata.obsp["spatial_connectivities"] = adjacency
    adata.obsp["adjacency_matrix"] = adjacency.copy()
    adata.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(adjacency)


def _adjacency_lists(adjacency_matrix) -> list[list[int]]:
    """Convert a sparse adjacency matrix into one neighbor list per cell."""

    adjacency_matrix = adjacency_matrix.tocoo()
    adjacency_lists = [[] for _ in range(adjacency_matrix.shape[0])]
    for row, column in zip(*adjacency_matrix.nonzero()):
        adjacency_lists[row].append(column)
    return adjacency_lists


def _apply_spatial_dropout(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    dropout: SpatialDropoutConfig,
) -> None:
    """Set expression to zero at graph-separated cells for every gene.

    For each gene, cells are visited in random order and greedily selected only
    when none of their already selected neighbors excludes them. Up to
    ``dropout.sparsity * n_obs`` selected cells then receive zero expression.
    This creates spatially dispersed dropout rather than independent dropout.

    """

    rng = np.random.default_rng(dropout.random_state)
    for sample in sample_axis.names:
        indices = sample_axis.indices(sample)
        adjacency_lists = _adjacency_lists(adata.obsp["adjacency_matrix"][indices][:, indices])
        num_dropout_cells = int(np.rint(dropout.sparsity * len(indices)))
        for gene_index in range(adata.n_vars):
            excluded = set()
            independent_set = []
            for cell_index in rng.permutation(len(indices)):
                if cell_index not in excluded:
                    independent_set.append(int(cell_index))
                    excluded.update(adjacency_lists[cell_index])
            adata.X[indices[independent_set[:num_dropout_cells]], gene_index] = 0


def generate_simulation(
    config: SimulationConfig | None = None,
    recipes: Mapping[str, SimulationRecipe] | None = None,
    replicates: Mapping[str, str] | None = None,
    domains: Mapping[str, LayerLandmarks] | None = None,
    dropout: SpatialDropoutConfig | None = None,
    output_path: Path | None = None,
    calculate_neighbors: bool | None = None,
    verbose: int = 0,
) -> SimulationResult:
    """Generate related synthetic datasets with shared metagenes."""

    from popari.simulation.recipes import default_hierarchical_cortex_recipes

    # Resolve defaults and validate the requested replicate recipes.
    config = config or SimulationConfig()
    recipes = dict(recipes or default_hierarchical_cortex_recipes())
    replicates = replicates or named_replicates(next(iter(recipes)))
    domains = domains or {}

    unknown_recipes = set(replicates.values()) - set(recipes)
    if unknown_recipes:
        raise ValueError(f"Unknown recipes in replicate specification: {sorted(unknown_recipes)}")

    # Generate each replicate while sharing metagenes across datasets.
    rng = np.random.default_rng(config.random_state)
    datasets = []
    shared_metagenes = None
    shared_magnitudes = None
    for replicate_name, recipe_name in replicates.items():
        recipe = recipes[recipe_name]
        dataset = _create_dataset(replicate_name, recipe, config)
        recipe_domains = (
            domains[recipe_name] if recipe_name in domains else default_domain_landmarks(recipe_name, recipe)
        )
        _assign_domains(dataset, recipe, recipe_domains)
        shared_metagenes, shared_magnitudes = _simulate_expression(
            dataset,
            recipe,
            config,
            rng,
            metagenes=shared_metagenes,
            metagene_magnitudes=shared_magnitudes,
        )
        datasets.append(dataset)
        if verbose:
            print(f"Simulated {replicate_name}.")

    # Finalize observation metadata before constructing the unified object.
    for dataset, recipe_name in zip(datasets, replicates.values()):
        recipe = recipes[recipe_name]
        dataset.obs_names = [f"{dataset.popari.name}_{index}" for index in range(dataset.n_obs)]
        dataset.obs["batch"] = dataset.popari.name
        for key in ("batch", "cell_type", recipe.domain_key):
            dataset.obs[key] = dataset.obs[key].astype("category")

    adata = ad.concat(
        datasets,
        index_unique=None,
        join="inner",
        merge="same",
        uns_merge="same",
    )
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["batch"] = adata.obs["batch"].cat.reorder_categories(list(replicates), ordered=True)
    adata.uns[DATASET_NAME_KEY] = "multisample" if len(replicates) > 1 else next(iter(replicates))
    adata.uns[SAMPLE_KEY_KEY] = "batch"
    adata.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    adata.simulation.ground_truth_M = shared_metagenes
    adata.uns["ground_truth_metagene_magnitudes"] = shared_magnitudes
    sample_axis = SampleAxis.from_anndata(adata, sample_key="batch")

    # Construct spatial graphs and optionally apply graph-aware dropout.
    should_calculate_neighbors = config.calculate_neighbors if calculate_neighbors is None else calculate_neighbors
    selected_dropout = dropout if dropout is not None else config.dropout
    if selected_dropout is not None and not should_calculate_neighbors:
        raise ValueError("Spatial dropout requires calculate_neighbors=True.")

    if should_calculate_neighbors:
        _calculate_grid_neighbors(adata, sample_axis)
    else:
        adata.obsp["adjacency_matrix"] = csr_array((adata.n_obs, adata.n_obs))
        adata.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(
            adata.obsp["adjacency_matrix"],
        )

    if selected_dropout is not None:
        _apply_spatial_dropout(adata, sample_axis, selected_dropout)
    adata.X = csr_array(adata.X)
    adata.popari.validate()

    # Persist the unified result when an output path is requested.
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_anndata(output_path, adata)

    return SimulationResult(
        adata=adata,
        config=config,
        recipes=recipes,
        output_path=output_path,
    )


def generate_simulation_sweep(
    base_config: SimulationConfig,
    sweep: Mapping[str, Sequence[object]],
    recipes: Mapping[str, SimulationRecipe],
    root_path: Path,
    recipe_names: Sequence[str] = ("progenitor", "layer"),
    replicate_planner=None,
    verbose: int = 0,
) -> tuple[SimulationResult, ...]:
    """Generate every configuration in a Cartesian-product sweep."""

    from popari.simulation.recipes import alternating_replicates

    replicate_planner = replicate_planner or alternating_replicates
    results = []
    keys = tuple(sweep)
    for values in itertools.product(*(sweep[key] for key in keys)):
        overrides = dict(zip(keys, values))
        num_replicates = int(overrides.pop("num_replicates", 1))
        config = replace(base_config, **overrides)
        results.append(
            generate_simulation(
                config=config,
                recipes=recipes,
                replicates=replicate_planner(recipe_names, num_replicates),
                output_path=simulation_sweep_output_path(root_path, config, num_replicates),
                verbose=verbose,
            ),
        )
    return tuple(results)


def make_disjoint_metagenes(num_genes: int, num_metagenes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Create normalized metagenes with disjoint gene support."""

    if num_genes < num_metagenes:
        raise ValueError("num_genes must be at least num_metagenes.")
    metagenes = np.zeros((num_genes, num_metagenes))
    for metagene_index, gene_indices in enumerate(np.array_split(np.arange(num_genes), num_metagenes)):
        metagenes[gene_indices, metagene_index] = 1 / len(gene_indices)
    return metagenes, np.ones(num_metagenes)


def spatial_affinity_demo_label_grid(scenario_name: str, grid_size: int) -> np.ndarray:
    """Return the cell-type grid for a named spatial-affinity scenario."""

    rows, columns = np.indices((grid_size, grid_size))
    if scenario_name == "Alternating":
        return np.where(columns % 2 == 0, "Type A", "Type B")
    if scenario_name == "Checkers":
        return np.where((rows + columns) % 2 == 0, "Type A", "Type B")
    if scenario_name == "Monotype":
        return np.full((grid_size, grid_size), "Type A", dtype=object)
    if scenario_name == "Layers":
        labels = np.empty((grid_size, grid_size), dtype=object)
        first_boundary = grid_size // 3
        second_boundary = 2 * grid_size // 3
        labels[:, :first_boundary] = "Type A"
        labels[:, first_boundary:second_boundary] = "Type B"
        labels[:, second_boundary:] = "Type C"
        return labels
    raise ValueError(f"Unknown spatial affinity demo scenario: {scenario_name!r}.")


def four_neighbor_grid_adjacency(grid_size: int) -> csr_array:
    """Return symmetric up/down/left/right grid adjacency."""

    rows = []
    columns = []
    for row in range(grid_size):
        for column in range(grid_size):
            index = row * grid_size + column
            for neighbor_row, neighbor_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if 0 <= neighbor_row < grid_size and 0 <= neighbor_column < grid_size:
                    rows.append(index)
                    columns.append(neighbor_row * grid_size + neighbor_column)
    return csr_array(
        (np.ones(len(rows)), (rows, columns)),
        shape=(grid_size**2, grid_size**2),
    )


def create_spatial_affinity_demo_datasets(
    config: SimulationConfig,
    scenario_names: Sequence[str] = SPATIAL_AFFINITY_DEMO_SCENARIOS,
    noiseless: bool = False,
) -> tuple[ad.AnnData, ...]:
    """Create datasets demonstrating canonical spatial-affinity patterns."""

    metagenes, magnitudes = make_disjoint_metagenes(config.num_genes, num_metagenes=3)
    adjacency = four_neighbor_grid_adjacency(config.grid_size)
    rows, columns = np.indices((config.grid_size, config.grid_size))
    coordinates = np.column_stack([columns.ravel(), rows.ravel()]).astype(float)
    datasets = []

    for scenario_name in scenario_names:
        labels = spatial_affinity_demo_label_grid(scenario_name, config.grid_size).ravel()
        embeddings = np.asarray(
            [SPATIAL_AFFINITY_DEMO_CELL_TYPES[label] for label in labels],
            dtype=float,
        )
        expression = embeddings @ metagenes.T

        if not noiseless:
            rng = np.random.default_rng(config.random_state)
            expressed = expression > 0
            noise = rng.normal(
                scale=config.sig_y_scale / config.num_genes,
                size=expression.shape,
            )
            expression[expressed] = np.abs(expression[expressed] + noise[expressed])

        dataset = ad.AnnData(X=csr_array(expression))
        dataset.popari.name = scenario_name
        dataset.obs_names = [f"{scenario_name}_{index}" for index in range(dataset.n_obs)]
        dataset.var_names = [f"gene_{index}" for index in range(config.num_genes)]
        dataset.obs["scenario"] = scenario_name
        dataset.obs["cell_type"] = labels
        dataset.obs["batch"] = scenario_name
        for key in ("scenario", "cell_type", "batch"):
            dataset.obs[key] = dataset.obs[key].astype("category")
        dataset.obs["cell_type_encoded"] = dataset.obs["cell_type"].cat.codes

        dataset.obsm["spatial"] = coordinates.copy()
        dataset.simulation.ground_truth_X = embeddings
        dataset.simulation.ground_truth_M = metagenes
        dataset.uns["simulation_parameters"] = {
            "cell_type_names": list(SPATIAL_AFFINITY_DEMO_CELL_TYPES),
            "scenario_name": scenario_name,
        }
        dataset.uns["cell_type_definitions"] = {
            scenario_name: SPATIAL_AFFINITY_DEMO_CELL_TYPES,
        }
        dataset.uns["ground_truth_metagene_magnitudes"] = magnitudes
        dataset.uns["domain_names"] = [scenario_name]
        dataset.obsp["adjacency_matrix"] = adjacency.copy()
        dataset.obsp["spatial_connectivities"] = adjacency.copy()
        dataset.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(adjacency)
        datasets.append(dataset)

    return tuple(datasets)


__all__ = [
    SimulationResult.__name__,
    create_spatial_affinity_demo_datasets.__name__,
    four_neighbor_grid_adjacency.__name__,
    generate_simulation.__name__,
    generate_simulation_sweep.__name__,
    make_disjoint_metagenes.__name__,
    spatial_affinity_demo_label_grid.__name__,
    "SPATIAL_AFFINITY_DEMO_SCENARIOS",
]
