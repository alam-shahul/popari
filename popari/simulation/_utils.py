"""Notebook-facing helpers for Popari simulation examples.

This module has two responsibilities:

* Wrap the legacy :mod:`popari.simulation_framework` classes with small
  dataclass-based configuration objects for the hierarchical simulation
  notebook.
* Build exact grid-pattern AnnData objects for the minimal spatial-affinity
  demo notebook.

"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_array

from popari.io import save_anndata
from popari.simulation_framework import MultiReplicateSyntheticDataset, SimulationParameters, SyntheticDataset
from popari.util import convert_adjacency_matrix_to_awkward_array

LayerLandmarks = Mapping[str, Sequence[Sequence[float]]]
SPATIAL_AFFINITY_DEMO_CELL_TYPES = {
    "Type A": [1, 0, 0],
    "Type B": [0, 1, 0],
    "Type C": [0, 0, 1],
}
SPATIAL_AFFINITY_DEMO_SCENARIOS = ("alternating_columns", "checker_diagonals", "only_type_a", "vertical_thirds")


@dataclass(frozen=True)
class SimulationRecipe:
    """Spatial and cell-type design for one synthetic replicate type."""

    cell_type_definitions: Mapping[str, Sequence[float]]
    spatial_distributions: Mapping[str, Mapping[str, float]]
    metagene_variation_probabilities: Sequence[float]
    width: float = 1.0
    height: float = 1.0
    domain_key: str = "layer"

    @property
    def domain_names(self) -> tuple[str, ...]:
        return tuple(self.spatial_distributions.keys())

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Compatibility alias for older layer-only notebook code."""

        return self.domain_names

    @property
    def layer_distributions(self) -> Mapping[str, Mapping[str, float]]:
        """Compatibility alias for older layer-only notebook code."""

        return self.spatial_distributions

    @property
    def num_real_metagenes(self) -> int:
        return len(self.metagene_variation_probabilities)


@dataclass(frozen=True)
class SimulationConfig:
    """Shared numerical settings for notebook simulations."""

    num_genes: int = 100
    grid_size: int = 36
    num_noise_metagenes: int = 0
    sig_y_scale: float = 1.0
    sig_x_scale: float = 1.0
    real_metagene_parameter: float = 8.0
    noise_metagene_parameter: float = 4.0
    lambda_s: float = 1.0
    random_state: int = 101
    calculate_neighbors: bool = True
    dropout: SpatialDropoutConfig | None = None


@dataclass(frozen=True)
class ReplicateSpec:
    """Ordered mapping from replicate names to recipe names."""

    recipe_by_replicate: Mapping[str, str]
    parameter_overrides: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.recipe_by_replicate)


@dataclass(frozen=True)
class ReplicateConfig:
    """Compatibility wrapper for older single-recipe notebook code."""

    names: Sequence[str] = ("layer_0",)
    parameter_overrides: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def to_spec(self, recipe_name: str) -> ReplicateSpec:
        return ReplicateSpec(
            recipe_by_replicate={name: recipe_name for name in self.names},
            parameter_overrides=self.parameter_overrides,
        )


@dataclass(frozen=True)
class SpatialDropoutConfig:
    """Post-simulation dropout settings for hierarchical examples."""

    sparsity: float = 0.15
    random_state: int = 0


@dataclass(frozen=True)
class GeneratedSimulation:
    """Result object returned by ``generate_simulation``."""

    datasets: tuple[AnnData, ...]
    replicate_names: tuple[str, ...]
    config: SimulationConfig
    recipes: Mapping[str, SimulationRecipe]
    output_path: Path | None = None

    @property
    def recipe(self) -> SimulationRecipe:
        """Compatibility alias for simulations with one recipe."""

        return next(iter(self.recipes.values()))


LayerRecipe = SimulationRecipe
SyntheticDataConfig = SimulationConfig


def default_cortex_layer_recipe() -> SimulationRecipe:
    """Return the layered cortex recipe from the hierarchical simulation
    notebook."""

    cell_type_definitions = {
        "Excitatory L1": [0.5, 0, 0, 0, 0, 1, 0, 0, 0],
        "Excitatory L2": [0.5, 0, 0, 0, 0, 0, 1, 0, 0],
        "Excitatory L3": [0.5, 0, 0, 0, 0, 0, 0, 1, 0],
        "Excitatory L4": [0.5, 0, 0, 0, 0, 0, 0, 0, 1],
        "Inhibitory 1": [0, 0.5, 0, 1, 0, 0, 0, 0, 0],
        "Inhibitory 2": [0, 0.5, 0, 0, 1, 0, 0, 0, 0],
        "Non-Neuron L1": [0, 0, 1, 0, 0, 0.5, 0, 0, 0],
        "Non-Neuron Ubiquitous": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    }

    layer_distributions = {
        "L1": {
            "Excitatory L1": 0.43,
            "Inhibitory 1": 0.1,
            "Non-Neuron L1": 0.4,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L2": {
            "Excitatory L2": 0.93,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L3": {
            "Excitatory L3": 0.53,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.3,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L4": {
            "Excitatory L4": 0.73,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.1,
            "Non-Neuron Ubiquitous": 0.07,
        },
    }

    return SimulationRecipe(
        cell_type_definitions=cell_type_definitions,
        spatial_distributions=layer_distributions,
        metagene_variation_probabilities=[0, 0.3, 0, 0, 0.3, 0, 0.3, 0.3, 0.3],
    )


def default_hierarchical_cortex_recipes() -> dict[str, SimulationRecipe]:
    """Return the named recipe set for the hierarchical cortex example."""

    return {"layer": default_cortex_layer_recipe()}


def default_differential_cortex_recipes() -> dict[str, SimulationRecipe]:
    """Return the progenitor/layer recipes from the differential simulation
    notebook."""

    cell_type_definitions = {
        "Excitatory L1": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Excitatory L2": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Excitatory L3": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Excitatory L4 normal": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "Inhibitory 1": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "Inhibitory 2": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "Non-Neuron L1 normal": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Non-Neuron Ubiquitous": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    metagene_variation_probabilities = [0, 0.25, 0, 0.25, 0, 0.25, 0, 0.1, 0.1, 0.1]

    progenitor_distributions = {
        "progenitor_L1_2": {
            "Excitatory L1": 0.2,
            "Inhibitory 1": 0.06,
            "Non-Neuron L1 normal": 0.3,
            "Excitatory L2": 0.37,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "progenitor_L3_4": {
            "Excitatory L3": 0.26,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.1,
            "Excitatory L4 normal": 0.37,
            "Non-Neuron Ubiquitous": 0.07,
        },
    }
    layer_distributions = {
        "L1": {
            "Excitatory L1": 0.33,
            "Inhibitory 1": 0.1,
            "Non-Neuron L1 normal": 0.5,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L2": {
            "Excitatory L2": 0.93,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L3": {
            "Excitatory L3": 0.53,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.3,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L4": {
            "Excitatory L4 normal": 0.73,
            "Inhibitory 1": 0.1,
            "Inhibitory 2": 0.1,
            "Non-Neuron Ubiquitous": 0.07,
        },
    }

    return {
        "progenitor": SimulationRecipe(
            cell_type_definitions=cell_type_definitions,
            spatial_distributions=progenitor_distributions,
            metagene_variation_probabilities=metagene_variation_probabilities,
            domain_key="layer",
        ),
        "layer": SimulationRecipe(
            cell_type_definitions=cell_type_definitions,
            spatial_distributions=layer_distributions,
            metagene_variation_probabilities=metagene_variation_probabilities,
            domain_key="layer",
        ),
    }


def default_joint_improvement_recipes() -> dict[str, SimulationRecipe]:
    """Return the paired progenitor/layer recipes for the joint-improvement
    simulation.

    This preset preserves the synthetic design used by
    ``copy_joint_improvement_simulation.ipynb``: 11 real metagenes, no required
    noise metagenes, shared cell-type definitions, and two domain-distribution
    regimes named ``progenitor`` and ``layer``.

    """

    cell_type_definitions = {
        "Excitatory L1": [0.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "Excitatory L2": [0.5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Excitatory L3": [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Excitatory L4": [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "Inhibitory L1": [0, 0.5, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Inhibitory L2": [0, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "Inhibitory L3": [0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Inhibitory L4": [0, 0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "Non-Neuron Ubiquitous": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    progenitor_distributions = {
        "L1": {
            "Excitatory L1": 0.53,
            "Inhibitory L1": 0.2,
            "Inhibitory L2": 0.2,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L2": {
            "Excitatory L2": 0.53,
            "Inhibitory L2": 0.2,
            "Inhibitory L3": 0.2,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L3": {
            "Excitatory L3": 0.53,
            "Inhibitory L3": 0.2,
            "Inhibitory L4": 0.2,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L4": {
            "Excitatory L4": 0.53,
            "Inhibitory L3": 0.2,
            "Inhibitory L4": 0.2,
            "Non-Neuron Ubiquitous": 0.07,
        },
    }

    layer_distributions = {
        "L1": {
            "Excitatory L1": 0.2,
            "Excitatory L2": 0.2,
            "Inhibitory L1": 0.53,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L2": {
            "Excitatory L2": 0.2,
            "Excitatory L3": 0.2,
            "Inhibitory L2": 0.53,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L3": {
            "Excitatory L3": 0.2,
            "Excitatory L4": 0.2,
            "Inhibitory L3": 0.53,
            "Non-Neuron Ubiquitous": 0.07,
        },
        "L4": {
            "Excitatory L3": 0.2,
            "Excitatory L4": 0.2,
            "Inhibitory L4": 0.53,
            "Non-Neuron Ubiquitous": 0.07,
        },
    }

    metagene_variation_probabilities = [0, 0.1, 0, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1]

    return {
        "progenitor": SimulationRecipe(
            cell_type_definitions=cell_type_definitions,
            spatial_distributions=progenitor_distributions,
            metagene_variation_probabilities=metagene_variation_probabilities,
            domain_key="domain",
        ),
        "layer": SimulationRecipe(
            cell_type_definitions=cell_type_definitions,
            spatial_distributions=layer_distributions,
            metagene_variation_probabilities=metagene_variation_probabilities,
            domain_key="domain",
        ),
    }


def two_cell_type_ratio_recipes(domain_key: str = "region") -> dict[str, SimulationRecipe]:
    """Return one minimal two-cell-type recipe per A:B tissue ratio."""

    ratio_definitions = {
        "A_to_B_1_to_9": {
            "Type A": 1 / 10,
            "Type B": 9 / 10,
        },
        "A_to_B_1_to_1": {
            "Type A": 1 / 2,
            "Type B": 1 / 2,
        },
        "A_to_B_9_to_1": {
            "Type A": 9 / 10,
            "Type B": 1 / 10,
        },
    }

    return {
        ratio_name: SimulationRecipe(
            cell_type_definitions={
                "Type A": [1, 0],
                "Type B": [0, 1],
            },
            spatial_distributions={
                ratio_name: ratio_distribution,
            },
            metagene_variation_probabilities=[0, 0],
            domain_key=domain_key,
        )
        for ratio_name, ratio_distribution in ratio_definitions.items()
    }


def make_disjoint_metagenes(num_genes: int, num_metagenes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Create predefined metagenes with non-overlapping gene support.

    Returns:
        A ``(num_genes, num_metagenes)`` metagene matrix and a magnitudes vector,
        matching ``SyntheticDataset.simulate_expression``'s predefined-metagene
        interface.

    """

    if num_genes < num_metagenes:
        raise ValueError("`num_genes` must be at least `num_metagenes`.")

    metagenes = np.zeros((num_genes, num_metagenes))
    gene_partitions = np.array_split(np.arange(num_genes), num_metagenes)

    for metagene_index, gene_indices in enumerate(gene_partitions):
        metagenes[gene_indices, metagene_index] = 1 / len(gene_indices)

    return metagenes, np.ones(num_metagenes)


def spatial_affinity_demo_label_grid(scenario_name: str, grid_size: int) -> np.ndarray:
    """Return the cell-type grid for one spatial-affinity demo scenario.

    The returned array has shape ``(grid_size, grid_size)`` and contains
    labels from ``SPATIAL_AFFINITY_DEMO_CELL_TYPES``. Scenario names are the
    values returned by ``spatial_affinity_demo_scenario_names``.

    """

    row_indices, column_indices = np.indices((grid_size, grid_size))

    if scenario_name == "alternating_columns":
        return np.where(column_indices % 2 == 0, "Type A", "Type B")

    if scenario_name == "checker_diagonals":
        return np.where((row_indices + column_indices) % 2 == 0, "Type A", "Type B")

    if scenario_name == "only_type_a":
        return np.full((grid_size, grid_size), "Type A", dtype=object)

    if scenario_name == "vertical_thirds":
        labels = np.empty((grid_size, grid_size), dtype=object)
        first_boundary = grid_size // 3
        second_boundary = (2 * grid_size) // 3
        labels[:, :first_boundary] = "Type A"
        labels[:, first_boundary:second_boundary] = "Type B"
        labels[:, second_boundary:] = "Type C"
        return labels

    raise ValueError(f"Unknown spatial affinity demo scenario: {scenario_name!r}.")


def spatial_affinity_demo_scenario_names() -> tuple[str, ...]:
    """Return scenarios in notebook display order."""

    return SPATIAL_AFFINITY_DEMO_SCENARIOS


def four_neighbor_grid_adjacency(grid_size: int) -> csr_array:
    """Return a symmetric up/down/left/right grid adjacency matrix."""

    rows = []
    cols = []

    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            if row > 0:
                rows.append(index)
                cols.append((row - 1) * grid_size + col)
            if row < grid_size - 1:
                rows.append(index)
                cols.append((row + 1) * grid_size + col)
            if col > 0:
                rows.append(index)
                cols.append(row * grid_size + col - 1)
            if col < grid_size - 1:
                rows.append(index)
                cols.append(row * grid_size + col + 1)

    values = np.ones(len(rows), dtype=float)
    return csr_array((values, (rows, cols)), shape=(grid_size**2, grid_size**2))


def _add_four_neighbor_grid_graph(dataset: AnnData, grid_size: int) -> None:
    """Attach exact four-neighbor graph fields expected by Popari."""

    adjacency_matrix = four_neighbor_grid_adjacency(grid_size)
    dataset.obsp["adjacency_matrix"] = adjacency_matrix
    dataset.obsp["spatial_connectivities"] = adjacency_matrix
    dataset.obsm["adjacency_list"] = convert_adjacency_matrix_to_awkward_array(adjacency_matrix)


def _create_spatial_affinity_demo_dataset(
    scenario_name: str,
    config: SyntheticDataConfig,
    metagenes: np.ndarray,
    metagene_magnitudes: np.ndarray | None = None,
    noiseless: bool = False,
) -> AnnData:
    """Create one exact-pattern dataset for the spatial affinity demo."""

    cell_type_definitions = SPATIAL_AFFINITY_DEMO_CELL_TYPES
    cell_type_names = list(cell_type_definitions)
    label_grid = spatial_affinity_demo_label_grid(scenario_name, config.grid_size)
    labels = label_grid.ravel()
    label_to_index = {label: index for index, label in enumerate(cell_type_names)}

    ground_truth_X = np.zeros((config.grid_size**2, len(cell_type_names)))
    for cell_index, label in enumerate(labels):
        ground_truth_X[cell_index] = cell_type_definitions[label]

    clean_expression = ground_truth_X @ metagenes.T
    expression = clean_expression.copy()

    if not noiseless:
        rng = np.random.default_rng(config.random_state)
        gene_stds = np.full(config.num_genes, config.sig_y_scale / config.num_genes)
        support = clean_expression > 0
        noise = rng.normal(0, gene_stds, size=clean_expression.shape)
        expression[support] = np.abs(clean_expression[support] + noise[support])

    row_indices, column_indices = np.indices((config.grid_size, config.grid_size))
    spatial = np.column_stack([column_indices.ravel(), row_indices.ravel()]).astype(float)
    dataset = AnnData(X=csr_array(expression))
    dataset.name = scenario_name
    dataset.obs_names = [f"{scenario_name}_{index}" for index in range(dataset.n_obs)]
    dataset.obs["scenario"] = scenario_name
    dataset.obs["cell_type"] = labels
    dataset.obs["cell_type_encoded"] = [label_to_index[label] for label in labels]
    dataset.obs["batch"] = scenario_name
    dataset.obs["scenario"] = dataset.obs["scenario"].astype("category")
    dataset.obs["cell_type"] = dataset.obs["cell_type"].astype("category")
    dataset.obs["batch"] = dataset.obs["batch"].astype("category")
    dataset.obsm["spatial"] = spatial
    dataset.obsm["ground_truth_X"] = ground_truth_X
    dataset.var_names = [f"gene_{index}" for index in range(config.num_genes)]
    dataset.uns["dataset_name"] = scenario_name
    dataset.uns["simulation_parameters"] = {
        "cell_type_names": cell_type_names,
        "scenario_name": scenario_name,
    }
    dataset.uns["cell_type_definitions"] = {
        scenario_name: cell_type_definitions,
    }
    dataset.uns["ground_truth_M"] = {
        scenario_name: metagenes,
    }
    dataset.uns["ground_truth_metagene_magnitudes"] = metagene_magnitudes
    dataset.uns["domain_names"] = [scenario_name]

    _add_four_neighbor_grid_graph(dataset, config.grid_size)
    return dataset


def create_spatial_affinity_demo_datasets(
    config: SyntheticDataConfig,
    scenario_names: Sequence[str] | None = None,
    noiseless: bool = False,
) -> tuple[AnnData, ...]:
    """Create exact grid-pattern datasets for the minimal affinity demo.

    All returned datasets share three disjoint metagenes and stable cell-type
    definitions, even when a scenario omits a cell type. This keeps
    ``ground_truth`` Popari initialization aligned as ``Type A -> m0``,
    ``Type B -> m1``, and ``Type C -> m2``.

    """

    scenario_names = tuple(scenario_names or spatial_affinity_demo_scenario_names())
    metagenes, metagene_magnitudes = make_disjoint_metagenes(config.num_genes, num_metagenes=3)
    return tuple(
        _create_spatial_affinity_demo_dataset(
            scenario_name,
            config,
            metagenes,
            metagene_magnitudes,
            noiseless=noiseless,
        )
        for scenario_name in scenario_names
    )


def build_simulation_parameters(
    recipe: SimulationRecipe,
    config: SimulationConfig,
    replicate_override: Mapping[str, object] | None = None,
) -> SimulationParameters:
    """Translate notebook config dataclasses into ``SimulationParameters``."""

    parameters = SimulationParameters(
        num_genes=config.num_genes,
        grid_size=config.grid_size,
        annotation_mode="domain",
        num_real_metagenes=recipe.num_real_metagenes,
        num_noise_metagenes=config.num_noise_metagenes,
        real_metagene_parameter=config.real_metagene_parameter,
        noise_metagene_parameter=config.noise_metagene_parameter,
        spatial_distributions=dict(recipe.spatial_distributions),
        cell_type_definitions=dict(recipe.cell_type_definitions),
        metagene_variation_probabilities=list(recipe.metagene_variation_probabilities),
        domain_key=recipe.domain_key,
        width=recipe.width,
        height=recipe.height,
        sig_y_scale=config.sig_y_scale,
        sig_x_scale=config.sig_x_scale,
        lambda_s=config.lambda_s,
    )

    for key, value in (replicate_override or {}).items():
        if not hasattr(parameters, key):
            raise ValueError(f"`{key}` is not a valid SimulationParameters field.")
        setattr(parameters, key, value)

    return parameters


def named_replicates(recipe_name: str, count: int = 1) -> ReplicateSpec:
    """Create repeated replicates that all use one recipe."""

    return ReplicateSpec({f"{recipe_name}_{index}": recipe_name for index in range(count)})


def alternating_replicates(recipe_names: Sequence[str], count: int) -> ReplicateSpec:
    """Create replicates that cycle through recipe names in order."""

    return ReplicateSpec(
        {
            f"{recipe_names[index % len(recipe_names)]}_{index}": recipe_names[index % len(recipe_names)]
            for index in range(count)
        },
    )


def paired_replicates(recipe_names: Sequence[str], count: int) -> ReplicateSpec:
    """Create ``count`` complete replicate pairs/groups for each recipe name."""

    return ReplicateSpec(
        {f"{recipe_name}_{index}": recipe_name for index in range(count) for recipe_name in recipe_names},
    )


def build_replicate_parameters(
    recipes: Mapping[str, SimulationRecipe],
    config: SimulationConfig,
    replicates: ReplicateSpec,
) -> dict[str, SimulationParameters]:
    """Build one legacy parameter object per replicate."""

    return {
        name: build_simulation_parameters(
            recipes[recipe_name],
            config,
            replicates.parameter_overrides.get(name),
        )
        for name, recipe_name in replicates.recipe_by_replicate.items()
    }


def create_multireplicate_dataset(
    recipes: Mapping[str, SimulationRecipe],
    config: SimulationConfig,
    replicates: ReplicateSpec,
    dataset_constructor: type[SyntheticDataset] = SyntheticDataset,
    verbose: int = 0,
) -> MultiReplicateSyntheticDataset:
    """Create the legacy multireplicate object from dataclass configs."""

    replicate_parameters = build_replicate_parameters(recipes, config, replicates)
    return MultiReplicateSyntheticDataset(
        replicate_parameters=replicate_parameters,
        dataset_constructor=dataset_constructor,
        random_state=config.random_state,
        verbose=verbose,
    )


def default_layer_landmarks(recipe: SimulationRecipe, points_per_layer: int = 9) -> dict[str, np.ndarray]:
    """Generate vertical stripe landmarks for automatic domain assignment."""

    domain_names = recipe.domain_names
    x_centers = np.linspace(
        recipe.width / (2 * len(domain_names)),
        recipe.width - recipe.width / (2 * len(domain_names)),
        len(domain_names),
    )
    y_coordinates = np.linspace(0, recipe.height, points_per_layer)

    return {
        domain_name: np.column_stack([np.full(points_per_layer, x_center), y_coordinates])
        for domain_name, x_center in zip(domain_names, x_centers)
    }


def default_progenitor_landmarks(recipe: SimulationRecipe, points_per_domain: int = 9) -> dict[str, np.ndarray]:
    """Generate broad vertical progenitor-domain landmarks."""

    return default_layer_landmarks(recipe, points_per_layer=points_per_domain)


def default_domain_landmarks(recipe_name: str, recipe: SimulationRecipe) -> dict[str, np.ndarray]:
    """Generate deterministic landmarks for a named preset recipe."""

    if recipe_name == "progenitor":
        return default_progenitor_landmarks(recipe)
    return default_layer_landmarks(recipe)


def load_or_assign_domains(
    multireplicate: MultiReplicateSyntheticDataset,
    recipes: Mapping[str, SimulationRecipe],
    replicates: ReplicateSpec,
    domains: Mapping[str, LayerLandmarks] | None = None,
) -> None:
    """Assign each synthetic cell to the nearest domain landmark."""

    domain_by_recipe = {recipe_name: dict(recipe_domains) for recipe_name, recipe_domains in (domains or {}).items()}
    for recipe_name, recipe in recipes.items():
        domain_by_recipe.setdefault(recipe_name, default_domain_landmarks(recipe_name, recipe))

    for replicate_name, recipe_name in replicates.recipe_by_replicate.items():
        multireplicate.datasets[replicate_name].domain_canvas.load_domains(domain_by_recipe[recipe_name])
    multireplicate.assign_domain_labels()


def simulate_expression_with_metagenes(
    multireplicate: MultiReplicateSyntheticDataset,
    metagenes: np.ndarray,
    metagene_magnitudes: np.ndarray | None = None,
    noiseless: bool = False,
    exact_cell_type_embeddings: bool = False,
    noise_on_expressed_genes_only: bool = False,
) -> None:
    """Simulate expression using a supplied metagene matrix.

    ``noise_on_expressed_genes_only`` keeps off-support genes exactly zero,
    which is useful for toy simulations with intentionally disjoint gene sets.

    """

    if metagene_magnitudes is None:
        metagene_magnitudes = np.ones(metagenes.shape[1])

    for dataset in multireplicate:
        dataset.simulate_expression(
            predefined_metagenes=metagenes,
            metagene_magnitudes=metagene_magnitudes,
        )
        if exact_cell_type_embeddings:
            exact_embeddings = np.zeros((dataset.n_obs, metagenes.shape[1]))
            for cell_type, definition in dataset.params.cell_type_definitions.items():
                cell_mask = dataset.obs["cell_type"].to_numpy() == cell_type
                exact_embeddings[cell_mask, : len(definition)] = definition
            dataset.obsm["ground_truth_X"] = exact_embeddings

        if noiseless:
            dataset.X = dataset.obsm["ground_truth_X"] @ dataset.uns["ground_truth_M"][dataset.name].T
        elif noise_on_expressed_genes_only:
            clean_expression = dataset.obsm["ground_truth_X"] @ dataset.uns["ground_truth_M"][dataset.name].T
            noisy_expression = clean_expression.copy()
            support = clean_expression > 0
            variance_y = dataset.variance_y

            for cell_index in range(dataset.n_obs):
                if isinstance(variance_y, dict):
                    cell_type = int(dataset.obs["cell_type_encoded"].iloc[cell_index])
                    cell_variance_y = variance_y[cell_type]
                else:
                    cell_variance_y = variance_y

                gene_stds = np.sqrt(np.diag(cell_variance_y))
                noise = dataset.rng.normal(0, gene_stds)
                expressed_genes = support[cell_index]
                noisy_expression[cell_index, expressed_genes] = np.abs(
                    clean_expression[cell_index, expressed_genes] + noise[expressed_genes],
                )

            dataset.X = noisy_expression


def calculate_grid_neighbors(multireplicate: MultiReplicateSyntheticDataset, n_neighs: int = 4) -> None:
    """Ask Squidpy to calculate grid neighbors for legacy synthetic datasets."""

    multireplicate.calculate_neighbors(coord_type="grid", n_neighs=n_neighs, delaunay=False)


def adjacency_matrix_to_list(adjacency_matrix) -> list[list[int]]:
    """Convert a sparse adjacency matrix into plain Python neighbor lists."""

    adjacency_matrix = adjacency_matrix.tocoo()
    num_cells, _ = adjacency_matrix.shape
    adjacency_list = [[] for _ in range(num_cells)]
    for x, y in zip(*adjacency_matrix.nonzero()):
        adjacency_list[x].append(y)
    return adjacency_list


def sample_graph_independent_set(
    adjacency_list: Sequence[Sequence[int]],
    sample_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample a greedy independent set from a graph adjacency list."""

    candidate_indices = rng.permutation(len(adjacency_list))[:sample_size]
    valid_indices = []
    excluded_indices = set()

    for index in candidate_indices:
        if index not in excluded_indices:
            valid_indices.append(int(index))
            excluded_indices.update(adjacency_list[index])

    return valid_indices


def apply_spatial_dropout(
    multireplicate: MultiReplicateSyntheticDataset,
    dropout: SpatialDropoutConfig,
) -> None:
    """Apply greedy graph-separated dropout to each gene."""

    rng = np.random.default_rng(dropout.random_state)

    for dataset in multireplicate:
        if "adjacency_matrix" not in dataset.obsp:
            raise ValueError("Neighbors must be calculated before spatial dropout.")

        num_cells, num_genes = dataset.shape
        num_dropout_cells = int(np.rint(dropout.sparsity * num_cells))
        adjacency_list = adjacency_matrix_to_list(dataset.obsp["adjacency_matrix"])

        for gene_index in range(num_genes):
            independent_set = sample_graph_independent_set(adjacency_list, num_cells, rng)
            dropout_cells = independent_set[:num_dropout_cells]
            dataset.X[dropout_cells, gene_index] = 0


def generate_simulation(
    config: SimulationConfig | None = None,
    recipes: Mapping[str, SimulationRecipe] | None = None,
    replicates: ReplicateSpec | ReplicateConfig | None = None,
    domains: Mapping[str, LayerLandmarks] | None = None,
    recipe: SimulationRecipe | None = None,
    dropout: SpatialDropoutConfig | None = None,
    output_path: Path | None = None,
    calculate_neighbors: bool | None = None,
    verbose: int = 0,
) -> GeneratedSimulation:
    """Generate a legacy-framework simulation from notebook-facing configs.

    ``recipe`` and ``ReplicateConfig`` are accepted for compatibility with the
    earlier single-recipe API. New code should pass ``recipes`` and
    ``ReplicateSpec``.

    """

    config = config or SimulationConfig()
    if recipes is None:
        recipes = {"layer": recipe or default_cortex_layer_recipe()}
    else:
        recipes = dict(recipes)

    if replicates is None:
        first_recipe_name = next(iter(recipes))
        replicates = named_replicates(first_recipe_name)
    elif isinstance(replicates, ReplicateConfig):
        if len(recipes) != 1:
            raise ValueError("ReplicateConfig can only be used with one recipe.")
        replicates = replicates.to_spec(next(iter(recipes)))

    multireplicate = create_multireplicate_dataset(recipes, config, replicates, verbose=verbose)
    load_or_assign_domains(multireplicate, recipes, replicates, domains=domains)
    multireplicate.simulate_expression()

    should_calculate_neighbors = config.calculate_neighbors if calculate_neighbors is None else calculate_neighbors
    if should_calculate_neighbors:
        calculate_grid_neighbors(multireplicate)

    dropout = dropout if dropout is not None else config.dropout
    if dropout is not None:
        apply_spatial_dropout(multireplicate, dropout)

    datasets = tuple(multireplicate)
    for dataset in datasets:
        recipe_name = replicates.recipe_by_replicate[dataset.name]
        dataset_recipe = recipes[recipe_name]
        dataset.obs_names = [f"{dataset.name}_{index}" for index in range(dataset.n_obs)]
        dataset.obs["batch"] = dataset.name
        dataset.obs["batch"] = dataset.obs["batch"].astype("category")
        dataset.obs["cell_type"] = dataset.obs["cell_type"].astype("category")
        dataset.obs[dataset_recipe.domain_key] = dataset.obs[dataset_recipe.domain_key].astype("category")
        dataset.X = csr_array(dataset.X)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_anndata(output_path, datasets)

    return GeneratedSimulation(
        datasets=datasets,
        replicate_names=tuple(replicates.names),
        config=config,
        recipes=recipes,
        output_path=output_path,
    )


def simulation_sweep_output_path(root_path: Path, config: SimulationConfig, num_replicates: int) -> Path:
    """Return the old notebook output path for one sweep configuration."""

    return (
        root_path
        / f"num_replicates_{num_replicates}"
        / (
            f"synthetic_{config.grid_size**2}_{config.num_genes}"
            f"_sigY-scale-{config.sig_y_scale}"
            f"_sigX-scale-{config.sig_x_scale}"
            f"_noise-metagene-parameter_{config.noise_metagene_parameter}"
        )
        / f"random_state_{config.random_state}"
        / "processed_dataset.h5ad"
    )


def generate_simulation_sweep(
    base_config: SimulationConfig,
    sweep: Mapping[str, Sequence[object]],
    recipes: Mapping[str, SimulationRecipe],
    root_path: Path,
    recipe_names: Sequence[str] = ("progenitor", "layer"),
    replicate_planner=alternating_replicates,
    verbose: int = 0,
) -> tuple[GeneratedSimulation, ...]:
    """Generate all simulations in a small Cartesian-product sweep."""

    results = []
    keys = tuple(sweep)
    for values in itertools.product(*(sweep[key] for key in keys)):
        overrides = dict(zip(keys, values))
        num_replicates = int(overrides.pop("num_replicates", 1))
        config = replace(base_config, **overrides)
        output_path = simulation_sweep_output_path(root_path, config, num_replicates)
        results.append(
            generate_simulation(
                config=config,
                recipes=recipes,
                replicates=replicate_planner(recipe_names, num_replicates),
                output_path=output_path,
                verbose=verbose,
            ),
        )

    return tuple(results)
