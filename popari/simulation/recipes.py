"""Declarative configurations and presets for Popari simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LayerLandmarks = Mapping[str, Sequence[Sequence[float]]]


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
class SpatialDropoutConfig:
    """Post-simulation dropout settings for hierarchical examples."""

    sparsity: float = 0.15
    random_state: int = 0


_HIERARCHICAL_CELL_TYPES = {
    "Excitatory L1": [0.5, 0, 0, 0, 0, 1, 0, 0, 0],
    "Excitatory L2": [0.5, 0, 0, 0, 0, 0, 1, 0, 0],
    "Excitatory L3": [0.5, 0, 0, 0, 0, 0, 0, 1, 0],
    "Excitatory L4": [0.5, 0, 0, 0, 0, 0, 0, 0, 1],
    "Inhibitory 1": [0, 0.5, 0, 1, 0, 0, 0, 0, 0],
    "Inhibitory 2": [0, 0.5, 0, 0, 1, 0, 0, 0, 0],
    "Non-Neuron L1": [0, 0, 1, 0, 0, 0.5, 0, 0, 0],
    "Non-Neuron Ubiquitous": [0, 0, 1, 0, 0, 0, 0, 0, 0],
}

_HIERARCHICAL_LAYER_DISTRIBUTIONS = {
    "L1": {
        "Excitatory L1": 0.43,
        "Inhibitory 1": 0.1,
        "Non-Neuron L1": 0.4,
        "Non-Neuron Ubiquitous": 0.07,
    },
    "L2": {"Excitatory L2": 0.93, "Non-Neuron Ubiquitous": 0.07},
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

_HIERARCHICAL_VARIATION = [0, 0.3, 0, 0, 0.3, 0, 0.3, 0.3, 0.3]


def default_cortex_layer_recipe() -> SimulationRecipe:
    """Return the layered cortex recipe from the hierarchical simulation
    notebook."""

    return SimulationRecipe(
        cell_type_definitions=_HIERARCHICAL_CELL_TYPES,
        spatial_distributions=_HIERARCHICAL_LAYER_DISTRIBUTIONS,
        metagene_variation_probabilities=_HIERARCHICAL_VARIATION,
    )


def default_hierarchical_cortex_recipes() -> dict[str, SimulationRecipe]:
    """Return the named recipe set for the hierarchical cortex example."""

    return {"layer": default_cortex_layer_recipe()}


_DIFFERENTIAL_CELL_TYPES = {
    "Excitatory L1": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Excitatory L2": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "Excitatory L3": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Excitatory L4 normal": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "Inhibitory 1": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "Inhibitory 2": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    "Non-Neuron L1 normal": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "Non-Neuron Ubiquitous": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}

_DIFFERENTIAL_PROGENITOR_DISTRIBUTIONS = {
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

_DIFFERENTIAL_LAYER_DISTRIBUTIONS = {
    "L1": {
        "Excitatory L1": 0.33,
        "Inhibitory 1": 0.1,
        "Non-Neuron L1 normal": 0.5,
        "Non-Neuron Ubiquitous": 0.07,
    },
    "L2": {"Excitatory L2": 0.93, "Non-Neuron Ubiquitous": 0.07},
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

_DIFFERENTIAL_VARIATION = [0, 0.25, 0, 0.25, 0, 0.25, 0, 0.1, 0.1, 0.1]


def default_differential_cortex_recipes() -> dict[str, SimulationRecipe]:
    """Return the progenitor/layer recipes from the differential simulation
    notebook."""

    return {
        "progenitor": SimulationRecipe(
            cell_type_definitions=_DIFFERENTIAL_CELL_TYPES,
            spatial_distributions=_DIFFERENTIAL_PROGENITOR_DISTRIBUTIONS,
            metagene_variation_probabilities=_DIFFERENTIAL_VARIATION,
            domain_key="layer",
        ),
        "layer": SimulationRecipe(
            cell_type_definitions=_DIFFERENTIAL_CELL_TYPES,
            spatial_distributions=_DIFFERENTIAL_LAYER_DISTRIBUTIONS,
            metagene_variation_probabilities=_DIFFERENTIAL_VARIATION,
            domain_key="layer",
        ),
    }


_JOINT_CELL_TYPES = {
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

_JOINT_PROGENITOR_DISTRIBUTIONS = {
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

_JOINT_LAYER_DISTRIBUTIONS = {
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

_JOINT_VARIATION = [0, 0.1, 0, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1]


def default_joint_improvement_recipes() -> dict[str, SimulationRecipe]:
    """Return the paired progenitor/layer joint-improvement recipes."""

    return {
        "progenitor": SimulationRecipe(
            cell_type_definitions=_JOINT_CELL_TYPES,
            spatial_distributions=_JOINT_PROGENITOR_DISTRIBUTIONS,
            metagene_variation_probabilities=_JOINT_VARIATION,
            domain_key="domain",
        ),
        "layer": SimulationRecipe(
            cell_type_definitions=_JOINT_CELL_TYPES,
            spatial_distributions=_JOINT_LAYER_DISTRIBUTIONS,
            metagene_variation_probabilities=_JOINT_VARIATION,
            domain_key="domain",
        ),
    }


_TWO_CELL_TYPE_DEFINITIONS = {"Type A": [1, 0], "Type B": [0, 1]}

_TWO_CELL_TYPE_RATIOS = {
    "A_to_B_1_to_9": {"Type A": 1 / 10, "Type B": 9 / 10},
    "A_to_B_1_to_1": {"Type A": 1 / 2, "Type B": 1 / 2},
    "A_to_B_9_to_1": {"Type A": 9 / 10, "Type B": 1 / 10},
}


def two_cell_type_ratio_recipes(domain_key: str = "region") -> dict[str, SimulationRecipe]:
    """Return one minimal two-cell-type recipe per A:B tissue ratio."""

    return {
        ratio_name: SimulationRecipe(
            cell_type_definitions=_TWO_CELL_TYPE_DEFINITIONS,
            spatial_distributions={ratio_name: ratio_distribution},
            metagene_variation_probabilities=[0, 0],
            domain_key=domain_key,
        )
        for ratio_name, ratio_distribution in _TWO_CELL_TYPE_RATIOS.items()
    }


def named_replicates(recipe_name: str, count: int = 1) -> dict[str, str]:
    """Create repeated replicates that all use one recipe."""

    return {f"{recipe_name}_{index}": recipe_name for index in range(count)}


def alternating_replicates(recipe_names: Sequence[str], count: int) -> dict[str, str]:
    """Create replicates that cycle through recipe names in order."""

    return {
        f"{recipe_names[index % len(recipe_names)]}_{index}": recipe_names[index % len(recipe_names)]
        for index in range(count)
    }


def paired_replicates(recipe_names: Sequence[str], count: int) -> dict[str, str]:
    """Create ``count`` complete replicate pairs/groups for each recipe name."""

    return {f"{recipe_name}_{index}": recipe_name for index in range(count) for recipe_name in recipe_names}


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


__all__ = [
    "LayerLandmarks",
    "SimulationConfig",
    "SimulationRecipe",
    "SpatialDropoutConfig",
    "alternating_replicates",
    "default_cortex_layer_recipe",
    "default_differential_cortex_recipes",
    "default_domain_landmarks",
    "default_hierarchical_cortex_recipes",
    "default_joint_improvement_recipes",
    "default_layer_landmarks",
    "default_progenitor_landmarks",
    "named_replicates",
    "paired_replicates",
    "simulation_sweep_output_path",
    "two_cell_type_ratio_recipes",
]
