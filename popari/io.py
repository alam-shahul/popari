"""Read, write, and migrate Popari AnnData artifacts."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse
from scipy.sparse import csr_array

from popari._sample_axis import SampleAxis
from popari.schema import (
    ADJACENCY_LIST_KEY,
    ADJACENCY_MATRIX_KEY,
    BIN_ASSIGNMENTS_KEY,
    DATASET_NAME_KEY,
    DEFAULT_SAMPLE_KEY,
    HYPERPARAMETERS_KEY,
    METAGENE_KEY,
    SAMPLE_KEY_KEY,
    SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    SPATIAL_AFFINITY_KEY,
)
from popari.util import convert_adjacency_matrix_to_awkward_array

LEVEL_FILE_PATTERN = re.compile(r"^level_(\d+)\.h5ad$")
_SAMPLE_PARAMETER_KEYS = (
    SPATIAL_AFFINITY_KEY,
    "Sigma_x_inv_bar",
)


def _resolved_sample_key(adata: AnnData, sample_key: str | None) -> str:
    if sample_key is not None:
        return sample_key
    return str(adata.uns.get(SAMPLE_KEY_KEY, DEFAULT_SAMPLE_KEY))


def _normalize_sample_labels(adata: AnnData, sample_key: str) -> tuple[str, ...]:
    if sample_key not in adata.obs:
        if DATASET_NAME_KEY not in adata.uns:
            raise KeyError(
                f"Missing sample column `obs[{sample_key!r}]` and fallback " f"`uns[{DATASET_NAME_KEY!r}]`.",
            )
        adata.obs[sample_key] = str(adata.uns[DATASET_NAME_KEY])

    labels = adata.obs[sample_key]
    if labels.isna().any():
        raise ValueError(f"`obs[{sample_key!r}]` contains missing sample labels.")

    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = [str(category) for category in labels.cat.categories]
        string_labels = labels.astype(str)
        observed = set(string_labels)
        categories = [category for category in categories if category in observed]
        ordered = labels.cat.ordered
    else:
        string_labels = labels.astype(str)
        categories = list(dict.fromkeys(string_labels))
        ordered = True

    adata.obs[sample_key] = pd.Categorical(
        string_labels,
        categories=categories,
        ordered=ordered,
    )
    return tuple(categories)


def _migrate_legacy_observation_names(
    adata: AnnData,
    *,
    sample_key: str,
    is_legacy: bool,
) -> None:
    if adata.obs_names.is_unique:
        return
    if not is_legacy:
        raise ValueError("Observation names must be unique.")

    identities = pd.MultiIndex.from_arrays(
        [adata.obs[sample_key].astype(str), adata.obs_names.astype(str)],
        names=[sample_key, "observation"],
    )
    if not identities.has_duplicates:
        adata.obs_names = [f"{sample}:{observation}" for sample, observation in identities]
    else:
        adata.obs["_legacy_obs_name"] = adata.obs_names.astype(str)
        sample_positions = adata.obs.groupby(sample_key, observed=True).cumcount()
        adata.obs_names = [
            f"{sample}:{position}"
            for sample, position in zip(
                adata.obs[sample_key].astype(str),
                sample_positions,
                strict=True,
            )
        ]
    if not adata.obs_names.is_unique:
        raise ValueError("Could not construct unique observation names for legacy artifact.")


def _graph_from_legacy_uns(
    adata: AnnData,
    *,
    sample_key: str,
    sample_names: Sequence[str],
) -> csr_array:
    adjacency_by_sample = adata.uns.get(ADJACENCY_MATRIX_KEY)
    if not isinstance(adjacency_by_sample, Mapping):
        raise KeyError(f"Missing spatial graph `obsp[{ADJACENCY_MATRIX_KEY!r}]`.")

    labels = adata.obs[sample_key].astype(str).to_numpy()
    row_parts = []
    column_parts = []
    data_parts = []
    for sample in sample_names:
        indices = np.flatnonzero(labels == sample)
        if sample not in adjacency_by_sample:
            raise KeyError(f"Missing legacy adjacency matrix for sample {sample!r}.")

        local_graph = sparse.coo_array(adjacency_by_sample[sample])
        expected_shape = (len(indices), len(indices))
        if local_graph.shape != expected_shape:
            raise ValueError(
                f"Legacy adjacency matrix for sample {sample!r} has shape "
                f"{local_graph.shape}; expected {expected_shape}.",
            )
        local_graph.eliminate_zeros()
        row_parts.append(indices[local_graph.row])
        column_parts.append(indices[local_graph.col])
        data_parts.append(local_graph.data)

    if not data_parts:
        return csr_array((adata.n_obs, adata.n_obs))

    return csr_array(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(adata.n_obs, adata.n_obs),
    )


def _restore_none_sentinels(adata: AnnData) -> None:
    for key in (SPATIAL_AFFINITY_KEY, "Sigma_x_inv_bar"):
        values = adata.uns.get(key)
        if not isinstance(values, Mapping):
            continue
        adata.uns[key] = {
            str(sample): (None if np.isscalar(value) and value == -1 else value) for sample, value in values.items()
        }


def _collapse_shared_matrix(adata: AnnData, key: str) -> None:
    value = adata.uns.get(key)
    if value is None:
        return

    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"`uns[{key!r}]` contains no matrices.")
        matrices = [np.asarray(matrix) for matrix in value.values()]
        matrix = matrices[0]
        if any(
            candidate.shape != matrix.shape or not np.allclose(candidate, matrix, rtol=1e-5, atol=1e-8)
            for candidate in matrices[1:]
        ):
            raise ValueError(
                f"Legacy `uns[{key!r}]` contains unequal sample-specific matrices. "
                "Differential metagenes are no longer supported.",
            )
        value = matrix

    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != adata.n_vars:
        raise ValueError(
            f"`uns[{key!r}]` must have shape (n_vars, K); found {matrix.shape}.",
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"`uns[{key!r}]` must contain only finite values.")
    adata.uns[key] = matrix


def _remove_legacy_metagene_hyperparameters(adata: AnnData) -> None:
    hyperparameters = adata.uns.get(HYPERPARAMETERS_KEY)
    if not isinstance(hyperparameters, Mapping):
        return
    hyperparameters = dict(hyperparameters)
    for key in ("metagene_groups", "metagene_tags", "metagene_mode", "lambda_M"):
        hyperparameters.pop(key, None)
    adata.uns[HYPERPARAMETERS_KEY] = hyperparameters


def convert_legacy_anndata(
    adata: AnnData,
    *,
    sample_key: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Return one canonical multisample AnnData from a Popari artifact.

    Existing canonical objects are validated, while legacy merged artifacts are
    normalized and upgraded in memory.

    """

    result = adata.copy() if copy else adata
    resolved_sample_key = _resolved_sample_key(result, sample_key)
    stored_version = result.uns.get(SCHEMA_VERSION_KEY)
    is_legacy = stored_version is None or int(stored_version) < SCHEMA_VERSION
    if stored_version is not None and int(stored_version) not in {1, SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported Popari schema version {stored_version!r}; expected 1 or {SCHEMA_VERSION}.",
        )

    sample_names = _normalize_sample_labels(result, resolved_sample_key)
    _migrate_legacy_observation_names(
        result,
        sample_key=resolved_sample_key,
        is_legacy=is_legacy,
    )

    if ADJACENCY_MATRIX_KEY in result.obsp:
        result.obsp[ADJACENCY_MATRIX_KEY] = csr_array(result.obsp[ADJACENCY_MATRIX_KEY])
    else:
        result.obsp[ADJACENCY_MATRIX_KEY] = _graph_from_legacy_uns(
            result,
            sample_key=resolved_sample_key,
            sample_names=sample_names,
        )

    result.uns.pop(ADJACENCY_MATRIX_KEY, None)
    result.obsm.pop(ADJACENCY_LIST_KEY, None)
    _restore_none_sentinels(result)
    _collapse_shared_matrix(result, METAGENE_KEY)
    _collapse_shared_matrix(result, "ground_truth_M")
    result.uns.pop("M_bar", None)
    _remove_legacy_metagene_hyperparameters(result)

    result.uns[SAMPLE_KEY_KEY] = resolved_sample_key
    result.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    if DATASET_NAME_KEY not in result.uns:
        result.uns[DATASET_NAME_KEY] = sample_names[0] if len(sample_names) == 1 else "multisample"

    result.popari.validate_spatial_graph()
    return result


def load_anndata(
    filepath: str | Path,
    *,
    sample_key: str | None = None,
) -> AnnData:
    """Load one canonical Popari AnnData from an H5AD file."""

    return convert_legacy_anndata(
        ad.read_h5ad(filepath),
        sample_key=sample_key,
        copy=False,
    )


def _sample_name(dataset: AnnData, sample_key: str) -> str:
    try:
        return dataset.popari.name
    except ValueError:
        fallback_key = (
            sample_key
            if sample_key in dataset.obs
            else DEFAULT_SAMPLE_KEY if DEFAULT_SAMPLE_KEY in dataset.obs else None
        )
        if fallback_key is None:
            raise
        names = dataset.obs[fallback_key].dropna().astype(str).unique()
        if len(names) != 1:
            raise ValueError(
                "Each compatibility input AnnData must represent exactly one named sample.",
            )
        return str(names[0])


def merge_anndata(
    datasets: Sequence[AnnData],
    ignore_raw_data: bool = False,
    *,
    sample_key: str = DEFAULT_SAMPLE_KEY,
) -> AnnData:
    """Compatibility adapter that combines single-sample AnnData objects."""

    datasets = list(datasets)
    if not datasets:
        raise ValueError("datasets must contain at least one AnnData object.")

    reference_variables = datasets[0].var_names
    if any(not dataset.var_names.equals(reference_variables) for dataset in datasets[1:]):
        raise ValueError("All datasets must contain the same variables in the same order.")

    sample_names = [_sample_name(dataset, sample_key) for dataset in datasets]
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("Sample names must be unique.")

    copies = []
    graphs = []
    bin_assignments = []
    for dataset, sample in zip(datasets, sample_names):
        dataset_copy = dataset.copy()
        if sample_key in dataset_copy.obs:
            del dataset_copy.obs[sample_key]
        dataset_copy.obsm.pop(ADJACENCY_LIST_KEY, None)
        if ADJACENCY_MATRIX_KEY not in dataset_copy.obsp:
            raise KeyError(
                f"Sample {sample!r} is missing `obsp[{ADJACENCY_MATRIX_KEY!r}]`.",
            )
        graphs.append(csr_array(dataset_copy.obsp[ADJACENCY_MATRIX_KEY]))

        assignment_keys = [
            key for key in dataset_copy.obsm if key == BIN_ASSIGNMENTS_KEY or key.startswith(f"{BIN_ASSIGNMENTS_KEY}_")
        ]
        if len(assignment_keys) > 1:
            raise ValueError(
                f"Sample {sample!r} has multiple bin-assignment matrices: {assignment_keys}.",
            )
        if assignment_keys:
            bin_assignments.append(csr_array(dataset_copy.obsm.pop(assignment_keys[0])))
        copies.append(dataset_copy)

    if bin_assignments and len(bin_assignments) != len(datasets):
        raise ValueError("Bin assignments must be present for either every sample or no samples.")

    observation_names = pd.Index(
        np.concatenate([dataset.obs_names.to_numpy() for dataset in copies]),
    )
    index_unique = None if observation_names.is_unique else "-"
    merged = ad.concat(
        copies,
        label=sample_key,
        keys=sample_names,
        index_unique=index_unique,
        join="inner",
        merge="unique",
        uns_merge="unique",
        pairwise=True,
    )
    merged.obs[sample_key] = pd.Categorical(
        merged.obs[sample_key].astype(str),
        categories=sample_names,
        ordered=True,
    )
    merged.obsp[ADJACENCY_MATRIX_KEY] = sparse.block_diag(graphs, format="csr")
    if bin_assignments:
        merged.obsm[BIN_ASSIGNMENTS_KEY] = sparse.block_diag(
            bin_assignments,
            format="csr",
        )
    merged.uns[DATASET_NAME_KEY] = sample_names[0] if len(sample_names) == 1 else "multisample"

    for key in _SAMPLE_PARAMETER_KEYS:
        values = {}
        for dataset, sample in zip(datasets, sample_names):
            source = dataset.uns.get(key)
            if isinstance(source, Mapping) and sample in source:
                values[sample] = source[sample]
        if values:
            merged.uns[key] = values

    merged = convert_legacy_anndata(
        merged,
        sample_key=sample_key,
        copy=False,
    )
    if ignore_raw_data:
        merged.X = csr_array(merged.shape)
    elif sparse.issparse(merged.X):
        merged.X = csr_array(merged.X)
    return merged


def _filter_hyperparameter_groups(hyperparameters: Mapping[str, Any], sample: str) -> dict:
    filtered = dict(hyperparameters)
    name_parts = sample.rsplit("_level_", maxsplit=1)
    level = int(name_parts[1]) if len(name_parts) == 2 and name_parts[1].isdigit() else 0

    for key in ("spatial_affinity_groups",):
        groups = filtered.get(key)
        if not isinstance(groups, Mapping):
            continue
        filtered[key] = {
            group: group_samples
            for group, group_samples in groups.items()
            if (
                (
                    int(str(group).rsplit("_level_", maxsplit=1)[1])
                    if "_level_" in str(group) and str(group).rsplit("_level_", maxsplit=1)[1].isdigit()
                    else 0
                )
                == level
            )
        }
    return filtered


def unmerge_anndata(
    merged_dataset: AnnData,
    *,
    sample_key: str | None = None,
) -> tuple[list[AnnData], list[str]]:
    """Compatibility adapter that splits a canonical AnnData by sample."""

    canonical = convert_legacy_anndata(
        merged_dataset,
        sample_key=sample_key,
        copy=True,
    )
    axis = SampleAxis.from_anndata(canonical, sample_key=canonical.popari.sample_key)
    canonical.popari.validate_spatial_graph()
    datasets = []
    for sample in axis.names:
        dataset = canonical[axis.indices(sample)].copy()
        dataset.obs[axis.sample_key] = dataset.obs[axis.sample_key].cat.remove_unused_categories()
        dataset.popari.name = sample

        for key in _SAMPLE_PARAMETER_KEYS:
            values = canonical.uns.get(key)
            if isinstance(values, Mapping) and sample in values:
                dataset.uns[key] = {sample: values[sample]}

        if HYPERPARAMETERS_KEY in dataset.uns:
            dataset.uns[HYPERPARAMETERS_KEY] = _filter_hyperparameter_groups(
                dataset.uns[HYPERPARAMETERS_KEY],
                sample,
            )

        adjacency = csr_array(dataset.obsp[ADJACENCY_MATRIX_KEY])
        dataset.obsp[ADJACENCY_MATRIX_KEY] = adjacency
        dataset.obsm[ADJACENCY_LIST_KEY] = convert_adjacency_matrix_to_awkward_array(
            adjacency.tocoo(),
        )
        if BIN_ASSIGNMENTS_KEY in canonical.obsm:
            assignment_rows = csr_array(canonical.obsm[BIN_ASSIGNMENTS_KEY])[axis.indices(sample)]
            used_columns = np.unique(assignment_rows.indices)
            if used_columns.size == 0:
                raise ValueError(f"Sample {sample!r} has an empty bin-assignment matrix.")
            dataset.obsm.pop(BIN_ASSIGNMENTS_KEY, None)
            dataset.obsm[f"{BIN_ASSIGNMENTS_KEY}_{sample}"] = assignment_rows[:, used_columns]
        datasets.append(dataset)

    return datasets, list(axis.names)


def _serializable_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return make_hdf5_compatible(value)
    if isinstance(value, Mapping):
        return {str(key): _serializable_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serializable_value(item) for item in value]
    if isinstance(value, list):
        return [_serializable_value(item) for item in value]
    return value


def _prepare_for_write(adata: AnnData) -> AnnData:
    result = adata.copy()
    result.uns = _serializable_value(dict(result.uns))
    for key in (SPATIAL_AFFINITY_KEY, "Sigma_x_inv_bar"):
        values = result.uns.get(key)
        if isinstance(values, Mapping):
            result.uns[key] = {sample: -1 if value is None else value for sample, value in values.items()}
    return result


def save_anndata(
    filepath: str | Path,
    adata: AnnData | Sequence[AnnData],
    ignore_raw_data: bool = False,
    *,
    sample_key: str | None = None,
) -> AnnData:
    """Write one canonical Popari AnnData to an H5AD file.

    A sequence of single-sample objects is accepted temporarily as a
    compatibility adapter.

    """

    if isinstance(adata, AnnData):
        canonical = convert_legacy_anndata(
            adata,
            sample_key=sample_key,
            copy=True,
        )
        if ignore_raw_data:
            canonical.X = csr_array(canonical.shape)
    else:
        warnings.warn(
            "Passing a sequence to save_anndata is deprecated; pass one multisample AnnData.",
            DeprecationWarning,
            stacklevel=2,
        )
        canonical = merge_anndata(
            adata,
            ignore_raw_data=ignore_raw_data,
            sample_key=sample_key or DEFAULT_SAMPLE_KEY,
        )

    writable = _prepare_for_write(canonical)
    writable.write_h5ad(filepath)
    return canonical


def _hierarchy_files(path: str | Path) -> dict[int, Path]:
    path = Path(path)
    if path.is_file():
        return {0: path}

    h5ad_path = path.with_suffix(".h5ad")
    if h5ad_path.is_file():
        return {0: h5ad_path}
    if not path.is_dir():
        raise FileNotFoundError(f"No Popari AnnData artifact found at {path}.")

    files = {}
    for candidate in path.glob("level_*.h5ad"):
        match = LEVEL_FILE_PATTERN.match(candidate.name)
        if match is not None:
            files[int(match.group(1))] = candidate
    if not files:
        raise ValueError(f"No level_*.h5ad files found in {path}.")
    expected = set(range(len(files)))
    if set(files) != expected:
        raise ValueError(
            f"Hierarchy levels must be contiguous from zero; found {sorted(files)}.",
        )
    return files


def load_anndata_hierarchy(
    path: str | Path,
    *,
    sample_key: str | None = None,
) -> dict[int, AnnData]:
    """Load one canonical AnnData for every saved hierarchy level."""

    hierarchy = {
        level: load_anndata(level_path, sample_key=sample_key)
        for level, level_path in sorted(_hierarchy_files(path).items())
    }
    return normalize_anndata_hierarchy(hierarchy, sample_key=sample_key)


def _rename_mapping_keys(mapping: Mapping, renames: Mapping[str, str]) -> dict:
    return {renames.get(str(key), str(key)): value for key, value in mapping.items()}


def _normalize_hierarchy_bin_assignments(
    previous_adata: AnnData,
    adata: AnnData,
    *,
    level: int,
    sample_key: str,
) -> None:
    if BIN_ASSIGNMENTS_KEY in adata.obsm:
        return

    legacy_keys = [key for key in adata.obsm if key.startswith(f"{BIN_ASSIGNMENTS_KEY}_")]
    if not legacy_keys:
        return

    previous_axis = SampleAxis.from_anndata(
        previous_adata,
        sample_key=sample_key,
    )
    previous_adata.popari.validate_spatial_graph()
    axis = SampleAxis.from_anndata(
        adata,
        sample_key=sample_key,
    )
    adata.popari.validate_spatial_graph()
    row_parts = []
    column_parts = []
    data_parts = []
    for sample in axis.names:
        candidates = (
            f"{BIN_ASSIGNMENTS_KEY}_{sample}",
            f"{BIN_ASSIGNMENTS_KEY}_{sample}_level_{level}",
        )
        matching_keys = [key for key in candidates if key in adata.obsm]
        if len(matching_keys) != 1:
            raise KeyError(
                f"Expected one legacy bin-assignment matrix for sample {sample!r}; " f"found {matching_keys}.",
            )

        coarse_indices = axis.indices(sample)
        fine_indices = previous_axis.indices(sample)
        legacy_assignments = csr_array(adata.obsm[matching_keys[0]])
        expected_shape = (adata.n_obs, len(fine_indices))
        if legacy_assignments.shape != expected_shape:
            raise ValueError(
                f"Legacy bin assignments for sample {sample!r} have shape "
                f"{legacy_assignments.shape}; expected {expected_shape}.",
            )

        local_assignments = sparse.coo_array(legacy_assignments[coarse_indices])
        row_parts.append(coarse_indices[local_assignments.row])
        column_parts.append(fine_indices[local_assignments.col])
        data_parts.append(local_assignments.data)

    adata.obsm[BIN_ASSIGNMENTS_KEY] = csr_array(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(adata.n_obs, previous_adata.n_obs),
    )
    for key in legacy_keys:
        del adata.obsm[key]


def normalize_anndata_hierarchy(
    hierarchy: Mapping[int, AnnData],
    *,
    sample_key: str | None = None,
) -> dict[int, AnnData]:
    """Normalize legacy hierarchy sample suffixes across unified levels."""

    if not hierarchy:
        raise ValueError("hierarchy must contain at least level 0.")

    result = dict(sorted(hierarchy.items()))
    levels = list(result)
    if levels != list(range(len(levels))):
        raise ValueError(f"Hierarchy levels must be contiguous from zero; found {levels}.")

    resolved_sample_key = _resolved_sample_key(result[0], sample_key)
    base_axis = SampleAxis.from_anndata(
        result[0],
        sample_key=resolved_sample_key,
    )
    result[0].popari.validate_spatial_graph()
    base_names = base_axis.names

    for level, adata in result.items():
        if _resolved_sample_key(adata, sample_key) != resolved_sample_key:
            raise ValueError("All hierarchy levels must use the same sample key.")

        axis = SampleAxis.from_anndata(adata, sample_key=resolved_sample_key)
        adata.popari.validate_spatial_graph()
        if axis.names != base_names:
            legacy_names = tuple(f"{sample}_level_{level}" for sample in base_names)
            if axis.names != legacy_names:
                raise ValueError(
                    f"Hierarchy level {level} has samples {axis.names}; expected "
                    f"{base_names} or legacy names {legacy_names}.",
                )

            renames = dict(zip(legacy_names, base_names))
            adata.obs[resolved_sample_key] = pd.Categorical(
                adata.obs[resolved_sample_key].astype(str).map(renames),
                categories=base_names,
                ordered=True,
            )
            for key in (*_SAMPLE_PARAMETER_KEYS, "sigma_yx"):
                values = adata.uns.get(key)
                if isinstance(values, Mapping):
                    adata.uns[key] = _rename_mapping_keys(values, renames)

            hyperparameters = adata.uns.get(HYPERPARAMETERS_KEY)
            if isinstance(hyperparameters, Mapping):
                hyperparameters = dict(hyperparameters)
                prior_x = hyperparameters.get("prior_x")
                if isinstance(prior_x, Mapping):
                    hyperparameters["prior_x"] = _rename_mapping_keys(prior_x, renames)
                for key in ("spatial_affinity_groups",):
                    groups = hyperparameters.get(key)
                    if isinstance(groups, Mapping):
                        hyperparameters[key] = {
                            group: [renames.get(str(sample), str(sample)) for sample in samples]
                            for group, samples in groups.items()
                        }
                for key in ("spatial_affinity_tags",):
                    tags = hyperparameters.get(key)
                    if isinstance(tags, Mapping):
                        hyperparameters[key] = _rename_mapping_keys(tags, renames)
                adata.uns[HYPERPARAMETERS_KEY] = hyperparameters

        adata.popari.validate_spatial_graph()
        if level > 0:
            _normalize_hierarchy_bin_assignments(
                result[level - 1],
                adata,
                level=level,
                sample_key=resolved_sample_key,
            )

    return result


def save_anndata_hierarchy(
    path: str | Path,
    hierarchy: Mapping[int, AnnData],
    *,
    ignore_raw_data: bool = False,
    sample_key: str | None = None,
) -> dict[int, AnnData]:
    """Write one canonical H5AD file for every hierarchy level."""

    levels = sorted(hierarchy)
    if levels != list(range(len(levels))):
        raise ValueError(f"Hierarchy levels must be contiguous from zero; found {levels}.")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return {
        level: save_anndata(
            path / f"level_{level}.h5ad",
            hierarchy[level],
            ignore_raw_data=ignore_raw_data,
            sample_key=sample_key,
        )
        for level in levels
    }


def make_hdf5_compatible(array: torch.Tensor | np.ndarray):
    """Convert a tensor to a CPU NumPy array for HDF5 serialization."""

    if isinstance(array, torch.Tensor):
        cpu_tensor = array.detach().cpu()
        if cpu_tensor.is_sparse:
            cpu_tensor = cpu_tensor.to_dense()
        return cpu_tensor.numpy()
    return array
