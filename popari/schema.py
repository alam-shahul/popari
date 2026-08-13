"""AnnData schema and namespace for Popari results."""

from collections.abc import Mapping

import anndata as ad
import numpy as np
from scipy import sparse

from popari._sample_axis import SampleAxis

DATASET_NAME_KEY = "dataset_name"
DEFAULT_SAMPLE_KEY = "batch"
SAMPLE_KEY_KEY = "popari_sample_key"
EMBEDDING_KEY = "X"
METAGENE_KEY = "M"
SPATIAL_AFFINITY_KEY = "Sigma_x_inv"
SPATIAL_AFFINITY_FACTORS_KEY = "spatial_affinity_factors"
SPATIAL_FACTOR_EMBEDDING_KEY = "spatial_factor_embeddings"
ADJACENCY_MATRIX_KEY = "adjacency_matrix"
BIN_ASSIGNMENTS_KEY = "bin_assignments"
HYPERPARAMETERS_KEY = "popari_hyperparameters"
SCHEMA_VERSION_KEY = "popari_schema_version"
SCHEMA_VERSION = 2


def _validate_spatial_graph(
    adata: ad.AnnData,
    sample_axis: SampleAxis,
    adjacency_key: str = ADJACENCY_MATRIX_KEY,
) -> None:
    """Validate one spatial graph against an existing sample axis."""

    if adjacency_key not in adata.obsp:
        raise KeyError(f"Missing spatial graph `obsp[{adjacency_key!r}]`.")

    adjacency = adata.obsp[adjacency_key]
    expected_shape = (adata.n_obs, adata.n_obs)
    if adjacency.shape != expected_shape:
        raise ValueError(
            f"`obsp[{adjacency_key!r}]` has shape {adjacency.shape}; expected {expected_shape}.",
        )

    if not sparse.issparse(adjacency) or adjacency.format != "csr":
        raise TypeError(f"`obsp[{adjacency_key!r}]` must use CSR sparse format.")

    row_chunk_size = 100_000
    for row_start in range(0, adata.n_obs, row_chunk_size):
        row_stop = min(row_start + row_chunk_size, adata.n_obs)
        edge_counts = np.diff(adjacency.indptr[row_start : row_stop + 1])
        rows = np.repeat(np.arange(row_start, row_stop), edge_counts)
        edge_start = adjacency.indptr[row_start]
        edge_stop = adjacency.indptr[row_stop]
        columns = adjacency.indices[edge_start:edge_stop]
        nonzero = adjacency.data[edge_start:edge_stop] != 0
        if np.any(sample_axis.codes[rows[nonzero]] != sample_axis.codes[columns[nonzero]]):
            raise ValueError(f"`obsp[{adjacency_key!r}]` contains cross-sample edges.")


@ad.register_anndata_namespace("popari")
class PopariNamespace:
    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    @property
    def name(self) -> str:
        if DATASET_NAME_KEY in self._adata.uns:
            return str(self._adata.uns[DATASET_NAME_KEY])

        raise ValueError(f"Dataset name is not set in `uns[{DATASET_NAME_KEY!r}]`.")

    @name.setter
    def name(self, replicate_name: str) -> None:
        self._adata.uns[DATASET_NAME_KEY] = f"{replicate_name}"

    @property
    def sample_names(self) -> tuple[str, ...]:
        """Ordered sample names from the canonical sample axis."""

        return SampleAxis.from_anndata(
            self._adata,
            sample_key=self.sample_key,
        ).names

    @property
    def sample_key(self) -> str:
        """Observation column containing sample identities."""

        return str(self._adata.uns.get(SAMPLE_KEY_KEY, DEFAULT_SAMPLE_KEY))

    def sample_mask(self, sample: str, sample_key: str | None = None) -> np.ndarray:
        """Return an observation mask for a named sample."""

        return SampleAxis.from_anndata(
            self._adata,
            sample_key=sample_key or self.sample_key,
        ).mask(sample)

    def sample_indices(self, sample: str, sample_key: str | None = None) -> np.ndarray:
        """Return observation indices for a named sample."""

        return SampleAxis.from_anndata(
            self._adata,
            sample_key=sample_key or self.sample_key,
        ).indices(sample)

    def validate_spatial_graph(self) -> None:
        """Validate the canonical spatial graph against the sample axis."""

        sample_axis = SampleAxis.from_anndata(
            self._adata,
            sample_key=self.sample_key,
        )
        _validate_spatial_graph(self._adata, sample_axis)

    def validate(self, *, require_schema_version: bool = True) -> SampleAxis:
        """Validate the canonical unified Popari AnnData schema."""

        if require_schema_version:
            version = self._adata.uns.get(SCHEMA_VERSION_KEY)
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Expected Popari schema version {SCHEMA_VERSION}; found {version!r}. "
                    "Migrate this artifact before loading it.",
                )

        sample_axis = SampleAxis.from_anndata(self._adata, sample_key=self.sample_key)
        _validate_spatial_graph(self._adata, sample_axis)
        return sample_axis

    def _single_sample_name(self) -> str:
        if DEFAULT_SAMPLE_KEY in self._adata.obs:
            samples = self._adata.obs[DEFAULT_SAMPLE_KEY].dropna().astype(str).unique()
            if len(samples) == 1:
                return str(samples[0])
            if len(samples) > 1:
                raise ValueError(
                    "This AnnData contains multiple samples; use a sample-specific Popari accessor.",
                )
        return self.name

    @property
    def embedding(self):
        """Learned cell-by-metagene embedding."""

        return self._adata.obsm[EMBEDDING_KEY]

    @embedding.setter
    def embedding(self, value) -> None:
        self._adata.obsm[EMBEDDING_KEY] = value

    @property
    def metagenes(self):
        """Shared learned gene-by-metagene matrix."""

        return self._adata.uns[METAGENE_KEY]

    @metagenes.setter
    def metagenes(self, value) -> None:
        self._adata.uns[METAGENE_KEY] = value

    @property
    def spatial_affinity(self) -> np.ndarray:
        """Spatial affinity matrix for this dataset."""

        return self.spatial_affinity_for(self._single_sample_name())

    @spatial_affinity.setter
    def spatial_affinity(self, value) -> None:
        self._adata.uns.setdefault(SPATIAL_AFFINITY_KEY, {})
        self._adata.uns[SPATIAL_AFFINITY_KEY][self._single_sample_name()] = value

    def spatial_affinity_for(self, sample: str) -> np.ndarray:
        """Return the spatial affinity associated with a named sample."""

        try:
            return self._adata.uns[SPATIAL_AFFINITY_KEY][str(sample)]
        except KeyError as error:
            raise KeyError(f"Missing spatial affinity matrix for sample {sample!r}.") from error

    @property
    def adjacency_matrix(self):
        """Sparse spatial adjacency matrix."""

        return self._adata.obsp[ADJACENCY_MATRIX_KEY]

    @adjacency_matrix.setter
    def adjacency_matrix(self, value) -> None:
        self._adata.obsp[ADJACENCY_MATRIX_KEY] = value

    @property
    def hyperparameters(self):
        """Hyperparameters saved with a trained Popari result."""

        return self._adata.uns[HYPERPARAMETERS_KEY]

    @hyperparameters.setter
    def hyperparameters(self, value) -> None:
        self._adata.uns[HYPERPARAMETERS_KEY] = value

    def affinity_difference(
        self,
        comparison: str,
        reference: str,
        spatial_affinity_key: str = SPATIAL_AFFINITY_KEY,
    ) -> np.ndarray:
        """Return one named spatial affinity matrix minus another."""

        if spatial_affinity_key not in self._adata.uns:
            raise KeyError(f"Missing spatial affinity key in `.uns`: {spatial_affinity_key!r}.")

        spatial_affinities = self._adata.uns[spatial_affinity_key]
        missing_names = [name for name in (comparison, reference) if name not in spatial_affinities]
        if missing_names:
            raise KeyError(
                f"Missing spatial affinity matrix/matrices under `.uns[{spatial_affinity_key!r}]`: "
                f"{missing_names}.",
            )

        return np.asarray(spatial_affinities[comparison]) - np.asarray(spatial_affinities[reference])


def validate_anndata_hierarchy(hierarchy: Mapping[int, ad.AnnData]) -> None:
    """Validate a canonical hierarchy containing one AnnData per level."""

    if not hierarchy:
        raise ValueError("hierarchy must contain at least level 0.")
    levels = sorted(hierarchy)
    if levels != list(range(len(levels))):
        raise ValueError(f"Hierarchy levels must be contiguous from zero; found {levels}.")

    base_axis = hierarchy[0].popari.validate()
    for level in levels[1:]:
        previous = hierarchy[level - 1]
        current = hierarchy[level]
        axis = current.popari.validate()
        if current.popari.sample_key != hierarchy[0].popari.sample_key:
            raise ValueError("All hierarchy levels must use the same sample key.")
        if axis.names != base_axis.names:
            raise ValueError(
                f"Hierarchy level {level} has samples {axis.names}; expected {base_axis.names}.",
            )

        if BIN_ASSIGNMENTS_KEY not in current.obsm:
            raise ValueError(f"Hierarchy level {level} is missing obsm[{BIN_ASSIGNMENTS_KEY!r}].")
        assignments = current.obsm[BIN_ASSIGNMENTS_KEY]
        expected_shape = (current.n_obs, previous.n_obs)
        if assignments.shape != expected_shape:
            raise ValueError(
                f"Hierarchy level {level} bin assignments have shape {assignments.shape}; "
                f"expected {expected_shape}.",
            )
        assignment = sparse.coo_array(assignments)
        previous_axis = SampleAxis.from_anndata(previous, sample_key=previous.popari.sample_key)
        if np.any(axis.codes[assignment.row] != previous_axis.codes[assignment.col]):
            raise ValueError(f"Hierarchy level {level} bin assignments cross sample boundaries.")


__all__ = [PopariNamespace.__name__, validate_anndata_hierarchy.__name__]
