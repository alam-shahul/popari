"""AnnData schema and namespace for Popari results."""

import anndata as ad
import numpy as np

from popari._sample_axis import SampleAxis

DATASET_NAME_KEY = "dataset_name"
DEFAULT_SAMPLE_KEY = "batch"
EMBEDDING_KEY = "X"
METAGENE_KEY = "M"
SPATIAL_AFFINITY_KEY = "Sigma_x_inv"
ADJACENCY_MATRIX_KEY = "adjacency_matrix"
ADJACENCY_LIST_KEY = "adjacency_list"
HYPERPARAMETERS_KEY = "popari_hyperparameters"
SCHEMA_VERSION_KEY = "popari_schema_version"
SCHEMA_VERSION = 1


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

        return SampleAxis.from_anndata(self._adata, sample_key=DEFAULT_SAMPLE_KEY).names

    def sample_mask(self, sample: str, sample_key: str = DEFAULT_SAMPLE_KEY) -> np.ndarray:
        """Return an observation mask for a named sample."""

        return SampleAxis.from_anndata(self._adata, sample_key=sample_key).mask(sample)

    def sample_indices(self, sample: str, sample_key: str = DEFAULT_SAMPLE_KEY) -> np.ndarray:
        """Return observation indices for a named sample."""

        return SampleAxis.from_anndata(self._adata, sample_key=sample_key).indices(sample)

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
        """Learned gene-by-metagene matrix for this dataset."""

        return self.metagenes_for(self._single_sample_name())

    @metagenes.setter
    def metagenes(self, value) -> None:
        self._adata.uns.setdefault(METAGENE_KEY, {})
        self._adata.uns[METAGENE_KEY][self._single_sample_name()] = value

    def metagenes_for(self, sample: str):
        """Return the metagene matrix associated with a named sample."""

        try:
            return self._adata.uns[METAGENE_KEY][str(sample)]
        except KeyError as error:
            raise KeyError(f"Missing metagene matrix for sample {sample!r}.") from error

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
    def adjacency_list(self):
        """Spatial neighbors stored as an Awkward array."""

        return self._adata.obsm[ADJACENCY_LIST_KEY]

    @adjacency_list.setter
    def adjacency_list(self, value) -> None:
        self._adata.obsm[ADJACENCY_LIST_KEY] = value

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


__all__ = [PopariNamespace.__name__]
