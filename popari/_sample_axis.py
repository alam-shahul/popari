"""Internal sample-axis indexing for multisample AnnData objects."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class SampleAxis:
    """Observation-to-sample indexing derived from a canonical AnnData."""

    sample_key: str
    names: tuple[str, ...]
    codes: np.ndarray
    _positions: MappingProxyType
    _indices: tuple[np.ndarray, ...]

    @classmethod
    def from_anndata(
        cls,
        adata: ad.AnnData,
        *,
        sample_key: str = "batch",
        adjacency_key: str = "adjacency_matrix",
    ) -> SampleAxis:
        """Validate a canonical multisample AnnData and index its samples."""

        if sample_key not in adata.obs:
            raise KeyError(f"Missing sample column `obs[{sample_key!r}]`.")

        samples = adata.obs[sample_key]
        if not isinstance(samples.dtype, pd.CategoricalDtype):
            raise TypeError(f"`obs[{sample_key!r}]` must have categorical dtype.")
        if samples.isna().any():
            raise ValueError(f"`obs[{sample_key!r}]` contains missing sample labels.")

        codes = samples.cat.codes.to_numpy(dtype=np.int64, copy=True)
        categories = tuple(str(category) for category in samples.cat.categories)
        observed_codes = set(np.unique(codes))
        unused = [categories[index] for index in range(len(categories)) if index not in observed_codes]
        if unused:
            raise ValueError(f"`obs[{sample_key!r}]` contains unused categories: {unused}.")

        if not adata.obs_names.is_unique:
            raise ValueError("Observation names must be unique.")
        if not adata.var_names.is_unique:
            raise ValueError("Variable names must be unique.")
        if adjacency_key not in adata.obsp:
            raise KeyError(f"Missing spatial graph `obsp[{adjacency_key!r}]`.")

        adjacency = adata.obsp[adjacency_key]
        expected_shape = (adata.n_obs, adata.n_obs)
        if adjacency.shape != expected_shape:
            raise ValueError(
                f"`obsp[{adjacency_key!r}]` has shape {adjacency.shape}; expected {expected_shape}.",
            )

        if sparse.issparse(adjacency):
            graph = adjacency.tocoo(copy=True)
            graph.eliminate_zeros()
            rows, columns = graph.row, graph.col
        else:
            rows, columns = np.nonzero(np.asarray(adjacency))
        if np.any(codes[rows] != codes[columns]):
            raise ValueError(f"`obsp[{adjacency_key!r}]` contains cross-sample edges.")

        positions = MappingProxyType({name: index for index, name in enumerate(categories)})
        indices = tuple(np.flatnonzero(codes == index) for index in range(len(categories)))
        return cls(
            sample_key=sample_key,
            names=categories,
            codes=codes,
            _positions=positions,
            _indices=indices,
        )

    def __len__(self) -> int:
        return len(self.names)

    def position(self, sample: str) -> int:
        """Return the integer position of a named sample."""

        try:
            return self._positions[str(sample)]
        except KeyError as error:
            raise KeyError(f"Unknown sample {sample!r}; expected one of {self.names}.") from error

    def indices(self, sample: str) -> np.ndarray:
        """Return observation indices belonging to a named sample."""

        return self._indices[self.position(sample)].copy()

    def mask(self, sample: str) -> np.ndarray:
        """Return an observation mask selecting a named sample."""

        return self.codes == self.position(sample)
