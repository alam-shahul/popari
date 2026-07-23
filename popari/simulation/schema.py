"""AnnData namespace defining Popari's simulation-result schema."""

from __future__ import annotations

import anndata as ad


@ad.register_anndata_namespace("simulation")
class MetageneSimulationNamespace:
    """Access simulation-specific AnnData fields through explicit properties."""

    def __init__(self, adata: ad.AnnData):
        self._adata = adata

    @property
    def ground_truth_X(self):
        """Ground-truth cell-by-metagene embeddings."""

        return self._adata.obsm["ground_truth_X"]

    @ground_truth_X.setter
    def ground_truth_X(self, value) -> None:
        self._adata.obsm["ground_truth_X"] = value

    @property
    def ground_truth_M(self):
        """Ground-truth gene-by-metagene matrix for this dataset."""

        return self._adata.uns["ground_truth_M"][self._adata.popari.name]

    @ground_truth_M.setter
    def ground_truth_M(self, value) -> None:
        dataset_name = self._adata.popari.name
        self._adata.uns.setdefault("ground_truth_M", {})
        self._adata.uns["ground_truth_M"][dataset_name] = value

    @property
    def learned_X(self):
        """Learned cell-by-metagene embeddings."""

        return self._adata.obsm["X"]

    @learned_X.setter
    def learned_X(self, value) -> None:
        self._adata.obsm["X"] = value

    @property
    def learned_M(self):
        """Learned gene-by-metagene matrix for this dataset."""

        return self._adata.uns["M"][self._adata.popari.name]

    @learned_M.setter
    def learned_M(self, value) -> None:
        dataset_name = self._adata.popari.name
        self._adata.uns.setdefault("M", {})
        self._adata.uns["M"][dataset_name] = value

    @property
    def ground_truth_expression(self):
        """Ground-truth reconstructed expression, ``ground_truth_X @
        ground_truth_M.T``."""

        return self.ground_truth_X @ self.ground_truth_M.T


__all__ = [MetageneSimulationNamespace.__name__]
