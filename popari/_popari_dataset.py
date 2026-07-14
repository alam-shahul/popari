import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import seaborn as sns
import squidpy as sq
from scipy.sparse import csr_matrix

DATASET_NAME_KEY = "dataset_name"


def remove_connectivity_artifacts(
    sparse_distance_matrix: csr_matrix,
    sparse_adjacency_matrix: csr_matrix,
    threshold: float = 94.5,
):
    """Remove long-range artifacts from spatial-connectivity heuristics."""

    dense_distances = sparse_distance_matrix.toarray()
    distances = sparse_distance_matrix.data
    cutoff = np.percentile(distances, threshold)
    mask = dense_distances < cutoff

    sparse_adjacency_matrix[~mask] = 0
    sparse_adjacency_matrix.eliminate_zeros()

    return sparse_adjacency_matrix


def plot_metagene_embedding(
    dataset: ad.AnnData,
    metagene_index: int,
    embedding_key: str = "X",
    coordinates_key: str = "spatial",
    **scatterplot_kwargs,
):
    points = dataset.obsm[coordinates_key]
    x, y = points.T
    embedding = dataset.obsm[embedding_key][:, metagene_index]

    biased_batch_effect = pd.DataFrame({"x": x, "y": y, f"Metagene {metagene_index}": embedding})
    sns.scatterplot(data=biased_batch_effect, x="x", y="y", hue=f"Metagene {metagene_index}", **scatterplot_kwargs)


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
        if "batch" in self._adata.obs:
            self._adata.obs["batch"] = f"{replicate_name}"

    def ensure_name(self, replicate_name: str | None = None, batch_key: str = "batch") -> ad.AnnData:
        if replicate_name is not None:
            self._adata.uns[DATASET_NAME_KEY] = f"{replicate_name}"
            if batch_key in self._adata.obs:
                self._adata.obs[batch_key] = f"{replicate_name}"
            return self._adata

        if DATASET_NAME_KEY in self._adata.uns:
            self._adata.uns[DATASET_NAME_KEY] = str(self._adata.uns[DATASET_NAME_KEY])
            return self._adata

        if batch_key not in self._adata.obs:
            raise ValueError(
                f"Dataset name is missing from `uns[{DATASET_NAME_KEY!r}]`, and legacy batch column {batch_key!r} is absent.",
            )

        included_datasets = self._adata.obs[batch_key].unique()
        if len(included_datasets) != 1:
            raise ValueError(
                f"Dataset name is missing from `uns[{DATASET_NAME_KEY!r}]`, and `obs[{batch_key!r}]` is not unique.",
            )

        self._adata.uns[DATASET_NAME_KEY] = str(included_datasets[0])
        return self._adata

    def compute_spatial_neighbors(self, threshold: float = 94.5):
        """Compute neighbor graph based on spatial coordinates."""

        sq.gr.spatial_neighbors(self._adata, coord_type="generic", delaunay=True)
        distance_matrix = self._adata.obsp["spatial_distances"]
        distances = distance_matrix.data
        cutoff = np.percentile(distances, threshold)

        sq.gr.spatial_neighbors(
            self._adata,
            coord_type="generic",
            delaunay=True,
            radius=[0, cutoff],
        )
        self._adata.obsp["adjacency_matrix"] = self._adata.obsp["spatial_connectivities"]

        num_cells, _ = self._adata.obsp["adjacency_matrix"].shape

        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*self._adata.obsp["adjacency_matrix"].nonzero()):
            adjacency_list[x].append(y)

        self._adata.obsm["adjacency_list"] = ak.Array(adjacency_list)

    def affinity_difference(
        self,
        numerator: str,
        denominator: str,
        spatial_affinity_key: str = "Sigma_x_inv",
    ) -> np.ndarray:
        """Return the difference between two named spatial affinity matrices.

        Args:
            numerator: Name of the dataset whose affinity matrix is subtracted from.
            denominator: Name of the dataset whose affinity matrix is subtracted.
            spatial_affinity_key: Key in ``.uns`` containing named affinity matrices.

        Returns:
            ``uns[spatial_affinity_key][numerator] - uns[spatial_affinity_key][denominator]``.

        """

        if spatial_affinity_key not in self._adata.uns:
            raise KeyError(f"Missing spatial affinity key in `.uns`: {spatial_affinity_key!r}.")

        spatial_affinities = self._adata.uns[spatial_affinity_key]
        missing_names = [name for name in (numerator, denominator) if name not in spatial_affinities]
        if missing_names:
            raise KeyError(
                f"Missing spatial affinity matrix/matrices under `.uns[{spatial_affinity_key!r}]`: "
                f"{missing_names}.",
            )

        return np.asarray(spatial_affinities[numerator]) - np.asarray(spatial_affinities[denominator])

    def plot_metagene_embedding(self, metagene_index: int, embedding_key: str = "X", **scatterplot_kwargs):
        return plot_metagene_embedding(
            self._adata,
            metagene_index=metagene_index,
            embedding_key=embedding_key,
            **scatterplot_kwargs,
        )

    @staticmethod
    def remove_connectivity_artifacts(
        sparse_distance_matrix: csr_matrix,
        sparse_adjacency_matrix: csr_matrix,
        threshold: float = 94.5,
    ):
        return remove_connectivity_artifacts(
            sparse_distance_matrix=sparse_distance_matrix,
            sparse_adjacency_matrix=sparse_adjacency_matrix,
            threshold=threshold,
        )


PopariDataset = ad.AnnData
