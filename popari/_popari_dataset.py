import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import seaborn as sns
import squidpy as sq
from scipy.sparse import csr_matrix

DATASET_NAME_KEY = "dataset_name"


def get_dataset_name(dataset: ad.AnnData) -> str:
    if DATASET_NAME_KEY in dataset.uns:
        return str(dataset.uns[DATASET_NAME_KEY])

    raise ValueError(f"Dataset name is not set in `uns[{DATASET_NAME_KEY!r}]`.")


def set_dataset_name(dataset: ad.AnnData, replicate_name: str, batch_key: str = "batch") -> ad.AnnData:
    dataset.uns[DATASET_NAME_KEY] = f"{replicate_name}"
    if batch_key in dataset.obs:
        dataset.obs[batch_key] = f"{replicate_name}"

    return dataset


def ensure_dataset_name(dataset: ad.AnnData, replicate_name: str | None = None, batch_key: str = "batch") -> ad.AnnData:
    if replicate_name is not None:
        return set_dataset_name(dataset, replicate_name, batch_key=batch_key)

    if DATASET_NAME_KEY in dataset.uns:
        dataset.uns[DATASET_NAME_KEY] = str(dataset.uns[DATASET_NAME_KEY])
        return dataset

    if batch_key not in dataset.obs:
        raise ValueError(
            f"Dataset name is missing from `uns[{DATASET_NAME_KEY!r}]`, and legacy batch column {batch_key!r} is absent.",
        )

    included_datasets = dataset.obs[batch_key].unique()
    if len(included_datasets) != 1:
        raise ValueError(
            f"Dataset name is missing from `uns[{DATASET_NAME_KEY!r}]`, and `obs[{batch_key!r}]` is not unique.",
        )

    dataset.uns[DATASET_NAME_KEY] = str(included_datasets[0])
    return dataset


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

    def name(self) -> str:
        return get_dataset_name(self._adata)

    def set_name(self, replicate_name: str, batch_key: str = "batch") -> ad.AnnData:
        return set_dataset_name(self._adata, replicate_name, batch_key=batch_key)

    def ensure_name(self, replicate_name: str | None = None, batch_key: str = "batch") -> ad.AnnData:
        return ensure_dataset_name(self._adata, replicate_name=replicate_name, batch_key=batch_key)

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


if not isinstance(getattr(ad.AnnData, "name", None), property):
    ad.AnnData.name = property(
        fget=lambda dataset: dataset.popari.name(),
        fset=lambda dataset, replicate_name: dataset.popari.set_name(replicate_name),
    )
