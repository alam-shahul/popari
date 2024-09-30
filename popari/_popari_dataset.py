from copy import deepcopy
from typing import Sequence

import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm, trange


class PopariDataset(ad.AnnData):
    r"""Wrapper around AnnData object. Allows for preprocessing of dataset for
    Popari.

    Attributes:
        dataset: AnnData dataset to convert into Popari-compatible object.
        replicate_name: name of dataset
        coordinates_key: location in ``.obsm`` dataframe of 2D coordinates for datapoints.

    """

    def __init__(
        self,
        dataset: ad.AnnData,
        replicate_name: str,
        coordinates_key: str = "spatial",
    ):
        super().__init__(
            X=dataset.X,
            obs=dataset.obs,
            obsm=dataset.obsm,
            obsp=dataset.obsp,
            var=dataset.var,
            varm=dataset.varm,
            varp=dataset.varp,
            uns=deepcopy(dataset.uns),
        )

        self.coordinates_key = coordinates_key
        if self.coordinates_key not in self.obsm:
            raise ValueError("Dataset must include spatial coordinates.")

        self.name = f"{replicate_name}"

    def compute_spatial_neighbors(self, threshold: float = 94.5):
        r"""Compute neighbor graph based on spatial coordinates.

        Stores resulting graph in ``self.obs["adjacency_list"]``.

        """

        sq.gr.spatial_neighbors(self, coord_type="generic", delaunay=True)
        distance_matrix = self.obsp["spatial_distances"]
        distances = distance_matrix.data
        cutoff = np.percentile(distances, threshold)

        sq.gr.spatial_neighbors(
            self,
            coord_type="generic",
            delaunay=True,
            radius=[0, cutoff],
        )
        self.obsp["adjacency_matrix"] = self.obsp["spatial_connectivities"]

        num_cells, _ = self.obsp["adjacency_matrix"].shape

        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*self.obsp["adjacency_matrix"].nonzero()):
            adjacency_list[x].append(y)

        self.obsm["adjacency_list"] = ak.Array(adjacency_list)

    @staticmethod
    def remove_connectivity_artifacts(
        sparse_distance_matrix: csr_matrix,
        sparse_adjacency_matrix: csr_matrix,
        threshold: float = 94.5,
    ):
        """Remove artifacts in adjacency matrices produced by heuristic
        algorithms.

        For example, when Delaunay triangulation is used to produce the adjacency matrix, some spots which are
        connected as a result may actually be far apart in Euclidean space.

        Args:
            sparse_distance_matrix:

        """
        dense_distances = sparse_distance_matrix.toarray()
        distances = sparse_distance_matrix.data
        cutoff = np.percentile(distances, threshold)
        mask = dense_distances < cutoff

        sparse_adjacency_matrix[~mask] = 0
        sparse_adjacency_matrix.eliminate_zeros()

        return sparse_adjacency_matrix

    def plot_metagene_embedding(self, metagene_index: int, embedding_key: str = "X", **scatterplot_kwargs):
        r"""Plot the embedding values for a embedding in-situ.

        Args:
            metagene_index: index of the embedding to plot.
            **scatterplot_kwargs: keyword args to pass to ``sns.scatterplot``.

        """
        points = self.obsm["spatial"]
        x, y = points.T
        embedding = self.obsm[embedding_key][:, metagene_index]

        biased_batch_effect = pd.DataFrame({"x": x, "y": y, f"Metagene {metagene_index}": embedding})
        sns.scatterplot(data=biased_batch_effect, x="x", y="y", hue=f"Metagene {metagene_index}", **scatterplot_kwargs)
