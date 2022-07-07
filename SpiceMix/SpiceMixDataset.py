import anndata as ad
import scanpy as sc
import squidpy as sq

import numpy as np
from scipy.sparse import csr_matrix

class SpiceMixDataset(ad.AnnData):
    """Wrapper around AnnData object.


    """

    def __init__(self, dataset, replicate_name, coordinates_key="spatial"):
        """
        """
        super()._init_as_view(self, dataset)
        # self.X = dataset.X
        # self.obs = dataset.obs
        # self.obsp = dataset.obsp
        # self.obsm = dataset.obsm
        # self.var = dataset.var
        # self.varp = dataset.varp
        # self.uns = dataset.uns
      
        self.coordinates_key = coordinates_key 
        if self.coordinates_key not in self.obsm:
            raise ValueError("Dataset must include spatial coordinates.")

        self.obs["name"] = f"{replicate_name}"

    def compute_spatial_neighbors(self):
        """Compute neighbor graph based on spatial coordinates.

        """

        sq.gr.spatial_neighbors(self, coord_type="generic", delaunay=True)
        distance_matrix, adjacency_matrix = self.obsp["spatial_distances"], self.obsp["spatial_connectivities"]
        self.obsp["spatial_connectivities"] = SpiceMixDataset.remove_connectivity_artifacts(distance_matrix, adjacency_matrix)
        self.obsp["adjacency_matrix"] = self.obsp["spatial_connectivities"]
        
        num_cells, _ = self.obsp["adjacency_matrix"].shape

        adjacency_list = [[] for _ in range(num_cells)]
        for x, y in zip(*self.obsp["adjacency_matrix"].nonzero()):
            adjacency_list[x].append(y)

        self.obs["adjacency_list"] = adjacency_list

    @staticmethod
    def remove_connectivity_artifacts(sparse_distance_matrix, sparse_adjacency_matrix, threshold=94.5):
        dense_distances = sparse_distance_matrix.toarray()
        distances = sparse_distance_matrix.data
        cutoff = np.percentile(distances, threshold)
        mask = dense_distances < cutoff

        return csr_matrix(sparse_adjacency_matrix * mask)
 
    def update_Sigma_x_inv(self):
        if "Sigma_x_inv" not in self.uns:
            raise ValueError("Dataset must include Sigma_x_inv")

        if "adjacency_matrix" not in self.obsp:
            raise ValueError("Dataset must include spatial adjacency matrix.")

    def initialize_Sigma_x_inv(self):
        """Another one?

        """
        pass

    def initialize_parameters(self, K, shared_metagenes=None):
        """Initialize state for metagenes and spatial affinities.

        """

        if shared_metagenes is not None:
            self.uns["M"] = {
                self.uns["name"]: shared_metagenes
            }
        else:
            M = np.zeros()
            self.uns["M"] = {
                self.uns["name"]: M
            }
            
        pass

    def initialize_weights(self):
        """Initialize state for cell embeddings.
        
        """

        pass
