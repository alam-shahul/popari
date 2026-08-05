import numpy as np
from scipy.sparse import csr_array


def graph_neighbors(adjacency: csr_array, index: int) -> np.ndarray:
    """Return neighbors of one node from a CSR graph."""

    return adjacency.indices[adjacency.indptr[index] : adjacency.indptr[index + 1]]
