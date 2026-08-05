import numpy as np
import torch
from scipy.sparse import csr_array


def convert_numpy_to_pytorch_sparse_coo(numpy_coo, context):
    """Convert a SciPy sparse matrix to a coalesced PyTorch COO tensor."""

    matrix = csr_array(numpy_coo)
    if np.any(matrix.data == 0):
        matrix = matrix.copy()
        matrix.eliminate_zeros()
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
    values = torch.as_tensor(coo.data, dtype=context["dtype"])
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, **context).coalesce()
