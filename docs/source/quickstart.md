# Quickstart

Popari operates on one unified, schema-v2 `AnnData` object. The categorical
sample column recorded in the Popari schema identifies the observations from
each spatial sample.

## Load a dataset

```python
import torch

from popari import Popari, Trainer
from popari.io import load_anndata, save_anndata_hierarchy

adata = load_anndata("./preprocessed_dataset.h5ad")
print(adata.popari.sample_names)
```

The dataset must contain spatial coordinates and a block-diagonal spatial
graph in `adata.obsp["adjacency_matrix"]`. Use the preprocessing tutorial to
construct a Popari dataset from raw spatial transcriptomics data.

## Initialize the model

```python
model = Popari(
    K=20,
    adata=adata,
    lambda_Sigma_x_inv=1e-4,
    initialization_method="leiden_fast",
    initial_context={"device": "cpu", "dtype": torch.float64},
    torch_context={"device": "cuda:0", "dtype": torch.float64},
)
```

`initial_context` controls where initialization runs, while `torch_context`
controls model optimization. Use `device="cpu"` for both when CUDA is not
available.

## Train

```python
trainer = Trainer(
    model,
    nmf_iterations=5,
    iterations=200,
    verbose=1,
)
trainer.train()
```

## Materialize and save results

```python
hierarchy = model.materialize_results()
save_anndata_hierarchy("./popari_results", hierarchy)
```

For a non-hierarchical model, the output directory contains `level_0.h5ad`.
Hierarchical models additionally write one H5AD file for each coarser level.

See the **Analysis Demo** for examples of analyzing materialized Popari
results.
