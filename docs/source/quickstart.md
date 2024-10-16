# Quickstart

## Load datasets
```python
from pathlib import Path

import anndata as ad
import torch

from popari.model import Popari

datasets = []
replicate_names = []
for fov in range(5):
    dataset = ad.read_h5ad(f"./example_st_dataset_fov_{replicate}.h5ad") # Each dataset must have spatial information stored as an adjacency matrix
    name = f"{fov}"
    datasets.append(dataset)
    replicate_names.append(name)
```

## Define hyperparameters
```python
K = 20 # Number of metagenes
lambda_Sigma_x_inv = 1e-4 # Spatial affinity regularization hyperparameter
torch_context = dict(device='cuda:0', dtype=torch.float32) # Context for PyTorch tensor instantiation 
```

## Initialize
```python
popari_demo = Popari(
    K=K,
    datasets=datasets,
    lambda_Sigma_x_inv=lambda_Sigma_x_inv,
    torch_context=torch_context
)
```    
## Train
```python
# Initialization with NMF
for iteration in range(5):
    popari_demo.estimate_parameters(update_spatial_affinities=False)
    popari_demo.estimate_weights(use_neighbors=False)

# Using spatial information
num_iterations = 200
for iteration in range(num_iterations):
    popari_demo.estimate_parameters()
    popari_demo.estimate_weights()
```

## Save to disk
```python
result_filepath = Path(f"./demo_{num_iterations}_iterations.h5ad")
popari_demo.save_results(result_filepath)
```

See **Analysis Demo** for examples of how to analyze Popari outputs.
