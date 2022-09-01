# SpiceMix

![overview](./SpiceMix_overview.png)

SpiceMix is an unsupervised tool for analyzing data of the spatial transcriptome. SpiceMix models the observed expression of genes within a cell as a mixture of latent factors. These factors are assumed to have some spatial affinity between neighboring cells. The factors and affinities are not known a priori, but are learned by SpiceMix directly from the data, by an alternating optimization method that seeks to maximize their posterior probability given the observed gene expression. In this way, SpiceMix learns a more expressive representation of the identity of cells from their spatial transcriptome data than other available methods. 

SpiceMix can be applied to any type of spatial transcriptomics data, including MERFISH, seqFISH, HDST, and Slide-seq.

## Install

```
pip install spicemix
```

## Usage

```python
from pathlib import Path

import anndata as ad
import torch

from spicemix.model import SpiceMixPlus

# Load datasets
datasets = []
replicate_names = []
for fov in range(5):
    dataset = ad.read_h5ad(f"./example_st_dataset_fov_{replicate}.h5ad") # Each dataset must have spatial information stored as an adjacency matrix
    name = f"{fov}"
    datasets.append(dataset)
    replicate_names.append(name)

# Define hyperparameters
K = 20 # Number of metagenes
lambda_Sigma_x_inv = 1e-4 # Spatial affinity regularization hyperparameter
torch_context = dict(device='cuda:0', dtype=torch.float32) # Context for PyTorch tensor instantiation 

# Initialize
spicemixplus_demo = SpiceMixPlus(
    K=K,
    datasets=datasets,
    lambda_Sigma_x_inv=lambda_Sigma_x_inv,
    torch_context=torch_context
)
    
# Train

## Initialization with NMF
for iteration in range(10):
    spicemixplus_demo.estimate_parameters(update_spatial_affinities=False)
    spicemixplus_demo.estimate_weights(use_neighbors=False)

## Using spatial information
num_iterations = 200
for iteration in range(num_iterations):
    spicemixplus_demo.estimate_parameters()
    spicemixplus_demo.estimate_weights()

# Save to disk
result_filepath = Path(f"./demo_{num_iterations}_iterations.h5ad")
spicemixplus_demo.save_results(result_filepath)
    
# Plot results

...
```

## Tests

To run the provided tests and ensure that SpiceMix can run on your platform, follow the instructions below:

- Download this repo.
```console
git clone https://github.com/alam-shahul/SpiceMixPlus.git
```
- Install `pytest` in your environment.
```console
pip install pytest
```
- Navigate to the root directory of this repo.
- Run the following command. With GPU resources, this test should execute without errors in ~2.5 minutes:
```console
python -m pytest -s tests/test_spicemix_shared.py
```

## Cite

Cite our paper:

```
@article{chidester2020spicemix,
  title={SPICEMIX: Integrative single-cell spatial modeling for inferring cell identity},
  author={Chidester, Benjamin and Zhou, Tianming and Ma, Jian},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

![paper](./paper.png)
