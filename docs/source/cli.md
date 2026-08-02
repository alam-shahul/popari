# Training from the command line

From a Popari repository checkout, run the Hydra entry point:

```bash
uv run python scripts/train.py \
  data.dataset_path=/path/to/input.h5ad \
  model.K=10
```

`data.dataset_path` must be an absolute path.

Configuration is organized into `data`, `model`, `training`, and `tracking`
sections, with execution settings at the top level. Override any setting with
Hydra's `section.key=value` syntax:

```bash
uv run python scripts/train.py \
  data.dataset_path=/path/to/input.h5ad \
  model.K=15 \
  model.lambda_Sigma_x_inv=1e-4 \
  training.nmf_iterations=5 \
  training.iterations=200 \
  initial_device=cuda \
  torch_device=cuda \
  output_dir=/path/to/results
```

Inspect the resolved configuration without training:

```bash
uv run python scripts/train.py \
  data.dataset_path=/path/to/input.h5ad \
  model.K=10 \
  --cfg job --resolve
```

Popari hashes the data, model, training, dtype, and device configuration and
uses `output_dir/<config-uuid>/` as the result directory. A single-level model
is written to `model.h5ad`; a hierarchical model is written to
`model/level_0.h5ad`, `model/level_1.h5ad`, and so on. Popari raises an error if
the result directory already exists. Tracking settings and verbosity do not
affect the derived path.
