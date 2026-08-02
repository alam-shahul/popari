# Tracking and parameter sweeps

Install the optional Weights & Biases dependency:

```bash
pip install popari[wandb]
```

## Track one run

Tracking is opt-in and defaults to the `popari/revisions` W&B project:

```bash
uv run --extra wandb python scripts/train.py \
  data.dataset_path=/path/to/input.h5ad \
  model.K=10 \
  initial_device=cuda \
  torch_device=cuda \
  output_dir=/path/to/results \
  tracking.enabled=true
```

The run records the resolved Hydra configuration and training losses. Its
final canonical model is uploaded as a `popari-model` artifact that can be
loaded with `popari.wandb_util.load_popari_model_from_wandb`.
The local result directory is `output_dir/<config-uuid>/`, independently of
whether W&B tracking is enabled. It contains `model.h5ad` for a single-level
model or a `model/` directory for a hierarchy. W&B uploads whichever result
representation Popari creates.

## Run a W&B sweep

Edit `sweeps/popari.yaml` to set the dataset and parameter grid, then create
the sweep:

```bash
uv run --extra wandb wandb sweep sweeps/popari.yaml
```

Run the agent command returned by W&B:

```bash
uv run --extra wandb wandb agent popari/revisions/<sweep-id>
```

The sweep uses `${args_no_hyphens}` to pass each selected parameter as a Hydra
override such as `model.K=10`. Launch additional agents locally or in separate
SLURM jobs to run trials concurrently.

For a local sweep without W&B scheduling, use Hydra's multirun mode:

```bash
uv run python scripts/train.py -m \
  data.dataset_path=/path/to/input.h5ad \
  model.K=10,15,20 \
  model.random_state=0,1,2,3,4
```
