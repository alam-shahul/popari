# Tracking and parameter sweeps

Popari can log training metrics and the final canonical model artifact to
Weights & Biases. Install the optional tracking dependency first:

```bash
pip install popari[wandb]
```

Enable tracking for a command-line training run with `--use-wandb`:

```bash
uv run python scripts/train.py \
  --dataset_path input.h5ad \
  --output_path output.h5ad \
  --K 10 \
  --nmf_preiterations 5 \
  --spatial_preiterations 0 \
  --num_iterations 200 \
  --initial_device cpu \
  --torch_device cuda \
  --use-wandb \
  --wandb-project popari
```

The run records Popari hyperparameters and loss metrics. Its final model is
uploaded as a `popari-model` artifact that can be loaded with
`popari.wandb_util.load_popari_model_from_wandb`.

Parameter sweeps use the standard W&B sweep mechanism. Define the desired
grid in a W&B sweep configuration and configure its command to invoke the
`popari` executable with `${args}`. Popari does not maintain a separate sweep
engine.
