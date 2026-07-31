import argparse
import json
from pathlib import Path

import torch

from popari.model import Popari
from popari.train import Trainer, TrainParameters


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value; received {value!r}.")


def get_parser() -> argparse.ArgumentParser:
    """Build the Popari training command-line parser."""

    parser = argparse.ArgumentParser(description="Train Popari on a canonical AnnData artifact.")
    parser.add_argument("--K", type=int, required=True, help="Number of shared metagenes.")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of spatial training iterations.")
    parser.add_argument("--nmf_preiterations", type=int, default=5, help="Number of NMF warmup iterations.")
    parser.add_argument("--spatial_preiterations", type=int, default=0, help="Spatial warmup iterations.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--torch_device", required=True, help="PyTorch device used during training.")
    parser.add_argument("--initial_device", required=True, help="PyTorch device used during initialization.")
    parser.add_argument("--output_path", type=Path, required=True, help="Destination for the trained model.")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Canonical input H5AD artifact.")
    parser.add_argument("--superresolution_epochs", type=int, default=0)
    parser.add_argument("--synchronization_frequency", type=int, default=10)

    parser.add_argument("--lambda_Sigma_x_inv", type=float)
    parser.add_argument("--lambda_Sigma_bar", type=float)
    parser.add_argument("--pretrained", type=_parse_bool)
    parser.add_argument("--initialization_method")
    parser.add_argument("--hierarchical_levels", type=int)
    parser.add_argument("--binning_downsample_rate", type=float)
    parser.add_argument("--superresolution_lr", type=float)
    parser.add_argument("--spatial_affinity_groups", type=json.loads)
    parser.add_argument("--betas", type=json.loads)
    parser.add_argument("--prior_x_modes", type=json.loads)
    parser.add_argument("--M_constraint")
    parser.add_argument("--sigma_yx_inv_mode")
    parser.add_argument("--spatial_affinity_mode")
    parser.add_argument("--downsampling_method")
    parser.add_argument("--spatial_affinity_lr", type=float)
    parser.add_argument("--spatial_affinity_tol", type=float)
    parser.add_argument("--spatial_affinity_constraint")
    parser.add_argument("--spatial_affinity_centering", type=_parse_bool)
    parser.add_argument("--spatial_affinity_scaling", type=float)
    parser.add_argument("--spatial_affinity_regularization_power", type=int)
    parser.add_argument("--embedding_mini_iterations", type=int)
    parser.add_argument("--embedding_acceleration_trick", type=_parse_bool)
    parser.add_argument("--embedding_step_size_multiplier", type=float)
    parser.add_argument("--use_inplace_ops", type=_parse_bool)
    parser.add_argument("--random_state", type=int)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--use-wandb", action="store_true", help="Track this run with Weights & Biases.")
    parser.add_argument("--wandb-project", default="popari")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    values = vars(args)

    training_keys = {
        "num_iterations",
        "nmf_preiterations",
        "spatial_preiterations",
        "output_path",
        "superresolution_epochs",
        "synchronization_frequency",
    }
    wandb_keys = {
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_name",
        "wandb_mode",
    }
    runtime_keys = {"dtype", "torch_device", "initial_device"}
    model_kwargs = {
        key: value
        for key, value in values.items()
        if key not in training_keys | wandb_keys | runtime_keys and value is not None
    }

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    model_kwargs["initial_context"] = {"device": args.initial_device, "dtype": dtype}
    model_kwargs["torch_context"] = {"device": args.torch_device, "dtype": dtype}
    model = Popari(**model_kwargs)

    parameters = TrainParameters(
        nmf_iterations=args.nmf_preiterations,
        spatial_preiterations=args.spatial_preiterations,
        iterations=args.num_iterations,
        savepath=args.output_path,
        synchronization_frequency=args.synchronization_frequency,
    )
    wandb_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "name": args.wandb_name,
        "mode": args.wandb_mode,
        "config": {
            "dataset_path": str(args.dataset_path),
            "output_path": str(args.output_path),
        },
    }
    wandb_kwargs = {key: value for key, value in wandb_kwargs.items() if value is not None}

    with Trainer(
        parameters,
        model,
        verbose=args.verbose,
        use_wandb=args.use_wandb,
        wandb_kwargs=wandb_kwargs,
    ) as trainer:
        trainer.train()
        if args.superresolution_epochs > 0:
            trainer.superresolve(n_epochs=args.superresolution_epochs)
        trainer.save_results(ignore_raw_data=False)


if __name__ == "__main__":
    main()
