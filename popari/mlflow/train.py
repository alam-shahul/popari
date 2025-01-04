import argparse
import json
import traceback
from pathlib import Path

import mlflow
import torch
from matplotlib import pyplot as plt

from popari import get_parser, pl, tl
from popari.model import Popari
from popari.train import MLFlowTrainer, MLFlowTrainParameters


def main():
    parser = get_parser()

    parser.add_argument(
        "--spatial_preiterations",
        type=int,
        default=0,
        help="Number of SpiceMix preiterations to use for initialization.",
    )

    args = parser.parse_args()
    filtered_args = {key: value for key, value in vars(args).items() if value is not None}

    print(filtered_args)

    num_iterations = filtered_args.pop("num_iterations")
    nmf_preiterations = filtered_args.pop("nmf_preiterations")
    spatial_preiterations = filtered_args.pop("spatial_preiterations")

    torch_device = filtered_args.pop("torch_device")
    initial_device = filtered_args.pop("initial_device")
    dtype = filtered_args.pop("dtype")

    superresolution_epochs = filtered_args.pop("superresolution_epochs", 0)

    dtype_object = torch.float32
    if dtype == "float64":
        dtype_object = torch.float64

    initial_context = {"device": initial_device, "dtype": dtype_object}
    filtered_args["initial_context"] = initial_context

    torch_context = {"device": torch_device, "dtype": dtype_object}
    filtered_args["torch_context"] = torch_context

    output_path = filtered_args.pop("output_path")
    output_path = Path(output_path)

    checkpoint_iterations = 50

    train_parameters = MLFlowTrainParameters(
        iterations=num_iterations,
        nmf_iterations=nmf_preiterations,
        spatial_preiterations=spatial_preiterations,
        savepath=output_path,
    )

    model = Popari(**filtered_args)

    mlflow_trainer = MLFlowTrainer(
        parameters=train_parameters,
        model=model,
        verbose=filtered_args["verbose"],
    )

    try:
        with mlflow_trainer:
            if "metagene_groups" in filtered_args:
                mlflow.set_tag("disjoint_metagenes", (filtered_args["metagene_groups"] == "disjoint"))
            if "spatial_affinity_groups" in filtered_args:
                mlflow.set_tag("disjoint_spatial_affinities", (filtered_args["spatial_affinity_groups"] == "disjoint"))

            trackable_hyperparameters = (
                "K",
                "lambda_Sigma_bar",
                "lambda_Sigma_x_inv",
                "hierarchical_levels",
                "downsampling_method",
                "binning_downsample_rate",
                "spatial_affinity_mode",
                "random_state",
                "dataset_path",
            )

            mlflow.log_params(
                {
                    **{
                        hyperparameter: filtered_args[hyperparameter]
                        for hyperparameter in trackable_hyperparameters
                        if hyperparameter in filtered_args
                    },
                    "nmf_preiterations": nmf_preiterations,
                    "spatial_preiterations": spatial_preiterations,
                    "num_iterations": num_iterations,
                },
            )
            mlflow_trainer.train()
            mlflow_trainer.superresolve(n_epochs=superresolution_epochs)
            mlflow_trainer.save_results()

    except Exception as e:
        with open(Path(f"./output_{torch_device}.txt"), "a") as f:
            tb = traceback.format_exc()
            f.write(f"\n{tb}")
    finally:
        if Path(f"./output_{torch_device}.txt").is_file():
            mlflow.log_artifact(f"output_{torch_device}.txt")


if __name__ == "__main__":
    main()
