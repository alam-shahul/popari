import argparse
import json
import traceback
from pathlib import Path

import mlflow
import torch
from matplotlib import pyplot as plt

from popari import get_parser, pl, tl
from popari.model import Popari


def main():
    parser = get_parser()

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

    with mlflow.start_run():
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
        try:
            model = Popari(**filtered_args)

            is_hierarchical = model.hierarchical_levels > 1

            if not is_hierarchical:
                path_without_extension = output_path.parent / output_path.stem
                output_path = f"{path_without_extension}.h5ad"

            nll = model.nll()
            mlflow.log_metric("nll", nll, step=-1)
            for preiteration in range(nmf_preiterations):
                print(f"----- Preiteration {preiteration} -----")
                model.estimate_parameters(update_spatial_affinities=False, differentiate_metagenes=False)
                model.estimate_weights(use_neighbors=False)
                nll = model.nll()
                mlflow.log_metric("nll", nll, step=preiteration)

            model.parameter_optimizer.reinitialize_spatial_affinities()
            model.synchronize_datasets()

            nll_spatial = model.nll(use_spatial=True)
            mlflow.log_metric("nll_spatial", nll_spatial, step=-1)

            for spatial_preiteration in range(spatial_preiterations):
                print(f"----- Spatial preiteration {spatial_preiteration} -----")
                model.estimate_parameters(differentiate_spatial_affinities=False, differentiate_metagenes=False)
                model.estimate_weights()
                nll_spatial = model.nll(use_spatial=True)
                mlflow.log_metric("nll_spatial_preiteration", nll_spatial, step=spatial_preiteration)
                if spatial_preiteration % checkpoint_iterations == 0:
                    model.save_results(output_path)
                    mlflow.log_artifact(output_path)

                    for hierarchical_level in range(model.hierarchical_levels):
                        save_popari_figs(model, level=hierarchical_level, save_spatial_figs=True)

                    if Path(f"./output_{torch_device}.txt").is_file():
                        mlflow.log_artifact(f"output_{torch_device}.txt")

            for iteration in range(num_iterations):
                print(f"----- Iteration {iteration} -----")
                model.estimate_parameters()
                model.estimate_weights()
                nll_spatial = model.nll(use_spatial=True)
                mlflow.log_metric("nll_spatial", nll_spatial, step=iteration)
                if iteration % checkpoint_iterations == 0:
                    checkpoint_path = f"{torch_device}_checkpoint_{iteration}_iterations"
                    if not is_hierarchical:
                        checkpoint_path = f"{checkpoint_path}.h5ad"

                    model.save_results(checkpoint_path)
                    mlflow.log_artifact(checkpoint_path)

                    for hierarchical_level in range(model.hierarchical_levels):
                        save_popari_figs(model, level=hierarchical_level, save_spatial_figs=True)

                    if Path(f"./output_{torch_device}.txt").is_file():
                        mlflow.log_artifact(f"output_{torch_device}.txt")

            if is_hierarchical:
                model.superresolve(n_epochs=superresolution_epochs)

            model.save_results(output_path, ignore_raw_data=False)
            mlflow.log_artifact(output_path)

            if nmf_preiterations + num_iterations > 0:
                for hierarchical_level in range(model.hierarchical_levels):
                    save_popari_figs(model, level=hierarchical_level, save_spatial_figs=True)

        except Exception as e:
            with open(Path(f"./output_{torch_device}.txt"), "a") as f:
                tb = traceback.format_exc()
                f.write(f"\n{tb}")
        finally:
            if Path(f"./output_{torch_device}.txt").is_file():
                mlflow.log_artifact(f"output_{torch_device}.txt")


def save_popari_figs(model: Popari, level: int = 0, save_spatial_figs: bool = False):
    """Save Popari figures."""

    is_hierarchical = model.hierarchical_levels > 1

    if not is_hierarchical:
        suffix = ".png"
    else:
        suffix = f"_level_{level}.png"

    tl.preprocess_embeddings(model, level=level)
    tl.leiden(model, level=level, joint=True)
    pl.in_situ(model, level=level, color="leiden")

    plt.savefig(f"leiden{suffix}")
    mlflow.log_artifact(f"leiden{suffix}")

    if save_spatial_figs:
        pl.spatial_affinities(model, level=level)

        plt.savefig(f"Sigma_x_inv{suffix}")
        mlflow.log_artifact(f"Sigma_x_inv{suffix}")

        pl.multireplicate_heatmap(model, level=level, uns="M", aspect=model.K / model.datasets[0].shape[1], cmap="hot")

        plt.savefig(f"metagenes{suffix}")
        mlflow.log_artifact(f"metagenes{suffix}")

    for metagene in range(model.K):
        pl.metagene_embedding(model, metagene, level=level)
        plt.savefig(f"metagene_{metagene}_in_situ{suffix}")
        mlflow.log_artifact(f"metagene_{metagene}_in_situ{suffix}")


if __name__ == "__main__":
    main()
