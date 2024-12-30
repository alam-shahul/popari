import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import mlflow.projects
import mlflow.tracking
import numpy as np
import toml
from mlflow.tracking import MlflowClient

from popari.mlflow.util import generate_mlproject_file


def run():
    parser = argparse.ArgumentParser(description="Run Popari hyperparameter grid search.")
    parser.add_argument("--configuration_filepath", type=str, required=True, help="Path to configuration file.")
    parser.add_argument("--mlflow_output_path", type=str, default=".", help="Where to output MLflow results.")
    parser.add_argument("--debug", action="store_true", help="Whether to dump print statements to console or to file.")
    # parser.add_argument('--include_benchmarks', action='store_true', help="Whether to include NMF and SpiceMix benchmarks for each hyperparameter combination.")
    parser.add_argument("--benchmarks", action="extend", nargs="+", type=str, help="Which benchmarks to include.")

    parser.add_argument("--only_benchmarks", action="store_true", help="Whether to only run benchmarks.")

    args = parser.parse_args()

    with open(args.configuration_filepath) as f:
        configuration = toml.load(f)

    mlflow_output_path = Path(args.mlflow_output_path)
    generate_mlproject_file(configuration["runtime"]["project_name"], mlflow_output_path / "MLproject")

    hyperparameter_names = [
        "K",
        "lambda_Sigma_x_inv",
        "lambda_Sigma_bar",
        "random_state",
        "hierarchical_levels",
        "binning_downsample_rate",
        "nmf_preiterations",
        "num_iterations",
        "spatial_preiterations",
        "metagene_groups",
        "downsampling_method",
        "spatial_affinity_groups",
    ]

    if "dataset_paths" in configuration["runtime"]:
        dataset_paths = configuration["runtime"]["dataset_paths"]
    elif "dataset_path" in configuration["runtime"]:
        dataset_paths = [configuration["runtime"]["dataset_path"]]

    configuration["hyperparameters"]["dataset_path"] = dataset_paths

    # Fix this up so that you can input a list of CUDA devices instead of a number of processes
    if "device_list" in configuration["runtime"]:
        device_list = configuration["runtime"]["device_list"]
        num_processes = len(device_list)
    elif "num_processes" in configuration["runtime"]:
        num_processes = configuration["runtime"]["num_processes"]
        device_list = [f"cuda:{device_number}" for device_number in range(num_processes)]

    tracking_client = MlflowClient()

    device_status = [False for _ in range(num_processes)]

    def generate_evaluate_function(parent_run, experiment_id, null_nll=0):
        """Generates function to evaluate Popari."""

        def evaluate(params):
            """Start parallel Popari run and track progress."""

            device_index = device_status.index(False)
            device = device_list[device_index]
            device_status[device_index] = True

            child_run = tracking_client.create_run(experiment_id, tags={"mlflow.parentRunId": parent_run.info.run_id})
            # with mlflow.start_run(nested=True) as child_run:
            p = mlflow.projects.run(
                run_id=child_run.info.run_id,
                uri=str(mlflow_output_path),
                entry_point="train_debug" if args.debug else "train",
                parameters={
                    **{
                        parameter_name: params[parameter_name]
                        for parameter_name in params
                        if params[parameter_name] is not None
                    },
                    "spatial_affinity_mode": (
                        "shared lookup" if params["lambda_Sigma_bar"] == 0 else "differential lookup"
                    ),
                    # "dataset_path": configuration['runtime']['dataset_path'],
                    "output_path": f"./device_{device}_result",
                    "dtype": "float64",
                    "torch_device": device,
                    "initial_device": device,
                    "spatial_affinity_mode": (
                        "shared lookup" if params["lambda_Sigma_bar"] == 0 else "differential lookup"
                    ),
                    "verbose": 1,
                },
                env_manager="local",
                experiment_id=experiment_id,
                synchronous=False,
            )
            succeeded = p.wait()
            device_status[device_index] = False

            tracking_client.set_terminated(child_run.info.run_id)

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics

                if "nll" in metrics:
                    nll = metrics["nll"]
                else:
                    raise RuntimeError(
                        "A run failed during initialization. This likely points "
                        "to an improperly formatted grid search configuration file.",
                    )
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")
                nll = null_nll

            mlflow.log_metrics({"nll": nll})

            return p.run_id, nll

        return evaluate

    with mlflow.start_run() as parent_run:
        experiment_id = parent_run.info.experiment_id
        null_evaluate = generate_evaluate_function(parent_run, experiment_id)
        null_hyperparameters = {
            "K": configuration["hyperparameters"]["K"]["start"],
            "dataset_path": dataset_paths[0],
            "lambda_Sigma_x_inv": 0,
            "lambda_Sigma_bar": 0,
            "hierarchical_levels": 2,
            "binning_downsample_rate": 0.1,
            "random_state": 0,
            "nmf_preiterations": 0,
            "spatial_preiterations": 0,
            "num_iterations": 0,
            "spatial_affinity_groups": json.dumps(None),
            "metagene_groups": json.dumps(None),
        }
        _, null_nll = null_evaluate(null_hyperparameters)

        benchmarks = {
            "nmf_benchmark": {
                "nmf_preiterations": 200,
                "spatial_preiterations": 0,
                "num_iterations": 0,
                "lambda_Sigma_bar": 0,
                "lambda_Sigma_x_inv": 0,
            },
            "spicemix_benchmark": {
                "nmf_preiterations": 5,
                "spatial_preiterations": 200,
                "num_iterations": 0,
                "lambda_Sigma_bar": 0,
            },
            "disjoint_spicemix_benchmark": {
                "nmf_preiterations": 5,
                "spatial_preiterations": 200,
                "num_iterations": 0,
                "lambda_Sigma_bar": 0,
                "metagene_groups": json.dumps("disjoint"),
                "spatial_affinity_groups": json.dumps("disjoint"),
            },
        }

        benchmarks = {name: params for name, params in benchmarks.items() if name in args.benchmarks}

        hyperparameter_options_list = []
        for hyperparameter_name in hyperparameter_names:
            if hyperparameter_name not in configuration["hyperparameters"]:
                default_options = [null_hyperparameters[hyperparameter_name]]
                hyperparameter_options_list.append(default_options)
                continue

            search_space = configuration["hyperparameters"][hyperparameter_name]

            # Categorical hyperparameters are specified via the "options" field
            if dtype == "categorical":
                hyperparameter_options = search_space["options"]
                continue

            # Handling numerical hyperparameters
            start = search_space["start"]
            end = search_space["end"]
            scale = search_space["scale"]
            gridpoints = search_space["gridpoints"]
            dtype = search_space["dtype"]

            if scale == "log":
                gridspace = np.logspace
            elif scale == "linear":
                gridspace = np.linspace

            hyperparameter_options = gridspace(start, end, num=gridpoints)

            if dtype == "int":
                hyperparameter_options = np.rint(hyperparameter_options).astype(int)

            hyperparameter_options_list.append(hyperparameter_options)

        hyperparameter_names.append("dataset_path")
        hyperparameter_options_list.append(dataset_paths)

        options = list(
            dict(zip(hyperparameter_names, hyperparameter_choice))
            for hyperparameter_choice in itertools.product(*hyperparameter_options_list)
        )

        special_options = []
        for benchmark_name, benchmark in benchmarks.items():
            is_novel = [
                True if hyperparameter_name in benchmark else False for hyperparameter_name in hyperparameter_names
            ]

            special_options_list = [
                hyperparameter_options if not is_novel[index] else [benchmark[hyperparameter_name]]
                for index, (hyperparameter_name, hyperparameter_options) in enumerate(
                    zip(hyperparameter_names, hyperparameter_options_list),
                )
            ]
            special_options.extend(
                list(
                    dict(zip(hyperparameter_names, hyperparameter_choice))
                    for hyperparameter_choice in itertools.product(*special_options_list)
                ),
            )

        if args.only_benchmarks:
            options = special_options
        else:
            options.extend(special_options)

        print(f"Number of grid search candidates: {len(options)}")

        evaluate = generate_evaluate_function(parent_run, experiment_id, null_nll)
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = executor.map(evaluate, options)

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id],
            f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' ",
        )
        best_nll = np.finfo(np.float64).max
        best_run = None

        nlls = [run.data.metrics["nll"] for run in runs]
        best_nll = np.min(nlls)
        best_run = runs[np.argmin(nlls)]

        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "nll": best_nll,
            },
        )


if __name__ == "__main__":
    run()
