from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import itertools
import toml

import numpy as np
import torch

import mlflow
import mlflow.tracking
import mlflow.projects
from mlflow.tracking import MlflowClient

from pathlib import Path
from popari.mlflow.util import generate_mlproject_file

def run():
    parser = argparse.ArgumentParser(description='Run Popari hyperparameter grid search.')
    parser.add_argument('--configuration_filepath', type=str, required=True, help="Path to configuration file.")
    parser.add_argument('--mlflow_output_path', type=str, default=".", help="Where to output MLflow results.")

    args = parser.parse_args()

    with open(args.configuration_filepath, "r") as f:
        configuration = toml.load(f)

    mlflow_output_path = Path(args.mlflow_output_path)
    generate_mlproject_file(configuration['runtime']['project_name'], mlflow_output_path / "MLproject")

    hyperparameter_names = ('K', 'lambda_Sigma_x_inv', 'lambda_Sigma_bar', 'hierarchical_levels', 'binning_downsample_rate')
    spatial_affinity_groups = configuration['spatial_affinity_groups'] if 'spatial_affinity_groups' in configuration else "null"
    nmf_preiterations = configuration['hyperparameters']['nmf_preiterations']
    spatial_preiterations = configuration['hyperparameters']['spatial_preiterations']
    num_iterations = configuration['hyperparameters']['num_iterations']

    max_p = configuration['runtime']['max_p']

    tracking_client = MlflowClient()

    device_status = [False for _ in range(max_p)]
    def generate_evaluate_function(parent_run, nmf_preiterations, spatial_preiterations, num_iterations, experiment_id, null_nll=0):
        """Generates function to evaluate Popari.

        """

        def evaluate(hyperparams):
            """Start parallel Popari run and track progress.


            """

            # (K, lambda_Sigma_x_inv, lambda_M, lambda_Sigma_bar) = hyperparams
          
            device = device_status.index(False)
            device_status[device] = True 
            child_run = tracking_client.create_run(experiment_id, tags={"mlflow.parentRunId": parent_run.info.run_id})
            # with mlflow.start_run(nested=True) as child_run:
            p = mlflow.projects.run(
                run_id=child_run.info.run_id,
                uri=str(mlflow_output_path),
                entry_point="train_debug",
                parameters={
                    **{hyperparameter_name: hyperparams[hyperparameter_name] for hyperparameter_name in hyperparameter_names},
                    "spatial_affinity_mode": "shared lookup" if hyperparams['lambda_Sigma_bar'] == 0 else "differential lookup",
                    "spatial_affinity_groups": "null" if hyperparams['lambda_Sigma_bar'] == 0 else spatial_affinity_groups,
                    "dataset_path": configuration['runtime']['dataset_path'],
                    "output_path": configuration['runtime']['output_path'],
                    "num_iterations": num_iterations,
                    "spatial_preiterations": spatial_preiterations,
                    "dtype": "float64",
                    "torch_device": f"cuda:{device}",
                    "initial_device": f"cuda:{device}",
                    "nmf_preiterations": nmf_preiterations,
                    "verbose": 1,
                    "random_state": 0,
                },
                env_manager="local",
                experiment_id=experiment_id,
                synchronous=False,
            )
            succeeded = p.wait()
            device_status[device] = False

            tracking_client.set_terminated(child_run.info.run_id)

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                nll = metrics["nll"]
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")
                nll = null_nll

            mlflow.log_metrics({"nll": nll})

            return p.run_id, nll

        return evaluate

    with mlflow.start_run() as parent_run:
        experiment_id = parent_run.info.experiment_id
        null_evaluate = generate_evaluate_function(parent_run, 0, 0, 0, experiment_id)
        null_hyperparameters = {
            "K": 2,
            "lambda_Sigma_x_inv": 0,
            "lambda_Sigma_bar": 0,
            "hierarchical_levels": 1,
            "binning_downsample_rate": 0.5
        }
        _, null_nll = null_evaluate(null_hyperparameters)

        hyperparameter_options_list = []
        for hyperparameter_name in hyperparameter_names:
            search_space = configuration['hyperparameters'][hyperparameter_name]

            start = search_space['start']
            end = search_space['end']
            scale = search_space['scale']
            gridpoints = search_space['gridpoints']
            dtype = search_space['dtype']

            if scale == 'log':
                gridspace = np.logspace
            elif scale == 'linear':
                gridspace = np.linspace
            
            hyperparameter_options = gridspace(start, end, num=gridpoints)

            if dtype == 'int':
                hyperparameter_options = np.rint(hyperparameter_options).astype(int)

            hyperparameter_options_list.append(hyperparameter_options)

        options = list(dict(zip(hyperparameter_names, hyperparameter_choice)) for hyperparameter_choice in itertools.product(*hyperparameter_options_list))

        print(f"Number of grid search candidates: {len(options)}")

        evaluate = generate_evaluate_function(parent_run, nmf_preiterations, spatial_preiterations, num_iterations, experiment_id, null_nll)
        with ThreadPoolExecutor(max_workers=max_p) as executor:
            _ = executor.map(evaluate, options,)

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' "
        )
        best_nll = np.finfo(np.float64).max
        best_run = None

        nlls = [run.data.metrics["nll"] for run in runs]
        best_nll = np.min(nlls)
        best_run = runs[np.argmin(nlls)]

        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                "nll": best_nll
            }
        )

if __name__ == "__main__":
    run()
