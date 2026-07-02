"""Training helpers for simulation experiments."""

from __future__ import annotations

from pathlib import Path


def train_popari_model(
    dataset,
    *,
    K: int,
    popari_kwargs=None,
    initialization_method: str = "ground_truth",
    nmf_iterations: int = 5,
    spatial_iterations: int = 20,
    random_state: int = 0,
    verbose: int = 1,
):
    """Train one Popari model for a single simulated dataset."""

    from scipy.sparse import csr_array

    from popari.model import Popari
    from popari.train import Trainer, TrainParameters

    popari_kwargs = {} if popari_kwargs is None else dict(popari_kwargs)

    dataset_copy = dataset.copy()
    dataset_copy.X = csr_array(dataset_copy.X)

    model_kwargs = {
        "K": K,
        "datasets": (dataset_copy,),
        "replicate_names": (dataset_copy.name,),
        "lambda_Sigma_x_inv": 1e-4,
        "initialization_method": initialization_method,
        "embedding_mini_iterations": 200,
        "random_state": random_state,
        "verbose": verbose,
    }
    model_kwargs.update(popari_kwargs)

    model = Popari(**model_kwargs)

    warmup_parameters = TrainParameters(
        nmf_iterations=nmf_iterations,
        iterations=0,
        savepath=Path("unused.h5ad"),
    )
    Trainer(warmup_parameters, model, verbose=verbose).train()

    model.parameter_optimizer.reinitialize_spatial_affinities()
    model.synchronize_datasets()

    spatial_parameters = TrainParameters(
        nmf_iterations=0,
        iterations=spatial_iterations,
        savepath=Path("unused.h5ad"),
    )
    Trainer(spatial_parameters, model, verbose=verbose).train()

    return model


def run_lambda_sweep(
    datasets,
    lambda_values,
    *,
    K: int,
    initialization_method: str = "ground_truth",
    nmf_iterations: int = 5,
    spatial_iterations: int = 20,
    random_state: int = 0,
    verbose: int = 1,
):
    """Train each dataset across a grid of ``lambda_Sigma_x_inv`` values."""

    return {
        dataset.name: {
            lambda_value: train_popari_model(
                dataset,
                K=K,
                initialization_method=initialization_method,
                nmf_iterations=nmf_iterations,
                spatial_iterations=spatial_iterations,
                random_state=random_state,
                verbose=verbose,
                popari_kwargs={"lambda_Sigma_x_inv": lambda_value},
            )
            for lambda_value in lambda_values
        }
        for dataset in datasets
    }


__all__ = [
    train_popari_model.__name__,
    run_lambda_sweep.__name__,
]
