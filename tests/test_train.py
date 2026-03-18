from pathlib import Path

import pytest

from popari.train import MLFlowTrainer, MLFlowTrainParameters, Trainer, TrainParameters


def test_trainer_runs_and_saves(shared_model_factory, tmp_path):
    model = shared_model_factory()
    savepath = tmp_path / "trainer_test.h5ad"
    trainer = Trainer(
        parameters=TrainParameters(
            nmf_iterations=0,
            iterations=1,
            savepath=savepath,
        ),
        model=model,
        verbose=False,
    )

    trainer.train()
    trainer.save_results()

    assert savepath.exists()


def test_mlflow_trainer_runs_if_available(shared_model_factory, tmp_path):
    try:
        import mlflow  # noqa: F401
    except ImportError:
        pytest.skip("mlflow is not installed in the current environment.")

    model = shared_model_factory()
    savepath = tmp_path / "mlflow_trainer_test.h5ad"
    trainer = MLFlowTrainer(
        parameters=MLFlowTrainParameters(
            nmf_iterations=0,
            spatial_preiterations=1,
            iterations=1,
            savepath=savepath,
            checkpoint_iterations=1,
        ),
        model=model,
        verbose=False,
    )

    with trainer:
        trainer.train()

    assert Path(savepath).exists()
