import shutil
from pathlib import Path

import pytest

from popari.train import MLFlowTrainer, MLFlowTrainParameters, Trainer, TrainParameters


@pytest.fixture(scope="module", autouse=True)
def cleanup_mlflow(request):
    def remove_mlflow_outputs():
        shutil.rmtree("mlruns")
        root = Path(".")
        for path in root.glob("*.h5ad"):
            path.unlink()

        for path in root.glob("metagene_*_in_situ.png"):
            path.unlink()

        (root / "leiden.png").unlink()
        (root / "Sigma_x_inv.png").unlink()
        (root / "metagenes.png").unlink()

    request.addfinalizer(remove_mlflow_outputs)


def test_trainer(test_datapath, shared_model):
    train_parameters = TrainParameters(
        nmf_iterations=0,
        iterations=1,
        savepath=(test_datapath / f"trainer_test.h5ad"),
    )

    trainer = Trainer(
        parameters=train_parameters,
        model=shared_model,
        verbose=True,
    )

    trainer.train()


def test_mlflow_trainer(test_datapath, shared_model):
    try:
        import mlflow
    except ImportError as e:
        pytest.skip("`[mlflow]` dependencyies must be installed for `MLFlowTrainer` test.")

    train_parameters = MLFlowTrainParameters(
        nmf_iterations=0,
        spatial_preiterations=1,
        iterations=1,
        savepath=(test_datapath / f"mlflow_trainer_test.h5ad"),
    )

    trainer = MLFlowTrainer(
        parameters=train_parameters,
        model=shared_model,
        verbose=True,
    )

    with trainer:
        trainer.train()
