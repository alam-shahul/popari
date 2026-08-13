from popari.train import Trainer


def train_model(
    model,
    *,
    nmf_iterations: int = 0,
    spatial_preiterations: int = 0,
    iterations: int = 1,
    spatial_affinity_epochs: int = 1000,
    verbose: int = 0,
    superresolve: bool = False,
) -> Trainer:
    """Train a model with concise defaults for tests."""

    trainer = Trainer(
        model,
        nmf_iterations=nmf_iterations,
        spatial_preiterations=spatial_preiterations,
        iterations=iterations,
        spatial_affinity_epochs=spatial_affinity_epochs,
        verbose=verbose,
    )
    if not superresolve:
        trainer.superresolution_completed = True
    trainer.train()
    return trainer
