from dataclasses import dataclass, field
from pathlib import Path

try:
    import mlflow
except ImportError:
    pass

from tqdm.auto import trange

from popari.model import Popari


@dataclass
class TrainParameters:
    nmf_iterations: int
    iterations: int
    savepath: Path
    synchronization_frequency: int = field(default=10, kw_only=True)


@dataclass
class MLFlowTrainParameters(TrainParameters): ...


class Trainer:
    def __init__(self, parameters: TrainParameters, model: Popari, verbose: int = 0):
        self.model = model
        self.parameters = parameters
        self.verbose = verbose
        self.nmf_iterations = 0
        self.iterations = 0

    def train(self):
        nmf_progress_bar = trange(self.parameters.nmf_iterations, leave=True, disable=not self.verbose)
        for _ in nmf_progress_bar:
            if self.verbose > 0:
                description = f"-------------- NMF Iteration {self.nmf_iterations} --------------"
                nmf_progress_bar.set_description(description)

            synchronize = not (self.nmf_iterations % self.parameters.synchronization_frequency)
            self.model.estimate_parameters(update_spatial_affinities=False, synchronize=synchronize)
            self.model.estimate_weights(use_neighbors=False, synchronize=synchronize)

            self.nmf_iterations += 1

        progress_bar = trange(self.parameters.iterations, leave=True, disable=not self.verbose)
        for _ in progress_bar:
            if self.verbose > 0:
                description = f"------------------ Iteration {self.iterations} ------------------"
                progress_bar.set_description(description)

            synchronize = not (self.iterations % self.parameters.synchronization_frequency)

            self.model.estimate_parameters(synchronize=synchronize)
            self.model.estimate_weights(synchronize=synchronize)

            self.iterations += 1

    def save_results(self, **kwargs):
        self.model.save_results(self.parameters.savepath, **kwargs)

    def superresolve(self, **kwargs):
        new_lr = kwargs.pop("new_lr", self.model.superresolution_lr)
        target_level = kwargs.pop("target_level", None)
        self.model.set_superresolution_lr(new_lr, target_level)
        self.model.superresolve(**kwargs)

    # TODO: decide whether to separate out saving of model training hyperparameters and
    # Popari parameters saving completely. This will obvious imply huge changes with how
    # MLflow should work. Perhaps even justifies switching entirely to WandB (a good excuse)


class MLFlowTrainer(Trainer):
    def save_results(self): ...
