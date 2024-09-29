from dataclasses import dataclass

from popari.model import Popari


@dataclass
class TrainParameters:
    nmf_iterations: int
    iterations: int
    synchronization_frequency: int = 10


class Trainer:
    def __init__(parameters: TrainParameters, model: Popari, verbose: int = 0):
        self.model = model
        self.parameters = parameters
        self.verbose = verbose
        self.nmf_iterations = 0
        self.iterations = 0

    def train():
        for _ in range(self.parameters.nmf_iterations):
            if verbose > 0:
                print(f"-------------- NMF Iteration {self.nmf_iterations} --------------")

            synchronize = not (self.nmf_iterations % self.parameters.synchronization_frequency)
            self.model.estimate_parameters(update_spatial_affinities=False, synchronize=synchronize)
            self.model.estimate_weights(use_neighbors=False, synchronize=synchronize)

            self.nmf_iterations += 1

        for _ in range(self.parameters.iterations):
            if verbose > 0:
                print(f"------------------ Iteration {self.iterations} ------------------")

            synchronize = not (self.iterations % self.parameters.synchronization_frequency)

            self.model.estimate_parameters(synchronize=synchronize)
            self.model.estimate_weights(synchronize=synchronize)

            self.iterations += 1

    def save_results(): ...

    # TODO: decide whether to separate out saving of model training hyperparameters and
    # Popari parameters saving completely. This will obvious imply huge changes with how
    # MLflow should work. Perhaps even justifies switching entirely to WandB (a good excuse)
