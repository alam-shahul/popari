from dataclasses import dataclass, field
from pathlib import Path

from matplotlib import pyplot as plt

try:
    import mlflow
except ImportError:
    pass

from tqdm.auto import trange

from popari import analysis as tl
from popari import plotting as pl
from popari._dataset_utils import setup_squarish_axes
from popari.model import Popari
from popari.util import get_datetime


@dataclass
class TrainParameters:
    nmf_iterations: int
    iterations: int
    savepath: Path
    synchronization_frequency: int = field(default=10, kw_only=True)


@dataclass
class MLFlowTrainParameters:
    nmf_iterations: int
    spatial_preiterations: int
    iterations: int
    savepath: Path
    synchronization_frequency: int = field(default=50, kw_only=True)
    checkpoint_iterations: int = field(default=50, kw_only=True)
    save_figs: bool = field(default=False, kw_only=True)


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

    def save_results(self, savepath=None, **kwargs):
        if savepath is None:
            savepath = self.parameters.savepath

        self.model.save_results(savepath, **kwargs)

    def superresolve(self, **kwargs):
        new_lr = kwargs.pop("new_lr", self.model.superresolution_lr)
        target_level = kwargs.pop("target_level", None)
        self.model.set_superresolution_lr(new_lr, target_level)
        self.model.superresolve(**kwargs)

    # TODO: decide whether to separate out saving of model training hyperparameters and
    # Popari parameters saving completely. This will obvious imply huge changes with how
    # MLflow should work. Perhaps even justifies switching entirely to WandB (a good excuse)


class MLFlowTrainer(Trainer):
    def __init__(self, parameters: MLFlowTrainParameters, model: Popari, verbose: int = 0):
        super().__init__(parameters, model, verbose)
        self.spatial_preiterations = 0
        self.torch_device = self.model.context["device"]
        self.log_system_metrics: bool = True

        self.is_hierarchical = self.model.hierarchical_levels > 1

        if not self.is_hierarchical:
            path_without_extension = self.parameters.savepath.parent / self.parameters.savepath.stem
            self.parameters.savepath = f"{path_without_extension}.h5ad"

    def train(self):
        nll = self.model.nll()
        mlflow.log_metric("nll", nll, step=-1)
        nmf_progress_bar = trange(self.parameters.nmf_iterations, leave=True, disable=not self.verbose)
        for _ in nmf_progress_bar:
            if self.verbose > 0:
                description = f"-------------- NMF Iteration {self.nmf_iterations} --------------"
                nmf_progress_bar.set_description(description)

            synchronize = not (self.nmf_iterations % self.parameters.synchronization_frequency)
            self.model.estimate_parameters(
                update_spatial_affinities=False,
                differentiate_metagenes=False,
                synchronize=synchronize,
            )
            self.model.estimate_weights(use_neighbors=False, synchronize=synchronize)

            self.nmf_iterations += 1

        nll = self.model.nll()
        mlflow.log_metric("nll", nll, step=self.nmf_iterations)

        self.model.parameter_optimizer.reinitialize_spatial_affinities()
        self.model.synchronize_datasets()

        nll_spatial = self.model.nll(use_spatial=True)
        mlflow.log_metric("nll_spatial", nll_spatial, step=-1)

        # if self.parameters.save_figs:
        #     for hierarchical_level in range(self.model.hierarchical_levels):
        #         self.save_popari_figs(level=hierarchical_level, save_spatial_figs=False)

        self.save_results(self.parameters.savepath)

        spatial_preprogress_bar = trange(self.parameters.spatial_preiterations, leave=True, disable=not self.verbose)
        for _ in spatial_preprogress_bar:
            if self.verbose > 0:
                description = f"-------------- Spatial Preiteration {self.spatial_preiterations} --------------"
                spatial_preprogress_bar.set_description(description)

            synchronize = not (self.spatial_preiterations % self.parameters.synchronization_frequency)
            self.model.estimate_parameters(differentiate_spatial_affinities=False, synchronize=synchronize)
            self.model.estimate_weights(synchronize=synchronize)

            if self.spatial_preiterations % self.parameters.checkpoint_iterations == 0:
                nll_spatial = self.model.nll(use_spatial=True)
                mlflow.log_metric("nll_spatial_preiteration", nll_spatial, step=self.spatial_preiterations)

                # if self.parameters.save_figs:
                #     for hierarchical_level in range(self.model.hierarchical_levels):
                #         self.save_popari_figs(level=hierarchical_level, save_spatial_figs=True)

                if Path(f"./output_{self.torch_device}.txt").is_file():
                    mlflow.log_artifact(f"output_{self.torch_device}.txt")

                self.save_results(self.parameters.savepath)

            self.spatial_preiterations += 1

        progress_bar = trange(self.parameters.iterations, leave=True, disable=not self.verbose)
        for _ in progress_bar:
            if self.verbose > 0:
                description = f"------------------------- Iteration {self.iterations} -------------------------"
                progress_bar.set_description(description)

            synchronize = not (self.iterations % self.parameters.synchronization_frequency)

            self.model.estimate_parameters(synchronize=synchronize)
            self.model.estimate_weights(synchronize=synchronize)

            if self.iterations % self.parameters.checkpoint_iterations == 0:
                if self.verbose:
                    print(f"{get_datetime()} Logging `nll_spatial`")

                checkpoint_path = f"{self.torch_device}_checkpoint_{self.iterations}_iterations"
                if not self.is_hierarchical:
                    checkpoint_path = f"{checkpoint_path}.h5ad"

                nll_spatial = self.model.nll(use_spatial=True)
                mlflow.log_metric("nll_spatial", nll_spatial, step=self.iterations)

                # if self.parameters.save_figs:
                #     for hierarchical_level in range(self.model.hierarchical_levels):
                #         self.save_popari_figs(level=hierarchical_level, save_spatial_figs=True)

                if Path(f"./output_{self.torch_device}.txt").is_file():
                    mlflow.log_artifact(f"output_{self.torch_device}.txt")

                self.save_results(checkpoint_path)

            self.iterations += 1

        self.save_results(self.parameters.savepath, ignore_raw_data=False)

        # if self.nmf_iterations + self.iterations > 0 and self.parameters.save_figs:
        #     for hierarchical_level in range(self.model.hierarchical_levels):
        #         self.save_popari_figs(level=hierarchical_level, save_spatial_figs=True)

    def superresolve(self, **kwargs):
        super().superresolve(**kwargs)

        # if self.nmf_iterations + self.iterations > 0 and self.parameters.save_figs:
        #     for hierarchical_level in range(self.model.hierarchical_levels):
        #         self.save_popari_figs(level=hierarchical_level, save_spatial_figs=True)

    def __enter__(self):
        return mlflow.start_run(log_system_metrics=self.log_system_metrics)

    def __exit__(self, error_type, value, traceback):
        mlflow.end_run()

    def save_results(self, savepath=None, **kwargs):
        if savepath is None:
            savepath = self.parameters.savepath

        super().save_results(savepath=savepath, **kwargs)
        mlflow.log_artifact(savepath)

    def save_popari_figs(self, level: int = 0, save_spatial_figs: bool = False):
        """Save Popari figures."""

        if not self.is_hierarchical:
            suffix = ".png"
        else:
            suffix = f"_level_{level}.png"

        if save_spatial_figs:
            if self.verbose:
                print(f"{get_datetime()} Plotting spatial affinities at level {level}")

            pl.spatial_affinities(self.model, level=level)

            plt.savefig(f"Sigma_x_inv{suffix}")
            plt.close()
            mlflow.log_artifact(f"Sigma_x_inv{suffix}")

        for metagene in range(self.model.K):
            if self.verbose:
                print(f"{get_datetime()} Plotting 'in situ' metagene {self.model.K} at level {level}")

            pl.metagene_embedding(self.model, metagene, level=level)
            plt.savefig(f"metagene_{metagene}_in_situ{suffix}")
            plt.close()

            mlflow.log_artifact(f"metagene_{metagene}_in_situ{suffix}")
