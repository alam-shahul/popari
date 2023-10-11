import argparse
import json
import traceback
from pathlib import Path

from matplotlib import pyplot as plt
import torch

from popari.model import Popari
from popari import pl, tl

import mlflow

def main():
    parser = argparse.ArgumentParser(description='Run SpiceMix on specified dataset and device.')
    parser.add_argument('--K', type=int, required=True, default=10, help="Number of metagenes to use for all replicates.")
    parser.add_argument('--num_iterations', type=int, required=True, default=200, help="Number of iterations to run Popari.")
    parser.add_argument('--nmf_preiterations', type=int, default=5, help="Number of NMF preiterations to use for initialization.")
    parser.add_argument('--spatial_preiterations', type=int, default=5, help="Number of spatial preiterations to use for initialization.")
    parser.add_argument('--dtype', type=str, required=True, default="float64", help="Datatype to use for PyTorch operations. Choose between ``float32`` or ``float64``")
    parser.add_argument('--torch_device', type=str, required=True, help="keyword args to use of PyTorch tensors during training.")
    parser.add_argument('--initial_device', type=str, required=True, help="keyword args to use during initialization of PyTorch tensors.")
    parser.add_argument('--output_path', type=str, required=True, help="Path at which to save Popari output.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to input dataset.")
    parser.add_argument('--lambda_Sigma_x_inv', default=None, type=float, help="Hyperparameter to balance importance of spatial information.")
    parser.add_argument('--pretrained', type=bool, help="if set, attempts to load model state from input files")
    parser.add_argument('--initialization_method', type=str, help="algorithm to use for initializing metagenes and embeddings. Default ``svd``")
    parser.add_argument('--hierarchical_levels', type=int, default=None, help="Number of hierarchical binning levels to use in Popari run.")
    parser.add_argument('--binning_downsample_rate', type=float, default=None, help="Approximate rate at which spots are aggregated into bins.")
    parser.add_argument('--superresolution_lr', type=float, help="Initial learning rate for superresolution optimization.")
    parser.add_argument('--superresolution_epochs', type=int, default=10000, help="Number of epochs to do superresolution optimization.")
    parser.add_argument('--metagene_groups', type=json.loads, help="defines a grouping of replicates for the metagene optimization.")
    parser.add_argument('--spatial_affinity_groups', type=json.loads, help="defines a grouping of replicates for the spatial affinity optimization.")
    parser.add_argument('--betas', type=json.loads, help="weighting of each dataset during optimization. Defaults to equally weighting each dataset")
    parser.add_argument('--prior_x_modes', type=json.loads, help="family of prior distribution for embeddings of each dataset")
    parser.add_argument('--M_constraint', type=str, help="constraint on columns of M. Default ``simplex``")
    parser.add_argument('--sigma_yx_inv_mode', type=str, help="form of sigma_yx_inv parameter. Default ``separate``")
    parser.add_argument('--spatial_affinity_mode', type=str, help="modality of spatial affinity parameters. Default ``shared lookup``")
    parser.add_argument('--metagene_mode', type=str, help="modality of metagene parameters. Default ``shared``")
    parser.add_argument('--lambda_M', type=float, help="hyperparameter to constrain metagene deviation in differential case.")
    parser.add_argument('--lambda_Sigma_bar', type=float, default=None, help="hyperparameter to constrain spatial affinity deviation in differential case.")
    parser.add_argument('--spatial_affinity_lr', type=float, help="learning rate for optimization of ``Sigma_x_inv``")
    parser.add_argument('--spatial_affinity_tol', type=float, help="convergence tolerance during optimization of ``Sigma_x_inv``")
    parser.add_argument('--spatial_affinity_constraint', type=str, help="method to ensure that spatial affinities lie within an appropriate range")
    parser.add_argument('--spatial_affinity_centering', type=bool, help="if set, spatial affinities are zero-centered after every optimization step")
    parser.add_argument('--spatial_affinity_scaling', type=float, help="magnitude of spatial affinities during initial scaling. Default ``10``")
    parser.add_argument('--spatial_affinity_regularization_power', type=int, help="exponent controlling penalization of spatial affinity magnitudes. Default ``2``")
    parser.add_argument('--embedding_mini_iterations', type=int, help="number of mini-iterations to use during each iteration of embedding optimization. Default ``1000``")
    parser.add_argument('--embedding_acceleration_trick', type=bool, help="if set, use trick to accelerate convergence of embedding optimization. Default ``True``")
    parser.add_argument('--embedding_step_size_multiplier', type=float, help="controls relative step size during embedding optimization. Default ``1.0``")
    parser.add_argument('--use_inplace_ops', type=bool, help="if set, inplace PyTorch operations will be used to speed up computation")
    parser.add_argument('--random_state', type=int, help="seed for reproducibility of randomized computations. Default ``0``")
    parser.add_argument('--verbose', type=int, help="level of verbosity to use during optimization. Default ``0`` (no print statements)")

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
        trackable_hyperparameters = (
            'K',
            'lambda_Sigma_bar',
            'lambda_Sigma_x_inv',
            'hierarchical_levels',
            'binning_downsample_rate',
            'spatial_affinity_mode',
            'random_state',
            'dataset_path'
        )
                                            
        mlflow.log_params({
            **{hyperparameter: filtered_args[hyperparameter] for hyperparameter in trackable_hyperparameters if hyperparameter in filtered_args},
            "nmf_preiterations": nmf_preiterations,
            "spatial_preiterations": spatial_preiterations,
            "num_iterations": num_iterations,
        })
        try:
            model = Popari(**filtered_args)
    
            is_hierarchical = (model.hierarchical_levels > 1)

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
            with open(Path(f"./output_{torch_device}.txt"), 'a') as f:
                tb = traceback.format_exc()
                f.write(f'\n{tb}')
        finally:
            if Path(f"./output_{torch_device}.txt").is_file():
                mlflow.log_artifact(f"output_{torch_device}.txt")

def save_popari_figs(model: Popari, level: int = 0, save_spatial_figs: bool = False):
    """Save Popari figures.

    """
    
    is_hierarchical = (model.hierarchical_levels > 1)

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
