import argparse
import json
from pathlib import Path

from popari.model import Popari

def main():
    parser = argparse.ArgumentParser(description='Run SpiceMix on specified dataset and device.')
    parser.add_argument('--K', type=int, required=True, default=10, help="Number of metagenes to use for all replicates.")
    parser.add_argument('--num_iterations', type=int, required=True, default=200, help="Number of iterations to run Popari.")
    parser.add_argument('--nmf_preiterations', type=int, default=5, help="Number of NMF preiterations to use for initialization.")
    parser.add_argument('--output_path', type=str, required=True, help="Path at which to save Popari output.")
    parser.add_argument('--dataset_path', type=str, help="Path to input dataset.")
    parser.add_argument('--lambda_Sigma_x_inv', type=float, help="Hyperparameter to balance importance of spatial information.")
    parser.add_argument('--pretrained', type=bool, help="if set, attempts to load model state from input files")
    parser.add_argument('--initialization_method', type=str, help="algorithm to use for initializing metagenes and embeddings. Default ``svd``")
    parser.add_argument('--metagene_groups', type=json.loads, help="defines a grouping of replicates for the metagene optimization.")
    parser.add_argument('--spatial_affinity_groups', type=json.loads, help="defines a grouping of replicates for the spatial affinity optimization.")
    parser.add_argument('--betas', type=json.loads, help="weighting of each dataset during optimization. Defaults to equally weighting each dataset")
    parser.add_argument('--prior_x_modes', type=json.loads, help="family of prior distribution for embeddings of each dataset")
    parser.add_argument('--M_constraint', type=str, help="constraint on columns of M. Default ``simplex``")
    parser.add_argument('--sigma_yx_inv_mode', type=str, help="form of sigma_yx_inv parameter. Default ``separate``")
    parser.add_argument('--dtype', type=str, help="Datatype to use for PyTorch operations. Choose between ``float32`` or ``float64``")
    parser.add_argument('--torch_device', type=str, help="keyword args to use of PyTorch tensors during training.")
    parser.add_argument('--initial_device', type=str, help="keyword args to use during initialization of PyTorch tensors.")
    parser.add_argument('--spatial_affinity_mode', type=str, help="modality of spatial affinity parameters. Default ``shared lookup``")
    parser.add_argument('--lambda_M', type=float, help="hyperparameter to constrain metagene deviation in differential case.")
    parser.add_argument('--lambda_Sigma_bar', type=float, help="hyperparameter to constrain spatial affinity deviation in differential case.")
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
    filtered_args = {key: value for key, value in vars(args).items() if key is not None}

    num_iterations = filtered_args.pop("num_iterations")
    nmf_preiterations = filtered_args.pop("nmf_preiterations")
    
    torch_device = filtered_args.pop("torch_device")
    initial_device = filtered_args.pop("initial_device")
    dtype = filtered_args.pop("dtype")

    dtype_object = torch.float32
    if dtype == "float64":
        dtype_object = torch.float64

    initial_context = {"device": initial_device, "dtype": dtype_object}
    filtered_args["initial_context"] = initial_context

    torch_context = {"device": torch_device, "dtype": dtype_object}
    filtered_args["torch_context"] = torch_context

    output_path = filtered_args.pop("output_path")
    output_path = Path(output_path)

    model = Popari(**filtered_args)
    for preiteration in range(nmf_preiterations):
        print(f"----- Preiteration {preiteration} -----")
        model.estimate_parameters(update_spatial_affinities=False)
        model.estimate_weights(use_neighbors=False)

    for iteration in range(num_iterations):
        print(f"----- Iterations {iteration} -----")
        model.estimate_parameters()
        model.estimate_weights()

    model.save_results(output_path)
