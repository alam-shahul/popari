# CLI

Popari can also be used as a command line tool. After {ref}`installation <installation>` via pip, you can run Popari as below:
```
popari --K={number of metagenes} \
       --num_iterations={number of iterations} \
       --dataset_path={path to input .h5ad file} \
       --output_path={where to save output .h5ad file}
```

The above example includes just the required arguments. Below is the help message that includes the CLI including the names of all optional arguments (see {doc}`the Popari class documentation </api/popari.model.Popari>` for more details on these arguments):
```
usage: popari [-h] --K K --num_iterations NUM_ITERATIONS --nmf_preiterations NMF_PREITERATIONS --output_path OUTPUT_PATH [--dataset_path DATASET_PATH] [--lambda_Sigma_x_inv LAMBDA_SIGMA_X_INV]
              [--pretrained PRETRAINED] [--initialization_method INITIALIZATION_METHOD] [--metagene_groups METAGENE_GROUPS] [--spatial_affinity_groups SPATIAL_AFFINITY_GROUPS] [--betas BETAS]
              [--prior_x_modes PRIOR_X_MODES] [--M_constraint M_CONSTRAINT] [--sigma_yx_inv_mode SIGMA_YX_INV_MODE] [--torch_context TORCH_CONTEXT] [--initial_context INITIAL_CONTEXT]
              [--spatial_affinity_mode SPATIAL_AFFINITY_MODE] [--lambda_M LAMBDA_M] [--lambda_Sigma_bar LAMBDA_SIGMA_BAR] [--spatial_affinity_lr SPATIAL_AFFINITY_LR]
              [--spatial_affinity_tol SPATIAL_AFFINITY_TOL] [--spatial_affinity_constraint SPATIAL_AFFINITY_CONSTRAINT] [--spatial_affinity_centering SPATIAL_AFFINITY_CENTERING]
              [--spatial_affinity_scaling SPATIAL_AFFINITY_SCALING] [--spatial_affinity_regularization_power SPATIAL_AFFINITY_REGULARIZATION_POWER]
              [--embedding_mini_iterations EMBEDDING_MINI_ITERATIONS] [--embedding_acceleration_trick EMBEDDING_ACCELERATION_TRICK] [--embedding_step_size_multiplier EMBEDDING_STEP_SIZE_MULTIPLIER]
              [--use_inplace_ops USE_INPLACE_OPS] [--random_state RANDOM_STATE] [--verbose VERBOSE]

Run SpiceMix on specified dataset and device.

optional arguments:
  -h, --help            show this help message and exit
  --K K                 Number of metagenes to use for all replicates.
  --num_iterations NUM_ITERATIONS
                        Number of iterations to run Popari.
  --nmf_preiterations NMF_PREITERATIONS
                        Number of NMF preiterations to use for initialization.
  --output_path OUTPUT_PATH
                        Path at which to save Popari output.
  --dataset_path DATASET_PATH
                        Path to input dataset.
  --lambda_Sigma_x_inv LAMBDA_SIGMA_X_INV
                        Hyperparameter to balance importance of spatial information.
  --pretrained PRETRAINED
                        if set, attempts to load model state from input files
  --initialization_method INITIALIZATION_METHOD
                        algorithm to use for initializing metagenes and embeddings. Default ``svd``
  --metagene_groups METAGENE_GROUPS
                        defines a grouping of replicates for the metagene optimization.
  --spatial_affinity_groups SPATIAL_AFFINITY_GROUPS
                        defines a grouping of replicates for the spatial affinity optimization.
  --betas BETAS         weighting of each dataset during optimization. Defaults to equally weighting each dataset
  --prior_x_modes PRIOR_X_MODES
                        family of prior distribution for embeddings of each dataset
  --M_constraint M_CONSTRAINT
                        constraint on columns of M. Default ``simplex``
  --sigma_yx_inv_mode SIGMA_YX_INV_MODE
                        form of sigma_yx_inv parameter. Default ``separate``
  --torch_context TORCH_CONTEXT
                        keyword args to use of PyTorch tensors during training.
  --initial_context INITIAL_CONTEXT
                        keyword args to use during initialization of PyTorch tensors.
  --spatial_affinity_mode SPATIAL_AFFINITY_MODE
                        modality of spatial affinity parameters. Default ``shared lookup``
  --lambda_M LAMBDA_M   hyperparameter to constrain metagene deviation in differential case.
  --lambda_Sigma_bar LAMBDA_SIGMA_BAR
                        hyperparameter to constrain spatial affinity deviation in differential case.
  --spatial_affinity_lr SPATIAL_AFFINITY_LR
                        learning rate for optimization of ``Sigma_x_inv``
  --spatial_affinity_tol SPATIAL_AFFINITY_TOL
                        convergence tolerance during optimization of ``Sigma_x_inv``
  --spatial_affinity_constraint SPATIAL_AFFINITY_CONSTRAINT
                        method to ensure that spatial affinities lie within an appropriate range
  --spatial_affinity_centering SPATIAL_AFFINITY_CENTERING
                        if set, spatial affinities are zero-centered after every optimization step
  --spatial_affinity_scaling SPATIAL_AFFINITY_SCALING
                        magnitude of spatial affinities during initial scaling. Default ``10``
  --spatial_affinity_regularization_power SPATIAL_AFFINITY_REGULARIZATION_POWER
                        exponent controlling penalization of spatial affinity magnitudes. Default ``2``
  --embedding_mini_iterations EMBEDDING_MINI_ITERATIONS
                        number of mini-iterations to use during each iteration of embedding optimization. Default ``1000``
  --embedding_acceleration_trick EMBEDDING_ACCELERATION_TRICK
                        if set, use trick to accelerate convergence of embedding optimization. Default ``True``
  --embedding_step_size_multiplier EMBEDDING_STEP_SIZE_MULTIPLIER
                        controls relative step size during embedding optimization. Default ``1.0``
  --use_inplace_ops USE_INPLACE_OPS
                        if set, inplace PyTorch operations will be used to speed up computation
  --random_state RANDOM_STATE
                        seed for reproducibility of randomized computations. Default ``0``
  --verbose VERBOSE     level of verbosity to use during optimization. Default ``0`` (no print statements)
```
