# Grid Search with MLflow

Using MLflow, we can effectively grid search for the best combination of Popari hyperparameters. Note that you must have {ref}`installed <installation>` the optional MLflow-related dependencies first. 

## Configuring your grid search
In order to specify the range of Popari hyperparameters over which to grid search, we have designed a [TOML](https://en.wikipedia.org/wiki/TOML)-formatted configuration file. Below is an example config, annotated with comments explaining the purpose of each setting:  

```
[runtime]
project_name = "example_project" # Name of the project
dataset_path = "/path/to/dataset.h5ad" # Path to the preprocessed Popari input
output_path =  "./result.h5ad" # Name of Popari output file
max_p = 4 # Max number of parallel Popari jobs to run at once; limited by number of available CUDA-enabled devices

[hyperparameters]

nmf_preiterations = 5 # Number of NMF preiterations to use (for initializing metagenes and embeddings)
spatial_preiterations = 50 # Number of SpiceMix preiterations to use (for initializing spatial affinities)
num_iterations = 200 # Number of Popari iterations to use

    [hyperparameters.K] # Grid search config for K (number of metagenes); the below yields [10, 15, 20] as options
    start = 10 # First value along grid search axis
    end = 20 # Last value along grid search axis
    gridpoints = 3 # Number of equally-spaced grid points to divide the range into
    scale = 'linear' # Scale on which to interpret the range 
    dtype = 'int' # Datatype of hyperparameter; if 'int', values will be rounded to the nearest integer

    [hyperparameters.lambda_Sigma_x_inv] # -> [1e-4]
    start = -4
    end = -4
    gridpoints = 1
    scale = 'log'
    dtype = 'float'

    [hyperparameters.lambda_Sigma_bar] # -> [1e-3, 1e-4, 1e-5]
    start = -5
    end = 3
    gridpoints = 3
    scale = 'log'
    dtype = 'float'

    [hyperparameters.binning_downsample_rate] # -> [0.25, 0.5]
    start = 0.25
    end = 0.5
    gridpoints = 2
    scale = 'linear'
    dtype = 'float'

    [hyperparameters.hierarchical_levels] # -> [2]
    start = 2
    end = 2
    gridpoints = 1
    scale = 'linear'
    dtype = 'int'
```

The above configuration would yield `3 * 1 * 3 * 2 * 1 = 12` hyperparameter combinations, implying that `12` Popari jobs will be executed by the MLflow framework. Furthermore, according to the `max_p` setting, a maximum of `4` of these processes will be running at any single moment in time.

To start a grid search, navigate to the folder where you want to store your grid search results (e.g. `example_grid_search`, and then run the `popari-grid-search` script:

```console
mkdir example_grid_search
cd example_grid_search
popari-grid-search --configuration_filepath=/path/to/config_file.toml
```

## View results
To view the results of your grid search (included relevant metrics), run the following command from the same folder

```console
mlflow ui --port=5000
```

and navigate to `localhost:5000` on your web browser.
