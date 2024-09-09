import textwrap
from pathlib import Path
from popari.model import load_trained_model

def generate_mlproject_file(project_name: str, output_filepath: str = "MLproject"):
    """Generate MLproject file for command line use.

    Args:
        project_name: name of project

    """

    mlproject_contents = f"""
    name: {project_name}

    entry_points:
      train_debug:
        parameters:
          dataset_path: {{type: str, default: "input.h5ad"}}
          output_path: {{type: str, default: "output.h5ad"}} 
          K: {{type: int}}
          dtype: {{type: str}}
          torch_device: {{type: str}}
          initial_device: {{type: str}}
          lambda_Sigma_x_inv: {{type: float, default: 1e-4}} 
          lambda_Sigma_bar: {{type: float, default: 0}}
          spatial_affinity_groups: {{type: str, default: 'null'}}
          spatial_affinity_mode: {{type: str, default: 'shared lookup'}}
          metagene_groups: {{type: str, default: 'null'}}
          hierarchical_levels: {{type: int}}
          binning_downsample_rate: {{type: float, default: 0.2}}
          num_iterations: {{type: int}} 
          spatial_preiterations: {{type: int}} 
          nmf_preiterations: {{type: int}} 
          verbose: {{type: int, default: 0}}
          random_state: {{type: int, default: 0}} 
        command: "popari-mlflow --dataset_path={{dataset_path}} --output_path={{output_path}}
                                     --K={{K}} --dtype={{dtype}} --torch_device={{torch_device}}
                                     --initial_device={{initial_device}}
                                     --lambda_Sigma_x_inv={{lambda_Sigma_x_inv}}
                                     --lambda_Sigma_bar={{lambda_Sigma_bar}}
                                     --spatial_affinity_groups={{spatial_affinity_groups}}
                                     --spatial_affinity_mode={{spatial_affinity_mode}}
                                     --metagene_groups={{metagene_groups}}
                                     --hierarchical_levels={{hierarchical_levels}}
                                     --binning_downsample_rate={{binning_downsample_rate}}
                                     --num_iterations={{num_iterations}}
                                     --spatial_preiterations={{spatial_preiterations}}
                                     --nmf_preiterations={{nmf_preiterations}}
                                     --verbose={{verbose}}
                                     --random_state={{random_state}}"
      
      train:
        parameters:
          dataset_path: {{type: str, default: "input.h5ad"}}
          output_path: {{type: str, default: "output.h5ad"}} 
          K: {{type: int}}
          dtype: {{type: str}}
          torch_device: {{type: str}}
          initial_device: {{type: str}}
          lambda_Sigma_x_inv: {{type: float, default: 1e-4}} 
          lambda_Sigma_bar: {{type: float, default: 0}}
          spatial_affinity_groups: {{type: str, default: 'null'}}
          spatial_affinity_mode: {{type: str, default: 'shared lookup'}}
          metagene_groups: {{type: str, default: 'null'}}
          hierarchical_levels: {{type: int}}
          binning_downsample_rate: {{type: float, default: 0.2}}
          num_iterations: {{type: int}} 
          spatial_preiterations: {{type: int}} 
          nmf_preiterations: {{type: int}} 
          verbose: {{type: int, default: 0}}
          random_state: {{type: int, default: 0}} 
        command: "popari-mlflow --dataset_path={{dataset_path}} --output_path={{output_path}}
                                     --K={{K}} --dtype={{dtype}} --torch_device={{torch_device}}
                                     --initial_device={{initial_device}}
                                     --lambda_Sigma_x_inv={{lambda_Sigma_x_inv}}
                                     --lambda_Sigma_bar={{lambda_Sigma_bar}}
                                     --spatial_affinity_groups={{spatial_affinity_groups}}
                                     --spatial_affinity_mode={{spatial_affinity_mode}}
                                     --metagene_groups={{metagene_groups}}
                                     --hierarchical_levels={{hierarchical_levels}}
                                     --binning_downsample_rate={{binning_downsample_rate}}
                                     --num_iterations={{num_iterations}}
                                     --spatial_preiterations={{spatial_preiterations}}
                                     --nmf_preiterations={{nmf_preiterations}}
                                     --verbose={{verbose}}
                                     --random_state={{random_state}} > output_{{torch_device}}.txt 2>&1"
      
      popari:
        parameters:
          configuration_filepath: path
        command: "popari-grid-search --configuration_filepath {{configuration_filepath}}"
    """

    with open(output_filepath, "w") as f:
        f.writelines(textwrap.dedent(mlproject_contents).strip())

def load_from_mlflow_run(run, client, file_suffix="_result", **loading_kwargs):
    """Load trained model from MLflow run.
    
    Args:
        run: MLflow run containing model artifact.
        client: MLflow client initialized to point to experiment location.
        loading_kwargs: kwargs that can be passed to Popari pretrained initialization.
    
    """
    run_id  = run.info.run_id
    artifacts = client.list_artifacts(run_id)

    trained_model_artifact, = list(filter(lambda artifact: Path(artifact.path).stem.endswith(file_suffix), artifacts))

    print(run.data.params)
    _, base_uri = run.info.artifact_uri.split(":")
    trained_model_filepath = Path(base_uri) / trained_model_artifact.path

    trained_model = load_trained_model(trained_model_filepath, **loading_kwargs)
    
    return trained_model
