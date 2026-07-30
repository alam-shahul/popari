import numpy as np
import pytest

from popari import pl, tl


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_parameter_updates_are_finite(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)

    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    for dataset in model.datasets:
        assert np.isfinite(model.parameter_optimizer.metagene_state[dataset.popari.name].detach().cpu().numpy()).all()
        assert np.isfinite(model.embedding_optimizer.embedding_state[dataset.popari.name].detach().cpu().numpy()).all()
        assert np.isfinite(model.parameter_optimizer.spatial_affinity[dataset.popari.name].detach().cpu().numpy()).all()


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_group_averages_track_replicate_parameters(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    model.estimate_parameters()

    for group_name, group_replicates in model.metagene_groups.items():
        average = sum(
            model.parameter_optimizer.metagene_state[dataset_name].detach().cpu().numpy()
            for dataset_name in group_replicates
        ) / len(group_replicates)
        assert np.allclose(
            average,
            model.parameter_optimizer.metagene_state.M_bar[group_name].detach().cpu().numpy(),
        )

    for group_name, group_replicates in model.spatial_affinity_groups.items():
        average = sum(
            model.parameter_optimizer.spatial_affinity[dataset_name].detach().cpu().numpy()
            for dataset_name in group_replicates
        ) / len(group_replicates)
        assert np.allclose(
            average,
            model.parameter_optimizer.spatial_affinity_bar[group_name].detach().cpu().numpy(),
        )


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_analysis_pipeline_runs(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    genes = tl.find_differential_genes(model.datasets, top_gene_limit=2)
    assert genes
    assert set(genes).issubset(set(model.datasets[0].var_names))

    pl.gene_trajectories(model.datasets, list(genes)[:2], covariate_values=list(range(len(model.metagene_groups))))
    pl.gene_activations(model.datasets, list(genes)[:2])
    tl.normalized_affinity_trends(model.datasets, timepoint_values=list(range(len(model.datasets))))
