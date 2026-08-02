import numpy as np
import pytest

from popari import tl


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_parameter_updates_are_finite(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)

    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    assert np.isfinite(model.parameter_optimizer.metagenes.detach().cpu().numpy()).all()
    for sample in model.adata.popari.sample_names:
        assert np.isfinite(model.embedding_optimizer.embedding_state[sample].detach().cpu().numpy()).all()
        assert np.isfinite(model.parameter_optimizer.spatial_affinity[sample].detach().cpu().numpy()).all()


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_affinity_group_averages_track_replicate_parameters(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    model.estimate_parameters()

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
def test_differential_affinity_analysis_pipeline_runs(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    for _ in range(2):
        model.estimate_parameters()
        model.estimate_weights()

    tl.normalized_affinity_trends(
        model.adata,
        timepoint_values=list(range(len(model.adata.popari.sample_names))),
    )
