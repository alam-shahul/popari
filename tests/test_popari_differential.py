import numpy as np
import pytest

from popari import tl
from tests._training import train_model


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_parameter_updates_are_finite(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)

    train_model(model, iterations=2)

    assert np.isfinite(model.hierarchy[-1].metagenes.detach().cpu().numpy()).all()
    for sample in model.adata.popari.sample_names:
        assert np.isfinite(model.hierarchy[-1].embedding(sample).detach().cpu().numpy()).all()
        assert np.isfinite(model.hierarchy[-1].spatial_affinity.for_sample(sample).detach().cpu().numpy()).all()


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_affinity_group_averages_track_replicate_parameters(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    train_model(model)

    group_means, _ = model.hierarchy[-1].spatial_affinity.regularization_structure()
    for group_name, parameter_names in model.regularization_groups.items():
        average = sum(
            model.hierarchy[-1].spatial_affinity.for_parameter(parameter_name).detach().cpu().numpy()
            for parameter_name in parameter_names
        ) / len(parameter_names)
        assert np.allclose(
            average,
            group_means[group_name].cpu().numpy(),
        )


@pytest.mark.gpu
@pytest.mark.expensive
def test_differential_affinity_analysis_pipeline_runs(differential_model_factory, gpu_context):
    model = differential_model_factory(torch_context=gpu_context, initial_context=gpu_context)
    train_model(model, iterations=2)
    model.materialize_results()

    tl.normalized_affinity_trends(
        model.adata,
        timepoint_values=list(range(len(model.adata.popari.sample_names))),
    )
