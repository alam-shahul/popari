import numpy as np
import pytest

from tests._training import train_model


@pytest.mark.gpu
@pytest.mark.expensive
def test_overlapping_regularization_groups_are_respected(adata_factory, differential_model_factory, gpu_context):
    adata = adata_factory(num_replicates=3, replicate_names=["top", "bottom", "central"])
    model = differential_model_factory(
        adata=adata,
        regularization_groups={
            "vertical_gradient": ["top", "bottom"],
            "central_group": ["central"],
        },
        torch_context=gpu_context,
        initial_context=gpu_context,
    )

    train_model(model)

    assert set(model.regularization_groups) == {"vertical_gradient", "central_group"}
    assert model.groups == {sample: [sample] for sample in model.replicate_names}

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
