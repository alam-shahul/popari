import numpy as np
import pytest

from tests._training import train_model


@pytest.mark.gpu
@pytest.mark.expensive
def test_multigroup_spatial_affinity_groups_are_respected(adata_factory, differential_model_factory, gpu_context):
    adata = adata_factory(num_replicates=3, replicate_names=["top", "bottom", "central"])
    model = differential_model_factory(
        adata=adata,
        spatial_affinity_groups={
            "vertical_gradient": ["top", "bottom"],
            "central_group": ["central"],
        },
        torch_context=gpu_context,
        initial_context=gpu_context,
    )

    train_model(model)

    assert set(model.spatial_affinity_groups) == {"vertical_gradient", "central_group"}
    assert set(model.hierarchy[-1].spatial_affinity.groups) == set(model.spatial_affinity_groups)

    group_means = model.hierarchy[-1].spatial_affinity.group_means()
    for group_name, group_replicates in model.spatial_affinity_groups.items():
        average = sum(
            model.hierarchy[-1].spatial_affinity.for_sample(dataset_name).detach().cpu().numpy()
            for dataset_name in group_replicates
        ) / len(group_replicates)
        assert np.allclose(
            average,
            group_means[group_name].cpu().numpy(),
        )
