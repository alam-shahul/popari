import numpy as np
import pytest


@pytest.mark.expensive
def test_multigroup_spatial_affinity_groups_are_respected(dataset_factory, differential_model_factory):
    datasets = dataset_factory(num_replicates=3, replicate_names=["top", "bottom", "central"])
    model = differential_model_factory(
        datasets=datasets,
        replicate_names=["top", "bottom", "central"],
        spatial_affinity_groups={
            "vertical_gradient": ["top", "bottom"],
            "central_group": ["central"],
        },
    )

    model.estimate_parameters()
    model.estimate_weights()

    assert set(model.spatial_affinity_groups) == {"vertical_gradient", "central_group"}
    assert set(model.parameter_optimizer.spatial_affinity_bar.groups) == set(model.spatial_affinity_groups)

    for group_name, group_replicates in model.spatial_affinity_groups.items():
        average = sum(
            model.parameter_optimizer.spatial_affinity[dataset_name].detach().cpu().numpy()
            for dataset_name in group_replicates
        ) / len(group_replicates)
        assert np.allclose(
            average,
            model.parameter_optimizer.spatial_affinity_bar[group_name].detach().cpu().numpy(),
        )
