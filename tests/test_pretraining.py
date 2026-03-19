import pytest

from popari.model import from_pretrained


@pytest.mark.expensive
def test_from_pretrained_switches_to_differential_mode(hierarchical_model_factory, context):
    model = hierarchical_model_factory(hierarchical_levels=2)

    pretrained = from_pretrained(model, popari_context=context, lambda_Sigma_bar=1e-3)

    assert pretrained.metagene_mode == "differential"
    assert pretrained.spatial_affinity_mode == "differential lookup"
    assert pretrained.hierarchical_levels == model.hierarchical_levels
