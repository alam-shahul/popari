import pytest

from popari.model import from_pretrained


@pytest.mark.expensive
def test_from_pretrained_switches_to_differential_affinities(hierarchical_model_factory, context):
    model = hierarchical_model_factory(hierarchical_levels=2)

    pretrained = from_pretrained(model, popari_context=context, lambda_Sigma_bar=1e-3)

    assert pretrained.groups == {sample: [sample] for sample in pretrained.replicate_names}
    assert pretrained.regularization_groups == {"_default": list(pretrained.replicate_names)}
    assert pretrained.adata.uns["M"] == pytest.approx(model.adata.uns["M"])
    assert pretrained.hierarchical_levels == model.hierarchical_levels
