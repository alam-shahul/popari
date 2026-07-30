import anndata as ad
import numpy as np
import pandas as pd
import pytest

from popari import pp


def _dataset(labels):
    return ad.AnnData(
        X=np.ones((len(labels), 1)),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(labels)},
            index=[f"cell_{index}" for index in range(len(labels))],
        ),
    )


def test_relabel_categories_supports_sequences_and_many_to_one_mappings():
    first = _dataset(["0", "1", "unknown", None])
    second = _dataset(["1", "2"])

    result = pp.relabel_categories(
        [first, second],
        source="cluster",
        target="domain",
        mapping={"0": "cortex", "1": "cortex", "2": "medulla"},
    )

    assert result is None
    assert first.obs["domain"].iloc[:3].tolist() == ["cortex", "cortex", "unknown"]
    assert pd.isna(first.obs["domain"].iloc[3])
    assert second.obs["domain"].tolist() == ["cortex", "medulla"]
    assert isinstance(first.obs["domain"].dtype, pd.CategoricalDtype)
    assert isinstance(second.obs["domain"].dtype, pd.CategoricalDtype)


def test_relabel_categories_rejects_unknown_labels_before_mutation():
    first = _dataset(["0"])
    second = _dataset(["unknown"])

    with pytest.raises(ValueError, match="unknown"):
        pp.relabel_categories(
            [first, second],
            source="cluster",
            target="domain",
            mapping={"0": "cortex"},
            unmapped="error",
        )

    assert "domain" not in first.obs
    assert "domain" not in second.obs


def test_relabel_categories_validates_source_and_unmapped_mode():
    dataset = _dataset(["0"])

    with pytest.raises(KeyError, match="missing"):
        pp.relabel_categories(
            dataset,
            source="missing",
            target="domain",
            mapping={},
        )

    with pytest.raises(ValueError, match="unmapped"):
        pp.relabel_categories(
            dataset,
            source="cluster",
            target="domain",
            mapping={},
            unmapped="ignore",
        )
