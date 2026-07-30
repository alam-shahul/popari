import anndata as ad
import numpy as np
import pytest
import torch

from popari.wandb_util import load_popari_model_from_wandb, read_popari_anndata_hierarchy


def _write_h5ad(path):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.write_h5ad(path)
    return dataset


def test_read_popari_anndata_hierarchy_loads_flat_h5ad(tmp_path, monkeypatch):
    h5ad_path = tmp_path / "output.h5ad"
    dataset = _write_h5ad(h5ad_path)
    monkeypatch.setattr("popari.wandb_util.unmerge_anndata", lambda merged: ([merged], ["replicate"]))

    hierarchy = read_popari_anndata_hierarchy(h5ad_path)

    assert tuple(hierarchy) == (0,)
    assert hierarchy[0][0].shape == dataset.shape


def test_read_popari_anndata_hierarchy_loads_level_h5ads(tmp_path, monkeypatch):
    level_0_path = tmp_path / "level_0.h5ad"
    level_1_path = tmp_path / "level_1.h5ad"
    _write_h5ad(level_0_path)
    _write_h5ad(level_1_path)
    monkeypatch.setattr("popari.wandb_util.unmerge_anndata", lambda merged: ([merged], ["replicate"]))

    hierarchy = read_popari_anndata_hierarchy(tmp_path)

    assert tuple(hierarchy) == (0, 1)
    assert hierarchy[0][0].shape == (2, 2)
    assert hierarchy[1][0].shape == (2, 2)


def test_read_popari_anndata_hierarchy_rejects_multiple_flat_h5ads(tmp_path):
    _write_h5ad(tmp_path / "first.h5ad")
    _write_h5ad(tmp_path / "second.h5ad")

    with pytest.raises(ValueError, match="Expected one flat"):
        read_popari_anndata_hierarchy(tmp_path)


def test_load_popari_model_from_wandb_uses_anndata_hierarchy(monkeypatch):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.popari.name = "replicate"
    hierarchy = {0: (dataset,)}
    captured = {}

    monkeypatch.setattr(
        "popari.wandb_util.load_popari_anndata_from_wandb",
        lambda *args, **kwargs: hierarchy,
    )

    def fake_load_pretrained(datasets, replicate_names, **kwargs):
        captured["datasets"] = datasets
        captured["replicate_names"] = replicate_names
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr("popari.wandb_util.load_pretrained", fake_load_pretrained)

    model = load_popari_model_from_wandb("run-id")

    assert model == "model"
    assert captured["datasets"] == hierarchy[0]
    assert captured["replicate_names"] == ["replicate"]
    assert captured["kwargs"]["reloaded_hierarchy"] == hierarchy
    assert captured["kwargs"]["hierarchical_levels"] == 1
    assert captured["kwargs"]["context"] == {"device": "cpu", "dtype": torch.float64}
