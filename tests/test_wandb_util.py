from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_array

from popari.io import save_anndata
from popari.wandb_util import (
    load_popari_model_from_wandb,
    load_popari_results_from_wandb,
    log_popari_results,
    read_popari_anndata_hierarchy,
)


class _Artifact:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.files = []

    def add_file(self, path, *, name):
        self.files.append((path, name))


def _write_h5ad(path, *, coarse=False):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.obs_names = ["cell_0", "cell_1"]
    dataset.var_names = ["gene_0", "gene_1"]
    dataset.obs["batch"] = pd.Categorical(["replicate", "replicate"])
    dataset.obsp["adjacency_matrix"] = csr_array(np.eye(2))
    if coarse:
        dataset.obsm["bin_assignments"] = csr_array(np.eye(2))
    dataset.popari.name = "replicate"
    save_anndata(path, dataset)
    return dataset


def test_read_popari_anndata_hierarchy_loads_flat_h5ad(tmp_path, monkeypatch):
    h5ad_path = tmp_path / "output.h5ad"
    dataset = _write_h5ad(h5ad_path)

    hierarchy = read_popari_anndata_hierarchy(h5ad_path)

    assert tuple(hierarchy) == (0,)
    assert hierarchy[0].shape == dataset.shape
    assert hierarchy[0].popari.sample_names == ("replicate",)


def test_read_popari_anndata_hierarchy_loads_level_h5ads(tmp_path, monkeypatch):
    level_0_path = tmp_path / "level_0.h5ad"
    level_1_path = tmp_path / "level_1.h5ad"
    _write_h5ad(level_0_path)
    _write_h5ad(level_1_path, coarse=True)

    hierarchy = read_popari_anndata_hierarchy(tmp_path)

    assert tuple(hierarchy) == (0, 1)
    assert hierarchy[0].shape == (2, 2)
    assert hierarchy[1].shape == (2, 2)


def test_read_popari_anndata_hierarchy_rejects_multiple_flat_h5ads(tmp_path):
    _write_h5ad(tmp_path / "first.h5ad")
    _write_h5ad(tmp_path / "second.h5ad")

    with pytest.raises(ValueError, match="Expected one flat"):
        read_popari_anndata_hierarchy(tmp_path)


def test_load_popari_model_from_wandb_uses_anndata_hierarchy(monkeypatch):
    dataset = ad.AnnData(X=np.ones((2, 2)))
    dataset.obs_names = ["cell_0", "cell_1"]
    dataset.var_names = ["gene_0", "gene_1"]
    dataset.obs["batch"] = pd.Categorical(["replicate", "replicate"])
    dataset.obsp["adjacency_matrix"] = csr_array(np.eye(2))
    dataset.popari.name = "replicate"
    hierarchy = {0: dataset}
    captured = {}

    monkeypatch.setattr("popari.wandb_util.load_popari_results_from_wandb", lambda *args, **kwargs: hierarchy)

    def fake_load_pretrained(adata, **kwargs):
        captured["adata"] = adata
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr("popari.wandb_util.load_pretrained", fake_load_pretrained)

    model = load_popari_model_from_wandb("run-id")

    assert model == "model"
    assert captured["adata"] is dataset
    assert captured["adata"].popari.sample_names == ("replicate",)
    assert tuple(captured["kwargs"]["reloaded_hierarchy"]) == (0,)
    assert captured["kwargs"]["reloaded_hierarchy"][0] is dataset
    assert captured["kwargs"]["hierarchical_levels"] == 1
    assert captured["kwargs"]["context"] == {"device": "cpu", "dtype": torch.float64}


def test_load_popari_results_from_wandb_reads_local_hierarchy(tmp_path, monkeypatch):
    result_path = tmp_path / "results"
    result_path.mkdir()
    _write_h5ad(result_path / "level_0.h5ad")
    api = Mock()
    api.run.return_value.config = {"result_directory": str(result_path)}
    monkeypatch.setattr("wandb.Api", Mock(return_value=api))
    download = Mock()
    monkeypatch.setattr(
        "popari.wandb_util.download_popari_results",
        download,
    )

    hierarchy = load_popari_results_from_wandb("run-id")

    assert tuple(hierarchy) == (0,)
    assert hierarchy[0].popari.sample_names == ("replicate",)
    download.assert_not_called()


def test_load_popari_results_from_wandb_falls_back_to_artifact(tmp_path, monkeypatch):
    result_path = tmp_path / "downloaded"
    result_path.mkdir()
    _write_h5ad(result_path / "level_0.h5ad")
    api = Mock()
    api.run.return_value.config = {"result_directory": str(tmp_path / "missing")}
    monkeypatch.setattr("wandb.Api", Mock(return_value=api))
    download = Mock(return_value=result_path)
    monkeypatch.setattr("popari.wandb_util.download_popari_results", download)

    hierarchy = load_popari_results_from_wandb("run-id", alias="manuscript")

    assert tuple(hierarchy) == (0,)
    download.assert_called_once_with(
        "run-id",
        entity="popari",
        project="revisions",
        alias="manuscript",
        root=None,
    )


def test_load_popari_results_from_wandb_reports_missing_sources(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing"
    api = Mock()
    api.run.return_value.config = {"result_directory": str(missing_path)}
    monkeypatch.setattr("wandb.Api", Mock(return_value=api))
    monkeypatch.setattr(
        "popari.wandb_util.download_popari_results",
        Mock(side_effect=RuntimeError("artifact missing")),
    )

    with pytest.raises(FileNotFoundError, match="local result_directory") as error:
        load_popari_results_from_wandb("run-id")

    assert str(missing_path) in str(error.value)
    assert "popari-results-run-id:latest" in str(error.value)


def test_log_popari_results_uses_requested_aliases(tmp_path, monkeypatch):
    (tmp_path / "level_0.h5ad").touch()
    monkeypatch.setattr("wandb.Artifact", _Artifact)
    run = type("Run", (), {"id": "run-id", "log_artifact": Mock()})()

    log_popari_results(run, tmp_path, aliases=("manuscript", "latest"))

    run.log_artifact.assert_called_once()
    assert run.log_artifact.call_args.kwargs["aliases"] == ["manuscript", "latest"]


def test_log_popari_results_requires_an_alias(tmp_path):
    (tmp_path / "level_0.h5ad").touch()
    run = type("Run", (), {"id": "run-id"})()

    with pytest.raises(ValueError, match="at least one"):
        log_popari_results(run, tmp_path, aliases=())
