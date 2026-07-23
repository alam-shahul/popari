from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.repair_inspire_artifacts import repair_inspire_artifact, resolve_source_path


def test_repair_inspire_artifact_restores_names_and_ground_truth(tmp_path):
    source_path = tmp_path / "source.h5ad"
    artifact_path = tmp_path / "result.h5ad"
    output_path = tmp_path / "repaired" / "result.h5ad"

    source = ad.AnnData(X=np.ones((4, 2)))
    source.obs["batch"] = pd.Categorical(
        ["progenitor", "progenitor", "layer", "layer"],
        categories=["progenitor", "layer"],
    )
    source.uns["ground_truth_M"] = {
        "progenitor": np.array([[1.0], [2.0]]),
        "layer": np.array([[3.0], [4.0]]),
    }
    source.write_h5ad(source_path)

    result = ad.AnnData(X=np.ones((4, 2)))
    result.obs_names = ["0-1", "1-1", "0-0", "1-0"]
    result.obs["batch"] = pd.Categorical(["0", "0", "1", "1"])
    result.obsm["ground_truth_X"] = np.ones((4, 1))
    result.uns["M"] = {
        "progenitor": np.array([[5.0], [6.0]]),
        "layer": np.array([[5.0], [6.0]]),
    }
    result.write_h5ad(artifact_path)

    mapping = repair_inspire_artifact(source_path, artifact_path, output_path)
    repaired = ad.read_h5ad(output_path)

    assert mapping == {"0": "layer", "1": "progenitor"}
    assert repaired.obs["batch"].astype(str).tolist() == ["layer", "layer", "progenitor", "progenitor"]
    assert set(repaired.uns["ground_truth_M"]) == {"layer", "progenitor"}
    np.testing.assert_allclose(repaired.uns["ground_truth_M"]["layer"], [[3.0], [4.0]])
    assert ad.read_h5ad(artifact_path).obs["batch"].astype(str).tolist() == ["0", "0", "1", "1"]


def test_resolve_source_path_relative_to_mlruns_parent(tmp_path):
    mlruns_path = tmp_path / "experiment" / "mlruns"
    source_path = tmp_path / "data" / "source.h5ad"
    mlruns_path.mkdir(parents=True)
    source_path.parent.mkdir()
    source_path.touch()

    resolved = resolve_source_path("../data/source.h5ad", mlruns_path)

    assert resolved == source_path


def test_repair_inspire_artifact_rejects_ambiguous_slice_suffixes(tmp_path):
    source_path = tmp_path / "source.h5ad"
    artifact_path = tmp_path / "result.h5ad"

    source = ad.AnnData(X=np.ones((2, 1)))
    source.obs["batch"] = pd.Categorical(["a", "b"])
    source.uns["ground_truth_M"] = {"a": np.ones((1, 1)), "b": np.ones((1, 1))}
    source.write_h5ad(source_path)

    result = ad.AnnData(X=np.ones((2, 1)))
    result.obs_names = ["cell-0", "cell-1"]
    result.obs["batch"] = pd.Categorical(["0", "0"])
    result.obsm["ground_truth_X"] = np.ones((2, 1))
    result.uns["M"] = {"a": np.ones((1, 1)), "b": np.ones((1, 1))}
    result.write_h5ad(artifact_path)

    with pytest.raises(ValueError, match="one INSPIRE slice index"):
        repair_inspire_artifact(source_path, artifact_path, tmp_path / "repaired.h5ad")
