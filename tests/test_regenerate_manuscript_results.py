from types import SimpleNamespace
from unittest.mock import Mock

import torch

from scripts import regenerate_manuscript_results as regeneration


class WandbRun:
    id = "run-id"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def configure_regeneration(monkeypatch, tmp_path, *, hierarchical_levels):
    source_artifact = SimpleNamespace(name="popari-model-run-id-level_0:v0", type="popari-model", aliases=["latest"])
    api_run = SimpleNamespace(logged_artifacts=lambda: [source_artifact])
    wandb = SimpleNamespace(
        Api=lambda: SimpleNamespace(run=lambda path: api_run),
        init=lambda **kwargs: WandbRun(),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", wandb)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = SimpleNamespace(
        hierarchical_levels=hierarchical_levels,
        materialize_results=Mock(return_value={level: object() for level in range(hierarchical_levels)}),
    )
    source_hierarchy = {level: object() for level in range(hierarchical_levels)}
    load_historical = Mock(return_value=source_hierarchy)
    load_pretrained = Mock(return_value=model)
    monkeypatch.setattr(regeneration, "load_popari_anndata_from_wandb", load_historical)
    monkeypatch.setattr(regeneration, "load_pretrained", load_pretrained)

    trainer = SimpleNamespace(superresolve=Mock())
    monkeypatch.setattr(regeneration, "Trainer", Mock(return_value=trainer))

    def save_hierarchy(path, hierarchy):
        path.mkdir(parents=True)
        for level in hierarchy:
            (path / f"level_{level}.h5ad").touch()

    monkeypatch.setattr(regeneration, "save_anndata_hierarchy", save_hierarchy)
    monkeypatch.setattr(
        regeneration,
        "load_anndata_hierarchy",
        lambda path: {level: object() for level in range(hierarchical_levels)},
    )
    log_results = Mock()
    monkeypatch.setattr(regeneration, "log_popari_results", log_results)
    return model, trainer, log_results, load_historical, load_pretrained


def test_flat_regeneration_materializes_without_superresolution(tmp_path, monkeypatch):
    model, trainer, log_results, load_historical, load_pretrained = configure_regeneration(
        monkeypatch,
        tmp_path,
        hierarchical_levels=1,
    )

    output = regeneration.regenerate_results("run-id", output_root=tmp_path)

    assert output == tmp_path / "run-id"
    assert (output / "level_0.h5ad").exists()
    trainer.superresolve.assert_not_called()
    model.materialize_results.assert_called_once_with()
    log_results.assert_called_once()
    load_historical.assert_called_once_with(
        "run-id",
        entity="popari",
        project="revisions",
        artifact_stem=None,
        legacy=True,
    )
    source_adata = load_historical.return_value[0]
    load_pretrained.assert_called_once_with(
        source_adata,
        reloaded_hierarchy=load_historical.return_value,
        hierarchical_levels=1,
        context={"device": "cuda:0", "dtype": torch.float64},
    )


def test_hierarchical_regeneration_uses_standard_schedule(tmp_path, monkeypatch):
    _, trainer, _, _, _ = configure_regeneration(monkeypatch, tmp_path, hierarchical_levels=2)

    regeneration.regenerate_results("run-id", output_root=tmp_path, upload=False)

    trainer.superresolve.assert_called_once_with()
