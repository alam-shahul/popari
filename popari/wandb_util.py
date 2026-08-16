"""Utilities for loading migrated Popari results from W&B artifacts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import anndata as ad
import torch

from popari.io import load_anndata
from popari.legacy_io import convert_legacy_anndata_hierarchy
from popari.model import load_pretrained
from popari.schema import validate_anndata_hierarchy

LEVEL_FILE_PATTERN = re.compile(r"^level_(\d+)\.h5ad$")


def log_popari_results(
    run,
    path: str | Path,
    *,
    metadata: dict | None = None,
    aliases: Sequence[str] = ("latest",),
) -> None:
    """Log a canonical AnnData hierarchy as a W&B result artifact."""

    import wandb

    result_path = Path(path)
    level_files = sorted(result_path.glob("level_*.h5ad"))
    if not level_files:
        raise ValueError(f"No level_*.h5ad files found in {result_path}.")
    aliases = tuple(aliases)
    if not aliases:
        raise ValueError("aliases must contain at least one artifact alias.")

    artifact = wandb.Artifact(
        f"popari-results-{run.id}",
        type="popari-results",
        metadata=metadata or {},
    )
    for level_file in level_files:
        artifact.add_file(str(level_file), name=level_file.name)
    run.log_artifact(artifact, aliases=list(aliases))


def download_popari_results(
    run_id: str,
    *,
    entity: str = "popari",
    project: str = "revisions",
    alias: str = "latest",
    root: str | Path | None = None,
) -> Path:
    """Download the canonical materialized results for one W&B run."""

    import wandb

    artifact = wandb.Api().artifact(
        f"{entity}/{project}/popari-results-{run_id}:{alias}",
        type="popari-results",
    )
    return Path(artifact.download(root=str(root) if root is not None else None))


def load_popari_results_from_wandb(
    run_id: str,
    *,
    entity: str = "popari",
    project: str = "revisions",
    alias: str = "latest",
    root: str | Path | None = None,
) -> dict[int, ad.AnnData]:
    """Load local results registered by W&B, falling back to an artifact."""

    import wandb

    run_path = f"{entity}/{project}/{run_id}"
    run = wandb.Api().run(run_path)
    result_directory = run.config.get("result_directory")
    if result_directory is not None:
        local_path = Path(result_directory)
        if local_path.exists():
            return read_popari_anndata_hierarchy(local_path)

    artifact_name = f"{entity}/{project}/popari-results-{run_id}:{alias}"
    try:
        result_path = download_popari_results(
            run_id,
            entity=entity,
            project=project,
            alias=alias,
            root=root,
        )
    except Exception as error:
        local_description = repr(result_directory) if result_directory is not None else "not configured"
        raise FileNotFoundError(
            f"W&B run {run_path} has no available Popari results: local result_directory "
            f"is {local_description}, and artifact {artifact_name!r} could not be downloaded.",
        ) from error
    return read_popari_anndata_hierarchy(result_path)


def download_popari_artifact(
    run_id: str,
    *,
    entity: str = "popari",
    project: str = "revisions",
    artifact_stem: str | None = "output",
    alias: str = "latest",
    root: str | Path | None = None,
) -> Path:
    """Download a migrated Popari artifact and return its local path."""

    import wandb

    api = wandb.Api()
    if artifact_stem is None:
        run = api.run(f"{entity}/{project}/{run_id}")
        artifacts = [
            artifact
            for artifact in run.logged_artifacts()
            if (
                artifact.type == "popari-model"
                and alias in artifact.aliases
                and any(path.endswith(".h5ad") for path in artifact.manifest.entries)
            )
        ]
        if len(artifacts) != 1:
            raise ValueError(
                f"Expected one latest popari-model artifact containing a .h5ad file for run {run_id}; "
                f"found {len(artifacts)}.",
            )
        artifact = artifacts[0]
    else:
        artifact_path = f"{entity}/{project}/popari-model-{run_id}-{artifact_stem}:{alias}"
        artifact = api.artifact(artifact_path, type="popari-model")

    download_path = artifact.download(root=str(root) if root is not None else None)
    return Path(download_path)


def _download_popari_artifacts(
    run_id: str,
    *,
    entity: str = "popari",
    project: str = "revisions",
    artifact_stem: str | None = "output",
    alias: str = "latest",
    root: str | Path | None = None,
) -> list[Path]:
    """Download one or more migrated Popari artifacts and return local paths."""

    if artifact_stem is not None:
        return [
            download_popari_artifact(
                run_id,
                entity=entity,
                project=project,
                artifact_stem=artifact_stem,
                alias=alias,
                root=root,
            ),
        ]

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    artifacts = [
        artifact
        for artifact in run.logged_artifacts()
        if (
            artifact.type == "popari-model"
            and alias in artifact.aliases
            and any(path.endswith(".h5ad") for path in artifact.manifest.entries)
        )
    ]
    if not artifacts:
        raise ValueError(
            f"Expected at least one latest popari-model artifact containing a .h5ad file for run {run_id}.",
        )

    return [Path(artifact.download(root=str(root) if root is not None else None)) for artifact in artifacts]


def _h5ad_files_from_artifact_path(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file() and path.suffix == ".h5ad":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.h5ad"))
    return []


def read_popari_anndata_hierarchy(paths: str | Path | list[str | Path]) -> dict[int, ad.AnnData]:
    """Read flat or hierarchical Popari results as ``level -> AnnData``."""

    if isinstance(paths, (str, Path)):
        paths = [paths]

    h5ad_files = []
    for path in paths:
        h5ad_files.extend(_h5ad_files_from_artifact_path(path))

    if not h5ad_files:
        raise ValueError("Expected at least one .h5ad file in downloaded Popari artifacts.")

    level_files = {}
    flat_files = []
    for h5ad_file in h5ad_files:
        match = LEVEL_FILE_PATTERN.match(h5ad_file.name)
        if match is None:
            flat_files.append(h5ad_file)
        else:
            level = int(match.group(1))
            if level in level_files:
                raise ValueError(f"Found multiple .h5ad files for hierarchical level {level}.")
            level_files[level] = h5ad_file

    if level_files and flat_files:
        raise ValueError("Found both hierarchical level_*.h5ad files and flat .h5ad result files.")
    if flat_files and len(flat_files) != 1:
        raise ValueError(f"Expected one flat .h5ad result file; found {len(flat_files)}.")

    files_by_level = level_files or {0: flat_files[0]}
    hierarchy = {level: load_anndata(h5ad_file) for level, h5ad_file in sorted(files_by_level.items())}
    validate_anndata_hierarchy(hierarchy)
    return hierarchy


def read_legacy_popari_anndata_hierarchy(paths: str | Path | list[str | Path]) -> dict[int, ad.AnnData]:
    """Read and convert historical Popari H5AD artifacts in memory."""

    if isinstance(paths, (str, Path)):
        paths = [paths]
    h5ad_files = []
    for path in paths:
        h5ad_files.extend(_h5ad_files_from_artifact_path(path))
    if not h5ad_files:
        raise ValueError("Expected at least one .h5ad file in downloaded Popari artifacts.")

    level_files = {}
    flat_files = []
    for h5ad_file in h5ad_files:
        match = LEVEL_FILE_PATTERN.match(h5ad_file.name)
        if match is None:
            flat_files.append(h5ad_file)
        else:
            level_files[int(match.group(1))] = h5ad_file
    if level_files and flat_files:
        raise ValueError("Found both hierarchical level_*.h5ad files and flat .h5ad result files.")
    if flat_files and len(flat_files) != 1:
        raise ValueError(f"Expected one flat .h5ad result file; found {len(flat_files)}.")

    files_by_level = level_files or {0: flat_files[0]}
    raw = {level: ad.read_h5ad(path) for level, path in sorted(files_by_level.items())}
    hierarchy = convert_legacy_anndata_hierarchy(raw)
    validate_anndata_hierarchy(hierarchy)
    return hierarchy


def load_popari_anndata_from_wandb(
    run_id: str,
    *,
    entity: str = "popari",
    project: str = "revisions",
    artifact_stem: str | None = "output",
    alias: str = "latest",
    root: str | Path | None = None,
    legacy: bool = False,
) -> dict[int, ad.AnnData]:
    """Load migrated Popari results as ``level -> AnnData``."""

    artifact_paths = _download_popari_artifacts(
        run_id,
        entity=entity,
        project=project,
        artifact_stem=artifact_stem,
        alias=alias,
        root=root,
    )
    if legacy:
        return read_legacy_popari_anndata_hierarchy(artifact_paths)
    return read_popari_anndata_hierarchy(artifact_paths)


def load_popari_model_from_wandb(
    run_id: str,
    *,
    context: dict | None = None,
    entity: str = "popari",
    project: str = "revisions",
    alias: str = "latest",
    root: str | Path | None = None,
    **popari_kwargs,
):
    """Reconstruct a trained Popari model from canonical W&B results."""

    if context is None:
        context = {"device": "cpu", "dtype": torch.float64}

    reloaded_hierarchy = load_popari_results_from_wandb(
        run_id,
        entity=entity,
        project=project,
        alias=alias,
        root=root,
    )
    popari_kwargs.setdefault("hierarchical_levels", len(reloaded_hierarchy))

    return load_pretrained(
        reloaded_hierarchy[0],
        reloaded_hierarchy=reloaded_hierarchy,
        context=context,
        **popari_kwargs,
    )
