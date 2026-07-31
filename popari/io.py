"""Strict I/O for canonical unified Popari AnnData artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_array

from popari.schema import SAMPLE_KEY_KEY, SCHEMA_VERSION, SCHEMA_VERSION_KEY, validate_anndata_hierarchy

LEVEL_FILE_PATTERN = re.compile(r"^level_(\d+)\.h5ad$")


def load_anndata(filepath: str | Path) -> AnnData:
    """Load and validate one canonical Popari H5AD artifact."""

    adata = ad.read_h5ad(filepath)
    version = adata.uns.get(SCHEMA_VERSION_KEY)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{filepath} uses Popari schema version {version!r}; expected {SCHEMA_VERSION}. "
            "Run `uv run python scripts/migrate_popari_artifact.py INPUT OUTPUT` first.",
        )
    adata.popari.validate()
    return adata


def save_anndata(
    filepath: str | Path,
    adata: AnnData,
    ignore_raw_data: bool = False,
    *,
    sample_key: str | None = None,
) -> AnnData:
    """Validate and write one unified Popari AnnData artifact."""

    if not isinstance(adata, AnnData):
        raise TypeError("save_anndata expects one unified AnnData object.")
    canonical = adata.copy()
    if sample_key is not None:
        canonical.uns[SAMPLE_KEY_KEY] = sample_key
    canonical.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    canonical.popari.validate()
    if ignore_raw_data:
        canonical.X = csr_array(canonical.shape)

    canonical.write_h5ad(filepath)
    return canonical


def _hierarchy_files(path: str | Path) -> dict[int, Path]:
    path = Path(path)
    if path.is_file():
        return {0: path}
    sibling = path.with_suffix(".h5ad")
    if sibling.is_file():
        return {0: sibling}
    if not path.is_dir():
        raise FileNotFoundError(f"No Popari AnnData artifact found at {path}.")
    files = {
        int(match.group(1)): candidate
        for candidate in path.glob("level_*.h5ad")
        if (match := LEVEL_FILE_PATTERN.match(candidate.name)) is not None
    }
    if not files:
        raise ValueError(f"No level_*.h5ad files found in {path}.")
    levels = sorted(files)
    if levels != list(range(len(levels))):
        raise ValueError(f"Hierarchy levels must be contiguous from zero; found {levels}.")
    return files


def load_anndata_hierarchy(path: str | Path) -> dict[int, AnnData]:
    """Load and validate one canonical AnnData per hierarchy level."""

    hierarchy = {level: load_anndata(file) for level, file in sorted(_hierarchy_files(path).items())}
    validate_anndata_hierarchy(hierarchy)
    return hierarchy


def save_anndata_hierarchy(
    path: str | Path,
    hierarchy: Mapping[int, AnnData],
    *,
    ignore_raw_data: bool = False,
    sample_key: str | None = None,
) -> dict[int, AnnData]:
    """Write one canonical H5AD file per hierarchy level."""

    prepared = {}
    for level, adata in hierarchy.items():
        current = adata.copy()
        if sample_key is not None:
            current.uns[SAMPLE_KEY_KEY] = sample_key
        current.uns[SCHEMA_VERSION_KEY] = SCHEMA_VERSION
        prepared[level] = current
    validate_anndata_hierarchy(prepared)

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return {
        level: save_anndata(path / f"level_{level}.h5ad", prepared[level], ignore_raw_data=ignore_raw_data)
        for level in sorted(prepared)
    }


__all__ = [
    load_anndata.__name__,
    load_anndata_hierarchy.__name__,
    save_anndata.__name__,
    save_anndata_hierarchy.__name__,
]
