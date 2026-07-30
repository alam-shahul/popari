"""Preprocessing helpers for observation annotations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import anndata as ad
import pandas as pd


def relabel_categories(
    dataset: ad.AnnData,
    *,
    source: str,
    target: str,
    mapping: Mapping,
    unmapped: Literal["keep", "error"] = "keep",
) -> None:
    """Map observation categories in a unified AnnData.

    Args:
        dataset: AnnData object whose observations will be relabeled.
        source: Observation column containing the original labels.
        target: Observation column in which to store mapped labels.
        mapping: Mapping from original labels to replacement labels.
        unmapped: Whether to preserve unknown labels or raise an error.

    Raises:
        KeyError: If ``source`` is absent from ``dataset``.
        ValueError: If ``unmapped`` is invalid or unknown labels are found
            when ``unmapped="error"``.

    """

    if unmapped not in {"keep", "error"}:
        raise ValueError("unmapped must be 'keep' or 'error'.")

    if source not in dataset.obs:
        raise KeyError(f"Observation key {source!r} is missing from dataset.obs.")

    if unmapped == "error":
        unknown_labels = {label for label in dataset.obs[source].dropna().unique() if label not in mapping}
        if unknown_labels:
            raise ValueError(f"Unmapped labels found: {sorted(unknown_labels, key=str)}.")

    labels = dataset.obs[source]
    dataset.obs[target] = pd.Categorical(
        labels.map(lambda label: mapping.get(label, label)),
    )
