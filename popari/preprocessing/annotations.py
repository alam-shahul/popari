"""Preprocessing helpers for observation annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import anndata as ad
import pandas as pd

from popari._datasets import as_datasets


def relabel_categories(
    data: ad.AnnData | Sequence[ad.AnnData],
    *,
    source: str,
    target: str,
    mapping: Mapping,
    unmapped: Literal["keep", "error"] = "keep",
) -> None:
    """Map observation categories across one or more datasets.

    Args:
        data: One AnnData object or a sequence of datasets.
        source: Observation column containing the original labels.
        target: Observation column in which to store mapped labels.
        mapping: Mapping from original labels to replacement labels.
        unmapped: Whether to preserve unknown labels or raise an error.

    Raises:
        KeyError: If ``source`` is absent from a dataset.
        ValueError: If ``unmapped`` is invalid or unknown labels are found
            when ``unmapped="error"``.

    """

    if unmapped not in {"keep", "error"}:
        raise ValueError("unmapped must be 'keep' or 'error'.")

    datasets = as_datasets(data)
    for dataset in datasets:
        if source not in dataset.obs:
            raise KeyError(f"Observation key {source!r} is missing from dataset.obs.")

    if unmapped == "error":
        unknown_labels = {
            label for dataset in datasets for label in dataset.obs[source].dropna().unique() if label not in mapping
        }
        if unknown_labels:
            raise ValueError(f"Unmapped labels found: {sorted(unknown_labels, key=str)}.")

    for dataset in datasets:
        labels = dataset.obs[source]
        dataset.obs[target] = pd.Categorical(
            labels.map(lambda label: mapping.get(label, label)),
        )
