"""Analysis helpers for aggregating sample-specific matrices."""

from collections.abc import Mapping, Sequence
from typing import Literal

import anndata as ad
import numpy as np


def aggregate_sample_matrices(
    dataset: ad.AnnData,
    sample_groups: Mapping[str, Sequence[str]],
    key: str,
    *,
    reduction: Literal["mean", "median"] = "mean",
) -> dict[str, np.ndarray]:
    """Aggregate sample-specific matrices over named groups.

    Args:
        dataset: AnnData object containing sample-specific matrices.
        sample_groups: Mapping from output group names to sample identifiers.
        key: Key of the sample-keyed matrix mapping in ``dataset.uns``.
        reduction: Aggregation to apply across samples in each group.

    Returns:
        Aggregated matrices keyed by group name.

    Raises:
        KeyError: If the matrix mapping or a requested sample is missing.
        TypeError: If ``dataset.uns[key]`` is not a mapping.
        ValueError: If a group is empty, matrix shapes differ, or the reduction
            is unsupported.

    """

    if key not in dataset.uns:
        raise KeyError(f"Matrix key {key!r} is missing from dataset.uns.")

    sample_matrices = dataset.uns[key]
    if not isinstance(sample_matrices, Mapping):
        raise TypeError(f"dataset.uns[{key!r}] must be a sample-keyed mapping.")
    if reduction not in {"mean", "median"}:
        raise ValueError("reduction must be 'mean' or 'median'.")

    aggregated = {}
    for group_name, group_samples in sample_groups.items():
        sample_names = list(dict.fromkeys(str(sample) for sample in group_samples))
        if not sample_names:
            raise ValueError(f"Sample group {group_name!r} is empty.")

        missing_samples = [name for name in sample_names if name not in sample_matrices]
        if missing_samples:
            raise KeyError(
                f"dataset.uns[{key!r}] is missing matrices for samples: {missing_samples}.",
            )

        matrices = [np.asarray(sample_matrices[name]) for name in sample_names]
        shapes = {matrix.shape for matrix in matrices}
        if len(shapes) != 1:
            raise ValueError(
                f"Sample matrices in group {group_name!r} have inconsistent shapes: {sorted(shapes)}.",
            )

        stacked = np.stack(matrices)
        aggregated[str(group_name)] = getattr(np, reduction)(stacked, axis=0)

    return aggregated
