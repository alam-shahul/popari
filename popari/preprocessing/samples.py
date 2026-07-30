"""Preprocessing helpers for selecting groups of spatial samples."""

from collections.abc import Mapping, Sequence

import anndata as ad
import pandas as pd

from popari.schema import SAMPLE_KEY_KEY


def subset_samples(
    dataset: ad.AnnData,
    sample_groups: Mapping[str, Sequence[str]],
    *,
    sample_key: str = "batch",
    sample_parameter_keys: Sequence[str] = ("M", "Sigma_x_inv", "Sigma_x_inv_bar"),
) -> dict[str, ad.AnnData]:
    """Create named AnnData subsets from groups of sample identifiers.

    Args:
        dataset: Multisample AnnData object to subset.
        sample_groups: Mapping from output group names to sample identifiers.
        sample_key: Observation column containing sample identifiers.
        sample_parameter_keys: Keys in ``dataset.uns`` whose sample-keyed
            mappings should be restricted to each subset.

    Returns:
        Copied AnnData subsets keyed by group name.

    Raises:
        KeyError: If ``sample_key`` is absent from ``dataset.obs``.
        ValueError: If a group is empty or contains an unknown sample.
        TypeError: If a requested parameter is not stored as a mapping.

    """

    if sample_key not in dataset.obs:
        raise KeyError(f"Sample key {sample_key!r} is missing from dataset.obs.")

    sample_labels = dataset.obs[sample_key].astype(str)
    available_samples = set(sample_labels.unique())
    subsets = {}

    for group_name, group_samples in sample_groups.items():
        sample_names = list(dict.fromkeys(str(sample) for sample in group_samples))
        if not sample_names:
            raise ValueError(f"Sample group {group_name!r} is empty.")

        unknown_samples = sorted(set(sample_names) - available_samples)
        if unknown_samples:
            raise ValueError(
                f"Sample group {group_name!r} contains unknown samples: {unknown_samples}.",
            )

        subset = dataset[sample_labels.isin(sample_names).to_numpy()].copy()
        subset.obs[sample_key] = pd.Categorical(
            subset.obs[sample_key].astype(str),
            categories=sample_names,
        )
        subset.uns[SAMPLE_KEY_KEY] = sample_key
        subset.popari.name = str(group_name)

        for key in sample_parameter_keys:
            if key not in dataset.uns:
                continue
            parameter = dataset.uns[key]
            if not isinstance(parameter, Mapping):
                raise TypeError(f"dataset.uns[{key!r}] must be a sample-keyed mapping.")
            subset.uns[key] = {name: parameter[name] for name in sample_names if name in parameter}

        subsets[str(group_name)] = subset

    return subsets
