"""Shared helpers for operations over one or more AnnData objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial, wraps

import anndata as ad
import numpy as np

from popari.util import concatenate, unconcatenate


def as_datasets(data: ad.AnnData | Sequence[ad.AnnData]) -> tuple[ad.AnnData, ...]:
    """Normalize one AnnData object or a sequence into a nonempty tuple."""

    if isinstance(data, ad.AnnData):
        return (data,)

    datasets = tuple(data)
    if not datasets:
        raise ValueError("At least one AnnData object is required.")
    if not all(isinstance(dataset, ad.AnnData) for dataset in datasets):
        raise TypeError("Expected an AnnData object or a sequence of AnnData objects.")
    return datasets


def copy_annotations(
    original_dataset: ad.AnnData,
    updated_dataset: ad.AnnData,
    annotations: Mapping[str, Sequence[str] | None] | None = None,
) -> None:
    """Copy selected AnnData annotation keys from an updated dataset."""

    for annotation, keys in (annotations or {}).items():
        original_annotation = getattr(original_dataset, annotation)
        updated_annotation = getattr(updated_dataset, annotation)
        for key in updated_annotation.keys() if keys is None else keys:
            original_annotation[key] = updated_annotation[key]


def enable_joint(function=None, *, annotations=None):
    """Allow a dataset-sequence operation to run on a merged dataset."""

    if function is None:
        return partial(enable_joint, annotations=annotations)

    @wraps(function)
    def joint_wrapper(data, *args, **kwargs):
        datasets = as_datasets(data)
        joint = kwargs.pop("joint", False)
        if not joint:
            return function(datasets, *args, **kwargs)

        merged_dataset = concatenate(datasets)
        outputs = function((merged_dataset,), *args, **kwargs)
        for updated_dataset, original_dataset in zip(unconcatenate(merged_dataset), datasets):
            copy_annotations(original_dataset, updated_dataset, annotations=annotations)
        if outputs is None:
            return None
        return datasets, outputs

    return joint_wrapper


def broadcast(function: Callable):
    """Apply a single-AnnData operation to one or more datasets."""

    @wraps(function)
    def broadcast_wrapper(data, *args, **kwargs):
        outputs = [function(dataset, *args, **kwargs) for dataset in as_datasets(data)]
        if all(output is None for output in outputs):
            return None
        return outputs[0] if len(outputs) == 1 else outputs

    return broadcast_wrapper


def broadcast_plottable(function: Callable):
    """Apply a plotting operation across datasets and allocated axes."""

    @wraps(function)
    def broadcast_wrapper(data, *args, **kwargs):
        from popari.plotting.utils import setup_squarish_axes

        datasets = as_datasets(data)
        axes = kwargs.pop("axes", None)
        sharex = kwargs.pop("sharex", False)
        sharey = kwargs.pop("sharey", False)
        fig = None
        if axes is None:
            fig, axes = setup_squarish_axes(len(datasets), sharex=sharex, sharey=sharey)

        axes = np.asarray(axes, dtype=object)
        outputs = [function(dataset, *args, ax=ax, **kwargs) for dataset, ax in zip(datasets, axes.flat)]
        return fig, outputs[0] if len(outputs) == 1 else outputs

    return broadcast_wrapper
