"""Internal helpers for selecting sample facets from unified AnnData objects."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad

from popari._sample_axis import SampleAxis


def resolve_samples(
    adata: ad.AnnData,
    *,
    samples: str | Sequence[str] | None = None,
    sample_key: str | None = None,
    require_graph: bool = False,
) -> tuple[SampleAxis, tuple[str, ...]]:
    """Return the sample axis and validated sample names to plot."""

    resolved_sample_key = sample_key or adata.popari.sample_key
    sample_axis = SampleAxis.from_anndata(
        adata,
        sample_key=resolved_sample_key,
    )
    if require_graph:
        adata.popari.validate_spatial_graph()

    if samples is None:
        selected = sample_axis.names
    elif isinstance(samples, str):
        selected = (samples,)
    else:
        selected = tuple(str(sample) for sample in samples)

    if not selected:
        raise ValueError("samples must select at least one sample.")
    for sample in selected:
        sample_axis.position(sample)
    if len(set(selected)) != len(selected):
        raise ValueError("samples must not contain duplicates.")
    return sample_axis, selected
