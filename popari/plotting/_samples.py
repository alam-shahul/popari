"""Internal helpers for selecting sample facets from unified AnnData objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np

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
    adjacency_key = "adjacency_matrix" if require_graph else None
    sample_axis = SampleAxis.from_anndata(
        adata,
        sample_key=resolved_sample_key,
        adjacency_key=adjacency_key,
    )

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


def sample_view(adata: ad.AnnData, sample_axis: SampleAxis, sample: str) -> ad.AnnData:
    """Return an observation view for one named sample."""

    return adata[sample_axis.indices(sample)]


def resolve_sample_matrix(
    adata: ad.AnnData,
    key: str,
    *,
    sample: str | None = None,
) -> np.ndarray:
    """Resolve a sample-keyed matrix, accepting an omitted sample when
    shared."""

    value = adata.uns[key]
    if not isinstance(value, Mapping):
        return np.asarray(value)
    if sample is not None:
        try:
            return np.asarray(value[str(sample)])
        except KeyError as error:
            raise KeyError(f"`uns[{key!r}]` has no matrix for sample {sample!r}.") from error
    if not value:
        raise ValueError(f"`uns[{key!r}]` contains no sample matrices.")

    matrices = [np.asarray(matrix) for matrix in value.values()]
    first = matrices[0]
    if any(matrix.shape != first.shape or not np.allclose(matrix, first) for matrix in matrices[1:]):
        raise ValueError(f"`uns[{key!r}]` varies by sample; specify sample=.")
    return first
