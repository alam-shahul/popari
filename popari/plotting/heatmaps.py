"""Heatmap plotting primitives for matrices and AnnData results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, colormaps, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator

from popari.plotting._samples import resolve_samples
from popari.plotting.utils import setup_squarish_axes


def _matrix_data_and_labels(matrix, xticklabels=None, yticklabels=None):
    """Extract array values and axis labels from a matrix-like object."""

    if isinstance(matrix, pd.DataFrame):
        if xticklabels is None:
            xticklabels = matrix.columns
        if yticklabels is None:
            yticklabels = matrix.index
        matrix = matrix.to_numpy()

    return np.asarray(matrix), xticklabels, yticklabels


def _matrix_absmax(matrices):
    """Return a nonzero absolute maximum across a collection of matrices."""

    max_value = 0.0
    for matrix in matrices:
        values, _, _ = _matrix_data_and_labels(matrix)
        if np.ma.isMaskedArray(values):
            values = values.compressed()
        if values.size:
            max_value = max(max_value, float(np.nanmax(np.abs(values))))

    return max_value if max_value != 0 else 1.0


def matrix_heatmap(
    matrix,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "bwr",
    center_zero: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    label_values: bool = False,
    label_font_size: float | None = None,
    cell_grid: bool = True,
    cell_grid_color: str = "gray",
    cell_grid_width: float = 0.5,
    mask: np.ndarray | None = None,
    xticklabels=None,
    yticklabels=None,
    render_mode: str = "mesh",
    return_image: bool = False,
    **imshow_kwargs,
):
    """Plot a standalone matrix-like object as a heatmap.

    ``render_mode="image"`` uses ``imshow``. ``render_mode="mesh"`` uses
    ``pcolormesh``, allowing cell borders and fills to share exact geometry.
    ``render_mode="rectangles"`` draws each matrix entry as a vector rectangle,
    which exports more reliably to PDF/Illustrator for small publication
    heatmaps. Set ``cell_grid=True`` to draw boundaries around matrix cells.

    """

    values, xticklabels, yticklabels = _matrix_data_and_labels(matrix, xticklabels, yticklabels)
    if mask is not None:
        values = np.ma.masked_where(mask, values)

    if center_zero:
        max_value = _matrix_absmax([values])
        imshow_kwargs.setdefault("vmin", -max_value)
        imshow_kwargs.setdefault("vmax", max_value)

    dpi = imshow_kwargs.pop("dpi", 100)
    figsize = imshow_kwargs.pop("figsize", None)
    aspect = imshow_kwargs.pop("aspect", "equal")
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    else:
        fig = ax.get_figure()

    if render_mode == "image":
        image = ax.imshow(values, cmap=cmap, interpolation="nearest", aspect=aspect, **imshow_kwargs)
    elif render_mode == "mesh":
        num_rows, num_columns = values.shape
        image = ax.pcolormesh(
            np.arange(num_columns + 1) - 0.5,
            np.arange(num_rows + 1) - 0.5,
            values,
            cmap=cmap,
            shading="flat",
            edgecolors=cell_grid_color if cell_grid else "none",
            linewidth=cell_grid_width if cell_grid else 0,
            **imshow_kwargs,
        )
        ax.set_xlim(-0.5, num_columns - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)
        ax.set_aspect(aspect)
    elif render_mode == "rectangles":
        norm = imshow_kwargs.pop("norm", None)
        vmin = imshow_kwargs.pop("vmin", None)
        vmax = imshow_kwargs.pop("vmax", None)
        alpha = imshow_kwargs.pop("alpha", None)
        if imshow_kwargs:
            unexpected = ", ".join(sorted(imshow_kwargs))
            raise TypeError(f"Unsupported rectangle heatmap arguments: {unexpected}")

        masked_values = np.ma.asarray(values)
        visible_values = masked_values.compressed()
        visible_values = visible_values[np.isfinite(visible_values)]
        if norm is None:
            if vmin is None:
                vmin = float(np.min(visible_values)) if visible_values.size else 0.0
            if vmax is None:
                vmax = float(np.max(visible_values)) if visible_values.size else 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)

        cmap_object = colormaps[cmap] if isinstance(cmap, str) else cmap
        mask_array = np.ma.getmaskarray(masked_values)
        num_rows, num_columns = masked_values.shape
        for row in range(num_rows):
            for column in range(num_columns):
                value = masked_values[row, column]
                if mask_array[row, column] or not np.isfinite(value):
                    continue
                rectangle = patches.Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cmap_object(norm(float(value))),
                    edgecolor=cell_grid_color if cell_grid else "none",
                    linewidth=cell_grid_width if cell_grid else 0,
                    alpha=alpha,
                )
                ax.add_patch(rectangle)

        ax.set_xlim(-0.5, num_columns - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)
        ax.set_aspect(aspect)
        image = cm.ScalarMappable(norm=norm, cmap=cmap_object)
        image.set_array(visible_values)
    else:
        raise ValueError("render_mode must be 'image', 'mesh', or 'rectangles'.")
    if title is not None:
        ax.set_title(title)

    num_rows, num_columns = values.shape
    if xticklabels is None:
        xticklabels = np.arange(num_columns)
    if yticklabels is None:
        yticklabels = np.arange(num_rows)
    ax.set_xticks(np.arange(num_columns), xticklabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(num_rows), yticklabels)
    ax.grid(False)
    if cell_grid and render_mode == "image":
        ax.set_xticks(np.arange(num_columns + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(num_rows + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=cell_grid_color, linewidth=cell_grid_width)
        ax.tick_params(which="minor", bottom=False, left=False)

    if label_values:
        font_size = 8 if label_font_size is None else label_font_size
        for (row, column), value in np.ndenumerate(np.asarray(values)):
            if mask is None or not mask[row, column]:
                ax.text(column, row, f"{value:.2g}", ha="center", va="center", fontsize=font_size)

    if colorbar:
        cbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    if return_image:
        return fig, image
    return fig


def category_marker_heatmap(
    scores: pd.DataFrame,
    *,
    ax: Axes | None = None,
    center_zero: bool = True,
    **heatmap_kwargs,
):
    """Plot a marker-by-category score matrix.

    Args:
        scores: Matrix returned by
            :func:`popari.tl.compute_category_marker_scores`.
        ax: Existing axis on which to draw.
        center_zero: Whether to use symmetric color limits around zero.
        **heatmap_kwargs: Additional arguments for
            :func:`matrix_heatmap`.

    Returns:
        Figure containing the heatmap.

    """

    expected_index_names = ["marker_category", "gene"]
    if not isinstance(scores.index, pd.MultiIndex) or list(scores.index.names) != expected_index_names:
        raise ValueError(f"scores must have index levels {expected_index_names}.")
    if scores.columns.nlevels not in (1, 2):
        raise ValueError("scores must have category columns or (category, dataset) columns.")

    heatmap_kwargs.setdefault("cmap", "bwr")
    heatmap_kwargs.setdefault("colorbar", True)
    heatmap_kwargs.setdefault("cell_grid", False)
    heatmap_kwargs.setdefault("aspect", scores.shape[1] / scores.shape[0])
    figure = matrix_heatmap(
        scores,
        ax=ax,
        center_zero=center_zero,
        xticklabels=[""] * scores.shape[1],
        yticklabels=[""] * scores.shape[0],
        **heatmap_kwargs,
    )
    ax = figure.axes[0] if ax is None else ax

    marker_categories = scores.index.get_level_values("marker_category")
    ordered_marker_categories = list(dict.fromkeys(marker_categories))
    marker_counts = np.array([(marker_categories == category).sum() for category in ordered_marker_categories])
    marker_bounds = np.insert(marker_counts, 0, 0).cumsum() - 0.5
    marker_centers = (marker_bounds[:-1] + marker_bounds[1:]) / 2
    marker_labels = [
        f"{category} - {', '.join(scores.loc[category].index.astype(str))}" for category in ordered_marker_categories
    ]
    ax.set_yticks(marker_centers, marker_labels)
    ax.yaxis.set_minor_locator(FixedLocator(marker_bounds))
    ax.tick_params(axis="y", which="major", length=0)
    ax.tick_params(axis="y", which="minor", length=15)

    categories = scores.columns.get_level_values(0)
    ordered_categories = list(dict.fromkeys(categories))
    category_widths = np.array([(categories == category).sum() for category in ordered_categories])
    category_bounds = np.insert(category_widths, 0, 0).cumsum() - 0.5
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(FixedLocator(category_bounds))
    return figure


def matrix_heatmap_panel(
    matrices,
    *,
    axes: Sequence[Axes] | None = None,
    titles: Sequence[str] | None = None,
    shared_scale: bool = True,
    center_zero: bool = False,
    colorbar: str | bool = "shared",
    colorbar_label: str | None = None,
    figsize=None,
    dpi: int = 100,
    sharex: bool = True,
    sharey: bool = True,
    **heatmap_kwargs,
):
    """Plot a collection of matrix-like objects as a heatmap panel."""

    if isinstance(matrices, Mapping):
        titles = list(matrices) if titles is None else titles
        matrices = list(matrices.values())
    else:
        matrices = list(matrices)

    if axes is None:
        subplots_kwargs = {"dpi": dpi, "sharex": sharex, "sharey": sharey}
        if figsize is not None:
            subplots_kwargs["figsize"] = figsize
        fig, axes = setup_squarish_axes(len(matrices), **subplots_kwargs)
    else:
        axes = np.asarray(axes)
        fig = axes.flat[0].get_figure()

    if titles is None:
        titles = [None] * len(matrices)

    panel_kwargs = dict(heatmap_kwargs)
    if shared_scale:
        if center_zero:
            max_value = _matrix_absmax(matrices)
            panel_kwargs.setdefault("vmin", -max_value)
            panel_kwargs.setdefault("vmax", max_value)
        elif "vmin" not in panel_kwargs and "vmax" not in panel_kwargs:
            values = [_matrix_data_and_labels(matrix)[0] for matrix in matrices]
            panel_kwargs["vmin"] = min(float(np.nanmin(value)) for value in values if value.size)
            panel_kwargs["vmax"] = max(float(np.nanmax(value)) for value in values if value.size)

    images = []
    per_axis_colorbar = colorbar == "each" or colorbar is True
    for index, ax in enumerate(axes.flat):
        if index >= len(matrices):
            ax.set_visible(False)
            continue

        _, image = matrix_heatmap(
            matrices[index],
            ax=ax,
            title=titles[index],
            center_zero=center_zero and not shared_scale,
            colorbar=per_axis_colorbar,
            colorbar_label=colorbar_label,
            return_image=True,
            **panel_kwargs,
        )
        images.append(image)

    if colorbar == "shared" and images:
        cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    return fig


def multireplicate_heatmap(
    adata: ad.AnnData,
    *,
    samples: str | Sequence[str] | None = None,
    sample_key: str | None = None,
    title_font_size: int | None = None,
    axes: Sequence[Axes] | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
    uns: str | None = None,
    label_values: bool = False,
    label_font_size: int = None,
    nested: bool | None = None,
    mask: np.ndarray | None = None,
    **heatmap_kwargs,
):
    r"""Plot sample-specific 2D heatmap data from a unified AnnData.

    Exactly one of ``obsm``, ``obsp``, or ``uns`` must be provided.

    Args:
        adata: Unified multisample AnnData object.
        samples: Sample names to plot. By default, plot every sample.
        sample_key: Observation column containing sample identities.
        axes: A predefined set of matplotlib axes to plot on.
        obsm: Key in ``obsm`` containing an observation-by-feature matrix.
        obsp: Key in ``obsp`` containing an observation-by-observation matrix.
        uns: Key in ``uns`` containing either a sample-keyed mapping or one matrix.
        nested: Whether ``uns`` contains a matrix for each sample. By default,
            infer this from whether the stored value is a mapping.
        **heatmap_kwargs: Arguments passed to :func:`matrix_heatmap_panel`.

    """

    provided_locations = [key is not None for key in (obsm, obsp, uns)]
    if sum(provided_locations) != 1:
        raise ValueError("Exactly one of obsm, obsp, or uns must be provided.")
    sample_axis, selected_samples = resolve_samples(
        adata,
        samples=samples,
        sample_key=sample_key,
    )
    cmap = heatmap_kwargs.pop("cmap", "hot")

    matrices = {}
    for sample in selected_samples:
        sample_indices = sample_axis.indices(sample)
        if obsm:
            image = np.asarray(adata.obsm[obsm])[sample_indices]
        elif obsp:
            image = adata.obsp[obsp][sample_indices][:, sample_indices]
            if hasattr(image, "toarray"):
                image = image.toarray()
        elif uns:
            image = adata.uns[uns]

        if uns and (isinstance(image, Mapping) if nested is None else nested):
            image = image[sample]
        if mask is not None:
            image = np.ma.masked_where(mask, image)

        matrices[sample] = image

    fig = matrix_heatmap_panel(
        matrices,
        axes=axes,
        cmap=cmap,
        shared_scale=False,
        colorbar="each",
        label_values=label_values,
        label_font_size=label_font_size,
        **heatmap_kwargs,
    )

    if title_font_size is not None:
        for ax in fig.axes:
            if ax.images:
                ax.title.set_fontsize(title_font_size)

    return fig


def spatial_affinity_heatmap(
    adata: ad.AnnData,
    *,
    samples: str | Sequence[str] | None = None,
    sample_key: str | None = None,
    spatial_affinity_key: str | None = "Sigma_x_inv",
    axes: Sequence[Axes] | None = None,
    metagene_order: Sequence[int] | None = None,
    **heatmap_kwargs,
):
    r"""Plot sample-specific spatial-affinity matrices.

    Args:
        adata: Unified multisample AnnData containing spatial-affinity matrices.
        samples: Sample names to plot. By default, plot every sample.
        sample_key: Observation column containing sample identities.
        axes: A predefined set of matplotlib axes to plot on.
        metagene_order: Optional permutation or subset of metagene indices.
        **heatmap_kwargs: Arguments passed to :func:`matrix_heatmap_panel`.

    """

    _, selected_samples = resolve_samples(
        adata,
        samples=samples,
        sample_key=sample_key,
    )
    cmap = heatmap_kwargs.pop("cmap") if "cmap" in heatmap_kwargs else "bwr"
    spatial_affinities = np.asarray(
        [adata.uns[spatial_affinity_key][sample] for sample in selected_samples],
    )

    _, K, _ = spatial_affinities.shape

    if metagene_order is not None:
        metagene_order = np.asarray(metagene_order)
    else:
        metagene_order = np.arange(K)

    metagene_labels = [f"m{k}" for k in metagene_order]
    matrices = {}
    for sample in selected_samples:
        spatial_affinity = adata.uns[spatial_affinity_key][sample]
        matrices[sample] = pd.DataFrame(
            spatial_affinity[metagene_order][:, metagene_order],
            index=metagene_labels,
            columns=metagene_labels,
        )

    fig = matrix_heatmap_panel(
        matrices,
        axes=axes,
        cmap=cmap,
        center_zero=True,
        shared_scale=True,
        colorbar="shared",
        sharex=True,
        sharey=True,
        **heatmap_kwargs,
    )

    for ax in fig.axes:
        if ax.images:
            ax.set_xticklabels(metagene_labels, fontsize="x-small", rotation=90)
            ax.set_yticklabels(metagene_labels, fontsize="x-small")

    return fig


def multigroup_heatmap(
    adata: ad.AnnData,
    groups: dict,
    title_font_size: int | None = None,
    axes: Sequence[Axes] | None = None,
    key: str | None = None,
    label_values: bool = False,
    label_font_size: int = None,
    **heatmap_kwargs,
):
    r"""Plot one stored matrix for each differential group.

    Args:
        adata: Unified AnnData containing group-keyed matrices in ``uns[key]``.
        groups: Mapping from group names to their member samples.
        axes: A predefined set of matplotlib axes to plot on.
        key: Key in ``uns`` containing matrices keyed by group name.
        **heatmap_kwargs: Arguments passed to Matplotlib's image plot.

    """

    if key is None:
        raise ValueError("key must identify a group-keyed mapping in adata.uns.")
    sharex = True if "sharex" not in heatmap_kwargs else heatmap_kwargs.pop("sharex", True)
    sharey = True if "sharey" not in heatmap_kwargs else heatmap_kwargs.pop("sharey", True)

    fig = None
    if axes is None:
        fig, axes = setup_squarish_axes(len(groups), sharex=sharex, sharey=sharey)
    else:
        axes = np.asarray(axes, dtype=object)
        fig = axes.flat[0].get_figure()

    aspect = heatmap_kwargs.pop("aspect", 0.05)
    cmap = heatmap_kwargs.pop("cmap", "hot")

    for group_index, (ax, group_name) in enumerate(zip(axes.flat, groups)):
        if group_index > len(groups):
            ax.set_visible(False)
            continue

        image = adata.uns[key][group_name]

        im = ax.imshow(image, cmap=cmap, interpolation="nearest", aspect=aspect, **heatmap_kwargs)
        if title_font_size is not None:
            ax.set_title(group_name, fontsize=title_font_size)
        if label_values:
            truncated_image = image.astype(int)
            for (j, i), label in np.ndenumerate(truncated_image):
                ax.text(i, j, label, ha="center", va="center", fontsize=label_font_size)

        fig.colorbar(im, ax=ax, orientation="vertical")

    return fig


def confusion_matrix(
    dataset: ad.AnnData,
    labels: str,
    ax=None,
    confusion_matrix_key: str = "confusion_matrix",
):
    """Plot a confusion matrix stored on a unified AnnData object."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ordered_labels = sorted(dataset.obs[labels].unique())
    sns.heatmap(
        dataset.uns[confusion_matrix_key],
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        annot=True,
        ax=ax,
        fmt="3d",
    )
    return fig, ax
