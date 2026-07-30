"""Internal matplotlib helpers shared by Popari plots."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def setup_squarish_axes(num_axes: int, **subplots_kwargs):
    """Create a compact, approximately square grid of axes."""

    if num_axes < 1:
        raise ValueError("num_axes must be positive.")

    height = int(np.sqrt(num_axes))
    width = num_axes // height
    height += width * height != num_axes

    constrained_layout = subplots_kwargs.pop("constrained_layout", True)
    dpi = subplots_kwargs.pop("dpi", 300)
    sharex = subplots_kwargs.pop("sharex", True)
    sharey = subplots_kwargs.pop("sharey", True)

    return plt.subplots(
        height,
        width,
        squeeze=False,
        constrained_layout=constrained_layout,
        dpi=dpi,
        sharex=sharex,
        sharey=sharey,
        **subplots_kwargs,
    )


def _highlight_cell(x: int, y: int, ax=None, **kwargs):
    """Draw a rectangle around one heatmap cell."""

    ax = ax or plt.gca()
    rectangle = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
    ax.add_patch(rectangle)
    return rectangle
