"""Global plotting presets."""

import warnings
from typing import Literal

import matplotlib
import scanpy as sc


def set_notebook_mode(mode: Literal["publication", "exploration"] = "exploration") -> None:
    """Configure plotting and warning behavior for a notebook workflow.

    Args:
        mode: ``"publication"`` preserves PDF text as editable TrueType text,
            rasterizes dense Scanpy scatter artists, and suppresses all Python
            warnings. ``"exploration"`` restores default PDF fonts, disables
            Scanpy's vector-friendly rasterization, and shows warnings normally.
            Both modes disable axes grids.

    Raises:
        ValueError: If ``mode`` is not supported.

    """

    if mode == "publication":
        matplotlib.rcParams["pdf.fonttype"] = 42
        sc.set_figure_params(vector_friendly=True)
        matplotlib.rcParams["axes.grid"] = False
        warnings.simplefilter("ignore")
    elif mode == "exploration":
        matplotlib.rcParams["pdf.fonttype"] = matplotlib.rcParamsDefault["pdf.fonttype"]
        sc.set_figure_params(vector_friendly=False)
        matplotlib.rcParams["axes.grid"] = False
        warnings.simplefilter("default")
    else:
        raise ValueError("mode must be 'publication' or 'exploration'.")


__all__ = [set_notebook_mode.__name__]
