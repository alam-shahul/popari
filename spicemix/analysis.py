from typing import Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import numpy as np
from spicemix.model import SpiceMixPlus

def plot_metagene_embedding(trained_model: SpiceMixPlus, metagene_index: int, axes: Optional[Sequence[Axes]] = None):
    datasets = trained_model.datasets

    if axes == None:
        height = int(np.sqrt(len(datasets)))
        width = len(datasets) // width + (width * height != len(datasets))
        fig, axes = plt.subplots((width, height))

    for dataset, ax in zip(datasets, axes):
        dataset.plot_metagene_embedding(metagene_index, ax=ax)

    return fig
