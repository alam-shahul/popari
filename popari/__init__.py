import sys

from popari.model import Popari, from_pretrained
from popari.train import Trainer, TrainParameters

from . import analysis as tl
from . import plotting as pl
from . import preprocessing as pp
from .__about__ import __version__
from .schema import PopariNamespace
from .simulation.schema import MetageneSimulationNamespace

sys.modules.update({f"{__name__}.{module}": globals()[module] for module in ("tl", "pl", "pp")})

__all__ = [
    "Popari",
    "PopariNamespace",
    "MetageneSimulationNamespace",
    "Trainer",
    "TrainParameters",
    "from_pretrained",
    "pl",
    "pp",
    "tl",
    "__version__",
]
