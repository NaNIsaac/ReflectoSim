"""Realistic synthetic X-ray reflectivity data for machine learning."""

from importlib.metadata import version

from reflectosim.priors import InstrumentPrior, OxidePrior, StackPrior, SubstratePrior
from reflectosim.simulate import Batch, simulate

__version__ = version("reflectosim")

__all__ = [
    "Batch",
    "InstrumentPrior",
    "OxidePrior",
    "StackPrior",
    "SubstratePrior",
    "__version__",
    "simulate",
]
