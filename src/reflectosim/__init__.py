"""X-ray reflectivity forward physics and realistic synthetic data for machine learning."""

from importlib.metadata import version

from reflectosim import materials
from reflectosim.forward import Instrument, forward
from reflectosim.priors import InstrumentPrior, OxidePrior, StackPrior, SubstratePrior
from reflectosim.simulate import Batch, simulate
from reflectosim.structure import Layer, Stack
from reflectosim.units import CU_KALPHA1, CU_KALPHA2, fwhm_to_sigma, q_to_tth, tth_to_q

__version__ = version("reflectosim")

__all__ = [
    "CU_KALPHA1",
    "CU_KALPHA2",
    "Batch",
    "Instrument",
    "InstrumentPrior",
    "Layer",
    "OxidePrior",
    "Stack",
    "StackPrior",
    "SubstratePrior",
    "__version__",
    "forward",
    "fwhm_to_sigma",
    "materials",
    "q_to_tth",
    "simulate",
    "tth_to_q",
]
