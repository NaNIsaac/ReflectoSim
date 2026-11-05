"""The q grid of the command lines."""

import numpy as np

from reflectosim.units import tth_to_q


def q_grid(spec: str, wavelength: float) -> np.ndarray:
    """'tth:START:STOP:N' (degrees) or 'q:START:STOP:N' (Å⁻¹) as q in Å⁻¹."""
    axis, start, stop, n = spec.split(":")
    values = np.linspace(float(start), float(stop), int(n))
    if axis == "tth":
        return tth_to_q(values, wavelength)
    if axis == "q":
        return values
    raise SystemExit(f"grid axis must be tth or q, got {axis!r}")
