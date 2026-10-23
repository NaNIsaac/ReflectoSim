"""Counting noise for simulated curves (ReflectoFit DESIGN.md section 2, Counts)."""

import numpy as np
from numpy.typing import NDArray


def poisson_counts(
    reflectivity: NDArray,
    counts_scale: NDArray | float,
    rng: np.random.Generator,
    dark: float = 0.0,
    read_sigma: float = 0.0,
) -> NDArray:
    """Counts y ~ Poisson(S · R + dark) + N(0, read_sigma), S the counts at R = 1 per row.

    `reflectivity` (B, Q) is the observed reflectivity before noise (I0 and background included);
    `counts_scale` is S per row (B,) or one value. Dark counts and read noise are detector terms on
    top of the background already in R; both default to off.
    """
    scale = (
        np.asarray(counts_scale, float).reshape(-1, 1) if np.ndim(counts_scale) else counts_scale
    )
    lam = scale * np.asarray(reflectivity) + dark
    y = rng.poisson(lam).astype(float)
    if read_sigma > 0:
        y += rng.normal(0.0, read_sigma, y.shape)
    return y
