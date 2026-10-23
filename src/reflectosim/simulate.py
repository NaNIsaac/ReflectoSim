"""Draw stacks and instruments from priors and simulate their curves with ReflectoFit.

Curves before noise are `reflectofit.forward` for each row (this package has no physics of its own).
Rows with fewer layers than the prior's maximum are padded with zero-thickness layers that match the
medium below, which leaves the curve unchanged. One resolution node count is fixed for the whole
dataset, so a row's curve does not depend on how the work is chunked.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from reflectofit import Instrument, forward
from reflectofit.forward import resolution_nodes

from reflectosim.noise import poisson_counts
from reflectosim.priors import InstrumentPrior, StackPrior, draw_instruments, draw_stacks

NOISE_BLOCK = 4096
"""Rows per noise stream; fixed, so the noise does not depend on `chunk`."""


@dataclass
class Batch:
    q: NDArray
    params: dict[str, NDArray]
    packed: tuple[NDArray, NDArray, NDArray]
    n_layers: NDArray
    counts_scale: NDArray
    reflectivity: NDArray
    """Observed reflectivity before noise: I0 · (smeared R) + background, per row."""
    counts: NDArray
    oxide: str
    nodes: int
    seed: object


INSTRUMENT_KEYS = (
    "i0",
    "background",
    "tth_offset",
    "sigma_q",
    "dq_over_q",
    "kalpha2_ratio",
    "sample_length",
    "beam_sigma",
)


def pack(prior: StackPrior, p: dict[str, NDArray]) -> tuple[NDArray, NDArray, NDArray]:
    """(n, N) media arrays, ambient first: [ambient, oxide if over, layers padded, oxide if under,
    substrate]. A missing layer copies the SLD of the medium below its stack, has zero thickness,
    and carries that medium's roughness on its top interface, so it is invisible."""
    n, hi = p["n_layers"].size, prior.n_layers[1]
    over, under = prior.oxide == "over", prior.oxide == "under"
    width = 1 + hi + (prior.oxide != "none") + 1
    sld = np.zeros((n, width), complex)
    thk = np.zeros((n, width))
    rgh = np.zeros((n, width))
    col = 1

    def put(c, s, d, r):
        sld[:, c], thk[:, c], rgh[:, c] = s, d, r

    if over:
        put(col, p["oxide.sld"] + 1j * p["oxide.isld"], p["oxide.thickness"], p["oxide.roughness"])
        col += 1
    below_name = "oxide" if under else "substrate"
    below_sld = p[f"{below_name}.sld"] + 1j * p[f"{below_name}.isld"]
    below_rough = p[f"{below_name}.roughness"]
    first_layer = col
    for j in range(hi):
        present = j < p["n_layers"]
        first_pad = j == p["n_layers"]
        s = np.where(present, p[f"layer{j}.sld"] + 1j * p[f"layer{j}.isld"], below_sld)
        d = np.where(present, p[f"layer{j}.thickness"], 0.0)
        r = np.where(present, p[f"layer{j}.roughness"], np.where(first_pad, below_rough, 0.0))
        put(first_layer + j, s, d, r)
    col = first_layer + hi
    if under:
        put(col, below_sld, p["oxide.thickness"], np.where(p["n_layers"] < hi, 0.0, below_rough))
        col += 1
        sub_rough = p["substrate.roughness"]
    else:
        sub_rough = np.where(p["n_layers"] < hi, 0.0, below_rough)
    put(col, p["substrate.sld"] + 1j * p["substrate.isld"], 0.0, sub_rough)
    return sld, thk, rgh


def simulate(
    prior: StackPrior,
    instrument: InstrumentPrior,
    q: NDArray,
    n: int,
    seed=0,
    chunk: int = 4096,
) -> Batch:
    """n curves on the q grid [Å⁻¹], with structure and instrument drawn from the priors.

    `seed` is an int or a tuple of ints (as ReflectoLearn#35: (20261024, n, j)); the parameters and
    the noise use separate streams spawned from it.
    """
    q = np.asarray(q, float)
    entropy = list(seed) if isinstance(seed, tuple | list) else seed
    rng = np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(0,)))
    params = draw_stacks(prior, n, rng) | draw_instruments(instrument, n, rng)
    packed = pack(prior, params)

    sigma_max = float(np.sqrt(params["sigma_q"] ** 2 + (params["dq_over_q"] * q.max()) ** 2).max())
    nodes = resolution_nodes(sigma_max, float(packed[1].sum(1).max())) if sigma_max > 0 else 1
    footprint = not np.isnan(params["sample_length"]).all()

    reflectivity = np.empty((n, q.size))
    for start in range(0, n, chunk):
        rows = slice(start, min(start + chunk, n))
        values = {k: params[k][rows] for k in INSTRUMENT_KEYS}
        if not footprint:
            values["sample_length"] = values["beam_sigma"] = None
        inst = Instrument(wavelength=instrument.wavelength, nodes=nodes, **values)
        reflectivity[rows] = forward(q, tuple(a[rows] for a in packed), inst)

    counts = np.empty_like(reflectivity)
    for block in range(0, n, NOISE_BLOCK):
        rows = slice(block, min(block + NOISE_BLOCK, n))
        noise_rng = np.random.default_rng(
            np.random.SeedSequence(entropy, spawn_key=(1, block // NOISE_BLOCK))
        )
        counts[rows] = poisson_counts(reflectivity[rows], params["counts_scale"][rows], noise_rng)

    return Batch(
        q=q,
        params=params,
        packed=packed,
        n_layers=params["n_layers"],
        counts_scale=params["counts_scale"],
        reflectivity=reflectivity,
        counts=counts,
        oxide=prior.oxide,
        nodes=nodes,
        seed=seed,
    )
