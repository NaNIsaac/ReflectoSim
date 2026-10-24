# ReflectoSim

[![Tests](https://github.com/NaNIsaac/ReflectoSim/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/ReflectoSim/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<img src="https://raw.githubusercontent.com/NaNIsaac/ReflectoSim/main/docs/media/curves.png" width="720" alt="Four curves drawn from examples/lab_films.toml: the reflectivity before noise and the Poisson counts">

Realistic synthetic X-ray reflectivity (XRR) data for machine learning: priors over layer stacks and
instruments, counting noise, and a dataset format that records how to regenerate it.

It is for anyone training or testing an inverse model for XRR who needs simulated curves that look
like measured ones: random multilayers with roughness tied to thickness and a minimum contrast
between neighbours, a native oxide placed where you say, Gaussian resolution, footprint, Kα₂, 2θ
offset, background and Poisson counts. The curves come from the forward model of
[ReflectoFit](https://github.com/NaNIsaac/ReflectoFit), the fitting engine, so a network trained on
them and a fit of the same measurement use the same physics; this package adds the data distribution
and nothing else.

Status: early development.

## Install

Not on PyPI yet. From a clone, with [uv](https://docs.astral.sh/uv/) (it fetches ReflectoFit from
GitHub):

```sh
git clone https://github.com/NaNIsaac/ReflectoSim
cd ReflectoSim
uv sync
```

## First run

Simulate 1000 curves of one to three films on SiO₂ on Si from the example prior, then summarise the
file:

<!-- test -->
```sh
uv run reflectosim sim --prior examples/lab_films.toml --n 1000 --seed 20261024,1 -o train.npz
uv run reflectosim describe train.npz
```

`train.npz` holds `q`, `counts`, `reflectivity` (before noise), `counts_scale` and one array per
parameter (`layer0.thickness`, `oxide.sld`, `tth_offset`, ...; NaN where a curve has fewer layers).
Its metadata holds the priors, the seed and the versions; `reflectosim.priors.from_metadata` turns it
back into priors and the same seed gives the same file. Use a `.h5` name for HDF5 (`uv sync --extra
hdf5`).

## Priors

A prior is a TOML file; `examples/lab_films.toml` shows every field. The native oxide position has no
default: `oxide = "under"` puts it on the substrate, `"over"` on top of the films, `"none"` leaves it
out. The two orders give curves that differ by tens of percent, so every prior states its choice.

## Python

```python
import numpy as np
from reflectofit import tth_to_q
from reflectosim import InstrumentPrior, StackPrior, simulate

stack = StackPrior(
    n_layers=(1, 3),
    thickness=(20, 250),
    sld=(5, 80),
    roughness=(2, 10),
    min_contrast=4.0,
    oxide="none",
)
instrument = InstrumentPrior(
    sigma_q=(5e-4, 3e-3), tth_offset=(-0.01, 0.01), log_background=(-8.5, -7.0), log_counts=(6, 8)
)
batch = simulate(
    stack, instrument, tth_to_q(np.linspace(0.05, 8, 500)), n=10_000, seed=(20261024, 1)
)
batch.counts, batch.reflectivity, batch.params["layer0.thickness"]
```

## Development

```sh
uv run pytest
```

## License

MIT
