# ReflectoSim

ReflectoSim generates realistic synthetic X-ray reflectivity (XRR) curves for training and testing
inverse models. You describe a distribution, a prior over layer stacks and instruments, and it draws
curves from it with counting noise and writes them to a dataset file that records how to regenerate
it.

The structure prior covers random multilayers: layer count, thickness, roughness tied to thickness
(σ ≤ ratio · d), SLD with a minimum contrast between neighbours, absorption, and a native oxide
whose position you must state (on the substrate, on top, or none). The instrument prior covers
resolution, 2θ offset, I0, background, Cu Kα₂, footprint and counting statistics. Counts are Poisson,
with optional dark counts and read noise.

The curves come from the forward model of [ReflectoFit](https://github.com/NaNIsaac/ReflectoFit),
the fitting engine; ReflectoSim adds the data distribution and has no physics of its own. Before
noise, every curve equals `reflectofit.forward` for its row, a curve does not depend on how the work
is chunked, and the same seed gives the same file.

## Install

Not on PyPI yet. From a clone, with [uv](https://docs.astral.sh/uv/) (it fetches ReflectoFit from
GitHub):

```sh
git clone https://github.com/NaNIsaac/ReflectoSim
cd ReflectoSim
uv sync
```

## First run

Simulate 1000 curves of one to three films on SiO₂ on Si from the example prior and summarise the
file:

```sh
uv run reflectosim sim --prior examples/lab_films.toml --n 1000 --seed 20261024,1 -o train.npz
uv run reflectosim describe train.npz
```

The dataset layout, the prior file format and the Python API are in the
[README](https://github.com/NaNIsaac/ReflectoSim#readme).

## Accuracy and cost

By default the resolution integral is converged to about 1e-5 (ReflectoFit's per-point rule), which
is the slow part for wide resolution and weakly absorbing films. `InstrumentPrior.nodes` fixes a
smaller node count for speed; per NaNIsaac DECISIONS #24 that is for generators who state an error
budget against the counting noise in advance and record it in the dataset metadata.
