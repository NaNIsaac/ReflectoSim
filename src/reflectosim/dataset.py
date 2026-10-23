"""Dataset files: HDF5 (with the `hdf5` extra) or compressed npz, chosen by the file suffix.

Arrays: `q` (Q,), `counts` and `reflectivity` (n, Q), `counts_scale` (n,), and one (n,) array per
parameter under `params/<name>` (HDF5) or `params.<name>` (npz). The metadata is one JSON string with
the priors, the seed, the resolution node count, the package versions and the conventions, enough
to regenerate the file.
"""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from reflectosim import priors
from reflectosim.simulate import Batch

CONVENTIONS = (
    "ReflectoFit DESIGN.md section 2: ambient first, roughness of the interface above, Im SLD > 0 "
    "absorbs, SLD 1e-6 A^-2, q A^-1, angles in degrees; reflectivity is I0 * smeared R + background "
    "before noise; counts are Poisson(counts_scale * reflectivity)"
)


def metadata(batch: Batch, stack: priors.StackPrior, instrument: priors.InstrumentPrior) -> dict:
    seed = list(batch.seed) if isinstance(batch.seed, tuple | list) else batch.seed
    return {
        "reflectosim": version("reflectosim"),
        "reflectofit": version("reflectofit"),
        "stack_prior": priors.to_dict(stack),
        "instrument_prior": priors.to_dict(instrument),
        "seed": seed,
        "n": int(batch.counts.shape[0]),
        "resolution_nodes": batch.nodes,
        "backend": "reflectofit numba float64",
        "conventions": CONVENTIONS,
    }


def write(
    path: str | Path, batch: Batch, stack: priors.StackPrior, instrument: priors.InstrumentPrior
) -> Path:
    path = Path(path)
    meta = json.dumps(metadata(batch, stack, instrument))
    arrays = {
        "q": batch.q,
        "counts": batch.counts.astype(np.float32),
        "reflectivity": batch.reflectivity.astype(np.float32),
        "counts_scale": batch.counts_scale,
    }
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=v, compression="gzip" if v.ndim == 2 else None)
            g = f.create_group("params")
            for k, v in batch.params.items():
                g.create_dataset(k, data=v)
            f.attrs["metadata"] = meta
    elif path.suffix == ".npz":
        np.savez_compressed(
            path, **arrays, **{f"params.{k}": v for k, v in batch.params.items()}, metadata=meta
        )
    else:
        raise ValueError(f"unknown dataset suffix {path.suffix!r}; use .h5 or .npz")
    return path


def read(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """(arrays, params, metadata) of a dataset file."""
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            arrays = {k: f[k][()] for k in ("q", "counts", "reflectivity", "counts_scale")}
            params = {k: f["params"][k][()] for k in f["params"]}
            meta = json.loads(f.attrs["metadata"])
        return arrays, params, meta
    with np.load(path) as z:
        arrays = {k: z[k] for k in ("q", "counts", "reflectivity", "counts_scale")}
        params = {k.removeprefix("params."): z[k] for k in z.files if k.startswith("params.")}
        meta = json.loads(str(z["metadata"]))
    return arrays, params, meta
