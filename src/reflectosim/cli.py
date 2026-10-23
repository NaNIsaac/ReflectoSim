"""Command line: reflectosim sim | describe."""

import argparse
import json
import sys

import numpy as np
from reflectofit import tth_to_q

from reflectosim import __version__, dataset, priors
from reflectosim.simulate import simulate


def _grid(spec: str, wavelength: float) -> np.ndarray:
    """'tth:START:STOP:N' (degrees) or 'q:START:STOP:N' (1/A)."""
    axis, start, stop, n = spec.split(":")
    values = np.linspace(float(start), float(stop), int(n))
    if axis == "tth":
        return tth_to_q(values, wavelength)
    if axis == "q":
        return values
    raise SystemExit(f"grid axis must be tth or q, got {axis!r}")


def _seed(text: str):
    parts = [int(p) for p in text.split(",")]
    return parts[0] if len(parts) == 1 else tuple(parts)


def _sim(args) -> int:
    stack, inst = priors.load(args.prior)
    q = _grid(args.grid, inst.wavelength)
    batch = simulate(stack, inst, q, n=args.n, seed=_seed(args.seed))
    out = dataset.write(args.out, batch, stack, inst)
    print(f"wrote {out}: {args.n} curves x {q.size} points, oxide {stack.oxide}, "
          f"{batch.nodes} resolution nodes")  # fmt: skip
    return 0


def _describe(args) -> int:
    arrays, params, meta = dataset.read(args.dataset)
    n, nq = arrays["counts"].shape
    print(
        f"{args.dataset}: {n} curves x {nq} points, q {arrays['q'][0]:.4g}-{arrays['q'][-1]:.4g} 1/A"
    )
    width = max(len(k) for k in params)
    for k, v in params.items():
        ok = np.isfinite(v)
        if ok.any():
            print(f"  {k:<{width}}  {v[ok].min():.4g} .. {v[ok].max():.4g}  ({ok.sum()} rows)")
    print(
        json.dumps({k: meta[k] for k in ("reflectosim", "reflectofit", "seed", "resolution_nodes")})
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reflectosim", description="synthetic X-ray reflectivity data")
    p.add_argument("--version", action="version", version=f"reflectosim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sim", help="simulate a dataset from a prior TOML")
    s.add_argument(
        "--prior", required=True, help="prior TOML: [stack], [oxide], [substrate], [instrument]"
    )
    s.add_argument("--n", type=int, required=True, help="number of curves")
    s.add_argument("--grid", default="tth:0.05:8:500", help="tth:START:STOP:N or q:START:STOP:N")
    s.add_argument("--seed", default="0", help="an int or comma-separated ints, e.g. 20261024,1")
    s.add_argument("-o", "--out", required=True, help="output .h5 (needs h5py) or .npz")
    s.set_defaults(run=_sim)

    d = sub.add_parser("describe", help="summarise a dataset file")
    d.add_argument("dataset")
    d.set_defaults(run=_describe)

    args = p.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
