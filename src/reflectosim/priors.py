"""Priors over structures and instruments.

Ranges are (low, high) tuples drawn uniformly; a range with low == high is a fixed value. Units as in
ReflectoFit: Å, 1e-6 Å⁻², Å⁻¹, degrees, mm.
"""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from reflectofit import materials

Range = tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class OxidePrior:
    """A native oxide layer; the defaults are amorphous SiO₂ (2.0-2.3 g/cm³) on a Si wafer."""

    thickness: Range = (5.0, 25.0)
    roughness: Range = (1.0, 5.0)
    sld: Range = (17.2, 19.7)
    isld_ratio: Range = (0.013, 0.013)
    """Im SLD as a fraction of Re SLD (SiO₂ at Cu Kα: 0.013)."""


@dataclass(frozen=True, kw_only=True)
class SubstratePrior:
    sld: complex = field(default_factory=lambda: materials.sld("Si", 2.33))
    roughness: Range = (1.0, 5.0)


@dataclass(frozen=True, kw_only=True)
class StackPrior:
    """Random films on a substrate, top first.

    `oxide` has no default on purpose: whether the native oxide lies under the film stack ("under",
    on the substrate), on top of it ("over") or is absent ("none") changes the curves by tens of
    percent and is an open decision (NaNIsaac DECISIONS #19); every prior states it.
    Roughness is drawn from U[lo, min(hi, roughness_max_ratio · d)] for each layer. Adjacent media
    (layer on layer, and the lowest layer on the medium below it) differ in Re SLD by at least
    `min_contrast`; rows that fail are redrawn.
    """

    n_layers: tuple[int, int]
    thickness: Range
    sld: Range
    oxide: Literal["under", "over", "none"]
    roughness: Range = (0.0, 10.0)
    roughness_max_ratio: float | None = 0.25
    isld_ratio: Range = (0.0, 0.0)
    """Im SLD as a fraction of Re SLD."""
    min_contrast: float = 0.0
    oxide_prior: OxidePrior = field(default_factory=OxidePrior)
    substrate: SubstratePrior = field(default_factory=SubstratePrior)

    def __post_init__(self) -> None:
        if self.oxide not in ("under", "over", "none"):
            raise ValueError(f"oxide must be 'under', 'over' or 'none', got {self.oxide!r}")
        lo, hi = self.n_layers
        if not 0 <= lo <= hi:
            raise ValueError(f"n_layers must be 0 <= low <= high, got {self.n_layers}")


@dataclass(frozen=True, kw_only=True)
class InstrumentPrior:
    """Per-curve instrument values. None switches a term off (footprint, background)."""

    sigma_q: Range = (0.0, 0.0)
    dq_over_q: Range = (0.0, 0.0)
    tth_offset: Range = (0.0, 0.0)
    i0: Range = (1.0, 1.0)
    log_background: Range | None = None
    kalpha2_ratio: Range = (0.0, 0.0)
    sample_length: Range | None = None
    beam_sigma: Range | None = None
    log_counts: Range = (8.0, 8.0)
    """log10 of the counts at R = 1."""
    wavelength: float = 1.5406
    nodes: int | None = None
    """None (default): ReflectoFit's converged per-point rule, within about 1e-5 of a brute-force
    convolution; it costs most for wide resolution and weakly absorbing films. An int fixes the node
    count for every point: faster, less accurate. Per NaNIsaac DECISIONS #24, fix it only with an
    error budget against the counting noise stated in advance; the count is recorded in the metadata."""


def _u(rng, r: Range, size):
    lo, hi = r
    return rng.uniform(lo, hi, size) if hi > lo else np.full(size, float(lo))


def draw_stacks(prior: StackPrior, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Parameter table of n stacks: layer{j}.{thickness,roughness,sld,isld} (NaN where a row has
    fewer layers), oxide.* when present, substrate.*, n_layers."""
    lo, hi = prior.n_layers
    count = rng.integers(lo, hi + 1, size=n)
    out: dict[str, np.ndarray] = {"n_layers": count}
    if prior.oxide != "none":
        ox = prior.oxide_prior
        out["oxide.thickness"] = _u(rng, ox.thickness, n)
        out["oxide.roughness"] = _u(rng, ox.roughness, n)
        out["oxide.sld"] = _u(rng, ox.sld, n)
        out["oxide.isld"] = out["oxide.sld"] * _u(rng, ox.isld_ratio, n)
    sub = complex(prior.substrate.sld)
    out["substrate.sld"] = np.full(n, sub.real)
    out["substrate.isld"] = np.full(n, sub.imag)
    out["substrate.roughness"] = _u(rng, prior.substrate.roughness, n)

    below = out["oxide.sld"] if prior.oxide == "under" else out["substrate.sld"]
    rho = _draw_contrasting_sld(prior, count, below, rng)
    for j in range(hi):
        present = j < count
        d = _u(rng, prior.thickness, n)
        top = np.full(n, prior.roughness[1])
        if prior.roughness_max_ratio is not None:
            top = np.minimum(top, prior.roughness_max_ratio * d)
        s = rng.uniform(np.minimum(prior.roughness[0], top), top)
        out[f"layer{j}.thickness"] = np.where(present, d, np.nan)
        out[f"layer{j}.roughness"] = np.where(present, s, np.nan)
        out[f"layer{j}.sld"] = np.where(present, rho[:, j], np.nan)
        out[f"layer{j}.isld"] = np.where(present, rho[:, j] * _u(rng, prior.isld_ratio, n), np.nan)
    return out


def _draw_contrasting_sld(prior, count, below, rng, max_rounds: int = 1000):
    n, hi = count.size, prior.n_layers[1]
    rho = _u(rng, prior.sld, (n, hi)) if hi else np.zeros((n, 0))
    for _ in range(max_rounds):
        bad = np.zeros(n, bool)
        for j in range(hi):
            under = np.where(j + 1 < count, rho[:, min(j + 1, hi - 1)], below)
            bad |= (j < count) & (np.abs(rho[:, j] - under) < prior.min_contrast)
        if not bad.any():
            return rho
        rho[bad] = _u(rng, prior.sld, (int(bad.sum()), hi))
    raise ValueError(
        f"min_contrast {prior.min_contrast} cannot be met within the sld range {prior.sld}"
    )


def draw_instruments(
    prior: InstrumentPrior, n: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    out = {
        "sigma_q": _u(rng, prior.sigma_q, n),
        "dq_over_q": _u(rng, prior.dq_over_q, n),
        "tth_offset": _u(rng, prior.tth_offset, n),
        "i0": _u(rng, prior.i0, n),
        "kalpha2_ratio": _u(rng, prior.kalpha2_ratio, n),
        "background": 10.0 ** _u(rng, prior.log_background, n)
        if prior.log_background
        else np.zeros(n),
        "counts_scale": 10.0 ** _u(rng, prior.log_counts, n),
    }
    if (prior.sample_length is None) != (prior.beam_sigma is None):
        raise ValueError("the footprint needs both sample_length and beam_sigma, or neither")
    if prior.sample_length is not None:
        out["sample_length"] = _u(rng, prior.sample_length, n)
        out["beam_sigma"] = _u(rng, prior.beam_sigma, n)
    else:
        out["sample_length"] = np.full(n, np.nan)
        out["beam_sigma"] = np.full(n, np.nan)
    return out


def to_dict(prior) -> dict:
    """JSON-ready form (complex as [re, im])."""

    def fix(v):
        if isinstance(v, complex):
            return [v.real, v.imag]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, tuple):
            return list(v)
        return v

    return fix(asdict(prior))


def load(path: str | Path) -> tuple[StackPrior, InstrumentPrior]:
    """Read a prior TOML: [stack] (with `oxide`), optional [oxide], [substrate], [instrument]."""
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    tup = {k: tuple(v) if isinstance(v, list) else v for k, v in spec.get("stack", {}).items()}
    oxide = OxidePrior(**{k: tuple(v) for k, v in spec.get("oxide", {}).items()})
    sub_spec = dict(spec.get("substrate", {}))
    if "material" in sub_spec:
        sld = materials.sld(sub_spec.pop("material"), sub_spec.pop("density"))
    elif "sld" in sub_spec:
        sld = complex(*sub_spec.pop("sld"))
    else:
        sld = SubstratePrior().sld
    substrate = SubstratePrior(sld=sld, **{k: tuple(v) for k, v in sub_spec.items()})
    inst = {
        k: tuple(v) if isinstance(v, list) else v for k, v in spec.get("instrument", {}).items()
    }
    return StackPrior(**tup, oxide_prior=oxide, substrate=substrate), InstrumentPrior(**inst)


def from_metadata(meta: dict) -> tuple[StackPrior, InstrumentPrior]:
    """The priors stored in a dataset's metadata (`dataset.read`), to regenerate it."""

    def tuples(d: dict) -> dict:
        return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}

    s = dict(meta["stack_prior"])
    oxide = OxidePrior(**tuples(s.pop("oxide_prior")))
    sub = s.pop("substrate")
    substrate = SubstratePrior(sld=complex(*sub["sld"]), roughness=tuple(sub["roughness"]))
    stack = StackPrior(**tuples(s), oxide_prior=oxide, substrate=substrate)
    return stack, InstrumentPrior(**tuples(meta["instrument_prior"]))
