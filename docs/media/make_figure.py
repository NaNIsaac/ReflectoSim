"""Draw docs/media/curves.png: four curves from examples/lab_films.toml, before and after noise.

uv run --with matplotlib docs/media/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reflectofit import tth_to_q

from reflectosim import priors, simulate

ROOT = Path(__file__).resolve().parents[2]
SEED = (20261024, 7)
N = 4
SHIFT = 1e-3  # each curve sits three decades below the one above it

stack, inst = priors.load(ROOT / "examples/lab_films.toml")
q = tth_to_q(np.linspace(0.05, 8, 500), inst.wavelength)
batch = simulate(stack, inst, q, n=N, seed=SEED)

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150, layout="constrained")
colors = plt.get_cmap("tab10").colors
for i in range(N):
    scale = SHIFT**i
    observed = batch.counts[i] / batch.counts_scale[i]
    keep = observed > 0
    films = int(batch.n_layers[i])
    thick = ", ".join(f"{batch.params[f'layer{j}.thickness'][i]:.0f}" for j in range(films))
    mantissa, exponent = f"{batch.counts_scale[i]:.1e}".split("e")
    label = (
        f"{films} film{'s' if films > 1 else ''}: {thick} Å; "
        f"{mantissa}×10$^{{{int(exponent)}}}$ counts at R = 1"
    )
    ax.semilogy(q[keep], observed[keep] * scale, ".", ms=2.5, color=colors[i], alpha=0.6)
    ax.semilogy(q, batch.reflectivity[i] * scale, "-", lw=1, color=colors[i], label=label)
ax.set_xlabel("q (Å$^{-1}$)")
ax.set_ylabel("reflectivity (offset)")
ax.set_xlim(0, q[-1])
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.set_title("lines: before noise    points: Poisson counts / scale", fontsize=9)
fig.savefig(ROOT / "docs/media/curves.png")
