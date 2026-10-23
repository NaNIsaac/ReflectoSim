"""What a simulated dataset promises, pinned before the generator is written.

1. Before noise, every curve is ReflectoFit's forward model for that row's stack and instrument:
   the simulator has no physics of its own (ReflectoFit DESIGN.md section 0.2).
2. Curves with fewer layers are padded to a common width without changing them.
3. The prior's constraints hold on every row: sigma <= ratio * d, adjacent contrast.
4. The oxide position is never implicit, and it is where the prior says.
5. A seed fixes the dataset, independent of how the work is chunked.
"""

import numpy as np
import pytest
from reflectofit import Instrument, Layer, Stack, forward, tth_to_q

from reflectosim import InstrumentPrior, OxidePrior, StackPrior, simulate

Q = tth_to_q(np.linspace(0.05, 8.0, 200))
STACK = StackPrior(
    n_layers=(1, 3),
    thickness=(20.0, 250.0),
    roughness=(2.0, 10.0),
    roughness_max_ratio=0.25,
    sld=(5.0, 80.0),
    isld_ratio=(0.0, 0.05),
    min_contrast=4.0,
    oxide="under",
    oxide_prior=OxidePrior(thickness=(5.0, 25.0), roughness=(1.0, 5.0), sld=(17.0, 20.0)),
)
INSTRUMENT = InstrumentPrior(
    sigma_q=(5e-4, 3e-3),
    dq_over_q=(0.0, 0.01),
    tth_offset=(-0.01, 0.01),
    i0=(0.8, 1.2),
    log_background=(-8.5, -7.0),
    kalpha2_ratio=(0.0, 0.5),
    sample_length=(8.0, 20.0),
    beam_sigma=(0.02, 0.06),
    log_counts=(6.0, 8.0),
)


@pytest.fixture(scope="module")
def batch():
    return simulate(STACK, INSTRUMENT, Q, n=64, seed=(20261024, 1))


def _row(batch, i):
    """Stack and instrument of row i, rebuilt from the stored parameter table alone."""
    p = batch.params
    layers = []
    for j in range(batch.n_layers[i]):
        sld = p[f"layer{j}.sld"][i] + 1j * p[f"layer{j}.isld"][i]
        layers.append(Layer(sld, p[f"layer{j}.thickness"][i], p[f"layer{j}.roughness"][i]))
    if "oxide.thickness" in p:
        oxide = Layer(
            p["oxide.sld"][i] + 1j * p["oxide.isld"][i],
            p["oxide.thickness"][i],
            p["oxide.roughness"][i],
        )
        layers = [*layers, oxide] if batch.oxide == "under" else [oxide, *layers]
    sub = Layer(
        p["substrate.sld"][i] + 1j * p["substrate.isld"][i], roughness=p["substrate.roughness"][i]
    )
    names = (
        "i0",
        "background",
        "tth_offset",
        "sigma_q",
        "dq_over_q",
        "kalpha2_ratio",
        "sample_length",
        "beam_sigma",
    )
    # the dataset fixes one resolution node count for all rows (independent of chunking)
    inst = Instrument(**{k: float(p[k][i]) for k in names}, nodes=batch.nodes)
    return Stack(layers, sub), inst


def test_curves_before_noise_are_reflectofit_forward(batch):
    for i in range(0, 64, 7):
        stack, inst = _row(batch, i)
        np.testing.assert_allclose(batch.reflectivity[i], forward(Q, stack, inst), rtol=1e-12)


def test_padding_does_not_change_a_curve(batch):
    assert len(set(batch.n_layers.tolist())) > 1, "the batch should mix layer counts"
    assert batch.packed[0].shape[1] == 1 + 3 + 1 + 1  # ambient, up to 3 layers, oxide, substrate
    thin = int(np.argmin(batch.n_layers))
    stack, _ = _row(batch, thin)
    np.testing.assert_allclose(
        forward(Q, tuple(a[thin] for a in batch.packed)), forward(Q, stack), rtol=1e-12
    )


def test_roughness_never_exceeds_the_ratio_of_its_layer(batch):
    for j in range(3):
        d, s = batch.params[f"layer{j}.thickness"], batch.params[f"layer{j}.roughness"]
        ok = ~np.isnan(d)
        assert np.all(s[ok] <= 0.25 * d[ok] + 1e-12)
        assert np.all(s[ok] >= min(2.0, 0.25 * 20.0) - 1e-12)


def test_adjacent_contrast_holds_down_to_the_medium_below(batch):
    p = batch.params
    for i in range(64):
        rho = [p[f"layer{j}.sld"][i] for j in range(batch.n_layers[i])]
        below = p["oxide.sld"][i] if batch.oxide == "under" else p["substrate.sld"][i]
        rho.append(below)
        assert np.all(np.abs(np.diff(rho)) >= 4.0 - 1e-12), i


def test_oxide_position_is_explicit():
    with pytest.raises(TypeError):
        StackPrior(n_layers=(1, 1), thickness=(20, 50), sld=(5, 80))  # oxide not given
    over = simulate(StackPrior(n_layers=(1, 1), thickness=(100, 100), sld=(40, 40), oxide="over",
                               oxide_prior=OxidePrior(thickness=(10, 10), sld=(18, 18))),
                    InstrumentPrior(), Q, n=1, seed=0)  # fmt: skip
    sld, thickness, _ = (a[0] for a in over.packed)
    assert sld[1].real == pytest.approx(18) and thickness[1] == 10  # oxide right under the ambient
    under = simulate(StackPrior(n_layers=(1, 1), thickness=(100, 100), sld=(40, 40), oxide="under",
                                oxide_prior=OxidePrior(thickness=(10, 10), sld=(18, 18))),
                     InstrumentPrior(), Q, n=1, seed=0)  # fmt: skip
    sld, thickness, _ = (a[0] for a in under.packed)
    assert sld[2].real == pytest.approx(18) and thickness[2] == 10  # oxide right on the substrate


def test_seed_fixes_the_dataset_independent_of_chunking(batch):
    again = simulate(STACK, INSTRUMENT, Q, n=64, seed=(20261024, 1), chunk=10)
    np.testing.assert_array_equal(again.counts, batch.counts)
    for k, v in batch.params.items():
        np.testing.assert_array_equal(again.params[k], v)
    other = simulate(STACK, INSTRUMENT, Q, n=64, seed=(20261024, 2))
    assert not np.array_equal(other.counts, batch.counts)


def test_counts_are_poisson_around_the_expected_counts(batch):
    lam = batch.counts_scale[:, None] * batch.reflectivity
    z = (batch.counts - lam) / np.sqrt(lam)
    big = lam > 100
    assert abs(np.mean(z[big])) < 0.05
    assert np.std(z[big]) == pytest.approx(1.0, abs=0.05)
