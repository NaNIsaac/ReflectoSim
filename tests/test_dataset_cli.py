"""Dataset files round-trip, the example prior loads, and the command line runs."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from reflectofit import tth_to_q

from reflectosim import dataset, priors, simulate
from reflectosim.cli import main
from reflectosim.noise import poisson_counts

EXAMPLE = Path(__file__).parents[1] / "examples" / "lab_films.toml"
Q = tth_to_q(np.linspace(0.05, 8.0, 120))


@pytest.fixture(scope="module")
def made():
    stack, inst = priors.load(EXAMPLE)
    return stack, inst, simulate(stack, inst, Q, n=40, seed=(1, 2))


def test_example_prior_loads(made):
    stack, inst, _ = made
    assert stack.oxide == "under" and stack.n_layers == (1, 3)
    assert stack.substrate.sld == pytest.approx(20.071 + 0.457j, abs=1e-3)
    assert inst.sigma_q == (5e-4, 3e-3)


@pytest.mark.parametrize("suffix", [".npz", ".h5"])
def test_dataset_round_trip(made, tmp_path, suffix):
    stack, inst, batch = made
    path = dataset.write(tmp_path / f"d{suffix}", batch, stack, inst)
    arrays, params, meta = dataset.read(path)
    np.testing.assert_array_equal(arrays["counts"], batch.counts.astype(np.float32))
    np.testing.assert_allclose(arrays["reflectivity"], batch.reflectivity, rtol=1e-6)
    for k, v in batch.params.items():
        np.testing.assert_array_equal(params[k], v)
    assert meta["seed"] == [1, 2] and meta["resolution_nodes"] == batch.nodes
    assert meta["stack_prior"]["oxide"] == "under"
    again = simulate(
        *priors.from_metadata(meta), arrays["q"], n=meta["n"], seed=tuple(meta["seed"])
    )
    np.testing.assert_array_equal(again.counts, batch.counts)


def test_noise_terms():
    rng = np.random.default_rng(0)
    r = np.full((2, 20000), 1e-4)
    y = poisson_counts(r, np.array([1e6, 1e4]), rng, dark=5.0, read_sigma=2.0)
    assert y[0].mean() == pytest.approx(105, rel=0.01)
    assert y[1].var() == pytest.approx(6.0 + 4.0, rel=0.05)


def test_unsatisfiable_contrast_is_reported():
    p = priors.StackPrior(
        n_layers=(2, 2), thickness=(50, 50), sld=(10, 11), oxide="none", min_contrast=4
    )
    with pytest.raises(ValueError, match="min_contrast"):
        simulate(p, priors.InstrumentPrior(), Q, n=3, seed=0)


def test_cli_sim_and_describe(tmp_path, capsys):
    out = tmp_path / "train.npz"
    args = [
        "sim",
        "--prior",
        str(EXAMPLE),
        "--n",
        "16",
        "--grid",
        "tth:0.05:6:100",
        "--seed",
        "20261024,1",
        "-o",
        str(out),
    ]
    assert main(args) == 0
    assert main(["describe", str(out)]) == 0
    text = capsys.readouterr().out
    assert "16 curves x 100 points" in text
    assert json.loads(text.strip().splitlines()[-1])["seed"] == [20261024, 1]


def test_node_count_can_be_fixed_for_speed():
    stack, inst = priors.load(EXAMPLE)
    auto = simulate(stack, inst, Q, n=4, seed=0)
    assert auto.nodes > 41  # the converged count for this sample's widest resolution
    fast = simulate(stack, dataclasses.replace(inst, nodes=41), Q, n=4, seed=0)
    assert fast.nodes == 41
    rel = np.max(np.abs(fast.reflectivity / auto.reflectivity - 1))
    assert 0 < rel < 1e-2
