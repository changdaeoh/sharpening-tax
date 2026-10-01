# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the PTGS controller (ptgs/controller.py) and its trainer hooks (ptgs/integration.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ptgs import PTGS, PTGSHooks, pivot_at, temperature_from_p

ROOT = Path(__file__).resolve().parents[1]


def at_pivot(pivot, **kw):
    return PTGS(pivot_start=pivot, pivot_end=pivot, **kw)


def mean_temperature(c: PTGS, key: str) -> float:
    """Deterministic stand-in for a Thompson draw: the tempering rule at the posterior mean."""
    return temperature_from_p(c.posterior_mean(key), c.pivot, c.tau, c.t_ref)


# -- Tempering rule -------------------------------------------------------------------------
@pytest.mark.parametrize("pivot", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("tau,t_ref", [(1.5, 1.0), (1.2, 1.0), (1.4, 0.7), (2.0, 1.3)])
def test_anchors_hold_for_every_pivot(pivot, tau, t_ref):
    assert temperature_from_p(0.0, pivot, tau, t_ref) == pytest.approx(tau * t_ref)
    assert temperature_from_p(pivot, pivot, tau, t_ref) == pytest.approx(t_ref)
    assert temperature_from_p(1.0, pivot, tau, t_ref) == pytest.approx(t_ref / tau)


@pytest.mark.parametrize("pivot", [0.25, 0.5, 0.75])
def test_monotone_decreasing_and_within_rails(pivot):
    temps = [temperature_from_p(p, pivot, 1.5) for p in np.linspace(0, 1, 101)]
    assert all(a > b for a, b in zip(temps, temps[1:]))
    assert min(temps) == pytest.approx(1.0 / 1.5)
    assert max(temps) == pytest.approx(1.5)
    assert temperature_from_p(-0.1, pivot, 1.5) == pytest.approx(1.5)
    assert temperature_from_p(1.1, pivot, 1.5) == pytest.approx(1.0 / 1.5)


def test_symmetric_form_at_pivot_half():
    """At p~ = 1/2 both branches collapse to T = T_ref * tau^(1-2p)."""
    for p in np.linspace(0, 1, 21):
        assert temperature_from_p(p, 0.5, 1.5, 0.8) == pytest.approx(0.8 * 1.5 ** (1 - 2 * p))


def test_tau_one_is_a_no_op():
    for pivot in (0.25, 0.5):
        assert all(temperature_from_p(p, pivot, 1.0, 0.7) == pytest.approx(0.7) for p in np.linspace(0, 1, 11))


# -- pivot schedule ---------------------------------------------------------------------------
def test_geometric_ramp_endpoints_and_clamping():
    assert pivot_at(0, 200) == pytest.approx(0.25)
    assert pivot_at(200, 200) == pytest.approx(0.5)
    assert pivot_at(-5, 200) == pytest.approx(0.25)
    assert pivot_at(250, 200) == pytest.approx(0.5)
    seq = [pivot_at(s, 200) for s in range(0, 201)]
    ratios = np.array(seq[1:]) / np.array(seq[:-1])
    assert np.allclose(ratios, 2 ** (1 / 200))


def test_one_indexed_steps_end_exactly_at_pivot_end():
    c = PTGS()
    assert c.pivot == pytest.approx(0.25)  # before the first step
    for step in range(1, 201):
        c.set_progress(step, 200)
    assert c.pivot == pytest.approx(0.5)
    assert pivot_at(199, 200) < 0.4985  # a 0-indexed loop stops one step short


# -- posterior ----------------------------------------------------------------------------------
def test_forgetting_update_math():
    c = PTGS(gamma=0.9)
    c.update("x", successes=3, n=5)
    assert c.counts["x"] == pytest.approx([3.0, 2.0])
    c.update("x", successes=0, n=4)
    assert c.counts["x"] == pytest.approx([0.9 * 3.0, 0.9 * 2.0 + 4.0])
    # The prior of mass 2 is added at read time, at the current pivot.
    s, f = c.counts["x"]
    assert c.posterior_mean("x") == pytest.approx((2 * 0.25 + s) / (2 + s + f))
    c.update("y", successes=0, n=0)  # an empty group is a no-op
    assert "y" not in c.counts


def test_unobserved_prompt_posterior_mean_is_the_pivot():
    c = PTGS()
    for step in (1, 50, 100, 200):
        c.set_progress(step, 200)
        assert c.posterior_mean("fresh") == pytest.approx(c.pivot)
        assert mean_temperature(c, "fresh") == pytest.approx(c.t_ref)
    assert "fresh" not in c.counts  # reading never inserts a key


def test_thompson_heats_an_unobserved_prompt_on_average_below_pivot_half():
    """The prior centres the success-rate estimate on p~, not T; the rule's steeper heating
    branch makes the draws heat on average."""
    c = at_pivot(0.25, seed=0)
    draws = np.array([c.temperature("fresh") for _ in range(20000)])
    assert 1.12 < draws.mean() < 1.18
    assert 0.58 < (draws > 1.0).mean() < 0.64


def test_repeated_failure_heats_and_repeated_success_cools():
    hard, easy = at_pivot(0.5), at_pivot(0.5)
    for _ in range(5):
        hard.update("x", 0, 16)
        easy.update("x", 16, 16)
    assert mean_temperature(hard, "x") > 1.3
    assert mean_temperature(easy, "x") < 0.75


def test_forgetting_lets_a_mastered_prompt_be_re_heated():
    c = at_pivot(0.5, gamma=0.5)
    for _ in range(10):
        c.update("x", 16, 16)
    before = mean_temperature(c, "x")
    for _ in range(10):
        c.update("x", 0, 16)
    assert mean_temperature(c, "x") > before


def test_thompson_draws_stay_within_the_rails():
    c = PTGS(tau=1.5, t_ref=0.8, seed=3)
    c.set_progress(100, 200)
    c.update("x", 2, 16)
    draws = [c.temperature(k) for k in ("x", "fresh") for _ in range(2000)]
    assert min(draws) >= 0.8 / 1.5 - 1e-12
    assert max(draws) <= 0.8 * 1.5 + 1e-12
    assert len(set(np.round(draws, 10))) > 1


def test_same_seed_same_stream():
    def run():
        c = PTGS(seed=7)
        out = []
        for _ in range(10):
            out.append(c.temperature("x"))
            c.update("x", 2, 16)
        return out

    assert run() == run()


def test_state_dict_round_trip_continues_the_same_draws():
    c = PTGS(seed=5)
    c.set_progress(120, 200)
    for i in range(6):
        c.update(f"t{i}", i, 8)
    _ = [c.temperature("t3") for _ in range(3)]
    state = c.state_dict()

    restored = PTGS(seed=999)
    restored.load_state_dict(state)
    assert restored.counts == c.counts
    assert (restored.step, restored.total_steps, restored.pivot) == (120, 200, c.pivot)
    assert [restored.temperature("t3") for _ in range(4)] == [c.temperature("t3") for _ in range(4)]


@pytest.mark.parametrize(
    "kwargs",
    [{"tau": 0.9}, {"t_ref": 0.0}, {"gamma": -0.1}, {"gamma": 1.5},
     {"pivot_start": 0.0}, {"pivot_start": 1.0}, {"pivot_end": 0.0}, {"pivot_end": 1.0}],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PTGS(**kwargs)


def test_config_record_ptgs_block_is_the_constructor_arguments():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "configs" / "ptgs_qwen2.5-7b.yaml").read_text())
    c = PTGS(**cfg["ptgs"])
    assert (c.tau, c.gamma, c.t_ref, c.pivot_start, c.pivot_end) == (1.5, 0.95, 1.0, 0.25, 0.5)


# -- trainer hooks --------------------------------------------------------------------------------
def test_hooks_end_to_end():
    hooks = PTGSHooks(pivot_start=0.5, pivot_end=0.5, seed=0)
    group_keys = {0: "task_a", 1: "task_b"}
    group_ids = [0] * 8 + [1] * 8

    hooks.on_step_start(1, 100)
    temps = hooks.assign_temperatures(group_keys)
    assert set(temps) == {0, 1}

    # each row carries its group's T: vLLM samples at it and the actor must score at it
    rows = hooks.temperature_for_rows(np.array(group_ids))
    assert rows.shape == (16,) and rows.dtype == np.float32
    assert np.allclose(rows, [temps[g] for g in group_ids])

    # group 0 fails everything, group 1 solves everything; a return of exactly 0.5 is a failure
    returns = [0.0, 0.2, 0.5, 0.0, 0.1, 0.3, 0.4, 0.0] + [1.0, 10.9, 0.6, 1.0, 2.0, 0.51, 5.0, 1.0]
    assert hooks.observe_group_outcomes(group_ids, returns) == 2
    assert hooks.controller.counts == {"task_a": [0.0, 8.0], "task_b": [8.0, 0.0]}

    m = hooks.metrics()
    assert m["ptgs/posterior_updates"] == 2
    assert m["ptgs/fallback_rows"] == 0
    assert m["ptgs/pivot"] == pytest.approx(0.5)
    assert m["ptgs/temp_min"] <= m["ptgs/temp_mean"] <= m["ptgs/temp_max"]

    hooks.on_step_start(2, 100)
    assert mean_temperature(hooks.controller, "task_a") > 1.0 > mean_temperature(hooks.controller, "task_b")


def test_hooks_threshold_binarises_shaped_returns():
    hooks = PTGSHooks(gamma=0.95)
    hooks.on_step_start(1, 10)
    hooks.assign_temperatures({0: "task_a"})
    hooks.observe_group_outcomes([0, 0, 0, 0], [0.0, 0.2, 0.9, 10.9])  # default: return > 0.5
    assert hooks.controller.counts["task_a"] == pytest.approx([2.0, 2.0])
    hooks.observe_group_outcomes([0, 0, 0, 0], [0.0, 0.2, 0.9, 10.9], threshold=5.0)
    assert hooks.controller.counts["task_a"] == pytest.approx([0.95 * 2 + 1, 0.95 * 2 + 3])
    assert hooks.metrics()["ptgs/posterior_updates"] == 2


def test_hooks_count_rows_that_fall_back_to_t_ref():
    hooks = PTGSHooks(t_ref=0.8)
    hooks.on_step_start(1, 10)
    hooks.assign_temperatures({0: "task_a"})
    rows = hooks.temperature_for_rows([0, 99, 99])
    assert rows[1:] == pytest.approx([0.8, 0.8])
    assert hooks.metrics()["ptgs/fallback_rows"] == 2
    assert hooks.observe_group_outcomes([99, 99], [1.0, 1.0]) == 0  # no key, no update

    # assignments and counters are per step
    hooks.on_step_start(2, 10)
    assert hooks.metrics()["ptgs/fallback_rows"] == 0
    assert hooks.temperature_for_rows([0])[0] == pytest.approx(0.8)
    assert hooks.metrics()["ptgs/fallback_rows"] == 1


def test_hooks_state_dict_round_trip_and_constructor_guard():
    hooks = PTGSHooks(seed=2)
    hooks.on_step_start(10, 200)
    hooks.assign_temperatures({0: "a"})
    hooks.observe_group_outcomes([0, 0], [1.0, 0.0])
    other = PTGSHooks(seed=3)
    other.load_state_dict(hooks.state_dict())
    assert other.controller.counts == hooks.controller.counts
    assert other.controller.temperature("a") == hooks.controller.temperature("a")
    with pytest.raises(ValueError):
        PTGSHooks(PTGS(), tau=1.2)


def test_score_function_has_zero_mean_only_at_the_sampling_temperature():
    """Why log-probs are scored at T_x: for a ~ softmax(z / T_x),
    E[grad_z log softmax(z / T)(a)] = p_x @ (I - p_T[None]) / T is zero only at T = T_x;
    at T = T_ref != T_x it is a drift that no advantage centring removes."""
    softmax = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()  # noqa: E731
    z = np.random.default_rng(0).normal(0.0, 2.0, size=8)
    eye, t_ref = np.eye(len(z)), 1.0

    def expected_score(t_x, T):
        p_x, p_T = softmax(z / t_x), softmax(z / T)
        return p_x @ (eye - p_T[None]) / T

    for t_x in (1 / 1.5, 1.2, 1.5):
        assert np.abs(expected_score(t_x, t_x)).sum() < 1e-12
        assert np.abs(expected_score(t_x, t_ref)).sum() > 1e-2
    assert np.abs(expected_score(t_ref, t_ref)).sum() < 1e-12


# -- entry points --------------------------------------------------------------------------------
def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_controller_main_runs():
    out = _run("-m", "ptgs")
    assert out.returncode == 0, out.stderr
    assert "step 200: 0.500" in out.stdout
    assert "RuntimeWarning" not in out.stderr


def test_toy_runs():
    out = _run("examples/ptgs_toy.py", "--seeds", "1", "--steps", "20")
    assert out.returncode == 0, out.stderr
    assert "Tax_S" in out.stdout and "seeds won (pass@1)" in out.stdout
