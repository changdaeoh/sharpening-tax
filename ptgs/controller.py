# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PTGS (posterior-tempered group sampling) in one file.

Numpy only. Each prompt keeps a discounted Beta posterior over its success rate, fed by the
success counts of its rollout groups; a Thompson draw p_hat_x sets the temperature of its next
group, above T_ref when p_hat_x is below the pivot and below T_ref when above. Nothing else in
the RL algorithm changes, provided the policy-gradient log-probs are scored at that same
temperature (ptgs/integration.py).

Notation: pivot p~ (target success rate), temperature spread tau >= 1 (T_x in
[T_ref/tau, tau*T_ref]), forgetting factor gamma, reference temperature T_ref.
"""

from __future__ import annotations

import numpy as np


# Geometric pivot ramp, by default 0.25 -> 0.5. Even at p = 0.25 a
# group of 16 is informative ~99% of the time (degenerate w.p. 0.25^16 + 0.75^16 ~ 0.01).
def pivot_at(step: int, total_steps: int, pivot_start: float = 0.25, pivot_end: float = 0.5) -> float:
    u = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return pivot_start * (pivot_end / pivot_start) ** u


# Tempering rule: log T piecewise linear in p_hat through T(0) = tau*T_ref, T(p~) = T_ref and
# T(1) = T_ref/tau, for any pivot; at p~ = 1/2, T = T_ref * tau^(1-2p).
def temperature_from_p(p_hat: float, pivot: float, tau: float, t_ref: float = 1.0) -> float:
    p_hat = min(max(p_hat, 0.0), 1.0)
    h = (pivot - p_hat) / (pivot if p_hat <= pivot else 1.0 - pivot)
    return t_ref * tau**h


class PTGS:
    """Per-prompt discounted Beta posterior + the tempering rule. Key prompts by a stable task id (dataset
    key / env seed), never the prompt text, which drifts with the observation shown. Not
    thread-safe: owned by the one process that forms rollout groups."""

    def __init__(self, tau=1.5, gamma=0.95, t_ref=1.0, pivot_start=0.25, pivot_end=0.5, seed=1):
        if tau < 1.0:
            raise ValueError(f"tau must be >= 1, got {tau}")
        if t_ref <= 0.0:
            raise ValueError(f"t_ref must be > 0, got {t_ref}")
        # gamma = 1 means no forgetting.
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        for name, val in (("pivot_start", pivot_start), ("pivot_end", pivot_end)):
            if not 0.0 < val < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {val}")
        self.tau, self.gamma, self.t_ref = tau, gamma, t_ref
        self.pivot_start, self.pivot_end = pivot_start, pivot_end
        self.counts: dict[str, list[float]] = {}  # key -> [s~_x, f~_x], discounted
        self.rng = np.random.default_rng(seed)
        self.step, self.total_steps = 0, None
        self.pivot = pivot_start

    def set_progress(self, step: int, total_steps: int) -> None:
        """Advance the pivot ramp. Call once per RL step, with step = 1..total_steps."""
        self.step, self.total_steps = step, total_steps
        self.pivot = pivot_at(step, total_steps, self.pivot_start, self.pivot_end)

    def _beta(self, key: str) -> tuple[float, float]:
        # Prior of mass 2 centred on the current pivot. An unobserved
        # prompt's estimate sits at the pivot, but its Thompson temperature is not neutral
        # unless p~ = 1/2: for p~ < 1/2 the heating branch of the rule is steeper (tau=1.5,
        # p~=0.25: mean T ~1.15, ~61% of draws above T_ref). A fixed Beta(1,1) prior would
        # instead cool it (mean T ~0.92).
        s, f = self.counts.get(key, (0.0, 0.0))
        return 2.0 * self.pivot + s, 2.0 * (1.0 - self.pivot) + f

    def posterior_mean(self, key: str) -> float:
        a, b = self._beta(key)
        return a / (a + b)

    def temperature(self, key: str) -> float:
        """Temperature for the next group of prompt ``key`` (one Thompson draw)."""
        p_hat = self.rng.beta(*self._beta(key))  # Thompson sampling
        return temperature_from_p(p_hat, self.pivot, self.tau, self.t_ref)

    def update(self, key: str, successes: float, n: float) -> None:
        """Fold a group's outcome into the discounted counts."""
        if n <= 0:
            return
        s, f = self.counts.get(key, (0.0, 0.0))
        self.counts[key] = [self.gamma * s + successes, self.gamma * f + (n - successes)]

    def state_dict(self) -> dict:
        """Includes the RNG state, so a resumed run continues the same draws on the same ramp."""
        return {
            "counts": {k: [float(s), float(f)] for k, (s, f) in self.counts.items()},
            "rng": self.rng.bit_generator.state,
            "step": self.step,
            "total_steps": self.total_steps,
            "pivot": self.pivot,
        }

    def load_state_dict(self, state: dict) -> None:
        self.counts = {k: [float(s), float(f)] for k, (s, f) in state["counts"].items()}
        self.rng.bit_generator.state = state["rng"]
        self.step, self.total_steps, self.pivot = state["step"], state["total_steps"], state["pivot"]


# PTGS in an RL loop; ptgs/integration.py gives the four call sites in a real trainer.
def rl_loop_sketch(task_pool, policy, rl_update, total_steps=200, group_size=16, batch_tasks=8):
    ptgs = PTGS(tau=1.5, gamma=0.95, t_ref=1.0, pivot_start=0.25, pivot_end=0.5)

    for step in range(1, total_steps + 1):  # 1-indexed, so the ramp ends exactly at pivot_end
        ptgs.set_progress(step, total_steps)
        batch = task_pool.next_batch(batch_tasks)  # a finite pool, cycled so every task recurs

        temps = {x: ptgs.temperature(x) for x in batch}  # one T per prompt group
        groups = {x: policy.generate(x, n=group_size, temperature=temps[x]) for x in batch}

        for x, group in groups.items():  # every group, before any rollout filter
            ptgs.update(x, successes=sum(g.success for g in group), n=group_size)

        rl_update(groups, log_probs=policy.log_probs(groups, temperature=temps))
