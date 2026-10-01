# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wiring PTGS into a group-based RL trainer: four call sites, nothing in the loss.

``PTGSHooks`` is the framework-agnostic half; the snippets show the rest for a verl-style
trainer (driver + vLLM rollout worker + sharded actor) such as RAGEN
(https://github.com/mll-lab-nu/RAGEN, MIT) on verl (https://github.com/volcengine/verl, Apache-2.0).

1. Advance the pivot at the top of each training step, with 1-indexed steps.

    for step in range(1, total_steps + 1):
        hooks.on_step_start(step, total_steps)
        batch = agent_proxy.rollout(batch)                  # call site 2 runs in here
        hooks.observe_group_outcomes(group_ids, returns)    # call site 3, full batch
        batch.meta_info["ptgs_correct_logprob"] = True      # call site 4 checks this flag
        batch.batch["ptgs_temperature"] = torch.as_tensor(
            hooks.temperature_for_rows(group_ids), dtype=torch.float32)
        batch = rollout_filter(batch)                       # e.g. StarPO-S, only after 3
        metrics.update(hooks.metrics())
        ...unchanged advantage estimation and policy update...

2. Draw one temperature per prompt group in the rollout driver at the first turn and hold it
   for every turn of every rollout in the group; it reaches vLLM as per-request
   SamplingParams. Training only: validation keeps the fixed evaluation temperature.

    if not val:
        if turn == 0:
            # Key by a task identity that recurs (env seed / dataset uid), never the prompt text.
            hooks.assign_temperatures({env.group_id: f"seed_{env.seed}" for env in envs})
        lm_inputs.non_tensor_batch["ptgs_temperature"] = hooks.temperature_for_rows(env_ids // group_size)

    # vLLM rollout worker
    temps = non_tensor_batch.pop("ptgs_temperature", None)
    if temps is not None and do_sample and not is_validate:
        params = [copy.deepcopy(self.sampling_params) for _ in temps]
        for sp, t in zip(params, temps):
            sp.temperature = float(t)
    else:
        params = self.sampling_params                       # the baseline path, unchanged
    outputs = self.inference_engine.generate(prompts=vllm_inputs, sampling_params=params)

3. Fold group outcomes (success = return > 0.5) into the posterior on the FULL batch, BEFORE
   any rollout filter. StarPO-S never keeps a zero-variance group, so it drops every all-fail
   group -- exactly the prompts PTGS must heat, which would otherwise never leave the prior.

4. Score log-probs at the sampling temperature: the actor divides logits row-wise by T_x, and
   a missing per-row temperature is a hard error, since a field dropped in transit (say, by a
   stale select_keys list) silently restores the scalar path.

    row_T = micro_batch.get("ptgs_temperature")             # (B,)
    if row_T is None and data.meta_info.get("ptgs_correct_logprob"):
        raise RuntimeError("ptgs_temperature was dropped between the driver and the actor")
    logits = self.actor_module(...).logits
    if row_T is None:
        logits.div_(temperature)                            # batch scalar, baseline
    else:
        logits.div_(row_T.to(logits.device).view(-1, 1, 1).to(logits.dtype))

   Scoring at T_ref instead would make every PTGS update off-policy with no importance weight,
   worst on the prompts PTGS moves furthest. The entropy bonus and logged entropy are then at
   T_x too, so compare entropies across arms at a common temperature.

Also required: the posterior key must recur, so training cycles a FINITE task pool (e.g.
200 tasks, 8 per step, so each recurs every 25 steps). Stock RAGEN never repeats a
training seed, so every group would be drawn from the prior.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .controller import PTGS


class PTGSHooks:
    """Framework-agnostic half, around one :class:`PTGS`. Holds this step's group id ->
    (task key, temperature) map, so any stage can recover a row's temperature from its group
    id. Group ids are per step; task keys are stable across steps."""

    def __init__(self, controller: Optional[PTGS] = None, success_threshold: float = 0.5, **ptgs_kwargs):
        if controller is None:
            controller = PTGS(**ptgs_kwargs)
        elif ptgs_kwargs:
            raise ValueError("pass either a PTGS instance or its constructor arguments, not both")
        self.controller = controller
        self.success_threshold = success_threshold
        self._group_key: dict[int, str] = {}
        self._group_temp: dict[int, float] = {}
        self._updates = 0
        self._fallback_rows = 0

    def on_step_start(self, step: int, total_steps: int) -> None:
        """Call site 1, with step = 1..total_steps."""
        self.controller.set_progress(step, total_steps)
        self._group_key, self._group_temp = {}, {}
        self._updates = self._fallback_rows = 0

    def assign_temperatures(self, group_keys: Mapping[int, str]) -> dict[int, float]:
        """Call site 2: one temperature per group, given group id -> stable task key."""
        self._group_key = {int(g): str(k) for g, k in group_keys.items()}
        self._group_temp = {g: self.controller.temperature(k) for g, k in self._group_key.items()}
        return dict(self._group_temp)

    def temperature_for_rows(self, group_ids: Iterable[int]) -> np.ndarray:
        """Rows of an unassigned group get T_ref and count toward ``ptgs/fallback_rows``;
        nonzero there means group ids and assigned groups have come apart."""
        out = []
        for g in np.asarray(list(group_ids)).ravel().tolist():
            t = self._group_temp.get(int(g))
            if t is None:
                self._fallback_rows += 1
                t = self.controller.t_ref
            out.append(t)
        return np.asarray(out, dtype=np.float32)

    def observe_group_outcomes(
        self, group_ids: Sequence[int], returns: Sequence[float], threshold: Optional[float] = None
    ) -> int:
        """Call site 3: success = return > threshold; returns the number of groups updated.
        Pass one entry per trajectory (dedupe per-turn rows), full batch, before any filter."""
        thr = self.success_threshold if threshold is None else threshold
        gids = np.asarray(group_ids).ravel().tolist()
        rets = np.asarray(returns, dtype=float).ravel().tolist()
        if len(gids) != len(rets):
            raise ValueError(f"{len(gids)} group ids but {len(rets)} returns")
        succ: dict[int, float] = {}
        tot: dict[int, int] = {}
        for g, r in zip(gids, rets):
            g = int(g)
            succ[g] = succ.get(g, 0.0) + float(r > thr)
            tot[g] = tot.get(g, 0) + 1
        n_updated = 0
        for g, n in tot.items():
            key = self._group_key.get(g)
            if key is not None:
                self.controller.update(key, succ[g], n)
                n_updated += 1
        self._updates += n_updated
        return n_updated

    def metrics(self) -> dict:
        out = {
            "ptgs/pivot": float(self.controller.pivot),
            "ptgs/posterior_updates": float(self._updates),
            "ptgs/fallback_rows": float(self._fallback_rows),
        }
        if self._group_temp:
            t = np.fromiter(self._group_temp.values(), dtype=float)
            out.update({"ptgs/temp_mean": float(t.mean()), "ptgs/temp_min": float(t.min()),
                        "ptgs/temp_max": float(t.max())})
        return out

    def state_dict(self) -> dict:
        return self.controller.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.controller.load_state_dict(state)
