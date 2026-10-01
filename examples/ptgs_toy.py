#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A tiny CPU-only RL loop for watching PTGS end to end in a few seconds.

A toy for intuition. For PTGS in an RL trainer for multi-turn LLM agents, see ptgs/integration.py
and configs/ptgs_qwen2.5-7b.yaml.

Each task is a chain of L decisions with A choices; a rollout succeeds only if all are
correct. Within a cluster most tasks share a canonical solution and a minority are
exceptions. The linear policy learns the canonical pattern by generalising across the
cluster, but an exception only from that task's own rare successes: one that never
succeeds in a group gets no gradient and is driven toward "never solved". That is the
bimodalization the Sharpening Tax measures and PTGS intervenes on.

ARMS (all trained arms share the lr tuned for the fixed arm's final pass@1)
  base     the initial policy, evaluated but never trained
  fixed    GRPO-style updates, every group sampled at T_ref                (the baseline)
  globalT  every group at one hotter T, by default PTGS's realised mean     (the control)
  ptgs     identical to fixed, but each group's temperature comes from the PTGS controller

globalT samples every group at one fixed, hotter temperature, heat-matched to PTGS by default,
so the two differ only in how heat is spread across prompts.

Run:  python examples/ptgs_toy.py                    (4 arms, 5 seeds)
      python examples/ptgs_toy.py --seeds 1 --verbose
      python examples/ptgs_toy.py --arms base,fixed,ptgs --lr 2.5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptgs.controller import PTGS  # noqa: E402
from sharpening_tax.metrics import pass_at_k, pass_hat_k, scalability  # noqa: E402


@dataclass
class ToyConfig:
    n_tasks: int = 96
    n_clusters: int = 8
    chain_length: int = 4  # L
    n_actions: int = 6  # A
    exception_rate: float = 0.30  # P(a decision deviates from its cluster's canonical action)
    task_code_dim: int = 24
    init_strength: float = 2.6  # initial logit bonus on each cluster's canonical action

    total_steps: int = 200
    tasks_per_step: int = 8
    group_size: int = 8
    lr: float = 6.0

    eval_rollouts: int = 128
    eval_temperature: float = 1.0


class ToyTasks:
    """Feature matrix, correct-action table, and the initial policy weights."""

    def __init__(self, cfg: ToyConfig, rng: np.random.Generator):
        self.cfg = cfg
        cluster = rng.integers(0, cfg.n_clusters, size=cfg.n_tasks)

        # phi(x) = [one-hot cluster | per-task code]; only the task block can encode an exception.
        onehot = np.eye(cfg.n_clusters)[cluster]
        code = rng.normal(0.0, 1.0, size=(cfg.n_tasks, cfg.task_code_dim))
        code /= np.linalg.norm(code, axis=1, keepdims=True)
        self.phi = np.concatenate([onehot, code], axis=1)  # (n_tasks, d)
        self.d = self.phi.shape[1]

        canonical = rng.integers(0, cfg.n_actions, size=(cfg.n_clusters, cfg.chain_length))
        self.correct = canonical[cluster].copy()  # (n_tasks, L)
        deviate = rng.random(self.correct.shape) < cfg.exception_rate
        self.correct[deviate] = rng.integers(0, cfg.n_actions, size=int(deviate.sum()))
        self.n_exceptions = (self.correct != canonical[cluster]).sum(axis=1)

        # Favouring the canonical actions gives a base-model-like spread of per-task success
        # rates: easy canonical tasks, very hard multi-exception ones.
        self.W0 = rng.normal(0.0, 0.05, size=(cfg.chain_length, cfg.n_actions, self.d))
        for c in range(cfg.n_clusters):
            for l in range(cfg.chain_length):
                self.W0[l, canonical[c, l], c] += cfg.init_strength


def logits(W: np.ndarray, phi_x: np.ndarray) -> np.ndarray:
    """(L, A) decision logits for one task."""
    return W @ phi_x


def softmax(z: np.ndarray, T: float) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def rollout_group(probs: np.ndarray, correct: np.ndarray, n: int, rng: np.random.Generator):
    """Sample ``n`` rollouts. Returns (actions (n, L), success (n,))."""
    L, A = probs.shape
    cdf = probs.cumsum(axis=-1)
    u = rng.random((n, L, 1))
    actions = (u > cdf[None, :, :]).sum(axis=-1).clip(0, A - 1)
    return actions, (actions == correct[None, :]).all(axis=1).astype(float)


def policy_gradient(probs, actions, adv, phi_x, temperature):
    """d/dW of mean_i adv_i * log pi_T(traj_i), via d/dz_l = (onehot(a) - softmax(z_l / T)) / T.

    ``temperature`` must be the group's sampling temperature, not T_ref (integration call site 4).
    """
    n, L = actions.shape
    A = probs.shape[1]
    onehot_weighted = np.zeros((L, A))
    for l in range(L):
        np.add.at(onehot_weighted[l], actions[:, l], adv)
    grad_z = (onehot_weighted - adv.sum() * probs) / (temperature * n)
    return grad_z[:, :, None] * phi_x[None, None, :]


def train(arm: str, tasks: ToyTasks, cfg: ToyConfig, seed: int, tau: float,
          global_temp: float = 1.0, verbose: bool = False):
    rng = np.random.default_rng(10_000 + seed)
    W = tasks.W0.copy()
    t_ref = 1.0

    ctrl = None
    if arm == "ptgs":
        ctrl = PTGS(tau=tau, gamma=0.95, t_ref=t_ref, pivot_start=0.25, pivot_end=0.5, seed=seed)
        arm_temp = None
    elif arm == "fixed":
        arm_temp = t_ref
    elif arm == "globalT":
        arm_temp = global_temp
    else:
        raise ValueError(f"unknown arm {arm!r}")

    trace = {"any_success": [], "mixed": [], "temp": []}
    for step in range(cfg.total_steps):
        if ctrl is not None:
            # 0-based here, so the toy's ramp stops one step short of pivot_end.
            ctrl.set_progress(step, cfg.total_steps)
        # A finite, cycled pool: the per-prompt posterior is only informative if tasks recur.
        lo = (step * cfg.tasks_per_step) % cfg.n_tasks
        batch = [(lo + j) % cfg.n_tasks for j in range(cfg.tasks_per_step)]

        step_any, step_mixed, step_temp = [], [], []
        for x in batch:
            key = f"task_{x}"
            T = ctrl.temperature(key) if ctrl is not None else arm_temp  # one T per group
            probs = softmax(logits(W, tasks.phi[x]), T)
            actions, success = rollout_group(probs, tasks.correct[x], cfg.group_size, rng)
            if ctrl is not None:
                ctrl.update(key, success.sum(), cfg.group_size)  # before the zero-variance skip

            adv = success - success.mean()  # group-relative advantage
            if np.any(adv):
                W += cfg.lr * policy_gradient(probs, actions, adv, tasks.phi[x], temperature=T)
            step_any.append(float(success.max()))
            step_mixed.append(float(0 < success.sum() < cfg.group_size))
            step_temp.append(T)

        trace["any_success"].append(np.mean(step_any))
        trace["mixed"].append(np.mean(step_mixed))
        trace["temp"].append(np.mean(step_temp))
        if verbose and (step + 1) % 50 == 0:
            pv = f"{ctrl.pivot:.3f}" if ctrl is not None else "  -- "
            print(f"    [{arm:>6}] step {step + 1:>3}  pivot={pv}  "
                  f"T={np.mean(step_temp):.3f}  groups with a success={np.mean(step_any):.2f}  "
                  f"mixed={np.mean(step_mixed):.2f}")
    return W, trace


def evaluate(W: np.ndarray, tasks: ToyTasks, cfg: ToyConfig, seed: int):
    """Every arm: same decoder, same seed, controller off."""
    rng = np.random.default_rng(777_000 + seed)
    counts = []
    for x in range(cfg.n_tasks):
        probs = softmax(logits(W, tasks.phi[x]), cfg.eval_temperature)
        _, success = rollout_group(probs, tasks.correct[x], cfg.eval_rollouts, rng)
        counts.append((int(success.sum()), cfg.eval_rollouts))
    return counts


def summarize(counts, ks, K):
    """pass@k for k in ks, pass^K and S(K) of one policy, from per-task (c, n) counts."""
    c, n = np.asarray(counts).T
    cov = pass_at_k(c, n, K).mean(axis=0)
    row = {f"pass@{k}": float(cov[k - 1]) for k in ks}
    row[f"pass^{K}"] = float(pass_hat_k(c, n, K).mean(axis=0)[K - 1])
    row["S"] = float(scalability(cov, K)[1][0])
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", default="base,fixed,globalT,ptgs")
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=None,
                    help="policy-gradient step size; default 6.0, tuned on the 'fixed' arm")
    ap.add_argument("--global-temp", type=float, default=None, dest="global_temp",
                    help="temperature of the globalT control; default = PTGS's own mean temperature")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = ToyConfig(total_steps=args.steps)
    if args.lr is not None:
        cfg.lr = args.lr
    arms = args.arms.split(",")
    K = cfg.eval_rollouts
    ks = [1, 4, 16, 64, K]

    # Heat-match globalT via a seed-0 PTGS probe run.
    global_temp = args.global_temp
    if global_temp is None and "globalT" in arms:
        _, probe = train("ptgs", ToyTasks(cfg, np.random.default_rng(0)), cfg, 0, args.tau)
        global_temp = float(np.mean(probe["temp"]))

    per_arm_counts: dict[str, list] = {a: [] for a in arms}
    per_arm_trace: dict[str, list] = {a: [] for a in arms}
    for seed in range(args.seeds):
        tasks = ToyTasks(cfg, np.random.default_rng(seed))
        if args.verbose:
            print(f"  seed {seed}: tasks with 0/1/2+ exception decisions = "
                  f"{(tasks.n_exceptions == 0).sum()}/{(tasks.n_exceptions == 1).sum()}/"
                  f"{(tasks.n_exceptions >= 2).sum()}")
        for arm in arms:
            if arm == "base":
                W = tasks.W0
            else:
                W, trace = train(arm, tasks, cfg, seed, args.tau, global_temp or 1.0, args.verbose)
                per_arm_trace[arm].append(trace)
            per_arm_counts[arm].append(evaluate(W, tasks, cfg, seed))

    print(f"\n{'=' * 96}")
    print(f"Toy RL: {cfg.n_tasks} tasks, chain of {cfg.chain_length} x {cfg.n_actions} choices, "
          f"{cfg.total_steps} steps x {cfg.tasks_per_step} tasks x {cfg.group_size} rollouts")
    print(f"Evaluated at T={cfg.eval_temperature} with {K} rollouts/task; "
          f"mean +- sd over {args.seeds} seeds; lr={cfg.lr}, tau={args.tau}"
          + (f", globalT={global_temp:.3f}" if global_temp else ""))
    print("=" * 96)

    header = f"{'arm':<16}" + "".join(f"{f'pass@{k}':>13}" for k in ks) + f"{f'pass^{K}':>13}{'Tax_S':>13}"
    print(header)
    print("-" * len(header))

    rows = {a: [summarize(c, ks, K) for c in per_arm_counts[a]] for a in arms}
    for arm in arms:
        per_seed = rows[arm]
        # Tax_S(K) = S_base(K) - S_arm(K), per seed, on the same tasks.
        taxes = [b["S"] - r["S"] for b, r in zip(rows["base"], per_seed)] \
            if arm != "base" and "base" in arms else None
        cells = ""
        for k in ks:
            v = np.array([s[f"pass@{k}"] for s in per_seed]) * 100
            cells += f"{v.mean():>8.1f}+-{v.std():<4.1f}"
        v = np.array([s[f"pass^{K}"] for s in per_seed]) * 100
        cells += f"{v.mean():>8.1f}+-{v.std():<4.1f}"
        if taxes is None:
            cells += f"{'--':>13}"
        else:
            t = np.array(taxes)
            cells += f"{t.mean():>8.3f}+-{t.std():<4.3f}"
        print(f"{arm:<16}{cells}")

    if any(per_arm_trace[a] for a in arms):
        print(f"\n{'-' * 96}\nTraining-time group statistics (mean over seeds, last 50 steps)")
        print(f"{'arm':<16}{'groups with >=1 success':>26}{'mixed groups':>16}{'mean T':>10}")
        for arm in arms:
            if not per_arm_trace[arm]:
                continue
            anyv = np.mean([np.mean(t["any_success"][-50:]) for t in per_arm_trace[arm]])
            mixv = np.mean([np.mean(t["mixed"][-50:]) for t in per_arm_trace[arm]])
            tv = np.mean([np.mean(t["temp"][-50:]) for t in per_arm_trace[arm]])
            print(f"{arm:<16}{anyv:>26.3f}{mixv:>16.3f}{tv:>10.3f}")

    # Seeds differ mostly in task-pool difficulty, shared by every arm: the paired difference
    # removes it, so the across-seed sd above can exceed a gap that is consistent across seeds.
    if "ptgs" in arms and "base" in arms:
        print(f"\n{'-' * 96}\nPaired per-seed difference vs PTGS (positive = PTGS ahead on that seed)")
        print(f"{'vs arm':<16}{'d pass@1':>24}{'d pass@' + str(K):>18}{'seeds won (pass@1)':>21}")
        ptgs_rows = rows["ptgs"]
        for arm in arms:
            if arm in ("base", "ptgs"):
                continue
            d1 = np.array([p["pass@1"] - o["pass@1"] for p, o in zip(ptgs_rows, rows[arm])]) * 100
            dk = np.array([p[f"pass@{K}"] - o[f"pass@{K}"] for p, o in zip(ptgs_rows, rows[arm])]) * 100
            won = int((d1 > 0).sum())
            print(f"{arm:<16}{d1.mean():>16.1f}+-{d1.std():<6.1f}"
                  f"{dk.mean():>10.1f}+-{dk.std():<6.1f}{won:>19}/{len(d1)}")


if __name__ == "__main__":
    main()
