# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""WebShop, first 500 goals over a 1,000-product index.

One ``search[...]`` / ``click[...]`` action per turn, at most 30; success iff the final reward
is 1.0 (a fully matching purchase). Prompt-based (no ``tools=``), so the harness path serves
the same prompt without a chat template.

Needs https://github.com/princeton-nlp/WebShop @ 64fa2a5 (the tested commit) with its
1,000-product index built (``setup.sh -d small``) and $WEBSHOP_DIR at it. The env is not
thread-safe: rollouts run sequentially, so parallelise with ``--shard i/n`` + ``outcomes merge``.
"""
from __future__ import annotations

import os
import random
import re
import sys

from ..policy import server_unreachable

N_GOALS = 500
MAX_STEPS = 30
NUM_PRODUCTS = 1000

SYSTEM_MESSAGE = (
    "You are a shopping agent operating on a simplified e-commerce site. "
    "Given a shopping instruction, interact with the environment step by "
    "step to find and purchase the product that best matches the instruction.\n\n"
    "At each turn you will see:\n"
    "  * the current page (text observation, with `[SEP]` separating UI chunks),\n"
    "  * the set of available actions.\n\n"
    "Respond with EXACTLY one action on a single line, using one of:\n"
    "  search[<keywords>]     — type into the search bar (only when a search "
    "bar is available).\n"
    "  click[<button text>]   — click the given button/link/option. The text "
    "must match one of the listed clickable elements.\n\n"
    "Do not output reasoning, explanations, or any text other than the action. "
    "Finish by clicking `Buy Now` when you are ready to purchase."
)

_ACTION_RE = re.compile(r"(search|click)\s*\[\s*(.+?)\s*\]", re.IGNORECASE)


def make_env():
    sys.path.insert(0, os.environ.get("WEBSHOP_DIR") or sys.exit("set WEBSHOP_DIR to the WebShop checkout"))
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the gym env)
    # Product prices and goal price caps come from the global RNG before WebShop's fixed-seed
    # goal shuffle, so unseeded processes see different constraints per goal. WEBSHOP_ENV_SEED
    # fixes them.
    if os.environ.get("WEBSHOP_ENV_SEED"):
        random.seed(int(os.environ["WEBSHOP_ENV_SEED"]))
    return gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=NUM_PRODUCTS,
                    human_goals=False, disable_env_checker=True)


def load_tasks() -> list[str]:
    return [f"goal_{i}" for i in range(N_GOALS)]


def _observation(obs: str, avail: dict, instruction: str | None = None) -> str:
    clickables = [c for c in (avail.get("clickables") or []) if str(c).lower() != "search"]
    actions = []
    if avail.get("has_search_bar"):
        actions.append("Search bar is available: emit `search[<keywords>]`.")
    if clickables:
        actions.append("Clickable buttons: " + ", ".join(f"`{c}`" for c in clickables))
    if not actions:
        actions.append("No available actions (the episode should terminate).")
    blocks = [f"Observation:\n{obs}", "Available actions:\n" + "\n".join(actions)]
    if instruction:
        blocks.insert(0, f"Instruction: {instruction}")
    return "\n\n".join(blocks)


def parse_action(text: str, avail: dict) -> str | None:
    """``search[...]`` / ``click[...]``; otherwise a bare button name (e.g. "Buy Now")."""
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if m:
        arg = m.group(2).strip().strip("`\"'")
        if arg:
            return f"{m.group(1).lower()}[{arg}]"
    clickables = {str(c).lower(): str(c) for c in (avail.get("clickables") or [])}
    strip = "`\"'.!,;: \n\t"
    candidates = [text.strip().strip(strip).lower()] + [l.strip().strip(strip).lower() for l in text.splitlines()]
    for cand in candidates:
        if cand in clickables:
            return f"click[{clickables[cand]}]"
    return None


def rollout(policy, env, goal_idx: int) -> dict:
    # No per-request seed: rollouts are independent draws (a fixed per-request seed makes
    # vLLM's sampling depend on how requests are batched).
    obs, _ = env.reset(session=goal_idx)
    avail = env.get_available_actions()
    messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": _observation(obs, avail, env.unwrapped.instruction_text)}]
    reward, error = 0.0, None
    for _ in range(MAX_STEPS):
        try:
            text = policy(messages).text.strip()
        except Exception as e:
            if server_unreachable(e):
                raise
            error = f"generation error: {e}"
            break
        messages.append({"role": "assistant", "content": text})
        action = parse_action(text, avail)
        if action is None:
            break
        try:
            obs, reward, done, _ = env.step(action)
        except Exception as e:
            error = f"env error: {e}"
            break
        if done:
            break
        avail = env.get_available_actions()
        messages.append({"role": "user", "content": _observation(obs, avail)})
    return {"messages": messages, "final_reward": float(reward),
            "passed": float(reward) >= 1.0 - 1e-6, "error": error}


def run_task(policy, env, task_id: str, n: int) -> dict:
    trajs = [rollout(policy, env, int(task_id.split("_")[1])) for _ in range(n)]
    return {"task_id": task_id, "pass_flags": [t["passed"] for t in trajs], "rollouts": trajs}
