# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Collect N rollouts per task for one model on one benchmark.

    python -m sharpening_tax.run --benchmark bfcl --model google/gemma-4-31B --mode harness \\
        --base-url http://127.0.0.1:8000/v1 --out runs/bfcl/gemma-4-31B.base.jsonl

Writes one JSONL row per task (``task_id``, ``pass_flags``, ``rollouts``) and the inference
configuration to ``<out>.config.json``. Re-running resumes (see load_done). ``--shard i/n``
splits the tasks over processes; ``python -m sharpening_tax.outcomes merge`` joins shards, or
extra rollout batches of the same tasks (on BFCL and ACEBench, each needs a disjoint seed range:
rollout i sends ``--seed`` + i).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import uuid
from pathlib import Path

from tqdm import tqdm

from . import config
from .outcomes import mismatched_settings
from .policy import Policy


def load_done(out: Path, n: int) -> set:
    """Task ids already in ``out`` with ``n`` rollouts. Rewrites the file without a torn last
    line, moving rows with another rollout count (e.g. a smoke test) to ``<out>.dropped``."""
    if not out.exists():
        return set()
    with open(out, encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    keep, done, other = [], set(), []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise SystemExit(f"{out}: line {i + 1} is not valid JSON")
            print(f"{out}: dropping a torn last line")
            continue
        if len(row["pass_flags"]) != n:
            other.append(line if line.endswith("\n") else line + "\n")
            continue
        keep.append(line if line.endswith("\n") else line + "\n")
        done.add(row["task_id"])
    if other:
        dropped = out.with_name(out.name + ".dropped")
        with open(dropped, "a", encoding="utf-8") as f:
            f.write("".join(other))
        print(f"{out}: moved {len(other)} rows that do not have {n} rollouts to {dropped}")
    if len(keep) < len(lines) or (lines and not lines[-1].endswith("\n")):
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        tmp.replace(out)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=config.BENCHMARKS)
    ap.add_argument("--model", required=True, help="model id served by vLLM")
    ap.add_argument("--mode", required=True, choices=["harness", "chat"],
                    help="harness: base model via the text harness; chat: native chat template")
    ap.add_argument("--base-url", default=None, help="vLLM /v1 endpoint (default $VLLM_BASE_URL)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=config.N_ROLLOUTS, help="rollouts per task")
    ap.add_argument("--seed", type=int, default=42,
                    help="BFCL, ACEBench single-shot: rollout i sends seed + i (WebShop and ACEBench "
                         "agent_multi_step send none)")
    ap.add_argument("--workers", type=int, default=32, help="concurrent rollouts per task")
    ap.add_argument("--shard", default=None, help='"i/n": run tasks[i::n]')
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="first tasks only, for smoke tests (ACEBench: per category)")
    ap.add_argument("--categories", default=None,
                    help="ACEBench only: comma-separated categories (default: all 14)")
    ap.add_argument("--temperature", type=float, default=None, help="override the benchmark default")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--fence-stop", choices=["default", "closing"], default=None,
                    help="harness tool-call stop rule (default: per config.default_fence_stop)")
    ap.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                    help="JSON dict for the chat path (default: thinking off for Qwen3.5)")
    args = ap.parse_args()
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard i/n needs 0 <= i < n")

    categories = None
    if args.benchmark == "acebench":
        from .benchmarks.acebench import CATEGORIES
        wanted = set(args.categories.split(",")) if args.categories else set(CATEGORIES)
        if wanted - set(CATEGORIES):
            ap.error(f"unknown ACEBench categories {sorted(wanted - set(CATEGORIES))}; "
                     f"choose from {','.join(CATEGORIES)}")
        categories = [c for c in CATEGORIES if c in wanted]
    elif args.categories:
        ap.error("--categories applies to --benchmark acebench only")

    sampling = dict(config.SAMPLING[args.benchmark])
    for k in ("temperature", "top_p", "max_tokens"):
        if getattr(args, k) is not None:
            sampling[k] = getattr(args, k)
    policy = Policy(
        args.model, args.mode, args.base_url, **sampling,
        fence_stop=args.fence_stop or config.default_fence_stop(args.model, args.benchmark),
        chat_template_kwargs=(args.chat_template_kwargs if args.chat_template_kwargs is not None
                              else config.default_chat_template_kwargs(args.model)))
    run_id = uuid.uuid4().hex[:8]       # keeps per-rollout environment state apart

    if args.benchmark == "bfcl":
        from .benchmarks import bfcl as bench
        tasks = bench.load_tasks()
        task_id = lambda t: t["id"]  # noqa: E731
        run = lambda t: bench.run_task(policy, t, args.n, args.seed, args.workers, run_id)  # noqa: E731
    elif args.benchmark == "webshop":
        from .benchmarks import webshop as bench
        env = bench.make_env()
        tasks = bench.load_tasks()
        task_id = lambda t: t  # noqa: E731
        run = lambda t: bench.run_task(policy, env, t, args.n)  # noqa: E731
    else:
        from .benchmarks import acebench as bench
        tasks = bench.load_tasks(categories)
        task_id = lambda t: t["id"]  # noqa: E731
        run = lambda t: bench.run_task(policy, t, args.n, args.seed, args.workers, run_id)  # noqa: E731

    if args.max_tasks and args.benchmark == "acebench":
        by_category = {}
        for t in tasks:
            by_category.setdefault(t["category"], []).append(t)
        tasks = [t for group in by_category.values() for t in group[: args.max_tasks]]
    elif args.max_tasks:
        tasks = tasks[: args.max_tasks]
    if args.shard:
        tasks = tasks[i::n]

    env_seed = os.environ.get("WEBSHOP_ENV_SEED") if args.benchmark == "webshop" else None
    cfg = {"benchmark": args.benchmark, "policy": policy.config(), "n_rollouts": args.n,
           "seed": args.seed, "shard": args.shard, "max_tasks": args.max_tasks,
           "categories": categories, "webshop_env_seed": int(env_seed) if env_seed else None,
           "python": platform.python_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    cfg_path = Path(f"{args.out}.config.json")
    if args.out.exists() and cfg_path.exists():
        changed = mismatched_settings(json.loads(cfg_path.read_text()), json.loads(json.dumps(cfg)),
                                      keys=("benchmark", "seed", "webshop_env_seed"))
        if changed:
            raise SystemExit(f"{args.out} was collected with a different {', '.join(changed)} "
                             f"(see {cfg_path}): write this run to another --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out, args.n)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    todo = [t for t in tasks if task_id(t) not in done]
    print(f"{args.benchmark} | {args.model} ({args.mode}) | {len(todo)} tasks to run, "
          f"{len(tasks) - len(todo)} already done | n={args.n}")
    n_pass = n_tot = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for t in tqdm(todo, desc=f"{args.benchmark}/{args.mode}", ncols=100):
            row = run(t)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            n_pass, n_tot = n_pass + sum(row["pass_flags"]), n_tot + len(row["pass_flags"])
    if n_tot:
        print(f"mean success rate over this run's rollouts: {n_pass / n_tot:.4f}")


if __name__ == "__main__":
    main()
