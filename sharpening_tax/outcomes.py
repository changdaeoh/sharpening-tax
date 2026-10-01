# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Outcome files: per-task success counts, merging runs, and the counts table.

    python -m sharpening_tax.outcomes merge runs/a.jsonl runs/b.jsonl -o runs/ab.jsonl
    python -m sharpening_tax.outcomes collect runs/ -o my_outcomes.csv.gz

``merge`` joins shards or seed-disjoint rollout batches (pass_flags concatenated). ``collect``
writes one row per (benchmark, pair, arm, task) with columns benchmark, pair, arm, model, task_id,
n, c (read it back with ``load_table``).
Arms: ``base`` (base model, harness) and ``rl`` (post-trained model, native chat);
``base_noharness`` (base model, chat) and ``rl_harness`` (post-trained model, harness); a non-default temperature
appends ``_T<temperature>`` (e.g. ``rl_T1.3``).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

COLUMNS = ["benchmark", "pair", "arm", "model", "task_id", "n", "c"]
SEEDED = ("bfcl", "acebench")       # rollout i of a task sends seed + i (ACEBench: single-shot only)


def load_counts(path) -> dict:
    """task_id -> (c, n) from an outcome JSONL (rows with ``pass_flags`` or ``n``/``c``)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                tid = str(row["task_id"])
                if tid in out:
                    raise ValueError(f"{path}: task {tid} has more than one row")
                if "pass_flags" in row:
                    out[tid] = (int(sum(map(bool, row["pass_flags"]))), len(row["pass_flags"]))
                else:
                    out[tid] = (int(row["c"]), int(row["n"]))
    return out


def load_table(path) -> dict:
    """{(benchmark, pair, arm): {task_id: (c, n)}} from a counts table (.csv or .csv.gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    cells = defaultdict(dict)
    with opener(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            cells[(r["benchmark"], r["pair"], r["arm"])][r["task_id"]] = (int(r["c"]), int(r["n"]))
    return dict(cells)


def mismatched_settings(a: dict, b: dict, keys=("benchmark", "webshop_env_seed")) -> list[str]:
    """Settings (``keys`` + every policy field) in which two ``<out>.config.json`` dicts differ."""
    return [k for k in keys if a.get(k) != b.get(k)] + [
        f"policy.{k}" for k in sorted(set(a.get("policy") or {}) | set(b.get("policy") or {}))
        if (a.get("policy") or {}).get(k) != (b.get("policy") or {}).get(k)]


def _seed_ranges(cfg: dict, n: int) -> list[tuple[int, int]]:
    """[start, end) of the request seeds behind a row of ``n`` rollouts."""
    if cfg.get("seed_ranges"):      # a merged file: the union over its sources (conservative)
        return [tuple(r) for r in cfg["seed_ranges"]]
    return [(cfg["seed"], cfg["seed"] + n)]


def merge(paths: list[Path], out: Path) -> None:
    rows: dict[str, dict] = {}
    used: dict[str, list] = {}      # task_id -> [(source, seed ranges)]
    cfgs, all_ranges = [], set()
    for p in paths:
        cfg_path = Path(f"{p}.config.json")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
        if cfg is None:
            print(f"{p}: no config.json, so its settings and seeds are not checked")
        else:
            diff = mismatched_settings(cfgs[0], cfg) if cfgs else []
            if diff:
                raise SystemExit(f"{p} differs from the first input in {', '.join(diff)}: only "
                                 "runs of one model with one configuration can be merged")
            cfgs.append(cfg)
        seeded = cfg is not None and cfg["benchmark"] in SEEDED and cfg.get("seed") is not None

        file_rows, n_lines = {}, 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    file_rows[r["task_id"]] = r         # the last row of a task wins
                    n_lines += 1
        if n_lines > len(file_rows):
            print(f"{p}: {n_lines - len(file_rows)} stale rows of repeated tasks ignored")

        for tid, r in file_rows.items():
            ranges = _seed_ranges(cfg, len(r["pass_flags"])) if seeded else []
            for q, other in used.get(tid, []):
                if any(a < d and c < b for a, b in ranges for c, d in other):
                    raise SystemExit(
                        f"task {tid} of {p} and {q} were sampled with the same request seeds: "
                        "give each extra rollout batch a disjoint --seed (e.g. --seed 42, then "
                        "--seed 170 for two batches of --n 128)")
            used.setdefault(tid, []).append((p, ranges))
            all_ranges.update(ranges)
            if tid in rows:
                rows[tid]["pass_flags"] += r["pass_flags"]
                rows[tid].setdefault("rollouts", []).extend(r.get("rollouts", []))
            else:
                rows[tid] = r
    with open(out, "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    cfg = dict(cfgs[0]) if cfgs else {}
    cfg.update(sources=[str(p) for p in paths], shard=None,
               n_rollouts=sorted({len(r["pass_flags"]) for r in rows.values()}))
    if all_ranges:
        cfg["seed_ranges"] = [list(r) for r in sorted(all_ranges)]
    Path(f"{out}.config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{out}: {len(rows)} tasks, rollouts per task {cfg['n_rollouts']}")


def collect(run_dir: Path, out: Path) -> None:
    from .config import ARM_OF, SAMPLING
    rows, seen = [], {}
    for cfg_path in sorted(run_dir.rglob("*.jsonl.config.json")):
        cfg = json.loads(cfg_path.read_text())
        if "policy" not in cfg:
            print(f"skip {cfg_path.name}: no run settings (merged from files without config.json)")
            continue
        model, mode = cfg["policy"]["model"], cfg["policy"]["mode"]
        if cfg.get("shard"):
            print(f"skip {cfg_path.name}: a shard (merge the shards and collect the merged file)")
            continue
        if model not in ARM_OF:
            print(f"skip {cfg_path.name}: {model} is not in config.PAIRS")
            continue
        pair, arm = ARM_OF[model]
        if arm == "base" and mode == "chat":
            arm = "base_noharness"
        elif arm == "rl" and mode == "harness":
            arm = "rl_harness"
        temperature = cfg["policy"]["temperature"]
        if temperature != SAMPLING[cfg["benchmark"]]["temperature"]:
            arm += f"_T{temperature}"
        for tid, (c, n) in load_counts(str(cfg_path)[: -len(".config.json")]).items():
            key = (cfg["benchmark"], pair, arm, tid)
            if key in seen:
                raise SystemExit(f"{key} appears in both {seen[key]} and {cfg_path}: merge rollout "
                                 "batches into one file and keep only the merged file in the directory")
            seen[key] = cfg_path
            rows.append({"benchmark": cfg["benchmark"], "pair": pair, "arm": arm,
                         "model": model, "task_id": tid, "n": n, "c": c})
    with gzip.open(out, "wt", newline="") if str(out).endswith(".gz") else open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} task rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="join shards or rollout batches of one model on one benchmark")
    m.add_argument("inputs", type=Path, nargs="+")
    m.add_argument("-o", "--out", type=Path, required=True)
    c = sub.add_parser("collect", help="build the counts table from every run in a directory")
    c.add_argument("run_dir", type=Path)
    c.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    merge(args.inputs, args.out) if args.cmd == "merge" else collect(args.run_dir, args.out)


if __name__ == "__main__":
    main()
