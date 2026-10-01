# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BFCL v4, ``multi_turn_base`` category (200 tasks).

BFCL's multi-turn protocol: at most MAXIMUM_STEP_LIMIT (20) tool-call rounds per user turn,
calls executed in BFCL's stateful Python environment, and the trajectory graded by BFCL's
state-based ``multi_turn_checker``; a rollout passes iff it returns ``valid``.

Needs https://github.com/ShishirPatil/gorilla @ 6ea5797 (the tested commit), with $BFCL_ROOT
at its ``berkeley-function-call-leaderboard`` directory.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import sys
import types
from concurrent.futures import ThreadPoolExecutor

from ..policy import ToolParserMismatch, server_unreachable

CATEGORY = "multi_turn_base"
GRADE_TIMEOUT_S = 180   # a hung checker counts as a failure

_bfcl = None


def _import_bfcl():
    global _bfcl
    if _bfcl is not None:
        return _bfcl
    root = os.environ.get("BFCL_ROOT")
    if not root:
        raise SystemExit("set BFCL_ROOT to gorilla/berkeley-function-call-leaderboard")
    sys.path.insert(0, root)
    # java/js parsers (unused by multi-turn) fail to import with some tree_sitter versions.
    for mod, fn in (("bfcl_eval.model_handler.parser.java_parser", "parse_java_function_call"),
                    ("bfcl_eval.model_handler.parser.js_parser", "parse_javascript_function_call")):
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            setattr(stub, fn, lambda *a, **k: None)
            sys.modules[mod] = stub
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions
    _bfcl = types.SimpleNamespace(
        data_dir=os.path.join(root, "bfcl_eval", "data"), MAXIMUM_STEP_LIMIT=MAXIMUM_STEP_LIMIT,
        populate_test_cases_with_predefined_functions=populate_test_cases_with_predefined_functions,
        convert_to_tool=convert_to_tool, GORILLA_TO_OPENAPI=GORILLA_TO_OPENAPI,
        ModelStyle=ModelStyle, convert_to_function_call=convert_to_function_call,
        multi_turn_utils=multi_turn_utils, multi_turn_checker=multi_turn_checker)
    return _bfcl


def load_tasks() -> list[dict]:
    b = _import_bfcl()
    read = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731
    tasks = read(os.path.join(b.data_dir, f"BFCL_v4_{CATEGORY}.json"))
    answers = {e["id"]: e["ground_truth"]
               for e in read(os.path.join(b.data_dir, "possible_answer", f"BFCL_v4_{CATEGORY}.json"))}
    tasks = b.populate_test_cases_with_predefined_functions(tasks)
    for t in tasks:
        t["ground_truth"] = answers.get(t["id"], [])
    return tasks


def _assistant_message(text: str, tool_calls: list[dict]) -> dict:
    if tool_calls:
        return {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
    return {"role": "assistant", "content": text or ""}


def rollout(policy, task: dict, tools: list[dict], state_key: str, seed: int | None) -> dict:
    """One trajectory through all user turns of a task. ``state_key`` isolates BFCL's
    per-rollout environment state (BFCL keys instances by model name + task id)."""
    b = _import_bfcl()
    messages, decoded_per_turn, error = [], [], None
    for turn in task["question"]:
        messages.extend(turn)
        steps = []
        for _ in range(b.MAXIMUM_STEP_LIMIT):
            try:
                reply = policy(messages, tools, seed=seed)
            except ToolParserMismatch as e:
                # The whole rollout fails: nothing is graded.
                return {"messages": messages, "decoded_steps_per_turn": None,
                        "error": f"tool-call parser mismatch: {e}"}
            except Exception as e:
                if server_unreachable(e):
                    raise
                error = f"generation error: {e}"    # the trajectory so far is graded
                break
            messages.append(_assistant_message(reply.text, reply.tool_calls))
            decoded = b.convert_to_function_call(
                [{tc["function"]["name"]: tc["function"]["arguments"]} for tc in reply.tool_calls]
            ) if reply.tool_calls else []
            if not decoded:        # no tool call: the model ended its turn
                break
            steps.append(decoded)
            try:
                results, _ = b.multi_turn_utils.execute_multi_turn_func_call(
                    func_call_list=decoded, initial_config=task.get("initial_config", {}),
                    involved_classes=task.get("involved_classes", []), model_name=state_key,
                    test_entry_id=task["id"], long_context=False, is_evaL_run=False)
            except Exception as e:
                results = [f"Execution error: {e}"] * len(decoded)
            for tc, res in zip(reply.tool_calls, results):
                messages.append({"role": "tool", "content": str(res), "tool_call_id": tc["id"]})
        else:
            error = "step limit reached"
        decoded_per_turn.append(steps)
        if error:
            break
    decoded_per_turn += [[] for _ in range(len(task["ground_truth"]) - len(decoded_per_turn))]
    return {"messages": messages, "decoded_steps_per_turn": decoded_per_turn, "error": error}


@contextlib.contextmanager
def _time_limit(seconds: int):
    def _raise(*_):
        raise TimeoutError(f"grading exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def grade(task: dict, traj: dict, state_key: str) -> bool:
    if traj["decoded_steps_per_turn"] is None:
        return False
    b = _import_bfcl()
    try:
        with _time_limit(GRADE_TIMEOUT_S):
            check = b.multi_turn_checker(
                multi_turn_model_result_list_decoded=traj["decoded_steps_per_turn"],
                multi_turn_ground_truth_list=task["ground_truth"], test_entry=task,
                test_category=CATEGORY, model_name=state_key)
        return bool(check.get("valid", False))
    except Exception:
        return False


def _free_state(state_keys: list[str], task: dict) -> None:
    """Drop BFCL's module-level environment instances of a finished task: ``{key}_*`` from
    rollouts, plus the checker's "_eval" copies for the model and the ground truth."""
    ns = _import_bfcl().multi_turn_utils.__dict__
    for key in state_keys:
        for suffix in ("", "_eval", "_ground_truth_eval"):
            for cls in task.get("involved_classes", []):
                ns.pop(re.sub(r"[-./:]", "_", f"{key}{suffix}_{task['id']}_{cls}_instance"), None)


def run_task(policy, task: dict, n: int, seed: int, workers: int, run_id: str) -> dict:
    b = _import_bfcl()
    tools = b.convert_to_tool(task.get("function", []) or [], b.GORILLA_TO_OPENAPI,
                              b.ModelStyle.OPENAI_COMPLETIONS)
    keys = [f"st_{run_id}_r{i}_{task['id']}".replace("/", "_").replace(".", "_") for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        trajs = list(ex.map(lambda i: rollout(policy, task, tools, keys[i], seed + i), range(n)))
    flags = [grade(task, t, k) for t, k in zip(trajs, keys)]   # SIGALRM: main thread only
    _free_state(keys, task)
    for t, ok in zip(trajs, flags):
        t["passed"] = ok
    return {"task_id": task["id"], "pass_flags": flags, "rollouts": trajs}
