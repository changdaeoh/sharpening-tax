# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions adapted from ACEBench (https://github.com/chenchen0103/ACEBench, commit 56dd66c:
# eval_main.py and model_inference/apimodel_inference.py), MIT License,
# Copyright (c) Microsoft Corporation.
"""ACEBench (English), 770 tasks: 13 single-shot categories (750) + ``agent_multi_step`` (20).

Prompts, category system prompts and graders are ACEBench's own: ``normal_*`` outputs are
AST-decoded and checked by ``normal_checker``; ``special_*`` outputs must contain the category's
detection sentence and values; an ``agent_multi_step`` episode passes iff the final states of
ACEBench's scenario classes match the ground truth. The multi-turn categories (agent_multi_turn,
which needs an LLM user simulator, and the two normal_multi_turn_* categories) are not used. Needs https://github.com/chenchen0103/ACEBench @ 56dd66c at $ACEBENCH_DIR.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from concurrent.futures import ThreadPoolExecutor

from ..policy import server_unreachable

# All 14 categories by default; SINGLE_SHOT (750 tasks) leaves out the multi-step agent tasks.
SINGLE_SHOT = [
    "normal_atom_bool", "normal_atom_enum", "normal_atom_list", "normal_atom_number",
    "normal_atom_object_deep", "normal_atom_object_short",
    "normal_single_turn_single_function", "normal_single_turn_parallel_function",
    "normal_similar_api", "normal_preference",
    "special_incomplete", "special_error_param", "special_irrelevant",
]
AGENT = "agent_multi_step"
CATEGORIES = SINGLE_SHOT + [AGENT]
MAX_DIALOG_TURNS = 40          # ACEBench default
# ACEBench's multi-step agent (APIAgent_step) has its own near-greedy defaults and Chinese agent
# prompt; agent_multi_step runs with them, as upstream does, not with the benchmark sampling.
AGENT_UPSTREAM_SAMPLING = {"temperature": 0.001, "top_p": 1.0, "max_tokens": 1000}

_ace = None


def _import_ace():
    global _ace
    if _ace is not None:
        return _ace
    root = os.environ.get("ACEBENCH_DIR") or sys.exit("set ACEBENCH_DIR to the ACEBench checkout")
    sys.path.insert(0, root)
    from model_eval.checker import agent_checker, normal_checker
    from model_eval.utils import is_function_call_format_valid
    from model_inference import prompt_en as P
    from model_inference.apimodel_inference import SAVED_CLASS
    from model_inference.multi_step import APIModel_agent as A
    from model_inference.multi_step import multi_step_utils
    from model_inference.multi_step.execution_role_step import EXECUTION_STEP
    from model_inference.multi_step.multi_step_scene import Mulit_Step_Scene
    from model_inference.utils import decode_ast
    _ace = types.SimpleNamespace(
        root=root, P=P, A=A, decode_ast=decode_ast, normal_checker=normal_checker,
        is_fc_valid=is_function_call_format_valid, agent_checker=agent_checker,
        SAVED_CLASS=SAVED_CLASS, EXECUTION_STEP=EXECUTION_STEP, Scene=Mulit_Step_Scene,
        state=multi_step_utils)
    return _ace


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_tasks(categories: list[str] = CATEGORIES) -> list[dict]:
    """The tasks of ``categories`` (default: all 770), each with its category and ground truth."""
    ace, tasks = _import_ace(), []
    for cat in categories:
        data = os.path.join(ace.root, "data_all", "data_en")
        gt = {r["id"]: r["ground_truth"] for r in _read(os.path.join(data, "possible_answer", f"data_{cat}.json"))}
        for t in _read(os.path.join(data, f"data_{cat}.json")):
            t["category"], t["ground_truth"] = cat, gt.get(t["id"], {})
            tasks.append(t)
    return tasks


# ---- Single-shot categories ------------------------------------------------ #

def _prompts(task: dict) -> list[dict]:
    P, cat = _import_ace().P, task["category"]
    functions = task.get("function", []) or []
    functions = [functions] if isinstance(functions, dict) else functions
    if "special" in cat:
        system = P.SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=task.get("time", "") or "", function=functions)
    elif "preference" in cat:
        system = P.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=task.get("profile", "") or "", function=functions)
    else:
        system = P.SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=task.get("time", "") or "", function=functions)
    return [{"role": "system", "content": system},
            {"role": "user", "content": P.USER_PROMPT_EN.format(question=task.get("question", ""))}]


def _outermost_brackets(text: str) -> str | None:
    start, depth = -1, 0
    for i, ch in enumerate(text):
        if ch == "[":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:i + 1]
    return None


def grade_single_shot(task: dict, output: str) -> bool:
    """ACEBench's checker for one single-shot output (mirrors eval_main.py)."""
    ace, cat, answer = _import_ace(), task["category"], task["ground_truth"] or {}
    if cat.startswith("special"):
        if "incomplete" in cat:
            return all("Missing necessary parameters" in output and name in output
                       and all(v in output for v in values) for name, values in answer.items())
        if "error" in cat:
            return all("There is incorrect value" in output and all(v in output for v in values)
                       for name, values in answer.items())
        return "the limitations of the function" in output          # special_irrelevant
    call = _outermost_brackets(output or "")
    if call is None:
        return False
    try:
        decoded = ace.decode_ast("vllm", call)
    except Exception:
        return False
    if not ace.is_fc_valid(decoded):
        return False
    functions = task.get("function", []) or []
    for ans in (answer if isinstance(answer, list) else [answer]):
        try:
            if ace.normal_checker(functions, decoded, ans, task.get("question", ""), cat).get("valid"):
                return True
        except Exception:
            pass
    return False


# ---- agent_multi_step ------------------------------------------------------ #

def agent_rollout(policy, task: dict, state_key: str) -> list:
    """One episode (mirrors APIModelInference.multi_step_inference) -> final states of the
    scenario classes the agent touched, JSON round-tripped as upstream does before grading."""
    ace = _import_ace()
    functions = task["function"]
    functions = [functions] if isinstance(functions, (dict, str)) else functions
    system_t, user_t = ace.A.MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH, ace.A.MULTI_TURN_AGENT_PROMPT_USER_ZH
    test_id = task["id"].split("_")[-1]
    scene = ace.Scene(question=task["question"], initial_state=task["initial_config"],
                      functions=functions, agent_role=None, language="en")
    execution = ace.EXECUTION_STEP(agent_model_name=state_key, initial_config=task["initial_config"],
                                   involved_classes=task["involved_classes"], test_id=test_id, language="en")
    history, instances = scene.dialogue_history, []
    for index in range(MAX_DIALOG_TURNS):
        if index == 0 or history[-1]["sender"] == "execution":
            prompt = [{"role": "system", "content": system_t.format(time="")},
                      {"role": "user", "content": user_t.format(functions=functions,
                                                                history=scene.get_inference_message())}]
            text = policy(prompt).text
            message = {"sender": "agent", "message": text,
                       "recipient": "execution" if re.match(r"\[.*?\]", text) else "user"}
        else:
            message, inst = execution.respond(history)
            if inst not in instances:
                instances.append(inst)
        scene.add_dialogue(message)
        if index > 1 and "finish conversation" in message["message"]:
            break
    result = [{name: {k: v for k, v in obj.__dict__.items() if k in ace.SAVED_CLASS[name]}}
              for inst in instances for name, obj in inst.items()]
    return json.loads(json.dumps(result, ensure_ascii=False))


def grade_agent(task: dict, result: list) -> bool:
    """ACEBench's end-to-end agent criterion (mirrors eval_main.agent_eval, including its
    behaviour when a ground-truth class has no counterpart in the result)."""
    answer = task["ground_truth"]
    answer = answer if isinstance(answer, list) else [answer]
    if len(answer) != len(result):
        return False
    check = {"valid": True}
    for expected in answer:
        match = next((r for r in result if set(r) == set(expected)), None)
        if match:
            check = _import_ace().agent_checker(match, expected)
        if check["valid"] is False:
            return False
    return True


def _free_agent_state(state_keys: list[str], task: dict) -> None:
    ns, test_id = _import_ace().state.__dict__, task["id"].split("_")[-1]
    for key in state_keys:
        prefix = key.replace("-", "_").replace(".", "_").replace("/", "_")
        for cls in task["involved_classes"]:
            ns.pop(f"{prefix}_{test_id}_{cls.lower()}_instance", None)


def run_task(policy, task: dict, n: int, seed: int, workers: int, run_id: str) -> dict:
    if task["category"] != AGENT:
        def one(i):
            try:
                text = policy(_prompts(task), seed=seed + i).text
            except Exception as e:
                if server_unreachable(e):
                    raise
                return {"output": None, "error": f"generation error: {e}", "passed": False}
            return {"output": text, "passed": grade_single_shot(task, text)}
    else:
        agent_policy = policy.with_sampling(**AGENT_UPSTREAM_SAMPLING)
        keys = [f"st_{run_id}_r{i}" for i in range(n)]

        def one(i):
            try:
                result = agent_rollout(agent_policy, task, keys[i])
            except Exception as e:
                if server_unreachable(e):
                    raise
                return {"result": None, "error": f"rollout error: {e}", "passed": False}
            return {"result": result, "passed": grade_agent(task, result)}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        trajs = list(ex.map(one, range(n)))
    if task["category"] == AGENT:
        _free_agent_state(keys, task)
    return {"task_id": task["id"], "category": task["category"],
            "pass_flags": [t["passed"] for t in trajs], "rollouts": trajs}
