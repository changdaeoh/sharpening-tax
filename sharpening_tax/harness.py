# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Text harness that lets a pre-trained (base) LLM act as a tool-calling agent.

Renders an OpenAI-style conversation (``messages`` + ``tools``) as one plain-text prompt for
the raw ``/v1/completions`` endpoint and parses the completion back into OpenAI tool calls.
One harness for every base model and benchmark (no per-model presets, no few-shot demos); the
only per-model setting is the closing-fence stop for Qwen3.5 base models on BFCL.

Prompt: system prompt, tools as Python stubs
(the model reads Python), a fenced ``tool_call`` JSON-array contract (it writes JSON), the
``### User / ### Assistant / ### Tool Output`` transcript, then a ``### Assistant`` cue.
A reply without a ``tool_call`` block is the final answer.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

MARKERS = {"user": "### User", "assistant": "### Assistant", "tool_output": "### Tool Output"}
CALL_FENCE = "tool_call"


def stop_sequences(fence_stop: str = "default") -> list[str]:
    """Stop strings standing in for the end-of-turn token a base model lacks.

    The two markers stop the model before it writes the tool's or user's turn; the fence ends
    the turn after the first tool-call block. ``closing`` ("\\n```\\n") cannot match an opening
    fence after a prose preamble: used only for the Qwen3.5 base models on BFCL, which write
    such a preamble and would otherwise be cut off before any JSON.
    """
    fence = {"default": "\n```", "closing": "\n```\n"}[fence_stop]
    return [MARKERS["tool_output"], MARKERS["user"], fence]


# ---- Tool catalog: OpenAI JSON schema -> Python stubs ---------------------- #

_SCALAR = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "null": "None"}


def _lit(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)


def _py_type(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "Any"
    if schema.get("enum"):
        return "Literal[" + ", ".join(_lit(v) for v in schema["enum"]) + "]"
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            subs = list(dict.fromkeys(_py_type(s) for s in schema[key]))
            if not subs:
                return "Any"
            return subs[0] if len(subs) == 1 else "Union[" + ", ".join(subs) + "]"
    t = schema.get("type")
    if isinstance(t, list):
        subs = list(dict.fromkeys(_SCALAR.get(x, "Any") for x in t if x != "null")) or ["Any"]
        base = subs[0] if len(subs) == 1 else "Union[" + ", ".join(subs) + "]"
        return f"Optional[{base}]" if "null" in t else base
    if t == "array":
        items = schema.get("items")
        return f"list[{_py_type(items)}]" if isinstance(items, dict) else "list"
    if t == "object":
        return "dict"
    return _SCALAR.get(t, "Any")


def _unwrap(tool: dict) -> dict:
    return tool["function"] if isinstance(tool.get("function"), dict) else tool


def render_tool_stub(tool: dict) -> str:
    """One tool as a Python ``def`` stub with a Google-style docstring."""
    fn = _unwrap(tool)
    name = fn.get("name", "tool")
    desc = (fn.get("description") or "").strip()
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    # Required parameters first so the signature is valid Python.
    ordered = [k for k in props if k in required] + [k for k in props if k not in required]

    sig = ", ".join(
        f"{k}: {_py_type(props[k])}" if k in required
        else f"{k}: {_py_type(props[k])} = {_lit(props[k].get('default', None))}"
        for k in ordered)
    doc = [desc or f"Call the {name} tool."]
    if ordered:
        doc += ["", "Args:"]
        for k in ordered:
            p = props[k]
            line = f"    {k} ({_py_type(p)}, {'required' if k in required else 'optional'}):"
            if (p.get("description") or "").strip():
                line += " " + p["description"].strip()
            doc.append(line)
            if p.get("enum"):
                doc.append(f"        choices: {p['enum']}")

    ind = "    "
    out = [f"def {name}({sig}) -> dict:"]
    if len(doc) == 1:
        out.append(f'{ind}"""{doc[0]}"""')
    else:
        out.append(f'{ind}"""{doc[0]}')
        out += [ind + l if l else "" for l in doc[1:]]
        out.append(f'{ind}"""')
    out.append(f"{ind}...")
    return "\n".join(out)


CALL_CONTRACT = (
    "# How to call tools\n"
    f"To call one or more tools, emit a fenced ```{CALL_FENCE}``` block containing "
    "a JSON array of calls. Each call is an object "
    '{"name": <tool_name>, "arguments": {<param>: <value>, ...}} whose '
    "argument keys match the function parameters above. Format:\n"
    f"```{CALL_FENCE}\n"
    '[{"name": "<tool_name>", "arguments": {"<param>": "<value>"}}]\n'
    "```\n"
    f'After the block, stop — the result will be provided under "{MARKERS["tool_output"]}". '
    "You may place several calls in the array to call tools in parallel. "
    "When the task is complete, reply in plain text with no "
    f"```{CALL_FENCE}``` block."
)


# ---- Transcript + prompt --------------------------------------------------- #

def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join((p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p)
                       for p in content)
    return str(content)


def render_call_block(tool_calls: list[dict]) -> str:
    """OpenAI tool_calls -> the fenced JSON array the model is asked to emit."""
    arr = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        arr.append({"name": fn.get("name"), "arguments": args or {}})
    return f"```{CALL_FENCE}\n{json.dumps(arr, ensure_ascii=False)}\n```"


def serialize(messages: list[dict]) -> str:
    """Non-system messages -> plain-text transcript. An assistant turn
    is its raw completion followed by the call block re-rendered from its parsed tool calls."""
    id2name = {tc.get("id"): (tc.get("function") or {}).get("name")
               for m in messages for tc in (m.get("tool_calls") or []) if tc.get("id")}
    blocks = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            blocks.append(f"{MARKERS['user']}\n{_text(m.get('content'))}")
        elif role == "assistant":
            parts = [p for p in (_text(m.get("content")),) if p]
            if m.get("tool_calls"):
                parts.append(render_call_block(m["tool_calls"]))
            blocks.append(f"{MARKERS['assistant']}\n" + "\n".join(parts))
        elif role == "tool":
            name = id2name.get(m.get("tool_call_id")) or m.get("name") or "tool"
            blocks.append(f"{MARKERS['tool_output']}\n[{name}] -> {_text(m.get('content'))}")
        else:
            blocks.append(f"### {role}\n{_text(m.get('content'))}")
    return "\n\n".join(blocks)


def build_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    """Raw completion prompt for one generation step."""
    system_prompt = next((_text(m.get("content")) for m in messages if m.get("role") == "system"), None)
    blocks = []
    if system_prompt and system_prompt.strip():
        blocks.append(system_prompt.strip())
    if tools:
        blocks.append("# Available tools\n```python\n"
                      + "\n\n".join(render_tool_stub(t) for t in tools) + "\n```")
        blocks.append(CALL_CONTRACT)
    transcript = serialize(messages)
    if transcript:
        blocks.append(transcript)
    return "\n\n".join(blocks) + f"\n\n{MARKERS['assistant']}\n"


# ---- Completion -> OpenAI tool calls --------------------------------------- #

_FENCE_RE = re.compile(r"```(?:tool_call|json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def _repair(s: str) -> str:
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _loads_lenient(s: str):
    """json.loads with light repairs (smart quotes, trailing commas, single quotes)."""
    if not s or not s.strip():
        return None
    s = s.strip()
    attempts = [s, _repair(s)]
    if '"' not in s and "'" in s:
        attempts.append(_repair(s).replace("'", '"'))
    for a in attempts:
        try:
            return json.loads(a)
        except ValueError:
            continue
    return None


def _first_json_blob(text: str) -> str | None:
    """First balanced [...] or {...} span (string-aware), for replies without a fence."""
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        depth, in_str, esc, quote = 0, False, False, ""
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    in_str = False
            elif c in "\"'":
                in_str, quote = True, c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
    return None


def _as_calls(obj) -> list[dict]:
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)]
    if isinstance(obj, dict):
        if "name" in obj:
            return [obj]
        for key in ("tool_calls", "calls", "tools"):
            if isinstance(obj.get(key), list):
                return [o for o in obj[key] if isinstance(o, dict)]
    return []


def _coerce(v, prop):
    """Cast a string argument to its declared JSON-Schema scalar type."""
    if not isinstance(v, str) or not isinstance(prop, dict):
        return v
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    s = v.strip()
    try:
        if t == "integer":
            return int(s)
        if t == "number":
            return float(s)
    except ValueError:
        return v
    if t == "boolean" and s.lower() in ("true", "false"):
        return s.lower() == "true"
    return v


def make_tool_call(name: str, args) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}}


def parse_tool_calls(text: str, tools: list[dict] | None = None) -> list[dict]:
    """Parse a base-model completion into OpenAI-shaped tool calls ([] = final answer).

    Fenced blocks first, else the first balanced JSON blob; the first block yielding valid
    calls wins. With ``tools``, unknown tool names are dropped and string args cast to schema types.
    """
    if not text:
        return []
    props = {}
    for t in tools or []:
        fn = _unwrap(t) if isinstance(t, dict) else {}
        if fn.get("name"):
            props[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    known = set(props) if tools else None

    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    if not candidates:
        blob = _first_json_blob(text)
        candidates = [blob] if blob else []
    for block in candidates:
        obj = _loads_lenient(block)
        if obj is None:
            continue
        calls = []
        for call in _as_calls(obj):
            name = call.get("name")
            if not isinstance(name, str) or (known is not None and name not in known):
                continue
            args = call.get("arguments", call.get("parameters", {}))
            if isinstance(args, str):
                parsed = _loads_lenient(args)
                args = parsed if isinstance(parsed, dict) else {}
            if not isinstance(args, dict):
                args = {}
            schema = props.get(name, {})
            calls.append(make_tool_call(name, {k: _coerce(v, schema.get(k)) for k, v in args.items()}))
        if calls:
            return calls
    return []
