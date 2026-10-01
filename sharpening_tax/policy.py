# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A policy = one served model + one inference path + fixed sampling parameters.

Paths: ``harness`` sends a base model the text-harness prompt
over ``/v1/completions``; ``chat`` gives a post-trained model the benchmark's default scaffolding
over ``/v1/chat/completions`` (native chat template + vLLM's family tool-call parser, see
scripts/serve_vllm.sh). A base model can also use ``chat``, with the chat template borrowed from
its post-trained sibling, to query it without the harness.

Only temperature, top_p, max_tokens (and seed) are sent; vLLM fills
other sampling fields (e.g. top_k, repetition_penalty) from the checkpoint's generation_config.json.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

try:
    import openai
except ImportError:     # only collecting rollouts needs the client
    openai = None

from . import harness

# Markup the server's tool-call parser should have turned into tool_calls.
_NATIVE_CALL_MARKERS = ("<tool_call>", "<|tool_call>", "<tool_call|>", "<function=")


class ToolParserMismatch(RuntimeError):
    """No tool_calls but native tool-call markup in the text: a malformed call, or a server
    without the family's tool-call parser. The BFCL runner fails the whole rollout on it."""


def server_unreachable(e: BaseException) -> bool:
    """True if the server is unusable (unreachable, or 404/401/403: wrong model, path or key),
    so the run stops. Timeouts and other error replies (a 400 for exceeding the context length,
    a 5xx) count as failed rollouts."""
    if openai is None:
        return False
    if isinstance(e, openai.APIConnectionError):
        return not isinstance(e, openai.APITimeoutError)
    return isinstance(e, (openai.NotFoundError, openai.AuthenticationError, openai.PermissionDeniedError))


@dataclass
class Reply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # OpenAI shape; [] = final answer


class Policy:
    def __init__(self, model: str, mode: str, base_url: str | None = None, *,
                 temperature: float, top_p: float, max_tokens: int,
                 fence_stop: str = "default", chat_template_kwargs: dict | None = None):
        if mode not in ("harness", "chat"):
            raise ValueError(f"mode must be 'harness' or 'chat', got {mode!r}")
        self.model, self.mode = model, mode
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.stop = harness.stop_sequences(fence_stop)
        self.chat_template_kwargs = chat_template_kwargs
        if openai is None:
            raise ImportError("collecting rollouts needs the openai client: pip install openai")
        base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=600, max_retries=2)

    def with_sampling(self, **overrides) -> "Policy":
        """A copy with different temperature / top_p / max_tokens (same endpoint)."""
        other = copy.copy(self)
        for k, v in overrides.items():
            if k not in ("temperature", "top_p", "max_tokens"):
                raise ValueError(k)
            setattr(other, k, v)
        return other

    def config(self) -> dict:
        return {"model": self.model, "mode": self.mode, "temperature": self.temperature,
                "top_p": self.top_p, "max_tokens": self.max_tokens,
                "stop": self.stop if self.mode == "harness" else None,
                "chat_template_kwargs": self.chat_template_kwargs}

    def __call__(self, messages: list[dict], tools: list[dict] | None = None,
                 seed: int | None = None) -> Reply:
        sampling = {"temperature": self.temperature, "top_p": self.top_p,
                    "max_tokens": self.max_tokens}
        if seed is not None:
            sampling["seed"] = seed

        if self.mode == "harness":
            resp = self.client.completions.create(
                model=self.model, prompt=harness.build_prompt(messages, tools), stop=self.stop,
                extra_body={"add_special_tokens": True}, **sampling)
            text = resp.choices[0].text or ""
            return Reply(text, harness.parse_tool_calls(text, tools))

        extra = {"chat_template_kwargs": self.chat_template_kwargs} if self.chat_template_kwargs else None
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, extra_body=extra,
            **({"tools": tools} if tools else {}), **sampling)
        msg = resp.choices[0].message
        text = msg.content or ""
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                 for tc in (msg.tool_calls or [])]
        if tools and not calls and any(m in text for m in _NATIVE_CALL_MARKERS):
            raise ToolParserMismatch("no tool_calls, but the text holds a native tool-call marker "
                                     "(a malformed call, or the server lacks the family's "
                                     f"tool-call parser). text[:200]={text[:200]!r}")
        return Reply(text, calls)
