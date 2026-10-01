# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model pairs and per-benchmark decoding defaults."""
from __future__ import annotations

# (pair name, base checkpoint, post-trained checkpoint, family).
PAIRS = [
    ("gemma-4-E4B", "google/gemma-4-E4B", "google/gemma-4-E4B-it", "Gemma-4"),
    ("gemma-4-12B", "google/gemma-4-12B", "google/gemma-4-12B-it", "Gemma-4"),
    ("gemma-4-26B-A4B", "google/gemma-4-26B-A4B", "google/gemma-4-26B-A4B-it", "Gemma-4"),
    ("gemma-4-31B", "google/gemma-4-31B", "google/gemma-4-31B-it", "Gemma-4"),
    ("Ministral-3-3B", "mistralai/Ministral-3-3B-Base-2512", "mistralai/Ministral-3-3B-Instruct-2512", "Ministral-3"),
    ("Ministral-3-8B", "mistralai/Ministral-3-8B-Base-2512", "mistralai/Ministral-3-8B-Instruct-2512", "Ministral-3"),
    ("Ministral-3-14B", "mistralai/Ministral-3-14B-Base-2512", "mistralai/Ministral-3-14B-Instruct-2512", "Ministral-3"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "Qwen2.5"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5"),
    ("Qwen2.5-14B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct", "Qwen2.5"),
    ("Qwen2.5-32B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", "Qwen2.5"),
    ("Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B", "Qwen3.5"),
    ("Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B", "Qwen3.5"),
    ("Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B-Base", "Qwen/Qwen3.5-35B-A3B", "Qwen3.5"),
]
# model id -> (pair, arm)
ARM_OF = {m: (p[0], arm) for p in PAIRS for arm, m in (("base", p[1]), ("rl", p[2]))}

BENCHMARKS = ["bfcl", "webshop", "acebench"]

# Decoding parameters, shared by base and post-trained models.
SAMPLING = {
    "bfcl": {"temperature": 0.4, "top_p": 0.95, "max_tokens": 1024},
    "webshop": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 256},
    "acebench": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 1200},
}
N_ROLLOUTS = 128


def default_fence_stop(model: str, benchmark: str) -> str:
    """Closing-fence stop for the Qwen3.5 base models on BFCL (see harness.stop_sequences)."""
    return "closing" if benchmark == "bfcl" and model.startswith("Qwen/Qwen3.5-") else "default"


def default_chat_template_kwargs(model: str) -> dict | None:
    """Thinking off for every Qwen3.5 model on the chat path (incl. w/o-harness base models)."""
    return {"enable_thinking": False} if model.startswith("Qwen/Qwen3.5-") else None
