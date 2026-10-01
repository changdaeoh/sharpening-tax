# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The base-model text harness: prompt layout, transcript replay and tool-call parsing."""
import json

from sharpening_tax import harness

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string", "description": "City name."},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius",
                 "description": "Temperature unit."}}}}}


def _episode():
    call = harness.make_tool_call("get_weather", {"city": "Palo Alto"})
    return [{"role": "system", "content": "You are a helpful assistant that can call tools."},
            {"role": "user", "content": "What's the weather in Palo Alto?"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": '{"temp_c": 21, "sky": "sunny"}'}]


def test_prompt_layout():
    prompt = harness.build_prompt(_episode(), [WEATHER])
    assert prompt.startswith("You are a helpful assistant that can call tools.\n\n# Available tools\n```python\n")
    assert 'def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:' in prompt
    assert "        choices: ['celsius', 'fahrenheit']" in prompt
    assert "# How to call tools" in prompt
    assert ('### User\nWhat\'s the weather in Palo Alto?\n\n### Assistant\n```tool_call\n'
            '[{"name": "get_weather", "arguments": {"city": "Palo Alto"}}]\n```') in prompt
    assert '### Tool Output\n[get_weather] -> {"temp_c": 21, "sky": "sunny"}' in prompt
    assert prompt.endswith("\n\n### Assistant\n")


def test_replay_of_a_stop_truncated_completion():
    # The stop string "\n```" is not part of the completion, so the stored assistant turn is an
    # unclosed block; the transcript replays it verbatim, then the canonical re-rendered block.
    raw = '```tool_call\n[{"name": "get_weather", "arguments": {"city": "Palo Alto"}}]'
    calls = harness.parse_tool_calls(raw, [WEATHER])
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == \
           [("get_weather", {"city": "Palo Alto"})]
    transcript = harness.serialize([{"role": "user", "content": "Weather?"},
                                    {"role": "assistant", "content": raw, "tool_calls": calls}])
    assert transcript == (
        "### User\nWeather?\n\n### Assistant\n"
        '```tool_call\n[{"name": "get_weather", "arguments": {"city": "Palo Alto"}}]\n'
        '```tool_call\n[{"name": "get_weather", "arguments": {"city": "Palo Alto"}}]\n```')


def test_call_block_round_trip():
    calls = [harness.make_tool_call("get_weather", {"city": "Paris", "unit": "fahrenheit"})]
    parsed = harness.parse_tool_calls(harness.render_call_block(calls), [WEATHER])
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in parsed] == \
           [("get_weather", {"city": "Paris", "unit": "fahrenheit"})]


def test_lenient_parsing():
    tools = [WEATHER]
    names = lambda text: [c["function"]["name"] for c in harness.parse_tool_calls(text, tools)]  # noqa: E731
    assert names('```tool_call\n[{"name": "get_weather", "arguments": {"city": "Rome",}}]') == ["get_weather"]
    assert names("Let me check. {'name': 'get_weather', 'arguments': {'city': 'Rome'}}") == ["get_weather"]
    assert names('```tool_call\n[{"name": "unknown", "arguments": {}}]\n```') == []
    assert names("It is sunny in Rome.") == []               # final answer


def test_stop_sequences():
    assert harness.stop_sequences() == ["### Tool Output", "### User", "\n```"]
    assert harness.stop_sequences("closing")[-1] == "\n```\n"
