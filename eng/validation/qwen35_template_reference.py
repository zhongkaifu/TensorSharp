#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Render the pi-style tool-loop conversation the way transformers' apply_chat_template does.

Writes InferenceWeb.Tests/Fixtures/Qwen35/pi_tool_loop.transformers.txt, the reference that
ClientToolLoopCompatibilityTests compares TensorSharp's renderer with. Needs only jinja2
(pip install jinja2); the environment below is the one transformers builds for chat templates.

    python3 eng/validation/qwen35_template_reference.py \
        InferenceWeb.Tests/Fixtures/Qwen35/chat_template.jinja \
        InferenceWeb.Tests/Fixtures/Qwen35/pi_tool_loop.transformers.txt
"""
import json
import sys

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def raise_exception(message):
    raise TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
MESSAGES = [
    {"role": "system", "content": "You are pi."},
    {"role": "user", "content": "Read notes.txt"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"path": "notes.txt"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "hello world"},
]


def main():
    template_path, out_path = sys.argv[1], sys.argv[2]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    with open(template_path, encoding="utf-8") as f:
        template = env.from_string(f.read())
    rendered = template.render(messages=MESSAGES, tools=TOOLS, add_generation_prompt=True, enable_thinking=False)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(rendered)


if __name__ == "__main__":
    main()
