"""Render Inkling 2's published chat template the way transformers does, for the renderer's reference strings.

    python eng/inkling2/render_template_reference.py chat_template.jinja cases.json

cases.json is a list of {"messages": [...], "tools": [...] (optional), "reasoning_effort": ... (optional)}. Prints a
JSON list of the rendered prompts. Needs jinja2. transformers renders with an ImmutableSandboxedEnvironment
(trim_blocks, lstrip_blocks, loopcontrols), a tojson that keeps non-ASCII, and the {% generation %} tag, which only
marks assistant spans for training masks and renders its body unchanged.
"""

from __future__ import annotations

import json
import sys

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise ValueError(message)


def main() -> int:
    template_path, cases_path = sys.argv[1], sys.argv[2]
    with open(template_path, encoding="utf-8") as handle:
        source = handle.read()
    # The generation tag is transformers' AssistantTracker; its body renders as written.
    source = source.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)
    with open(cases_path, encoding="utf-8") as handle:
        cases = json.load(handle)
    rendered = []
    for case in cases:
        kwargs = {"messages": case["messages"], "add_generation_prompt": True}
        if case.get("tools"):
            kwargs["tools"] = case["tools"]
        if "reasoning_effort" in case:
            kwargs["reasoning_effort"] = case["reasoning_effort"]
        rendered.append(template.render(**kwargs))
    json.dump(rendered, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
