"""Bounded coordination-only agent checks on the caller's owned local server.

The host must use --no-skills, omit code execution and worker-tool opt-ins, scrub
inherited tool settings, and apply host_arguments(). Only arithmetic text enters
the request. No client tool, URL, attachment or filesystem path is supplied.
"""
from __future__ import annotations

import json
import math
import re


_TOOLS = {"spawn_agent", "wait_agent", "send_input", "close_agent", "list_agents"}
_EXPECTED = {
    "A": {"revenue": 3000, "cost": 1860, "profit": 1140, "overstatement": 160},
    "B": {"revenue": 3600, "cost": 1800, "profit": 1800, "overstatement": 100},
}
_CHILDREN = {"/root/proposal_a": "A", "/root/proposal_b": "B"}
_PROMPT = (
    "This is a bounded arithmetic delegation fixture. Use only coordination tools; do not use files, "
    "skills, shell commands, websites or other tools. Spawn exactly two independent reviewer subagents "
    "named proposal_a and proposal_b, both with permissions read-only and no input_files or depends_on. "
    "Spawn both before waiting. Give proposal_a these facts: Proposal A sells 120 units for 25 each, "
    "pays 13 per unit and 300 fixed cost, and claims profit 1300. Give proposal_b these facts: Proposal B "
    "has 200 customers paying 18 each, pays 7 per customer and 400 fixed cost, and claims profit 1900. "
    "Tell each child to calculate revenue, total cost, profit and claimed-profit overstatement, and return "
    "only a JSON object with the four integer keys revenue, cost, profit, overstatement. Tell children to "
    "calculate directly without creating agents or using tools. Use wait_agent to collect both completed "
    "results. After checking both child reports, return only a JSON object with keys A and B containing "
    "their four-field integer objects. Do not supply a final answer before both children complete."
)


def host_arguments(timeout_seconds):
    """Source-backed host limits; these never enable code execution or worker tools."""
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("agent timeout must be a positive finite number")
    return ["--agents-max-concurrent", "2", "--agents-max-count", "2", "--agents-max-depth", "1",
            "--agents-max-rounds", "3", "--agents-max-generations", "6",
            "--agents-timeout", str(max(1, min(120, int(timeout_seconds)))), "--agents-max-result-chars", "2000"]


def _require(result, name, passed, expected=None, actual=None):
    result["assertions"].append({"name": name, "passed": bool(passed), "expected": expected, "actual": actual})
    if not passed:
        raise ValueError(name)


def _json_object(text):
    if not isinstance(text, str):
        raise ValueError("agent report is not text")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")].strip()
    def pairs(values):
        value = {}
        for key, item in values:
            if key in value:
                raise ValueError("duplicate agent result key: " + key)
            value[key] = item
        return value
    def reject(value):
        raise ValueError("nonfinite agent result number: " + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("agent report must be a JSON object")
    return value


def _exact_numbers(value, expected):
    return (isinstance(value, dict) and set(value) == set(expected) and
            all(type(value[key]) is int and value[key] == number for key, number in expected.items()))


def validate_delegation(stream, result):
    """Only structured successful tool events and completed snapshots qualify."""
    frames = stream.get("chunks", [])
    evidence = result["evidence"].setdefault("agents", {})
    steps = [frame for frame in frames if "skill_step" in frame]
    evidence["tool_steps"] = steps
    for frame in frames:
        if "skill_step" in frame:
            _require(result, "agent tool is permitted coordination", frame["skill_step"] in _TOOLS, sorted(_TOOLS), frame["skill_step"])
            _require(result, "agent coordination call succeeded", frame.get("ok") is True, True, frame.get("ok"))
            _require(result, "agent fixture exported no files", not frame.get("files"), [], frame.get("files"))
        if "tool_progress" in frame:
            _require(result, "agent progress belongs to permitted coordination", frame.get("tool") in _TOOLS, sorted(_TOOLS), frame.get("tool"))
    spawns = [frame for frame in steps if frame["skill_step"] == "spawn_agent"]
    waits = [frame for frame in steps if frame["skill_step"] == "wait_agent"]
    _require(result, "exactly two root children successfully spawned", len(spawns) == 2 and all(frame.get("agent_id") == "/root" for frame in spawns), 2, len(spawns))
    _require(result, "root observed a successful wait", bool(waits) and all(frame.get("agent_id") == "/root" for frame in waits), ">=1 root wait_agent", waits)
    first_wait = next(index for index, frame in enumerate(frames) if frame.get("skill_step") == "wait_agent")
    spawn_indices = [index for index, frame in enumerate(frames) if frame.get("skill_step") == "spawn_agent"]
    _require(result, "both children admitted before first wait", max(spawn_indices) < first_wait)
    final = {}
    for frame in frames:
        snapshots = frame.get("agents")
        if snapshots is None:
            snapshots = []
        _require(result, "agent progress array is structured", isinstance(snapshots, list), "array", type(snapshots).__name__)
        for agent in snapshots:
            _require(result, "agent belongs to expected request tree", isinstance(agent, dict) and agent.get("agent_id") in _CHILDREN and agent.get("parent_id") == "/root", list(_CHILDREN), agent)
            _require(result, "child has enforced read-only reviewer permissions", agent.get("permissions") == "read-only" and agent.get("agent_type") == "reviewer", "read-only reviewer", {key: agent.get(key) for key in ("permissions", "agent_type")})
            _require(result, "independent arithmetic child has no dependencies", agent.get("depends_on") == [], [], agent.get("depends_on"))
            _require(result, "child performed no host tool work", agent.get("tool") in (None, ""), None, agent.get("tool"))
            if frame.get("tool_progress") == "finished" and frame.get("tool") == "wait_agent":
                final[agent["agent_id"]] = agent
    evidence["observed_children"] = final
    _require(result, "finished wait exposes both named children", set(final) == set(_CHILDREN), list(_CHILDREN), list(final))
    workspaces = []
    for child_id, label in _CHILDREN.items():
        agent = final[child_id]
        _require(result, child_id + " completed successfully", agent.get("status") == "completed" and not agent.get("error"), "completed without error", agent)
        workspace = agent.get("workspace_id")
        _require(result, child_id + " has a workspace identity", isinstance(workspace, str) and bool(workspace), "nonempty logical identity", workspace)
        workspaces.append(workspace)
        value = _json_object(agent.get("result"))
        _require(result, child_id + " semantic arithmetic result", _exact_numbers(value, _EXPECTED[label]), _EXPECTED[label], value)
    _require(result, "children have distinct workspace identities", len(set(workspaces)) == 2, 2, len(set(workspaces)))
    answer = _json_object(stream.get("content"))
    _require(result, "final synthesis contains exact child results", set(answer) == set(_EXPECTED) and all(_exact_numbers(answer[label], expected) for label, expected in _EXPECTED.items()), _EXPECTED, answer)
    terminal = [frame for frame in frames if frame.get("done") is True]
    _require(result, "agent stream has one normal completion", len(terminal) == 1 and not any(terminal[0].get(key) for key in ("error", "aborted", "truncated")), "one normal done", terminal)
    evidence["final_result"] = answer


def run_agents(http, model, context, result):
    """Exercise bounded delegation/result delivery and a separate opt-out request."""
    result["inputs"]["agent_fixture"] = {"prompt": _PROMPT, "expected": _EXPECTED, "seed": 42, "temperature": 0}
    result["evidence"]["agents"] = {"limits": host_arguments(context.get("timeout_seconds", 600)),
        "limitations": ["Only two independent arithmetic children and request opt-out are covered.",
                        "Distinct workspace identities do not establish filesystem confinement.",
                        "Dependency scheduling, follow-up turns, disconnect cancellation and mutable worker tools remain untested."]}
    payload = {"messages": [{"role": "user", "content": _PROMPT}], "maxTokens": 256,
               "seed": 42, "temperature": 0, "think": False, "newChat": True, "multi_agent": True}
    delegated = http.stream("/api/chat", payload, web_ui=True)
    validate_delegation(delegated, result)
    single = dict(payload, messages=[{"role": "user", "content": "What is 6 multiplied by 7? Reply with only the integer answer."}],
                  multi_agent=False, maxTokens=32)
    response = http.stream("/api/chat", single, web_ui=True)
    _require(result, "request opt-out answer is correct", re.fullmatch(r"\s*42[.!]?\s*", response.get("content", "")) is not None, "42", response.get("content"))
    events = [frame for frame in response.get("chunks", []) if "skill_step" in frame or "tool_progress" in frame or frame.get("agents")]
    _require(result, "request opt-out performs no tool or agent work", not events, [], events)
    result["evidence"]["agents"]["opt_out_verified"] = True
