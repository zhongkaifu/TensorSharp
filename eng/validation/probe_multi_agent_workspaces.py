"""Probe real-model private worker workspaces and dependency scheduling over SSE.

Requires a running TensorSharp tool-capable model with code execution, mutable
workers, at least two child slots, three agents, and sufficient tool-loop rounds.
This is one executable acceptance scenario, not a latency or model-quality benchmark.
Failures and incomplete evidence are retained and cause a nonzero exit status.
"""
from __future__ import annotations

import argparse
import datetime
import json
import html
import re
import pathlib
import sys
import time
import urllib.parse
import urllib.request


EXPECTED = {"sum": 48, "count": 4, "min": 7, "max": 17, "range": 10, "verified": True}
EXPECTED_FILES = {
    ("/root/totals", "totals.json"): {"sum": 48, "count": 4},
    ("/root/range", "range.json"): {"min": 7, "max": 17, "range": 10},
    ("/root", "verified.json"): EXPECTED,
}
PROMPT = """Execute this small workspace/dependency acceptance task using the available tools.
Do not simulate tool execution. Do not use shell, skills, or nested subagents.
1. In the root workspace, use write_file or apply_patch to create seed.txt containing exactly:
7
11
13
17
2. Spawn two independent workers BEFORE waiting: task_name totals and task_name range,
agent_type worker, permissions workspace-write, input_files seed.txt. Each assignment
must be self-contained: read seed.txt with read_file, calculate its own result, write
its JSON file with write_file or apply_patch, and report the relative output path.
The totals worker writes totals.json with exactly numeric keys sum and count.
The range worker writes range.json with exactly numeric keys min, max, and range
(max minus min). Neither worker may spawn children, use shell, or modify seed.txt.
3. After both spawn calls return IDs, spawn task_name review, agent_type reviewer,
permissions read-only, input_files seed.txt, depends_on containing BOTH returned
worker IDs. Tell the reviewer to read dependencies/totals/totals.json and
dependencies/range/range.json plus seed.txt with read_file, independently verify
all five values, and report one JSON object with sum, count, min, max, range,
and verified (true only if both actual files are correct). Do not give it computed
answers. This reviewer must not create files or spawn children.
4. Call wait_agent without agent_id until all three are completed. Then call
wait_agent once more with timeout_ms 0 so the completed agent states are visible.
5. Root must read the actual worker exports at the parent-relative paths returned
by wait_agent, compare them with the reviewer and seed.txt, and write verified.json
in the root workspace with exactly the reviewer's six checked fields. Use file
tools only. End with that same compact JSON object. If any step fails, report the
failure accurately instead of inventing successful execution or missing artifacts.
"""


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, origin):
        super().__init__()
        self.origin = origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if origin_of(newurl) != self.origin:
            raise ValueError("Refusing an artifact redirect outside the endpoint origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def origin_of(url):
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.scheme, parts.hostname, port


def check(report, name, passed, detail):
    report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})


def json_objects(text):
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[position:])
                if isinstance(value, dict):
                    yield value
            except json.JSONDecodeError:
                pass


def request(endpoint, max_tokens, timeout, report):
    body = {"messages": [{"role": "user", "content": PROMPT}], "think": False,
            "maxTokens": max_tokens, "newChat": True, "multi_agent": True}
    report["request"] = body
    req = urllib.request.Request(endpoint + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter()
    frames = report["frames"]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            report["http_status"] = response.status
            data = []
            for raw in response:
                if time.perf_counter() - started > timeout:
                    raise TimeoutError(f"SSE request exceeded {timeout:g} seconds")
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    payload = "\n".join(data)
                    data.clear()
                    if payload == "[DONE]":
                        continue
                    frame = json.loads(payload)
                    if not isinstance(frame, dict):
                        raise ValueError("Expected an SSE JSON object")
                    frame["_probe_seconds"] = round(time.perf_counter() - started, 6)
                    frames.append(frame)
                    if frame.get("done"):
                        break
            if data:
                frame = json.loads("\n".join(data))
                frame["_probe_seconds"] = round(time.perf_counter() - started, 6)
                frames.append(frame)
    finally:
        report["seconds"] = time.perf_counter() - started
        report["answer"] = "".join(f.get("token", "") for f in frames)


def fetch_artifacts(endpoint, report):
    origin = origin_of(endpoint)
    opener = urllib.request.build_opener(SameOriginRedirect(origin))
    seen = set()
    for frame in report["frames"]:
        for artifact in frame.get("files") or []:
            owner = frame.get("agent_id") or "/root"
            name, url = artifact.get("name", ""), artifact.get("url", "")
            key = (owner, name, url)
            if key in seen:
                continue
            seen.add(key)
            record = {"agent_id": owner, "name": name, "url": url}
            report["artifacts"].append(record)
            # Handoff manifests can contain local paths; never open those or follow
            # arbitrary model-selected URLs. Only server artifact HTTP routes qualify.
            parsed = urllib.parse.urlsplit(url)
            if not url or (not parsed.scheme and not url.startswith("/api/code/artifacts/")):
                record["not_downloaded"] = "No server artifact URL (possibly a workspace handoff path)"
                continue
            target = urllib.parse.urljoin(endpoint + "/", url)
            if origin_of(target) != origin or not urllib.parse.urlsplit(target).path.startswith("/api/code/artifacts/"):
                record["not_downloaded"] = "Outside the endpoint artifact route"
                continue
            try:
                with opener.open(target, timeout=15) as response:
                    content = response.read(65537)
                if len(content) > 65536:
                    raise ValueError("Artifact exceeds probe's 64 KiB inspection bound")
                record["content"] = content.decode("utf-8")
                try:
                    record["json"] = json.loads(record["content"])
                except json.JSONDecodeError:
                    pass
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"



def root_file_reads(frames):
    """Correlate root invocation outcomes with streamed path arguments.

    Prefer the execution progress detail when provided. For older servers,
    Qwen streams XML parameters and other templates can stream JSON. An
    unrecognized format leaves the path unobserved.
    """
    pending, paths, reads = [], [], []
    current_path = None
    for frame in frames:
        if frame.get("tool_progress") == "running" and frame.get("tool") == "read_file":
            current_path = frame.get("detail") or current_path
        if frame.get("tool_progress") == "writing" and frame.get("tool") == "read_file":
            pending.append(frame.get("text") or "")
        if frame.get("tool_progress") == "running" and frame.get("tool") == "read_file" and pending:
            text = "".join(pending)
            pending.clear()
            xml_paths = re.findall(r"<parameter\s*=\s*path>\s*(.*?)\s*</parameter>", text, re.DOTALL)
            if xml_paths:
                paths.extend(html.unescape(path).strip() for path in xml_paths)
            else:
                paths.extend(value["path"] for value in json_objects(text)
                             if isinstance(value.get("path"), str))
        if frame.get("skill_step") == "read_file" and frame.get("agent_id", "/root") == "/root":
            parsed_path = paths.pop(0) if paths else None
            reads.append({"path": current_path or parsed_path or frame.get("detail"),
                          "ok": frame.get("ok"), "seconds": frame.get("_probe_seconds")})
            current_path = None
    return reads


def assess(report):
    frames = report["frames"]
    steps = [f for f in frames if "skill_step" in f]
    snapshots = [a for f in frames for a in f.get("agents") or []]
    latest = {a["agent_id"]: a for a in snapshots if "agent_id" in a}
    report["latest_agents"] = latest
    spawns = [f for f in steps if f["skill_step"] == "spawn_agent" and f.get("ok") is True
              and f.get("agent_id", "/root") == "/root"]
    workers = [latest.get("/root/" + name, {}) for name in ("totals", "range")]
    reviewer = latest.get("/root/review", {})
    check(report, "worker_spawns", len(spawns) >= 3 and all(
        a.get("agent_type") == "worker" and a.get("permissions") == "workspace-write"
        and not a.get("depends_on") for a in workers),
        {"successful_root_spawns": len(spawns), "workers": workers})
    check(report, "reviewer_dependencies", reviewer.get("agent_type") == "reviewer"
          and reviewer.get("permissions") == "read-only"
          and set(reviewer.get("depends_on") or []) == {"/root/totals", "/root/range"}, reviewer)
    check(report, "terminal_children", all(a.get("status") == "completed" for a in workers + [reviewer]),
          {key: value.get("status") for key, value in latest.items()})
    bad_agents = [a for a in snapshots if a.get("error") or a.get("status") in
                  {"failed", "cancelled", "limit_reached", "blocked", "cancelling"}]
    check(report, "no_child_failures", not bad_agents, bad_agents)
    check(report, "no_failed_tools", all(f.get("ok") is True for f in steps),
          [f for f in steps if f.get("ok") is not True])
    nested = [f for f in steps if f.get("skill_step") == "spawn_agent" and f.get("agent_id", "/root") != "/root"]
    check(report, "no_nested_or_execution_tools", not nested and not any(
        f["skill_step"] in {"shell", "skills_run", "run_code"} for f in steps), nested)
    workspace_ids = [a.get("workspace_id") for a in workers + [reviewer]]
    check(report, "distinct_workspace_ids", all(workspace_ids) and len(set(workspace_ids)) == 3, workspace_ids)
    # Child invocation frames expose tool name/owner/success, not call arguments.
    # Require real reads and inspect their produced artifacts instead of guessing paths.
    for owner, minimum in (("/root/totals", 1), ("/root/range", 1), ("/root/review", 3)):
        reads = [f for f in steps if f.get("agent_id") == owner
                 and f.get("skill_step") == "read_file" and f.get("ok")]
        check(report, "child_file_reads:" + owner, len(reads) >= minimum,
              {"successful_reads": len(reads), "required": minimum,
               "limitation": "Child SSE invocation frames do not expose requested file paths"})
    seed_files = [a for a in report["artifacts"] if a["agent_id"] == "/root"
                  and pathlib.PurePosixPath(a["name"]).name == "seed.txt"
                  and a.get("content", "").strip() == "7\n11\n13\n17"]
    check(report, "root_seed_contents", bool(seed_files), {"matching_artifacts": len(seed_files)})
    report["root_file_reads"] = root_file_reads(frames)
    root_export_reads = [read for read in report["root_file_reads"] if read["ok"] is True
                         and str(read["path"]).startswith("agent-results-")]
    check(report, "root_reads_worker_exports", all(any(name == pathlib.PurePosixPath(read["path"]).name
          for read in root_export_reads) for name in ("totals.json", "range.json")), root_export_reads)
    for (owner, name), expected in EXPECTED_FILES.items():
        matches = [a for a in report["artifacts"] if a["agent_id"] == owner
                   and pathlib.PurePosixPath(a["name"]).name == name and a.get("json") == expected]
        check(report, "artifact:" + owner + ":" + name, bool(matches),
              {"expected": expected, "matching_artifacts": len(matches)})
    review_objects = list(json_objects(reviewer.get("result") or ""))
    answer_objects = list(json_objects(report.get("answer", "")))
    check(report, "reviewer_exact_result", bool(review_objects) and review_objects[-1] == EXPECTED,
          reviewer.get("result"))
    check(report, "final_exact_result", bool(answer_objects) and answer_objects[-1] == EXPECTED,
          report.get("answer"))
    done = [f for f in frames if f.get("done")]
    check(report, "successful_stream_completion", bool(done) and not any(
        f.get("error") or f.get("aborted") or f.get("truncated") for f in done), done)
    counts = [sum(a.get("agent_type") == "worker" and a.get("status") == "running"
                  for a in f.get("agents") or []) for f in frames]
    report["max_observed_running_workers"] = max(counts, default=0)
    report["passed"] = not report.get("error") and all(c["passed"] for c in report["checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5039", help="TensorSharp server origin")
    parser.add_argument("--out", default="artifacts/multi-agent-workspaces/workspace-real-model.json")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if urllib.parse.urlsplit(endpoint).scheme not in {"http", "https"}:
        parser.error("--endpoint must be an HTTP(S) server origin")
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    repository = pathlib.Path(__file__).resolve().parents[2]
    output = pathlib.Path(args.out).resolve()
    if not any(output.is_relative_to(repository / folder) for folder in ("artifacts", "docs/validation")):
        parser.error("--out must be inside this repository's ignored artifacts/ or docs/validation/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"endpoint": endpoint, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "frames": [], "checks": [], "artifacts": [], "passed": False,
              "limitations": "One prompted real-model acceptance scenario. Sampled running states do not prove hardware parallelism or speedup. It does not measure autonomous task decomposition, general quality, or other devices/models."}
    try:
        with urllib.request.urlopen(endpoint + "/v1/models", timeout=10) as response:
            report["reported_models"] = json.loads(response.read(65536))
        request(endpoint, args.max_tokens, args.timeout, report)
        fetch_artifacts(endpoint, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        assess(report)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "seconds": report.get("seconds"),
                      "failed_checks": [c["name"] for c in report["checks"] if not c["passed"]],
                      "error": report.get("error"), "out": str(output)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
