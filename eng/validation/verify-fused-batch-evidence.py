#!/usr/bin/env python3
"""Verify completed native joint graphs against actual concurrent HTTP requests.

Consumes opt-in TS_BATCHED_FUSED_TRACE=1 JSONL, never invokes a server or model.
One logical graph may use multiple backend dispatches around host MoE seams.
This is execution engagement evidence, not quality, throughput or a GPU hard cap.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP_EVENT = "generation.sequence.bound"
BATCH_EVENT = "batch.fused.accepted"
MAX_LOG_BYTES = 128 << 20


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def native_execution_id(value):
    # The production structured logger serializes UInt64 as an invariant
    # decimal string (Int64 is a JSON number). Accept these exact two forms,
    # not floats, booleans, signed/whitespace text or lossy JSON coercions.
    if isinstance(value, str):
        if (not value or len(value) > 20 or not value.isascii() or not value.isdecimal()
                or value.startswith("0")):
            raise ValueError("Native execution ID must be canonical positive UInt64 decimal")
        value = int(value)
    if not integer(value, 1) or value > (1 << 64) - 1:
        raise ValueError("Native execution ID is outside positive UInt64 range")
    return value


def array(props, key):
    # The logger stores explicit JSON strings, avoiding provider-dependent
    # IEnumerable.ToString(). Also accept an already structured JSON array.
    value = props.get(key)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    return value


def request_scope(record):
    # Canonical JSONL is singular. An invalid canonical member must not be
    # repaired by an unrelated legacy plural member.
    if "scope" in record:
        scope = record["scope"]
        if not isinstance(scope, dict):
            raise ValueError("Canonical scope is not an object")
        return scope
    scopes = record.get("scopes", [])
    if isinstance(scopes, dict):
        return scopes
    if not isinstance(scopes, list) or any(not isinstance(s, dict) for s in scopes):
        raise ValueError("Invalid legacy scopes")
    merged = {}
    for scope in scopes:
        merged.update(scope)
    return merged


def event_is(record, prefix):
    template = record.get("template", record.get("message", ""))
    return isinstance(template, str) and (template == prefix or template.startswith(prefix + " "))


def generation_timing_evidence(report, records):
    """Keep attributed forward time separate from inclusive HTTP/server time.

    This diagnostic does not change the native engagement or quality gates.
    A child agent may inherit its parent's HTTP scope, so its different session
    must not be added to the root generation's duration or output token count.
    """
    result = {"all_current_requests_measured": False, "rows": [], "errors": [],
              "missing_http_request_ids": [], "ignored_other_session_rows": 0,
              "scope": "Per-generation attributed model forward durations, not request wall time or isolated queue/preparation/tool latency. Shared batch time is apportioned between sequences. Pipeline and engine token counts are distinct. Missing completion metrics remain null; no throughput is inferred."}
    targets = {}
    for mode in ("concurrent", "serial"):
        rows = report.get(mode, [])
        if not isinstance(rows, list):
            result["errors"].append(f"{mode} is not an HTTP row list")
            continue
        for row in rows:
            if not isinstance(row, dict):
                result["errors"].append(f"{mode} contains a malformed HTTP row")
                continue
            http_id = row.get("request_id")
            session_record = row.get("session")
            session = session_record.get("sessionId") if isinstance(session_record, dict) else None
            if not isinstance(http_id, str) or not http_id or not isinstance(session, str) or not session:
                result["errors"].append(f"{mode} lacks request/session identity for timing")
                continue
            if http_id in targets:
                result["errors"].append("HTTP timing identity is reused")
                continue
            targets[http_id] = (session, mode)
    seen, measured, unavailable = set(), set(), set()
    for line_number, record in enumerate(records, 1):
        if not event_is(record, "generation.timings"):
            continue
        try:
            scope = request_scope(record)
            http_id = scope.get("RequestId")
            if http_id not in targets:
                continue
            props = record.get("props", {})
            session, mode = targets[http_id]
            if props.get("SessionId") != session:
                result["ignored_other_session_rows"] += 1
                continue
            internal = props.get("InternalRequestId")
            if not isinstance(internal, str) or not internal or internal in seen:
                raise ValueError("Missing or duplicated generation timing identity")
            if props.get("TimingBasis") != "attributed_forward":
                raise ValueError("Unknown forward timing basis")
            for key in ("PromptTokens", "KvReusedTokens", "GeneratedTokens", "TotalNs"):
                if not integer(props.get(key)):
                    raise ValueError(f"{key} must be a measured nonnegative integer")
            if props["KvReusedTokens"] > props["PromptTokens"]:
                raise ValueError("Reused prompt tokens exceed prompt length")
            available = props.get("EngineCompletionAvailable")
            if type(available) is not bool:
                raise ValueError("Engine completion availability must be explicit")
            for key in ("PromptNs", "EvalNs", "EngineOutputTokens"):
                if (available and not integer(props.get(key))) or (not available and props.get(key) is not None):
                    raise ValueError(f"{key} must be measured when completion exists, otherwise null")
            reason = props.get("FinishReason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Missing generation finish reason")
            seen.add(internal)
            if available:
                measured.add(http_id)
            else:
                unavailable.add(http_id)
            result["rows"].append({"line": line_number, "mode": mode, "http_request_id": http_id,
                "internal_request_id": internal, "session_id": session,
                "prompt_tokens": props["PromptTokens"], "kv_reused_tokens": props["KvReusedTokens"],
                "recorded_pipeline_tokens": props["GeneratedTokens"],
                "engine_output_tokens": props.get("EngineOutputTokens"),
                "attributed_prefill_ns": props.get("PromptNs"), "attributed_decode_ns": props.get("EvalNs"),
                "server_generation_total_ns": props["TotalNs"], "finish_reason": reason,
                "engine_completion_available": available})
        except (ValueError, TypeError, AttributeError) as error:
            result["errors"].append(f"Timing line {line_number}: {error}")
    # A later measured tool round cannot repair missing completion metrics for
    # an earlier generation under the same HTTP request and root session.
    result["unavailable_completion_http_request_ids"] = sorted(unavailable)
    result["missing_http_request_ids"] = sorted((set(targets) - measured) | unavailable)
    result["all_current_requests_measured"] = bool(targets) and not result["errors"] and not result["missing_http_request_ids"]
    return result


def verify(report, records, minimum_steps=2):
    result = {"schema_version": 1, "passed": False,
              "batched_fused_graph_verified": False, "actual_joint_engagement_verified": False,
              "full_interval_continuity_verified": False, "errors": [],
              "continuity_gaps": [], "contiguous_joint_runs": [],
              "matched_executions": [], "request_mapping": {},
              "scope": "At least two current HTTP requests advanced together through repeated successful native logical joint graphs, followed by managed publication. Host MoE seams may dispatch multiple GPU/CPU segments. Does not establish natural EOS, output quality, full-lifetime batching, performance or physical hard limits."}
    result["generation_timings"] = generation_timing_evidence(report, records)
    errors = result["errors"]
    if not integer(minimum_steps, 2):
        errors.append("minimum_steps must be at least 2")
        return result
    result["source_report_passed"] = report.get("passed")
    result["source_greedy_text_parity"] = report.get("greedy_text_parity")
    if report.get("completed") is not True:
        errors.append("Concurrent HTTP report did not complete")
    if not isinstance(report.get("model"), dict) or report["model"].get("unifiedMemory") is not True:
        errors.append("Concurrent HTTP report does not identify an active unified session")
    rows = report.get("concurrent")
    if not isinstance(rows, list) or len(rows) != 2:
        errors.append("Exactly two current concurrent HTTP rows are required")
        return result
    targets = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("Concurrent HTTP row is not an object")
            return result
        http_id = row.get("request_id")
        session_record = row.get("session")
        session = session_record.get("sessionId") if isinstance(session_record, dict) else None
        terminal = row.get("terminal")
        if (not isinstance(http_id, str) or not http_id or http_id in targets
                or not isinstance(session, str) or not session
                or not isinstance(terminal, dict)
                or row.get("completed") is not True or row.get("status") != 200
                or row.get("error") or row.get("cleanup_error")
                or terminal.get("done") is not True or terminal.get("aborted") or terminal.get("error")):
            errors.append("Each concurrent row needs a unique request ID, session, HTTP 200 and completed non-aborted stream")
            return result
        targets[http_id] = session
    if len(set(targets.values())) != 2:
        errors.append("Concurrent requests must belong to two distinct sessions")
        return result

    # Resolve all bindings first: worker success logging can outrun the HTTP
    # thread's append. Each engine request must have one unambiguous HTTP owner.
    bindings = {}
    target_sessions = set(targets.values())
    target_internal_ids = set()
    for record in records:
        if not event_is(record, MAP_EVENT):
            continue
        try:
            props = record.get("props", {})
            if request_scope(record).get("RequestId") in targets:
                internal = props.get("InternalRequestId")
                if isinstance(internal, str) and internal:
                    target_internal_ids.add(internal)
        except (ValueError, TypeError, AttributeError):
            pass  # The relevant record's precise validation below fails closed.
    ignored_mappings = 0
    for line_number, record in enumerate(records, 1):
        if not event_is(record, MAP_EVENT):
            continue
        try:
            props = record.get("props", {})
            internal = props.get("InternalRequestId")
            session = props.get("SessionId")
            target_hint = session in target_sessions or internal in target_internal_ids
            try:
                scope = request_scope(record)
            except (ValueError, TypeError, AttributeError):
                if target_hint:
                    raise
                ignored_mappings += 1
                continue
            http_id = scope.get("RequestId")
            if http_id not in targets and not target_hint:
                # Startup prefix warm-up is a real internal generation with no
                # HTTP owner. It cannot qualify or invalidate these requests.
                ignored_mappings += 1
                continue
            if not isinstance(internal, str) or not internal or not isinstance(http_id, str) or not http_id:
                raise ValueError("Missing internal/HTTP request ID")
            binding = (http_id, session)
            if internal in bindings and bindings[internal] != binding:
                raise ValueError("One internal request is attributed to different HTTP/session owners")
            bindings[internal] = binding
            if http_id in targets and session != targets[http_id]:
                raise ValueError("Mapped session differs from current HTTP response")
        except (ValueError, TypeError, AttributeError) as error:
            errors.append(f"Mapping line {line_number}: {error}")
    relevant = {internal: pair for internal, pair in bindings.items() if pair[0] in targets}
    result["ignored_unrelated_mapping_count"] = ignored_mappings
    result["request_mapping"] = {k: {"http_request_id": v[0], "session_id": v[1]} for k, v in relevant.items()}
    if {pair[0] for pair in relevant.values()} != set(targets):
        errors.append("Both current HTTP requests need generation.sequence.bound mappings")

    seen_batch, last_native, last_positions = set(), 0, {}
    native_ids, covered, pair_steps = set(), set(), {}
    current_run, current_pair = None, None
    for line_number, record in enumerate(records, 1):
        if not event_is(record, BATCH_EVENT):
            continue
        props = record.get("props", {})
        try:
            ids = array(props, "SequenceRequestIdsJson")
            if any(not isinstance(i, str) or not i for i in ids):
                raise ValueError("Sequence IDs must be nonempty strings")
            # Other model traffic is not used to qualify this test. Malformed
            # evidence that purports to include a current request fails closed.
            if not set(ids).intersection(relevant):
                continue
            if len(ids) < 2 or len(set(ids)) != len(ids):
                raise ValueError("A joint graph needs at least two distinct sequence IDs")
            before = array(props, "PositionsBeforeJson")
            after = array(props, "PositionsAfterJson")
            count = props.get("ActiveSequenceCount")
            if not integer(count, 2) or count != len(ids) or len(before) != count or len(after) != count:
                raise ValueError("Sequence count and position-array widths disagree")
            if any(not integer(a) or not integer(b) or b != a + 1 for a, b in zip(before, after)):
                raise ValueError("Each joint decode must advance exactly one token")
            if props.get("ManagedPublishCompleted") is not True:
                raise ValueError("Native execution without completed managed publication is not acceptance")
            native_id = native_execution_id(props.get("NativeAcceptedExecutionId"))
            nodes = props.get("NativeNodeCount")
            graphs = props.get("NativeGraphCount")
            segments = props.get("NativeHostSegmentCount")
            execution_kind = props.get("NativeExecutionKind")
            batch_id = props.get("BatchExecutionId")
            if not ((integer(batch_id, 1)) or (isinstance(batch_id, str) and bool(batch_id))):
                raise ValueError("Missing batch execution identity")
            if batch_id in seen_batch or native_id in native_ids or native_id <= last_native:
                raise ValueError("Duplicate or non-increasing successful execution identity")
            if not integer(nodes, 1) or not integer(graphs, 1) or graphs != 1 or not integer(segments):
                raise ValueError("Actual native node count, one logical graph and explicit segment count are required")
            if (not integer(execution_kind) or execution_kind not in (0, 1)
                    or (execution_kind == 1) != (segments > 0)):
                raise ValueError("Native whole-graph/host-segmented kind disagrees with its seam count")
            seen_batch.add(batch_id); native_ids.add(native_id); last_native = native_id
            owners = {bindings[i][0] for i in ids if i in bindings}
            # Only graphs containing both targeted HTTP generations establish
            # the requested engagement. Other graph traffic remains separate.
            if not set(targets).issubset(owners):
                continue
            pair = tuple(tuple(i for i in ids if bindings.get(i, (None,))[0] == owner)
                         for owner in targets)
            if any(len(members) != 1 for members in pair):
                raise ValueError("Each targeted HTTP generation needs one unambiguous joint member")
            gaps = []
            for internal, start, end in zip(ids, before, after):
                if internal in relevant:
                    previous = last_positions.get(internal)
                    if previous is not None and start < previous:
                        raise ValueError("Current generation position went backwards/replayed between native acceptances")
                    if previous is not None and previous != start:
                        gaps.append({"internal_request_id": internal, "previous_after": previous,
                                     "current_before": start, "untraced_tokens": start - previous})
            if gaps:
                result["continuity_gaps"].append({"line": line_number,
                    "batch_execution_id": batch_id, "native_accepted_execution_id": native_id,
                    "members": gaps,
                    "scope": "Position advancement absent from accepted joint trace; the log alone does not identify its cause."})
            # A forward gap is not a fabricated native success. Retain it and
            # start a new interval, while the strict final gate still fails.
            if gaps or pair != current_pair or current_run is None:
                current_run = {"sequence_request_ids": [members[0] for members in pair],
                    "steps": 0, "first_line": line_number, "last_line": line_number,
                    "first_native_id": native_id, "last_native_id": native_id}
                result["contiguous_joint_runs"].append(current_run)
                current_pair = pair
            current_run["steps"] += 1
            current_run["last_line"] = line_number
            current_run["last_native_id"] = native_id
            for internal, start, end in zip(ids, before, after):
                if internal in relevant:
                    last_positions[internal] = end
                    covered.add(bindings[internal][0])
            pair_steps[pair] = pair_steps.get(pair, 0) + 1
            result["matched_executions"].append({
                "line": line_number, "batch_execution_id": batch_id,
                "native_accepted_execution_id": native_id, "sequence_request_ids": ids,
                "positions_before": before, "positions_after": after,
                "native_node_count": nodes, "native_logical_graph_count": graphs,
                "native_host_segment_count": segments, "native_execution_kind": execution_kind})
        except (ValueError, TypeError, AttributeError) as error:
            errors.append(f"Batch line {line_number}: {error}")
    count = len(result["matched_executions"])
    repeated_steps = max(pair_steps.values(), default=0)
    contiguous_steps = max((run["steps"] for run in result["contiguous_joint_runs"]), default=0)
    result.update(joint_steps=count, maximum_same_generation_pair_steps=repeated_steps,
                  maximum_contiguous_joint_steps=contiguous_steps,
                  minimum_joint_steps=minimum_steps,
                  native_segmented_execution_observed=any(r["native_host_segment_count"] > 0 for r in result["matched_executions"]))
    if contiguous_steps < minimum_steps or covered != set(targets):
        errors.append(f"Need at least {minimum_steps} consecutive completed joint steps of the same generation pair containing both current HTTP requests")
    result["actual_joint_engagement_verified"] = not errors
    result["full_interval_continuity_verified"] = not errors and not result["continuity_gaps"]
    for gap in result["continuity_gaps"]:
        errors.append(f"Batch line {gap['line']}: Untraced/serial position gap inside the qualifying joint interval")
    result["passed"] = result["batched_fused_graph_verified"] = result["full_interval_continuity_verified"]
    return result


def read_input(path, limit):
    with Path(path).open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"Input exceeds {limit} bytes: {path}")
    return payload, {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency-report", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--output", required=True, help="Fresh JSON in ignored artifacts/ or docs/validation/")
    parser.add_argument("--min-joint-steps", type=int, default=2)
    args = parser.parse_args(argv)
    lexical = Path(os.path.abspath(args.output))
    if ".." in Path(args.output).parts or not any(lexical.is_relative_to(ROOT / name) for name in ("artifacts", "docs/validation")):
        parser.error("Output must be inside ignored validation evidence")
    if not integer(args.min_joint_steps, 2):
        parser.error("min-joint-steps must be at least 2")
    result = {"passed": False, "batched_fused_graph_verified": False}
    try:
        report_bytes, report_pin = read_input(args.concurrency_report, 32 << 20)
        log_bytes, log_pin = read_input(args.server_log, MAX_LOG_BYTES)
        records = [json.loads(line) for line in log_bytes.decode("utf-8-sig").splitlines() if line.strip()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("Every JSONL record must be an object")
        result = verify(json.loads(report_bytes.decode("utf-8-sig")), records, args.min_joint_steps)
        result["inputs"] = {"concurrency_report": report_pin, "server_log": log_pin}
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lexical.parent.mkdir(parents=True, exist_ok=True)
    with lexical.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({key: result.get(key) for key in ("passed", "joint_steps", "errors", "error")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
