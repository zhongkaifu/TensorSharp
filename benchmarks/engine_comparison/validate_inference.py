#!/usr/bin/env python3
"""Strict, repeatable quality and HTTP performance checks against a running server.

No server is started and no model is downloaded. See README.md for matching
launch settings and the distinction between layer split and tensor parallelism.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

import engines
import scenarios

SCENARIOS = ("short", "short_zh", "decode", "decode_8k", "json", "json_schema", "json_unicode", "long_8k", "long_32k", "long_64k",
             "multi_turn", "tool_round_trip", "agentic")
SAMPLING = {"temperature": 0, "top_p": 1, "top_k": 0, "seed": 42,
            "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def exact_json(text, expected):
    try:
        return json.loads(text) == expected
    except (ValueError, TypeError):
        return False


def function(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


WEATHER_TOOL = function("get_weather", "Look up the current temperature.",
                       {"city": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius"]}},
                       ["city", "units"])
INVOICE_TOOL = function("read_invoice", "Read an invoice by its identifier.",
                       {"invoice_id": {"type": "string"}}, ["invoice_id"])
TOTAL_TOOL = function("calculate_total", "Calculate the total for an invoice.",
                     {"unit_price": {"type": "number"},
                      "quantity": {"type": "integer"}}, ["unit_price", "quantity"])


def case_spec(name, tag):
    # A different leading marker for every trial/client prevents one engine's
    # automatic prefix cache from turning repeated prefill into a cache benchmark.
    # The marker is deterministic and identical across the compared engines.
    prefix = f"[validation {tag}]\n"
    spec = {"messages": [], "max_tokens": 256, "tools": None, "response_format": None}
    if name == "short":
        prompt = "What is 17 + 25? Reply with only the integer."
    elif name == "short_zh":
        prompt = "用阿拉伯数字回答：十七加二十五等于多少？只输出结果。"
    elif name in ("decode", "decode_8k"):
        prompt = ("Write a detailed explanation of how a hash table works, including collisions, "
                  "resizing, complexity, and a worked example. Continue until the token limit.")
        spec["max_tokens"] = 512
        if name == "decode_8k":
            prompt = (scenarios._sliced_corpus(8192) +
                      "\nThe preceding document is background. For your answer, follow this request:\n" + prompt)
    elif name in ("json", "json_schema"):
        prompt = ('Return exactly a JSON object with name "Mars", moons 2, and habitable false. '
                  'Use exactly those three keys and no other text.')
        spec["response_format"] = {"type": "json_object"}
        if name == "json_schema":
            spec["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "planet", "strict": True, "schema": {"type": "object",
                "properties": {"name": {"type": "string", "enum": ["Mars"]},
                               "moons": {"type": "integer", "enum": [2]},
                               "habitable": {"type": "boolean", "enum": [False]}},
                "required": ["name", "moons", "habitable"], "additionalProperties": False}}}
    elif name == "json_unicode":
        prompt = '把以下字段原样写入 JSON，不添加其他内容：city 为 "北京"，message 为 "你好，世界"，emoji 为 "🚀"。'
        spec["response_format"] = {"type": "json_object"}
    elif name.startswith("long_"):
        target = int(name.split("_")[1][:-1]) * 1024
        doc = scenarios._sliced_corpus(target)
        facts = ["The verification code for ALPHA is silver-4821.",
                 "The verification code for BETA is amber-7392.",
                 "The verification code for GAMMA is violet-1605."]
        for fraction, fact in reversed(list(zip((0.1, 0.5, 0.9), facts))):
            index = int(len(doc) * fraction)
            doc = doc[:index] + "\n" + fact + "\n" + doc[index:]
        prompt = (doc + '\nReturn only a JSON object mapping ALPHA, BETA, GAMMA to their '
                  'verification codes from this document. Do not guess.')
    elif name == "multi_turn":
        prompt = 'Remember the secret code juniper-5938. Acknowledge with exactly "OK".'
    elif name == "tool_round_trip":
        prompt = ("Use get_weather for Paris in celsius, then report the returned temperature "
                  'as a JSON object with exactly city and temperature_c. Do not invent weather.')
        spec["tools"] = [WEATHER_TOOL]
    elif name == "agentic":
        prompt = ("Read invoice INV-472 with read_invoice, then pass its returned unit_price "
                  "and quantity to calculate_total. You must call both tools in that order. "
                  'Finally return only JSON with invoice_id and total using the tool results.')
        spec["tools"] = [INVOICE_TOOL, TOTAL_TOOL]
    else:
        raise ValueError(f"unknown scenario {name}")
    spec["messages"] = [{"role": "user", "content": prefix + prompt}]
    return spec


def assistant_content(metrics):
    """Return only the API assistant content channel for correctness checks.

    engines.output_text is a display/throughput convenience and can fall back
    to reasoning-only output. Keep that diagnostic field and all token/timing
    metrics intact, but never use it as the final answer.
    """
    message = metrics.get("assistant_message")
    if not isinstance(message, dict):
        raise ValueError("server omitted the assistant message")
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("assistant content must be text or null")
    return content


def strip_json_fence(text):
    """The JSON inside a single ```json (or bare ```) fence, or None.

    Only whitespace may surround the fence and only one fence may be present;
    anything else is not "a fence around otherwise exact JSON" and stays a
    strict failure. Used by --accept-fenced-json for the lenient status only.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```") or stripped.count("```") != 2:
        return None
    body = stripped[3:-3]
    first_line, newline, rest = body.partition("\n")
    if not newline:
        return None
    if first_line.strip().lower() not in ("", "json"):
        return None
    return rest.strip()


def check_answer(name, text):
    if name in ("short", "short_zh"):
        return text.strip() == "42"
    if name in ("decode", "decode_8k"):
        lower = text.lower()
        return len(text) > 300 and "hash" in lower and "collision" in lower
    if name in ("json", "json_schema"):
        # Python's equality conflates false and 0: enforce JSON types too.
        if not exact_json(text, {"name": "Mars", "moons": 2, "habitable": False}):
            return False
        data = json.loads(text)
        return type(data["moons"]) is int and type(data["habitable"]) is bool
    if name == "json_unicode":
        return exact_json(text, {"city": "北京", "message": "你好，世界", "emoji": "🚀"})
    if name.startswith("long_"):
        return exact_json(text, {"ALPHA": "silver-4821", "BETA": "amber-7392",
                                "GAMMA": "violet-1605"})
    if name == "multi_turn":
        return exact_json(text, {"code": "juniper-5938"})
    if name == "tool_round_trip":
        return exact_json(text, {"city": "Paris", "temperature_c": 19})
    if name == "agentic":
        return exact_json(text, {"invoice_id": "INV-472", "total": 68.75})
    return False


def execute_tool(name, metrics, step):
    """Validate full OpenAI tool structure before returning deterministic fixture data.

    These tools only return fixtures. Model output never executes shell/code or
    accesses local files, and no finish_reason-only shortcut can pass.
    """
    calls = metrics.get("tool_call_details") or []
    if metrics.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise ValueError("expected exactly one structured tool call and finish_reason=tool_calls")
    call = calls[0]
    if not isinstance(call.get("id"), str) or not call["id"] or call.get("type") != "function":
        raise ValueError("tool call needs a nonempty id and type=function")
    fn = call.get("function") or {}
    arguments = json.loads(fn.get("arguments", ""))
    if name == "tool_round_trip":
        expected_name, expected_args = "get_weather", {"city": "Paris", "units": "celsius"}
        result = {"city": "Paris", "temperature_c": 19}
    elif step == 0:
        expected_name, expected_args = "read_invoice", {"invoice_id": "INV-472"}
        result = {"invoice_id": "INV-472", "unit_price": 13.75, "quantity": 5}
    else:
        expected_name, expected_args = "calculate_total", {"unit_price": 13.75, "quantity": 5}
        result = {"total": 68.75}
    if fn.get("name") != expected_name or arguments != expected_args:
        raise ValueError(f"expected {expected_name}({expected_args}), got {fn}")
    return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}


def run_case(url, model, engine, name, tag, thinking=False, stream=True, timeout=1200,
             structured_tool_results=False, max_tokens=None, serial_tool_workflows=False,
             accept_fenced_json=False):
    spec = case_spec(name, tag)
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        spec["max_tokens"] = max_tokens
    initial = {**spec, "sampling": SAMPLING, "thinking": thinking, "stream": stream}
    if structured_tool_results:
        initial["structured_tool_results"] = True
    serial_workflow = bool(serial_tool_workflows and spec["tools"] and
                           name in ("tool_round_trip", "agentic"))
    if serial_workflow:
        initial["serial_tool_workflows"] = True
    result = {"scenario": name, "tag": tag, "status": "fail",
              "input_sha256": digest(initial), "turns": []}
    messages = list(spec["messages"])
    started = time.monotonic()
    try:
        count = {"multi_turn": 2, "tool_round_trip": 2, "agentic": 3}.get(name, 1)
        for step in range(count):
            extra = {**SAMPLING, **engines.thinking_body(engine, thinking)}
            if serial_workflow:
                # Dependent fixture calls must consume the previous result.
                # This is a client constraint; execute_tool still validates
                # exact call count, order and arguments on every response.
                extra["parallel_tool_calls"] = False
            # Final tool answer has the same explicit output contract in both engines.
            response_format = spec["response_format"]
            if name in ("tool_round_trip", "agentic") and step == count - 1:
                extra["tool_choice"] = "none"
                if structured_tool_results:
                    response_format = {"type": "json_object"}
            request = {"messages": messages.copy(), "tools": spec["tools"],
                       "response_format": response_format, "extra_body": extra,
                       "max_tokens": spec["max_tokens"], "stream": stream}
            metrics = engines.run_openai_chat(url, model, timeout_s=timeout, **request)
            result["turns"].append({"request": request, "metrics": metrics})
            if not metrics.get("usage_present"):
                raise ValueError("server omitted token usage; token throughput cannot be validated")
            if not metrics.get("finish_reason"):
                raise ValueError("completion ended without a finish reason")
            if name in ("tool_round_trip", "agentic") and step < count - 1:
                tool_result = execute_tool(name, metrics, step)
                messages.extend([metrics["assistant_message"], tool_result])
            elif name == "multi_turn" and step == 0:
                if assistant_content(metrics).strip() != "OK":
                    raise ValueError("first conversation turn did not acknowledge the code")
                messages.extend([metrics["assistant_message"], {"role": "user", "content":
                    'What was the code? Return only JSON with one key "code".'}])
        # Never count reasoning-only output as a correct final answer.
        text = assistant_content(metrics)
        result["validated_content"] = text
        truncated = metrics["finish_reason"] == "length" and name not in ("decode", "decode_8k")
        if not check_answer(name, text):
            # Opt-in secondary verdict: a ```json fence around otherwise exact JSON
            # is recorded as lenient-ok. The strict status stays the case status.
            if accept_fenced_json and not truncated:
                unfenced = strip_json_fence(text)
                if unfenced is not None and check_answer(name, unfenced):
                    result["lenient_status"] = "ok"
                    result["lenient_detail"] = "exact JSON inside a code fence"
            raise ValueError("final answer failed the scenario's semantic/structural check")
        if truncated:
            raise ValueError("answer reached the token limit")
        result["status"] = "ok"
    except Exception as error:
        result["detail"] = f"{type(error).__name__}: {error}"
    if accept_fenced_json:
        result.setdefault("lenient_status", result["status"])
    result["total_wall_ms"] = (time.monotonic() - started) * 1000
    return result


def summarize(cases):
    groups = {}
    for case in cases:
        key = f"{case['scenario']}@c{case.get('concurrency', 1)}"
        groups.setdefault(key, []).append(case)
    summaries = {}
    for key, cells in groups.items():
        passed = [c for c in cells if c["status"] == "ok"]
        metrics = [c["turns"][0]["metrics"] for c in passed]
        item = {"passed": len(passed), "total": len(cells)}
        if any("lenient_status" in c for c in cells):
            item["lenient_passed"] = sum(c.get("lenient_status") == "ok" for c in cells)
        for field in ("ttft_ms", "prefill_tps", "decode_tps"):
            values = [m[field] for m in metrics if m.get(field, 0) > 0]
            if values:
                item[field + "_median"] = statistics.median(values)
                item[field + "_samples"] = values
        if passed:
            item["conversation_wall_ms_median"] = statistics.median(c["total_wall_ms"] for c in passed)
        summaries[key] = item
    return summaries


def compare(current, reference, tolerance):
    """A comparison is inconclusive if provenance, coverage or output checks differ."""
    errors = []
    for label, data in (("current", current), ("reference", reference)):
        if data.get("run_complete") is False:
            errors.append(f"{label}: benchmark run is incomplete")
        expected = data.get("execution_plan", {}).get("expected_cases")
        if expected is not None and len(data["cases"]) != expected:
            errors.append(f"{label}: measured cases do not match the execution plan")
    for field in ("weights_id", "profile", "thinking", "stream", "sampling"):
        if current.get(field) != reference.get(field):
            errors.append(f"mismatched {field}")
    if bool(current.get("structured_tool_results")) != bool(reference.get("structured_tool_results")):
        errors.append("mismatched structured_tool_results")
    if bool(current.get("serial_tool_workflows")) != bool(reference.get("serial_tool_workflows")):
        errors.append("mismatched serial_tool_workflows")
    if not current.get("weights_id") or current.get("weights_id") == "unspecified":
        errors.append("weights_id must identify the exact checkpoint/quantization/revision")
    def keyed(data):
        return {(c["scenario"], c["tag"], c["concurrency"]): c for c in data["cases"]}
    left, right = keyed(current), keyed(reference)
    if not left or not right:
        errors.append("both engines must have measured cases; unavailable references cannot establish parity")
    if left.keys() != right.keys():
        errors.append("scenario/trial/concurrency coverage differs")
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if a["input_sha256"] != b["input_sha256"]:
            errors.append(f"{key}: request inputs differ")
        if a["status"] != "ok" or b["status"] != "ok":
            errors.append(f"{key}: both outputs must pass correctness checks")
        for at, bt in zip(a["turns"], b["turns"]):
            am, bm = at["metrics"], bt["metrics"]
            if am.get("prompt_tokens") != bm.get("prompt_tokens"):
                errors.append(f"{key}: tokenized prompt lengths differ; inspect chat templates")
            if am.get("decode_timing_source") != bm.get("decode_timing_source"):
                errors.append(f"{key}: decode timer sources differ")
    ratios = {}
    if not errors:
        for key, a in summarize(current["cases"]).items():
            b = summarize(reference["cases"])[key]
            ratios[key] = {}
            # Decode-only case provides enough output for a meaningful decode rate.
            fields = ["conversation_wall_ms_median", "ttft_ms_median"]
            if key.startswith(("decode@", "decode_8k@")):
                fields.append("decode_tps_median")
            for field in fields:
                if a.get(field, 0) <= 0 or b.get(field, 0) <= 0:
                    continue
                ratio = a[field] / b[field] if "tps" in field else b[field] / a[field]
                ratios[key][field + "_speedup"] = ratio
                if ratio < 1 - tolerance:
                    errors.append(f"{key}: {field} speedup {ratio:.3f} below {1-tolerance:.3f}")
    return {"status": "pass" if not errors else "fail", "errors": errors, "ratios": ratios,
            "tolerance": tolerance,
            "scope": "Fixture correctness and measured HTTP latency/throughput; not general model-quality equivalence."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", choices=("tensorsharp", "llamacpp"), required=True)
    ap.add_argument("--model", required=True, help="served model id from /v1/models")
    ap.add_argument("--weights-id", required=True, help="same immutable revision + quantization or shard hashes on both engines")
    ap.add_argument("--profile", required=True, help="same placement/context/batch/cache settings on both engines")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int,
                    help="Override every turn's generation budget, including reasoning; recorded in request hashes")
    ap.add_argument("--blocking", action="store_true")
    ap.add_argument("--structured-tool-results", action="store_true",
                    help="Request JSON mode on the final tool-workflow answer; retain strict output checks")
    ap.add_argument("--serial-tool-workflows", action="store_true",
                    help="Send parallel_tool_calls=false on tool-bearing workflow turns; preserve exact call order/argument checks")
    ap.add_argument("--accept-fenced-json", action="store_true",
                    help="Also record a lenient per-case status that accepts exact JSON wrapped in a ```json fence; "
                         "the strict status remains the case status and the exit code")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    names = args.scenarios.split(",")
    degrees = [int(value) for value in args.concurrency.split(",")]
    if args.repeats < 1 or any(n < 1 for n in degrees) or not 0 <= args.tolerance < 1:
        ap.error("repeats/concurrency must be positive and tolerance in [0,1)")
    if args.max_tokens is not None and args.max_tokens < 1:
        ap.error("max-tokens must be positive")
    if set(names) - set(SCENARIOS):
        ap.error(f"unknown scenarios: {set(names) - set(SCENARIOS)}")
    report = {"format_version": 1, "engine": args.engine, "url": args.url,
              "harness_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in (__file__, engines.__file__, scenarios.__file__)},
              "model": args.model, "weights_id": args.weights_id, "profile": args.profile,
              "thinking": args.thinking, "stream": not args.blocking, "sampling": SAMPLING,
              "structured_tool_results": args.structured_tool_results,
              "serial_tool_workflows": args.serial_tool_workflows,
              "accept_fenced_json": args.accept_fenced_json,
              "max_tokens_override": args.max_tokens,
              "execution_plan": {"scenarios": names, "concurrency": degrees,
                  "repeats": args.repeats,
                  "expected_cases": len(names) * sum(degrees) * args.repeats},
              "run_complete": False,
              "cases": [], "waves": [], "not_covered": {"image": "requires a released, compatible vision companion and separate media validation",
              "audio": "no audio validation in this suite", "video": "requires compatible vision companion and separate media validation"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Warm graph allocation with a separate prompt. Keep the result for auditing,
    # but never include it in measured summaries.
    report["warmup"] = run_case(args.url, args.model, args.engine, "short", "warmup",
                               args.thinking, not args.blocking, args.timeout,
                               max_tokens=args.max_tokens)
    for name in names:
        for concurrency in degrees:
            for repeat in range(args.repeats):
                start = time.monotonic()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    jobs = [pool.submit(run_case, args.url, args.model, args.engine,
                            name, f"{name}-c{concurrency}-r{repeat}-i{i}", args.thinking,
                            not args.blocking, args.timeout, args.structured_tool_results,
                            args.max_tokens, args.serial_tool_workflows,
                            args.accept_fenced_json) for i in range(concurrency)]
                    cases = [job.result() for job in jobs]
                wall = time.monotonic() - start
                for case in cases:
                    case.update(concurrency=concurrency, repeat=repeat)
                report["cases"].extend(cases)
                tokens = sum(t["metrics"].get("completion_tokens", 0)
                             for c in cases for t in c["turns"])
                report["waves"].append({"scenario": name, "concurrency": concurrency,
                    "repeat": repeat, "wall_ms": wall * 1000,
                    "generated_tokens": tokens, "end_to_end_tokens_per_second": tokens / wall,
                    "all_passed": all(c["status"] == "ok" for c in cases)})
                lenient = (f"; lenient {sum(c.get('lenient_status') == 'ok' for c in cases)}/{concurrency}"
                           if args.accept_fenced_json else "")
                print(f"{name} c{concurrency} repeat{repeat}: "
                      f"{sum(c['status'] == 'ok' for c in cases)}/{concurrency} passed{lenient}; {wall:.2f}s", flush=True)
                report["summary"] = summarize(report["cases"])
                args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report["run_complete"] = True
    if args.reference:
        report["comparison"] = compare(report, json.loads(args.reference.read_text()), args.tolerance)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    failed = any(c["status"] != "ok" for c in report["cases"])
    failed |= report.get("comparison", {}).get("status") == "fail"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
