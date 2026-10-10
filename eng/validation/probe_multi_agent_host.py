"""Exercise real TensorSharp Web UI SSE delegation, opt-out, and disconnect cleanup.

Requires a running tool-capable model. Results are model-dependent; failure is
reported rather than counting an unavailable model or missing delegation as a pass.
"""
import argparse
import json
import pathlib
import time
import urllib.request


def shown_answer(frames):
    """The answer a page is left showing: tokens append, a replace sets the whole of it
    (a host takes back text that proved to be reasoning that way)."""
    answer = ""
    for frame in frames:
        if isinstance(frame.get("token"), str):
            answer += frame["token"]
        if isinstance(frame.get("replace"), str):
            answer = frame["replace"]
    return answer


def request(base, prompt, *, enabled=True, cancel_on_spawn=False, max_tokens=768):
    body = {"messages": [{"role": "user", "content": prompt}], "think": False,
            "maxTokens": max_tokens, "newChat": True, "multi_agent": enabled}
    req = urllib.request.Request(base.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    frames = []
    cancelled = False
    with urllib.request.urlopen(req, timeout=240) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: {"):
                continue
            frame = json.loads(line[6:])
            frames.append(frame)
            if cancel_on_spawn and frame.get("skill_step") == "spawn_agent" and frame.get("ok"):
                # Let the admitted child begin; closing this streaming response
                # disconnects the parent and must cancel its descendants.
                time.sleep(1)
                cancelled = True
                break
    answer = shown_answer(frames)
    return {"seconds": time.perf_counter() - started, "cancelled_on_spawn": cancelled,
            "answer": answer, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5037")
    parser.add_argument("--out", default="docs/validation/multi-agent/host-final-probe.json")
    parser.add_argument("--complex-only", action="store_true",
        help="Run one single-agent and one automatic code-review task; report model delegation without forcing it.")
    parser.add_argument("--max-tokens", type=int, default=768,
        help="Output budget for each arm of --complex-only; server cap must permit it.")
    args = parser.parse_args()
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    try:
        if args.complex_only:
            fixture = pathlib.Path(__file__).resolve().parents[2] / "InferenceWeb.Tests/Fixtures/MultiAgentHost/complex-review.md"
            prompt = fixture.read_text(encoding="utf-8")
            for name, enabled in (("complex_single", False), ("complex_auto", True)):
                result = request(args.endpoint, prompt, enabled=enabled, max_tokens=args.max_tokens)
                result["spawn_count"] = sum(f.get("skill_step") == "spawn_agent" and f.get("ok", False)
                                            for f in result["frames"])
                result["done"] = [f for f in result["frames"] if f.get("done")]
                report[name] = result
            report["assessment"] = "One qualitative pair, not a performance benchmark. Review answers against the fixture's concurrency/crash cases; zero spawns means autonomous delegation was not exercised."
            report["requested_max_tokens"] = args.max_tokens
            report["completed"] = True
            return
        report["simple_auto"] = request(args.endpoint, "What is 17+25? Reply with just the number.")
        assert report["simple_auto"]["answer"].strip() == "42"
        assert not any(f.get("skill_step") == "spawn_agent" for f in report["simple_auto"]["frames"])
        report["delegated"] = request(args.endpoint,
            "Use two independent subagents to verify two proposals. Give each all its numbers. "
            "Spawn both before waiting, then collect both results and synthesize a concise table. "
            "Proposal A sells 120 units for $25 each, pays $13 per unit and $300 fixed cost, "
            "and claims profit $1300. Proposal B has 200 customers paying $18 each, pays $7 "
            "per customer and $400 fixed cost, and claims profit $1900. Ask each child to "
            "calculate revenue, cost, profit and exact profit overstatement. Final answer must "
            "include all these numbers for both proposals. Do not guess the causes of the errors.")
        delegated = report["delegated"]
        assert sum(f.get("skill_step") == "spawn_agent" and f.get("ok", False)
                   for f in delegated["frames"]) >= 2, "Model did not successfully delegate twice"
        normalized = delegated["answer"].replace(",", "")
        assert all(number in normalized for number in ("3000", "1860", "1140", "160", "3600", "1800", "100")), normalized
        assert any(f.get("done") and not f.get("aborted") and not f.get("error") for f in delegated["frames"])
        report["disconnect"] = request(args.endpoint,
            "Immediately spawn one subagent named long_review to write a detailed 1200-word "
            "review of database transaction isolation, including five independent examples. "
            "Give it the full task and then wait for its result. Do not answer the review yourself.",
            cancel_on_spawn=True)
        assert report["disconnect"]["cancelled_on_spawn"], "Model did not spawn before disconnect"
        time.sleep(2)
        report["after_disconnect_single"] = request(args.endpoint,
            "What is 6 times 7? Reply with just the number.", enabled=False)
        assert report["after_disconnect_single"]["answer"].strip() == "42"
        assert not any(f.get("skill_step") == "spawn_agent" for f in report["after_disconnect_single"]["frames"])
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: {"seconds": value.get("seconds"), "answer": value.get("answer")}
                          for key, value in report.items() if isinstance(value, dict)}, indent=2))


if __name__ == "__main__":
    main()
