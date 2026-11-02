#!/usr/bin/env python3
"""Render offline PNG/SVG scientific figures from original engine-bench JSONL.

No engine is run and no response is rescored. The existing report parser and
original quality gate are reused. Matplotlib is imported only when rendering;
--data-only and data-extraction tests need only the Python standard library.

  python eng/qwen38-engine-figures.py 'docs/validation/run/raw/*.jsonl' \
      --output-dir docs/validation/run/figures --exclude-label '*warmup*'
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
VERSION = "qwen38-engine-figures/1"
REPORT_PATH = Path(__file__).with_name("qwen38-eval-report.py")
spec = importlib.util.spec_from_file_location("qwen38_figure_report", REPORT_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
metrics = report.metrics
KINDS = ("code", "prose", "copy")
MODES = ("plain", "dflash2")
LONG = ("long_prompt_128k", "long_prompt_250k")
RATE_SOURCE = "tokenizer_visible_content_minus_first_semantic_payload"
TOKEN_SOURCES = ("usage", "tokenizer", "server_timings")
LIMITATIONS = [
    "Original benchmark checks are reused without rescoring. In particular, the code check is not a proof of program correctness.",
    "Speed is withheld unless all expected repeats pass every original check. Decode additionally requires >=128 completion and visible tokens, reliable token counts, and the recorded v3 first-payload tokenizer evidence.",
    "Decode is a client stream estimate after the first semantic payload, not GPU kernel throughput; speculative output bursts affect it. TTFT includes client/server overhead.",
    "95% intervals bootstrap whole repeats (2000 deterministic resamples), not individual tokens. Three repeats provide weak uncertainty evidence; no ranking or winner is inferred.",
    "GSM8K repeats reuse the same public question IDs. Eight questions repeated three times are eight unique questions, not 24 independent questions. Quality figures show no confidence intervals.",
    "Strict generated-task and GSM8K counts retain their original scorer, including format failures; any request.error withholds the whole quality point as availability evidence, not a model zero. These small tests do not establish comprehensive model quality.",
    "Long-context TTFT is shown only after the original retrieval/order/token-budget checks pass. Missing, failed, incomplete and unsupported records produce no speed point or zero substitute.",
    "Checkpoints, weight/KV precision, drafters, contexts and active lanes can differ between labelled configurations. These are stack comparisons, not isolated universal engine rankings.",
]


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def complete_repeats(header, cases):
    expected = header.get("repeats")
    ids = [case.get("repeat") for case in cases]
    return (integer(expected, 1) and len(ids) == expected and all(integer(i) for i in ids)
            and set(ids) == set(range(expected)))


def selected(label, include, exclude):
    return ((not include or any(fnmatch.fnmatchcase(label, pattern) for pattern in include))
            and not any(fnmatch.fnmatchcase(label, pattern) for pattern in exclude))


def identity(key, run, scenario, summaries):
    row = summaries.get(scenario, {})
    return {"engine": key[0], "mode": key[1], "label": key[2], "scenario": scenario,
            "run_id": run["header"].get("run_id"), "source": run["source"],
            "original_gate": row.get("status", "NOT_RUN"),
            "expected_repeats": run["header"].get("repeats"),
            "qualified": False, "withheld_reason": row.get("reason", "scenario not recorded")}


def source_ref(case, request):
    return {"repeat": case.get("repeat"), "request_sha256": request.get("request_sha256"),
            "content_sha256": request.get("content_sha256"),
            "response_sha256": request.get("response_sha256")}


def decode_reason(request, minimum):
    """Additional display qualification; never replace the original scorer."""
    if (not integer(request.get("completion_tokens"), minimum)
            or request.get("completion_tokens_source") not in TOKEN_SOURCES):
        return "completion token count is missing, unreliable or below minimum"
    if request.get("decode_tok_s_source") != RATE_SOURCE or request.get("decode_numerator_source") != RATE_SOURCE:
        return "v3 tokenizer-based post-first-payload decode rate is unavailable"
    if (request.get("decode_token_prefix_stable") is not True
            or not integer(request.get("visible_content_tokens"), minimum)
            or not integer(request.get("first_payload_content_tokens"), 1)
            or not integer(request.get("decode_numerator_tokens"), 1)):
        return "tokenizer prefix/count evidence is incomplete"
    if request["visible_content_tokens"] - request["first_payload_content_tokens"] != request["decode_numerator_tokens"]:
        return "post-first-payload token counts disagree"
    rate, duration = request.get("decode_tok_s"), request.get("decode_duration_s")
    if not report.finite(rate) or rate <= 0 or not report.finite(duration) or duration <= 0:
        return "decode rate/duration is not finite and positive"
    if not math.isclose(rate, request["decode_numerator_tokens"] / duration, rel_tol=1e-8, abs_tol=1e-8):
        return "recorded decode rate disagrees with its numerator and duration"
    if (request.get("decode_unavailable_reason") or request.get("error")
            or not integer(request.get("n_content_deltas"), 2)
            or not integer(request.get("n_model_deltas"), 2)
            or request.get("n_reasoning_deltas", 0) or request.get("n_tool_deltas", 0)
            or request.get("reasoning_chars", 0)):
        return "request is not an eligible complete pure-content decode stream"
    return None


def decode_rows(key, run, summaries, minimum):
    cases = [case for case in run["cases"] if case["scenario"] == "long_generation"]
    rows = []
    for kind in KINDS:
        row = identity(key, run, "long_generation", summaries)
        row.update(kind=kind, median=None, ci95=None, repeat_samples=[])
        rows.append(row)
    if not complete_repeats(run["header"], cases) or rows[0]["original_gate"] != "PASS":
        if not complete_repeats(run["header"], cases):
            for row in rows:
                row["withheld_reason"] = "expected repeats are missing, duplicated or invalid"
        return rows
    mapped = []
    for case in sorted(cases, key=lambda case: case["repeat"]):
        per = case.get("metrics", {}).get("per_prompt", [])
        reqs = case.get("requests", [])
        if (not isinstance(per, list) or len(per) != 3 or len(reqs) != 3
                or any(not isinstance(item, dict) for item in per)
                or sorted(item.get("kind", "") for item in per) != sorted(KINDS)):
            for row in rows:
                row["withheld_reason"] = "missing or ambiguous code/prose/copy request mapping"
            return rows
        pairs = {}
        for item, request in zip(per, reqs):
            if (not item.get("content_sha256") or item["content_sha256"] != request.get("content_sha256")
                    or item.get("completion_tokens") != request.get("completion_tokens")
                    or item.get("decode_tok_s") != request.get("decode_tok_s")):
                for row in rows:
                    row["withheld_reason"] = "per-prompt metadata does not match its raw request"
                return rows
            pairs[item["kind"]] = request
        mapped.append((case, pairs))
    for row in rows:
        samples, reason = [], None
        for case, pairs in mapped:
            request = pairs[row["kind"]]
            reason = decode_reason(request, minimum)
            if reason:
                break
            samples.append({**source_ref(case, request), "value": request["decode_tok_s"],
                            "completion_tokens": request["completion_tokens"],
                            "visible_content_tokens": request["visible_content_tokens"]})
        if reason:
            row["withheld_reason"] = reason
            continue
        low, high = metrics.repeat_median_ci([[sample["value"]] for sample in samples])
        row.update(qualified=True, withheld_reason=None, median=statistics.median(s["value"] for s in samples),
                   ci95=[low, high] if low is not None else None, repeat_samples=samples)
    return rows


def quality_row(key, run, scenario, summaries):
    row = identity(key, run, scenario, summaries)
    cases = [case for case in run["cases"] if case["scenario"] == scenario]
    row.update(report.functional_answer_counts(run["header"], cases, scenario))
    return row


def quality_withheld_label(row):
    availability = row.get("request_availability", {})
    if row.get("withheld_reason_code") == "request_error":
        return f"withheld (request errors {availability['requests_with_errors']}/{availability['requests_observed']})"
    if row.get("withheld_reason_code") == "unavailable_case":
        return "withheld (unavailable case)"
    return "withheld (incomplete counts)"


def long_row(key, run, scenario, summaries):
    row = identity(key, run, scenario, summaries)
    row.update(median=None, ci95=None, repeat_samples=[])
    cases = [case for case in run["cases"] if case["scenario"] == scenario]
    if row["original_gate"] != "PASS" or not complete_repeats(run["header"], cases):
        if not complete_repeats(run["header"], cases):
            row["withheld_reason"] = "expected repeats are missing, duplicated or invalid"
        return row
    groups, samples = [], []
    for case in sorted(cases, key=lambda case: case["repeat"]):
        reqs = case.get("requests", [])
        if not reqs or any(not report.finite(req.get("ttft_s")) or req["ttft_s"] <= 0 for req in reqs):
            row["withheld_reason"] = "a passing repeat lacks finite positive TTFT"
            return row
        groups.append([req["ttft_s"] for req in reqs])
        samples.extend({**source_ref(case, req), "value": req["ttft_s"],
                        "prompt_tokens": req.get("prompt_tokens")} for req in reqs)
    low, high = metrics.repeat_median_ci(groups)
    row.update(qualified=True, withheld_reason=None, median=statistics.median(s["value"] for s in samples),
               ci95=[low, high] if low is not None else None, repeat_samples=samples)
    return row


def build_data(paths, include=(), exclude=(), min_decode_tokens=128, initial_warnings=()):
    if not integer(min_decode_tokens, 128):
        raise ValueError("min_decode_tokens must be an integer >=128")
    runs, warnings, sources = report.load_runs(paths)
    kept = {key: run for key, run in sorted(runs.items()) if selected(key[2], include, exclude)}
    data = {"schema": VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "sources": sources, "diagnostics": list(initial_warnings) + warnings,
            "selection": {"include_label_globs": list(include), "exclude_label_globs": list(exclude),
                          "order": "engine, mode, label alphabetically; never by measured performance",
                          "invocations": "existing report policy: last run per engine/mode/label; never merged",
                          "selected_runs": len(kept), "excluded_runs": len(runs) - len(kept)},
            "method": {"min_decode_tokens": min_decode_tokens, "bootstrap": "repeat clusters, 2000 resamples, seed1234, percentile95%",
                       "quality_counts": "original strict n_correct/n_questions, validated against question checks and complete identities; any request.error withholds the quality point; no CI",
                       "long_context": "original all-repeat quality gate; no failed/missing speed substitutes"},
            "tools": [{"path": str(p.resolve()), "sha256": report.digest_file(p)} for p in
                      (Path(__file__), REPORT_PATH, Path(__file__).with_name("qwen38-engine-report.py"))],
            "runs": [], "decode": {mode: [] for mode in MODES}, "quality": [],
            "long_context": {scenario: [] for scenario in LONG}, "limitations": LIMITATIONS}
    for key, run in kept.items():
        summaries = {row["scenario"]: row for row in report.summarise_run(key, run, min_decode_tokens)}
        data["runs"].append({"engine": key[0], "mode": key[1], "label": key[2], "source": run["source"],
                             "header_line": run["header_line"], "header": run["header"]})
        if key[1] in MODES:
            data["decode"][key[1]].extend(decode_rows(key, run, summaries, min_decode_tokens))
        data["quality"].extend(quality_row(key, run, scenario, summaries)
                               for scenario in ("quality_tasks", "quality_gsm8k"))
        for scenario in LONG:
            data["long_context"][scenario].append(long_row(key, run, scenario, summaries))
    # A malformed later run header could otherwise leave a complete earlier run
    # selected by the shared parser. Preserve its evidence, but do not plot it.
    malformed = {source["path"] for source in sources if any(
        warning.startswith("无效 JSON：") and source["path"] + ":" in warning for warning in warnings)}
    for rows in list(data["decode"].values()) + list(data["long_context"].values()) + [data["quality"]]:
        for row in rows:
            if row["source"] in malformed:
                row.update(qualified=False, withheld_reason="source contains malformed JSON; run boundaries are unverified")
                if "median" in row:
                    row.update(median=None, ci95=None, repeat_samples=[])
                else:
                    row.update(correct=None, total=None, counts_complete=False)
    return report.safe_json(data)


def wrapped_label(row, mode=True):
    title = row["engine"] + (f" / {row['mode']}" if mode else "")
    label = textwrap.fill(row["label"] or "(unlabelled)", width=29, break_long_words=True, break_on_hyphens=False)
    return title + "\n" + label


def render(data, output_dir):
    # Deliberately lazy: extraction/tests do not require a plotting dependency.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data["matplotlib_version"] = matplotlib.__version__
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "svg.fonttype": "none", "axes.spines.top": False,
                                "axes.spines.right": False})
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, name, note):
        fig.text(.02, .025, textwrap.fill(note, 165), ha="left", va="bottom", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, .14, 1, .93))
        for extension in ("png", "svg"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
            outputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                            "sha256": report.digest_file(path)})
        plt.close(fig)

    def speeds(ax, rows, title, units, show_modes=True):
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_yticks(range(len(rows)), [wrapped_label(row, show_modes) for row in rows], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        shown = []
        for y, row in enumerate(rows):
            if not row["qualified"]:
                status = row["original_gate"]
                if status == "PASS":
                    status = "metric unavailable"
                ax.text(.025, y, f"withheld ({status})", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=9, color="#777777")
                continue
            v, ci = row["median"], row["ci95"]
            xerr = [[max(0, v - ci[0])], [max(0, ci[1] - v)]] if ci else None
            ax.errorbar(v, y, xerr=xerr, fmt="o", color="#176b87", ecolor="#176b87", capsize=3, markersize=6)
            repeats = len({sample["repeat"] for sample in row["repeat_samples"]})
            ax.annotate(f"{v:.1f} (n={repeats})", (ci[1] if ci else v, y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8)
            shown.append(max(v, ci[1]) if ci else v)
        ax.set_ylim(len(rows) - .45, -.55)
        if shown:
            ax.set_xlim(0, max(shown) * 1.42)
            ax.set_xlabel(units)
        else:
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_xlabel("No qualified measurements")
        if not rows:
            ax.text(.5, .5, "No selected runs", transform=ax.transAxes, ha="center")

    for mode in MODES:
        rows = data["decode"][mode]
        count = len(rows) // 3
        fig, axes = plt.subplots(1, 3, figsize=(17, max(4.9, 2.5 + count * .68)))
        fig.suptitle(f"Client decode rate · {mode}", fontsize=16, fontweight="bold")
        for kind, ax in zip(KINDS, axes):
            speeds(ax, [row for row in rows if row["kind"] == kind], kind.title(), "visible tokens / second", False)
        save(fig, f"figure1-decode-{mode}",
             "Median and repeat-cluster 95% bootstrap interval. Original all-repeat quality PASS and reliable >=128-token decode required. "
             "Alphabetical configuration order; no winner inference. Few repeats give weak uncertainty evidence; n counts repeats, not tokens. Code checks do not establish full program correctness.")

    n = len(data["runs"])
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5.5, 2.7 + n * .68)))
    fig.suptitle("Original strict functional quality · no confidence intervals", fontsize=15, fontweight="bold")
    for scenario, ax in zip(("quality_tasks", "quality_gsm8k"), axes):
        rows = [row for row in data["quality"] if row["scenario"] == scenario]
        ax.set_title("Generated tasks" if scenario == "quality_tasks" else "GSM8K fixed public subset", fontweight="bold", pad=12)
        ax.set_yticks(range(len(rows)), [wrapped_label(row) for row in rows], fontsize=9)
        ax.set_ylim(len(rows) - .45, -.55)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Original strict correct / total presentations")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        for y, row in enumerate(rows):
            if not row["qualified"]:
                ax.text(.02, y, quality_withheld_label(row), va="center", color="#777777", fontsize=9)
                continue
            rate = row["correct"] / row["total"]
            ax.barh(y, rate, height=.38, color="#3a8064", alpha=.80)
            annotation = f"{row['correct']}/{row['total']}"
            if scenario == "quality_gsm8k":
                annotation += f" · {row['unique_questions']} unique × {row['repeats_observed']} repeats"
            ax.text(.02, y, annotation, va="center", fontsize=9, color="#17251d",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .84, "pad": 2})
    save(fig, "figure2-strict-quality",
         "Actual original question scores, including FAIL scenarios when all question counts are complete. "
         "GSM8K labels show actual unique IDs × repeats; repeated presentations are not independent questions. No confidence intervals or comprehensive quality claim.")

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5.5, 2.7 + n * .68)))
    fig.suptitle("Long-context client TTFT · original quality gate required", fontsize=15, fontweight="bold")
    for scenario, ax in zip(LONG, axes):
        speeds(ax, data["long_context"][scenario], "128K (131,072 tokens)" if scenario.endswith("128k") else "250K (256,000 tokens)", "seconds")
    save(fig, "figure3-long-context-ttft",
         "Only configurations passing every original retrieval/order/context-budget check in every expected repeat receive a TTFT point. "
         "FAIL, INCOMPLETE, NOT_RUN and unsupported measurements are withheld, never drawn as zero. TTFT is client-observed latency, not kernel time.")
    return outputs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--include-label", action="append", default=[], help="case-sensitive label glob; repeatable")
    ap.add_argument("--exclude-label", action="append", default=[], help="case-sensitive label glob; repeatable, takes precedence")
    ap.add_argument("--min-decode-tokens", type=int, default=128)
    ap.add_argument("--data-only", action="store_true", help="extract data.json without importing matplotlib")
    args = ap.parse_args(argv)
    if args.min_decode_tokens < 128:
        ap.error("--min-decode-tokens cannot be lower than128")
    paths, warnings = report.collect_inputs(args.inputs)
    data = build_data(paths, args.include_label, args.exclude_label, args.min_decode_tokens, warnings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data["figures"] = [] if args.data_only else render(data, args.output_dir)
    data["rendered"] = not args.data_only
    (args.output_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(data['runs'])} selected runs; {len(data['figures'])} figure files; {args.output_dir / 'data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
