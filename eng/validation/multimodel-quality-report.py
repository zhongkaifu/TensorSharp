#!/usr/bin/env python3
"""Recompute model task checks and timing denominators before making a report.

Manifest: {"runs": [{"label": "...", "report": "absolute/report.json",
"cases": ["squares", ...], "repetitions": 3}], "notes": ["..."]}.
Failed quality remains visible. This validates evidence structure, not logits.
"""
import argparse
import base64
import hashlib
import html
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("multimodel-quality-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def verify(report, cases, repetitions):
    if report.get("run_complete") is not True or report.get("error"):
        raise ValueError("Incomplete model run")
    native = report.get("loaded_native", {}).get("sha256")
    native_name = report.get("native_name", "GgmlOps.dll")
    if native_name not in ("GgmlOps.dll", "libGgmlOps.so") or not isinstance(native, str) or not re.fullmatch("[0-9a-f]{64}", native) or native != report.get("binaries", {}).get(native_name, {}).get("sha256"):
        raise ValueError("Loaded native identity mismatch")
    expected = {(case, repetition) for case in cases for repetition in range(repetitions)}
    rows = report.get("records", [])
    keys = [(row.get("case"), row.get("repetition")) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or not rows:
        raise ValueError("Missing, duplicate or unexpected case rows")
    for row in rows:
        response = row.get("response", {})
        complete = response.get("done") is True and response.get("done_reason") == "stop" and not response.get("error")
        passed = bench.quality(row["case"], response.get("message", {}).get("content"), complete)
        if row.get("http_status") != 200 or row.get("error") or row.get("quality_passed") != passed:
            raise ValueError("Request error or stored quality result does not match raw response")
        request = row.get("request", {})
        messages = request.get("messages", [])
        if len(messages) != 1 or messages[0].get("role") != "user" or messages[0].get("content") != bench.PROMPTS[row["case"]]:
            raise ValueError("Changed task prompt")
        required = dict(temperature=0, top_k=0, top_p=1, min_p=0, repeat_penalty=1,
                        presence_penalty=0, frequency_penalty=0, seed=17, stop=[])
        if any(request.get("options", {}).get(k) != v for k, v in required.items()) or request.get("think") is not False or request.get("stream") is not False or request.get("multi_agent") is not False:
            raise ValueError("Changed sampling or protocol settings")
        if request.get("skills") != [] or request.get("skills_discovery") is not False or request.get("tools"):
            raise ValueError("Unexpected tools or skills")
        if row["case"] == "image_ocr":
            images = messages[0].get("images", [])
            if len(images) != 1 or hashlib.sha256(base64.b64decode(images[0], validate=True)).hexdigest() != report.get("image", {}).get("sha256"):
                raise ValueError("Image identity mismatch")
        if json.loads(row.get("raw_response", "null")) != response:
            raise ValueError("Decoded response differs from raw transport evidence")
        measured = bench.metrics(response)
        for key, value in measured.items():
            stored = row.get("metrics", {}).get(key)
            if not isinstance(stored, (int, float)) or not math.isfinite(stored) or not math.isclose(stored, value, rel_tol=1e-12):
                raise ValueError("Changed timing or denominator")
    return bench.summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not any(out.is_relative_to(ROOT / p) for p in ("artifacts", "docs/validation")):
        parser.error("Generated reports belong in ignored evidence directories")
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results, identities = [], []
    for entry in manifest["runs"]:
        path = Path(entry["report"])
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = verify(data, entry["cases"], entry["repetitions"])
        identities.append({k: v["sha256"] for k, v in data["binaries"].items()})
        results.append(dict(label=entry["label"], report=str(path.resolve()), report_sha256=bench.sha(path),
                            summary=summary, memory=data["memory"], checkpoint=data["checkpoint"],
                            failures=[dict(case=r["case"], repetition=r["repetition"], reason=r["response"].get("done_reason"),
                                           answer=r["response"].get("message", {}).get("content"))
                                      for r in data["records"] if not r["quality_passed"]]))
    if not identities or any(i != identities[0] for i in identities):
        raise ValueError("Main matrix mixes binary deployments; report variants separately")
    result = dict(evidence_validated=True, quality_all_passed=all(not r["failures"] for r in results), runs=results,
                  notes=manifest.get("notes", []), binaries=identities[0], manifest_sha256=bench.sha(args.manifest),
                  reporter_sha256=bench.sha(__file__), checker_sha256=bench.sha(bench.__file__))
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    esc = lambda x: html.escape(str(x))
    def median(run, case, metric):
        value = run["summary"].get(case, {}).get("warm", {}).get(metric, {}).get("median")
        return f"{value:.2f}" if value is not None else "—"
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多模型质量与性能验证</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;margin:40px auto;max-width:1240px;padding:0 24px;background:#f8fafc;color:#172033}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:12px;border:1px solid #dbe1e8;text-align:left}th{background:#e9eff6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:18px;border:1px solid #ddd}h1,h2{line-height:1.25}.bad{color:#a12b20}</style>',
        f'<h1>多模型质量与性能验证</h1><p>{esc(manifest.get("hardware", "硬件、容器限制与放置参数见原始运行记录。"))} 相同冻结二进制；无推测解码、无前缀复用、贪心且关闭惩罚。</p>',
        '<p>下表为每类任务后续请求的中位数；首次请求与完整范围见明细。速度单位 tokens/s。RAM 为整个进程生命周期的采样 RSS；VRAM 为各卡同时采样占用总和，均非硬配额。失败、截断请求保留计时，不能视为成功回答速度。</p>',
        '<table><tr><th>模型</th><th>严格任务通过</th><th>短 prefill</th><th>平方数 decode</th><th>较长 prefill</th><th>RAM GiB</th><th>整卡 VRAM GiB</th></tr>']
    for run in results:
        checks = run["summary"].values()
        score = f'{sum(c["quality_passes"] for c in checks)}/{sum(c["requests"] for c in checks)}'
        memory = run["memory"]
        parts.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in (run["label"], score,
            median(run, "squares", "prefill_tps"), median(run, "squares", "decode_tps"), median(run, "long_extract", "prefill_tps"),
            f'{memory["peak_rss_bytes"]/2**30:.2f}', f'{memory.get("peak_all_boards_mib", memory["peak_board_mib"])/1024:.2f}')) + '</tr>')
    parts.append('</table><h2>结论与限制</h2><ul>')
    parts.extend(f'<li>{esc(note)}</li>' for note in result["notes"])
    parts.append('</ul><h2>逐模型完整观测</h2>')
    for run in results:
        parts.append(f'<details><summary>{esc(run["label"])}</summary><p>{esc(run["checkpoint"]["path"])}</p><pre>{esc(json.dumps(run, ensure_ascii=False, indent=2))}</pre></details>')
    parts.append(f'<h2>冻结二进制身份</h2><pre>{esc(json.dumps(result["binaries"], indent=2))}</pre></html>')
    (out / "report.html").write_text(''.join(parts), encoding="utf-8")
    print(json.dumps(dict(evidence_validated=True, quality_all_passed=result["quality_all_passed"], report=str(out / "report.html"))))


if __name__ == "__main__":
    main()
