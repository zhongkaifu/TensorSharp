"""Local evidence only. Never uploads reports or includes environment dumps."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

STATUSES = {"pass", "fail", "skip", "unsupported", "untested"}
SECRET_KEY = re.compile(r"(?i)(password|secret|authorization|api[_-]?key|access[_-]?token|hf[_-]?token)")


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): "<redacted>" if SECRET_KEY.search(str(k)) else sanitize(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        result, hide_next = [], False
        for item in value:
            result.append("<redacted>" if hide_next else sanitize(item))
            hide_next = isinstance(item, str) and item.startswith("--") and bool(SECRET_KEY.search(item))
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        value = re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+", r"\1<redacted>", value)
        value = re.sub(r"(?i)([?&](?:token|key|signature|credential|x-amz-[^=]+)=)[^&\s]+", r"\1<redacted>", value)
        value = re.sub(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@", r"\1<redacted>@", value)
        value = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "<redacted>", value)
        value = re.sub(r"(?i)((?:password|api[_-]?key|access[_-]?token|secret)\s*[=:]\s*)[^\s,;]+", r"\1<redacted>", value)
        return value
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(repo):
    result = {"repository": "https://github.com/zhongkaifu/TensorSharp", "root": str(repo)}
    for name, command in [("commit", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]),
                          ("working_tree", ["status", "--porcelain"])]:
        try:
            probe = subprocess.run(["git", "-C", str(repo), *command], capture_output=True,
                                   text=True, timeout=10, check=False)
            result[name] = probe.stdout.strip() if probe.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[name] = "unavailable"
    return result


def runtime_identity(hardware):
    """Hash explicit deployment assets; do not imply they were mapped by a process."""
    result = []
    for name, paths in hardware.get("runtime", {}).get("files", {}).items():
        if not any(part in name for part in ("tensorsharp", "ggml", "mlx", ".ptx", ".fatbin")):
            continue
        for filename in paths:
            path = Path(filename)
            result.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_hash(path),
                           "provenance": "explicit deployment candidate; mapped module identity is not independently observed"})
    return result


def aggregate_exit(results, *, strict=False, cancelled=False):
    if cancelled:
        return 130
    if any(r.get("status") not in STATUSES for r in results):
        return 2
    if any(r["status"] == "fail" for r in results):
        return 1
    if not results or not any(r["status"] == "pass" for r in results):
        return 3
    if strict and any(r["status"] in {"skip", "untested"} for r in results):
        return 3
    return 0


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_reports(root, report, *, bundle=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report = sanitize(report)
    results = report.get("results", [])
    report["counts"] = {state: sum(r.get("status") == state for r in results) for state in sorted(STATUSES)}
    atomic_json(root / "report.json", report)
    lines = ["# TensorSharp local regression", "", f"Mode: **{report.get('mode', 'unknown')}**. Source: `{report.get('source', {}).get('commit', 'unknown')}`.",
             "", " | ".join(f"{k}: {v}" for k, v in report["counts"].items()), "",
             "A planned, skipped, unsupported or untested case is never a passing test.", "",
             "| Case | Backend | Status | Reason |", "|---|---|---|---|"]
    esc = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    for result in results:
        lines.append("| " + " | ".join(esc(result.get(k, "")) for k in ("id", "backend", "status", "reason")) + " |")
    lines += ["", "## Scope and limitations", ""] + [f"- {item}" for item in report.get("limitations", [])]
    lines += ["", "Full machine inventory, budgets, source/model identities, assertions and reproduction arguments are in `report.json`.",
              "Per-case sanitized logs and artifacts remain beside this report. No model weights are bundled.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    suite = ET.Element("testsuite", name="TensorSharp.local-regression", tests=str(len(results)),
                       failures=str(report["counts"]["fail"]),
                       skipped=str(sum(report["counts"][s] for s in ("skip", "unsupported", "untested"))))
    for result in results:
        case = ET.SubElement(suite, "testcase", name=str(result.get("id", "unknown")),
                             classname=str(result.get("backend", "inventory")),
                             time=str(result.get("duration_seconds", 0)))
        state = result.get("status", "untested")
        if state == "fail":
            ET.SubElement(case, "failure", message=str(result.get("reason", "failure"))).text = json.dumps(result.get("assertions", []))
        elif state != "pass":
            ET.SubElement(case, "skipped", type=state, message=str(result.get("reason", state)))
    ET.ElementTree(suite).write(root / "junit.xml", encoding="utf-8", xml_declaration=True)
    if bundle:
        # Deliberate allowlist: never zip arbitrary binaries, media, local weights or process memory.
        with zipfile.ZipFile(root / "diagnostics.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.is_file() or path.suffix not in {".json", ".md", ".txt", ".log", ".xml"}:
                    continue
                if path.stat().st_size > 16 * 1024 * 1024:
                    continue
                archive.writestr(path.relative_to(root).as_posix(), sanitize(path.read_text(encoding="utf-8", errors="replace")))
    return report
