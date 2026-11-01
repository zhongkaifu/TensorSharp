"""Execute explicitly supplied, prebuilt native tests and preserve per-test evidence.

Discovery never runs tests. Unknown backend attribution is deliberately untested;
the adapter does not configure/build native code or silently run another device.
"""
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from .processes import (run_process, containment_failure_evidence, unsafe_process_state,
                        UNSAFE_RESOURCES_REASON, ProcessContainmentError)
from .reports import file_hash


def test_backend(test):
    """Prefer explicit command device arguments, then a single device name token."""
    devices = {"cpu", "cuda", "metal", "vulkan"}
    arguments = set(test.get("command", [])[1:]) & devices
    if len(arguments) == 1:
        return arguments.pop()
    if arguments:
        return None
    tokens = set(re.split(r"[-_]", test["name"])) & devices
    return tokens.pop() if len(tokens) == 1 else None


def inventory(payload, backend, pattern):
    if pattern:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError("invalid native test filter: " + str(error)) from error
    if payload.get("kind") != "ctestInfo" or payload.get("version", {}).get("major") != 1:
        raise ValueError("unsupported CTest discovery format")
    tests = payload.get("tests", [])
    names = [test.get("name") for test in tests]
    if not tests or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("CTest discovery is empty or has invalid/duplicate names")
    selected, omitted = [], []
    for test in tests:
        observed = test_backend(test)
        reason = None
        if observed is None:
            reason = "backend cannot be established from native test command/name"
        elif observed != backend:
            reason = f"native test declares {observed}; requested {backend}"
        elif pattern and not re.search(pattern, test["name"]):
            reason = "outside explicit native test filter"
        if reason:
            omitted.append({"id": "native_ctest::" + test["name"], "status": "untested" if observed is None else "skip", "reason": reason})
        else:
            selected.append(test)
    # Never let CTest auto-add an unselected fixture on a different device.
    # Repeatedly remove dependants whose fixture setup is not selected.
    while True:
        setups = set()
        for test in selected:
            for prop in test.get("properties", []):
                if prop.get("name") == "FIXTURES_SETUP":
                    setups.update(_property_values(prop.get("value")))
        retained = []
        for test in selected:
            needed = set()
            for prop in test.get("properties", []):
                if prop.get("name") == "FIXTURES_REQUIRED":
                    needed.update(_property_values(prop.get("value")))
            if needed - setups:
                omitted.append({"id": "native_ctest::" + test["name"], "status": "skip",
                                "reason": "fixture setup outside selected backend/filter: " + ", ".join(sorted(needed - setups))})
            else:
                retained.append(test)
        if len(retained) == len(selected):
            return retained, omitted
        selected = retained


def _property_values(value):
    return value if isinstance(value, list) else str(value or "").split(";")


def parse_results(path, selected):
    expected = {test["name"]: test for test in selected}
    root = ET.parse(path).getroot()
    rows = root.findall(".//testcase")
    if not rows or len(rows) != len(expected) or {row.get("name") for row in rows} != set(expected):
        raise ValueError("CTest JUnit test names/count differ from selected inventory")
    results = []
    for row in rows:
        name = row.get("name")
        failures = row.findall("failure") + row.findall("error")
        skipped = row.find("skipped")
        duration = float(row.get("time", "0"))
        if not 0 <= duration < float("inf"):
            raise ValueError("invalid native test duration")
        status = "fail" if failures else "skip" if skipped is not None else "pass"
        text = "\n".join("".join(item.itertext()) for item in failures)
        if skipped is not None:
            text = skipped.get("message", "") + " " + "".join(skipped.itertext())
        results.append({"id": "native_ctest::" + name, "status": status,
                        "reason": text.strip() or ("native assertions passed" if status == "pass" else "native test skipped"),
                        "duration_seconds": duration, "command": expected[name].get("command", []),
                        "stdout": row.findtext("system-out", ""), "stderr": row.findtext("system-err", "")})
    return results


def run_native_tests(repo, build_dir, output, mode, backend="cpu", pattern=None, timeout_seconds=600, gpu_index=0):
    repo, build_dir, output = Path(repo).resolve(), Path(build_dir).resolve(), Path(output)
    result = {"id": "native_ctest", "status": "untested", "requested_backend": backend,
              "build_dir": str(build_dir), "scope": "prebuilt native assertions; not model inference or other-device coverage",
              "reason": "planned; native tests were not executed", "processes": []}
    if mode != "run":
        return [result]
    if unsafe_process_state() is not None:
        result.update(reason=UNSAFE_RESOURCES_REASON, stage="execution_halted")
        return [result]
    ctest = shutil.which("ctest")
    if not ctest:
        result.update(status="skip", reason="CTest unavailable")
        return [result]
    cache = build_dir / "CMakeCache.txt"
    try:
        entries = {}
        for line in cache.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith(("#", "//")):
                key, value = line.split("=", 1)
                entries[key.split(":", 1)[0]] = value
        source = Path(entries.get("CMAKE_HOME_DIRECTORY", "")).resolve()
        if source != repo / "TensorSharp.GGML.Native":
            raise ValueError("native build source must be this checkout's TensorSharp.GGML.Native")
        if entries.get("TENSORSHARP_GGML_NATIVE_BUILD_TESTS", "").upper() not in {"ON", "TRUE", "1", "YES"}:
            raise ValueError("native build did not enable TensorSharp regression tests")
        if backend != "cpu" and entries.get("TENSORSHARP_GGML_NATIVE_ENABLE_" + backend.upper(), "").upper() not in {"ON", "TRUE", "1", "YES"}:
            raise ValueError("requested native backend is not enabled in CMake cache")
        result["cmake_cache_sha256"] = file_hash(cache)
        result["build_identity_note"] = "Cache/source and binary hashes identify deployment; they do not prove binaries rebuilt from current source."
        output.mkdir(parents=True, exist_ok=True)
        common = [ctest, "--test-dir", str(build_dir), "-C", "Release"]
        discovery = run_process(common + ["--show-only=json-v1"], cwd=repo, artifact_dir=output,
                                stage="discovery", timeout_seconds=min(timeout_seconds, 60))
        result["processes"].append(discovery)
        containment = containment_failure_evidence(discovery)
        if containment is not None:
            result.update(status="fail", reason=UNSAFE_RESOURCES_REASON,
                          containment_failure=True, containment_evidence=containment)
            # Reading an already captured discovery payload is safe; use it to
            # disclose selected native cases without starting another command.
            try:
                pending, omitted = inventory(json.loads(discovery.get("stdout", "")), backend, pattern)
            except (ValueError, KeyError, TypeError):
                return [result]
            return [result] + [{"id": "native_ctest::" + test["name"], "status": "untested",
                              "reason": UNSAFE_RESOURCES_REASON} for test in pending] + omitted
        if discovery["exit_code"] != 0 or discovery.get("timed_out"):
            raise ValueError("CTest discovery failed; see process evidence")
        selected, omitted = inventory(json.loads(discovery["stdout"]), backend, pattern)
        if not selected:
            result.update(status="untested", reason="no native tests with verified requested backend and complete fixtures")
            return [result] + omitted
        result["test_binaries"] = {test["command"][0]: file_hash(test["command"][0])
                                   for test in selected if test.get("command")}
        junit = output / "ctest.xml"
        # A successful process exit is insufficient without a report from this
        # invocation. Invalidate only our known result file before execution.
        junit.unlink(missing_ok=True)
        expression = "^(" + "|".join(re.escape(test["name"]) for test in selected) + ")$"
        command = common + ["--parallel", "1", "--timeout", str(timeout_seconds), "--no-tests=error",
                            "--fixture-exclude-any", ".*", "-R", expression, "--output-on-failure", "--output-junit", str(junit.resolve())]
        execution = run_process(command, cwd=repo, artifact_dir=output, stage="native-tests",
                                timeout_seconds=timeout_seconds * len(selected) + 30,
                                env={"CUDA_VISIBLE_DEVICES": str(gpu_index) if backend == "cuda" else ""})
        result["processes"].append(execution)
        containment = containment_failure_evidence(execution)
        if containment is not None:
            result.update(status="fail", reason=UNSAFE_RESOURCES_REASON,
                          containment_failure=True, containment_evidence=containment)
            return [result] + [{"id": "native_ctest::" + test["name"], "status": "untested",
                              "reason": UNSAFE_RESOURCES_REASON} for test in selected] + omitted
        rows = parse_results(junit, selected)
        if execution.get("timed_out") or (execution["exit_code"] != 0 and not any(row["status"] == "fail" for row in rows)):
            raise ValueError("native test process failed or timed out without matching failed test evidence")
        result.update(status="fail" if any(row["status"] == "fail" for row in rows) else "pass" if any(row["status"] == "pass" for row in rows) else "untested",
                      reason="individual native assertion outcomes retained; omitted tests remain explicit",
                      counts={status: sum(row["status"] == status for row in rows) for status in ("pass", "fail", "skip")})
        return [result] + rows + omitted
    except ProcessContainmentError as error:
        result.update(status="fail", reason=UNSAFE_RESOURCES_REASON,
                      containment_failure=True, containment_evidence=error.evidence)
        return [result]
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        result.update(status="fail", reason=str(error))
        return [result]
