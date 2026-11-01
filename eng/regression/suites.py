"""Reuse correctness suites; optional managed builds never reconfigure native assets."""
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from .processes import (run_process, containment_failure_evidence, unsafe_process_state,
                        UNSAFE_RESOURCES_REASON, ProcessContainmentError)

MANAGED_BUILD_PROPERTIES = ["-p:TensorSharpSkipGgmlNative=true", "-p:TensorSharpSkipMlxNative=true"]


def list_suites(repo):
    repo = Path(repo)
    suites = []
    for identifier, project in (("server_unit", "InferenceWeb.Tests"),
                                ("knowledge_unit", "TensorSharp.Knowledge.Tests"),
                                ("studio_unit", "TensorSharp.Studio.Tests")):
        suites.append({"id": identifier, "kind": "dotnet", "project": project + "/" + project + ".csproj",
                       "built": repo / project / "bin/Release/net10.0" / (project + ".dll"),
                       "filter": "Category!=Bench&Requires!=Models&Requires!=Cuda&Requires!=Mlx",
                       "assertions": ["TRX contains executed passed tests", "zero failed tests", "skipped tests disclosed separately"],
                       "scope": "Environment-independent correctness only; not native GPU/model/performance coverage"})
    suites.append({"id": "client_native_contract", "kind": "client_native",
                   "built": repo / "eng/tests/NativeChatCompletion/bin/Release/net10.0/NativeChatCompletion.dll",
                   "example": repo / "examples/tensorsharp-native-chat/bin/Release/net10.0/TensorSharp.NativeChat.dll",
                   "assertions": ["all seven completion-boundary checks", "adapter contracts execute"]})
    suites.append({"id": "client_http_contract", "kind": "client_http",
                   "built": repo / "examples/dotnet-local-chat/bin/Release/net10.0/TensorSharp.LocalChat.dll",
                   "assertions": ["mock HTTP integration assertions execute", "zero FAIL records"]})
    return suites


def trx_counts(path):
    root = ET.parse(path).getroot()
    counters = root.find(".//{*}Counters")
    if counters is None:
        raise ValueError("TRX contains no test counters")
    counts = {key: int(counters.attrib.get(key, "0")) for key in ("total", "executed", "passed", "failed", "error", "timeout", "notExecuted", "aborted", "inconclusive", "notRunnable")}
    if any(v < 0 for v in counts.values()) or counts["executed"] <= 0 or counts["passed"] <= 0 or not counts["total"] >= counts["executed"] >= counts["passed"]:
        raise ValueError("suite did not execute any passing tests or counters are inconsistent")
    if counts["failed"] + counts["error"] + counts["timeout"] + counts["aborted"]:
        raise ValueError("TRX reports failure/error/timeout/aborted tests")
    if counts["executed"] != counts["passed"] + counts["inconclusive"]:
        raise ValueError("TRX executed count includes unexplained nonpassing tests")
    counts["unqualified"] = counts["total"] - counts["passed"]
    return counts


def run_suites(repo, output, mode, allow_builds, timeout_seconds):
    repo, output = Path(repo), Path(output)
    results, dotnet = [], shutil.which("dotnet")
    halt = unsafe_process_state() if mode == "run" else None
    for spec in list_suites(repo):
        result = {"id": spec["id"], "scope": spec.get("scope", "client contracts without model execution"),
                  "assertions": spec["assertions"], "status": "untested", "reason": "planned; suite not executed",
                  "build_policy": {"managed_builds_allowed": bool(allow_builds), "native_builds_allowed": False}}
        if halt is not None:
            result.update(reason=UNSAFE_RESOURCES_REASON, stage="execution_halted")
            results.append(result)
            continue
        if mode != "run":
            results.append(result)
            continue
        if not dotnet:
            result.update(status="skip", reason=".NET SDK/runtime unavailable")
            results.append(result)
            continue
        target = output / spec["id"]
        target.mkdir(parents=True, exist_ok=True)
        commands = []
        if spec["kind"] == "dotnet":
            if not allow_builds and not spec["built"].is_file():
                result.update(status="skip", reason="test assembly not built; build explicitly or use --allow-builds")
                results.append(result)
                continue
            command = [dotnet, "test", str(repo / spec["project"]), "-c", "Release", "--filter", spec["filter"],
                       "--logger", "trx;LogFileName=tests.trx", "--results-directory", str(target)]
            if not allow_builds:
                command += ["--no-build", "--no-restore"]
            commands.append(command)
        else:
            if spec["kind"] == "client_native":
                if allow_builds:
                    commands.append([dotnet, "build", "eng/tests/NativeChatCompletion", "-c", "Release"])
                elif not spec["built"].is_file() or not spec["example"].is_file():
                    result.update(status="skip", reason="native client contract assemblies not built; see documented recipe")
                    results.append(result)
                    continue
                commands.append([dotnet, str(spec["built"]), str(spec["example"])])
            else:
                if allow_builds:
                    commands += [[dotnet, "restore", "examples/dotnet-local-chat", "--locked-mode", "--configfile", "examples/dotnet-local-chat/NuGet.Config"],
                                 [dotnet, "build", "examples/dotnet-local-chat", "-c", "Release", "--no-restore"]]
                elif not spec["built"].is_file():
                    result.update(status="skip", reason="HTTP example assembly not built; see documented recipe")
                    results.append(result)
                    continue
                commands += [[sys.executable, str(repo / "eng/tests/dotnet-local-chat.py")],
                             [sys.executable, str(repo / "eng/tests/dotnet-local-chat-e2e.py"), "--self-test"]]
        # Even dotnet test --no-build evaluates MSBuild targets. Always prevent
        # transitive native build/restore targets from reconfiguring a shared
        # CUDA/Metal build directory or replacing an existing runtime library.
        commands = [command + MANAGED_BUILD_PROPERTIES
                    if len(command) > 1 and command[0] == dotnet and command[1] in {"test", "build", "restore"}
                    else command for command in commands]
        result["processes"] = []
        try:
            for index, command in enumerate(commands):
                process = run_process(command, cwd=repo, artifact_dir=target, stage=f"step-{index}",
                                      timeout_seconds=timeout_seconds,
                                      env={"TS_TEST_GGML_BACKEND": "cpu"})
                result["processes"].append(process)
                containment = containment_failure_evidence(process)
                if containment is not None:
                    halt = containment
                    result.update(containment_failure=True, containment_evidence=containment)
                    result["remaining_commands"] = [{"command": pending, "status": "untested", "reason": UNSAFE_RESOURCES_REASON}
                                                    for pending in commands[index + 1:]]
                    raise ValueError(UNSAFE_RESOURCES_REASON)
                if process["exit_code"] != 0 or process.get("timed_out"):
                    raise ValueError(f"suite process failed or timed out at step {index}; see sanitized logs")
                if len(command) > 1 and command[0] == dotnet and command[1] == "build":
                    expected = [spec["built"]] + ([spec["example"]] if spec["kind"] == "client_native" else [])
                    missing = [str(path) for path in expected if not path.is_file()]
                    if missing:
                        raise ValueError("managed build did not produce required contract assemblies: " + ", ".join(missing))
                is_assertion_command = spec["kind"] != "dotnet" and not (len(command) > 1 and command[0] == dotnet and command[1] in {"build", "restore"})
                if is_assertion_command:
                    content = process.get("stdout", "") + process.get("stderr", "")
                    if "FAIL " in content or not re.search(r"\bPASS\b|\bchecks passed\b", content):
                        raise ValueError(f"client test step {index} emitted failure or no assertion evidence")
            if spec["kind"] == "dotnet":
                counts = trx_counts(target / "tests.trx")
                result["counts"] = counts
                if counts["failed"] + counts["error"] + counts["timeout"]:
                    raise ValueError("TRX reports failed/error/timeout tests")
                if counts["unqualified"]:
                    results.append({"id": spec["id"] + "::not-executed", "status": "skip",
                                    "reason": f"{counts['unqualified']} test(s) unexecuted or inconclusive; see TRX"})
            else:
                content = "\n".join(p.get("stdout", "") + p.get("stderr", "") for p in result["processes"])
                if "FAIL " in content or not re.search(r"\bPASS\b|\bchecks passed\b", content):
                    raise ValueError("client suite emitted failure or no assertion evidence")
                if spec["kind"] == "client_native" and "7 completion boundary checks passed" not in content:
                    raise ValueError("completion boundary checks did not complete")
            result.update(status="pass", reason="existing suite assertions passed; detailed counts/scope retained")
        except ProcessContainmentError as error:
            halt = error.evidence
            result.update(status="fail", reason=UNSAFE_RESOURCES_REASON,
                          containment_failure=True, containment_evidence=halt)
        except (OSError, ValueError, ET.ParseError) as error:
            result.update(status="fail", reason=str(error))
        results.append(result)
    return results
