#!/usr/bin/env python3
"""Rebuild two owned fixture objects and qualify portable expert-file tests.

Requires an unchanged, successful pinned Linux native build. Only the current
Native CMake file and expert-cache fixture may differ from that build. This is
test-bridge qualification, not a new production library or model qualification.
No CMake configure, upstream build, source patch, or old object reuse occurs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

TARGET = "GgmlOpsHostMoeExpertCacheTest"
REVISION = "3612c0af93f2cfc8f048295300a61187723b12ae"
ALLOWED_DELTAS = {"TensorSharp.GGML.Native/CMakeLists.txt",
                  "TensorSharp.GGML.Native/tests/host_moe_expert_cache_test.cpp"}
COMPILED = {"tests/host_moe_expert_cache_test.cpp", "ggml_ops_host_moe_cache.cpp"}
MIN_MEM, MIN_C, MIN_OUTPUT, MAX_PROCESS = 1 << 30, 500 << 20, 10 << 30, 8 << 30


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_names(names):
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise RuntimeError("Unsafe manifest path: " + name)
    return names


def source_deltas(expected, current, require_port):
    if set(expected) != set(current):
        raise RuntimeError("The complete Native input closure changed.")
    delta = {name: {"qualified_sha256": expected[name], "current_sha256": current[name]}
             for name in expected if expected[name] != current[name]}
    if set(delta) - ALLOWED_DELTAS:
        raise RuntimeError("A production source/header or unrelated fixture changed: "
                           + repr(sorted(set(delta) - ALLOWED_DELTAS)))
    if require_port and set(delta) != ALLOWED_DELTAS:
        raise RuntimeError("Both explicit test-only portability changes are required.")
    return delta


def parse_commands(raw):
    """Strip only Ninja's known no-op link wrapper; never run a shell."""
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        arguments = shlex.split(line)
        if arguments[:2] == [":", "&&"] and arguments[-2:] == ["&&", ":"]:
            arguments = arguments[2:-2]
        if any(item in {"&&", ";", "|", "||", ">", "<", ":"} for item in arguments):
            raise RuntimeError("Unexpected shell operation in Ninja command.")
        if not arguments or arguments[0] != "/usr/bin/g++-12":
            raise RuntimeError("Unexpected actual compiler/linker.")
        rows.append(arguments)
    if len(rows) != 3 or any("-c" not in row for row in rows[:2]) or "-c" in rows[2]:
        raise RuntimeError("Exactly two owned compilations and one link are required.")
    return rows


def without_dependency_flags(arguments):
    answer, pos = [], 0
    while pos < len(arguments):
        value = arguments[pos]
        if value in ("-MT", "-MF"):
            pos += 2
        elif value == "-MD":
            pos += 1
        else:
            answer.append(value)
            pos += 1
    return answer


def mapped_arguments(arguments, old_driver, new_source):
    """Keep flags, upstream includes, static libraries and object names exact."""
    answer = []
    prefix = str(old_driver)
    for value in arguments:
        if value == prefix or value.startswith(prefix + "/"):
            value = str(new_source) + value[len(prefix):]
        elif value == "-I" + prefix or value.startswith("-I" + prefix + "/"):
            value = "-I" + str(new_source) + value[len("-I" + prefix):]
        answer.append(value)
    return answer


def validate_output(code, output):
    if code != 0:
        raise RuntimeError("Actual fixture exit was not zero (skip is not a pass): " + str(code))
    if re.search(r"^\s*(?:FAIL|SKIP)\b", output, re.M | re.I):
        raise RuntimeError("Fixture reported failure or skipped coverage.")
    for marker in ("PASS: exact file extents, cached reuse, out-of-order staging, partial-read recovery and source release",
                   "PASS: scoped N3 prefill microgroups for 2/3/4/5/8/17/32 tokens, K10, scalar CUDA bit-exact, 3 bridges"):
        if marker not in output:
            raise RuntimeError("Missing actual portable file/microgroup coverage: " + marker)


def environment(root):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("TS_", "TENSORSHARP_", "GGML_"))}
    env.update(TMPDIR=str(root / "tmp"), TMP=str(root / "tmp"), TEMP=str(root / "tmp"),
               XDG_CACHE_HOME=str(root / "cache"), CCACHE_DIR=str(root / "cache/ccache"),
               CCACHE_DISABLE="1", CUDA_HOME="/usr/local/cuda-12.6", CUDA_PATH="/usr/local/cuda-12.6",
               PATH="/usr/local/cuda-12.6/bin:" + env["PATH"],
               LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64", OMP_NUM_THREADS="8")
    return env


def verify_inputs(root, repo, manifest, receipt, captured_records=None):
    expected = checked_names(manifest["source"])
    if (receipt.get("passed") is not True or receipt.get("production_no_test_hooks") is not True
            or receipt.get("before") != receipt.get("after")
            or receipt.get("libraries_before") != receipt.get("libraries_after")
            or receipt.get("source") != expected or manifest.get("snapshot") != expected
            or len(expected) != 242 or len(manifest["upstream_files"]) != 2321
            or manifest.get("upstream_revision") != REVISION):
        raise RuntimeError("Complete unchanged-source Native16 build evidence is required.")
    snapshot = {name: sha(root / "source" / name) for name in expected}
    if snapshot != expected:
        raise RuntimeError("Immutable Native16 snapshot changed.")
    for name in expected:
        if name != "TensorSharp.GGML.Native/CMakeLists.txt":
            if sha(root / "driver" / Path(name).relative_to("TensorSharp.GGML.Native")) != expected[name]:
                raise RuntimeError("Native16 compiled source changed: " + name)
    for relative, field in (("driver/CMakeLists.txt", "driver_cmake_sha256"),
                            ("import-upstream.cmake", "import_driver_sha256"),
                            ("build.py", "build_tool_sha256")):
        if sha(root / relative) != manifest[field]:
            raise RuntimeError("Immutable generated Native16 driver changed: " + relative)
    listed = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--cached", "--others",
                                      "--exclude-standard", "--", "TensorSharp.GGML.Native"], text=True).splitlines()
    if set(listed) != set(expected):
        raise RuntimeError("Current complete Native source inventory differs.")
    current = {name: sha(repo / name) for name in expected}
    source_deltas(expected, current, False)
    reference = Path(manifest["reference"])
    upstream = reference / "ExternalProjects/ggml"
    for name, expected_hash in checked_names(manifest["reference_files"]).items():
        if sha(reference / name) != expected_hash:
            raise RuntimeError("Qualified frozen dependency record changed: " + name)
    for name, expected_hash in checked_names(manifest["upstream_files"]).items():
        if sha(upstream / name) != expected_hash:
            raise RuntimeError("Actual unchanged upstream source changed: " + name)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or status:
        raise RuntimeError("Frozen ggml revision/clean status changed.")
    library_root = reference / "build-v13-cuda126-attempt2"
    libraries = {name: sha(library_root / name) for name in manifest["libraries"]}
    if len(libraries) != 4 or libraries != manifest["libraries"] or libraries != receipt["libraries_before"]:
        raise RuntimeError("Frozen static library identity changed.")
    artifacts = {}
    for field, branch in (("hooks_artifacts", "components"), ("production_artifacts", "production")):
        for name, expected_hash in checked_names(receipt[field]).items():
            actual = sha(root / branch / name)
            if actual != expected_hash:
                raise RuntimeError("Immutable Native16 artifact changed: " + branch + "/" + name)
            artifacts[branch + "/" + name] = actual
    for branch, field in (("hooks", "hooks_artifacts"), ("production", "production_artifacts")):
        if sha(root / ("build-" + branch) / "compile_commands.json") != receipt[field]["compile_commands.json"]:
            raise RuntimeError("Actual Native16 compiler database changed.")
    records = {name: sha(root / name) for name in
               ("prepare-manifest.json", "build-summary.json", "build-hooks/build.ninja",
                "build-hooks/CMakeCache.txt", "build-hooks/CMakeFiles/rules.ninja",
                "build-hooks/compile_commands.json", "build-production/build.ninja",
                "build-production/CMakeCache.txt", "build-production/CMakeFiles/rules.ninja",
                "build-production/compile_commands.json")}
    if captured_records is not None and records != captured_records:
        raise RuntimeError("Original Native16 build records changed during follow-up.")
    return {"current_sources": current, "snapshot_sources": snapshot, "libraries": libraries,
            "artifacts": artifacts, "native_records": records,
            "upstream": {"revision": revision, "status": status, "tracked_files": 2321}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Verify and show actual argv; write nothing and run no compilation/GPU.")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Execution and read-only provenance plans require Linux/WSL (actual upstream symlink reads).")
    root, repo, output = (path.resolve() for path in (args.native_root, args.repo, args.output))
    if (output.exists() or not any(output.is_relative_to(Path(drive)) for drive in ("/mnt/d", "/mnt/e"))
            or output.is_relative_to(root) or output.is_relative_to(repo)):
        parser.error("A fresh D/E output outside repository and immutable Native16 root is required.")
    manifest, receipt = read_json(root / "prepare-manifest.json"), read_json(root / "build-summary.json")
    if output.is_relative_to(Path(manifest["reference"]).resolve()):
        parser.error("Output cannot be inside the frozen dependency reference.")
    report = {"passed": False, "scope": "test-only portable expert-file fixture qualification",
              "production_library_rebuilt": False, "production_model_qualification": False,
              "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "runner_sha256": sha(__file__), "steps": [], "resource_samples": [],
              "limits": {"compile_link_address_space_bytes": MAX_PROCESS,
                         "gpu_test_rss_bytes": MAX_PROCESS, "gpu_virtual_address_space_capped": False,
                         "min_mem_available_bytes": MIN_MEM, "min_c_free_bytes": MIN_C,
                         "min_output_free_bytes": MIN_OUTPUT}}
    created = False
    base_env = environment(output)

    def save():
        if created:
            (output / "progress.json").write_text(json.dumps(report, indent=2))

    def resources(pid=None):
        mem = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
               for line in Path("/proc/meminfo").read_text().splitlines()}
        sample = {"mem_available_bytes": mem["MemAvailable"], "c_free_bytes": shutil.disk_usage("/mnt/c").free,
                  "output_free_bytes": shutil.disk_usage(output).free}
        if pid:
            rss = 0
            # The fixture creates threads, not child processes. Include any future
            # process-group children so a wrapper cannot escape the physical guard.
            for status_path in Path("/proc").glob("[0-9]*/status"):
                try:
                    if os.getpgid(int(status_path.parent.name)) == pid:
                        match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status_path.read_text(), re.M)
                        rss += int(match.group(1)) * 1024 if match else 0
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            sample["process_group_rss_bytes"] = rss
        return sample

    def run(label, arguments, gpu=False, timeout=300):
        import resource
        sample = resources()
        if sample["mem_available_bytes"] < MIN_MEM or sample["c_free_bytes"] < MIN_C or sample["output_free_bytes"] < MIN_OUTPUT:
            raise RuntimeError("Physical resource guard reached before " + label)
        step = {"name": label, "arguments": arguments, "cwd": str(output / "build"), "gpu_test": gpu}
        report["steps"].append(step)
        log_path = output / (label + ".log")
        env = dict(base_env, CUDA_VISIBLE_DEVICES="0" if gpu else "-1")
        step["environment"] = {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "TMPDIR", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")}
        process = None
        started = time.monotonic()
        failure = None
        try:
            def limit():
                resource.setrlimit(resource.RLIMIT_AS, (MAX_PROCESS, MAX_PROCESS))
            with log_path.open("wb") as log:
                process = subprocess.Popen(arguments, cwd=output / "build", env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           preexec_fn=None if gpu else limit)
                step["owned_process_group"] = process.pid
                save()
                while process.poll() is None:
                    sample = resources(process.pid)
                    sample.update(step=label, elapsed_seconds=time.monotonic() - started)
                    report["resource_samples"].append(sample)
                    if (sample["mem_available_bytes"] < MIN_MEM or sample["c_free_bytes"] < MIN_C
                            or sample["output_free_bytes"] < MIN_OUTPUT
                            or sample.get("process_group_rss_bytes", 0) > MAX_PROCESS):
                        failure = "Physical resource guard reached."
                        break
                    if time.monotonic() - started > timeout:
                        failure = "Bounded command timeout."
                        break
                    save()
                    time.sleep(1)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            step.update(exit_code=process.returncode if process else None, seconds=time.monotonic() - started,
                        output_sha256=sha(log_path), output_path=str(log_path))
            if failure:
                step["guard_failure"] = failure
            save()
        text = log_path.read_text(errors="replace")
        if failure or step["exit_code"] != 0:
            raise RuntimeError(label + " failed: " + str(failure or step["exit_code"]))
        if gpu:
            validate_output(step["exit_code"], text)
            step["actual_file_and_microgroup_coverage"] = True
        return text

    try:
        before = verify_inputs(root, repo, manifest, receipt)
        report["before"] = before
        report["source_deltas"] = source_deltas(manifest["source"], before["current_sources"], not args.plan)
        versions = {name: subprocess.check_output([executable, "--version"], env=base_env, text=True)
                    for name, executable in (("gcc", "/usr/bin/gcc-12"), ("g++", "/usr/bin/g++-12"),
                                             ("nvcc", "/usr/local/cuda-12.6/bin/nvcc"), ("cmake", "cmake"))}
        if versions != receipt["compiler_versions"]:
            raise RuntimeError("Actual compiler/toolkit versions differ from Native16.")
        report["compiler_versions"] = versions
        report["compiler_sha256"] = sha("/usr/bin/g++-12")
        raw = subprocess.check_output(["ninja", "-C", str(root / "build-hooks"), "-t", "commands", TARGET],
                                      env=base_env, text=True)
        commands = parse_commands(raw)
        database = read_json(root / "build-hooks/compile_commands.json")
        rows = [row for row in database if row.get("output", "").startswith("CMakeFiles/" + TARGET + ".dir/")]
        if len(rows) != 2 or {str(Path(row["file"]).relative_to(root / "driver")) for row in rows} != COMPILED:
            raise RuntimeError("Qualified compiler database does not contain the exact two owned objects.")
        for command, row in zip(commands[:2], rows):
            if without_dependency_flags(command) != shlex.split(row["command"]):
                raise RuntimeError("Actual Ninja compiler flags differ from the qualified database.")
        objects = {command[command.index("-o") + 1] for command in commands[:2]}
        if {value for value in commands[2] if value.endswith(".o")} != objects:
            raise RuntimeError("Link must consume exactly both freshly compiled owned objects.")
        if commands[2][commands[2].index("-o") + 1] != TARGET:
            raise RuntimeError("Link output differs from the qualified fixture target.")
        dependencies = {value: sha(value) for value in commands[2] if value.startswith("/") and Path(value).is_file()}
        report.update(actual_ninja_commands_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                      actual_original_commands=commands, linked_file_sha256=dependencies,
                      fresh_commands=[mapped_arguments(row, root / "driver", output / "source/TensorSharp.GGML.Native")
                                      for row in commands])
        if args.plan:
            print(json.dumps({"plan_only": True, "passed": False, "scope": report["scope"],
                              "source_deltas": report["source_deltas"], "commands": report["fresh_commands"]}, indent=2))
            return 0
        output.mkdir(parents=True, exist_ok=False)
        created = True
        for directory in ("tmp", "cache", "build"):
            (output / directory).mkdir()
        (output / "original-ninja-commands.txt").write_text(raw)
        shutil.copy2(__file__, output / "file_fixture_followup.py")
        for name in before["current_sources"]:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / name, destination)
        snapshot = {name: sha(output / "source" / name) for name in before["current_sources"]}
        if snapshot != before["current_sources"]:
            raise RuntimeError("Fresh complete source snapshot changed during copy.")
        report["fresh_snapshot_sources"] = snapshot
        save()
        for index, arguments in enumerate(report["fresh_commands"]):
            for flag in ("-o", "-MF"):
                if flag in arguments:
                    relative = Path(arguments[arguments.index(flag) + 1])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Owned outputs must stay inside the new build directory.")
                    destination = output / "build" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise RuntimeError("Old compiled objects/output cannot be reused.")
            run("compile-" + str(index + 1) if index < 2 else "link", arguments)
        executable = output / "build" / TARGET
        with executable.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise RuntimeError("Fresh linked fixture is not an actual Linux ELF executable.")
        report["fresh_artifacts"] = {name: sha(output / "build" / name) for name in sorted(objects | {TARGET})}
        loader = run("runtime-dependencies", ["ldd", str(executable)], timeout=30)
        if "not found" in loader:
            raise RuntimeError("Fresh fixture has an unresolved runtime dependency.")
        runtime_paths = set(re.findall(r"(?:=>\s+|^\s*)(/[^\s]+)\s+\(", loader, re.M))
        if not runtime_paths:
            raise RuntimeError("Actual loader dependency evidence is absent.")
        report["loaded_runtime_file_sha256"] = {name: sha(name) for name in sorted(runtime_paths)}
        for name, arguments in (("arena-file-q8", ["--arena", "--file-read"]),
                                ("arena-file-iq1", ["--arena", "--file-read", "--quantization", "iq1_m"])):
            run(name, [str(executable), *arguments], gpu=True, timeout=180)
        report["tests_passed"] = 2
        report["after"] = verify_inputs(root, repo, manifest, receipt, before["native_records"])
        if report["after"] != before:
            raise RuntimeError("Complete original/current source or dependency bookends disagree.")
        if {name: sha(output / "source" / name) for name in snapshot} != snapshot:
            raise RuntimeError("Fresh compiled snapshot changed.")
        if {name: sha(output / "build" / name) for name in report["fresh_artifacts"]} != report["fresh_artifacts"]:
            raise RuntimeError("Fresh fixture objects/executable changed during tests.")
        if {name: sha(name) for name in dependencies} != dependencies:
            raise RuntimeError("A linked compiler/runtime dependency changed.")
        if {name: sha(name) for name in report["loaded_runtime_file_sha256"]} != report["loaded_runtime_file_sha256"]:
            raise RuntimeError("An actual loader runtime dependency changed.")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        if created and "before" in report:
            try:
                report["after_failure"] = verify_inputs(root, repo, manifest, receipt, report["before"]["native_records"])
            except BaseException as bookend_error:
                report["bookend_error"] = repr(bookend_error)
    finally:
        report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if created:
            (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "error": report.get("error"),
                      "summary": str(output / "summary.json") if created else None}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
