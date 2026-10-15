#!/usr/bin/env python3
"""Build only fresh TensorSharp objects against pinned, unchanged Linux ggml libs."""
import datetime
import hashlib
import json
import os
import pathlib
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

TARGETS = ["GgmlOps", "GgmlOpsHostMoeExpertCacheTest", "GgmlOpsHostMoeCacheIdentityTest",
           "GgmlOpsGlmHostMappingTest", "GgmlOpsHostMoeCacheConfigTest", "GgmlOpsHostMoeCachePolicyTest",
           "GgmlOpsHostMoeFilePipelineConfigTest", "GgmlOpsHostMoeFilePipelinePlanTest",
           "GgmlOpsGlmBatchedExecutionTest", "GgmlOpsGlmGraphBudgetTest", "GgmlOpsGlmExpertBoundaryTest",
           "GgmlOpsHostMoeFileBatchTest", "GgmlOpsHostMoeFileBatchParallelTest"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compiler_identity(row, native, upstream):
    args = shlex.split(row["command"])
    flags, includes = [], []
    pos = 0
    while pos < len(args):
        item = args[pos]
        if item in ("-o", "-c"):
            pos += 2
            continue
        if item in ("-I", "-isystem"):
            value = args[pos + 1]
            pos += 2
            prefix = item
        elif item.startswith("-I"):
            prefix, value = "-I", item[2:]
            pos += 1
        else:
            flags.append(item)
            pos += 1
            continue
        path = pathlib.Path(value).resolve()
        if path.is_relative_to(native):
            value = "<OWNED>/" + str(path.relative_to(native))
        elif path.is_relative_to(upstream):
            value = "<UPSTREAM>/" + str(path.relative_to(upstream))
        else:
            value = str(path)
        includes.append((prefix, value))
    # Include ordering is immaterial for the verified disjoint owned/upstream/toolkit
    # roots, but changing -I into -isystem or dropping a root is rejected.
    return {"flags": flags, "includes": sorted(set(includes))}


def main():
    if sys.platform != "linux":
        raise RuntimeError("The prepared build must run under Linux/WSL.")
    root = pathlib.Path(__file__).resolve().parent
    manifest = json.loads((root / "prepare-manifest.json").read_text())
    reference = pathlib.Path(manifest["reference"])
    upstream = reference / "ExternalProjects/ggml"
    source = root / "source/TensorSharp.GGML.Native"
    driver = root / "driver"
    reference_build = reference / "build-v13-cuda126-attempt2"
    if not root.is_relative_to(pathlib.Path("/mnt/e")) and not root.is_relative_to(pathlib.Path("/mnt/d")):
        raise RuntimeError("Build outputs must be on D/E, outside the C-backed WSL filesystem.")
    report = {"utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False,
              "source": manifest["source"], "libraries_before": manifest["libraries"],
              "steps": [], "resource_samples": [], "flags_comparisons": [],
              "limits": {"parallel_jobs": 2, "per_process_address_space_bytes": 8 << 30,
                         "min_mem_available_bytes": 1 << 30, "min_c_free_bytes": 500 << 20,
                         "min_output_free_bytes": 10 << 30}, "native_tests_run": False}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TS_", "TENSORSHARP_", "GGML_"))}
    for name in ("tmp", "cache"):
        (root / name).mkdir(exist_ok=True)
    env.update(TMPDIR=str(root / "tmp"), TMP=str(root / "tmp"), TEMP=str(root / "tmp"),
               XDG_CACHE_HOME=str(root / "cache"), CCACHE_DIR=str(root / "cache/ccache"), CCACHE_DISABLE="1",
               CUDA_HOME="/usr/local/cuda-12.6", CUDA_PATH="/usr/local/cuda-12.6", CUDA_VISIBLE_DEVICES="-1",
               PATH="/usr/local/cuda-12.6/bin:" + env["PATH"],
               LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64", OMP_NUM_THREADS="8")
    report["environment"] = {k: v for k, v in env.items() if k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "CCACHE_DIR", "CCACHE_DISABLE", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")}

    def save():
        (root / "build-progress.json").write_text(json.dumps(report, indent=2))

    def verify():
        current = {name: sha(pathlib.Path(manifest["repository"]) / name) for name in manifest["source"]}
        snapshot = {name: sha(root / "source" / name) for name in manifest["source"]}
        effective = {name: sha(driver / pathlib.Path(name).relative_to("TensorSharp.GGML.Native"))
                     for name in manifest["source"] if name != "TensorSharp.GGML.Native/CMakeLists.txt"}
        if current != manifest["source"] or snapshot != manifest["source"]:
            raise RuntimeError("Current source or immutable snapshot changed.")
        if any(effective[name] != manifest["source"][name] for name in effective):
            raise RuntimeError("Effective compiled TensorSharp source bytes changed.")
        if sha(driver / "CMakeLists.txt") != manifest["driver_cmake_sha256"] or sha(root / "import-upstream.cmake") != manifest["import_driver_sha256"] or sha(root / "build.py") != manifest["build_tool_sha256"]:
            raise RuntimeError("Prepared validation driver changed.")
        for name, expected in manifest["reference_files"].items():
            if sha(reference / name) != expected:
                raise RuntimeError("Frozen reference build record changed: " + name)
        for name, expected in manifest["libraries"].items():
            if sha(reference_build / name) != expected:
                raise RuntimeError("Frozen dependency library changed: " + name)
        files = {name: sha(upstream / name) for name in manifest["upstream_files"]}
        if files != manifest["upstream_files"]:
            raise RuntimeError("Frozen upstream source bytes changed.")
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], env=env, text=True).strip()
        status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], env=env, text=True).strip()
        if revision != manifest["upstream_revision"] or status:
            raise RuntimeError("Frozen upstream revision/clean status changed.")
        return {"revision": revision, "status": status, "tracked_files": len(files), "current_source_count": len(current)}

    def resources():
        mem = {x.split(":", 1)[0]: int(x.split()[1]) * 1024 for x in pathlib.Path("/proc/meminfo").read_text().splitlines()}
        return {"mem_available_bytes": mem["MemAvailable"], "c_free_bytes": shutil.disk_usage("/mnt/c").free,
                "output_free_bytes": shutil.disk_usage(root).free}

    def run(label, arguments, timeout=1800):
        step = {"name": label, "arguments": arguments}
        report["steps"].append(step)
        save()
        started = time.monotonic()
        with (root / (label + ".log")).open("wb") as log:
            def cap():
                resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
            process = subprocess.Popen(arguments, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=cap)
            step["owned_process_group"] = process.pid
            save()
            failure = None
            while process.poll() is None:
                sample = resources()
                sample.update(step=label, elapsed_seconds=time.monotonic() - started)
                report["resource_samples"].append(sample)
                if sample["mem_available_bytes"] < (1 << 30) or sample["c_free_bytes"] < (500 << 20) or sample["output_free_bytes"] < (10 << 30):
                    failure = "Resource guard reached."
                elif time.monotonic() - started > timeout:
                    failure = "Bounded command timeout."
                if failure:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                save()
                time.sleep(2)
            step.update(exit_code=process.returncode, seconds=time.monotonic() - started)
            if failure:
                step["guard_failure"] = failure
            save()
            if process.returncode or failure:
                raise RuntimeError(label + " failed.")

    def compare_flags(build, branch):
        reference_commands = reference / ("components-linux-v13/compile_commands.json" if branch == "hooks" else "build-v13-cuda126-attempt2/compile_commands.json")
        def owned(rows, native):
            result = {}
            for row in rows:
                path = pathlib.Path(row["file"])
                if not path.is_relative_to(native):
                    continue
                key = (row["output"], str(path.relative_to(native)))
                result[key] = compiler_identity(row, native, upstream)
            return result
        old = owned(json.loads(reference_commands.read_text()), reference / "TensorSharp.GGML.Native")
        rows = json.loads((build / "compile_commands.json").read_text())
        if any(not pathlib.Path(row["file"]).is_relative_to(driver) for row in rows):
            raise RuntimeError("A fresh build unexpectedly includes an upstream/non-owned compilation target.")
        actual = owned(rows, driver)
        missing, changed, added = sorted(set(old) - set(actual)), [key for key in old.keys() & actual.keys() if old[key] != actual[key]], sorted(set(actual) - set(old))
        details = {"branch": branch, "old_owned_entries": len(old), "actual_owned_entries": len(actual),
                   "missing": missing, "changed": [{"key": key, "old": old[key], "actual": actual[key]} for key in changed], "added": added}
        report["flags_comparisons"].append(details)
        save()
        if missing or changed or added:
            raise RuntimeError("Actual owned compiler arguments differ from the pinned successful Linux configuration.")

    try:
        if (root / "build-hooks").exists() or (root / "build-production").exists():
            raise RuntimeError("Fresh build directories required; previous attempts must remain intact.")
        report["before"] = verify()
        report["compiler_versions"] = {name: subprocess.check_output([path, "--version"], env=env, text=True)
                                       for name, path in (("gcc", "/usr/bin/gcc-12"), ("g++", "/usr/bin/g++-12"), ("nvcc", "/usr/local/cuda-12.6/bin/nvcc"), ("cmake", "cmake"))}
        common = ["cmake", "-S", str(driver), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                  "-DCMAKE_C_COMPILER=/usr/bin/gcc-12", "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12",
                  "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc", "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12",
                  "-DCUDAToolkit_ROOT=/usr/local/cuda-12.6", "-DCMAKE_CUDA_ARCHITECTURES=86-real",
                  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON",
                  "-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF", "-DTENSORSHARP_GGML_NATIVE_ENABLE_NCCL=OFF",
                  "-DTENSORSHARP_VALIDATION_UPSTREAM=" + str(upstream),
                  "-DTENSORSHARP_VALIDATION_LIBRARIES=" + str(reference_build)]
        for branch, enabled, targets in (("hooks", "ON", TARGETS), ("production", "OFF", ["GgmlOps"])):
            build = root / ("build-" + branch)
            run("configure-" + branch, common + ["-B", str(build), "-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=" + enabled], 300)
            compare_flags(build, branch)
            run("build-" + branch, ["cmake", "--build", str(build), "--parallel", "2", "--target", *targets], 3600)
            destination = root / ("components" if branch == "hooks" else "production")
            destination.mkdir(exist_ok=False)
            for target in targets:
                name = "libGgmlOps.so" if target == "GgmlOps" else target
                shutil.copy2(build / name, destination / name)
            shutil.copy2(build / "compile_commands.json", destination / "compile_commands.json")
            report[branch + "_artifacts"] = {path.name: sha(path) for path in destination.iterdir() if path.is_file()}
        symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(root / "production/libGgmlOps.so")], env=env, text=True)
        (root / "production/exports.txt").write_text(symbols)
        if any("TSGgml_" in line and "Test" in line for line in symbols.splitlines()):
            raise RuntimeError("Production unexpectedly exports native test hooks.")
        report["production_no_test_hooks"] = True
        report["after"] = verify()
        report["libraries_after"] = {name: sha(reference_build / name) for name in manifest["libraries"]}
        report["passed"] = report["before"] == report["after"] and report["libraries_before"] == report["libraries_after"]
    except BaseException as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (root / "build-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "error": report.get("error"), "summary": str(root / "build-summary.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
