#!/usr/bin/env python3
"""Prepare an immutable Linux native validation candidate; never compile it."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

REVISION = "3612c0af93f2cfc8f048295300a61187723b12ae"
LIBRARIES = ("libggml.a", "libggml-cpu.a", "libggml-cuda.a", "libggml-base.a")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def linux_path(path):
    path = path.resolve()
    if sys.platform == "win32":
        return "/mnt/" + path.drive[0].lower() + "/" + path.as_posix()[3:]
    return str(path)


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--native-summary", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo, reference, output = (p.resolve() for p in (args.repo, args.reference, args.output))
    if output.exists() or output.is_relative_to(reference) or output.is_relative_to(repo):
        raise RuntimeError("A new output outside the repository and frozen reference is required.")
    summary = json.loads(args.native_summary.read_text())
    old_summary_path = reference / "linux-build-summary-attempt2.json"
    old = json.loads(old_summary_path.read_text())
    if not summary["passed"] or not old["passed"]:
        raise RuntimeError("Both native source qualification and successful Linux dependency build are required.")
    source = summary["source_after"]
    if source != summary["source_before"] or source != summary["source_snapshot"]:
        raise RuntimeError("Qualified native source bookends disagree.")
    current = {name: sha(repo / name) for name in source}
    listed = git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "--", "TensorSharp.GGML.Native").splitlines()
    if set(listed) != set(source) or current != source:
        raise RuntimeError("Current complete native input closure differs from the qualified source.")
    if old["upstream_before"] != old["upstream_after"] or old["upstream_after"]["revision"] != REVISION:
        raise RuntimeError("Frozen Linux upstream bookends disagree.")
    if old["upstream_after"]["status"] or old["upstream_after"]["diff_index_exit"]:
        raise RuntimeError("Frozen Linux upstream was not clean.")
    upstream = reference / "ExternalProjects/ggml"
    upstream_files = old["upstream_after"]["files"]
    pending_linux_reads = []
    for name, expected in upstream_files.items():
        try:
            actual = sha(upstream / name)
        except OSError:
            if sys.platform != "win32":
                raise
            # NTFS Linux symlinks cannot be opened by Windows pathlib. They must
            # be checked under Linux before configuration/compilation starts.
            pending_linux_reads.append(name)
            continue
        if actual != expected:
            raise RuntimeError("Frozen Linux upstream source bytes changed: " + name)
    build = reference / "build-v13-cuda126-attempt2"
    library_pins = {name: sha(build / name) for name in LIBRARIES}
    if any(library_pins[name] != old["production_artifacts"][name] for name in LIBRARIES):
        raise RuntimeError("Frozen Linux static library bytes differ from the actual build record.")
    reference_files = [old_summary_path, build / "CMakeCache.txt", build / "compile_commands.json",
                       build / "build.ninja", build / "CMakeFiles/rules.ninja",
                       reference / "components-linux-v13/compile_commands.json"]
    pins = {path.relative_to(reference).as_posix(): sha(path) for path in reference_files}
    output.mkdir(parents=True, exist_ok=False)
    for name in source:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, destination)
    snapshot = {name: sha(output / "source" / name) for name in source}
    if snapshot != source:
        raise RuntimeError("Prepared native source snapshot differs.")
    # This generated build driver changes CMake dependency consumption only. Every
    # compiled TensorSharp source remains identical to the qualified snapshot.
    driver = output / "driver"
    shutil.copytree(output / "source/TensorSharp.GGML.Native", driver)
    formal_cmake = (driver / "CMakeLists.txt").read_text()
    replacements = {
        'add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml" g)':
            'include("${CMAKE_CURRENT_SOURCE_DIR}/../import-upstream.cmake")',
        'target_compile_definitions(ggml-base PRIVATE GGML_SCHED_MAX_BACKENDS=32)':
            '# Frozen ggml-base already has GGML_SCHED_MAX_BACKENDS=32; verified by identity.',
        '${CMAKE_CURRENT_SOURCE_DIR}/../ExternalProjects/ggml': '${TENSORSHARP_VALIDATION_UPSTREAM}',
    }
    for before, after in replacements.items():
        count = formal_cmake.count(before)
        if count < 1 or (before in tuple(replacements)[:2] and count != 1):
            raise RuntimeError("Unexpected formal CMake anchor: " + before)
        formal_cmake = formal_cmake.replace(before, after)
    (driver / "CMakeLists.txt").write_text(formal_cmake, newline="\n")
    imports = '''# Generated private validation driver. No upstream build targets.
find_package(OpenMP REQUIRED COMPONENTS C CXX)
foreach(name ggml ggml-base ggml-cpu ggml-cuda)
  add_library(${name} STATIC IMPORTED GLOBAL)
  set_target_properties(${name} PROPERTIES
    IMPORTED_LOCATION "${TENSORSHARP_VALIDATION_LIBRARIES}/lib${name}.a"
    IMPORTED_NO_SYSTEM TRUE)
endforeach()
set_target_properties(ggml-base PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORSHARP_VALIDATION_UPSTREAM}/src/../include"
  INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:OpenMP::OpenMP_C>;$<LINK_ONLY:OpenMP::OpenMP_CXX>;$<LINK_ONLY:Threads::Threads>;$<LINK_ONLY:m>")
set_target_properties(ggml-cpu PROPERTIES INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:ggml-base>")
set_target_properties(ggml-cuda PROPERTIES
  INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:ggml-base>;$<LINK_ONLY:CUDA::cudart>;$<LINK_ONLY:CUDA::cublas>;$<LINK_ONLY:CUDA::cuda_driver>")
set_target_properties(ggml PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "GGML_USE_CPU;GGML_USE_CUDA"
  INTERFACE_LINK_LIBRARIES "ggml-base;$<LINK_ONLY:dl>;ggml-cpu;ggml-cuda")
add_library(ggml::ggml ALIAS ggml)
'''
    (output / "import-upstream.cmake").write_text(imports, newline="\n")
    for tool in ("build.py",):
        shutil.copy2(pathlib.Path(__file__).parent / tool, output / tool)
    include_rows = []
    suffixes = {".h", ".hpp", ".c", ".cpp", ".cu", ".cuh", ".inc", ".m", ".mm", ".metal"}
    by_name = {}
    for name in source:
        by_name.setdefault(pathlib.Path(name).name, []).append(name)
    upstream_by_name = {}
    for name in upstream_files:
        upstream_by_name.setdefault(pathlib.Path(name).name, []).append(name)
    for name in source:
        if pathlib.Path(name).suffix.lower() not in suffixes:
            continue
        for include in re.findall(r'^\s*#\s*include\s*"([^"\r\n]+)"', (repo / name).read_text(errors="replace"), re.M):
            local = (repo / name).parent / include
            candidates = ([str(local.relative_to(repo)).replace("\\", "/")] if local.is_file() else [])
            candidates += by_name.get(pathlib.Path(include).name, [])
            candidates += ["ExternalProjects/ggml/" + x for x in upstream_by_name.get(pathlib.Path(include).name, [])]
            include_rows.append({"source": name, "include": include, "candidates": sorted(set(candidates))})
    unresolved = [row for row in include_rows if not row["candidates"]]
    if unresolved:
        raise RuntimeError("Unresolved quoted native include closure: " + repr(unresolved))
    manifest = {
        "prepared": True, "built": False, "qualified": False,
        "repository": linux_path(repo), "reference": linux_path(reference),
        "output": linux_path(output), "native_summary_sha256": sha(args.native_summary),
        "source": source, "snapshot": snapshot, "upstream_revision": REVISION,
        "upstream_files": upstream_files, "reference_files": pins, "libraries": library_pins,
        "upstream_windows_unreadable_pending_linux_check": pending_linux_reads,
        "driver_cmake_sha256": sha(driver / "CMakeLists.txt"),
        "import_driver_sha256": sha(output / "import-upstream.cmake"),
        "build_tool_sha256": sha(output / "build.py"),
        "quoted_includes": include_rows, "unresolved_includes": unresolved,
        "rules": {"no_upstream_compile_targets": True, "fresh_all_owned_objects": True,
                  "old_native_qualification_inherited": False, "source_cmake_changes": replacements},
    }
    (output / "prepare-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"prepared": True, "built": False, "sources": len(source),
                      "quoted_includes": len(include_rows), "output": str(output),
                      "pending_linux_upstream_reads": pending_linux_reads,
                      "run_after_benchmark_release": "python3 " + linux_path(output / "build.py")}))


if __name__ == "__main__":
    main()
