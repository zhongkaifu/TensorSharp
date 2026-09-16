#!/usr/bin/env python3
"""Archive completed Q8 embedding evidence without executing model workloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    results = args.wsl_base / "results"
    build = results / "q8-projection-build"
    correctness = results / "embedding-q8-f32-correctness"
    abba = results / "embedding-q8-f32-abba"
    oracle = results / "embedding-q8-f32-numpy-oracle"
    model_tests = results / "embedding-q8-f32-model-tests"
    expected = {"before": "c0abb8bc954ba62ac616b4225d3cba69bf0ced78b973b822416a8df53221d692",
                "candidate": "b4ed6a28b2c88f5ba4aef06cf3245c86e0a24e4ad8d93839cd4c8694ab9471df"}
    record = read(build / "build.json")
    require(record["status"] == "passed" and record["exit_code"] == 0 and record["finished_at_unix"], "Build incomplete")
    require(record["native_sha256"] == expected["candidate"], "Candidate build digest differs")
    require(all(row["exit_code"] == 0 for row in read(build / "native-tests.json")), "Native gates failed")
    before, after = read(build / "ggml-before.json"), read(build / "ggml-after.json")
    require(before == after and not after["modified_or_missing"] and not after["extra_files"], "Upstream tree changed")
    for relative, digest in record["source_sha256"].items():
        require(sha(build / "sources" / relative) == digest, f"Compiled source changed: {relative}")
    runs = read(abba / "runs.json")
    require(len(runs) == 8 and all(row["exit_code"] == 0 for row in runs), "ABBA incomplete")
    quality_runs = read(correctness / "runs.json")
    require(len(quality_runs) == 2 and all(row["exit_code"] == 0 for row in quality_runs), "Correctness incomplete")
    performance = read(abba / "summary.json")
    require(performance["latency_gate_ratio"] == 1.05, "Latency threshold changed")
    require(len(read(oracle / "run.json")) == 2 and all(r["exit_code"] == 0 for r in read(oracle / "run.json")), "Oracle incomplete")
    test_run, qualification = read(model_tests / "run.json"), read(model_tests / "qualification.json")
    require(test_run["finished_at_unix"] and test_run["exit_code"] == 0, "Model tests incomplete")
    require(not test_run["passed"] and qualification["passed"] and qualification["native_binding_passed"], "Original/qualified status differs")
    require(qualification["test_counters"]["passed"] == "19" and qualification["test_counters"]["failed"] == "0", "Test counts differ")
    require(qualification["native_sha256"] == expected["candidate"] and qualification["opt_in_banner_observed"], "Test binding differs")
    for relative, digest in qualification["evidence_sha256"].items():
        require(sha(model_tests / relative) == digest, f"Test evidence changed: {relative}")
    all_profiles, model_summary = [], {}
    for model in ("minilm", "snowflake"):
        profile = read(correctness / model / "profile.json")
        diagnostic = read(correctness / model / "diagnostic.json")
        quality = read(correctness / model / "embeddings.json")
        boundaries = read(correctness / model / "embedding-boundaries.json")
        require(profile["status"] == quality["status"] == boundaries["status"] == "passed", "Correctness failure")
        require(diagnostic["consistency_gate"]["passed"], "Consistency gate failed")
        require(len(boundaries["cases"]) == 7 and all(c["status"] == "passed" for c in boundaries["cases"]), "Boundary count differs")
        require(profile["profile"]["env"]["TS_EMBEDDING_Q8_F32"] == "1", "Candidate not enabled")
        all_profiles.append((model + "-correctness", profile, "candidate"))
        selected = [row for row in runs if row["model"] == model]
        require([r["label"] for r in selected] == ["before", "candidate", "candidate", "before"], "ABBA order differs")
        processes = {"before": [], "candidate": []}
        shape_keys = None
        for row in selected:
            folder = abba / Path(row["output"]).name
            p, d = read(folder / "profile.json"), read(folder / "diagnostic.json")
            require(p["status"] == "passed" and d["status"] == "diagnostic-complete", "Benchmark process failed")
            require(d["consistency_gate"]["passed"] == (row["label"] == "candidate"), "Baseline/candidate consistency verdict differs")
            all_profiles.append((folder.name, p, row["label"]))
            require(p["profile"]["env"]["TS_EMBEDDING_Q8_F32"] == ("1" if row["label"] == "candidate" else "0"), "Dispatch label differs")
            require(d["measurement_settings"]["warmup"] == 5, "Warmups differ")
            shape_keys = shape_keys or set(d["benchmarks"])
            require(set(d["benchmarks"]) == shape_keys and len(shape_keys) == 7, "Benchmark shapes differ")
            processes[row["label"]].append(d)
        for shape in performance["models"][model]["shapes"]:
            medians = {label: [p["benchmarks"][shape["shape"]]["median_ms"] for p in ps] for label, ps in processes.items()}
            require(medians == shape["process_medians_ms"], "Reported process medians differ")
            a, b = statistics.median(medians["before"]), statistics.median(medians["candidate"])
            require(a == shape["before"] and b == shape["candidate"] and b / a == shape["ratio"], "Aggregated latency differs")
            require(shape["passed"] == (b / a <= 1.05), "Performance verdict differs")
        comparison = quality["comparison"]
        c = quality["correctness"]
        model_summary[model] = {
            "correctness_status": quality["status"], "minimum_batch_single_cosine": c["batch_min_cosine"],
            "batch_single_cosine_gate": c["batch_consistency_gate"],
            "maximum_diagnostic_component_error": max(row["batch_max_component_error"] for row in diagnostic["rows"]),
            "cross_backend_status": comparison["status"], "cross_backend_minimum_cosine": comparison["min_cosine"],
            "cross_backend_maximum_component_error": comparison["max_component_error"],
            "dimensions": c["dimensions"], "dimensions_tested": c["dimensions_tested"],
            "invalid_requests": c["invalid_requests"], "concurrent_requests": c["concurrent_requests"],
            "boundary_cases": 7, "boundary_passed": 7,
            "model_assets": profile["profile"]["model_assets"],
            "performance": performance["models"][model],
        }
        oracle_result = read(oracle / (model + ".json"))
        oracle_input = read(oracle / (model + "-http.json"))
        require(oracle_result["passed"], "Independent oracle failed")
        require(oracle_result["http_results_sha256"] == sha(oracle / (model + "-http.json")), "Oracle input differs")
        require(oracle_input["source_sha256"] == sha(correctness / model / "embeddings.json"), "Oracle source report differs")
        require(oracle_result["model_sha256"] == profile["profile"]["model_assets"][0]["fresh_observed_sha256"], "Oracle model digest differs")
        model_summary[model]["independent_numpy_oracle"] = {k: oracle_result[k] for k in ("passed", "minimum_cosine", "maximum_absolute_error", "model_sha256", "oracle_script_sha256", "reference_vectors_sha256", "token_fixture_sha256")}
    assemblies = all_profiles[0][1]["managed_assemblies"]
    for name, p, label in all_profiles:
        require(p["finished_at_unix"] and all(s["exit_code"] == 0 for s in p["suites"]), f"Profile incomplete: {name}")
        require(set(p["loaded_native_libraries"].values()) == {expected[label]}, f"Mapped native differs: {name}")
        require(p["loaded_native_libraries"] == p["loaded_native_libraries_after_suites"], f"Native changed: {name}")
        require(p["managed_assemblies"] == assemblies, f"Managed assemblies differ: {name}")
    manifest = {"archived_at_unix": time.time(), "scope": "Only completed build/correctness/ABBA/oracle/model-test directories; no inference or build run by archiver",
                "files": [], "external_binaries": [], "native_sha256": expected,
                "native_source_files_verified": len(record["source_sha256"]),
                "upstream_revision": after["revision"], "upstream_manifest_sha256": after["manifest_sha256"],
                "all_ten_process_native_mappings_verified_before_and_after": True,
                "all_ten_process_managed_assemblies_identical": True,
                "excluded": ["Large native binaries remain in the WSL evidence store; digests freshly verified below", "Model files", "Later or running campaigns", "Python bytecode"]}
    def copy(origin, relative, pin=None):
        digest = sha(origin)
        require(pin is None or digest == pin, f"Pinned file changed: {origin}")
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists() or sha(destination) == digest, f"Refusing different evidence overwrite: {destination}")
        if any(row["archive"] == relative.as_posix() for row in manifest["files"]):
            return
        shutil.copy2(origin, destination)
        require(sha(destination) == sha(origin) == digest, f"Copy changed: {origin}")
        manifest["files"].append({"source": str(origin), "archive": relative.as_posix(), "bytes": destination.stat().st_size,
                                  "source_sha256": digest, "archived_sha256": digest})
    for root in (build, correctness, abba, oracle, model_tests):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix != ".so" and "__pycache__" not in path.parts:
                copy(path, Path(root.name) / path.relative_to(root))
    for label, path in (("candidate", build / "candidate-libGgmlOps.so"),
                        ("before", results / "embedding-attention-control-build/before-libGgmlOps.so")):
        require(sha(path) == expected[label], "Preserved native artifact differs")
        manifest["external_binaries"].append({"label": label, "path": str(path), "bytes": path.stat().st_size, "sha256": expected[label]})
    for campaign in (correctness, abba):
        for relative, digest in read(campaign / "harness-manifest.json").items():
            matches = [p for p in (args.repo / relative, args.wsl_base / "repo" / relative) if p.exists() and sha(p) == digest]
            require(matches, f"No hash-matching harness copy: {relative}")
            copy(matches[0], Path("harness") / digest / relative, digest)
    for name in ("build-wsl-q8-projection.py", "run-wsl-q8-embedding-correctness.py", "run-wsl-q8-embedding-abba.py"):
        copy(args.repo / "TestResults/ggml-no-patch-2026-09-15" / name, Path("drivers") / name)
    copy(args.repo / "TestResults/ggml-no-patch-2026-09-15/test-wsl-q8-embedding-bound.py", Path("drivers/test-wsl-q8-embedding-bound-current-fixed.py"))
    copy(args.repo / "InferenceWeb.Tests/EmbeddingModelTests.cs", Path("test-sources/EmbeddingModelTests.cs"), test_run["test_source_sha256"])
    for model in ("minilm", "snowflake"):
        result = read(oracle / (model + ".json"))
        copy(args.repo / "eng/embedding-reference.py", Path("oracle-inputs/embedding-reference.py"), result["oracle_script_sha256"])
        for field in ("token_fixture", "reference_vectors"):
            original = Path(result[field])
            copy(original, Path("oracle-inputs") / original.name, result[field + "_sha256"])
    copy(Path(__file__), Path("archive-q8-embedding-prototype.py"))
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    (args.destination / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {"status": "not_release_qualified", "opt_in_only": True, "native_sha256": expected,
               "models": model_summary, "performance_gate_ratio": 1.05, "performance_cases": 14,
               "performance_passed": sum(s["passed"] for m in model_summary.values() for s in m["performance"]["shapes"]),
               "independent_numpy_oracle": "Both models passed unchanged thresholds using preserved NumPy F32 reference vectors, exact token/model/report hashes",
               "mapped_managed_actual_model_tests": {"passed": 19, "failed": 0, "native_sha256": expected["candidate"], "original_runner_passed": False, "qualified_same_execution_passed": True, "note": qualification["qualification_note"], "performance_qualified": False},
               "qualification": "Ten HTTP processes completed with exact native mappings and identical managed assembly hashes. Both HTTP correctness suites, 14 boundary cases, independent NumPy comparisons and 19 mapped managed model tests pass. All 14 ABBA 5% latency gates fail. Baseline has known batch-consistency failures; its successful timing process does not imply correctness."}
    (args.destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    parent_path = args.destination.parent / "summary.json"
    parent = read(parent_path)
    require("q8_f32_prototype" not in parent, "Summary already exists; review before replacing")
    parent["q8_f32_prototype"] = {"evidence": "q8-f32-prototype/summary.json", **summary}
    parent_path.write_text(json.dumps(parent, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": manifest["file_count"], "bytes": manifest["total_bytes"],
                      "compiled_sources_verified": manifest["native_source_files_verified"],
                      "profile_mappings_verified": len(all_profiles), "boundary_passed": 14,
                      "performance_passed": summary["performance_passed"], "performance_cases": 14}, indent=2))


if __name__ == "__main__":
    main()
