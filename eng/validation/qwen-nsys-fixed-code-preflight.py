#!/usr/bin/env python3
"""CPU-only admission for a source-pinned single-request Nsight diagnostic.

Reconstructs the original code request with the unchanged benchmark body/nonce
functions and exercises the unchanged profiler select_row for prime+measurement.
No HTTP, process, CUDA or model execution is performed by this module.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

BENCH_SHA = "4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c"
RUNTIME = "/dev/shm/ts-ninfer-r12/runtime"
MODEL = "/workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer"
SCENARIO = "r12-long_generation-code-r0"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_spec(spec, historical):
    expected = list(historical["argv"])
    expected[expected.index("--port") + 1] = "5089"
    expected.append("--no-prefix-cache")
    need(spec["argv"] == expected and spec["cwd"] == RUNTIME, "Server command differs from the pinned R12 diagnostic plan")
    need(expected[:2] == ["/root/dotnet/dotnet", RUNTIME + "/TensorSharp.Server.Host.dll"], "R12 runtime changed")
    need(expected[expected.index("--model") + 1] == MODEL, "Original artifact changed")
    environment = dict(historical["environment"])
    environment.update(TS_SCHED_MAX_RUNNING_SEQS="1", TS_Q35_NINFER_DECODE_GRAPHS="0",
                       TS_Q35_NINFER_HOST_EMBEDDING="0")
    need(spec["env"] == environment, "Unexpected environment differences from R12")
    for key, value in {"MAX_CONTEXT": "40960", "TS_Q35_UBATCH": "2048", "TS_Q35_NINFER_VERIFY": "1",
                       "TS_Q35_GREEDY_VERIFY": "1", "TS_Q35_NINFER_DECODE_GRAPHS": "0",
                       "TS_SCHED_MAX_RUNNING_SEQS": "1"}.items():
        need(environment.get(key) == value, "Fixed diagnostic policy differs: " + key)
    need("TS_Q35_NINFER_DRAFT_SHORTLIST_PTX" not in environment
         and not any("FI_" in key or "PREFILL_ATTN" in key for key in environment), "Unexpected alternative arithmetic option")


def reconstruct(bench, header):
    args = SimpleNamespace(temperature=header["sampling"]["temperature"], top_p=header["sampling"]["top_p"],
                           sampling_seed=header["sampling"]["seed"], nonce_seed=header["nonce_seed"])
    client = bench.Client.__new__(bench.Client)
    client.args, client.engine, client.model, client.extra = args, header["engine"], header["model"], header["extra_body"]
    context = bench.Ctx(args, client, {})
    need(bench.GEN_PROMPTS[0][0] == "code", "Original request index is no longer code")
    return client.body(context.messages("long_generation", 0, bench.GEN_PROMPTS[0][1], 0), header["decode_tokens"])


def validate_rows(profile, bench, rows, original, provenance):
    need(len(rows) == 2 and [(row["scenario"], row["lane"], row["step"]) for row in rows]
         == [(SCENARIO, 0, 0), (SCENARIO, 0, 1)], "Exactly one prime and one identical measurement required")
    for row in rows:
        need(row["body"] == original and row["expected"] is None and row["request_sha256"] == digest(original), "Original fixture bytes or compact digest differ")
    logical = sha(json.dumps(bench.request_identity(original), sort_keys=True, ensure_ascii=False).encode())
    need(logical == provenance["originalRecordedRequestSha256"], "Independent frozen request reconstruction differs from R12")
    prime = profile.select_row(rows, SCENARIO, 0, 0, 128)
    measured = profile.select_row(rows, SCENARIO, 0, 1, 128)
    capped = copy.deepcopy(original)
    capped["max_tokens"] = 128
    need(prime["body"] == measured["body"] == capped, "Prime/profile body must differ only by the diagnostic cap")
    need(prime["request_sha256"] == measured["request_sha256"] == digest(capped)
         == provenance["diagnosticWireRequestSha256"], "Diagnostic compact digest differs")
    need(original["max_tokens"] == 1024 and provenance["originalFullWireCompactSha256"] == digest(original), "Original1024-token source body changed")
    return {"originalLogicalSha256": logical, "originalFullWireCompactSha256": digest(original),
            "diagnosticWireRequestSha256": digest(capped), "primeEqualsMeasured": True,
            "originalPromptTokens": provenance["originalPromptTokens"], "maxTokensDiagnosticOnly": 128}


def run(package, source_root, self_test=False):
    manifest = json.loads((package / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = Path(entry["path"])
        need(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
        data = (source_root / path).read_bytes()
        need(len(data) == entry["bytes"] and sha(data) == entry["sha256"], "Frozen source differs: " + str(path))
    for entry in manifest["controlFiles"]:
        data = (package / entry["path"]).read_bytes()
        need(len(data) == entry["bytes"] and sha(data) == entry["sha256"], "Pinned package control changed: " + entry["path"])
    bench_path = source_root / "eng/qwen38-engine-bench.py"
    need(sha(bench_path.read_bytes()) == BENCH_SHA, "Frozen benchmark source changed")
    bench = load_module("nsys_r6_frozen_bench", bench_path)
    profile = load_module("nsys_r6_frozen_profile", source_root / "eng/validation/qwen-nsys-profile.py")
    spec = json.loads((package / "server-spec.json").read_text())
    historical = json.loads((package / "r12-source-manifest.json").read_text())
    header = json.loads((package / "r12-source-header.json").read_text())
    provenance = json.loads((package / "input-provenance.json").read_text())
    need(header["harness_sha256"] == BENCH_SHA and historical["sourceCatalogSha256"] == provenance["r12SourceCatalogSha256"], "Historical source identity differs")
    # Portable source/fixture admission must not pretend the VM's cwd exists on
    # this CPU host. The controller separately verifies the real R12 directory
    # before execution; validate all other spec structure against a real local cwd.
    structural = copy.deepcopy(spec)
    structural["cwd"] = str(source_root.resolve())
    profile.validate_server_spec(structural)
    validate_spec(spec, historical)
    original = reconstruct(bench, header)
    rows = [json.loads(line) for line in (package / "fixture.jsonl").read_text().splitlines() if line]
    result = validate_rows(profile, bench, rows, original, provenance)
    if self_test:
        mutations = []
        wrong = copy.deepcopy(rows); wrong[0]["request_sha256"] = "0" * 64
        mutations.append(lambda: validate_rows(profile, bench, wrong, original, provenance))
        different = copy.deepcopy(rows); different[1]["body"]["messages"][1]["content"] += " altered"
        mutations.append(lambda: validate_rows(profile, bench, different, original, provenance))
        for field, value in [("TS_Q35_NINFER_DECODE_GRAPHS", "1"), ("TS_Q35_UBATCH", "512"), ("TS_Q35_NINFER_HOST_EMBEDDING", "1")]:
            bad = copy.deepcopy(spec); bad["env"][field] = value
            mutations.append(lambda bad=bad: validate_spec(bad, historical))
        for mutation in mutations:
            refused = False
            try:
                mutation()
            except ValueError:
                refused = True
            need(refused, "Changed fixture/policy was admitted")
        result["mutationRefusals"] = len(mutations)
    return {"CPU": "PASS", "fixture": result, "sourceFilesVerified": len(manifest["files"]),
            "controlFilesVerified": len(manifest["controlFiles"]), "HTTP": "NOT_RUN", "GPU": "NOT_RUN", "performance": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.package_dir, args.source_root, args.self_test), indent=2))


if __name__ == "__main__":
    main()
