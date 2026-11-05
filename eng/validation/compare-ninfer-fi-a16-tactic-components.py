#!/usr/bin/env python3
"""Require all 48 full synthetic outputs to retain the frozen N128 tactic bits."""
import argparse
import hashlib
import json
import pathlib

REVISION = "61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0"
PREFIX = "flashinfer61bc8-native-raw-nv4-bf16-mma-f32-alpha-post-"
BASE = PREFIX + "s1-w4-k128-st2-v1"
KNOWN = {BASE: (4, 2, 128), PREFIX + "s1-w8-k128-st2-v1": (8, 2, 128),
         PREFIX + "s1-w4-k128-st3-v1": (4, 3, 128),
         "flashinfer61bc8-n64-raw-nv4-bf16-mma-f32-alpha-post-s1-w4-k128-st2-v1": (4, 2, 64)}
FAMILIES = {"signed-zero", "exact-lattice", "f32-output", "broad-finite"}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def body(data, size):
    need(len(data) == size + 512 and data[:256] == b"\xa5" * 256 and data[-256:] == b"\xa5" * 256, "Raw output length/guards differ")
    return data[256:-256]


def audit(path):
    raw = path.read_bytes()
    need(path.with_name(path.name + ".sha256").read_text() == sha(raw) + "  completed-result.json\n", "Write-once completed certificate sidecar differs")
    root = json.loads(raw)
    need(root["schema"] == "ninfer-flashinfer-native-w4a16-component/1" and root["complete"] is True and root["passed"] is True
         and root["GPU"] == "EXECUTED" and root["localWeightProductsExactlyBf16"] == 2032, "Actual passing component certificate required")
    folder = path.parent
    manifest_bytes = (folder / "module-input.json").read_bytes()
    need(sha(manifest_bytes) == root["manifestSha256"], "Captured AOT manifest differs")
    manifest = json.loads(manifest_bytes)
    policy = manifest["arithmeticPolicy"]
    need(policy in KNOWN and manifest["status"] == "complete" and manifest["sourceVerification"]["revision"] == REVISION, "Unknown AOT source/policy")
    w, stages, tile_n = KNOWN[policy]
    need(manifest["splits"] == 1 and manifest["warps"] == w and manifest["stages"] == stages and manifest["tileK"] == 128
         and manifest["enablePdl"] is False and manifest["outputDtype"] == "Float32", "Mislabeled AOT tactic")
    images = {x["rows"]: x["cubinSha256"].lower() for x in manifest["specializations"]}
    need(set(images) == {1, 8}, "Both static image identities required")
    records = root["records"]
    need(len(records) == 16 and len({x["id"] for x in records}) == 16, "All16 unique lifecycle records required")
    outputs, cases = {}, set()
    for record in records:
        family, rows, lifetime = record["family"], record["rows"], record["lifetime"]
        case = family + "-t" + str(rows)
        need(family in FAMILIES and rows in (1, 8) and lifetime in (0, 1) and record["case"] == case
             and record["id"] == "lifetime" + str(lifetime) + "-" + case and record["passed"] is True
             and record["imageSha256"].lower() == images[rows], "Unexpected case geometry, image or failed status")
        cases.add((case, lifetime))
        passes = record["passes"]
        need(len(passes) == 3 and {x["pass"] for x in passes} == {0, 1, 2}, "All three poison passes required")
        for p in passes:
            need(all(p[k] is True for k in ("passed", "guardsIntact", "inputsImmutable", "repeatedBits"))
                 and p["nonfinite"] == 0 and p["boundFailures"] == 0, "Original numerical/immutability/repeat gate failed")
            if record["exactBitsRequired"]:
                need(p["differentF32BitsFromExact"] == 0, "Original exact bit gate failed")
            if lifetime == 1:
                need(p["crossLifetimeBits"] is True, "Original cross-lifetime gate failed")
            name = p["file"]
            need(pathlib.Path(name).name == name, "Unsafe original capture path")
            data = (folder / name).read_bytes()
            need(sha(data) == p["sha256"], "Original full guarded capture SHA differs")
            values = body(data, rows * 34816 * 4)
            need(sha(values) == p["bodySha256"], "Original active body SHA differs")
            key = (record["id"], p["pass"])
            need(key not in outputs, "Duplicate raw comparison identity")
            outputs[key] = values
    need(len(cases) == 16 and len(outputs) == 48, "Complete 8case/2lifetime/3pass coverage required")
    inputs = {}
    for family in sorted(FAMILIES):
        for name in ("original-codes.bin", "original-scales.bin"):
            rel = "weights-" + family + "/" + name
            inputs[rel] = file_sha(folder / rel)
        for rows in (1, 8):
            case = family + "-t" + str(rows)
            for name in ("input.bf16", "alpha.f32", "ideal.f64", "bounds.f64") + (() if family == "broad-finite" else ("expected.f32",)):
                rel = case + "/" + name
                inputs[rel] = file_sha(folder / rel)
    return {"certificateSha256": sha(raw), "manifestSha256": root["manifestSha256"], "policy": policy, "tileN": tile_n}, inputs, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        values = bytes(range(16)); need(body(b"\xa5" * 256 + values + b"\xa5" * 256, 16) == values, "Raw extraction test failed")
        try:
            body(b"\x00" * 256 + values + b"\xa5" * 256, 16)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed guard accepted")
        print(json.dumps({"rawBodyGuardSelfTest": "PASS", "GPU": "NOT_RUN", "comparison": "NOT_RUN"}))
        return
    if not all((args.reference, args.candidate, args.out)):
        parser.error("--reference --candidate --out required")
    need(not args.out.exists() and not args.out.is_symlink(), "Use a fresh comparison artifact")
    need("artifacts" in args.out.parts or any(args.out.parts[i:i + 2] == ("docs", "validation") for i in range(len(args.out.parts) - 1)), "Ignored evidence destination required")
    ref, ref_inputs, ref_outputs = audit(args.reference)
    candidate, candidate_inputs, candidate_outputs = audit(args.candidate)
    need(ref["policy"] == BASE and candidate["policy"] != BASE, "Compare a distinct known tactic to the frozen original policy")
    need(ref_inputs == candidate_inputs, "Original fixtures, FP64 bounds or exact CPU expectations changed")
    need(ref_outputs.keys() == candidate_outputs.keys(), "Original output coverage differs")
    comparisons = [{"id": key[0], "pass": key[1], "bytes": len(value), "sameBits": value == candidate_outputs[key],
                    "referenceSha256": sha(value), "candidateSha256": sha(candidate_outputs[key])} for key, value in sorted(ref_outputs.items())]
    passed = all(x["sameBits"] for x in comparisons)
    result = {"schema": "ninfer-flashinfer-tactic-component-bitcomparison/1", "passed": passed, "reference": ref, "candidate": candidate,
              "originalInputsAndOraclesIdentical": True, "fullOutputComparisons": len(comparisons), "comparisons": comparisons,
              "scope": "All fixed synthetic component outputs only; not a universal floating-point proof, whole-model quality, or performance qualification"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, indent=2) + "\n").encode()
    with args.out.open("xb") as output:
        output.write(data)
    with args.out.with_name(args.out.name + ".sha256").open("x") as output:
        output.write(sha(data) + "  " + args.out.name + "\n")
    print(json.dumps({"passed": passed, "fullOutputComparisons": len(comparisons), "wholeModelQuality": "NOT_RUN"}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
