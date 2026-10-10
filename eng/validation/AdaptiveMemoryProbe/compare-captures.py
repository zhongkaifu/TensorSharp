#!/usr/bin/env python3
"""Gate complete matched-history model logits, allowing bounded numerical error.

Every row requires relative L2 <= .001, cosine >= .999999 and equal argmax.
This numerical regression gate is not semantic quality or quiet throughput.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def load(path):
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    require(report.get("Executed") is True and not report.get("Error")
            and report.get("NativeShutdown") is True, "Run or native shutdown did not complete")
    require(report.get("CaptureLogits") is True, "Full-logit capture was not enabled")
    require(is_hash(report.get("ModelSha256")) and is_hash(report.get("ModelsAssemblySha256"))
            and is_hash(report.get("ProbeAssemblySha256")) and report.get("ManagedAssembliesSha256")
            and all(is_hash(value) for value in report["ManagedAssembliesSha256"].values())
            and type(report.get("ModelBytes")) is int and report["ModelBytes"] > 0
            and report.get("Native") and all(is_hash(n.get("Sha256")) for n in report["Native"]),
            "Missing checkpoint or actual binary identity")
    geometry = report.get("ModelGeometry", {})
    require(isinstance(geometry.get("Architecture"), str) and geometry["Architecture"]
            and all(type(geometry.get(key)) is int and geometry[key] > 0 for key in
                    ("HiddenSize", "NumLayers", "NumHeads", "NumKVHeads", "Vocabulary", "Context")),
            "Invalid model geometry")
    vocabulary = geometry["Vocabulary"]
    steps, repeats = report.get("Steps"), report.get("Repeats")
    require(type(steps) is int and steps >= 2 and type(repeats) is int and repeats >= 1,
            "Missing requested row counts")
    prompt = report.get("Prompt")
    valid_tokens = lambda ids: isinstance(ids, list) and all(type(x) is int and 0 <= x < vocabulary for x in ids)
    require(valid_tokens(prompt) and prompt and len(prompt) + steps <= geometry["Context"],
            "Invalid prompt or context")
    teacher = report.get("Teacher")
    if report.get("Generation") == "teacher-forced":
        require(valid_tokens(teacher) and len(teacher) == steps and is_hash(report.get("TeacherSha256")),
                "Incomplete teacher conditioning")
    else:
        require(report.get("Generation") == "raw-greedy" and teacher is None, "Unknown conditioning")
    records = report.get("Records", [])
    require([record.get("Run") for record in records] == list(range(-1, repeats))
            and all(record.get("Warmup") is (i == 0) for i, record in enumerate(records)),
            "Missing warmup or measured runs")
    for record in records:
        require(record.get("PromptTokens") == len(prompt) and record.get("DecodeCalls") == steps - 1
                and valid_tokens(record.get("Generated")) and len(record["Generated"]) == steps
                and valid_tokens(record.get("Consumed")) and len(record["Consumed"]) == steps - 1
                and record["Consumed"] == (teacher if teacher is not None else record["Generated"])[:-1]
                and is_hash(record.get("LogitsSha256")), "Incomplete run or changed token history")
    if report.get("Mode") == "adaptive":
        pools = report.get("AfterDispose")
        require(isinstance(pools, list) and pools and all(p.get("Reserved") == 0 and p.get("Committed") == 0
                                                        for p in pools), "Adaptive budget owners remain")
    else:
        require(report.get("Mode") == "resident", "Unknown placement mode")
    capture = report.get("LogitCapture") or {}
    require(capture.get("Format") == "f32le" and is_hash(capture.get("Sha256")), "Invalid capture format/hash")
    data_path, index_path = Path(capture["DataPath"]), Path(capture["IndexPath"])
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    require(index.get("Format") == "f32le" and Path(index["DataPath"]).resolve() == data_path.resolve()
            and index.get("Rows") == capture.get("Rows"), "Capture index is not bound to final report")
    rows = capture["Rows"]
    total_bytes = repeats * steps * vocabulary * 4
    require(len(rows) == repeats * steps and capture.get("Bytes") == total_bytes
            and data_path.stat().st_size == total_bytes, "Capture is incomplete or has trailing rows")
    for position, row in enumerate(rows):
        run, step = divmod(position, steps)
        require(row.get("Run") == run and row.get("Step") == step
                and row.get("Stage") == ("prefill" if step == 0 else "decode")
                and row.get("InputTokens") == prompt + records[run + 1]["Consumed"][:step]
                and row.get("Elements") == vocabulary and row.get("ByteOffset") == position * vocabulary * 4
                and row.get("Argmax") == records[run + 1]["Generated"][step] and is_hash(row.get("Sha256")),
                "Capture row is unbound, reordered or has a different history")
    return report, rows, data_path


def read_row(stream, row):
    data = stream.read(row["Elements"] * 4)
    require(len(data) == row["Elements"] * 4 and hashlib.sha256(data).hexdigest() == row["Sha256"].lower(),
            "Capture row bytes/hash differ")
    values = array.array("f", data)
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), "Nonfinite captured logits")
    require(max(range(len(values)), key=values.__getitem__) == row["Argmax"], "Recorded argmax differs from captured values")
    return values, data


def compare(left_path, right_path):
    result = {"Passed": False, "RunComplete": False, "Rows": [], "Errors": [],
              "Gate": {"MaximumRelativeL2": .001, "MinimumCosine": .999999, "EqualArgmax": True},
              "Scope": "Matched full-vocabulary numerical regression; correctness capture is not quiet performance or semantic quality."}
    try:
        left, left_rows, left_data = load(left_path)
        right, right_rows, right_data = load(right_path)
        for key in ("ModelSha256", "ModelBytes", "ModelsAssemblySha256", "ProbeAssemblySha256",
                    "ManagedAssembliesSha256", "ModelGeometry", "Prompt",
                    "Generation", "Teacher", "Steps", "Repeats"):
            require(left[key] == right[key], "Model identity, shape or conditioning differs: " + key)
        require([r["InputTokens"] for r in left_rows] == [r["InputTokens"] for r in right_rows],
                "Prediction histories differ")
        result["Identities"] = [{"Report": str(path), "ReportSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ModelSha256": report["ModelSha256"], "ModelsAssemblySha256": report["ModelsAssemblySha256"],
            "ProbeAssemblySha256": report["ProbeAssemblySha256"],
            "ManagedAssembliesSha256": report["ManagedAssembliesSha256"], "Environment": report.get("Environment"),
            "Native": report["Native"], "CaptureSha256": report["LogitCapture"]["Sha256"]}
            for path, report in ((left_path, left), (right_path, right))]
        full_hashes = [hashlib.sha256(), hashlib.sha256()]
        run_hashes = [hashlib.sha256(), hashlib.sha256()]
        with left_data.open("rb") as astream, right_data.open("rb") as bstream:
            for arow, brow in zip(left_rows, right_rows):
                a, abytes = read_row(astream, arow)
                b, bbytes = read_row(bstream, brow)
                for index, data in enumerate((abytes, bbytes)):
                    full_hashes[index].update(data); run_hashes[index].update(data)
                aa, bb = math.fsum(x * x for x in a), math.fsum(x * x for x in b)
                error = math.fsum((x - y) ** 2 for x, y in zip(a, b))
                relative = math.sqrt(error) / max(math.sqrt(aa), 1e-150)
                cosine = (1.0 if error == 0 else 0.0) if aa == 0 or bb == 0 else math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(aa * bb)
                result["Rows"].append({"Run": arow["Run"], "Step": arow["Step"], "Stage": arow["Stage"],
                    "InputTokens": arow["InputTokens"], "Elements": len(a), "RelativeL2": relative,
                    "Cosine": cosine, "MaximumAbsoluteError": max(abs(x - y) for x, y in zip(a, b)),
                    "Argmax": [arow["Argmax"], brow["Argmax"]], "BitwiseEqual": abytes == bbytes,
                    "Passed": relative <= .001 and cosine >= .999999 and arow["Argmax"] == brow["Argmax"]})
                if arow["Step"] + 1 == left["Steps"]:
                    for index, report in enumerate((left, right)):
                        require(run_hashes[index].hexdigest() == report["Records"][arow["Run"] + 1]["LogitsSha256"].lower(),
                                "Complete measured-run hash differs")
                        run_hashes[index] = hashlib.sha256()
        for digest, report in zip(full_hashes, (left, right)):
            require(digest.hexdigest() == report["LogitCapture"]["Sha256"].lower(), "Complete capture hash differs")
        result["RunComplete"] = True
        result["Passed"] = all(row["Passed"] for row in result["Rows"])
    except Exception as error:
        result["Errors"].append(str(error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "Rows"}, indent=2))
    raise SystemExit(0 if result["Passed"] else 1)
