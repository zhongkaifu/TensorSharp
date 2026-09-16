#!/usr/bin/env python3
"""Drive the Qwen4Exp MTP executor through ggml-cuda's CUDA-graph *update* path.

eng/tests/qwen4exp-mtp-operator.py never runs two identical forwards after a
shape change, so ggml only ever instantiates a fresh executable graph and never
calls cudaGraphExecUpdate on an instance captured for another topology. That
update is the call compute-sanitizer reports as a "handled graph-update API
error" (cudaErrorGraphExecUpdateFailure, cleared and re-instantiated by
ggml_cuda_graph_update_executable). This probe alternates two token counts and
calls each shape three times back to back:

  call 0  node properties differ from the captured graph -> "warmup reset",
          executed directly;
  call 1  properties stable -> "warmup complete": stream capture, then
          cudaGraphExecUpdate(old instance, new graph); a topology change makes
          that return cudaErrorGraphExecUpdateFailure and ggml destroys and
          re-instantiates the executable;
  call 2  replay of the instantiated graph.

Every call is compared elementwise against the operator fixture's F64 oracle
(atol=rtol=2e-5), and round N>0 must reproduce round 0 bit for bit, so a wrong
or stale executable graph fails the probe rather than only the sanitizer.
Run it with TS_GGML_LOG_DEBUG=1 to see ggml's warmup lines, and under
compute-sanitizer with and without GGML_CUDA_DISABLE_GRAPHS=1.
"""
import argparse
import ctypes as C
import datetime
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np

P, I, L = C.c_void_p, C.c_int, C.c_int64


def load_operator(path):
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_operator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--operator", type=Path, default=None,
                        help="path to eng/tests/qwen4exp-mtp-operator.py (default: relative to this repository)")
    parser.add_argument("--capacity", type=int, choices=(32, 512), default=512)
    parser.add_argument("--kv-alignment", type=int, choices=(4, 64), default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shapes", default="1,5", help="comma-separated token counts to alternate between")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path")
    if args.operator is None:
        args.operator = Path(__file__).resolve().parents[3] / "eng" / "tests" / "qwen4exp-mtp-operator.py"
    if not args.operator.is_file():
        parser.error(f"operator fixture not found: {args.operator}; pass --operator")
    op = load_operator(args.operator)
    shapes = [int(x) for x in args.shapes.split(",")]
    report = dict(schema_version=1, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope=__doc__, atol=op.ATOL, rtol=op.RTOL, rounds=args.rounds, shapes=shapes,
                  cache_capacity=args.capacity, kv_alignment=args.kv_alignment,
                  cuda_graphs_disabled_by_env=os.environ.get("GGML_CUDA_DISABLE_GRAPHS") is not None,
                  native_sha256=op.sha(args.library), operator_sha256=op.sha(args.operator),
                  sample_sha256=op.sha(args.sample), checks=[], calls=[])
    sample = json.loads(args.sample.read_text())
    fixture = op.Fixture(sample, capacity=args.capacity, kv_alignment=args.kv_alignment)
    os.environ["TS_GGML_CPU_THREADS"] = "2"
    handles, dll = [], None

    def check(name, passed, **extra):
        report["checks"].append(dict(name=name, passed=bool(passed), **extra))
        if not passed:
            raise AssertionError(name)

    def compare(name, actual, expected):
        diff = actual.astype(np.float64) - expected
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        check(name, finite and np.allclose(actual, expected, atol=op.ATOL, rtol=op.RTOL), finite=finite,
              max_abs=float(np.max(np.abs(diff))) if finite else None)

    try:
        dll = C.CDLL(str(args.library.resolve()))
        error = dll.TSGgml_GetLastError; error.argtypes = []; error.restype = C.c_char_p
        create = dll.TSGgml_Qwen4ExpMtpCreate
        create.argtypes = [C.POINTER(x) for x in (op.Config, op.Attn, op.Ffn, op.Head)]; create.restype = P
        forward = dll.TSGgml_Qwen4ExpMtpForward
        forward.argtypes = [P, P, P, I, I, I, P, P, P]; forward.restype = I
        free = dll.TSGgml_Qwen4ExpMtpFree; free.argtypes = [P]; free.restype = None
        check("cuda_backend_initialized", op.initialize_fixture_backend(dll, "CUDA", report))
        handle = create(C.byref(fixture.c), C.byref(fixture.a), C.byref(fixture.f), C.byref(fixture.o))
        check("create", bool(handle), native_error=(error() or b"").decode(errors="replace"))
        handles.append(handle)
        first_round = {}

        def call(name, count, seed):
            e, h = fixture.inputs(count, seed)
            expected_hidden, expected_logits = fixture.oracle(e, h, 0, 0)
            out_h = np.full((fixture.HC, fixture.H), np.nan, np.float32)
            out_l = np.full(fixture.VOCAB, np.nan, np.float32)
            rc = forward(handle, e.ctypes.data, h.ctypes.data, count, 0, 0, None, out_h.ctypes.data, out_l.ctypes.data)
            check(name + "_forward", rc == 1, native_error=(error() or b"").decode(errors="replace"))
            compare(name + "_wide_hidden", out_h, expected_hidden[-1])
            compare(name + "_logits", out_l, expected_logits[-1])
            report["calls"].append(dict(name=name, tokens=count, seed=seed, hidden=out_h.tolist(), logits=out_l.tolist()))
            return out_h, out_l

        for round_index in range(args.rounds):
            for count in shapes:
                for rep in range(3):
                    name = f"round{round_index}_tokens{count}_call{rep}"
                    seed = 900 + 10 * count + rep
                    out = call(name, count, seed)
                    key = (count, rep)
                    if round_index == 0:
                        first_round[key] = out
                    else:
                        check(name + "_bitwise_equals_round0",
                              np.array_equal(out[0], first_round[key][0]) and np.array_equal(out[1], first_round[key][1]))
    except op.BackendUnavailable as exc:
        report["unavailable"] = str(exc)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if dll is not None:
            for handle in handles:
                dll.TSGgml_Qwen4ExpMtpFree(handle)
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["passed"] = not report.get("unavailable") and not report.get("error") and all(x["passed"] for x in report["checks"])
        report["status"] = "unavailable" if report.get("unavailable") else "passed" if report["passed"] else "failed"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{report['status'].upper()}: {len(report['checks'])} checks; {len(report['calls'])} forwards; {args.report}", flush=True)
        if report.get("error"):
            print(report["error"], flush=True)
    return 77 if report.get("unavailable") else 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
