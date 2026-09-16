#!/usr/bin/env python3
"""CPU/CUDA native Qwen4Exp target taps and recurrent rollback fixture.

Direct descriptors exercise two nonzero layers (GDN then attention), a PLE
block and routed/shared FFNs. Compares new wide/all-logit taps to ordinary
native inference and an independent F64 HC/output-head oracle. Recurrent
restoration is checked by same-schedule cold continuation, not by inspecting
opaque snapshot memory. No GGUF loading, managed holder, full-model, or
speculative acceptance claim is made. Self-comparison gate: atol=rtol=2e-5.
"""
import argparse
import ctypes as C
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

_operator_path = Path(__file__).with_name("qwen4exp-mtp-operator.py")
_spec = importlib.util.spec_from_file_location("mtp_operator_fixture", _operator_path)
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)
P, I, F = C.c_void_p, C.c_int, C.c_float


class Gdn(C.Structure):
    _fields_ = op.fields("hc_norm hc_down hc_up hc_inject qkv gate beta alpha conv1d ssm_dt ssm_a ssm_norm out_proj conv_state ssm_state",
                        "hc_down hc_up hc_inject qkv gate beta alpha out_proj",
                        "hc_down hc_up hc_inject qkv gate beta alpha out_proj")


class Ple(C.Structure):
    _fields_ = op.fields("key_w value_w norm_key norm_query norm_conv conv1d_t conv_state",
                        "key value", "key value") + [("kern", I), ("dil", I)]


class Target:
    GK, GV, GKH, GVH, CONV = 4, 4, 1, 2, 3

    def __init__(self, sample, geometry="tiny", attention_head_dim=8, attention_heads=4):
        if geometry == "gdn32":
            # CUDA ssm_conv requires a multiple of128 channels, and its GDN
            # supports head32. Keep the exact H8/HC4/attention/PLE subspace.
            self.GK = self.GV = 32
        elif geometry != "tiny":
            raise ValueError("Unknown target fixture geometry")
        self.base = b = op.Fixture(sample, seed=701, capacity=512, kv_alignment=64,
                                   attention_head_dim=attention_head_dim, attention_heads=attention_heads)
        self.g = Gdn()
        self.p = Ple()
        for field in ("hc_norm", "hc_down", "hc_up", "hc_inject",
                      "hc_down_bytes", "hc_up_bytes", "hc_inject_bytes",
                      "hc_down_type", "hc_up_type", "hc_inject_type"):
            setattr(self.g, field, getattr(b.a, field))
        cd = 2 * self.GK * self.GKH + self.GV * self.GVH
        vd = self.GV * self.GVH
        for field, shape, scale in (("qkv", (cd, b.H), .6), ("gate", (vd, b.H), .5),
                                     ("beta", (self.GVH, b.H), .5), ("alpha", (self.GVH, b.H), .5),
                                     ("conv1d", (cd, self.CONV), .7), ("out_proj", (b.H, vd), .8)):
            b.random(self.g, field, "gdn." + field, shape, scale)
        b.put(self.g, "ssm_dt", "gdn.ssm_dt", [-.2, .1])
        b.put(self.g, "ssm_a", "gdn.ssm_a", [-.7, -1.1])
        b.put(self.g, "ssm_norm", "gdn.ssm_norm", np.linspace(.8, 1.2, self.GV))
        b.put(self.g, "conv_state", "gdn.conv_state", np.zeros((cd, self.CONV - 1)))
        b.put(self.g, "ssm_state", "gdn.ssm_state", np.zeros((self.GVH, self.GV, self.GV)))
        self.p.kern, self.p.dil = 3, 2
        b.random(self.p, "key_w", "ple.key", (b.HC * b.H, b.H), .4)
        self.p.key_bytes = b.arrays["ple.key"].nbytes
        b.random(self.p, "value_w", "ple.value", (b.H, b.H), .5)
        self.p.value_bytes = b.arrays["ple.value"].nbytes
        for field in ("norm_key", "norm_query", "norm_conv"):
            b.put(self.p, field, "ple." + field, np.linspace(.7, 1.3, b.HC * b.H))
        b.random(self.p, "conv1d_t", "ple.conv1d", (self.p.kern, b.HC * b.H), .6)
        b.put(self.p, "conv_state", "ple.conv_state", np.zeros(((self.p.kern - 1) * self.p.dil, b.HC * b.H)))
        self.ffn = (op.Ffn * 2)(b.f, b.f)
        self.attn = (op.Attn * 2)(b.a, b.a)
        self.gdn = (Gdn * 2)(self.g, self.g)
        self.kinds = (C.c_uint8 * 2)(1, 0)
        self.keys = (P * 2)(self.g.conv_state, self.p.conv_state)
        self.devices = (I * 2)(0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA", "Metal"), default="CPU")
    parser.add_argument("--geometry", choices=("tiny", "gdn32"), default="tiny",
                        help="Explicit gdn32 uses128 convchannels supported byCUDA; tiny preserves the original CPU case")
    args = parser.parse_args()
    if args.report.exists(): parser.error("Use a fresh report path")
    if args.backend == "CUDA" and args.geometry == "tiny":
        parser.error("Original tiny GDN4 is unsupported by upstream CUDA ssm_conv; use explicit --geometry gdn32")
    report = dict(schema_version=1, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope=__doc__, source_sha256=op.sha(__file__), operator_source_sha256=op.sha(_operator_path),
                  native_sha256=op.sha(args.library), sample_sha256=op.sha(args.sample),
                  atol=2e-5, rtol=2e-5, checks=[], calls=[])
    os.environ["TS_GGML_CPU_THREADS"] = "2"
    report["environment"] = {k: os.environ.get(k) for k in ("TS_GGML_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "TS_Q4E_SPAN_STATE")}
    sample = json.loads(args.sample.read_text())
    target = Target(sample, args.geometry)
    report["geometry"] = dict(name=args.geometry, head_k=target.GK, head_v=target.GV,
        conv_channels=2 * target.GK * target.GKH + target.GV * target.GVH)
    b = target.base
    report["abi_sizes"] = {"Gdn": C.sizeof(Gdn), "Ple": C.sizeof(Ple)}
    report["tensor_pins"] = {k: dict(shape=list(v.shape), sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in b.arrays.items()}
    dll, directories, snapshots = None, [], []

    def check(name, ok, **details):
        report["checks"].append(dict(name=name, passed=bool(ok), **details))
        # A native assert can terminate Python before finally. Preserve completed
        # checks and identity as an explicitly unfinished report for the runner.
        report["status"] = "running"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
        if not ok: raise AssertionError(name)

    def compare(name, actual, expected):
        actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        finite = np.isfinite(actual).all() and np.isfinite(expected).all()
        difference = actual - expected
        check(name, finite and np.allclose(actual, expected, atol=2e-5, rtol=2e-5), finite=bool(finite),
              max_abs=float(np.max(np.abs(difference))) if finite else None,
              rel_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-300)) if finite else None)

    try:
        if os.name == "nt":
            directories.append(os.add_dll_directory(str(args.library.resolve().parent)))
            if os.environ.get("CUDA_PATH"):
                directories.append(os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin")))
        dll = C.CDLL(str(args.library.resolve()))
        init = dll.TSGgml_IsBackendAvailable; init.argtypes = [I]; init.restype = I
        last_error = dll.TSGgml_GetLastError; last_error.argtypes = []; last_error.restype = C.c_char_p
        # Exact Q4E_SPAN_PARAMETERS order, with explicit ctypes widths.
        signature = [P, P, P, P, I, I, P, P] + [I] * 16 + [F] * 3 + [I] * 4 + [F, I, I, P, P, P, I, P, P, P, I, I]
        normal = dll.TSGgml_Qwen4ExpTokenSpan; normal.argtypes = signature; normal.restype = I
        extended = dll.TSGgml_Qwen4ExpTokenSpanEx; extended.argtypes = signature + [P, I]; extended.restype = I
        create = dll.TSGgml_Qwen4ExpStateSnapshotCreate; create.argtypes = [P, P, I, P, P, P]; create.restype = P
        capture = dll.TSGgml_Qwen4ExpStateSnapshotCapture; capture.argtypes = [P]; capture.restype = I
        restore = dll.TSGgml_Qwen4ExpStateSnapshotRestore; restore.argtypes = [P]; restore.restype = I
        free = dll.TSGgml_Qwen4ExpStateSnapshotFree; free.argtypes = [P]; free.restype = None
        release = dll.TSGgml_Qwen4ExpReleaseAllSeqState; release.argtypes = []; release.restype = None
        release_keys = dll.TSGgml_Qwen4ExpReleaseSeqState; release_keys.argtypes = [P, I]; release_keys.restype = None
        clear = dll.TSGgml_ClearHostBufferCache; clear.argtypes = []; clear.restype = None
        check("explicit_backend_initialized", op.initialize_fixture_backend(dll, args.backend, report))

        def reset():
            release(); clear()
            for key in ("k_cache", "v_cache", "gdn.conv_state", "gdn.ssm_state", "ple.conv_state"):
                b.arrays[key].fill(0)

        def run(name, count, position, seed, mode="extended"):
            e, h = b.inputs(count, seed)
            residual = h.copy()
            mask = np.full((count, position + count), -np.inf, np.float16)
            for row in range(count): mask[row, :position + row + 1] = 0
            output = np.full((count if mode == "extended" else 1, b.VOCAB), np.nan, np.float32)
            hidden = np.full((count, b.HC, b.H), np.nan, np.float32)
            values = [C.addressof(target.ffn), C.addressof(target.gdn), C.addressof(target.attn),
                C.addressof(target.kinds), 0, 2, residual.ctypes.data, mask.ctypes.data,
                b.H, b.HC, b.LOW, count, target.GK, target.GV, target.GKH, target.GVH, target.CONV,
                b.HD, b.NH, b.NK, b.capacity, position + count, position, b.c.n_rot,
                b.c.rope_base, b.c.rope_scale, b.c.attn_scale, b.EXP, b.USED, b.FF, b.SH, b.eps,
                17, 0, C.addressof(b.o) if mode != "no-head" else None,
                output.ctypes.data if mode != "no-head" else None,
                C.addressof(target.p), 0, e.ctypes.data, None, None, position, 0]
            assert len(values) == len(signature), (len(values), len(signature))
            rc = extended(*values, hidden.ctypes.data, count) if mode == "extended" else normal(*values)
            check(name + "_forward", rc == 1, native_error=(last_error() or b"").decode(errors="replace"))
            if mode == "no-head": hidden[:] = residual
            else: check(name + "_input_immutable", np.array_equal(residual, h))
            if mode == "extended":
                mixed, _ = b.mix(hidden.astype(np.float64), "head")
                compare(name + "_all_head_rows", output, mixed @ b.w["head"].T)
            check(name + "_finite", np.isfinite(hidden).all() if mode == "no-head" else np.isfinite(output).all())
            report["calls"].append(dict(name=name, tokens=count, position=position, mode=mode,
                inputs_sha256=hashlib.sha256(e.tobytes() + h.tobytes()).hexdigest(),
                logits=output.tolist() if mode != "no-head" else None,
                hidden=hidden.tolist() if mode != "ordinary" else None))
            return hidden.copy(), output.copy()

        def make_snapshot(indices=(0, 1)):
            keys = (P * len(indices))(*(target.keys[i] for i in indices))
            devices = (I * len(indices))(*([0] * len(indices)))
            value = create(keys, devices, len(indices), C.addressof(target.attn), C.addressof(target.gdn), C.addressof(target.p))
            check("snapshot_create", bool(value))
            snapshots.append(value)
            return value

        check("null_capture_rejected", capture(None) == 0)
        check("null_restore_rejected", restore(None) == 0)
        snap = make_snapshot()
        check("restore_without_capture_rejected", restore(snap) == 0)
        free(snap); snapshots.remove(snap)

        for count in (1, 3, 5, 17):
            reset(); eh, el = run(f"taps_{count}", count, 0, 901 + count)
            reset(); _, ol = run(f"ordinary_{count}", count, 0, 901 + count, "ordinary")
            compare(f"ordinary_last_logit_{count}", el[-1], ol[-1])
            reset(); nh, _ = run(f"no_head_{count}", count, 0, 901 + count, "no-head")
            compare(f"ordinary_wide_residual_{count}", eh, nh)

        reset(); run("cold_prefix", 5, 0, 910)
        cold_h, cold_l = run("cold_continuation", 3, 5, 911)
        for indices, label in (((0, 1), "both"), ((0,), "gdn_only"), ((1,), "ple_only")):
            reset(); run(label + "_prefix", 5, 0, 910)
            snap = make_snapshot(indices)
            check(label + "_capture", capture(snap) == 1)
            run(label + "_discarded_branch", 3, 5, 912)
            check(label + "_restore", restore(snap) == 1)
            got_h, got_l = run(label + "_continuation", 3, 5, 911)
            if len(indices) == 2:
                compare("full_snapshot_hidden_vs_cold", got_h, cold_h)
                compare("full_snapshot_logits_vs_cold", got_l, cold_l)
            else:
                # Negative controls measure observable state sensitivity; this
                # is not an acceptance tolerance for a partial restoration.
                maxdiff = float(np.max(np.abs(got_l - cold_l)))
                check(label + "_insufficient_for_rollback", maxdiff > 1e-6, max_abs=maxdiff)
            free(snap); snapshots.remove(snap)

        reset()
        snap = make_snapshot()
        check("initial_unseeded_capture", capture(snap) == 1)
        run("mutate_after_unseeded_capture", 5, 0, 919)
        check("unseeded_restore", restore(snap) == 1)
        seed_h, seed_l = run("unseeded_restored_prefix", 5, 0, 910)
        free(snap); snapshots.remove(snap)
        reset(); compare_h, compare_l = run("unseeded_cold_prefix", 5, 0, 910)
        compare("unseeded_hidden_vs_cold", seed_h, compare_h)
        compare("unseeded_logits_vs_cold", seed_l, compare_l)

        snap = make_snapshot()
        check("capture_before_free", capture(snap) == 1)
        release_keys(target.keys, 2)
        check("freed_source_restore_refused", restore(snap) == 0)
        free(snap); snapshots.remove(snap)
        reset(); retry_h, retry_l = run("retry_after_freed_source", 5, 0, 910)
        compare("retry_after_free_hidden", retry_h, compare_h)
        compare("retry_after_free_logits", retry_l, compare_l)
    except op.BackendUnavailable as exc:
        report["unavailable"] = str(exc)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if dll is not None:
            for snapshot in snapshots:
                try: dll.TSGgml_Qwen4ExpStateSnapshotFree(snapshot)
                except Exception: report.setdefault("cleanup_errors", []).append(traceback.format_exc())
            try:
                dll.TSGgml_Qwen4ExpReleaseAllSeqState()
                dll.TSGgml_ClearHostBufferCache()
            except Exception: report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        for directory in reversed(directories): directory.close()
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["native_sha256_after"] = op.sha(args.library)
        if report["native_sha256_after"] != report["native_sha256"]:
            report["error"] = "Native library changed during fixture execution"
        report["passed"] = not report.get("unavailable") and not report.get("error") and not report.get("cleanup_errors") and all(c["passed"] for c in report["checks"])
        report["status"] = "unavailable" if report.get("unavailable") else "passed" if report["passed"] else "failed"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{report['status'].upper()}: {len(report['checks'])} checks; {args.report}", flush=True)
        if report.get("error"): print(report["error"], flush=True)
    return 77 if report.get("unavailable") else 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
