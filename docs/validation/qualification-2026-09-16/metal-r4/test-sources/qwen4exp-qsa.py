#!/usr/bin/env python3
"""Independent QSA arithmetic probe; synthetic inputs, not trained model quality.

The unchanged fixture gate is elementwise atol=rtol=2e-5 against F64 arithmetic
after explicitly typed F32/F16 inputs. All four arithmetic stages are retained.
Top-k ties have no index-order contract in public ggml_top_k: require every
strictly higher score, exclude every strictly lower score, and require the exact
cardinality at the cutoff. Record exact selected indices for each query.
This probe rebuilds graphs and supplies cache bytes. It does not establish
managed/native persistent cache, speculative rollback, or media encoder quality.
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

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np

ATOL = RTOL = 2e-5
P, I, L, F = C.c_void_p, C.c_int, C.c_int64, C.c_float


class Args(C.Structure):
    _fields_ = [(x, P) for x in "k_proj q_proj k_norm q_norm cache".split()] + [
        (x, L) for x in "k_bytes q_bytes cache_bytes".split()] + [
        (x, I) for x in "k_type q_type cache_type head_dim heads ratio top_k".split()] + [
        ("rope_sections", I * 4)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pointer(array):
    return array.ctypes.data_as(P)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reference_plan(positions, padded, start, count, ratio):
    """Declarative single-sequence interpretation, independent of native buffers."""
    coordinates = [tuple(map(int, row)) for row in positions]
    ranked = len({x[0] for x in coordinates}) != len(coordinates)
    order = sorted(range(len(coordinates)), key=lambda i: coordinates[i])
    ranks = {cell: rank for rank, cell in enumerate(order)}
    index = [ranks[i] if ranked else c[0] for i, c in enumerate(coordinates)]
    groups = {}
    for i, p in enumerate(index):
        groups.setdefault(p // ratio, []).append(i)
    complete = [(b, sorted(cells, key=lambda i: index[i]))
                for b, cells in sorted(groups.items())
                if len({index[i] % ratio for i in cells}) == ratio]
    blocks = (padded + ratio - 1) // ratio
    members = np.zeros((blocks, ratio), dtype=np.int32)
    key_positions = np.zeros((blocks, 3), dtype=np.int32)
    dead = min(len(complete), blocks - 1)
    cell_blocks = np.full(padded, dead, dtype=np.int32)
    bias = np.full((count, blocks), -np.inf, dtype=np.float64)
    for b, (bucket, cells) in enumerate(complete):
        members[b] = cells
        key_positions[b] = coordinates[cells[0]] if ranked else [bucket * ratio] * 3
        cell_blocks[cells] = b
    for q in range(count):
        coord = coordinates[start + q]
        q_index = sum(c <= coord for c in coordinates) - 1 if ranked else coord[0]
        tail_start = (q_index + 1) // ratio * ratio
        for b, (bucket, _) in enumerate(complete):
            bias[q, b] = 1e9 if bucket * ratio >= tail_start else 0
        if len(complete) < blocks:
            bias[q, dead] = 1e9
    return members, key_positions, cell_blocks, bias


def rms(x, gamma, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * gamma


def imrope(x, positions, nrot, sections, base, scale):
    """Half-paired rotary dimensions; interleaved THW assignment per frequency."""
    out = x.copy()
    half = nrot // 2
    axes = []
    for frequency in range(half):
        sector = frequency % sum(sections)
        axis = sector % 3
        if sector >= 3 * sections[axis]:
            axis = 0  # fourth planar coordinate duplicates time
        axes.append(axis)
    angles = positions[:, axes] * (scale * base ** (-2 * np.arange(half) / nrot))
    angles = angles.reshape((x.shape[0],) + (1,) * (x.ndim - 2) + (half,))
    cosine, sine = np.cos(angles), np.sin(angles)
    out[..., :half] = x[..., :half] * cosine - x[..., half:nrot] * sine
    out[..., half:nrot] = x[..., :half] * sine + x[..., half:nrot] * cosine
    return out


class Fixture:
    E, D, H, NROT = 12, 12, 3, 8
    SECTIONS = (2, 1, 1, 0)
    BASE, SCALE = 10000.0, 0.75
    EPS = float(np.float32(1e-5))

    def __init__(self, seed, live, padded, start, count, ratio, mode, dtype):
        self.seed, self.live, self.padded, self.start, self.count = seed, live, padded, start, count
        self.capacity, self.ratio = padded + 7, ratio
        self.topk = ratio + 1  # width=2*ratio, sparse for every selected geometry
        rng = np.random.default_rng(seed)
        self.arrays = {
            "k_proj": rng.normal(0, .27, (self.D, self.E)).astype(np.float32),
            "q_proj": rng.normal(0, .25, (self.H, self.D, self.E)).astype(np.float32),
            "k_norm": rng.uniform(.7, 1.3, self.D).astype(np.float32),
            "q_norm": rng.uniform(.6, 1.4, self.D).astype(np.float32),
            "cache": rng.normal(0, .5, (self.capacity, self.D)).astype(dtype),
            "mixed": rng.normal(0, .7, (count, self.E)).astype(np.float32)}
        p = np.repeat(np.arange(live, dtype=np.int32)[:, None], 3, axis=1)
        if mode == "offset":
            p += 101
        elif mode == "media":
            # Multiple spatial cells share time; arrival order differs from THW order.
            p = np.array([(i // 5 + 2, (i * 7) % 5, (i * 3) % 4) for i in range(live)], dtype=np.int32)
            if live > 3: p[3] = p[2]  # exact duplicate coordinate
        self.arrays["positions"] = p
        causal = np.full((count, padded), -np.inf, dtype=np.float16)
        for q in range(count):
            for k in range(live):
                if tuple(p[k]) <= tuple(p[start + q]): causal[q, k] = 0
        self.arrays["mask"] = causal
        self.args = Args()
        for name in ("k_proj", "q_proj", "k_norm", "q_norm", "cache"):
            setattr(self.args, name, pointer(self.arrays[name]))
        self.args.k_bytes = self.arrays["k_proj"].nbytes
        self.args.q_bytes = self.arrays["q_proj"].nbytes
        self.args.cache_bytes = self.arrays["cache"].nbytes
        self.args.cache_type = 1 if dtype == np.float16 else 0
        self.args.head_dim, self.args.heads = self.D, self.H
        self.args.ratio, self.args.top_k = ratio, self.topk
        self.args.rope_sections[:] = self.SECTIONS

    def oracle(self):
        a = {k: v.astype(np.float64) for k, v in self.arrays.items()}
        members, kp, mapping, bias = reference_plan(self.arrays["positions"], self.padded,
                                                   self.start, self.count, self.ratio)
        cache = a["cache"].copy()
        raw = a["mixed"] @ a["k_proj"].T
        # Native set_rows stores through the declared cache format before pooling.
        cache[self.start:self.start + self.count] = raw.astype(self.arrays["cache"].dtype).astype(np.float64)
        pooled = np.mean(cache[members], axis=1)
        keys = imrope(rms(pooled, a["k_norm"], self.EPS), kp, self.NROT,
                      self.SECTIONS, self.BASE, self.SCALE)
        qraw = np.einsum("te,hde->thd", a["mixed"], a["q_proj"])
        qp = self.arrays["positions"][self.start:self.start + self.count]
        queries = imrope(rms(qraw, a["q_norm"], self.EPS), qp, self.NROT,
                         self.SECTIONS, self.BASE, self.SCALE)
        head_scores = np.einsum("bd,thd->thb", keys, queries)
        scores = np.maximum(head_scores, 0).sum(axis=1)
        # The 1e9 tail marker is intentionally an F32 operation: it erases tiny
        # score differences within that forced tail. Do not invent a tie order.
        expanded = (scores.astype(np.float32) + bias.astype(np.float32))[:, mapping]
        expanded = expanded + a["mask"]
        wrong_pool = np.mean(rms(cache[members], a["k_norm"], self.EPS), axis=1)
        wrong_relu = np.maximum(head_scores.sum(axis=1), 0)
        return dict(pooled=pooled, keys=keys, queries=queries, scores=scores,
                    expanded=expanded, members=members, block_positions=kp,
                    cell_blocks=mapping, bias=bias,
                    wrong_pool_max=float(np.max(np.abs(wrong_pool - rms(pooled, a["k_norm"], self.EPS)))),
                    wrong_relu_max=float(np.max(np.abs(wrong_relu - scores))))

    def run(self, function, descriptor=None, **overrides):
        B = (self.padded + self.ratio - 1) // self.ratio
        outputs = {"mask": np.full((self.count, self.padded), 123, np.float16),
                   "pooled": np.full((B, self.D), 123, np.float32),
                   "keys": np.full((B, self.D), 123, np.float32),
                   "queries": np.full((self.count, self.H, self.D), 123, np.float32),
                   "scores": np.full((self.count, B), 123, np.float32)}
        vals = dict(n_embd=self.E, T=self.count, capacity=self.capacity, live=self.live,
                    padded=self.padded, start=self.start, n_rot=self.NROT,
                    base=self.BASE, scale=self.SCALE, eps=self.EPS, device=0)
        vals.update(overrides)
        rc = function(C.byref(self.args if descriptor is None else descriptor),
            pointer(self.arrays["mixed"]), pointer(self.arrays["mask"]), pointer(self.arrays["positions"]),
            *[vals[n] for n in ("n_embd", "T", "capacity", "live", "padded", "start", "n_rot", "base", "scale", "eps", "device")],
            *[pointer(outputs[n]) for n in ("mask", "pooled", "keys", "queries", "scores")])
        return rc, outputs


def metrics(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected)
    idx = tuple(int(i) for i in np.unravel_index(np.argmax(delta), delta.shape))
    return dict(passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                           and np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
                max_abs=float(delta[idx]), rel_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-300)),
                worst_index=idx, actual=float(actual[idx]), expected=float(expected[idx]))


def selection_checks(actual, expected, causal, width):
    rows = []
    for q in range(len(actual)):
        eligible = np.flatnonzero(np.isfinite(causal[q]))
        chosen = np.flatnonzero(np.isfinite(actual[q]))
        take = min(width, len(eligible))
        cutoff = np.sort(expected[q, eligible])[-take] if take else np.inf
        # Only exact mathematical ties may choose different cells. Values within
        # floating error of a distinct cutoff are retained as ambiguous failures.
        must = np.flatnonzero(expected[q] > cutoff)
        allowed = np.flatnonzero(expected[q] >= cutoff)
        passed = (not np.isnan(actual[q]).any() and len(chosen) == take
                  and set(must) <= set(chosen) <= set(allowed)
                  and np.all(actual[q, chosen] == causal[q, chosen])
                  and np.all(np.isneginf(actual[q, ~np.isfinite(causal[q])])))
        rows.append(dict(passed=bool(passed), selected=chosen.tolist(), required=must.tolist(),
                         allowed=allowed.tolist(), cutoff=None if not np.isfinite(cutoff) else float(cutoff),
                         boundary_tied=len(allowed) > take))
    return dict(passed=all(x["passed"] for x in rows), queries=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA", "Metal"), default="CPU")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(started_utc=stamp(), passed=False, scope="synthetic production QSA builder arithmetic only",
                  gate=dict(atol=ATOL, rtol=RTOL, mask="exact cutoff membership/cardinality; unspecified ties"),
                  native_path=str(args.library.resolve()), native_sha256=sha(args.library),
                  script_sha256=sha(__file__), cases=[], abi=[])
    code, handles = 1, []
    try:
        if report["native_sha256"] != args.expected_native_sha256.lower():
            raise ValueError("Native SHA256 mismatch")
        helper_path = Path(__file__).with_name("qwen4exp-mtp-operator.py")
        spec = importlib.util.spec_from_file_location("q4e_operator_helpers", helper_path)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        report["helper_sha256"] = sha(helper_path)
        if os.name == "nt":
            handles.append(os.add_dll_directory(str(args.library.resolve().parent)))
            if os.environ.get("CUDA_PATH"):
                handles.append(os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin")))
        dll = C.CDLL(str(args.library.resolve()))
        helper.initialize_fixture_backend(dll, args.backend, report)
        fn = dll.TSGgml_Qwen4ExpTestQsa
        fn.argtypes = [C.POINTER(Args), P, P, P] + [I] * 7 + [F] * 3 + [I] + [P] * 5
        fn.restype = I
        error = dll.TSGgml_GetLastError; error.restype = C.c_char_p
        # Ordered small/large/small shapes, every tail residue, offset and ranked
        # media, two cache formats. Every graph supplies changed seed/cache/input.
        geometries = [("text", 16, 20, 0, 16, 4), ("text", 17, 20, 16, 1, 4),
            ("text", 18, 24, 15, 3, 4), ("text", 19, 24, 15, 4, 4),
            ("text", 20, 24, 16, 4, 4), ("offset", 19, 28, 14, 5, 4),
            ("media", 23, 28, 16, 7, 4), ("text", 37, 40, 28, 9, 2),
            ("text", 97, 104, 64, 33, 4), ("text", 3, 12, 1, 2, 4)]
        for format_index, dtype in enumerate((np.float32, np.float16)):
            for i, (mode, live, padded, start, count, ratio) in enumerate(geometries):
                name = f"{np.dtype(dtype).name}-{i:02d}-{mode}-L{live}-T{count}-R{ratio}"
                f = Fixture(160916 + i + 100 * format_index, live, padded, start, count, ratio, mode, dtype)
                oracle = f.oracle()
                before = {k: v.copy() for k, v in f.arrays.items()}
                rc, out = f.run(fn)
                item = dict(name=name, return_code=rc,
                            native_error=(error() or b"").decode(errors="replace"), stages={})
                for stage in ("pooled", "keys", "queries", "scores"):
                    item["stages"][stage] = metrics(out[stage], oracle[stage])
                item["mask"] = selection_checks(out["mask"], oracle["expanded"], f.arrays["mask"], f.topk + ratio - 1)
                item["inputs_unchanged"] = all(np.array_equal(v, before[k]) for k, v in f.arrays.items())
                item["wrong_math_controls"] = {k: oracle[k] for k in ("wrong_pool_max", "wrong_relu_max")}
                item["passed"] = bool(rc == 1 and item["inputs_unchanged"] and item["mask"]["passed"]
                                      and all(x["passed"] for x in item["stages"].values()))
                data = {"input_" + k: v for k, v in f.arrays.items()}
                data.update({"actual_" + k: v for k, v in out.items()})
                data.update({"reference_" + k: v for k, v in oracle.items() if isinstance(v, np.ndarray)})
                path = args.output / (name + ".npz"); np.savez_compressed(path, **data)
                item["data"] = dict(path=path.name, sha256=sha(path))
                report["cases"].append(item)
                print(f"{name}: {'PASS' if item['passed'] else 'FAIL'} "
                      f"max_score_error={item['stages']['scores']['max_abs']:.9g}", flush=True)
        f = Fixture(991, 17, 24, 16, 1, 4, "text", np.float32)
        invalid = [("short_k_bytes", "k_bytes", f.args.k_bytes - 1),
                   ("short_q_bytes", "q_bytes", f.args.q_bytes - 1),
                   ("short_cache_bytes", "cache_bytes", f.args.cache_bytes - 1),
                   ("bad_cache_type", "cache_type", 30), ("bad_k_type", "k_type", 1),
                   ("null_cache", "cache", None), ("null_k", "k_proj", None),
                   ("null_q", "q_proj", None), ("null_k_norm", "k_norm", None),
                   ("null_q_norm", "q_norm", None), ("zero_heads", "heads", 0),
                   ("zero_ratio", "ratio", 0), ("large_ratio", "ratio", 65),
                   ("zero_topk", "top_k", 0), ("nonsparse_probe", "top_k", 24)]
        for name, field, value in invalid:
            desc = Args.from_buffer_copy(f.args); setattr(desc, field, value)
            rc, out = f.run(fn, descriptor=desc)
            report["abi"].append(dict(name=name, passed=rc == 0 and all(np.all(v == 123) for v in out.values()),
                                      return_code=rc, error=(error() or b"").decode(errors="replace")))
        for name, change in [("bad_start", {"start": 17}), ("padded_below_live", {"padded": 16}),
                             ("odd_rotary", {"n_rot": 7}), ("zero_epsilon", {"eps": 0}),
                             ("nan_base", {"base": float("nan")}), ("negative_scale", {"scale": -1})]:
            rc, out = f.run(fn, **change)
            report["abi"].append(dict(name=name, passed=rc == 0 and all(np.all(v == 123) for v in out.values()),
                                      return_code=rc, error=(error() or b"").decode(errors="replace")))
        rc, out = f.run(fn)
        report["recovery"] = dict(return_code=rc, score=metrics(out["scores"], f.oracle()["scores"]))
        report["wrong_math_detected"] = all(any(x["wrong_math_controls"][k] > .01 for x in report["cases"])
                                               for k in ("wrong_pool_max", "wrong_relu_max"))
        report["passed"] = (all(x["passed"] for x in report["cases"] + report["abi"])
                            and rc == 1 and report["recovery"]["score"]["passed"] and report["wrong_math_detected"])
        code = 0 if report["passed"] else 1
    except Exception as exc:
        report["exception"] = traceback.format_exc()
        if type(exc).__name__ == "BackendUnavailable": code = 77
    finally:
        report["finished_utc"] = stamp()
        report["exit_code"] = code
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for handle in reversed(handles): handle.close()
        print(f"QSA fixture {'PASS' if report['passed'] else 'FAIL'}; {len(report['cases'])} numeric cases, {len(report['abi'])} ABI cases", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
