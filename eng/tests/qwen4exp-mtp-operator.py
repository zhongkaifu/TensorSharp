#!/usr/bin/env python3
"""Independent F64 oracle for the actual Qwen4Exp MTP C ABI on CPU or CUDA.

Uses the pinned actual-head EH/HC *subspace* sample plus synthetic nonzero
attention, routed/shared expert and output weights. This is not full-model
MTP accuracy, target GDN/PLE rollback, or speculative engagement evidence.
The predeclared F32 fixture gate is elementwise atol=rtol=2e-5, unchanged
across every shape/replay. All oracle arithmetic after F32 inputs is F64.
"""
import argparse
import ctypes as C
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np


P, I, L, F = C.c_void_p, C.c_int, C.c_int64, C.c_float
ATOL = RTOL = 2e-5


def fields(pointers, sizes, types):
    return ([(x, P) for x in pointers.split()] + [(x + "_bytes", L) for x in sizes.split()]
            + [(x + "_type", I) for x in types.split()])


class Config(C.Structure):
    _fields_ = [(x, P) for x in ("enorm", "hnorm", "eh_proj")] + [
        ("eh_bytes", L), ("eh_type", I)] + [(x, I) for x in (
        "n_embd hc hc_low_rank head_dim n_head n_head_kv n_rot n_expert "
        "n_expert_used n_ff n_ff_sh capacity device").split()] + [
        (x, F) for x in ("eps", "rope_base", "rope_scale", "attn_scale")] + [("rope_sections", I * 4)]


class Attn(C.Structure):
    _fields_ = fields("hc_norm hc_down hc_up hc_inject wq wk wv wo q_norm k_norm k_cache v_cache",
                      "hc_down hc_up hc_inject wq wk wv wo kv",
                      "hc_down hc_up hc_inject wq wk wv wo kv")


class Ffn(C.Structure):
    _fields_ = fields("hc_norm hc_down hc_up hc_inject router gate_exps up_exps down_exps "
                      "sh_gate_inp sh_gate sh_up sh_down",
                      "hc_down hc_up hc_inject router gate_exps up_exps down_exps sh_gate sh_up sh_down",
                      "hc_down hc_up hc_inject router gate_exps up_exps down_exps sh_gate sh_up sh_down")


class Head(C.Structure):
    _fields_ = fields("hc_norm hc_down hc_up head", "hc_down hc_up head", "hc_down hc_up head") + [("vocab", I)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class BackendUnavailable(RuntimeError):
    pass


def initialize_fixture_backend(dll, requested, report):
    """Select the strict singleton backend, then verify rank0 and describe it.

    The owned executor allocates and computes on active_backend directly; it has
    no CPU scheduler fallback. IsBackendAvailable rejects a different backend.
    """
    backend_type = {"Metal": 1, "CPU": 2, "CUDA": 3}[requested]
    init = dll.TSGgml_IsBackendAvailable; init.argtypes = [I]; init.restype = I
    error = dll.TSGgml_GetLastError; error.argtypes = []; error.restype = C.c_char_p
    report["backend"] = dict(requested=requested, native_type=backend_type)
    if init(backend_type) != 1:
        reason = (error() or b"Backend initialization failed").decode(errors="replace")
        if requested in ("CUDA", "Metal"): raise BackendUnavailable(reason)
        raise RuntimeError(reason)
    select = dll.TSGgml_SetActiveDevice; select.argtypes = [I]; select.restype = I
    current = dll.TSGgml_GetActiveDevice; current.argtypes = []; current.restype = I
    if select(0) != 1 or current() != 0:
        raise RuntimeError("Fixture did not select actual active rank0")
    description = dll.TSGgml_GetGpuDeviceDescription
    description.argtypes = [I, I, P, I]; description.restype = I
    buffer = C.create_string_buffer(1024)
    if description(backend_type, 0, buffer, len(buffer)) != 1 or not buffer.value:
        raise RuntimeError("Cannot describe the initialized fixture backend device")
    report["backend"].update(initialized=True, active_rank=current(),
        device_description=buffer.value.decode(errors="replace"), execution="single active backend; no scheduler fallback")
    return True


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rms(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def softmax(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / np.sum(y, axis=-1, keepdims=True)


def rotary(x, positions, n_rot, base, scale):
    out = x.copy()
    # NEOX partial rotation: first half paired with second half, leaving the
    # dimensions beyond n_rot unchanged. All heads share this text position.
    angles = positions[:, None] * scale * base ** (-2 * np.arange(n_rot // 2, dtype=np.float64) / n_rot)
    cs, sn = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    a, b = x[..., :n_rot // 2], x[..., n_rot // 2:n_rot]
    out[..., :n_rot // 2] = a * cs - b * sn
    out[..., n_rot // 2:n_rot] = a * sn + b * cs
    return out


class Fixture:
    H, HC, LOW, HD, NH, NK, FF, SH, EXP, USED, VOCAB = 8, 4, 3, 8, 4, 2, 6, 5, 4, 2, 13

    def __init__(self, sample, seed=701, capacity=512, kv_alignment=4, attention_head_dim=8, attention_heads=4):
        if attention_head_dim not in (8, 64, 256):
            raise ValueError("Unsupported synthetic attention head dimension")
        self.HD = attention_head_dim
        if attention_heads not in (4, 24):
            raise ValueError("Unsupported synthetic attention head count")
        self.NH = attention_heads
        self.arrays = {}
        self.rng = np.random.default_rng(seed)
        self.eps = float(np.float32(sample["epsilon"]))
        self.capacity = capacity
        self.c, self.a, self.f, self.o = Config(), Attn(), Ffn(), Head()
        c = self.c
        for field, value in dict(n_embd=self.H, hc=self.HC, hc_low_rank=self.LOW,
                                 head_dim=self.HD, n_head=self.NH, n_head_kv=self.NK, n_rot=4,
                                 n_expert=self.EXP, n_expert_used=self.USED, n_ff=self.FF,
                                 n_ff_sh=self.SH, capacity=capacity, device=0).items():
            setattr(c, field, value)
        c.eps, c.rope_base, c.rope_scale, c.attn_scale = self.eps, 10000, 0.75, 1 / math.sqrt(self.HD)
        c.rope_sections[:] = (2, 1, 1, 0)
        a = sample["arrays"]
        self.put(c, "enorm", "enorm", np.array(a["embedding_norm"]))
        self.put(c, "hnorm", "hnorm", np.array(a["hidden_norm"]).reshape(self.HC, self.H))
        self.put(c, "eh_proj", "eh", np.array(a["eh"]).reshape(self.H, 2 * self.H), sizes=False)
        c.eh_bytes = self.arrays["eh"].nbytes
        for desc, prefix in ((self.a, "attn"), (self.f, "ffn"), (self.o, "head")):
            self.put(desc, "hc_norm", prefix + ".norm", np.array(a["head_norm"]).reshape(self.HC, self.H))
            self.put(desc, "hc_down", prefix + ".down", np.array(a["head_down"]).reshape(self.LOW, self.H * self.HC))
            self.put(desc, "hc_up", prefix + ".up", np.array(a["head_up"]).reshape(self.H * self.HC, self.LOW))
            if prefix != "head":
                self.random(desc, "hc_inject", prefix + ".inject", (self.HC, self.H * self.HC), .2)
        self.random(self.a, "wq", "wq", (2 * self.NH * self.HD, self.H), .3)
        self.random(self.a, "wk", "wk", (self.NK * self.HD, self.H), .3)
        self.random(self.a, "wv", "wv", (self.NK * self.HD, self.H), .4)
        self.random(self.a, "wo", "wo", (self.H, self.NH * self.HD), .3)
        self.put(self.a, "q_norm", "q_norm", np.linspace(.7, 1.3, self.HD))
        self.put(self.a, "k_norm", "k_norm", np.linspace(1.2, .8, self.HD))
        for name in ("k_cache", "v_cache"):
            # Borrowed float buffers need only natural float alignment at the
            # C ABI. Exercise both host-wrap and fallback paths deterministically.
            size = self.NK * capacity * self.HD
            raw = np.zeros(size + 32, dtype=np.float32)
            first = (-raw.ctypes.data % 64) // 4 + (1 if kv_alignment == 4 else 0)
            cache = raw[first:first + size].reshape(self.NK, capacity, self.HD)
            assert cache.ctypes.data % 64 == (4 if kv_alignment == 4 else 0)
            self.put(self.a, name, name, cache)
        self.a.kv_bytes = self.arrays["k_cache"].nbytes
        self.random(self.f, "router", "router", (self.EXP, self.H), .5)
        self.random(self.f, "gate_exps", "gate_exps", (self.EXP, self.FF, self.H), .6)
        self.random(self.f, "up_exps", "up_exps", (self.EXP, self.FF, self.H), .6)
        self.random(self.f, "down_exps", "down_exps", (self.EXP, self.H, self.FF), .6)
        self.random(self.f, "sh_gate_inp", "sh_gate_inp", (self.H,), .4)
        self.random(self.f, "sh_gate", "sh_gate", (self.SH, self.H), .5)
        self.random(self.f, "sh_up", "sh_up", (self.SH, self.H), .5)
        self.random(self.f, "sh_down", "sh_down", (self.H, self.SH), .5)
        self.random(self.o, "head", "head", (self.VOCAB, self.H), .7)
        self.o.vocab = self.VOCAB
        self.w = {k: v.astype(np.float64) for k, v in self.arrays.items()}
        self.reference_k = np.zeros((capacity, self.NK, self.HD), np.float64)
        self.reference_v = self.reference_k.copy()
        self.last_stages = {}

    def put(self, desc, field, name, value, sizes=True):
        value = np.ascontiguousarray(value, dtype=np.float32)
        self.arrays[name] = value  # owns every pointer through native Free
        setattr(desc, field, value.ctypes.data)
        if sizes and hasattr(desc, field + "_bytes"):
            setattr(desc, field + "_bytes", value.nbytes)

    def random(self, desc, field, name, shape, scale):
        self.put(desc, field, name, self.rng.uniform(-scale, scale, shape))

    def inputs(self, count, seed):
        r = np.random.default_rng(seed)
        e = np.ascontiguousarray(r.normal(0, .7, (count, self.H)), np.float32)
        h = r.normal(0, 1, (count, self.HC, self.H))
        h *= np.array([.07, .5, 2.0, 7.0])[None, :, None]
        h += np.array([-.4, .9, -.2, 1.8])[None, :, None]
        return e, np.ascontiguousarray(h, np.float32)

    def mix(self, residual, prefix):
        xn = rms(residual, self.eps) * self.w[prefix + ".norm"]
        flat = xn.reshape(len(xn), -1)
        lo = silu(flat @ self.w[prefix + ".down"].T / self.HC)
        gate = sigmoid(lo @ self.w[prefix + ".up"].T).reshape(xn.shape)
        mixed = (xn * gate).mean(axis=1)
        scatter = None if prefix == "head" else 2 * sigmoid(flat @ self.w[prefix + ".inject"].T / self.HC)
        return mixed, scatter

    def eh(self, embedding, previous, wrong=None):
        en = rms(embedding, self.eps) * self.w["enorm"]
        if wrong == "global-rms":
            hn = rms(previous.reshape(len(previous), -1), self.eps).reshape(previous.shape) * self.w["hnorm"]
        else:
            hn = rms(previous, self.eps) * self.w["hnorm"]
        en = np.repeat(en[:, None, :], self.HC, axis=1)
        joined = np.concatenate((hn, en) if wrong == "hidden-first" else (en, hn), axis=2)
        out = joined @ self.w["eh"].T
        if wrong == "collapsed":
            out = np.repeat(out.mean(axis=1, keepdims=True), self.HC, axis=1)
        return out

    def oracle(self, embedding, previous, position, rope_position):
        e, h = embedding.astype(np.float64), previous.astype(np.float64)
        T, w = len(e), self.w
        residual = self.eh(e, h)
        self.last_stages = {"eh": residual.copy()}
        mixed, scatter = self.mix(residual, "attn")
        qg = (mixed @ w["wq"].T).reshape(T, self.NH, 2, self.HD)
        q = rms(qg[:, :, 0], self.eps) * w["q_norm"]
        k = rms((mixed @ w["wk"].T).reshape(T, self.NK, self.HD), self.eps) * w["k_norm"]
        v = (mixed @ w["wv"].T).reshape(T, self.NK, self.HD)
        pos = np.arange(T, dtype=np.float64) + rope_position
        q = rotary(q, pos, self.c.n_rot, self.c.rope_base, self.c.rope_scale)
        k = rotary(k, pos, self.c.n_rot, self.c.rope_base, self.c.rope_scale)
        if position == 0:
            self.reference_k.fill(0); self.reference_v.fill(0)
        self.reference_k[position:position + T] = k
        self.reference_v[position:position + T] = v
        attended = np.zeros((T, self.NH, self.HD))
        for token in range(T):
            for head in range(self.NH):
                kh = head // (self.NH // self.NK)
                keys = self.reference_k[:position + token + 1, kh]
                values = self.reference_v[:position + token + 1, kh]
                prob = softmax(keys @ q[token, head] * self.c.attn_scale)
                attended[token, head] = prob @ values
        projected = (attended * sigmoid(qg[:, :, 1])).reshape(T, -1) @ w["wo"].T
        delta = projected[:, None, :] * scatter[:, :, None]
        self.last_stages["attention_delta"] = delta
        residual = residual + delta
        mixed, scatter = self.mix(residual, "ffn")
        router = mixed @ w["router"].T
        selected = np.argsort(-router, axis=1, kind="stable")[:, :self.USED]
        probs = softmax(np.take_along_axis(router, selected, axis=1))
        moe = np.zeros((T, self.H))
        for token in range(T):
            for j, expert in enumerate(selected[token]):
                gate = w["gate_exps"][expert] @ mixed[token]
                up = w["up_exps"][expert] @ mixed[token]
                moe[token] += probs[token, j] * (w["down_exps"][expert] @ (silu(gate) * up))
        shared = (silu(mixed @ w["sh_gate"].T) * (mixed @ w["sh_up"].T)) @ w["sh_down"].T
        shared *= sigmoid(mixed @ w["sh_gate_inp"])[:, None]
        self.last_stages["routed_delta"] = moe[:, None, :] * scatter[:, :, None]
        self.last_stages["shared_delta"] = shared[:, None, :] * scatter[:, :, None]
        residual = residual + (moe + shared)[:, None, :] * scatter[:, :, None]
        final, _ = self.mix(residual, "head")
        return residual, final @ w["head"].T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA", "Metal"), default="CPU")
    parser.add_argument("--capacity", type=int, choices=(32, 512), default=512,
                        help="32 covers the shared binder's <4096-byte cache boundary; 512 covers larger buffers")
    parser.add_argument("--kv-alignment", type=int, choices=(4, 64), default=4,
                        help="Guaranteed pointer offset4 modulo64, or64-byte-aligned buffers")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Use a fresh report path; historical failures must remain intact")
    report = dict(schema_version=1, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope=__doc__, atol=ATOL, rtol=RTOL, checks=[], cases=[])
    report.update(source_sha256=sha(__file__), native_sha256=sha(args.library), sample_sha256=sha(args.sample))
    sample = json.loads(args.sample.read_text())
    assert sample["hidden"] == 8 and sample["streams"] == 4 and sample["rank"] == 3
    report["sample_provenance"] = {k: v for k, v in sample.items() if k != "arrays"}
    fixture = Fixture(sample, capacity=args.capacity, kv_alignment=args.kv_alignment)
    report["cache_capacity"] = args.capacity
    report["kv_pointer_mod64"] = {k: v.ctypes.data % 64 for k, v in fixture.arrays.items() if k in ("k_cache", "v_cache")}
    report["weights"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest())
                         for k, v in fixture.arrays.items()}
    report["abi_sizes"] = {x.__name__: C.sizeof(x) for x in (Config, Attn, Ffn, Head)}
    os.environ["TS_GGML_CPU_THREADS"] = "2"
    report["environment"] = {k: os.environ.get(k) for k in ("TS_GGML_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    dirs, handles, dll = [], [], None

    def check(name, passed, **extra):
        report["checks"].append(dict(name=name, passed=bool(passed), **extra))
        if not passed:
            raise AssertionError(name)

    def compare(name, actual, expected):
        diff = actual.astype(np.float64) - expected
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        worst = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        def scalar(value):
            return float(value) if np.isfinite(value) else str(value)
        check(name, finite and np.allclose(actual, expected, atol=ATOL, rtol=RTOL), finite=finite,
              max_abs=scalar(np.max(np.abs(diff))), rel_l2=scalar(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-300)),
              worst_index=[int(x) for x in worst], expected=scalar(expected[worst]), actual=scalar(actual[worst]))

    try:
        if os.name == "nt":
            dirs.append(os.add_dll_directory(str(args.library.resolve().parent)))
            if os.environ.get("CUDA_PATH"):
                dirs.append(os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin")))
        dll = C.CDLL(str(args.library.resolve()))
        init = dll.TSGgml_IsBackendAvailable; init.argtypes = [I]; init.restype = I
        error = dll.TSGgml_GetLastError; error.argtypes = []; error.restype = C.c_char_p
        create = dll.TSGgml_Qwen4ExpMtpCreate
        create.argtypes = [C.POINTER(x) for x in (Config, Attn, Ffn, Head)]; create.restype = P
        forward = dll.TSGgml_Qwen4ExpMtpForward
        forward.argtypes = [P, P, P, I, I, I, P, P, P]; forward.restype = I
        copy_kv = dll.TSGgml_Qwen4ExpMtpCopyKv
        copy_kv.argtypes = [P, P, P, L]; copy_kv.restype = I
        free = dll.TSGgml_Qwen4ExpMtpFree; free.argtypes = [P]; free.restype = None
        check("explicit_backend_initialized", initialize_fixture_backend(dll, args.backend, report))
        check("null_create_rejected", not create(None, None, None, None))
        check("null_forward_rejected", forward(None, None, None, 0, 0, 0, None, None, None) == 0)
        free(None)
        check("null_free_returns", True)
        for field, value in (("n_embd", 0), ("capacity", 0), ("n_head_kv", 0),
                             ("n_expert_used", fixture.EXP + 1), ("device", -1),
                             ("eps", 0.0), ("eps", float("inf")), ("rope_base", 0.0),
                             ("n_rot", fixture.HD + 2), ("eh_bytes", fixture.c.eh_bytes - 4),
                             ("eh_type", 26), ("enorm", None)):
            bad = Config.from_buffer_copy(fixture.c)
            setattr(bad, field, value)
            rejected = create(C.byref(bad), C.byref(fixture.a), C.byref(fixture.f), C.byref(fixture.o))
            if rejected: handles.append(rejected)
            check("invalid_config_" + field, not rejected)
        for desc, field, value in (("attn", "q_norm", None), ("attn", "kv_bytes", fixture.a.kv_bytes - 4),
                                   ("ffn", "router", None), ("ffn", "gate_exps_bytes", fixture.f.gate_exps_bytes - 4),
                                   ("head", "hc_norm", None), ("head", "head_bytes", fixture.o.head_bytes - 4)):
            ad, fd, hd = Attn.from_buffer_copy(fixture.a), Ffn.from_buffer_copy(fixture.f), Head.from_buffer_copy(fixture.o)
            setattr({"attn": ad, "ffn": fd, "head": hd}[desc], field, value)
            rejected = create(C.byref(fixture.c), C.byref(ad), C.byref(fd), C.byref(hd))
            if rejected: handles.append(rejected)
            check("invalid_descriptor_" + desc + "_" + field, not rejected)

        def new():
            handle = create(C.byref(fixture.c), C.byref(fixture.a), C.byref(fixture.f), C.byref(fixture.o))
            check("valid_create", bool(handle), native_error=(error() or b"").decode(errors="replace"))
            handles.append(handle)
            check("valid_create_clears_prior_error", not error())
            return handle

        handle = new()
        outcomes = {}

        def run(name, count, position, seed, rope_position=None, output_head=True):
            nonlocal handle
            if rope_position is None: rope_position = position
            e, h = fixture.inputs(count, seed)
            untouched = (e.copy(), h.copy())
            expected_hidden, expected_logits = fixture.oracle(e, h, position, rope_position)
            out_h = np.full((fixture.HC, fixture.H), np.nan, np.float32)
            out_l = np.full(fixture.VOCAB, np.nan, np.float32)
            rc = forward(handle, e.ctypes.data, h.ctypes.data, count, position, rope_position,
                         None, out_h.ctypes.data, out_l.ctypes.data if output_head else None)
            check(name + "_forward", rc == 1, native_error=(error() or b"").decode(errors="replace"))
            check(name + "_input_immutable", np.array_equal(e, untouched[0]) and np.array_equal(h, untouched[1]))
            compare(name + "_wide_hidden", out_h, expected_hidden[-1])
            if output_head: compare(name + "_logits", out_l, expected_logits[-1])
            for stage in ("attention_delta", "routed_delta", "shared_delta"):
                check(name + "_nonzero_" + stage, np.max(np.abs(fixture.last_stages[stage])) > 1e-6,
                      max_abs=float(np.max(np.abs(fixture.last_stages[stage]))))
            report["cases"].append(dict(name=name, tokens=count, position=position, rope_position=rope_position,
                output_head=output_head, input_sha256=hashlib.sha256(e.tobytes() + h.tobytes()).hexdigest(),
                hidden=out_h.tolist(), logits=out_l.tolist() if output_head else None))
            outcomes[name] = (out_h.copy(), out_l.copy())
            return e, h

        e, h = run("first_token", 1, 0, 801)
        correct_eh = fixture.eh(e.astype(np.float64), h.astype(np.float64))
        for wrong in ("global-rms", "hidden-first", "collapsed"):
            distance = float(np.max(np.abs(correct_eh - fixture.eh(e.astype(np.float64), h.astype(np.float64), wrong))))
            check("fixture_distinguishes_" + wrong, distance > 1e-3, max_abs=distance)
        run("same_shape_changed_inputs", 1, 0, 802)
        run("same_shape_original_again", 1, 0, 801)
        compare("replayed_original_hidden", outcomes["same_shape_original_again"][0], outcomes["first_token"][0])
        run("prefix_five", 5, 0, 803)
        run("append_one", 1, 5, 804)
        run("append_five", 5, 6, 805)
        # Caller-directed rollback overwrites suffix rows. The MTP head has KV
        # only; this does not restore or prove target GDN/PLE snapshot state.
        run("replacement_suffix", 1, 5, 806)
        retained = outcomes["replacement_suffix"]
        free(handle); handles.remove(handle); handle = new()
        run("fresh_prefix_five", 5, 0, 803)
        run("fresh_replacement_suffix", 1, 5, 806)
        compare("rollback_vs_fresh_hidden", outcomes["fresh_replacement_suffix"][0], retained[0])
        compare("rollback_vs_fresh_logits", outcomes["fresh_replacement_suffix"][1], retained[1])

        # A managed cache growth must export the executor's authoritative bytes,
        # preserve each head's row stride, and seed the replacement executor.
        exported = [np.full_like(fixture.arrays[n], -77) for n in ("k_cache", "v_cache")]
        check("kv_export_bad_size_refused", copy_kv(handle, exported[0].ctypes.data,
              exported[1].ctypes.data, exported[0].nbytes - 4) == 0)
        check("kv_export_refusal_mutation_free", all(np.all(x == -77) for x in exported))
        check("kv_export", copy_kv(handle, exported[0].ctypes.data,
              exported[1].ctypes.data, exported[0].nbytes) == 1)
        compare("exported_keys", exported[0], fixture.reference_k.transpose(1, 0, 2))
        compare("exported_values", exported[1], fixture.reference_v.transpose(1, 0, 2))
        old_fixture = fixture
        free(handle); handles.remove(handle)
        fixture = Fixture(sample, capacity=512, kv_alignment=args.kv_alignment)
        for name, values in zip(("k_cache", "v_cache"), exported):
            fixture.arrays[name][:, :old_fixture.capacity] = values
        fixture.reference_k[:old_fixture.capacity] = old_fixture.reference_k
        fixture.reference_v[:old_fixture.capacity] = old_fixture.reference_v
        handle = new()
        report["recreated_cache"] = dict(old_capacity=old_fixture.capacity, new_capacity=fixture.capacity,
                                         seeded_position=6, layout="[kv_heads,capacity,head_dim]")
        run("export_seed_recreated_append", 1, 6, 814)
        run("prefill_sixteen_no_head", 16, 0, 807, rope_position=7, output_head=False)
        run("prefill_sixteen_with_head", 16, 0, 807, rope_position=7)
        compare("head_option_preserves_hidden", outcomes["prefill_sixteen_no_head"][0], outcomes["prefill_sixteen_with_head"][0])
        run("append_changed_k", 1, 16, 808, rope_position=23)
        run("prefill_thirtyone", 31, 0, 809)
        e, h = fixture.inputs(1, 811)
        check("out_of_capacity_refused", forward(handle, e.ctypes.data, h.ctypes.data, 1,
              fixture.capacity, fixture.capacity, None, None, None) == 0)
        check("failed_executor_declines_retry", forward(handle, e.ctypes.data, h.ctypes.data, 1,
              0, 0, None, None, None) == 0)
        free(handle); handles.remove(handle); handle = new()
        run("new_executor_after_refusal", 1, 0, 812)
    except BackendUnavailable as exc:
        report["unavailable"] = str(exc)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if dll is not None:
            for handle in handles:
                try: dll.TSGgml_Qwen4ExpMtpFree(handle)
                except Exception: report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        for directory in reversed(dirs): directory.close()
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["native_sha256_after"] = sha(args.library)
        if report["native_sha256_after"] != report["native_sha256"]:
            report["error"] = "Native library changed during fixture execution"
        report["passed"] = not report.get("unavailable") and not report.get("error") and not report.get("cleanup_errors") and all(x["passed"] for x in report["checks"])
        report["status"] = "unavailable" if report.get("unavailable") else "passed" if report["passed"] else "failed"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{report['status'].upper()}: {len(report['checks'])} checks; {args.report}", flush=True)
        if report.get("error"): print(report["error"], flush=True)
    return 77 if report.get("unavailable") else 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
