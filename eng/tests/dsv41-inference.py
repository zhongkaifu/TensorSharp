#!/usr/bin/env python3
"""Compare native V4.1 inference with the independent PyTorch fixture oracle.

Generate the fixture with eng/dsv41-fixture.py DIRECTORY --f32, then run this
script DIRECTORY --library /absolute/path/to/libGgmlOps.so. CPU checks use
strict tolerances; GPU tolerances can be explicitly selected if necessary.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpu-moe", type=int, default=0, help="Number of initial routed-MoE layers placed on CPU")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--self-atol", type=float, default=None,
                        help="Tolerance for the same-backend self-consistency checks: a rewind served "
                             "from the slot checkpoint against the same rewind served from the live "
                             "rings, and a truncate-to-zero against a plain reset. Defaults to --atol. "
                             "Keep it TIGHT even where --atol is loose - a backend's distance from the "
                             "PyTorch oracle is not a licence for a rewind to change the answer")
    parser.add_argument("--report", type=Path, help="Write a distinct report when comparing native precision modes")
    parser.add_argument("--dump-logits", type=Path,
                        help="Save every native forward's logits, in call order, to this .npz so two native "
                             "libraries can be compared byte for byte on the same fixture and settings")
    parser.add_argument("--long-sparse-tokens", type=int, default=0,
                        help="Also prefill a synthetic prompt of this many tokens, long enough that the owned F32 "
                             "attention sees >= 8,192 keys, once with the default sparse prefill and once with "
                             "TS_DSV41_SPARSE_FA=0 (tiled), and compare the logits and a greedy continuation. "
                             "GPU backends only: the CPU backend has no owned sparse kernel. 0 skips the case")
    parser.add_argument("--long-sparse-context", type=int, default=16384)
    parser.add_argument("--long-sparse-steps", type=int, default=32)
    parser.add_argument("--long-sparse-atol", type=float, default=2e-5)
    args = parser.parse_args()
    if args.long_sparse_tokens and args.backend.upper() == "CPU":
        raise ValueError("--long-sparse-tokens needs a GPU backend: only the owned CUDA attention has the sparse gate")
    if args.long_sparse_tokens and args.long_sparse_tokens + args.long_sparse_steps > args.long_sparse_context:
        raise ValueError("--long-sparse-tokens plus --long-sparse-steps must fit --long-sparse-context")
    config = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not config.get("fixture"):
        raise ValueError("This test is for the small deterministic fixture, not downloaded model weights")
    spec = importlib.util.spec_from_file_location("dsv41_reference", Path(__file__).parents[1] / "dsv41-reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    weights = reference.GgufWeights(args.fixture_dir / "deepseek41-fixture.gguf")
    engram = reference.load_engram(args.fixture_dir / "deepseek41.engram.bin")
    tokens = np.array(json.loads((args.fixture_dir / "tokens.json").read_text()), dtype=np.int32)
    other = np.array([0, 9, 21, 85, 11, 19, 6, 44, 35, 72, 11], dtype=np.int32)
    def oracle(ids):
        model = reference.Reference(weights, config["config"], engram, "model")
        return model.forward(ids.tolist()).cpu().numpy()
    expected, expected_other = oracle(tokens), oracle(other)
    lib = ctypes.CDLL(str(args.library.resolve()))
    signatures = {
        "LoadModel": ([ctypes.c_char_p] + [ctypes.c_int] * 5 + [ctypes.c_char_p], ctypes.c_void_p),
        "Forward": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
        "Reset": ([ctypes.c_void_p], None), "Free": ([ctypes.c_void_p], None),
        "SlotAlloc": ([ctypes.c_void_p], ctypes.c_int),
        "SetActiveSlot": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        "SlotFree": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        "NPast": ([ctypes.c_void_p], ctypes.c_int),
        "Rewind": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        "Truncate": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        "TruncateAlign": ([ctypes.c_void_p], ctypes.c_int),
    }
    api = {}
    for name, (parameters, result) in signatures.items():
        function = getattr(lib, "TSGgml_Dsv4" + name)
        function.argtypes, function.restype = parameters, result
        api[name] = function
    handle = api["LoadModel"](str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                                args.gpus, 256, 32, 2, args.cpu_moe, args.backend.encode())
    if not handle:
        raise RuntimeError("Native fixture model load failed")
    checks = []
    dumped = []
    def forward(ids):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        output = np.empty(config["config"]["text_config"]["vocab_size"], dtype=np.float32)
        result = api["Forward"](handle, ids.ctypes.data, len(ids), output.ctypes.data)
        if result:
            raise RuntimeError(f"Native forward returned {result}")
        if args.dump_logits:
            dumped.append(output.copy())
        return output
    def compare(name, output, target, atol=None, rtol=None):
        atol = args.atol if atol is None else atol
        rtol = args.rtol if rtol is None else rtol
        checks.append(dict(name=name, max_absolute_error=float(np.max(np.abs(output - target))),
                           relative_l2=float(np.linalg.norm(output - target) / np.linalg.norm(target)),
                           argmax=int(output.argmax()), reference_argmax=int(target.argmax()),
                           passed=bool(np.allclose(output, target, atol=atol, rtol=rtol))))
    self_atol = args.atol if args.self_atol is None else args.self_atol
    def cold(ids, split=None):
        """This backend's own logits for a prompt forwarded from an empty cache, optionally
        in the SAME two calls the truncation case makes. Matching the call shape is what
        isolates the rewind: feeding [0,split) and [split,end) separately differs from one
        whole-prompt call by the graph's n_kv bucketing alone, which on CUDA is ~2e-4 and
        has nothing to do with the cache path."""
        api["Reset"](handle)
        if split is None or split == 0:
            return forward(ids)
        forward(ids[:split])
        return forward(ids[split:])
    def compare_self(name, output, baseline):
        """Compare two results produced by the SAME backend, at the tight tolerance. Used
        where the two paths ought to be indistinguishable - the checkpoint route against the
        live route, a truncate-to-zero against a plain reset - so the backend's own distance
        from the PyTorch oracle is not folded into the verdict."""
        compare(name, output, baseline, atol=self_atol, rtol=self_atol)
    try:
        for chunk in (len(tokens), 1, 3, 5):
            api["Reset"](handle)
            for start in range(0, len(tokens), chunk):
                stop = min(start + chunk, len(tokens))
                compare(f"chunk_{chunk}_position_{stop}", forward(tokens[start:stop]), expected[stop - 1])
                assert api["NPast"](handle) == stop
        # Reset must clear Engram token history as well as raw/compressed KV.
        api["Reset"](handle)
        compare("reset_other_prompt", forward(other), expected_other[-1])
        # V4.1's rolling compressor state cannot be restored by moving only
        # n_past. Rejected rewind must preserve both position and continuation.
        api["Reset"](handle)
        forward(tokens[:4])
        assert api["Rewind"](handle, 1) == 0, "V4.1 rewind must be rejected"
        assert api["NPast"](handle) == 4, "Rejected rewind changed position"
        compare("rejected_rewind_continuation", forward(tokens[4:8]), expected[7])
        assert api["NPast"](handle) == 8
        # --- Truncate: partial KV reuse (conversational rewind) --------------
        #
        # Fixture geometry: sliding_window 8, ubatch 32, so ring_raw = 256 and a rewind
        # reaches 256 - 8 + 1 = 249 positions back; compress_ratios {0, 1, 2} make the
        # alignment 2. n_ctx is 256, so a rewind DEEPER than the ring - the one only the
        # prompt-boundary checkpoint can serve - is reachable for targets up to 6.
        align = api["TruncateAlign"](handle)
        assert align == 2, f"V4.1 truncation alignment should be 2, got {align}"
        span, ctx = 249, 256
        filler = np.array([(3 * i + 7) % config["config"]["text_config"]["vocab_size"]
                           for i in range(ctx)], dtype=np.int32)

        def truncate_from_live(head, target):
            """Prefill `head`, rewind to `target` from the LIVE rings, refill."""
            api["Reset"](handle)
            forward(tokens[:head])
            assert api["Truncate"](handle, target) == 1, f"live truncate {head}->{target} refused"
            assert api["NPast"](handle) == target
            return forward(tokens[target:head])

        def truncate_from_checkpoint(head, target):
            """Same rewind, but first decode far enough past `head` that the live rings can
            no longer hold the window - so only the checkpoint taken at the end of the
            `head`-token prefill can serve it. Single-token forwards deliberately do NOT
            move the checkpoint."""
            api["Reset"](handle)
            forward(tokens[:head])                 # multi-token: checkpoints at `head`
            for position in range(head, ctx):
                forward(filler[position:position + 1])
            depth = api["NPast"](handle) - target
            assert depth > span, f"the deep case must exceed the ring span, got {depth}"
            assert api["Truncate"](handle, target) == 1, "a rewind to the prompt boundary was refused"
            assert api["NPast"](handle) == target
            return forward(tokens[target:head])

        # (a) The checkpoint route must be INDISTINGUISHABLE from the live route: same
        # retained prefix, same refill, so the restored rings must be the rings the live
        # case never lost. This is the tight check - it isolates the restore, with no
        # difference in call shape or token provenance to excuse a discrepancy.
        for head, target in ((8, 6), (8, 4), (8, 2), (6, 2)):
            live = truncate_from_live(head, target)
            compare_self(f"truncate_checkpoint_matches_live_{head}_to_{target}",
                         truncate_from_checkpoint(head, target), live)
            # And the answer itself is the oracle's, to the run's ordinary tolerance.
            compare(f"truncate_live_{head}_to_{target}", live, expected[head - 1])

        # (b) Deeper heads through the live route only (n_ctx leaves no room to push a
        # rewind to 8 or beyond past the ring span).
        for head, target in ((16, 8), (16, 4), (12, 10)):
            compare(f"truncate_live_{head}_to_{target}", truncate_from_live(head, target),
                    expected[head - 1])

        # (c) Against this backend's own chunked cold prefill, with a SELF-CALIBRATED
        # tolerance. The retained prefix was computed inside a larger forward here and as
        # its own call there, and those differ by the graph's n_kv bucketing alone - the
        # same effect the chunk_* checks above measure, which on CUDA is far larger than
        # any tolerance worth hard-coding. So the bar is: the rewind must not move the
        # answer further than this backend's own chunk boundary already does.
        for head, target in ((8, 6), (8, 2), (16, 8), (16, 4)):
            chunked, whole = cold(tokens[:head], split=target), cold(tokens[:head])
            control = float(np.max(np.abs(chunked - whole)))
            checks.append(dict(name=f"cold_chunk_spread_{head}_at_{target}", max_absolute_error=control,
                               relative_l2=0.0, argmax=int(chunked.argmax()), reference_argmax=int(whole.argmax()),
                               passed=True))
            bar = max(self_atol, control)
            compare(f"truncate_live_{head}_to_{target}_vs_cold",
                    truncate_from_live(head, target), chunked, atol=bar, rtol=bar)

        # (d) A truncate to zero IS a reset, so the Engram token history must go with it.
        api["Reset"](handle)
        forward(tokens[:12])
        assert api["Truncate"](handle, 0) == 1, "zero is always reachable - it is a reset"
        assert api["NPast"](handle) == 0
        got = forward(other)
        compare("truncate_to_zero_other_prompt", got, expected_other[-1])
        compare_self("truncate_to_zero_other_prompt_vs_cold", got, cold(other))

        # (e) The same depth with no checkpoint in reach must be refused, and refusing must
        # not disturb the sequence.
        api["Reset"](handle)
        for position in range(0, 252):             # every forward is one token: no checkpoint
            forward(filler[position:position + 1])
        head = api["NPast"](handle)
        assert head - 2 > span, f"the no-checkpoint case must exceed the ring span, got {head - 2}"
        assert api["Truncate"](handle, 2) == 0, "a rewind past every reachable state must be refused"
        assert api["NPast"](handle) == head, "a refused truncate moved the head"
        assert api["Truncate"](handle, 0) == 1, "zero is always reachable - it is a reset"
        assert api["NPast"](handle) == 0

        # (f) A misaligned target must be refused rather than straddling a compression
        # block, and must leave the sequence usable.
        api["Reset"](handle)
        forward(tokens[:8])
        assert api["Truncate"](handle, 5) == 0, "an odd truncation target must be refused"
        assert api["NPast"](handle) == 8
        got = forward(tokens[8:12])
        compare("refused_alignment_continuation", got, expected[11])
        compare_self("refused_alignment_continuation_vs_cold", got, cold(tokens[:12], split=8))

        # (g) Out of range, and the no-op.
        api["Reset"](handle)
        forward(tokens[:8])
        assert api["Truncate"](handle, 9) == 0, "a truncate past the head must be refused"
        assert api["Truncate"](handle, -1) == 0
        assert api["Truncate"](handle, 8) == 1, "truncating to the head is a no-op, not a refusal"
        assert api["NPast"](handle) == 8
        got = forward(tokens[8:12])
        compare("truncate_noop_continuation", got, expected[11])
        compare_self("truncate_noop_continuation_vs_cold", got, cold(tokens[:12], split=8))

        api["Reset"](handle)
        slot = api["SlotAlloc"](handle)
        assert slot > 0
        positions = [0, 0]
        while positions[0] < len(tokens) or positions[1] < len(other):
            for stream, (slot_id, prompt, target, chunk) in enumerate(((0, tokens, expected, 3), (slot, other, expected_other, 2))):
                start = positions[stream]
                if start == len(prompt):
                    continue
                assert api["SetActiveSlot"](handle, slot_id) == 0
                stop = min(start + chunk, len(prompt))
                compare(f"interleaved_slot_{stream}_position_{stop}", forward(prompt[start:stop]), target[stop - 1])
                assert api["NPast"](handle) == stop
                positions[stream] = stop
        assert api["SetActiveSlot"](handle, 0) == 0
        assert api["SlotFree"](handle, slot) == 0
    finally:
        api["Free"](handle)
    environment = {name: os.environ.get(name) for name in ("TS_DSV4_FA", "TS_DSV4_GATHER", "TS_DSV41_TP", "TS_DSV41_SPARSE_FA",
                  "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_WARM", "NVIDIA_TF32_OVERRIDE")}
    long_sparse = long_sparse_case(args, config, api, checks, dumped) if args.long_sparse_tokens else None
    if args.dump_logits:
        np.savez(args.dump_logits, logits=np.stack(dumped))
    result = dict(backend=args.backend, gpus=args.gpus, cpu_moe=args.cpu_moe, atol=args.atol, rtol=args.rtol,
                  environment=environment, checks=checks)
    if long_sparse:
        result["long_sparse"] = long_sparse
    path = args.report or args.fixture_dir / f"validation-{args.backend.lower()}-{args.gpus}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    passed = sum(check["passed"] for check in checks)
    print(f"Passed {passed}/{len(checks)} native/reference checks; {path}")
    if passed != len(checks):
        raise SystemExit(1)


def long_sparse_case(args, config, api, checks, dumped):
    """Default sparse prefill against its TS_DSV41_SPARSE_FA=0 escape hatch at >= 8,192 keys.

    The owned CUDA attention compacts each prefill query's mask to its sliding window plus
    the indexer's selection once a launch has more than 8 queries and at least 8,192 keys;
    below that, and with =0, it runs the tiled dense kernel. Both are F32 and must agree to
    the fixture tolerance, and a greedy continuation (decode stays dense in both) must pick
    the same tokens. Each arm is a fresh load, so no graph built under one setting is reused
    by the other; the native loader reads the variable when it builds a graph.
    """
    text = config["config"]["text_config"]
    vocabulary, window = text["vocab_size"], text["sliding_window"]
    context, count, steps = args.long_sparse_context, args.long_sparse_tokens, args.long_sparse_steps
    ubatch = 32
    ring = (window + ubatch + 255) // 256 * 256
    ratios = sorted({r for r in text["compress_ratios"] if r > 0})
    # build_comp_plan: a power-of-two bucket (min 256) of the call's final row count, capped
    # at 8,192 positions of hint; K is the raw ring plus that bucket.
    def keys_at(ratio):
        needed = max(min(count, 8192) // ratio, (count + 1) // ratio, 1)
        bucket = 256
        while bucket < needed:
            bucket <<= 1
        return ring + min(bucket, (context // ratio + 1 + 255) // 256 * 256)
    widest = max(keys_at(r) for r in ratios)
    if widest < 8192:
        raise ValueError(f"--long-sparse-tokens {count} reaches only {widest} attention keys; the sparse gate needs 8,192")
    prompt = np.array([(7 * i * i + 13 * i + 5) % vocabulary for i in range(count)], dtype=np.int32)

    def run(setting):
        saved = os.environ.get("TS_DSV41_SPARSE_FA")
        if setting is None:
            os.environ.pop("TS_DSV41_SPARSE_FA", None)
        else:
            os.environ["TS_DSV41_SPARSE_FA"] = setting
        try:
            handle = api["LoadModel"](str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                                      args.gpus, context, ubatch, 2, args.cpu_moe, args.backend.encode())
            if not handle:
                raise RuntimeError("Native fixture model load failed for the long sparse case")
            try:
                def step(ids):
                    ids = np.ascontiguousarray(ids, dtype=np.int32)
                    output = np.empty(vocabulary, dtype=np.float32)
                    result = api["Forward"](handle, ids.ctypes.data, len(ids), output.ctypes.data)
                    if result:
                        raise RuntimeError(f"Native forward returned {result}")
                    return output
                rows, tokens = [step(prompt)], []
                for _ in range(steps):
                    tokens.append(int(rows[-1].argmax()))
                    rows.append(step(np.array([tokens[-1]], dtype=np.int32)))
                assert api["NPast"](handle) == count + steps
                return rows, tokens
            finally:
                api["Free"](handle)
        finally:
            if saved is None:
                os.environ.pop("TS_DSV41_SPARSE_FA", None)
            else:
                os.environ["TS_DSV41_SPARSE_FA"] = saved

    sparse_rows, sparse_tokens = run(None)
    tiled_rows, tiled_tokens = run("0")
    if args.dump_logits:
        dumped.extend(sparse_rows + tiled_rows)
    worst = 0.0
    for index, (sparse, tiled) in enumerate(zip(sparse_rows, tiled_rows)):
        name = "long_sparse_prefill_logits" if index == 0 else f"long_sparse_decode_{index}_logits"
        error = float(np.max(np.abs(sparse - tiled)))
        worst = max(worst, error)
        checks.append(dict(name=name, max_absolute_error=error,
                           relative_l2=float(np.linalg.norm(sparse - tiled) / max(np.linalg.norm(tiled), 1e-30)),
                           argmax=int(sparse.argmax()), reference_argmax=int(tiled.argmax()),
                           passed=bool(np.allclose(sparse, tiled, atol=args.long_sparse_atol, rtol=0))))
    checks.append(dict(name=f"long_sparse_greedy_{steps}_tokens", max_absolute_error=0.0, relative_l2=0.0,
                       argmax=sparse_tokens[-1], reference_argmax=tiled_tokens[-1],
                       passed=sparse_tokens == tiled_tokens))
    identical = all(np.array_equal(a, b) for a, b in zip(sparse_rows, tiled_rows))
    print(f"long sparse case: {count} tokens, context {context}, widest attention {widest} keys; "
          f"max |default - TS_DSV41_SPARSE_FA=0| = {worst:.3g} over {len(sparse_rows)} logit rows; "
          f"greedy {'identical' if sparse_tokens == tiled_tokens else 'DIFFERENT'}; "
          f"bit-identical logits: {identical}")
    return dict(tokens=count, context=context, ubatch=ubatch, steps=steps, widest_keys=widest,
                atol=args.long_sparse_atol, max_absolute_error=worst, bit_identical=identical,
                default_tokens=sparse_tokens, tiled_tokens=tiled_tokens)


if __name__ == "__main__":
    main()
