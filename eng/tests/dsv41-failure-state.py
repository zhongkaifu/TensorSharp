#!/usr/bin/env python3
"""Exercise V4.1 slot failure recovery with native test-build fault injection.

Requires a native library built with its tests enabled (TSG_GGML_TEST_HOOKS).
The hooks interrupt a real CPU fixture forward after Engram history mutation
or after graph execution; a production build intentionally has no such hooks.
"""
import argparse
import ctypes
import importlib.util
import json
import os
import hashlib
from pathlib import Path

import numpy as np
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--vision-fixture", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--tp", type=int, default=0)
    args = parser.parse_args()
    if args.tp and (args.backend != "CUDA" or args.tp != args.gpus or args.tp < 2):
        parser.error("Whole-model TP failure checks require CUDA and --tp matching --gpus >= 2")
    reference = module("reference", Path(__file__).parents[1] / "dsv41-reference.py")
    helper = module("vision_test", Path(__file__).with_name("dsv41-vision.py"))
    config = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not config.get("fixture"):
        raise ValueError("Failure injection tests only accept the synthetic fixture")
    torch.set_num_threads(2)
    os.environ["TS_DSV4_FA"] = "0"
    os.environ["TS_DSV4_GATHER"] = "0"
    os.environ["TS_DSV41_TP"] = str(args.tp)
    weights = reference.GgufWeights(args.fixture_dir / "deepseek41-fixture.gguf")
    visual_weights = reference.GgufWeights(args.vision_fixture / "deepseek41.vision.gguf")
    layout = reference.load_engram(args.fixture_dir / "deepseek41.engram.bin")
    visual = json.loads((args.vision_fixture / "manifest.json").read_text())
    image_id = visual["config"]["image_token_id"]
    dim = config["config"]["text_config"]["hidden_size"]
    vocab = config["config"]["text_config"]["vocab_size"]
    span = np.fromfile(args.vision_fixture / "grid_4x5.span.f32", dtype=np.float32).reshape(-1, dim)
    text_ids = np.array(json.loads((args.fixture_dir / "tokens.json").read_text()), dtype=np.int32)
    image_ids = np.array([0, 15] + [image_id] * len(span) + [32, 64, 128, 13, 254, 18], dtype=np.int32)
    image_mask = np.array([0, 0] + [1] * len(span) + [0] * 6, dtype=np.uint8)
    streams = [(text_ids, np.zeros(len(text_ids), dtype=np.uint8), np.empty((0, dim), dtype=np.float32)),
               (image_ids, image_mask, span)]
    targets = []
    for ids, mask, rows in streams:
        oracle = reference.Reference(weights, config["config"], layout, "model")
        targets.append(oracle.forward(ids.tolist(), rows, mask, visual_weights)[-1].cpu().numpy())
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr, number = ctypes.c_void_p, ctypes.c_int
    bind = lambda name, args, result=number: helper.bind(lib, name, args, result)
    load = bind("Dsv4LoadModel", [ctypes.c_char_p] + [number] * 5 + [ctypes.c_char_p], ptr)
    free = bind("Dsv4Free", [ptr], None)
    reset = bind("Dsv4Reset", [ptr], None)
    past = bind("Dsv4NPast", [ptr])
    rewind = bind("Dsv4Rewind", [ptr, number])
    slot_alloc = bind("Dsv4SlotAlloc", [ptr])
    slot_select = bind("Dsv4SetActiveSlot", [ptr, number])
    slot_free = bind("Dsv4SlotFree", [ptr, number])
    plain = bind("Dsv4Forward", [ptr, ptr, number, ptr])
    speculative = bind("Dsv4ForwardSpec", [ptr, ptr, number, ptr])
    visual_forward = bind("Dsv41ForwardVision", [ptr, ptr, ptr, ptr, number, number, ptr])
    vision_load = bind("Dsv41VisionLoad", [ctypes.c_char_p, ctypes.c_char_p, number, number], ptr)
    vision_free = bind("Dsv41VisionFree", [ptr], None)
    attach = bind("Dsv41AttachVision", [ptr, ptr])
    handle = load(str(args.fixture_dir / "deepseek41-fixture.gguf").encode(), args.gpus, 256, 3, 2, 0, args.backend.encode())
    if not handle:
        raise RuntimeError("Text fixture load failed")
    encoder = vision_load(str(args.vision_fixture / "deepseek41.vision.gguf").encode(), args.backend.encode(), 0, 2)
    checks = []
    def check(name, passed, **extra):
        checks.append(dict(name=name, passed=bool(passed), **extra))
        if not passed:
            raise AssertionError(name)
    def call(stream, vision):
        ids, mask, rows = stream
        logits = np.empty(vocab, dtype=np.float32)
        status = (visual_forward(handle, ids.ctypes.data, mask.ctypes.data, rows.ctypes.data,
                                 len(ids), int(mask.sum()), logits.ctypes.data) if vision else
                  plain(handle, ids.ctypes.data, len(ids), logits.ctypes.data))
        return status, logits
    def slice_stream(stream, start, stop):
        ids, mask, rows = stream
        return ids[start:stop], mask[start:stop], rows[int(mask[:start].sum()):int(mask[:stop].sum())]
    def compare(name, actual, expected):
        check(name, np.allclose(actual, expected, atol=2e-5, rtol=2e-5),
              max_absolute_error=float(np.max(np.abs(actual - expected))),
              relative_l2=float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
    try:
        check("attach", encoder and attach(handle, encoder) == 0)
        vision_free(encoder)
        encoder = None
        healthy = slot_alloc(handle)
        check("second_slot", healthy > 0)
        # This is the model's first native forward: the graph cannot already
        # be cached. Interrupt construction before allocation, then request
        # exactly the same shape after Reset. A half-built cached entry would
        # crash or return invalid logits on that retry.
        os.environ["TS_DSV41_TEST_FAIL_STAGE"] = "graph"
        os.environ["TS_DSV41_TEST_FAIL_POSITION"] = "0"
        check("partial_graph_failure", call(streams[1], True)[0] < 0)
        check("partial_graph_position", past(handle) == 0)
        os.environ.pop("TS_DSV41_TEST_FAIL_STAGE")
        os.environ.pop("TS_DSV41_TEST_FAIL_POSITION")
        check("partial_graph_sticky", call(streams[1], True)[0] < 0)
        reset(handle)
        status, output = call(streams[1], True)
        check("partial_graph_same_shape_recovery", status == 0)
        compare("partial_graph_recovery_oracle", output, targets[1])
        for vision in (False, True):
            stream, target = streams[int(vision)], targets[int(vision)]
            stages = ("engram", "compute", "tp-rank") if args.tp else ("engram", "compute")
            faults = [(stage, position, "runtime_error") for stage in stages for position in (0, 3)]
            # Checkpointing runs after every microbatch has advanced n_past.
            # Both exception classes must be contained there too, and resetting
            # must restore the same independent model result afterward.
            faults += [("checkpoint", len(stream[0]), kind) for kind in ("bad_alloc", "unknown")]
            for stage, position, kind in faults:
                name = f"{'vision' if vision else 'text'}_{stage}_{kind}_position{position}"
                check(name + "_prepare_other", slot_select(handle, healthy) == 0)
                reset(handle)
                status, _ = call(slice_stream(stream, 0, 1), vision)
                check(name + "_other_prefix", status == 0 and past(handle) == 1)
                check("select_failed_slot", slot_select(handle, 0) == 0)
                reset(handle)
                os.environ["TS_DSV41_TEST_FAIL_STAGE"] = stage
                os.environ["TS_DSV41_TEST_FAIL_POSITION"] = str(position)
                os.environ["TS_DSV41_TEST_FAIL_KIND"] = kind
                status, _ = call(stream, vision)
                check(name + "_injected_failure", status < 0,
                      note="If this fails, ensure the native library was built with test hooks enabled")
                check(name + "_committed_prefix", past(handle) == position)
                os.environ.pop("TS_DSV41_TEST_FAIL_STAGE")
                os.environ.pop("TS_DSV41_TEST_FAIL_POSITION")
                os.environ.pop("TS_DSV41_TEST_FAIL_KIND")
                # Both APIs must keep rejecting after the transient hook
                # disappears; moving only the logical position cannot heal
                # a partially written compressed/raw cache.
                for use_visual in (False, True):
                    status, _ = call(streams[int(use_visual)], use_visual)
                    check(name + f"_sticky_{use_visual}", status < 0 and past(handle) == position)
                check(name + "_rewind_rejected", rewind(handle, 0) == 0 and past(handle) == position)
                scratch = np.empty(vocab, dtype=np.float32)
                check(name + "_speculative_rejected", speculative(handle, text_ids.ctypes.data, 1,
                      scratch.ctypes.data) == 0 and past(handle) == position)
                check(name + "_still_failed_after_rewind", call(stream, vision)[0] < 0)
                check(name + "_select_other", slot_select(handle, healthy) == 0)
                check(name + "_other_prefix_preserved", past(handle) == 1)
                status, output = call(slice_stream(stream, 1, len(stream[0])), vision)
                check(name + "_other_slot_healthy", status == 0)
                compare(name + "_other_slot_oracle", output, target)
                check(name + "_select_original", slot_select(handle, 0) == 0)
                reset(handle)
                check(name + "_reset_position", past(handle) == 0)
                status, output = call(stream, vision)
                check(name + "_reset_recovered", status == 0)
                compare(name + "_reset_oracle", output, target)
        # Input validation errors must not poison an otherwise healthy slot.
        reset(handle)
        bad_ids = np.array([vocab], dtype=np.int32)
        output = np.empty(vocab, dtype=np.float32)
        check("healthy_speculative_rejected", speculative(handle, text_ids.ctypes.data, 1,
              output.ctypes.data) == 0 and past(handle) == 0)
        check("invalid_text_token_retryable", plain(handle, bad_ids.ctypes.data, 1, output.ctypes.data) < 0 and past(handle) == 0)
        for mask, count in ((np.array([2], dtype=np.uint8), 1), (np.array([1], dtype=np.uint8), 0)):
            ids = np.array([image_id], dtype=np.int32)
            check("invalid_visual_input_retryable", visual_forward(handle, ids.ctypes.data, mask.ctypes.data,
                  span.ctypes.data, 1, count, output.ctypes.data) < 0 and past(handle) == 0)
        status, output = call(streams[0], False)
        check("valid_after_invalid_input", status == 0)
        compare("validation_errors_preserve_fresh_state", output, targets[0])
        check("free_other_slot", slot_free(handle, healthy) == 0)
    finally:
        for key in ("TS_DSV41_TEST_FAIL_STAGE", "TS_DSV41_TEST_FAIL_POSITION", "TS_DSV41_TEST_FAIL_KIND"):
            os.environ.pop(key, None)
        if encoder:
            vision_free(encoder)
        free(handle)
        with args.library.open("rb") as library:
            hasher = hashlib.sha256()
            for block in iter(lambda: library.read(1048576), b""):
                hasher.update(block)
            library_sha256 = hasher.hexdigest()
        args.report.write_text(json.dumps(dict(backend=args.backend, gpus=args.gpus, tensor_parallel=args.tp,
            library_sha256=library_sha256, test_hooks_required=True, checks=checks), indent=2) + "\n")
    print(f"Passed {len(checks)}/{len(checks)} failure-state checks; {args.report}")


if __name__ == "__main__":
    main()
