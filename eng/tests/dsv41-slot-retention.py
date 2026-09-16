#!/usr/bin/env python3
"""Check V4.1 retained-slot eligibility against real CPU/CUDA fixture continuation.

Requires eng/dsv41-fixture.py --f32 and a test-enabled native library. Queries
must not select or mutate a slot. This validates native admission/continuation,
not the managed scheduler's retention policy or full-model performance.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--gpus", type=int, choices=(1,), default=1)
    parser.add_argument("--cpu-admission-reserve-mb", type=int,
                        help="CPU-only regression: accelerator reserve must not affect explicit CPU byte-budget admission")
    args = parser.parse_args()
    if args.cpu_admission_reserve_mb is not None and (args.backend != "CPU" or
            not 0 <= args.cpu_admission_reserve_mb <= (2**64 - 1) // (1024 * 1024)):
        parser.error("The explicit CPU reserve probe requires CPU and a representable nonnegative MiB value")
    if args.report.exists():
        parser.error("Use a fresh report path to preserve prior evidence")
    config_path = args.fixture_dir / "deepseek41.config.json"
    config = json.loads(config_path.read_text())
    if not config.get("fixture"):
        raise ValueError("This test accepts only the small deterministic fixture")
    cfg = config["config"]["text_config"]
    if cfg["sliding_window"] != 8 or cfg["compress_ratios"] != [0, 2, 2, 1, 1]:
        raise ValueError("The explicit ring/alignment boundary cases require the canonical fixture geometry")
    tokens = np.array(json.loads((args.fixture_dir / "tokens.json").read_text()), dtype=np.int32)
    if len(tokens) < 16:
        raise ValueError("Fixture needs at least 16 deterministic tokens")
    tokens = tokens[:16]
    other = np.array([0, 9, 21, 85, 11, 19, 6, 44, 35, 72, 11, 5], dtype=np.int32)
    filler = np.array([(3 * i + 7) % cfg["vocab_size"] for i in range(256)], dtype=np.int32)
    torch.set_num_threads(2)
    reference_path = Path(__file__).parents[1] / "dsv41-reference.py"
    spec = importlib.util.spec_from_file_location("slot_retention_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    weights_path = args.fixture_dir / "deepseek41-fixture.gguf"
    engram_path = args.fixture_dir / "deepseek41.engram.bin"
    weights = reference.GgufWeights(weights_path)
    engram = reference.load_engram(engram_path)

    def oracle(ids):
        return reference.Reference(weights, config["config"], engram, "model").forward(ids.tolist()).cpu().numpy()

    expected, expected_other = oracle(tokens), oracle(other)
    ptr, number = ctypes.c_void_p, ctypes.c_int
    signatures = {
        "LoadModel": ([ctypes.c_char_p] + [number] * 5 + [ctypes.c_char_p], ptr),
        "Free": ([ptr], None), "ResetChecked": ([ptr], number), "NPast": ([ptr], number),
        "Forward": ([ptr, ptr, number, ptr], number),
        "SlotAlloc": ([ptr], number), "SlotFree": ([ptr, number], number),
        "SetActiveSlot": ([ptr, number], number), "Truncate": ([ptr, number], number),
        "SlotStatus": ([ptr, number] + [ctypes.POINTER(number)] * 3, number),
        "SlotCanReuse": ([ptr, number, number, number], number),
        "SlotCanRetain": ([ptr, number, number, ctypes.c_uint64], number),
        "SlotReleaseGraphs": ([ptr, number], number),
        "TestSharedPlacement": ([ptr, number, number, ptr, number], number),
    }
    api = {}
    dll_directories = []
    failure_keys = ("TS_DSV41_TEST_FAIL_STAGE", "TS_DSV41_TEST_FAIL_POSITION", "TS_DSV41_TEST_FAIL_KIND")
    settings = {"TS_DSV41_REWIND_CHECKPOINT": "1", "TS_DSV41_TP": "0", "TS_DSV4_FA": "0",
                "TS_DSV41_ENGRAM_THREADS": "2"}
    if args.cpu_admission_reserve_mb is not None:
        settings["TS_DSV4_GRAPH_CACHE_HEADROOM_MB"] = str(args.cpu_admission_reserve_mb)
    environment_keys = tuple(settings) + failure_keys
    environment_before = {key: os.environ.get(key) for key in environment_keys}
    os.environ.update(settings)
    for key in failure_keys:
        os.environ.pop(key, None)
    checks = []
    backend_observations = []
    handle = None
    error = None

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
        if not passed:
            raise AssertionError(name)

    def status(slot, selected_handle=None):
        values = [number(-777), number(-778), number(-779)]
        result = api["SlotStatus"](handle if selected_handle is None else selected_handle, slot,
                                   *(ctypes.byref(value) for value in values))
        return result, tuple(value.value for value in values)

    def select(slot):
        check(f"select_{slot}", api["SetActiveSlot"](handle, slot) == 0)

    def reset():
        check("checked_reset_succeeds", api["ResetChecked"](handle) == 1)
        check("checked_reset_position", api["NPast"](handle) == 0)

    def release_graphs(name, slot):
        before, active_head = status(slot), api["NPast"](handle)
        check(name, api["SlotReleaseGraphs"](handle, slot) == 1)
        check(name + "_preserves_cache_and_binding",
              status(slot) == before and api["NPast"](handle) == active_head)

    def forward(ids, expect_success=True):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        output = np.empty(cfg["vocab_size"], dtype=np.float32)
        result = api["Forward"](handle, ids.ctypes.data, len(ids), output.ctypes.data)
        if expect_success:
            check("forward_success", result == 0, status=result, tokens=len(ids))
            check("forward_finite", np.isfinite(output).all())
        return result, output

    def compare(name, actual, target):
        delta = actual.astype(np.float64) - target.astype(np.float64)
        check(name, np.isfinite(actual).all() and np.isfinite(target).all()
              and np.allclose(actual, target, atol=2e-5, rtol=2e-5),
              max_absolute_error=float(np.abs(delta).max()),
              relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(target.astype(np.float64)), 1e-300)),
              argmax=int(actual.argmax()), reference_argmax=int(target.argmax()), atol=2e-5, rtol=2e-5)

    def inspect_query(name, slot, cached_head, target, expected_result):
        active_head = api["NPast"](handle)
        before = status(slot)
        result = api["SlotCanReuse"](handle, slot, cached_head, target)
        check(name, result == expected_result, result=result, cached_head=cached_head, target=target)
        check(name + "_preserves_slot_status", status(slot) == before)
        check(name + "_preserves_active_binding", api["NPast"](handle) == active_head)

    def budget_query(name, slot, count, budget, expected_result):
        before, active_head = status(slot), api["NPast"](handle)
        result = api["SlotCanRetain"](handle, slot, count, budget)
        check(name, result == expected_result, retained_count=count, budget_per_device=budget, result=result)
        check(name + "_preserves_state_and_binding", status(slot) == before and api["NPast"](handle) == active_head)

    try:
        if os.name == "nt":
            dll_directories.append(os.add_dll_directory(str(args.library.resolve().parent)))
            if args.backend == "CUDA":
                cuda_path = os.environ.get("CUDA_PATH")
                if not cuda_path:
                    raise RuntimeError("CUDA_PATH must identify the CUDA runtime used by the isolated library")
                dll_directories.append(os.add_dll_directory(str(Path(cuda_path) / "bin")))
        lib = ctypes.CDLL(str(args.library.resolve()))
        for name, (parameters, result) in signatures.items():
            function = getattr(lib, "TSGgml_Dsv4" + name)
            function.argtypes, function.restype = parameters, result
            api[name] = function
        handle = api["LoadModel"](str(weights_path).encode(), args.gpus, 512, 32, 2, 0, args.backend.encode())
        check("fixture_loaded", bool(handle))
        source, observer = api["SlotAlloc"](handle), api["SlotAlloc"](handle)
        check("distinct_slots_allocated", source > 0 and observer > 0 and source != observer)
        select(observer)
        forward(other[:3])
        for layer in range(cfg["num_hidden_layers"]):
            description = ctypes.create_string_buffer(256)
            placed = api["TestSharedPlacement"](handle, layer, 0, description, len(description))
            observed = description.value.decode()
            backend_observations.append(dict(layer=layer, matches_layer_backend=placed == 1, description=observed))
            # TSDSV4 is the TensorSharp CUDA wrapper's actual backend name;
            # its implementation delegates device work to the paired CUDA backend.
            prefixes = ("CUDA", "TSDSV4-") if args.backend == "CUDA" else ("CPU",)
            names = observed.split(" -> ")
            matches_request = len(names) == 2 and all(name.startswith(prefixes) for name in names)
            check("actual_backend_matches_requested", placed == 1 and matches_request,
                  layer=layer, requested=args.backend, observed=observed)
        check("fresh_inactive_status", status(source) == (1, (0, -1, 1)))
        budget_query("empty_slot_not_retained", source, 0, 2**64 - 1, 0)
        for missing in (-1, observer + 100000):
            check("missing_status_preserves_outputs", status(missing) == (0, (-777, -778, -779)))
            inspect_query("missing_slot_not_reusable", missing, 0, 0, 0)
            budget_query("missing_slot_not_retained", missing, 0, 2**64 - 1, 0)
        check("null_handle_status_preserves_outputs", status(source, ctypes.c_void_p()) == (0, (-777, -778, -779)))
        check("null_handle_not_reusable", api["SlotCanReuse"](None, source, 0, 0) == 0)
        check("null_handle_not_retained", api["SlotCanRetain"](None, source, 0, 2**64 - 1) == 0)
        check("null_handle_checked_reset_refused", api["ResetChecked"](None) == 0)
        check("null_handle_graph_release_refused", api["SlotReleaseGraphs"](None, source) == 0)
        check("missing_slot_graph_release_refused", api["SlotReleaseGraphs"](handle, -1) == 0)

        # Inactive inspection cannot steal the active slot. Budget rejection
        # cannot evict/clear it: the exact same retained buffers remain usable.
        for head, target in ((8, 6), (8, 0), (3, 3), (16, 8)):
            select(source)
            reset()
            forward(tokens[:head])
            check("prefix_status", status(source) == (1, (head, head, 1)))
            select(observer)
            inspect_query("head_match_eligible", source, head, target, 1)
            inspect_query("stale_managed_head_rejected", source, head + 1, target, 0)
            inspect_query("negative_target_rejected", source, head, -1, 0)
            inspect_query("beyond_head_rejected", source, head, head + 1, 0)
            if head >= 8:
                inspect_query("misaligned_rewind_rejected", source, head, 5, 0)
            budget_query("zero_budget_rejected", source, 0, 0, 0)
            budget_query("one_byte_budget_rejected", source, 0, 1, 0)
            budget_query("negative_retained_count_rejected", source, -1, 2**64 - 1, 0)
            budget_query("sufficient_explicit_budget_and_device_headroom", source, 3, 2**64 - 1, 1)
            select(source)
            check("actual_truncate_matches_query", api["Truncate"](handle, target) == 1)
            check("actual_target_position", api["NPast"](handle) == target)
            stop = max(head, target + 1)
            if stop == target:
                stop += 1
            if target == head:
                stop = min(len(tokens), head + 4)
            _, actual = forward(tokens[target:stop])
            compare("retained_continuation_reference", actual, expected[stop - 1])
            select(0)
            reset()
            if target:
                forward(tokens[:target])
            _, cold = forward(tokens[target:stop])
            compare("retained_continuation_cold", actual, cold)

        # Only a saved prompt boundary can serve 256 -> 6 (span is 249).
        # Compare with the same prefill/truncate/refill schedule before decode
        # overwrites the live ring; this isolates copying the shadow state.
        select(0)
        reset()
        forward(tokens[:8])
        check("control_live_truncate", api["Truncate"](handle, 6) == 1)
        _, live = forward(tokens[6:12])
        select(source)
        reset()
        forward(tokens[:8])
        for position in range(8, 256):
            forward(filler[position:position + 1])
        check("decode_keeps_checkpoint", status(source) == (1, (256, 8, 1)))
        select(observer)
        inspect_query("checkpoint_route_eligible", source, 256, 6, 1)
        inspect_query("checkpoint_does_not_waive_alignment", source, 256, 5, 0)
        select(source)
        check("checkpoint_truncate_matches_query", api["Truncate"](handle, 6) == 1)
        _, actual = forward(tokens[6:12])
        compare("checkpoint_continuation_matches_live", actual, live)
        compare("checkpoint_continuation_reference", actual, expected[11])

        # No multi-token call means no rewind shadow, despite the same head.
        select(source)
        reset()
        for position in range(252):
            forward(filler[position:position + 1])
        check("single_token_history_has_no_checkpoint", status(source) == (1, (252, -1, 1)))
        select(observer)
        inspect_query("unreachable_depth_rejected", source, 252, 2, 0)
        inspect_query("reset_remains_reachable", source, 252, 0, 1)
        select(source)
        check("unreachable_actual_truncate_rejected", api["Truncate"](handle, 2) == 0 and api["NPast"](handle) == 252)
        check("zero_actual_truncate_accepted", api["Truncate"](handle, 0) == 1)
        _, actual = forward(tokens[:12])
        compare("reset_after_refusal_reference", actual, expected[11])

        # A shadow after the rewind target belongs to the abandoned branch.
        # Keep decoding one token per call so no new prompt-boundary snapshot
        # can hide stale lineage. At258, target8 is beyond live span249.
        branch = np.array([(7 * i + 23) % cfg["vocab_size"] for i in range(259)], dtype=np.int32)
        select(source)
        reset()
        forward(tokens[:8])
        check("lineage_initial_checkpoint", status(source) == (1, (8, 8, 1)))
        check("lineage_first_rewind", api["Truncate"](handle, 6) == 1)
        check("future_checkpoint_invalidated", status(source) == (1, (6, -1, 1)))
        for position in range(6, 258):
            forward(branch[position:position + 1])
        check("new_branch_has_no_abandoned_checkpoint", status(source) == (1, (258, -1, 1)))
        select(observer)
        inspect_query("abandoned_branch_checkpoint_refused", source, 258, 8, 0)
        select(source)
        check("abandoned_branch_actual_truncate_refused", api["Truncate"](handle, 8) == 0
              and status(source) == (1, (258, -1, 1)))
        _, actual = forward(branch[258:259])
        # Reproduce the same branch and exact forward-call partition with no
        # intervening retention query/refused truncate. This checks that a
        # refusal leaves continuation intact without relaxing oracle limits.
        select(0)
        reset()
        forward(tokens[:8])
        check("lineage_control_first_rewind", api["Truncate"](handle, 6) == 1)
        for position in range(6, 259):
            _, cold = forward(branch[position:position + 1])
        compare("lineage_refusal_preserves_exact_branch", actual, cold)

        for stage, fault_position in (("engram", 4), ("compute", 4), ("checkpoint", 8)):
            select(source)
            reset()
            forward(tokens[:4])
            os.environ["TS_DSV41_TEST_FAIL_STAGE"] = stage
            os.environ["TS_DSV41_TEST_FAIL_POSITION"] = str(fault_position)
            result, _ = forward(tokens[4:8], expect_success=False)
            for key in failure_keys:
                os.environ.pop(key, None)
            check("fault_injected_" + stage, result < 0, test_hooks_required=True)
            available, (head, checkpoint, healthy) = status(source)
            check("failed_slot_is_available_but_unhealthy", available == 1 and healthy == 0,
                  stage=stage, head=head, checkpoint=checkpoint)
            select(observer)
            inspect_query("failed_slot_cannot_reuse_even_noop", source, head, head, 0)
            inspect_query("failed_slot_cannot_reuse_reset", source, head, 0, 0)
            budget_query("failed_slot_cannot_retain", source, 0, 2**64 - 1, 0)
            select(source)
            reset()
            check("reset_recovers_health", status(source) == (1, (0, -1, 1)))
            _, actual = forward(tokens[:12])
            compare("failure_reset_reference_" + stage, actual, expected[11])

        select(observer)
        check("observer_prefix_never_changed", api["NPast"](handle) == 3)
        release_graphs("inactive_source_graphs_released", source)
        release_graphs("selected_observer_graphs_released", observer)
        _, actual = forward(other[3:])
        compare("observer_continuation_reference", actual, expected_other[-1])
        check("retained_source_freed", api["SlotFree"](handle, source) == 0)
        check("freed_slot_status_unavailable", status(source) == (0, (-777, -778, -779)))
        inspect_query("freed_slot_not_reusable", source, 12, 12, 0)
        replacement = api["SlotAlloc"](handle)
        check("replacement_has_fresh_identity", replacement > source and replacement != observer)
        select(replacement)
        check("replacement_has_fresh_state", status(replacement) == (1, (0, -1, 1)))
        forward(other[:3])
        release_graphs("replacement_selected_graphs_released", replacement)
        _, actual = forward(other[3:])
        compare("replacement_after_graph_disposal_reference", actual, expected_other[-1])
        select(0)
        check("replacement_freed", api["SlotFree"](handle, replacement) == 0)
        check("observer_freed", api["SlotFree"](handle, observer) == 0)
    except Exception:
        error = traceback.format_exc()
    finally:
        if handle:
            api["Free"](handle)
        for directory in reversed(dll_directories):
            directory.close()
        for key, value in environment_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report = dict(backend=args.backend, gpus=args.gpus if args.backend == "CUDA" else 0,
                      requested_gpus=args.gpus, backend_observations=backend_observations,
                      settings=settings, test_hooks_required=True, atol=2e-5, rtol=2e-5,
                      library=str(args.library.resolve()), library_sha256=digest(args.library),
                      source_sha256=digest(__file__), reference_sha256=digest(reference_path),
                      fixture_sha256={path.name: digest(path) for path in (weights_path, config_path, engram_path)},
                      checks=checks, passed=error is None and all(item["passed"] for item in checks), error=error)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    outcome = "PASS" if report["passed"] else "FAIL"
    print(f"{outcome}: {sum(item['passed'] for item in checks)}/{len(checks)} retained-slot checks; {args.report}")
    if error:
        print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
