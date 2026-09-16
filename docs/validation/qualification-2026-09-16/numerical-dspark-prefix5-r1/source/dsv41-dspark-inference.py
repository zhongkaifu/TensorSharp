#!/usr/bin/env python3
"""Actual native V4.1 DSpark graph, verify, rewind and sequence-slot checks.

Uses a deterministic small target fixture and an independent PyTorch drafter
oracle. No downloaded-model quality or serving performance claim follows.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from gguf import GGUFWriter


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(target, directory):
    metadata = json.loads((target / "deepseek41.config.json").read_text())
    if metadata.get("fixture") is not True:
        raise ValueError("Use the small deterministic fixture, not downloaded model weights")
    c = metadata["config"]["text_config"]
    reference = load_module("reference", Path(__file__).parents[1] / "dsv41-reference.py")
    trunk = reference.GgufWeights(target / "deepseek41-fixture.gguf")
    layers, count, used, rank, block = [1, 3, 4], 3, 2, 32, 5
    output = directory / "deepseek41-dspark-fixture.gguf"
    writer = GGUFWriter(str(output), "deepseek41-dspark")
    writer.add_string("dspark.target_architecture", "deepseek41")
    for name, value in {"block_size": block, "markov_rank": rank, "noise_token_id": 250,
                        "stage_count": 3, "expert_count": count, "expert_used_count": used}.items():
        writer.add_uint32("dspark." + name, value)
    writer.add_array("dspark.target_layer_ids", layers)
    rng = np.random.default_rng(410915)
    dim, hc = c["hidden_size"], c["hc_mult"]
    suffixes = ["attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_sinks.weight",
                "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
                "hc_attn_fn.weight", "hc_attn_scale.weight", "hc_attn_base.weight", "hc_ffn_fn.weight",
                "hc_ffn_scale.weight", "hc_ffn_base.weight", "ffn_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias",
                "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]
    for stage, layer in enumerate(layers):
        for suffix in suffixes:
            name = f"blk.{layer}.{suffix}"
            # Existing trunk stores router bias under ffn_exp_probs_b.bias.
            if name not in trunk.tensors and suffix == "exp_probs_b.bias":
                name = f"blk.{layer}.ffn_exp_probs_b.bias"
            shape = tuple(reversed(trunk.shape(name)))
            value = trunk.weight(name).numpy().reshape(shape)
            if suffix.endswith("_exps.weight") or suffix == "ffn_gate_inp.weight":
                value = value[:count]
            elif suffix == "exp_probs_b.bias":
                value = value.reshape(-1)[:count]
            if suffix == "attn_output_a.weight":
                value = value.reshape(c["o_groups"] * c["o_lora_rank"], -1)
            writer.add_tensor(f"mtp.{stage}.{suffix}", np.ascontiguousarray(value, np.float32))
    writer.add_tensor("mtp.0.main_norm.weight", np.ones(dim, np.float32))
    writer.add_tensor("mtp.0.main_proj.weight", rng.normal(0, 1 / (dim * 3)**.5, (dim, dim * 3)).astype(np.float32))
    writer.add_tensor("mtp.2.norm.weight", np.ones(dim, np.float32))
    writer.add_tensor("mtp.2.markov_head.markov_w1.weight", rng.normal(0, .05, (c["vocab_size"], rank)).astype(np.float32))
    writer.add_tensor("mtp.2.markov_head.markov_w2.weight", rng.normal(0, .05, (c["vocab_size"], rank)).astype(np.float32))
    writer.add_tensor("mtp.2.confidence_head.proj.weight", rng.normal(0, .03, (1, dim + rank)).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    settings = dict(block_size=block, stages=3, expert_count=count, expert_used_count=used,
                    noise_token=250, markov_rank=rank, target_layers=layers)
    (directory / "draft-config.json").write_text(json.dumps(settings, indent=2) + "\n")
    return metadata["config"], settings, output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--library", type=Path)
    ap.add_argument("--backend", default="CPU")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--require-split-dspark", action="store_true",
                    help="Require target feature layers on at least two actual devices")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--state-only", action="store_true", help="Run state fixtures separately; exclude independent reference checks explicitly")
    ap.add_argument("--reference-only", action="store_true", help="Run independent reference checks only; exclude state fixtures explicitly")
    ap.add_argument("--keep-going", action="store_true", help="Continue numerical comparisons after failure, retaining failed status and nonzero exit")
    ap.add_argument("--native-ring-control", action="store_true", help="Add diagnostic draft arithmetic using observed native KV; does not replace independent end-to-end oracle")
    ap.add_argument("--trace-native", action="store_true", help="Record native committed-KV inputs for numerical diagnosis")
    ap.add_argument("--reference-prefixes", nargs="+", type=int, default=[1, 5, 17],
                    help="Explicit reference prefix lengths; separate diagnostic scopes retain their own results")
    args = ap.parse_args()
    if args.state_only and args.reference_only:
        ap.error("--state-only and --reference-only are mutually exclusive")
    if any(length < 1 or length > 32 for length in args.reference_prefixes):
        ap.error("Reference prefixes must contain 1..32 tokens")
    args.out.mkdir(exist_ok=False, parents=True)
    if args.trace_native:
        os.environ["TS_DSV41_TRACE_DIR"] = str(args.out / "native-trace")
    torch.set_num_threads(2)
    report = dict(started_at_unix=time.time(), release_qualified=False, checks=[], status="preparing",
                  source_sha256=sha(__file__), backend=args.backend, gpus=args.gpus,
                  atol=2e-5, rtol=2e-5, scope="Deterministic native numerical/state fixture, not a language model")
    report["reference_checks_excluded"] = args.state_only
    report["state_checks_excluded"] = args.reference_only
    report["native_ring_control"] = args.native_ring_control
    report["reference_prefixes"] = [] if args.state_only else args.reference_prefixes
    handles = []
    api = {}

    def save():
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def check(name, ok, recoverable=False, **details):
        report["checks"].append(dict(name=name, passed=bool(ok), **details))
        save()
        if not ok:
            if args.keep_going and recoverable:
                report.setdefault("failed_checks", []).append(name)
                save()
                return
            raise AssertionError(name)

    def compare(name, actual, expected, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        np.savez(args.out / f"check-{len(report['checks']):04}-{name}.npz", actual=actual, expected=expected)
        check(name, np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=2e-5, rtol=2e-5),
              recoverable=True, shape=list(actual.shape), max_abs=float(difference.max(initial=0)),
              relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)))

    try:
        config, settings, draft_path = prepare(args.fixture_dir, args.out)
        report["inputs"] = {str(p.resolve()): sha(p) for p in [
            args.fixture_dir / "deepseek41-fixture.gguf", args.fixture_dir / "deepseek41.config.json",
            args.fixture_dir / "deepseek41.engram.bin", draft_path, args.out / "draft-config.json"]}
        if args.prepare_only:
            report["status"] = "prepared-not-executed"
            return
        if args.library is None:
            raise ValueError("--library is required for native execution")
        report["native_sha256"] = sha(args.library)
        lib = ct.CDLL(str(args.library.resolve()))
        signatures = {
            "LoadModelDspark": ([ct.c_char_p] + [ct.c_int]*4 + [ct.c_char_p, ct.c_int, ct.c_char_p], ct.c_void_p),
            "Forward": ([ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p], ct.c_int),
            "ForwardSpec": ([ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p], ct.c_int),
            "DsparkDraft": ([ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p], ct.c_int),
            "DsparkBlockSize": ([ct.c_void_p], ct.c_int),
            "Rewind": ([ct.c_void_p, ct.c_int], ct.c_int), "Truncate": ([ct.c_void_p, ct.c_int], ct.c_int),
            "NPast": ([ct.c_void_p], ct.c_int), "ResetChecked": ([ct.c_void_p], ct.c_int),
            "SlotAlloc": ([ct.c_void_p], ct.c_int), "SetActiveSlot": ([ct.c_void_p, ct.c_int], ct.c_int),
            "SlotFree": ([ct.c_void_p, ct.c_int], ct.c_int), "Free": ([ct.c_void_p], None),
        }
        for name, (arguments, result) in signatures.items():
            fn = getattr(lib, "TSGgml_Dsv4" + name)
            fn.argtypes, fn.restype = arguments, result
            api[name] = fn
        if args.native_ring_control:
            read_ring = lib.TSGgml_Dsv4TestDsparkReadRing
            read_ring.argtypes, read_ring.restype = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p], ct.c_int
        for _ in range(2):
            handle = api["LoadModelDspark"](str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                args.gpus, 1024, 32, 2, str(draft_path).encode(), 0, args.backend.encode())
            check("load_draft_model", bool(handle))
            handles.append(handle)
        live, cold = handles
        check("draft_block_size", api["DsparkBlockSize"](live) == 5)
        if args.require_split_dspark:
            layer_device = lib.TSGgml_Dsv4TestLayerDevice
            layer_device.argtypes, layer_device.restype = [ct.c_void_p, ct.c_int], ct.c_int
            devices = [layer_device(live, layer) for layer in settings["target_layers"]]
            check("draft_target_features_cross_devices", min(devices) >= 0 and len(set(devices)) >= 2,
                  target_layers=settings["target_layers"], devices=devices)
        vocab = config["text_config"]["vocab_size"]
        rng = np.random.default_rng(159410)
        corpus = rng.integers(3, 240, 620, dtype=np.int32)

        def reset(handle):
            if api["ResetChecked"](handle) != 1:
                raise AssertionError("Native reset failed")

        def forward(handle, ids, spec=False):
            ids = np.ascontiguousarray(ids, np.int32)
            result = np.empty((len(ids) if spec else 1, vocab), np.float32)
            code = api["ForwardSpec" if spec else "Forward"](handle, ids.ctypes.data, len(ids), result.ctypes.data)
            if code != (1 if spec else 0):
                raise AssertionError(f"Native {'verify' if spec else 'forward'} returned {code}")
            return result if spec else result[0]

        def prefill(handle, ids):
            for begin in range(0, len(ids), 32):
                output = forward(handle, ids[begin:begin+32])
            return output

        def draft(handle, anchor):
            tokens, confidence = np.empty(5, np.int32), np.empty(5, np.float32)
            if api["DsparkDraft"](handle, int(anchor), tokens.ctypes.data, confidence.ctypes.data) != 5:
                raise AssertionError("Native DSpark did not produce its complete block")
            return tokens, confidence

        oracle = load_module("dspark_reference", Path(__file__).parents[1] / "dsv41-dspark-reference.py")
        weights = oracle.reference.GgufWeights(args.fixture_dir / "deepseek41-fixture.gguf")
        draft_weights = oracle.reference.GgufWeights(draft_path)
        engram = oracle.reference.load_engram(args.fixture_dir / "deepseek41.engram.bin")
        report["oracle_sources"] = {str(p): sha(p) for p in [
            Path(__file__).parents[1] / "dsv41-dspark-reference.py", Path(__file__).parents[1] / "dsv41-reference.py"]}
        for length in (() if args.state_only else args.reference_prefixes):
            target = oracle.TargetWithDraftFeatures(weights, config, engram, "model", target_layers=settings["target_layers"])
            if args.trace_native:
                target.output_dir = args.out / "oracle-trace" / f"prefix{length}"
                target.output_dir.mkdir(parents=True)
            expected_logits = target.forward(corpus[:length].tolist()).numpy()
            head = oracle.DSparkReference(draft_weights, weights, config,
                **{k: v for k, v in settings.items() if k != "target_layers"})
            head.commit_features(target.draft_features())
            np.savez(args.out / f"committed-inputs-prefix{length}.npz",
                     features=head.committed_features.numpy(), main=head.committed_main.numpy(),
                     **{f"stage{stage}": value.numpy() for stage, value in enumerate(head.committed_prequant)})
            anchor = int(expected_logits[-1].argmax())
            tokens, confidence, trace = head.draft(anchor)
            np.savez(args.out / f"oracle-prefix{length}.npz", tokens=np.array(tokens), confidence=confidence.numpy(),
                     target_logits=expected_logits, **{k: v.numpy() for k, v in trace.items()})
            reset(live)
            compare(f"target_prefix{length}", prefill(live, corpus[:length]), expected_logits[-1])
            actual_tokens, actual_confidence = draft(live, anchor)
            compare(f"draft_tokens_prefix{length}", actual_tokens, tokens, exact=True)
            if args.native_ring_control:
                native_keys = []
                for stage in range(3):
                    native_key = np.empty((length, config["text_config"]["head_dim"]), np.float32)
                    check("read_native_draft_ring", read_ring(live, stage, length, native_key.ctypes.data) == 1)
                    np.savez(args.out / f"ring-prefix{length}-stage{stage}.npz", native=native_key, independent=head.keys[stage].numpy())
                    native_keys.append(torch.from_numpy(native_key))
                independent_keys = head.keys
                head.keys = native_keys
                control_tokens, control_confidence, control_trace = head.draft(anchor)
                np.savez(args.out / f"native-ring-control-prefix{length}.npz", **{k:v.numpy() for k,v in control_trace.items()})
                head.keys = independent_keys
                compare(f"native_ring_draft_tokens_prefix{length}", actual_tokens, control_tokens, exact=True)
                compare(f"native_ring_draft_confidence_prefix{length}", actual_confidence, control_confidence.numpy())
            batch = np.array([anchor] + tokens, np.int32)
            expected_verify = target.forward(batch.tolist()).numpy()
            compare(f"verify_full_vocabulary_prefix{length}", forward(live, batch, spec=True), expected_verify)
            compare(f"draft_confidence_prefix{length}", actual_confidence, confidence.numpy())
        if args.reference_only:
            if report.get("failed_checks"):
                raise AssertionError(f"{len(report['failed_checks'])} numerical comparisons failed")
            report["status"] = "passed"
            return
        # Same-backend comparisons isolate rollback from independent oracle rounding.
        for length in (1, 2, 7, 8, 9, 255, 257):
            for accepted in range(6):
                reset(live)
                reset(cold)
                prefill(live, corpus[:length])
                prefill(cold, corpus[:length])
                batch = corpus[length:length+6]
                forward(live, batch, spec=True)
                kept = length + accepted + 1
                check(f"rewind_p{length}_accepted{accepted}", api["Rewind"](live, kept) == 1)
                check("rewind_position", api["NPast"](live) == kept)
                forward(cold, batch[:accepted+1])
                if args.native_ring_control:
                    # Compare the committed window in logical order, including
                    # wrapped rings; masked rejected rows are outside this view.
                    count = min(kept, config["text_config"]["sliding_window"])
                    for stage in range(3):
                        rows = []
                        for handle in (live, cold):
                            row = np.empty((count, config["text_config"]["head_dim"]), np.float32)
                            check("read_native_draft_ring", read_ring(handle, stage, count, row.ctypes.data) == 1)
                            rows.append(row)
                        compare(f"rewound_ring_p{length}_accepted{accepted}_stage{stage}", rows[0], rows[1], exact=True)
                left, left_conf = draft(live, 243)
                right, right_conf = draft(cold, 243)
                compare(f"rewound_draft_p{length}_accepted{accepted}", left, right, exact=True)
                compare(f"rewound_confidence_p{length}_accepted{accepted}", left_conf, right_conf)
                compare(f"divergent_continuation_p{length}_accepted{accepted}",
                        forward(live, [243, 241, 239]), forward(cold, [243, 241, 239]))
                check("ordinary_forward_invalidates_old_verify", api["Rewind"](live, kept) == 0)
        reset(live)
        prefill(live, corpus[:7])
        position = api["NPast"](live)
        output = np.empty((7, vocab), np.float32)
        for name, invalid in [("overwide", np.arange(7, dtype=np.int32)),
                              ("invalid_token", np.array([3, vocab, 4], np.int32))]:
            check(name + "_verify_refused", api["ForwardSpec"](live, invalid.ctypes.data, len(invalid), output.ctypes.data) == 0)
            check(name + "_position_unchanged", api["NPast"](live) == position)
        forward(live, corpus[7:13], spec=True)
        check("rewind_cannot_drop_anchor", api["Rewind"](live, 7) == 0)
        check("repeated_rewind_first", api["Rewind"](live, 11) == 1)
        check("repeated_rewind_second", api["Rewind"](live, 8) == 1)
        check("repeated_rewind_cannot_extend_interval", api["Rewind"](live, 7) == 0)
        # Preserve the prompt-boundary checkpoint while the live raw/draft rings wrap.
        reset(live)
        reset(cold)
        prefill(live, corpus[:2])
        for token in corpus[2:302]:
            forward(live, [token])
        check("conversation_rewind_uses_checkpoint", api["Truncate"](live, 2) == 1)
        prefill(cold, corpus[:2])
        left, left_conf = draft(live, 243)
        right, right_conf = draft(cold, 243)
        compare("checkpoint_restores_draft_ring_tokens", left, right, exact=True)
        compare("checkpoint_restores_draft_ring_confidence", left_conf, right_conf)
        compare("checkpoint_branch_continuation", forward(live, [243, 241, 239]), forward(cold, [243, 241, 239]))
        # Two real slot buffers keep their draft rings isolated under interleaving.
        reset(live)
        prefill(live, corpus[:9])
        expected_a = draft(live, 243)
        second = api["SlotAlloc"](live)
        check("slot_allocation", second > 0)
        check("select_second_slot", api["SetActiveSlot"](live, second) == 0)
        prefill(live, corpus[30:43])
        expected_b = draft(live, 241)
        for slot, anchor, expected in [(0, 243, expected_a), (second, 241, expected_b)] * 2:
            check("interleaved_slot_select", api["SetActiveSlot"](live, slot) == 0)
            actual = draft(live, anchor)
            compare("interleaved_draft_tokens", actual[0], expected[0], exact=True)
            compare("interleaved_draft_confidence", actual[1], expected[1])
        check("select_primary_before_free", api["SetActiveSlot"](live, 0) == 0)
        check("free_second_slot", api["SlotFree"](live, second) == 0)
        if report.get("failed_checks"):
            raise AssertionError(f"{len(report['failed_checks'])} numerical comparisons failed")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for handle in reversed(handles):
            api["Free"](handle)
        report["finished_at_unix"] = time.time()
        if args.library and "native_sha256" in report:
            report["native_unchanged"] = sha(args.library) == report["native_sha256"]
        report["inputs_unchanged"] = all(sha(path) == digest for path, digest in report.get("inputs", {}).items())
        save()
        print(json.dumps({key: report[key] for key in ("status", "release_qualified", "finished_at_unix")} |
                         {"checks": len(report["checks"]), "passed": sum(c["passed"] for c in report["checks"])}))


if __name__ == "__main__":
    main()
