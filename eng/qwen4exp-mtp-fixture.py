#!/usr/bin/env python3
"""Generate tiny, non-language Qwen4Exp target/shared-MTP GGUF fixtures.

Uses the independently exercised native fixture's nonzero arrays. EH/HC
values originate in a bounded publisher-head subspace; every other weight is
synthetic. This tests loading and execution ownership, not trained acceptance
or actual checkpoint quality. Routed experts use F16 storage to exercise the
loader's mapped stacked-expert route; other tensors are F32. No native library
or model inference is run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, GGUFValueType


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("q4e_target_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def byte_tokens():
    # GPT-2's reversible byte alphabet, indexed here by the original byte.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in visible}
    next_char = 256
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(next_char)
            next_char += 1
    return [mapping[b] for b in range(256)] + ["<|bos|>", "<|eos|>", "<|pad|>", "ab"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--geometry", choices=("tiny", "gdn32"), default="tiny",
                        help="gdn32 uses supported CUDA GDN/conv dimensions; tiny preserves the CPU fixture")
    parser.add_argument("--qsa", action="store_true", help="Enable synthetic target QSA with ratio4/topk8; head remains dense.")
    parser.add_argument("--attention-head-dim", type=int, choices=(8, 64, 256), default=8,
                        help="Explicit synthetic attention geometry; 64/256 exercise the flash-attention route with F16 KV.")
    parser.add_argument("--attention-heads", type=int, choices=(4, 24), default=4,
                        help="24 attention heads over the two KV heads exercises the trained target's GQA ratio.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).parent / "tests/qwen4exp-target-snapshot.py"
    module = load_module(module_path)
    target = module.Target(json.loads(args.sample.read_text(encoding="utf-8")), geometry=args.geometry,
                           attention_head_dim=args.attention_head_dim, attention_heads=args.attention_heads)
    b = target.base
    arrays = b.arrays
    rng = np.random.default_rng(481048)
    tokens = byte_tokens()
    vocab = len(tokens)
    weights = {}

    def put(name, value):
        weights[name] = np.ascontiguousarray(value, dtype=np.float32)

    def shared_blocks(prefix):
        for part in ("attn", "ffn"):
            put(prefix + f"hc_{part}_norm.weight", arrays[f"{part}.norm"].reshape(-1))
            for suffix in ("down", "up", "inject"):
                put(prefix + f"hc_{part}_{suffix}.weight", arrays[f"{part}.{suffix}"])
        for destination, source in {
            "ffn_gate_inp.weight": "router", "ffn_gate_exps.weight": "gate_exps",
            "ffn_up_exps.weight": "up_exps", "ffn_down_exps.weight": "down_exps",
            "ffn_gate_inp_shexp.weight": "sh_gate_inp", "ffn_gate_shexp.weight": "sh_gate",
            "ffn_up_shexp.weight": "sh_up", "ffn_down_shexp.weight": "sh_down",
        }.items():
            put(prefix + destination, arrays[source])

    def attention(prefix):
        for destination, source in {
            "attn_q.weight": "wq", "attn_k.weight": "wk", "attn_v.weight": "wv",
            "attn_output.weight": "wo", "attn_q_norm.weight": "q_norm", "attn_k_norm.weight": "k_norm",
        }.items():
            put(prefix + destination, arrays[source])

    put("token_embd.weight", rng.normal(0, .4, (vocab, b.H)))
    put("output.weight", rng.normal(0, .4, (vocab, b.H)))
    put("output_hc_norm.weight", arrays["head.norm"].reshape(-1))
    put("output_hc_down.weight", arrays["head.down"])
    put("output_hc_up.weight", arrays["head.up"])
    put("per_layer_token_embd.weight", rng.normal(0, .35, (17, b.H)))
    for layer in (0, 1):
        shared_blocks(f"blk.{layer}.")
    attention("blk.1.")
    for destination, source in {
        "attn_qkv.weight": "qkv", "attn_gate.weight": "gate", "ssm_beta.weight": "beta",
        "ssm_alpha.weight": "alpha", "ssm_conv1d.weight": "conv1d", "ssm_out.weight": "out_proj",
        "ssm_dt.bias": "ssm_dt", "ssm_a": "ssm_a", "ssm_norm.weight": "ssm_norm",
    }.items():
        put("blk.0." + destination, arrays["gdn." + source])
    for destination, source in {
        "ple_key.weight": "key", "ple_value.weight": "value", "ple_norm_key.weight": "norm_key",
        "ple_norm_query.weight": "norm_query", "ple_norm_conv.weight": "norm_conv",
    }.items():
        put("blk.0." + destination, arrays["ple." + source])
    # The managed loader transposes this once to the native tap-major layout.
    put("blk.0.ple_conv1d.weight", arrays["ple.conv1d"].T)
    if args.qsa:
        put("blk.1.indexer.k_norm.weight", 1 + rng.normal(0, .1, b.HD))
        put("blk.1.indexer.q_norm.weight", 1 + rng.normal(0, .1, b.HD))
        put("blk.1.indexer.k_proj.weight", rng.normal(0, .4, (b.HD, b.H)))
        put("blk.1.indexer.q_proj.weight", rng.normal(0, .4, (b.HD, b.H)))
    trunk_weights = dict(weights)
    weights.clear()
    shared_blocks("blk.2.")
    attention("blk.2.")
    for destination, source in {
        "nextn.enorm.weight": "enorm", "nextn.hnorm.weight": "hnorm", "nextn.eh_proj.weight": "eh",
        "nextn.hc_head_norm.weight": "head.norm", "nextn.hc_head_down.weight": "head.down",
        "nextn.hc_head_up.weight": "head.up",
    }.items():
        value = arrays[source]
        put("blk.2." + destination, value.reshape(-1) if "norm" in destination else value)
    put("blk.2.indexer.k_norm.weight", np.ones(b.HD))
    put("blk.2.indexer.q_norm.weight", np.ones(b.HD))
    put("blk.2.indexer.k_proj.weight", rng.normal(0, .1, (b.HD, b.H)))
    put("blk.2.indexer.q_proj.weight", rng.normal(0, .1, (b.HD, b.H)))
    head_weights = dict(weights)
    assert len(head_weights) == 32
    for tensor_set in (trunk_weights, head_weights):
        for name, data in tensor_set.items():
            if "_exps." in name:
                tensor_set[name] = np.ascontiguousarray(data, dtype=np.float16)

    def write(path, tensors, draft):
        writer = GGUFWriter(path, "qwen4exp")
        writer.add_name("Synthetic Qwen4Exp shared MTP ownership fixture")
        integers = {
            "block_count": 3 if draft else 2, "nextn_predict_layers": 1 if draft else 0,
            "context_length": 1024, "embedding_length": b.H, "attention.head_count": b.NH,
            "attention.head_count_kv": b.NK, "attention.key_length": b.HD, "attention.value_length": b.HD,
            "hyper_connection.count": b.HC, "hyper_connection.low_rank": b.LOW,
            "expert_count": b.EXP, "expert_used_count": b.USED,
            "expert_feed_forward_length": b.FF, "expert_shared_feed_forward_length": b.SH,
            "rope.dimension_count": b.c.n_rot, "attention.indexer.head_count": 1,
            "attention.indexer.key_length": b.HD, "attention.indexer.top_k": 8,
            "ssm.conv_kernel": target.CONV, "ssm.state_size": target.GK,
            "ssm.group_count": target.GKH, "ssm.time_step_rank": target.GVH,
            "ple.ngram_size": 2, "ple.heads_per_ngram": 1, "ple.conv_kernel": 3,
            "ple.eos_token_id": 257, "ple.image_token_id": 258, "embedding_length_per_layer_input": b.H,
        }
        for key, value in integers.items(): writer.add_uint32("qwen4exp." + key, value)
        for key, value in {
            "attention.layer_norm_rms_epsilon": b.eps, "rope.freq_base": b.c.rope_base,
            "rope.scaling.factor": 1 / b.c.rope_scale, "attention.scale": b.c.attn_scale,
        }.items(): writer.add_float32("qwen4exp." + key, value)
        writer.add_bool("qwen4exp.nextn_shared_target_tensors", draft)
        for key, values, subtype in (
            ("rope.dimension_sections", list(b.c.rope_sections), GGUFValueType.INT32),
            ("attention.compress_ratios", [0, 4 if args.qsa else 0] + ([0] if draft else []), GGUFValueType.INT32),
            ("attention.recurrent_layers", [1, 0] + ([0] if draft else []), GGUFValueType.UINT32),
            ("ple.layers", [0], GGUFValueType.INT32),
            ("ple.layer_multipliers", [3, 7], GGUFValueType.UINT64),
            ("ple.head_offsets", [0], GGUFValueType.UINT64),
            ("ple.head_vocab_sizes", [17], GGUFValueType.UINT64),
        ):
            writer.add_key_value("qwen4exp." + key, values, GGUFValueType.ARRAY, subtype)
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        writer.add_token_list(tokens)
        writer.add_token_types([1] * 256 + [3, 3, 3, 1])
        writer.add_token_merges(["a b"])
        writer.add_bos_token_id(256)
        writer.add_eos_token_id(257)
        writer.add_pad_token_id(258)
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)
        for name, data in tensors.items(): writer.add_tensor(name, data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    target_path, head_path = args.output_dir / "target.gguf", args.output_dir / "head.gguf"
    write(target_path, trunk_weights, False)
    write(head_path, head_weights, True)
    report = dict(schema_version=1, fixture=True, scope=__doc__, geometry=args.geometry,
                  attention_head_dim=args.attention_head_dim, attention_heads=args.attention_heads,
                  qsa=args.qsa, source_sha256=sha(__file__),
                  target_fixture_sha256=sha(module_path),
                  operator_fixture_sha256=sha(module_path.with_name("qwen4exp-mtp-operator.py")),
                  sample_sha256=sha(args.sample), vocab=vocab, eos_token_id=257, context_length=1024,
                  files={p.name: dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in (target_path, head_path)},
                  tensors={kind: {name: dict(shape=list(data.shape), dtype=str(data.dtype),
                      sha256=hashlib.sha256(data.tobytes()).hexdigest()) for name, data in values.items()}
                      for kind, values in (("target", trunk_weights), ("head", head_weights))})
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["files"], indent=2))


if __name__ == "__main__":
    main()
