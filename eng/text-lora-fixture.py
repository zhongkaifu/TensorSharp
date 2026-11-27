#!/usr/bin/env python3
"""Write a seeded text-LoRA test adapter for a base GGUF, plus its merged-weights oracle.

Given a base model GGUF, draws a deterministic rank-R LoRA on the chosen projections and writes,
under the output directory (default artifacts/lora/<base stem>/, which Git ignores):

  loras/lora-gguf.gguf          llama.cpp GGUF adapter (general.type=adapter, adapter.type=lora,
                                adapter.lora.alpha, general.architecture of the base)
  loras/lora-st.safetensors     the same factors as <tensor>.lora_a [r, in] / .lora_b [out, r],
                                rank and alpha in __metadata__ (no adapter_config.json needed)
  peft/adapter_model.safetensors  (--peft-copy) the same factors under PEFT-style names, which
                                TensorSharp must refuse with a pointer to convert_lora_to_gguf.py
  merged.gguf                   the base with every target replaced by F32 dequant(W) + s * B @ A
                                (s = alpha / r) and every other tensor and key copied unchanged
  fixture.json                  what was written, for InferenceWeb.Tests' TextLoraFixtureValidationTests
                                (TS_TEXT_LORA_FIXTURE_DIR=<output dir>)

merged.gguf is an exact oracle for an F32 base. A quantized base's own matmuls round (Q8_0 rounds the
activations too), so against the F32 merged twin expect differences of that size (about 2e-3 of the
largest logit for Q8_0) and compare top-1 agreement rather than bits.

Start the server with --loras-dir <output>/loras to load both adapters, and compare a request with
"lora" against the same request on merged.gguf (or llama.cpp's llama-server --lora loras/lora-gguf.gguf,
or llama-export-lora). A ~ N(0, 1/in); B is scaled per module so that ||s * B @ A||_F is
--relative-delta (default 4%) of ||W||_F, enough to change greedy text without wrecking it.

Targets default to the projections TensorSharp applies a LoRA to for the base's architecture
(mistral3 / hunyuan-dense: attention q/k/v/output and ffn gate/up/down plus output.weight), and to
a subset of them for nemotron_h (attention q/k/v, dense and shared-expert up, latent-in, router,
output.weight; Nemotron also adapts attn_output, ffn_down, ssm_in/ssm_out, ffn_down_shexp and
ffn_latent_up). --targets REGEX overrides them (matched against GGUF tensor names).

--self-test DIR writes a tiny mistral3 base GGUF (F32 with one F16 and one Q8_0 projection) into
DIR, runs the fixture on it and checks the merged GGUF and both adapters against each other.

Requires numpy and gguf (gguf-py, tested with 0.17.1).
"""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize, quantize

DEFAULT_TARGETS = {
    "mistral3": r"^(blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight|output\.weight)$",
    "hunyuan-dense": r"^(blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight|output\.weight)$",
    # A subset of what Nemotron adapts (NemotronRuntimeModel.s_textLoraProjections, which also covers
    # attn_output, ffn_down, ssm_in/ssm_out, ffn_down_shexp and ffn_latent_up/ffn_latent_out).
    "nemotron_h": r"^(blk\.\d+\.(attn_q|attn_k|attn_v|ffn_up|ffn_up_shexp|ffn_latent_down|ffn_latent_in|ffn_gate_inp)\.weight|output\.weight)$",
}
DEFAULT_TARGETS["nemotron_h_moe"] = DEFAULT_TARGETS["nemotron_h"]
DEFAULT_TARGETS["nemotron_h_omni"] = DEFAULT_TARGETS["nemotron_h"]


def field_value(reader, key):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def logical(tensor):
    """The tensor as F32 with numpy shape (ne[n-1], ..., ne[0]) (row-major [out, in] for a matrix)."""
    data = np.asarray(tensor.data)
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return data.astype(np.float32).reshape(tuple(reversed([int(d) for d in tensor.shape])))
    return dequantize(data, tensor.tensor_type).astype(np.float32).reshape(tuple(reversed([int(d) for d in tensor.shape])))


def draw_factors(name, w, rank, alpha, relative, seed):
    """Seeded A [r, in], B [out, r] for one module; the seed mixes in the name, so a module's
    factors do not depend on which other modules are selected."""
    out_dim, in_dim = w.shape
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    a = rng.standard_normal((rank, in_dim)) / np.sqrt(in_dim)
    b = rng.standard_normal((out_dim, rank))
    scale = alpha / rank
    delta = scale * (b @ a)
    norm_w = float(np.linalg.norm(w.astype(np.float64)))
    norm_d = float(np.linalg.norm(delta))
    if norm_d > 0 and norm_w > 0:
        b *= relative * norm_w / norm_d
    return a.astype(np.float32), b.astype(np.float32)


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    blobs, offset = [], 0
    for name, array in tensors:
        raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * ((8 - len(head) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        for raw in blobs:
            f.write(raw)


def write_gguf_adapter(path, arch, alpha, factors, outtype):
    writer = GGUFWriter(str(path), arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(alpha))
    dtype = np.float16 if outtype == "f16" else np.float32
    for name, (a, b) in factors.items():
        # numpy [r, in] is GGUF ne [in, r]; [out, r] is ne [r, out]: llama.cpp's layout.
        writer.add_tensor(name + ".lora_a", a.astype(dtype))
        writer.add_tensor(name + ".lora_b", b.astype(dtype))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_merged(reader, path, arch, merged):
    """Copy the base GGUF (keys, tensor order, raw bytes) with the targets replaced by F32."""
    writer = GGUFWriter(str(path), arch)
    alignment = field_value(reader, "general.alignment")
    if alignment is not None:
        writer.data_alignment = int(alignment)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        if field.types[0] == GGUFValueType.ARRAY and len(field.data) == 0:
            # gguf-py cannot write an empty array; the merged file goes without the key.
            print(f"warning: {field.name} is an empty array and is not copied to the merged GGUF", file=sys.stderr)
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
    for t in reader.tensors:
        if t.name in merged:
            m = merged[t.name]
            writer.add_tensor_info(t.name, m.shape, m.dtype, m.nbytes, GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(merged[t.name] if t.name in merged else t.data)
    writer.close()


def run(base, out_dir, rank, alpha, relative, seed, targets_regex, outtype, peft_copy):
    reader = GGUFReader(str(base))
    arch = field_value(reader, "general.architecture")
    if not arch:
        sys.exit(f"{base}: no general.architecture")
    if field_value(reader, "general.type") == "adapter":
        sys.exit(f"{base} is an adapter, not a base model")
    pattern = targets_regex or DEFAULT_TARGETS.get(arch)
    if pattern is None:
        sys.exit(f"no default targets for architecture '{arch}'; pass --targets REGEX")
    regex = re.compile(pattern)
    selected = [t for t in reader.tensors if regex.match(t.name) and len(t.shape) == 2]
    if not selected:
        sys.exit(f"no 2-D tensor of {base} matches {pattern}")

    (out_dir / "loras").mkdir(parents=True, exist_ok=True)
    factors, merged, summary = {}, {}, []
    for t in selected:
        w = logical(t)
        a, b = draw_factors(t.name, w, rank, alpha, relative, seed)
        factors[t.name] = (a, b)
        delta = (alpha / rank) * (b.astype(np.float64) @ a.astype(np.float64))
        merged[t.name] = (w.astype(np.float64) + delta).astype(np.float32)
        summary.append({"tensor": t.name, "type": t.tensor_type.name, "out": int(w.shape[0]), "in": int(w.shape[1]),
                        "relative_delta": float(np.linalg.norm(delta) / max(np.linalg.norm(w.astype(np.float64)), 1e-30))})

    gguf_adapter = out_dir / "loras" / "lora-gguf.gguf"
    st_adapter = out_dir / "loras" / "lora-st.safetensors"
    write_gguf_adapter(gguf_adapter, arch, alpha, factors, outtype)
    tensors = []
    for name, (a, b) in factors.items():
        tensors += [(name + ".lora_a", a), (name + ".lora_b", b)]
    write_safetensors(st_adapter, tensors, {"rank": str(rank), "alpha": str(alpha)})
    if peft_copy:
        (out_dir / "peft").mkdir(exist_ok=True)
        peft = [(f"base_model.model.{n.removesuffix('.weight')}.lora_A.weight", a) for n, (a, _) in factors.items()]
        peft += [(f"base_model.model.{n.removesuffix('.weight')}.lora_B.weight", b) for n, (_, b) in factors.items()]
        write_safetensors(out_dir / "peft" / "adapter_model.safetensors", peft, {})
    write_merged(reader, out_dir / "merged.gguf", arch, merged)

    manifest = {
        "base": str(Path(base).resolve()),
        "architecture": arch,
        "rank": rank,
        "alpha": alpha,
        "seed": seed,
        "relative_delta": relative,
        "gguf_outtype": outtype,
        "adapter_gguf": "loras/lora-gguf.gguf",
        "adapter_safetensors": "loras/lora-st.safetensors",
        "merged": "merged.gguf",
        "targets": summary,
    }
    (out_dir / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(selected)} target(s) of {base} -> {out_dir}")
    return manifest


# ---------------------------------------------------------------------------- self-test

def write_tiny_base(path, all_f32=False):
    """A 2-layer mistral3 GGUF shaped like InferenceWeb.Tests' DenseDecoderSyntheticModelBuilder,
    loadable by TensorSharp, with one F16 and one Q8_0 projection to exercise the dequantizer
    (unless all_f32: then the merged GGUF is an exact oracle, with no quantized base matmul)."""
    hidden, heads, kv_heads, head_dim, ffn, layers = 64, 4, 2, 16, 128, 2
    rng = np.random.default_rng(7)
    w = GGUFWriter(str(path), "mistral3")
    for key, value in [("block_count", layers), ("context_length", 4096), ("embedding_length", hidden),
                       ("feed_forward_length", ffn), ("attention.head_count", heads), ("attention.head_count_kv", kv_heads),
                       ("attention.key_length", head_dim), ("attention.value_length", head_dim),
                       ("rope.dimension_count", head_dim), ("vocab_size", 261)]:
        w.add_uint32(f"mistral3.{key}", value)
    w.add_float32("mistral3.rope.freq_base", 1_000_000.0)
    w.add_float32("mistral3.attention.layer_norm_rms_epsilon", 1e-5)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_map = dict(zip(bs, cs))
    # gguf-py cannot write an empty merges array, so one merge of two control bytes (never in
    # text) joins the 256 byte tokens.
    merge = chr(byte_map[0]) + chr(byte_map[1])
    tokens = [chr(byte_map[b]) for b in range(256)] + [merge, "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"]
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "tekken")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", [1] * 257 + [3] * 4)
    w.add_array("tokenizer.ggml.merges", [chr(byte_map[0]) + " " + chr(byte_map[1])])
    w.add_uint32("tokenizer.ggml.bos_token_id", ord("A"))
    w.add_uint32("tokenizer.ggml.eos_token_id", 0)
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_bool("tokenizer.ggml.add_eos_token", False)

    def gen(shape, scale=0.08, offset=0.0):
        return (offset + scale * rng.uniform(-1, 1, size=shape)).astype(np.float32)

    w.add_tensor("token_embd.weight", gen((261, hidden)))
    w.add_tensor("output_norm.weight", gen((hidden,), 0.5, 1.0))
    w.add_tensor("output.weight", gen((261, hidden)))
    for l in range(layers):
        p = f"blk.{l}."
        w.add_tensor(p + "attn_norm.weight", gen((hidden,), 0.2, 1.0))
        w.add_tensor(p + "attn_q.weight", gen((heads * head_dim, hidden)))
        w.add_tensor(p + "attn_k.weight", gen((kv_heads * head_dim, hidden)))
        w.add_tensor(p + "attn_v.weight", gen((kv_heads * head_dim, hidden)))
        attn_output = gen((hidden, heads * head_dim))
        w.add_tensor(p + "attn_output.weight", attn_output if l or all_f32 else attn_output.astype(np.float16))
        w.add_tensor(p + "ffn_norm.weight", gen((hidden,), 0.2, 1.0))
        w.add_tensor(p + "ffn_gate.weight", gen((ffn, hidden)))
        w.add_tensor(p + "ffn_up.weight", gen((ffn, hidden)))
        down = gen((hidden, ffn))
        if l == 1 and not all_f32:
            w.add_tensor(p + "ffn_down.weight", quantize(down, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(p + "ffn_down.weight", down)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def self_test(directory, all_f32):
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / "base.gguf"
    write_tiny_base(base, all_f32)
    out = directory / "fixture"
    manifest = run(base, out, rank=8, alpha=16.0, relative=0.05, seed=1, targets_regex=None, outtype="f32", peft_copy=True)

    base_r, merged_r = GGUFReader(str(base)), GGUFReader(str(out / "merged.gguf"))
    adapter_r = GGUFReader(str(out / "loras" / "lora-gguf.gguf"))
    st = read_safetensors(out / "loras" / "lora-st.safetensors")
    targets = {t["tensor"] for t in manifest["targets"]}
    assert len(targets) == 2 * 7 + 1, sorted(targets)
    scale = 16.0 / 8
    for t in base_r.tensors:
        m = next(x for x in merged_r.tensors if x.name == t.name)
        if t.name in targets:
            assert m.tensor_type == GGMLQuantizationType.F32, t.name
            a, b = st[t.name + ".lora_a"], st[t.name + ".lora_b"]
            ga = next(x for x in adapter_r.tensors if x.name == t.name + ".lora_a")
            gb = next(x for x in adapter_r.tensors if x.name == t.name + ".lora_b")
            assert [int(d) for d in ga.shape] == [a.shape[1], a.shape[0]], (t.name, ga.shape)   # ne [in, r]
            assert [int(d) for d in gb.shape] == [b.shape[1], b.shape[0]], (t.name, gb.shape)   # ne [r, out]
            assert np.array_equal(logical(ga), a) and np.array_equal(logical(gb), b), t.name
            expect = logical(t).astype(np.float64) + scale * (b.astype(np.float64) @ a.astype(np.float64))
            err = float(np.max(np.abs(logical(m) - expect)))
            assert err < 1e-6, (t.name, err)
        else:
            assert m.tensor_type == t.tensor_type and bytes(np.asarray(m.data).tobytes()) == bytes(np.asarray(t.data).tobytes()), t.name
    for key in base_r.fields:
        if not key.startswith("GGUF."):
            assert key in merged_r.fields and merged_r.fields[key].contents() == base_r.fields[key].contents(), key
    assert field_value(adapter_r, "general.type") == "adapter" and field_value(adapter_r, "adapter.type") == "lora"
    assert abs(field_value(adapter_r, "adapter.lora.alpha") - 16.0) < 1e-6
    assert field_value(adapter_r, "general.architecture") == "mistral3"
    rel = [t["relative_delta"] for t in manifest["targets"]]
    assert all(abs(r - 0.05) < 1e-3 for r in rel), rel
    print(f"self-test OK: {len(targets)} targets, merged == base + {scale} * B @ A, "
          f"non-targets byte-identical, GGUF adapter == safetensors adapter. Fixture: {out}")


def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    data = raw[8 + n:]
    result = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        result[name] = np.frombuffer(data[start:end], dtype=np.float32).reshape(info["shape"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", nargs="?", type=Path, help="base model GGUF (one file; a split model is not supported)")
    parser.add_argument("--out", type=Path, help="output directory (default artifacts/lora/<base stem>)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--relative-delta", type=float, default=0.04, help="||s*B@A||_F / ||W||_F per module")
    parser.add_argument("--targets", help="regex over GGUF tensor names (default: by architecture)")
    parser.add_argument("--outtype", choices=["f32", "f16"], default="f32", help="factor type in the GGUF adapter")
    parser.add_argument("--peft-copy", action="store_true", help="also write a PEFT-named copy (must be refused)")
    parser.add_argument("--self-test", type=Path, metavar="DIR", help="build a tiny base in DIR and check the fixture")
    parser.add_argument("--self-test-f32", action="store_true",
                        help="with --self-test: an all-F32 base, so merged.gguf is an exact oracle for the runtime adapter")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.self_test, args.self_test_f32)
        return
    if args.base is None:
        parser.error("a base GGUF is required (or --self-test DIR)")
    if args.rank < 1 or args.alpha <= 0 or not 0 < args.relative_delta < 1:
        parser.error("--rank >= 1, --alpha > 0 and 0 < --relative-delta < 1 are required")
    out = args.out or Path("artifacts") / "lora" / args.base.stem
    run(args.base, out, args.rank, args.alpha, args.relative_delta, args.seed, args.targets, args.outtype, args.peft_copy)


if __name__ == "__main__":
    main()
