#!/usr/bin/env python3
"""Independent NumPy BERT/XLM-R GGUF sentence-embedding correctness oracle.

Only GGUF reading and weight dequantization come from llama.cpp's gguf-py.
No llama/TensorSharp model inference, tokenizer, native kernels, padding masks,
QKV fusion, or final-layer pruning are used. Every sequence gets its own full
bidirectional float32 attention; every token runs through every layer.

Example:
  PYTHONPATH=/path/to/llama.cpp/gguf-py VECLIB_MAXIMUM_THREADS=1 \
    python eng/embedding-reference.py --model models/encoder.gguf \
      --tokens InferenceWeb.Tests/Fixtures/EmbeddingTokenizer/snowflake-tokenization.json \
      --indices 0,1,2,3,7,8 --http-results path/to/results.json \
      --output path/to/numpy-reference.json

Requires NumPy and gguf-py. On Apple Silicon use NumPy >= 2.3.1 to include
Accelerate floating-point-status fixes. Token IDs come from saved fixtures.
Exact erf GELU is the default, matching the models' original GELU definition.
--gelu tanh is available to diagnose ggml's approximation; it is not necessary
for the primary acceptance gate. This is a small architecture/numerical check,
not an embedding-quality dataset, MTEB score, or performance benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


class Weights:
    def __init__(self, path: Path):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(str(path), mode="r")
        self.dequantize = dequantize
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}

    def metadata(self, name):
        return self.reader.fields[name].contents()

    def rows(self, name, row_ids):
        tensor = self.tensors[name]
        rows = int(np.prod(tensor.shape[1:], dtype=np.int64)) or 1
        raw = tensor.data.reshape(rows, -1)[row_ids]
        return np.asarray(self.dequantize(np.ascontiguousarray(raw), tensor.tensor_type), dtype=np.float32)

    def matrix(self, name):
        return self.rows(name, slice(None))

    def vector(self, name):
        return self.matrix(name).reshape(-1)

    def linear(self, hidden, name):
        # GGUF ne[0] is the input dimension: numpy rows are output channels.
        matrix = self.matrix(name + ".weight")
        output = hidden @ matrix.T
        del matrix
        if name + ".bias" in self.tensors:
            output += self.vector(name + ".bias")
        return output

    def layer_norm(self, hidden, name, epsilon):
        centered = hidden - np.mean(hidden, axis=-1, keepdims=True, dtype=np.float32)
        variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
        normalized = centered / np.sqrt(variance + np.float32(epsilon))
        return normalized * self.vector(name + ".weight") + self.vector(name + ".bias")


def gelu(hidden, mode):
    if mode == "tanh":
        root = np.float32(math.sqrt(2.0 / math.pi))
        return np.float32(0.5) * hidden * (np.float32(1) + np.tanh(
            root * (hidden + np.float32(0.044715) * hidden * hidden * hidden)))
    # Scalar libm erf provides an independent exact-GELU reference without SciPy.
    # Convert back to float32 before the activation multiply and all matmuls.
    scaled = hidden / np.float32(math.sqrt(2.0))
    error_function = np.fromiter((math.erf(float(x)) for x in scaled.flat),
                                dtype=np.float32, count=scaled.size).reshape(hidden.shape)
    return np.float32(0.5) * hidden * (np.float32(1) + error_function)


def encode(weights, token_rows, activation):
    dim = int(weights.metadata("bert.embedding_length"))
    heads = int(weights.metadata("bert.attention.head_count"))
    layers = int(weights.metadata("bert.block_count"))
    epsilon = float(weights.metadata("bert.attention.layer_norm_epsilon"))
    pooling = int(weights.metadata("bert.pooling_type"))
    if weights.metadata("general.architecture") != "bert" or dim % heads:
        raise ValueError("Only GGUF BERT/XLM-R with evenly sized attention heads is supported.")
    lengths = [len(ids) for ids in token_rows]
    offsets = np.cumsum([0] + lengths)
    head_dim = dim // heads
    # Only selected embedding rows are materialized; the full vocabulary is not.
    tokens = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in token_rows])
    positions = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    hidden = weights.rows("token_embd.weight", tokens) + weights.rows("position_embd.weight", positions)
    if "token_types.weight" in weights.tensors:
        hidden += weights.rows("token_types.weight", [0])
    hidden = weights.layer_norm(hidden, "token_embd_norm", epsilon)
    layer_seconds = []
    for layer in range(layers):
        started = time.monotonic()
        prefix = f"blk.{layer}."
        if prefix + "attn_qkv.weight" in weights.tensors:
            projected = weights.linear(hidden, prefix + "attn_qkv")
            query, key, value = np.split(projected, 3, axis=-1)
        else:
            query = weights.linear(hidden, prefix + "attn_q")
            key = weights.linear(hidden, prefix + "attn_k")
            value = weights.linear(hidden, prefix + "attn_v")
        attended = np.empty_like(hidden)
        for first, last in zip(offsets[:-1], offsets[1:]):
            length = int(last - first)
            q = query[first:last].reshape(length, heads, head_dim).transpose(1, 0, 2)
            k = key[first:last].reshape(length, heads, head_dim).transpose(1, 0, 2)
            v = value[first:last].reshape(length, heads, head_dim).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) * np.float32(1.0 / math.sqrt(head_dim))
            scores -= np.max(scores, axis=-1, keepdims=True)
            probability = np.exp(scores)
            probability /= np.sum(probability, axis=-1, keepdims=True, dtype=np.float32)
            attended[first:last] = (probability @ v).transpose(1, 0, 2).reshape(length, dim)
        del query, key, value
        hidden = weights.layer_norm(hidden + weights.linear(attended, prefix + "attn_output"),
                                    prefix + "attn_output_norm", epsilon)
        feed_forward = gelu(weights.linear(hidden, prefix + "ffn_up"), activation)
        hidden = weights.layer_norm(hidden + weights.linear(feed_forward, prefix + "ffn_down"),
                                    prefix + "layer_output_norm", epsilon)
        del attended, feed_forward
        if hidden.dtype != np.float32 or not np.isfinite(hidden).all():
            raise ValueError(f"Layer {layer} produced non-finite/non-FP32 hidden states.")
        layer_seconds.append(time.monotonic() - started)
        print(f"layer {layer + 1}/{layers}: {layer_seconds[-1]:.3f}s", flush=True)
    pooled = []
    for first, last in zip(offsets[:-1], offsets[1:]):
        if pooling == 1:
            vector = np.mean(hidden[first:last], axis=0, dtype=np.float32)
        elif pooling == 2:
            vector = hidden[first]
        elif pooling == 3:
            vector = hidden[last - 1]
        else:
            raise ValueError(f"Unsupported sentence pooling type {pooling}.")
        norm = math.sqrt(float(np.sum(vector.astype(np.float64) ** 2)))
        if not norm:
            raise ValueError("Encoder produced a zero-norm vector.")
        pooled.append((vector / norm).astype(np.float32))
    return np.stack(pooled), {
        "dimensions": dim, "heads": heads, "layers": layers, "layer_norm_epsilon": epsilon,
        "pooling": {1: "mean", 2: "CLS", 3: "last"}[pooling],
        "token_counts": lengths, "layer_seconds": layer_seconds,
    }


def cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--indices", default="0,1,2,3,7,8")
    parser.add_argument("--http-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--http-engine", choices=["tensorsharp", "llama"], default="tensorsharp")
    parser.add_argument("--gguf-py", type=Path)
    parser.add_argument("--gelu", choices=["exact", "tanh"], default="exact")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-absolute-error", type=float, default=0.005)
    parser.add_argument("--reference-vectors", type=Path,
                        help="Reuse saved oracle vectors for comparison-only runs; performs no model forward.")
    args = parser.parse_args()
    np.seterr(divide="raise", over="raise", invalid="raise")
    if args.gguf_py:
        sys.path.insert(0, str(args.gguf_py))
    fixtures = json.loads(args.tokens.read_text())
    if isinstance(fixtures, dict):
        fixtures = fixtures["cases"]
    selected = [fixtures[int(i)] for i in args.indices.split(",")]
    selected_tokens_hash = hashlib.sha256(json.dumps([item["tokens"] for item in selected]).encode()).hexdigest()
    results = json.loads(args.http_results.read_text())
    by_text = dict(zip(results["inputs"], results["engines"][args.http_engine]["correctness"]["vectors"]))
    expected_http = np.asarray([by_text[item["text"]] for item in selected], dtype=np.float64)
    started = time.monotonic()
    model_hash = sha256(args.model)
    if model_hash != results["model_sha256"]:
        raise ValueError("The HTTP results and oracle must use exactly the same GGUF SHA-256.")
    if args.reference_vectors:
        saved = np.load(args.reference_vectors, allow_pickle=False)
        vectors = saved["vectors"]
        if list(saved["texts"]) != [item["text"] for item in selected]:
            raise ValueError("Saved reference texts do not match selected token fixtures.")
        metadata = json.loads(str(saved["metadata"]))
        if str(saved["model_sha256"]) != model_hash or str(saved["gelu"]) != args.gelu:
            raise ValueError("Saved reference model or GELU does not match requested oracle.")
        if str(saved["selected_tokens_sha256"]) != selected_tokens_hash:
            raise ValueError("Saved reference token IDs do not match selected token fixtures.")
        creation_script_hash = str(saved["oracle_script_sha256"])
        vectors_path = args.reference_vectors
    else:
        vectors, metadata = encode(Weights(args.model), [item["tokens"] for item in selected], args.gelu)
        vectors_path = args.output.with_suffix(".vectors.npz")
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(vectors_path, vectors=vectors,
                            texts=np.asarray([item["text"] for item in selected]),
                            metadata=json.dumps(metadata), model_sha256=model_hash, gelu=args.gelu,
                            selected_tokens_sha256=selected_tokens_hash,
                            oracle_script_sha256=sha256(Path(__file__)))
        creation_script_hash = sha256(Path(__file__))
    if vectors.shape != expected_http.shape or not np.isfinite(vectors).all() or not np.isfinite(expected_http).all():
        raise ValueError("Oracle vector shape/values do not match the HTTP response.")
    comparisons = []
    for i, item in enumerate(selected):
        error = np.abs(vectors[i].astype(np.float64) - expected_http[i])
        agreement = cosine(vectors[i], expected_http[i])
        comparisons.append({"text": item["text"], "tokens": len(item["tokens"]),
                            "cosine": agreement, "max_absolute_error": float(error.max()),
                            "rmse": float(np.sqrt(np.mean(error * error))),
                            "passed": agreement >= args.min_cosine and float(error.max()) <= args.max_absolute_error})
    # Small retrieval sanity check: compare each query with the two saved code documents.
    document_indices = [i for i, item in enumerate(selected) if item["text"].startswith("def ")]
    retrieval = []
    for i, item in enumerate(selected):
        if not item["text"].startswith("query: "):
            continue
        oracle_scores = {str(j): cosine(vectors[i], vectors[j]) for j in document_indices}
        http_scores = {str(j): cosine(expected_http[i], expected_http[j]) for j in document_indices}
        oracle_order = sorted(document_indices, key=lambda j: oracle_scores[str(j)], reverse=True)
        http_order = sorted(document_indices, key=lambda j: http_scores[str(j)], reverse=True)
        retrieval.append({"query": item["text"], "document_indices": document_indices,
                          "numpy_scores": oracle_scores, "http_scores": http_scores,
                          "numpy_order": oracle_order, "http_order": http_order,
                          "same_order": oracle_order == http_order})
    report = {
        "purpose": "Independent mathematical BERT/XLM-R check on quantized GGUF weights; not MTEB or a speed benchmark.",
        "implementation": "NumPy FP32 matmul, per-sequence bidirectional attention, layernorm, GELU, full final layer, metadata pooling and L2; gguf-py used only for format/dequantization.",
        "model_file": str(args.model), "model_sha256": model_hash,
        "oracle_script_sha256": sha256(Path(__file__)),
        "reference_creation_script_sha256": creation_script_hash,
        "gguf_py": str(args.gguf_py) if args.gguf_py else "PYTHONPATH/import environment",
        "token_fixture": str(args.tokens), "token_fixture_sha256": sha256(args.tokens),
        "selected_tokens_sha256": selected_tokens_hash,
        "http_engine": args.http_engine,
        "http_results": str(args.http_results), "http_results_sha256": sha256(args.http_results),
        "reference_vectors": str(vectors_path), "reference_vectors_sha256": sha256(vectors_path),
        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "numpy_error_policy": np.geterr(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ["VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
        "gelu": args.gelu, "thresholds": {"min_cosine": args.min_cosine, "max_absolute_error": args.max_absolute_error},
        "metadata": metadata, "comparisons": comparisons, "retrieval": retrieval,
        "minimum_cosine": min(row["cosine"] for row in comparisons),
        "maximum_absolute_error": max(row["max_absolute_error"] for row in comparisons),
        "passed": all(row["passed"] for row in comparisons) and all(row["same_order"] for row in retrieval),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["minimum_cosine", "maximum_absolute_error", "passed", "elapsed_seconds"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
