#!/usr/bin/env python3
"""Create a fixture from complete payload-hash evidence and equivalent tokenizers.

Reads only bounded GGUF metadata, never model payloads. Full SHA256s must already
have been recorded in the supplied evidence; ResetLogitProbe rechecks every byte
before CUDA/model execution. Does not generate or download model weights.
"""
import argparse
import hashlib
import json
import pathlib
import struct


def sha256(data):
    return hashlib.sha256(data).hexdigest().upper()


def tokenizer_metadata(path):
    limit = 32 * 1024 * 1024
    with path.open("rb") as stream:
        def read(count):
            if count < 0 or stream.tell() + count > limit:
                raise ValueError("GGUF metadata exceeds 32 MiB budget")
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("Truncated GGUF metadata")
            return data

        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

        def string():
            return read(number("Q")).decode("utf-8")

        def value(kind):
            formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in formats:
                return number(formats[kind])
            if kind == 8:
                return string()
            if kind == 9:
                element, count = number("I"), number("Q")
                if count > 2_000_000 or element == 9:
                    raise ValueError("Unsupported GGUF metadata array")
                return [value(element) for _ in range(count)]
            raise ValueError("Unknown GGUF metadata type")

        if read(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError("Unsupported GGUF")
        number("Q")  # tensor count; tensor descriptors/payloads are not needed
        count = number("Q")
        if count > 100_000:
            raise ValueError("Excessive GGUF metadata count")
        result = {}
        for _ in range(count):
            name = string()
            item = value(number("I"))
            if name.startswith("tokenizer.ggml."):
                result[name] = item
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--payload-hashes", type=pathlib.Path, required=True)
    parser.add_argument("--input-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--input-model", type=pathlib.Path, required=True,
                        help="GGUF supplying the original fixture's tokenizer; headers only")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-lengths", default="32,64")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite fixture")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    hashes = json.loads(args.payload_hashes.read_text(encoding="utf-8"))
    source = json.loads(args.input_fixture.read_text(encoding="utf-8"))
    if not inventory["structurally_complete"] or not hashes["complete"]:
        raise ValueError("Model inventory or payload hashes are incomplete")
    if hashes["inventorySha256"] != sha256(args.inventory.read_bytes()):
        raise ValueError("Payload hash evidence belongs to another inventory")
    if len(hashes["files"]) != len(inventory["shards"]):
        raise ValueError("Payload hashes do not cover every shard")
    files = []
    canonical = "ResetLogitProbe model-files v1\n"
    for shard, evidence in zip(inventory["shards"], hashes["files"]):
        path = pathlib.Path(shard["path"])
        stat = path.stat()
        digest = evidence["Sha256"].upper()
        if (path.name != evidence["FileName"] or stat.st_size != shard["file_bytes"]
                or stat.st_size != evidence["Bytes"] or stat.st_mtime_ns != evidence["modifiedNs"]
                or len(digest) != 64 or any(c not in "0123456789ABCDEF" for c in digest)):
            raise ValueError("Shard differs from completed payload-hash evidence: " + str(path))
        files.append({"FileName": path.name, "Bytes": stat.st_size, "Sha256": digest})
        canonical += f"{path.name}\t{stat.st_size}\t{digest}\n"
    target_path = pathlib.Path(inventory["shards"][0]["path"])
    target_tokens = tokenizer_metadata(target_path)
    source_tokens = tokenizer_metadata(args.input_model)
    # Forward receives explicit IDs without padding or chat-template rendering.
    # A padding-ID difference is recorded and allowed only if neither ID occurs
    # in the selected histories. All other tokenizer fields must match exactly.
    differences = {k: {"source": source_tokens.get(k), "target": target_tokens.get(k)}
                   for k in sorted(target_tokens.keys() | source_tokens.keys())
                   if target_tokens.get(k) != source_tokens.get(k)}
    if set(differences) - {"tokenizer.ggml.padding_token_id"}:
        raise ValueError("Tokenizers differ; encode new model-specific inputs instead of reusing IDs: " + ", ".join(differences))
    vocab = len(target_tokens["tokenizer.ggml.tokens"])
    lengths = [int(n) for n in args.prompt_lengths.split(",")]
    if not lengths or len(lengths) != len(set(lengths)) or min(lengths) < 1 or args.steps < 1:
        raise ValueError("Invalid prompt lengths or step count")
    cases = []
    for length in lengths:
        candidates = sorted((c for c in source["Cases"] if len(c["PromptTokens"]) >= length
                             and len(c["ForcedTokens"]) >= args.steps), key=lambda c: len(c["PromptTokens"]))
        if not candidates or length + args.steps > args.context:
            raise ValueError("Input fixture or context cannot supply requested dimensions")
        case = candidates[0]
        prompt, forced = case["PromptTokens"][:length], case["ForcedTokens"][:args.steps]
        if any(not isinstance(t, int) or not 0 <= t < vocab for t in prompt + forced):
            raise ValueError("Input token outside target vocabulary")
        if differences:
            padding_ids = {source_tokens.get("tokenizer.ggml.padding_token_id"), target_tokens.get("tokenizer.ggml.padding_token_id")}
            if padding_ids.intersection(prompt + forced):
                raise ValueError("History includes a token affected by the padding-ID difference")
        cases.append({"Name": "p" + str(length), "PromptTokens": prompt, "ForcedTokens": forced})
    names = [case["Name"] for case in cases]
    sequence = [name for name in names for _ in range(2)] + [names[0]]
    tokenizer_json = json.dumps(target_tokens, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    fixture = {
        "Version": 1, "Steps": args.steps, "Context": args.context, "VocabSize": vocab,
        "ModelBytes": sum(f["Bytes"] for f in files), "ModelSha256": sha256(canonical.encode("utf-8")),
        "ModelFiles": files, "Sequence": sequence, "Cases": cases,
        "InputProvenance": {
            "Purpose": "Fixed input histories only; continuation tokens are inputs, not an output oracle.",
            "InputFixtureSha256": sha256(args.input_fixture.read_bytes()),
            "TokenizerMetadataEquivalent": not differences, "TokenizerMetadataSha256": sha256(tokenizer_json),
            "TokenizerDifferences": differences, "InputTokenMeaningEquivalent": True,
            "TokenizerComparison": "All tokenizer.ggml.* fields match except recorded unused padding IDs; token strings, types, merges, special IDs and pre-tokenizer match exactly.",
            "PayloadHashEvidenceSha256": sha256(args.payload_hashes.read_bytes()),
            "InventorySha256": sha256(args.inventory.read_bytes()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(fixture, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "modelSha256": fixture["ModelSha256"],
                      "fileCount": len(files), "vocabSize": vocab, "sequence": sequence,
                      "tokenizerMetadataSha256": sha256(tokenizer_json)}, indent=2))


if __name__ == "__main__":
    main()
