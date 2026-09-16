#!/usr/bin/env python3
"""Add missing synthetic BPE merges metadata without changing inference tensors.

The original fixture declares gpt2 but omits the required merges array. This
derived fixture uses the original synthetic token IDs and an empty merge table;
it is not a language tokenizer or a trained-model-quality fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from gguf import GGUFReader, GGUFWriter, GGUFValueType


class ExplicitEmptyStringArrayWriter(GGUFWriter):
    # The installed writer refuses every empty array even with an explicit
    # element type. Emit the standard typed zero-length encoding in this local
    # fixture helper; the dependency and all nonempty serialization are intact.
    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        if vtype == GGUFValueType.ARRAY and sub_type == GGUFValueType.STRING and len(val) == 0:
            prefix = self._pack("I", vtype) if add_vtype else b""
            return prefix + self._pack("I", GGUFValueType.STRING) + self._pack("Q", 0)
        return super()._pack_val(val, vtype, add_vtype, sub_type)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): parser.error("Output must be fresh; preserve prior fixture evidence")
    source_hash = sha(args.source)
    if source_hash != "f3917170dd3b8bf2b3220a1c41ff5a6dfd81517b2ae37079afbf5224d7dca99d":
        parser.error("Unexpected source fixture")
    source = GGUFReader(args.source)
    assert "tokenizer.ggml.merges" not in source.fields
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / args.source.name
    writer = ExplicitEmptyStringArrayWriter(output, source.fields["general.architecture"].contents())
    for name, field in source.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture": continue
        writer.add_key_value(name, field.contents(), field.types[0],
            field.types[1] if field.types[0] == GGUFValueType.ARRAY else None)
    writer.add_key_value("tokenizer.ggml.merges", [], GGUFValueType.ARRAY, GGUFValueType.STRING)
    for tensor in source.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    derived = GGUFReader(output)
    assert derived.fields["tokenizer.ggml.merges"].contents() == []
    assert len(source.tensors) == len(derived.tensors) == 142
    tensors = []
    for old, new in zip(source.tensors, derived.tensors):
        old_hash = hashlib.sha256(old.data.tobytes()).hexdigest()
        new_hash = hashlib.sha256(new.data.tobytes()).hexdigest()
        assert old.name == new.name and list(old.shape) == list(new.shape) and old.tensor_type == new.tensor_type
        assert old_hash == new_hash
        tensors.append(dict(name=old.name, shape=[int(n) for n in old.shape], dtype=old.tensor_type.name,
                            bytes=old.n_bytes, sha256=old_hash))
    sidecars = {}
    for name in ("deepseek41.config.json", "deepseek41.engram.bin"):
        src, dst = args.source.parent / name, args.output_dir / name
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        sidecars[name] = sha(dst)
    report = dict(schema_version=1, fixture=True, scope=__doc__, generator_sha256=sha(__file__),
        source=str(args.source.resolve()), source_sha256=source_hash, target=str(output.resolve()),
        target_sha256=sha(output), metadata_change={"tokenizer.ggml.merges": []},
        unchanged_tensor_count=len(tensors), tensors=tensors, sidecars=sidecars)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("tensors", "scope")}, indent=2))


if __name__ == "__main__": main()
