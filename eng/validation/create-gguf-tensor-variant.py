#!/usr/bin/env python3
"""Create an explicitly labelled diagnostic GGUF with selected donor tensors.

This changes the checkpoint, not the inference engine. A passing variant is
never evidence that the original quantization or production model is repaired.
Inputs are immutable and SHA-pinned. Requires the unmodified gguf Python package.
"""
import argparse
import hashlib
import json
from pathlib import Path
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_tensor(tensor):
    # Hash the mapped buffer directly, without duplicating a large embedding.
    return hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()


class TypedArrayWriter(GGUFWriter):
    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        if vtype == GGUFValueType.ARRAY and len(val) == 0:
            require(sub_type is not None and sub_type != GGUFValueType.ARRAY, "Untyped/nested empty array")
            prefix = self._pack("I", vtype) if add_vtype else b""
            return prefix + self._pack("I", sub_type) + self._pack("Q", 0)
        return super()._pack_val(val, vtype, add_vtype, sub_type)


def create(source_path, donor_path, source_hash, donor_hash, selected, output_dir):
    require(not output_dir.exists(), "Use a fresh output directory; preserve earlier evidence")
    require(selected and len(set(selected)) == len(selected), "Select distinct tensor names")
    require(sha_file(source_path) == source_hash.lower(), "Source SHA256 mismatch")
    require(sha_file(donor_path) == donor_hash.lower(), "Donor SHA256 mismatch")
    source, donor = GGUFReader(source_path), GGUFReader(donor_path)
    architecture = source.fields["general.architecture"].contents()
    require(donor.fields["general.architecture"].contents() == architecture, "Architecture mismatch")
    require(source.endianess == donor.endianess, "Byte order mismatch")
    for reader in (source, donor):
        splits = reader.fields.get("split.count")
        require(splits is None or splits.contents() == 1, "Only complete, single-file checkpoints are supported")
    contract = lambda reader: {name: (field.types, field.contents()) for name, field in reader.fields.items()
        if name.startswith((architecture + ".", "tokenizer."))}
    require(contract(source) == contract(donor), "Architecture parameters or tokenizer/template differ")
    old = {t.name: t for t in source.tensors}
    donors = {t.name: t for t in donor.tensors}
    require(set(old) == set(donors) and set(selected) <= set(old), "Tensor inventory mismatch")
    for name, tensor in old.items():
        require(list(tensor.shape) == list(donors[name].shape), "Tensor shape mismatch: " + name)
        if tensor.tensor_type == donors[name].tensor_type == GGMLQuantizationType.F32:
            require(sha_tensor(tensor) == sha_tensor(donors[name]), "Unquantized parameter mismatch: " + name)
    output_dir.mkdir(parents=True)
    output = output_dir / "diagnostic-tensor-variant.gguf"
    writer = TypedArrayWriter(output, architecture, endianess=source.endianess)
    # Metadata is copied with its original type below; preserve the matching
    # physical padding too, without inserting a duplicate alignment field.
    writer.data_alignment = source.alignment
    for name, field in source.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.name"):
            continue
        require(len(field.types) <= 2, "Nested metadata arrays are unsupported")
        writer.add_key_value(name, field.contents(), field.types[0],
            field.types[1] if field.types[0] == GGUFValueType.ARRAY else None)
    writer.add_name("DIAGNOSTIC tensor variant; see adjacent manifest.json")
    for tensor in source.tensors:
        chosen = donors[tensor.name] if tensor.name in selected else tensor
        writer.add_tensor(chosen.name, chosen.data, raw_dtype=chosen.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    derived = GGUFReader(output)
    require(contract(derived) == contract(source), "Derived metadata contract changed")
    require(len(derived.tensors) == len(source.tensors), "Derived tensor count changed")
    tensors = []
    for before, after in zip(source.tensors, derived.tensors):
        chosen = donors[before.name] if before.name in selected else before
        digest = sha_tensor(chosen)
        require(after.name == chosen.name and list(after.shape) == list(chosen.shape)
            and after.tensor_type == chosen.tensor_type and sha_tensor(after) == digest,
            "Derived payload differs from selected input: " + before.name)
        tensors.append(dict(name=before.name, donor=before.name in selected,
            original_type=before.tensor_type.name, type=after.tensor_type.name,
            shape=[int(v) for v in after.shape], bytes=int(after.n_bytes), sha256=digest))
    report = dict(complete=True, diagnostic_only=True, original_model_fixed=False,
        source=str(source_path.resolve()), source_sha256=source_hash.lower(),
        donor=str(donor_path.resolve()), donor_sha256=donor_hash.lower(), selected=selected,
        output=str(output.resolve()), output_sha256=sha_file(output), output_bytes=output.stat().st_size,
        generator_sha256=sha_file(Path(__file__)), tensors=tensors,
        qualification="Exact selected tensor substitution and unchanged architecture/tokenizer. No inference or semantic quality claim.")
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "donor", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--donor-sha256", required=True)
    parser.add_argument("--tensor", action="append", required=True)
    args = parser.parse_args()
    result = create(args.source, args.donor, args.source_sha256, args.donor_sha256, args.tensor, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, indent=2))
