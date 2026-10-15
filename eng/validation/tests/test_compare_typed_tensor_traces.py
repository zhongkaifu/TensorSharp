"""Independent packed fixtures; no native backend, model or GPU execution."""
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("typed_trace", Path(__file__).parents[1] / "compare-typed-tensor-traces.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def native_checksum(data):
    # Independent producer fixture, also checked against published FNV vectors.
    checksum = 0xcbf29ce484222325
    for octet in data:
        checksum ^= octet
        checksum *= 0x100000001b3
        checksum %= 0x10000000000000000
    return format(checksum, "016x")


class Capture:
    def __init__(self, directory, *, nt=3, rows=2, p0=0, endian="little", records=None, overrides=None):
        self.root = directory
        self.root.mkdir()
        self.pin_path = directory.with_suffix(".pins.json")
        self.meta = {"schema": "tensorsharp.glm-trace.v1", "pass_id": 0, "model_id": 1,
            "kind": "trunk", "p0": p0, "nt": nt, "n_kv": nt + p0, "slot_id": 0,
            "sequences": [{"slot_id": 0, "position": p0}], "rows_limit": rows,
            "max_bytes": 1 << 20, "endian": endian, "token_axis_overrides": overrides or {},
            "scope": "Independent fixture, not physical-memory validation", "traces": []}
        records = records if records is not None else [{"values": [1., 2., 3., 4.]}]
        for index, settings in enumerate(records):
            typ = settings.get("type", "f32")
            ne = settings.get("ne", [2, nt, 1, 1])
            nb = settings.get("nb", [4, 16, 16 * nt, 16 * nt])
            axis = settings.get("axis", 1)
            tag = settings.get("tag", "activation" + str(index))
            shape = ne.copy(); shape[axis] = min(nt, rows)
            data = struct.pack(("<" if endian == "little" else ">") + ("f" if typ == "f32" else "i") * len(settings["values"]), *settings["values"])
            row = {"ordinal": index, "tag": tag, "name": tag + ".tensor", "layer": settings.get("layer", 0), "rank": 0,
                "type": typ, "type_code": 0 if typ == "f32" else 26, "ne": ne, "nb": nb,
                "source_bytes": 4 + sum((n - 1) * stride for n, stride in zip(ne, nb)),
                "candidate_token_axes": [d for d, n in enumerate(ne) if n == nt], "token_axis": axis,
                "token_axis_source": "override" if tag in self.meta["token_axis_overrides"] else "unique-extent",
                "shape": shape, "slice_start": 0, "token_count": min(nt, rows),
                "logical_elements": len(settings["values"]), "bytes": len(data),
                "file": f"pass-000000.tensor-{index:04d}.bin", "content_fnv1a64": native_checksum(data)}
            self.meta["traces"].append(row)
            (self.root / row["file"]).write_bytes(data)
        self.meta["total_bytes"] = sum(r["bytes"] for r in self.meta["traces"])
        self.pin = {"schema": "tensorsharp.tensor-trace-pins.v1", "capture_root": str(self.root.resolve()),
            "fresh_directory": True, "controller_exit_code": 0, "producer_exit_code": 1,
            "expected_producer_exit_code": 1, "producer_binary_sha256": "1" * 64,
            "producer_build_receipt_sha256": "2" * 64, "model_identity_sha256": "3" * 64,
            "probe_report_sha256": "4" * 64, "execution_receipt_sha256": "5" * 64,
            "input_ids": list(range(p0 + nt + 5))}
        self.save()

    def save(self):
        (self.root / "pass-000000.json").write_text(json.dumps(self.meta), encoding="utf-8")
        self.repin()

    def repin(self):
        self.pin["files"] = {p.name: digest(p.read_bytes()) for p in self.root.iterdir() if p.is_file()}
        self.pin_path.write_text(json.dumps(self.pin), encoding="utf-8")

    def read(self, **kwargs):
        return TRACE.read_capture(self.root, self.pin_path, **kwargs)

    def replace_values(self, ordinal, values, fmt="f"):
        row = self.meta["traces"][ordinal]
        raw = struct.pack("<" + fmt * len(values), *values)
        (self.root / row["file"]).write_bytes(raw)
        row["content_fnv1a64"] = native_checksum(raw)
        self.save()


class TypedTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.counter = 0

    def fixture(self, **kwargs):
        self.counter += 1
        return Capture(self.base / f"capture-{self.counter}", **kwargs)

    def pair(self, **kwargs):
        return self.fixture(**kwargs), self.fixture(**kwargs)

    def compare(self, a, b, prefix=2):
        return TRACE.compare_captures(a.read(), b.read(), prefix_tokens=prefix)

    def test_standard_fnv_vectors(self):
        for data, expected in [(b"", "cbf29ce484222325"), (b"a", "af63dc4c8601ec8c"), (b"foobar", "85944171f73967e8")]:
            self.assertEqual(expected, native_checksum(data))
            self.assertEqual(expected, TRACE.fnv1a64(data))

    def test_original_strides_do_not_reinterpret_packed_payload(self):
        f = self.fixture(records=[{"ne": [2, 3, 1, 1], "nb": [16, 4, 40, 40], "values": [1., 2., 3., 4.]}])
        actual = f.read()["selected"]["rows"][0]["values"]
        self.assertEqual([[1., 3.], [2., 4.]], actual[:, :, 0, 0].tolist())

    def test_different_widths_compare_only_actual_common_512_prefix(self):
        a = self.fixture(nt=512, rows=512, records=[{"values": list(range(1024))}])
        b = self.fixture(nt=774, rows=774, records=[{"values": list(range(1024)) + [-99.] * 524}])
        b.pin["input_ids"][512:] = [987] * (len(b.pin["input_ids"]) - 512); b.repin()
        report = self.compare(a, b, 512)
        self.assertEqual("matched", report["status"])
        self.assertEqual(512, report["verified_input_history_extent"])
        self.assertEqual([{"p0": 0, "nt": 512}, {"p0": 0, "nt": 774}], report["capture_pass_extents"])
        self.assertEqual(1024, report["observations"][0]["elements"])
        self.assertIsNone(report["numerical_quality_passed"])
        self.assertIsNone(report["semantic_quality_passed"])
        self.assertIsNone(report["performance_accepted"])

    def test_endian_normalization(self):
        a = self.fixture(endian="little")
        b = self.fixture(endian="big")
        self.assertEqual("matched", self.compare(a, b)["status"])

    def test_localizes_registration_order_and_i32_ids_with_coordinates(self):
        records = [{"tag": "embedding", "values": [1., 2., 3., 4.]},
            {"tag": "router", "layer": 7, "values": [1., 2., 3., 4.]},
            {"tag": "selected", "type": "i32", "layer": 7, "values": [10, 20, 11, 21]}]
        a, b = self.pair(p0=4, records=records)
        b.replace_values(1, [1., 2., 3., 6.])
        b.replace_values(2, [10, 20, 12, 21], "i")
        report = self.compare(a, b)
        first = report["first_divergent_observation"]
        self.assertEqual(("router", 7, 1), (first["tag"], first["layer"], first["reference_ordinal"]))
        self.assertEqual(1., first["rmse"])
        self.assertEqual(2., first["max_absolute"])
        ids = report["first_differing_i32_observation"]
        self.assertFalse(ids["exact_ids_equal"])
        self.assertEqual([0, 1, 0, 0], ids["first_differences"][0]["index_ne0_first"])
        self.assertEqual(5, ids["first_differences"][0]["absolute_token"])

    def test_axis_zero_i32_and_negative_ids_are_exact(self):
        records = [{"type": "i32", "ne": [3, 2, 1, 1], "axis": 0, "nb": [16, 4, 40, 40], "values": [-1, 2, 3, 4]}]
        a, b = self.pair(records=records)
        b.replace_values(0, [-1, 8, 3, 4], "i")
        item = self.compare(a, b)["observations"][0]
        self.assertEqual([1, 0, 0, 0], item["first_differences"][0]["index_ne0_first"])
        self.assertEqual(1, item["first_differences"][0]["absolute_token"])

    def test_signed_zero_observed_without_inventing_numerical_error(self):
        a, b = self.pair(records=[{"values": [0., 0., 0., 0.]}])
        b.replace_values(0, [-0., 0., 0., 0.])
        report = self.compare(a, b)
        item = report["observations"][0]
        self.assertEqual("localized", report["status"])
        self.assertEqual(1, item["bitwise_differences"])
        self.assertEqual(0, item["numeric_differences"])
        self.assertEqual(0., item["relative_l2"])
        json.dumps(report, allow_nan=False)

    def test_ambiguous_token_axis_requires_declared_override(self):
        record = {"ne": [3, 3, 1, 1], "values": [1., 2., 3., 4., 5., 6.]}
        f = self.fixture(records=[record])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            f.read()
        f.meta["token_axis_overrides"] = {"activation0": 1}
        f.meta["traces"][0]["token_axis_source"] = "override"; f.save()
        f.read()
        f.meta["token_axis_overrides"]["activation0"] = 0; f.save()
        with self.assertRaisesRegex(ValueError, "override"):
            f.read()

    def test_malformed_tensor_metadata_is_rejected(self):
        alterations = {"ordinal": 2, "ne": [2, 3, 1], "nb": [4, 2, 48, 48], "source_bytes": 4,
            "shape": [2, 1, 1, 1], "logical_elements": 3, "bytes": 12,
            "candidate_token_axes": [True], "token_axis": 0, "token_count": 1, "slice_start": 1,
            "type_code": 1, "rank": True, "layer": -2, "type": "f64"}
        for field, value in alterations.items():
            with self.subTest(field=field):
                f = self.fixture(); f.meta["traces"][0][field] = value; f.save()
                with self.assertRaises(ValueError):
                    f.read()

    def test_independent_snapshot_metadata_is_typed_and_complete(self):
        f = self.fixture()
        row = f.meta["traces"][0]
        snapshot = {"kind": "independent_contiguous_snapshot", "layer": 0, "rank": 0,
            "tensor_name": "", "op": "CONT", "ne": row["ne"].copy(), "nb": [4, 8, 24, 24], "source_bytes": 24}
        row.update(snapshot_checked_elements=4, snapshot_source=snapshot); f.save()
        loaded = f.read()
        self.assertEqual(4, loaded["selected"]["rows"][0]["metadata"]["snapshot_checked_elements"])
        for field, value in [("snapshot_checked_elements", 3), ("snapshot_source", None)]:
            with self.subTest(field=field):
                other = self.fixture(); other.meta["traces"][0].update(copy.deepcopy(row))
                other.meta["traces"][0][field] = value; other.save()
                with self.assertRaisesRegex(ValueError, "Snapshot check"):
                    other.read()
        for field, value in [("ne", [2, 2, 1, 1]), ("nb", [4, 2, 24, 24]), ("source_bytes", 4),
                             ("op", 35), ("tensor_name", None), ("rank", True)]:
            with self.subTest(source_field=field):
                other = self.fixture(); other.meta["traces"][0].update(copy.deepcopy(row))
                other.meta["traces"][0]["snapshot_source"][field] = value; other.save()
                with self.assertRaises(ValueError):
                    other.read()

    def test_capture_context_is_explicit_and_sequences_nonempty(self):
        for field, value in [("p0", "missing"), ("sequences", []), ("nt", 0), ("slot_id", True),
                             ("total_bytes", 12), ("max_bytes", 12), ("endian", "native")]:
            with self.subTest(field=field):
                f = self.fixture()
                if value == "missing": del f.meta[field]
                else: f.meta[field] = value
                f.save()
                with self.assertRaises(ValueError): f.read()

    def test_nonfinite_logical_f32_values_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value):
                f = self.fixture(); f.replace_values(0, [1., 2., value, 4.])
                with self.assertRaisesRegex(ValueError, "Nonfinite F32"):
                    f.read()

    def test_truncated_oversized_and_unpinned_payloads_rejected(self):
        for changed, message in [(b"123", "Truncated"), (b"0" * 20, "oversized")]:
            f = self.fixture(); (f.root / f.meta["traces"][0]["file"]).write_bytes(changed); f.repin()
            with self.assertRaisesRegex(ValueError, message):
                f.read()
        f = self.fixture(); (f.root / f.meta["traces"][0]["file"]).write_bytes(b"0" * 16)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            f.read()

    def test_native_checksum_independently_verified_after_sha_pin(self):
        f = self.fixture(); f.meta["traces"][0]["content_fnv1a64"] = "0" * 16; f.save()
        with self.assertRaisesRegex(ValueError, "FNV1a64"):
            f.read()

    def test_path_traversal_absolute_drive_and_separator_rejected(self):
        for name in ["../outside.bin", "/tmp/other.bin", "C:/data.bin", "..\\outside.bin", "pass-000000.tensor-0000.bin:stream"]:
            with self.subTest(name=name):
                f = self.fixture(); f.meta["traces"][0]["file"] = name; f.save()
                with self.assertRaisesRegex(ValueError, "filename"):
                    f.read()

    def test_incomplete_temporary_and_orphan_files_rejected(self):
        for name, message in [("pass-000000.incomplete", "Incomplete"), ("data.tmp", "temporary"),
                              ("pass-000001.tensor-0000.bin", "Orphan")]:
            with self.subTest(name=name):
                f = self.fixture(); (f.root / name).write_bytes(b"1234"); f.repin()
                with self.assertRaisesRegex(ValueError, message):
                    f.read()

    def test_duplicate_registration_identity_and_order_rejected(self):
        f = self.fixture(records=[{"tag": "same", "values": [1., 2., 3., 4.]}] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate trace"):
            f.read()

    def test_duplicate_json_keys_and_nonfinite_extra_fields_rejected(self):
        for raw in [b'{"schema":"a","schema":"b"}', b'{"unused":NaN}', b'{"unused":1e999}', b'[]', b'\xff']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                TRACE.load_json(raw)

    def test_run_pin_requires_execution_identity_and_actual_expected_exit(self):
        mutations = {"fresh_directory": False, "controller_exit_code": 1, "producer_exit_code": 0,
            "expected_producer_exit_code": True, "probe_report_sha256": None, "execution_receipt_sha256": "a",
            "producer_binary_sha256": "B" * 64, "model_identity_sha256": "x", "capture_root": "other",
            "input_ids": [True], "files": {}}
        for key, value in mutations.items():
            with self.subTest(field=key):
                f = self.fixture(); f.pin[key] = value
                f.pin_path.write_text(json.dumps(f.pin), encoding="utf-8")
                with self.assertRaises(ValueError):
                    f.read()

    def test_external_pin_digest_and_location(self):
        f = self.fixture()
        actual = digest(f.pin_path.read_bytes())
        self.assertTrue(f.read(expected_pin_sha256=actual)["external_pin_sha256_verified"])
        with self.assertRaisesRegex(ValueError, "run-pin SHA256"):
            f.read(expected_pin_sha256="a" * 64)
        inside = f.root / "pin.json"; inside.write_bytes(f.pin_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "outside"):
            TRACE.read_capture(f.root, inside)

    def test_joint_and_mismatched_contexts_history_model_refused(self):
        for mutation in ["joint", "p0", "kind", "sequence", "history", "model", "input_short"]:
            with self.subTest(mutation=mutation):
                a, b = self.pair()
                if mutation == "joint": b.meta["p0"] = None; b.meta["kind"] = "batched"
                elif mutation == "p0": b.meta["p0"] = 1
                elif mutation == "kind": b.meta["kind"] = "mtp"
                elif mutation == "sequence": b.meta["sequences"][0]["position"] = 1
                elif mutation == "history": b.pin["input_ids"][1] = 123
                elif mutation == "model": b.pin["model_identity_sha256"] = "8" * 64
                else: b.pin["input_ids"] = [0]
                b.save()
                with self.assertRaises(ValueError): self.compare(a, b)

    def test_native_shaped_joint_zero_extent_has_explicit_rows_but_cannot_prefix_compare(self):
        records = [{"type": "i32", "tag": "moe_sel", "ne": [2, 3, 1, 1],
                    "nb": [4, 16, 48, 48], "values": [0, 1, 1, 2, 2, 3]}]
        f = self.fixture(nt=3, rows=4, records=records, overrides={"moe_sel": 1})
        f.meta.update(kind="batched", p0=None, n_kv=0, slot_id=-1,
                      sequences=[{"slot_id": slot, "position": 4} for slot in range(3)])
        f.save()
        capture = f.read()
        self.assertEqual(0, capture["selected"]["metadata"]["n_kv"])
        self.assertEqual(3, len(capture["selected"]["metadata"]["sequences"]))
        with self.assertRaisesRegex(ValueError, "explicit row mapping"):
            TRACE.compare_captures(capture, capture, prefix_tokens=3)
        # The row identity is slot plus position; multiple distinct positions
        # from one slot are permitted and are not collapsed into one KV extent.
        f.meta["sequences"] = [{"slot_id": 0, "position": p} for p in range(4, 7)]
        f.save(); f.read()

    def test_zero_kv_extent_is_not_permitted_without_valid_joint_row_metadata(self):
        variants = [
            {"kind": "trunk"}, {"kind": "mtp"}, {"n_kv": -1}, {"n_kv": False},
            {"p0": 0}, {"sequences": []},
            {"sequences": [{"slot_id": 0, "position": 4}]},
            {"sequences": [{"slot_id": 0, "position": 4}] * 3},
            {"sequences": [{"slot_id": -1, "position": 4}, {"slot_id": 1, "position": 4}, {"slot_id": 2, "position": 4}]},
            {"sequences": [{"slot_id": 0, "position": False}, {"slot_id": 1, "position": 4}, {"slot_id": 2, "position": 4}]},
        ]
        for mutation in variants:
            with self.subTest(mutation=mutation):
                f = self.fixture()
                f.meta.update(kind="batched", p0=None, n_kv=0, slot_id=-1,
                              sequences=[{"slot_id": slot, "position": 4} for slot in range(3)])
                f.meta.update(mutation); f.save()
                with self.assertRaises(ValueError):
                    f.read()

    def test_inadequate_dump_prefix_is_not_silently_compared(self):
        a, b = self.pair()
        with self.assertRaisesRegex(ValueError, "prefix extent"):
            self.compare(a, b, 3)

    def test_missing_or_reordered_stages_are_incomplete_not_localized(self):
        records = [{"tag": "a", "values": [1., 2., 3., 4.]}, {"tag": "b", "values": [1., 2., 3., 4.]}]
        a = self.fixture(records=records)
        for other in [records[:1], list(reversed(records))]:
            b = self.fixture(records=other)
            report = self.compare(a, b)
            self.assertEqual("incomplete", report["status"])
            self.assertFalse(report["diagnostic_complete"])
            self.assertIsNone(report["first_divergent_observation"])

    def test_capture_mutation_and_inventory_changes_detected_after_read(self):
        for change in ["data", "pin", "inventory"]:
            with self.subTest(change=change):
                f = self.fixture(); capture = f.read()
                if change == "data": (f.root / f.meta["traces"][0]["file"]).write_bytes(b"0" * 16)
                elif change == "pin": f.pin_path.write_text("{}", encoding="utf-8")
                else: (f.root / "extra").write_text("extra", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    TRACE.unchanged(capture)

    def test_bound_checks_before_large_reads(self):
        f = self.fixture()
        with self.assertRaisesRegex(ValueError, "aggregate byte bound"):
            f.read(maximum_capture_bytes=4)
        with self.assertRaisesRegex(ValueError, "parser bound"):
            f.read(maximum_tensor_bytes=4)

    def test_multiple_complete_passes_require_explicit_selection(self):
        f = self.fixture()
        meta = copy.deepcopy(f.meta); meta["pass_id"] = 1
        row = meta["traces"][0]; old = row["file"]; row["file"] = "pass-000001.tensor-0000.bin"
        (f.root / row["file"]).write_bytes((f.root / old).read_bytes())
        (f.root / "pass-000001.json").write_text(json.dumps(meta), encoding="utf-8"); f.repin()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            f.read()
        self.assertEqual("pass-000001.json", f.read(manifest_name="pass-000001.json")["selected"]["file"])

    def test_symlink_payload_is_not_a_regular_capture_file(self):
        f = self.fixture(); original = f.root / f.meta["traces"][0]["file"]
        target = self.base / "external.bin"; target.write_bytes(original.read_bytes()); original.unlink()
        try:
            original.symlink_to(target)
        except OSError as error:
            self.skipTest("OS did not allow the independent symlink fixture: " + str(error))
        f.repin()
        with self.assertRaisesRegex(ValueError, "regular file"):
            f.read()

    def test_cli_diagnostic_exit_and_immutable_output(self):
        a, b = self.pair()
        output = self.base / "comparison.json"
        command = ["compare-typed-tensor-traces.py", "--reference", str(a.root), "--reference-pin", str(a.pin_path),
            "--candidate", str(b.root), "--candidate-pin", str(b.pin_path), "--prefix-tokens", "2", "--output", str(output)]
        with patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, TRACE.main())
        original = output.read_bytes()
        report = json.loads(original)
        self.assertEqual("matched", report["status"])
        self.assertEqual(1, report["sources"][0]["run_bindings"]["producer_exit_code"])
        with patch("sys.argv", command), self.assertRaisesRegex(ValueError, "Never overwrite"):
            TRACE.main()
        self.assertEqual(original, output.read_bytes())
        command[-1] = str(a.root / "new.json")
        with patch("sys.argv", command), self.assertRaisesRegex(ValueError, "immutable"):
            TRACE.main()
        self.assertFalse((a.root / "new.json").exists())

    def test_cli_invalid_report_does_not_claim_quality_or_success(self):
        a, b = self.pair(); b.pin["producer_exit_code"] = 99; b.repin()
        output = self.base / "invalid.json"
        command = ["compare-typed-tensor-traces.py", "--reference", str(a.root), "--reference-pin", str(a.pin_path),
            "--candidate", str(b.root), "--candidate-pin", str(b.pin_path), "--output", str(output)]
        with patch("sys.argv", command), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(2, TRACE.main())
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual("invalid", report["status"])
        self.assertIsNone(report["numerical_quality_passed"])


if __name__ == "__main__":
    unittest.main()
