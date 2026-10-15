import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("attention_dump", Path(__file__).parents[1] / "compare-qwen4exp-attention-dumps.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class AttentionDumpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0

    def row(self, values=(1., 2.), *, tensor="q_pre_norm", kind="f32"):
        self.count += 1
        path = self.root / f"{self.count}.bin"
        data = struct.pack("<" + ("f" if kind == "f32" else "e") * len(values), *values)
        path.write_bytes(data)
        h = 1469598103934665603
        for b in data:
            h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
        return {"file": str(path), "tensor": tensor, "type": kind, "ne": [len(values), 1, 1, 1], "bytes": len(data), "hash": f"{h:016x}"}

    def test_equal_f32(self):
        result = MOD.compare(self.row(), self.row(), self.root)
        self.assertTrue(result["bytes_equal"])
        self.assertEqual(0, result["different_values"])

    def test_difference_coordinates_and_error(self):
        result = MOD.compare(self.row(), self.row((1., 3.)), self.root)
        self.assertFalse(result["bytes_equal"])
        self.assertEqual(1, result["max_abs"])
        self.assertEqual([1, 0, 0, 0], result["first_differences"][0]["index_ne0_first"])

    def test_mask_negative_infinity_valid_other_nonfinite_rejected(self):
        a = self.row((0., -math.inf), tensor="mask", kind="f16")
        self.assertTrue(MOD.compare(a, a, self.root)["bytes_equal"])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            MOD.decode(self.row((1., math.nan)), self.root)

    def test_changed_bytes_rejected(self):
        row = self.row()
        Path(row["file"]).write_bytes(b"\0" * row["bytes"])
        with self.assertRaisesRegex(ValueError, "native hash"):
            MOD.decode(row, self.root)

    def test_shape_and_path_escape_rejected(self):
        row = self.row()
        row["ne"] = [3, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, "shape"):
            MOD.decode(row, self.root)
        with self.assertRaisesRegex(ValueError, "directory"):
            MOD.decode(self.row(), self.root / "nested")

    def test_no_complete_layer_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, "correspondence"):
            MOD.analyze("", self.root)

    def log(self, *, omit_label=False, duplicate=False):
        text = "" if omit_label else "diagnostic serial rebuild: id=serial-2-0-0, row=0, step=22, position=1501, token=100\n"
        for path, call in (("solo", 0), ("arena", 1)):
            for name in MOD.NAMES:
                r = self.row(tensor=name)
                entry = f"[q4e-attn-diagnostic] path={path} call={call} layer=0 pos=1501 caller=0 slot=0 tensor={name} type=f32 ne=2,1,1,1 bytes=8 hash={r['hash']} file={r['file']}\n"
                text += entry
                if duplicate and name == "q_pre_norm":
                    text += entry
            if path == "solo":
                text += "[q4e-trace] span 0..1 T=1 pos=1501: 0:1.0@0000000000000001 1:2.0@0000000000000002\n"
        text += "[q4e-arena-trace] row=0 pos=1501 l=0 half=0 l2=1.0 hash=0000000000000001\n[q4e-arena-trace] row=0 pos=1501 l=0 half=1 l2=2.0 hash=0000000000000002\n"
        return text

    def test_labelled_invocation_and_residual_mapping(self):
        result = MOD.analyze(self.log(), self.root, layer=0, layers=1, rows=1)
        self.assertEqual(5, len(result["pairs"][0]["tensors"]))
        self.assertTrue(all(x["bytes_equal"] for x in result["pairs"][0]["tensors"]))

    def test_unlabelled_solo_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            MOD.analyze(self.log(omit_label=True), self.root, layer=0, layers=1, rows=1)
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            MOD.analyze(self.log(duplicate=True), self.root, layer=0, layers=1, rows=1)

    def test_completed_serial_span_cannot_lend_label_to_later_solo(self):
        text = self.log()
        span = "[q4e-trace] span 0..1 T=1 pos=1501: 0:1.0@0000000000000001 1:2.0@0000000000000002\n"
        text = text.replace(span, "")
        marker, rest = text.split("\n", 1)
        with self.assertRaisesRegex(ValueError, "Missing labelled"):
            MOD.analyze(marker + "\n" + span + rest, self.root, layer=0, layers=1, rows=1)

    def cross_logs(self):
        pp = "[q4e-trace] span 0..1 T=2 pos=0: 0:3.0@0000000000000003 1:4.0@0000000000000004\n"
        text = pp + self.log()
        source = "\n".join(x for x in text.splitlines() if "path=arena" not in x and "arena-trace" not in x)
        target = "\n".join(x for x in text.splitlines() if "path=solo" not in x)
        return source, target

    def test_cross_run_explicitly_matches_reference_before_comparing(self):
        source, target = self.cross_logs()
        result = MOD.analyze(target, self.root, layer=0, layers=1, rows=1,
            serial_text=source, serial_dump_root=self.root, serial_rows=[0])
        self.assertTrue(result["cross_run"])
        self.assertTrue(result["cross_run_residual_and_first_prefill_match"])
        self.assertEqual(5, len(result["pairs"][0]["tensors"]))
        with self.assertRaisesRegex(ValueError, "Missing labelled"):
            MOD.analyze(target, self.root, layer=0, layers=1, rows=1)

    def test_cross_run_rejects_different_serial_residual(self):
        source, target = self.cross_logs()
        source = source.replace("1:2.0@0000000000000002", "1:5.0@0000000000000005")
        with self.assertRaisesRegex(ValueError, "residuals differ"):
            MOD.analyze(target, self.root, layer=0, layers=1, rows=1,
                serial_text=source, serial_dump_root=self.root, serial_rows=[0])

    def test_cross_run_rejects_different_teacher_and_prefill(self):
        source, target = self.cross_logs()
        for changed, message in ((source.replace("token=100", "token=101"), "teacher token"),
                                 (source.replace("0:3.0@0000000000000003", "0:9.0@0000000000000009"), "first-prefill")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                MOD.analyze(target, self.root, layer=0, layers=1, rows=1,
                    serial_text=changed, serial_dump_root=self.root, serial_rows=[0])

    def test_cross_run_requires_explicit_valid_selection_and_directory(self):
        source, target = self.cross_logs()
        for kwargs in ({"serial_dump_root": self.root, "serial_rows": [0, 0]},
                       {"serial_dump_root": None, "serial_rows": [0]}):
            with self.assertRaises(ValueError):
                MOD.analyze(target, self.root, layer=0, layers=1, rows=1, serial_text=source, **kwargs)

    def test_invalid_tensor_preserves_partial_evidence_without_comparison_pass(self):
        text = self.log()
        _, arena = MOD.collect(text, position=1501, layer=0, serial_prefix="serial-2-0-")
        Path(arena[0]["mask"]["file"]).write_bytes(b"\0" * 8)
        result = MOD.analyze(text, self.root, layer=0, layers=1, rows=1)
        self.assertEqual("incomplete_invalid_tensor", result["status"])
        tensors = result["pairs"][0]["tensors"]
        self.assertEqual(4, sum(t["status"] == "compared" for t in tensors))
        self.assertEqual("invalid", tensors[-1]["status"])
        self.assertNotIn("bytes_equal", tensors[-1])

    def test_one_invocation_cannot_mix_callers_or_slots(self):
        text = self.log()
        for new in ("caller=1 slot=0 tensor=k_cache", "caller=0 slot=1 tensor=k_cache"):
            changed = text.replace("caller=0 slot=0 tensor=k_cache", new)
            with self.subTest(header=new), self.assertRaisesRegex(ValueError, "mixes caller or slot"):
                MOD.analyze(changed, self.root, layer=0, layers=1, rows=1)


if __name__ == "__main__":
    unittest.main()
