"""Coverage audit negatives use synthetic repositories, never model payloads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression.audit import audit_catalog


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "config").mkdir()
        self.config = self.repo / "config/tiny.json"
        self.config.write_text(json.dumps({"model": {"path": "../models/tiny.gguf",
            "urls": ["https://huggingface.co/owner/model/resolve/" + "a" * 40 + "/tiny.gguf"], "sha256": "b" * 64}}))
        (self.repo / "source.cs").write_text("// source evidence")
        self.catalog = {
            "schema_version": 1, "source_revision": "c" * 40, "note": "synthetic audit fixture",
            "inventory": {"config_glob": "config/**/*.json", "excluded": ["config/regression/**"]},
            "backends": [{"id": "ggml_cpu", "os": ["windows", "linux", "macos"], "requires": "native runtime"}],
            "families": [{"id": "test", "aliases": [], "backends": ["ggml_cpu"], "modalities": ["text"],
                "layout": "dense", "notes": "fixture", "source_evidence": ["source.cs"],
                "configuration_status": "configured", "presets": ["tiny"]}],
            "presets": [{"id": "tiny", "config": "config/tiny.json", "kind": "text", "family": "test",
                "modalities": ["text"], "backends": ["ggml_cpu"], "scenarios": ["cli_text"],
                "source_evidence": ["source.cs"], "layout": "dense", "status": "configured"}],
            "scenarios": [{"id": "cli_text", "entry_point": "TensorSharp.Cli", "status": "automated",
                "recipe": "fixed arithmetic fixture", "assertions": ["answer equals 4"],
                "evidence": ["source.cs:1"], "requirements": []}],
        }

    def test_valid_inventory_passes_without_reading_models(self):
        result = audit_catalog(self.repo, self.catalog)
        self.assertEqual(result["status"], "pass", result["errors"])
        self.assertEqual(result["config_count"], 1)
        self.assertEqual(result["inventory"][0]["file_count"], 1)
        self.assertFalse((self.repo / "models").exists())

    def test_new_config_cannot_disappear_from_coverage(self):
        (self.repo / "config/new.json").write_text('{"model":"other.gguf"}')
        result = audit_catalog(self.repo, self.catalog)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("new.json" in error for error in result["errors"]))

    def test_unknown_scenario_and_missing_evidence_fail(self):
        self.catalog["presets"][0]["scenarios"] = ["missing"]
        self.catalog["scenarios"][0]["evidence"] = ["does-not-exist.cs"]
        result = audit_catalog(self.repo, self.catalog)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("unknown scenario" in error for error in result["errors"]))
        self.assertTrue(any("missing source evidence" in error for error in result["errors"]))

    def test_duplicate_ids_fail(self):
        self.catalog["presets"].append(copy.deepcopy(self.catalog["presets"][0]))
        self.catalog["scenarios"].append(copy.deepcopy(self.catalog["scenarios"][0]))
        self.assertEqual(audit_catalog(self.repo, self.catalog)["status"], "fail")

    def test_config_escape_is_refused(self):
        outside = self.repo / "outside.json"
        outside.write_text('{"model":"outside.gguf"}')
        self.catalog["presets"][0]["config"] = "config/../outside.json"
        result = audit_catalog(self.repo, self.catalog)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("within config/" in error for error in result["errors"]))

    def test_scenario_ids_cannot_escape_artifact_directory(self):
        self.catalog["scenarios"][0]["id"] = "../../outside"
        self.catalog["presets"][0]["scenarios"] = ["../../outside"]
        self.assertEqual(audit_catalog(self.repo, self.catalog)["status"], "fail")

    def test_unknown_backend_cannot_be_claimed_as_supported(self):
        self.catalog["presets"][0]["backends"] = ["invented_gpu"]
        self.assertEqual(audit_catalog(self.repo, self.catalog)["status"], "fail")

    def test_sizes_and_budget_metadata_are_validated(self):
        for value in (-1, 0, True, "100"):
            with self.subTest(value=value):
                catalog = copy.deepcopy(self.catalog)
                catalog["presets"][0]["file_sizes"] = {"tiny.gguf": value}
                self.assertEqual(audit_catalog(self.repo, catalog)["status"], "fail")
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(resource=value):
                catalog = copy.deepcopy(self.catalog)
                catalog["presets"][0]["resource_requirements"] = {"ram_gib": value, "basis": "fixture"}
                self.assertEqual(audit_catalog(self.repo, catalog)["status"], "fail")

    def test_malformed_config_and_unverified_download_are_visible(self):
        self.config.write_text('{"model":{"path":"x", "sha256":"bad"}}')
        self.assertEqual(audit_catalog(self.repo, self.catalog)["status"], "fail")
        self.config.write_text('{"model":{"path":"x", "urls":["https://example.com/x"]}}')
        result = audit_catalog(self.repo, self.catalog)
        self.assertTrue(any("SHA256" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
