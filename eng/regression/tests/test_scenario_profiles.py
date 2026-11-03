"""Offline profile contract, resource and preparation regression tests."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import scenario_profiles as profiles
from regression import runner


class ProfileContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "synthetic input.txt"
        self.fixture.write_bytes(b"only synthetic fixture")
        self.path = self.root / "profiles.json"
        self.profile = {"id": "local-example", "scenario": "retrieval", "preset": "tiny", "backend": "cpu",
                        "models": {"embedding": "encoder"},
                        "fixtures": {"document": {"path": self.fixture.name, "sha256": hashlib.sha256(self.fixture.read_bytes()).hexdigest()}},
                        "parameters": {}, "expected": {}, "resources": {"ram_required_gib": 4}}
        self.catalog = {"presets": [{"id": "tiny"}, {"id": "encoder"}], "scenarios": [{"id": "retrieval"}]}
        self.module = Mock()
        self.mock = patch.object(profiles, "adapter", return_value=self.module)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def load(self, *items):
        self.path.write_text(json.dumps({"schema_version": 1, "profiles": list(items or [self.profile])}))
        return profiles.load_profiles(self.path, self.catalog)

    def test_relative_paths_spaces_and_checksum(self):
        profile = self.load()[0]
        self.assertEqual(str(self.fixture.resolve()), profile["fixtures"]["document"]["path"])
        identity, missing = profiles.verify_fixtures(profile, hash_files=True)
        self.assertFalse(missing)
        self.assertTrue(identity["document"]["verified"])
        self.module.validate_profile.assert_called_once()

    def test_plan_does_not_hash_large_payload(self):
        profile = self.load()[0]
        with patch.object(profiles, "file_hash", side_effect=AssertionError("payload hash during plan")):
            identity, missing = profiles.verify_fixtures(profile, hash_files=False)
        self.assertFalse(missing)
        self.assertFalse(identity["document"]["verified"])

    def test_changed_and_empty_fixture_rejected(self):
        profile = self.load()[0]
        self.fixture.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            profiles.verify_fixtures(profile, hash_files=True)
        self.fixture.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            profiles.verify_fixtures(profile, hash_files=False)

    def test_missing_fixture_is_explicit_requirement(self):
        profile = self.load()[0]
        self.fixture.unlink()
        identity, missing = profiles.verify_fixtures(profile, hash_files=True)
        self.assertFalse(identity)
        self.assertIn("document:", missing[0])

    def test_unknown_fields_or_model_ids_rejected(self):
        for mutate in (lambda p: p.update(command=["arbitrary"]),
                       lambda p: p["models"].update(draft="unverified-model"),
                       lambda p: p.update(backend="auto"),
                       lambda p: p.update(id="../escape"),
                       lambda p: p["resources"].update(ram_required_gib=0),
                       lambda p: p["fixtures"]["document"].update(sha256="missing"),
                       lambda p: p["fixtures"]["document"].update(path="https://example.test/private"),
                       lambda p: p["fixtures"]["document"].update(path="\\/server/share/file"),
                       lambda p: p["fixtures"]["document"].update(path="/\\server/share/file"),
                       lambda p: p["fixtures"]["document"].update(path="\\\\server\\share\\file")):
            with self.subTest(mutate=mutate):
                profile = copy.deepcopy(self.profile)
                mutate(profile)
                with self.assertRaises(ValueError):
                    self.load(profile)

    def test_duplicate_id_and_primary_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load(self.profile, self.profile)
        self.profile["models"]["primary"] = "encoder"
        with self.assertRaisesRegex(ValueError, "primary role"):
            self.load()

    def test_module_typed_validation_runs_before_execution(self):
        self.module.validate_profile.side_effect = ValueError("invalid typed parameter")
        with self.assertRaisesRegex(ValueError, "typed parameter"):
            self.load()

    def test_gpu_requires_total_and_each_device_budget_without_pooling(self):
        self.profile.update(scenario="multi_gpu", backend="ggml_cuda")
        self.catalog["scenarios"] = [{"id": "multi_gpu"}]
        with self.assertRaisesRegex(ValueError, "vram_required"):
            self.load()
        self.profile["resources"].update(vram_required_gib=4, per_gpu_required_gib={"0": 4, "1": 4})
        self.profile["parameters"] = {"devices": [0, 1]}
        profile = self.load()[0]
        hardware = {"gpus": [{"index": 0, "memory_free_bytes": 20 * 1024 ** 3},
                             {"index": 1, "memory_free_bytes": 5 * 1024 ** 3}]}
        ok, why = profiles.per_device_budget(profile, hardware)
        self.assertFalse(ok)
        self.assertIn("not pooled", why)
        hardware["gpus"][1]["memory_free_bytes"] = 8 * 1024 ** 3
        self.assertTrue(profiles.per_device_budget(profile, hardware)[0])

    def test_parser_accepts_portable_profile_flag_and_reproduce_preserves_it(self):
        args = runner.parse_args(["--scenario-profiles", str(self.path)])
        command = runner.reproduce_args(args, self.root, self.root, {"id": "tiny"}, "cpu", {"id": "retrieval"})
        self.assertEqual(str(self.path), command[command.index("--scenario-profiles") + 1])

    def test_distributed_controller_reserves_only_single_device_control(self):
        profile = {"scenario": "distributed", "parameters": {"devices": [0, 1]},
                   "resources": {"per_gpu_required_gib": {"0": 8, "1": 8}}}
        hardware = {"gpus": [{"index": 0, "memory_free_bytes": 16 * 1024 ** 3}]}
        self.assertTrue(profiles.per_device_budget(profile, hardware)[0])
        profile["scenario"] = "multi_gpu"
        self.assertFalse(profiles.per_device_budget(profile, hardware)[0])

    def test_selected_profile_replaces_missing_recipe_inventory(self):
        catalog = {"presets": [], "scenarios": [{"id": "retrieval", "status": "automated", "execution": "profile"}]}
        self.assertEqual(1, len(runner.unbound_catalog_recipes(catalog, "retrieval")))
        self.assertEqual([], runner.unbound_catalog_recipes(catalog, "retrieval", [self.profile]))

    def test_one_profile_does_not_hide_other_selected_model_device_gaps(self):
        scenario = {"id": "retrieval", "status": "automated", "execution": "profile"}
        preset = {"id": "tiny", "scenarios": ["retrieval"], "backends": ["cpu", "ggml_cpu"]}
        args = runner.parse_args(["--backend", "all"])
        rows = runner.missing_profile_cases(preset, {"retrieval": scenario}, args, [self.profile])
        self.assertEqual(["ggml_cpu"], [row["backend"] for row in rows])
        other = dict(preset, id="other")
        self.assertEqual(2, len(runner.missing_profile_cases(other, {"retrieval": scenario}, args, [self.profile])))
        catalog = {"presets": [preset], "scenarios": [scenario]}
        self.assertEqual([], runner.unbound_catalog_recipes(catalog, "retrieval"))


if __name__ == "__main__":
    unittest.main()
