"""Source-linked guards: added configs/architectures must not vanish from coverage."""
import json
from pathlib import Path
import re
import sys
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eng"))
from regression.configs import read_config


class CatalogInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((REPO / "config/regression/catalog.json").read_text(encoding="utf-8"))
        cls.presets = {row["id"]: row for row in cls.catalog["presets"]}
        cls.families = {row["id"]: row for row in cls.catalog["families"]}

    def test_every_config_has_one_inventory_entry(self):
        actual = {path.relative_to(REPO).as_posix() for path in (REPO / "config").rglob("*.json")
                  if "regression" not in path.relative_to(REPO / "config").parts}
        inventoried = [row["config"] for row in self.presets.values()]
        self.assertEqual(actual, set(inventoried))
        self.assertEqual(len(inventoried), len(set(inventoried)))

    def test_registered_architecture_sources_have_family_inventory(self):
        manifest = (REPO / "TensorSharp.Models/Architecture/BuiltInArchitectures.cs").read_text()
        classes = set(re.findall(r"Register\((?:\w+\.)?(\w+Architecture)\.(?:Descriptor|FlashDescriptor)\)", manifest))
        self.assertTrue(classes)
        evidence = {path for family in self.families.values() for path in family["source_evidence"]}
        for architecture in classes:
            matches = list((REPO / "TensorSharp.Models").rglob(architecture + ".cs"))
            self.assertEqual(1, len(matches), architecture)
            self.assertIn(matches[0].relative_to(REPO).as_posix(), evidence, architecture)
        for embedding in ("bert", "nomic-bert", "gemma-embedding2"):
            self.assertIn(embedding, self.families)

    def test_evidence_and_references_exist(self):
        scenarios = {row["id"] for row in self.catalog["scenarios"]}
        self.assertEqual(len(scenarios), len(self.catalog["scenarios"]))
        for row in self.catalog["families"] + self.catalog["presets"] + self.catalog["scenarios"]:
            for path in row.get("source_evidence", row.get("evidence", [])):
                self.assertTrue((REPO / path).is_file(), path)
        for preset in self.presets.values():
            self.assertIn(preset["family"], self.families)
            self.assertFalse(set(preset["scenarios"]) - scenarios)
            self.assertRegex(preset["id"], r"^[A-Za-z0-9_.-]+$")

    def test_refused_backend_routes_are_not_automatically_planned(self):
        forbidden = {
            "qwen4exp": {"mlx"}, "qwen_image": {"cuda", "mlx"},
            "bert": {"cuda", "mlx", "ggml_vulkan"}, "nomic-bert": {"cuda", "mlx", "ggml_vulkan"},
            "gemma-embedding2": {"ggml_vulkan"}, "bonsai2": {"cpu", "cuda", "mlx"},
            "minimax-m3": {"cpu", "cuda", "mlx", "ggml_metal", "ggml_vulkan"},
        }
        for family, backends in forbidden.items():
            self.assertFalse(set(self.families[family]["backends"]) & backends, family)
            for preset in self.presets.values():
                if preset["family"] == family:
                    self.assertFalse(set(preset["backends"]) & backends, preset["id"])
        self.assertEqual({"cpu", "cuda", "ggml_cpu", "ggml_cuda", "ggml_metal"},
                         set(next(s for s in self.catalog["scenarios"] if s["id"] == "forced_logits_sequence")["backends"]))

    def test_family_gaps_are_explicit_and_do_not_look_configured(self):
        for family in self.families.values():
            configs = [p["id"] for p in self.presets.values() if p["family"] == family["id"]]
            self.assertEqual(sorted(configs), sorted(family["presets"]))
            if not configs:
                self.assertEqual("missing_preset", family["configuration_status"])
                self.assertTrue(family["gap_reason"])

    def test_new_model_presets_use_source_pins_and_complete_shards(self):
        additions = ["bonsai-2-27b-ptq1", "nemotron-h-8b", "nemotron-h-47b",
                     "nemotron-3-nano-omni-30b-a3b", "nemotron-3.5-lightning-30b-a3b",
                     "mistral-small-3.1-24b", "hunyuan-hy-mt2-1.8b", "deepseek-v4-flash-0731",
                     "deepseek-v4.1-flash", "glm-5.2", "glm-5.3", "glm-5.3-flash", "qwen3.8-27b"]
        source = "\n".join(path.read_text(encoding="utf-8") for path in
                           (REPO / "TensorAgent/src/TensorAgent.Core/Catalog").glob("ModelCatalog*.cs"))
        source += (REPO / "config/agent-qwen3.8-27b.json").read_text(encoding="utf-8")
        for name in additions:
            resolved = read_config(REPO / self.presets[name]["config"], REPO / "models", environ={})
            self.assertTrue(resolved.files, name)
            shards = []
            for spec in resolved.files:
                self.assertRegex(spec.sha256 or "", r"^[0-9a-f]{64}$")
                self.assertIn(spec.sha256, source, spec.path.name)
                self.assertTrue(spec.urls)
                for url in spec.urls:
                    self.assertRegex(url, r"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/")
                match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", spec.path.name)
                if match and spec.option == "model":
                    shards.append(tuple(map(int, match.groups())))
            if shards:
                total = shards[0][1]
                self.assertEqual(list(range(1, total + 1)), sorted(number for number, count in shards))
                self.assertTrue(all(count == total for number, count in shards))


if __name__ == "__main__":
    unittest.main()
