"""Partial-file planning uses remaining bytes without understating final memory."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression.configs import FileSpec
from regression.downloads import DownloadPolicy, PrepareError, prepare_files
from regression.runner import sizes_for
import test_downloads as download_fixture
import test_runner as runner_fixture


class ResumeAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"GGUF" + b"synthetic" * 100
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.target = self.root / "model.gguf"
        self.spec = FileSpec("model", self.target,
            ("https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/model.gguf",),
            self.sha, len(self.payload))
        self.policy = DownloadPolicy(self.root, 200, 0, True)

    def partial(self, content, **metadata_overrides):
        Path(str(self.target) + ".part").write_bytes(content)
        Path(str(self.target) + ".part.json").write_text(json.dumps({
            "sha256": self.sha, "size_bytes": len(self.payload), "version": 1, **metadata_overrides}))

    def test_offline_known_size_credits_owned_partial_without_payload_io(self):
        self.partial(self.payload[:-100])
        with patch("regression.downloads._open", side_effect=AssertionError("offline")), \
             patch("regression.downloads._hash", side_effect=AssertionError("no payload hashing in plan")):
            footprint, remaining = sizes_for([self.spec], self.policy)
        self.assertEqual(footprint, len(self.payload))
        self.assertEqual(remaining, 100)
        self.assertLess(remaining, self.policy.max_download_bytes)
        self.assertGreater(footprint, self.policy.max_download_bytes)

    def test_head_discovered_size_uses_remaining_not_full_target(self):
        unknown = FileSpec("model", self.target, self.spec.urls, self.sha)
        with patch("regression.downloads.probe_files", return_value=[{
            "size_bytes": 10 * 1024**3, "remaining_bytes": 1024**3, "status": "available"}]):
            self.assertEqual(sizes_for([unknown], self.policy, network=True), (10 * 1024**3, 1024**3))

    def test_unowned_or_identity_mismatched_partial_is_blocked_and_retained(self):
        partial = Path(str(self.target) + ".part")
        partial.write_bytes(b"user partial")
        with self.assertRaises(PrepareError) as error:
            sizes_for([self.spec], self.policy)
        self.assertEqual(error.exception.code, "partial_unowned")
        self.assertEqual(partial.read_bytes(), b"user partial")
        self.partial(self.payload[:-100], sha256="f" * 64)
        with self.assertRaises(PrepareError) as error:
            sizes_for([self.spec], self.policy)
        self.assertEqual(error.exception.code, "partial_unowned")

    def test_oversized_partial_is_blocked_and_complete_partial_costs_zero_transfer(self):
        self.partial(self.payload + b"extra")
        with self.assertRaises(PrepareError) as error:
            sizes_for([self.spec], self.policy)
        self.assertEqual(error.exception.code, "partial_oversized")
        self.partial(self.payload)
        self.assertEqual(sizes_for([self.spec], self.policy), (len(self.payload), 0))

    def test_local_final_and_partial_have_separate_footprint_and_remaining(self):
        local = self.root / "cached.gguf"
        local.write_bytes(self.payload)
        self.partial(self.payload[:-100])
        self.assertEqual(sizes_for([FileSpec("model", local), self.spec], self.policy),
                         (2 * len(self.payload), 100))


class ResumeExecutionTests(unittest.TestCase):
    def test_tiny_loopback_resume_fits_budget_smaller_than_target(self):
        # Reuse the tested 8 KB fixture server, never a real model source.
        download_fixture.DownloadTests.setUpClass()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = download_fixture.PAYLOAD
                spec = FileSpec("model", root / "model.gguf",
                    (download_fixture.DownloadTests.url + "/file",), download_fixture.SHA, len(payload))
                policy = DownloadPolicy(root, 2048, 0, True, chunk_bytes=1024, allow_loopback_http=True)
                Path(str(spec.path) + ".part").write_bytes(payload[:-1024])
                Path(str(spec.path) + ".part.json").write_text(json.dumps({
                    "sha256": spec.sha256, "size_bytes": len(payload), "version": 1}))
                self.assertEqual(sizes_for([spec], policy, network=True), (len(payload), 1024))
                identities = prepare_files([spec], policy)
                self.assertEqual(policy.downloaded_bytes, 1024)
                self.assertEqual(spec.path.read_bytes(), payload)
                self.assertEqual(identities[0].sha256, download_fixture.SHA)
        finally:
            download_fixture.DownloadTests.tearDownClass()


class ResumeRunnerTests(unittest.TestCase):
    setUp = runner_fixture.RunnerTests.setUp
    invoke = runner_fixture.RunnerTests.invoke

    def test_two_gib_budget_allows_one_gib_remainder_of_ten_gib_model(self):
        with patch("regression.runner.sizes_for", return_value=(10 * 1024**3, 1024**3)):
            code, report = self.invoke("--mode", "run", "--allow-download", "--max-download-gib", "2")
        self.assertEqual(code, 0)
        self.prepare.assert_called_once()
        estimates = report["results"][-1]["resource_plan"]["estimates"]
        self.assertEqual(estimates["weight_bytes"], 10 * 1024**3)
        self.assertEqual(estimates["disk_required_bytes"], 1024**3)

    def test_actual_restart_bytes_are_charged_to_whole_run_budget(self):
        first = self.catalog_data["presets"][0]
        self.catalog_data["presets"] = [{**first, "id": "first"}, {**first, "id": "second"}]
        self.catalog.write_text(json.dumps(self.catalog_data))

        def prepare(files, policy):
            # A range-ignorant server needed more than the initially reserved bytes.
            policy.downloaded_bytes = 2 * 1024**3
            return [{"status": "local", "path": str(self.model)}]

        self.prepare.side_effect = prepare
        with patch("regression.runner.sizes_for", return_value=(10 * 1024**3, 1024**3)):
            _, report = self.invoke("--mode", "run", "--allow-download", "--max-download-gib", "2")
        self.assertEqual(self.prepare.call_count, 1)
        self.assertEqual(report["results"][-1]["status"], "skip")
        self.assertIn("whole-run download budget", report["results"][-1]["reason"])
