"""No SSH, network, model or build execution: typed profiles and mocked supervision."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import signal
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import scaling_remote as remote


def digest(value):
    return hashlib.sha256(value).hexdigest()


class RemoteProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tokens = self.root / "tokens.json"
        tokens.write_bytes(b"[ 1, 2, 3, 4, 5, 6, 7, 8 ]\n")
        hosts = self.root / "known_hosts"
        hosts.write_bytes(b"synthetic pinned host fixture; no connection made\n")
        probe = self.root / "DeepSeek41DistributedProbe.dll"
        probe.write_bytes(b"synthetic pinned probe; never executed")
        model = self.root / "model.gguf"
        model.write_bytes(b"synthetic fixture marker; no model parser runs")
        names = ["TensorSharp.Core/Fixture.cs", "TensorSharp.Models/ModelBase.cs", "TensorSharp.Distributed/TcpCommunicator.cs",
                 "eng/validation/DeepSeek41DistributedProbe/Program.cs",
                 "eng/validation/DeepSeek41DistributedProbe/DeepSeek41DistributedProbe.csproj"]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source " + name)
        nodes = []
        for index in range(2):
            nodes.append({"host": f"node{index}.invalid", "port": 22, "user": "fixture", "authorized": True,
                          "expected_machine_id_sha256": digest(f"synthetic-machine-{index}\n".encode()),
                          "root": f"/node{index}/artifacts", "model_path": f"/node{index}/model.gguf",
                          "probe_path": f"/node{index}/bin/DeepSeek41DistributedProbe.dll", "dotnet": "/opt/dotnet/dotnet",
                          "model_files": [{"path": f"/node{index}/model.gguf", "sha256": digest(model.read_bytes())}],
                          "runtime_files": [{"path": f"/node{index}/bin/DeepSeek41DistributedProbe.dll", "sha256": digest(probe.read_bytes())},
                                            {"path": f"/node{index}/bin/libGgmlOps.so", "sha256": "a" * 64},
                                            {"path": f"/node{index}/bin/TensorSharp.Models.dll", "sha256": "b" * 64},
                                            {"path": "/opt/dotnet/dotnet", "sha256": "d" * 64}],
                          "source_files": [{"path": f"/node{index}/repo/" + name, "sha256": digest((self.root / name).read_bytes())} for name in names]})
        self.profile = {"parameters": {"mode": "tp", "local_degree": 1, "devices": [0], "nodes": nodes,
                                       "peer_endpoints": ["node0.invalid:19000", "node1.invalid:19001"],
                                       "chunk": 3, "decode_count": 2, "timeout_seconds": 10},
                        "fixtures": {name: {"path": str(path), "sha256": digest(path.read_bytes())}
                                     for name, path in (("tokens", tokens), ("known_hosts", hosts), ("probe", probe))},
                        "expected": {"weights_id": "synthetic-checkpoint@pinned"},
                        "resources": {"ram_required_gib": 8, "vram_required_gib": 8, "per_gpu_required_gib": {"0": 8}},
                        "backend": "ggml_cuda"}
        self.context = {"repo": self.root, "artifact_dir": self.root / "artifacts", "backend": "ggml_cuda", "context_tokens": 256,
                        "disk_headroom_gib": 0,
                        "profile_models": {"primary": {"identity": [{"path": str(model), "sha256": digest(model.read_bytes())}]}}}

    def test_valid_schema_and_two_phase_availability_never_connect(self):
        remote.validate_remote_profile(self.profile)
        with patch.object(remote.shutil, "which", return_value="synthetic-tool"), patch.object(remote, "ManagedProcess") as process:
            self.assertTrue(remote.availability(self.profile, self.context)[0])
            early = {**self.context, "profile_models": {}, "preparing_model": True}
            self.assertTrue(remote.availability(self.profile, early)[0])
            early["preparing_model"] = False
            self.assertFalse(remote.availability(self.profile, early)[0])
            process.assert_not_called()

    def test_unauthorized_ambiguous_or_injected_nodes_rejected(self):
        for change in (lambda p: p["parameters"]["nodes"][0].update(authorized=False),
                       lambda p: p["parameters"]["nodes"][0].update(host="host; touch /tmp/escape"),
                       lambda p: p["parameters"]["nodes"][0].update(root="/safe/../outside"),
                       lambda p: p["parameters"]["nodes"][0].update(dotnet="/bin/rm"),
                       lambda p: p["parameters"].update(command="arbitrary"),
                       lambda p: p["parameters"].update(peer_endpoints=["same:1", "same:1"]),
                       lambda p: p["parameters"].update(peer_endpoints=["unrelated.invalid:1", "node1.invalid:2"]),
                       lambda p: p["parameters"]["nodes"].pop(),
                       lambda p: p["parameters"].update(timeout_seconds=3601)):
            profile = copy.deepcopy(self.profile); change(profile)
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                remote.validate_remote_profile(profile)

    def test_resource_requirements_are_positive_finite_per_device(self):
        for resources in ({}, {"ram_required_gib": float("nan")}, {"per_gpu_required_gib": {"1": 8}},
                          {"per_gpu_required_gib": {"0": -1}}, {"ram_required_gib": True}):
            profile = copy.deepcopy(self.profile)
            if resources:
                profile["resources"].update(resources)
            else:
                profile["resources"] = {}
            with self.subTest(resources=resources), self.assertRaises(ValueError):
                remote.validate_remote_profile(profile)

    def test_alias_nodes_require_distinct_pinned_physical_machine_identities(self):
        profile = copy.deepcopy(self.profile)
        profile["parameters"]["nodes"][1]["expected_machine_id_sha256"] = profile["parameters"]["nodes"][0]["expected_machine_id_sha256"]
        with self.assertRaisesRegex(ValueError, "unique.*machine-id"):
            remote.validate_remote_profile(profile)

    def test_all_shards_and_source_anchors_required(self):
        profile = copy.deepcopy(self.profile)
        for node in profile["parameters"]["nodes"]:
            node["model_path"] = "/model-00001-of-00002.gguf"
            node["model_files"] = [{"path": node["model_path"], "sha256": "a" * 64}]
        with self.assertRaisesRegex(ValueError, "all and only"):
            remote.validate_remote_profile(profile)
        profile = copy.deepcopy(self.profile)
        for node in profile["parameters"]["nodes"]:
            node["source_files"] = node["source_files"][:1]
        with self.assertRaisesRegex(ValueError, "anchors"):
            remote.validate_remote_profile(profile)

    def test_model_probe_and_source_identity_checked_against_local_bytes(self):
        with patch.object(remote.shutil, "which", return_value="synthetic-tool"):
            wrong = copy.deepcopy(self.context)
            wrong["profile_models"]["primary"]["identity"][0]["sha256"] = "c" * 64
            self.assertFalse(remote.availability(self.profile, wrong)[0])
            (self.root / "TensorSharp.Models/ModelBase.cs").write_text("changed source")
            self.assertFalse(remote.availability(self.profile, self.context)[0])

    def test_ssh_and_scp_ignore_ambient_config_and_require_pinned_hosts(self):
        node = self.profile["parameters"]["nodes"][0]
        command = remote.ssh_prefix(node, "owned_known_hosts", "ssh", "owned_empty_config")
        for option in ("StrictHostKeyChecking=yes", "ControlMaster=no", "ControlPath=none", "ProxyCommand=none", "ProxyJump=none",
                       'UserKnownHostsFile="owned_known_hosts"', 'GlobalKnownHostsFile="owned_known_hosts"'):
            self.assertIn(option, command)
        self.assertEqual(command[command.index("-F") + 1], "owned_empty_config")
        copy_command = remote._scp_command(node, "owned_known_hosts", "/owned/receipt.json", "local.json", "scp", "owned_empty_config")
        self.assertIn("-P", copy_command)
        self.assertNotIn("-T", copy_command)
        self.assertNotIn("-p", copy_command)
        self.assertEqual(copy_command[-2], "fixture@node0.invalid:/owned/receipt.json")
        spaces = remote.ssh_prefix(node, self.root / "space directory" / "known_hosts", "ssh", "empty")
        value = next(arg for arg in spaces if arg.startswith("UserKnownHostsFile="))
        self.assertTrue(value.startswith('UserKnownHostsFile="') and value.endswith('"'))
        self.assertIn("space directory/known_hosts", value)

    def _probe_files(self, index):
        tokens = json.loads((self.root / "tokens.json").read_bytes())
        p = self.profile["parameters"]
        data = {"schema_version": 1, "status": "completed" if index == 0 else "worker_loop_returned", "disposed": True,
                "tokens": tokens, "tokens_sha256": digest((self.root / "tokens.json").read_bytes()),
                "weights_id": self.profile["expected"]["weights_id"], "vocab_size": 16,
                "topology": {"mode": "tensor_parallel", "local_degree": 1, "global_degree": 2, "rank_offset": index, "node_count": 2}, "steps": []}
        files = {}
        if index == 0:
            prefix = len(tokens) - p["decode_count"]
            wanted = [("reset", []), ("prefill", tokens[:prefix])]
            wanted += [("decode", [t]) for t in tokens[prefix:]] + [("reset", [])]
            wanted += [("chunk", tokens[i:min(prefix, i + p["chunk"])]) for i in range(0, prefix, p["chunk"])]
            wanted += [("refill", [t]) for t in tokens[prefix:]] + [("reset", []), ("reset-prefill", tokens[:prefix])]
            for i, (operation, inputs) in enumerate(wanted):
                step = {"operation": operation, "tokens": inputs, "logits": None}
                if operation != "reset":
                    step.update(logits=f"step-{i:03d}-{operation}.f32", count=16)
                    files["probe/" + step["logits"]] = struct.pack("<16f", *([.5, -.25] * 8))
                data["steps"].append(step)
        files["probe/probe.json"] = json.dumps(data).encode()
        files["process.log"] = (f"Backend: GgmlCuda\n[TcpCommunicator] Rank {index}/2 connected to all peers (protocol1)\n"
                                f"Distributed tensor parallelism: node {index}/2, 1 local GPU(s), 2 total across cluster.\n"
                                "Qwen3.5 TP weight sharding complete (2 GPUs, 1 local).\n").encode()
        return files

    def test_ipv6_host_is_unbracketed_for_ssh_and_bracketed_for_scp(self):
        node = {**self.profile["parameters"]["nodes"][0], "host": "2001:db8::7", "port": 2222}
        command = remote.ssh_prefix(node, "pinned_hosts", "ssh", "empty_config")
        self.assertEqual(command[-1], "fixture@2001:db8::7")
        self.assertEqual(command[command.index("-p") + 1], "2222")
        copy_command = remote._scp_command(node, "pinned_hosts", "/owned/receipt.json", "local.json", "scp", "empty_config")
        self.assertEqual(copy_command[-2], "fixture@[2001:db8::7]:/owned/receipt.json")
        self.assertEqual(copy_command[copy_command.index("-P") + 1], "2222")

    def _bundle(self, files):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        content = stream.getvalue()
        return content, [{"path": name, "bytes": len(value), "sha256": digest(value)} for name, value in files.items()]

    def test_artifact_hashes_paths_and_archive_members_are_strict(self):
        body, manifest = self._bundle(self._probe_files(1))
        bundle = self.root / "bundle.zip"; bundle.write_bytes(body)
        receipt = {"bundle": {"bytes": len(body), "sha256": digest(body)}, "artifacts": manifest}
        path = remote.extract_bundle(bundle, receipt, self.root / "good")
        self.assertTrue(path.is_file())
        for mutate in (lambda r: r["bundle"].update(sha256="f" * 64),
                       lambda r: r["artifacts"][0].update(path="../outside"),
                       lambda r: r["artifacts"][0].update(sha256="f" * 64),
                       lambda r: r["bundle"].update(bytes=remote.MAX_ARTIFACT_BYTES + 1)):
            wrong = copy.deepcopy(receipt); mutate(wrong)
            with self.assertRaises(ValueError):
                remote.extract_bundle(bundle, wrong, self.root / "bad")
            self.assertFalse((self.root / "bad").exists())

    def test_zero_exit_or_requested_topology_does_not_replace_join_and_placement(self):
        files = self._probe_files(0)
        folder = self.root / "node"; (folder / "probe").mkdir(parents=True)
        for name, content in files.items():
            (folder / name).write_bytes(content)
        path = folder / "probe/probe.json"
        proof = remote.qualify_node_probe(path, self.profile, 0, list(range(1, 9)), digest((self.root / "tokens.json").read_bytes()))
        self.assertTrue(proof["placement_lines"])
        (folder / "process.log").write_text("Backend: GgmlCuda\nrequested tp2\n")
        with self.assertRaisesRegex(ValueError, "connection"):
            remote.qualify_node_probe(path, self.profile, 0, list(range(1, 9)), digest((self.root / "tokens.json").read_bytes()))

    def _execute_mocked(self, cleanup=True, cancel=False, receipt_missing=False, truncated=False, low_disk=False, unavailable=False, duplicate_machine=False, preflight_mismatch=False):
        parent = self
        instances = []
        class Managed:
            def __init__(self, command, **kwargs):
                self.spec = json.loads(Path(command[-1]).read_text())
                self.stage = kwargs["stage"]
                self.folder = Path(kwargs["artifact_dir"])
                self.process = SimpleNamespace(poll=lambda: 1 if preflight_mismatch else 0)
                self.stopped = False
                instances.append(self)
            def wait(self, timeout):
                if cancel: raise KeyboardInterrupt()
                return self.snapshot()
            def stop(self):
                self.stopped = True
            def snapshot(self):
                snapshot = {"stage": self.stage, "exit_code": 1 if preflight_mismatch else 0, "teardown": {"attempted": True, "verified": True}}
                if preflight_mismatch:
                    payload = self.spec["payload"]
                    snapshot["stdout"] = json.dumps({"schema_version": 1, "run_id": payload["run_id"], "node_index": payload["node_index"],
                                                    "stage": "preflight", "status": "failed", "unavailable": False,
                                                    "teardown": {"attempted": False, "verified": True}, "hashes": [],
                                                    "error": "Remote machine identity hash mismatch; model launch refused"})
                return snapshot
        bundles = {}
        def copy_command(command, **kwargs):
            self.assertFalse(preflight_mismatch, "authenticated never-launched preflight receipt must not fetch a nonexistent remote file")
            index = int(kwargs["stage"].split("-")[0][4:])
            managed = next(item for item in instances if item.stage == f"node{index}")
            payload = managed.spec["payload"]
            self.assertEqual(base64.b64decode(payload["tokens_base64"]), (self.root / "tokens.json").read_bytes())
            self.assertEqual(Path(managed.spec["ssh_config"]).read_bytes(), b"")
            self.assertEqual(Path(managed.spec["known_hosts"]).parent, managed.folder)
            destination = Path(command[-1])
            remote_dir = payload["node"]["root"] + "/.tensorsharp-regression-" + payload["run_id"] + "-node" + str(index)
            if "-receipt-" in kwargs["stage"]:
                if receipt_missing:
                    return {"exit_code": 1, "stage": kwargs["stage"]}
                content, manifest = self._bundle(self._probe_files(index))
                bundles[index] = content
                receipt = {"schema_version": 1, "run_id": payload["run_id"], "node_index": index,
                           "status": "passed", "child": {"exit_code": 0, "log_truncated": truncated, "log_complete": not truncated}, "anchor_identity": {"pid": 123, "start_ticks": 456},
                           "hardware_budget": {"eligible": True}, "teardown": {"attempted": True, "verified": cleanup},
                           "node_identity": {"machine_id_sha256": self.profile["parameters"]["nodes"][0 if duplicate_machine else index]["expected_machine_id_sha256"]},
                           "hashes": [{"role": role, **f} for role in ("model_files", "runtime_files", "source_files") for f in payload["node"][role]],
                           "bundle": {"path": remote_dir + "/evidence.zip", "bytes": len(content), "sha256": digest(content)}, "artifacts": manifest}
                if unavailable:
                    receipt.update(status="failed", stage="preflight", unavailable=True,
                                   teardown={"attempted": False, "verified": True}, error="no remote GPU")
                    receipt.pop("child"); receipt.pop("anchor_identity")
                destination.write_text(json.dumps(receipt))
            else:
                self.assertFalse(low_disk, "no artifact transfer should start without local capacity")
                destination.write_bytes(bundles[index])
            return {"exit_code": 0, "stage": kwargs["stage"], "teardown": {"attempted": True, "verified": True}}
        result = {"assertions": [], "evidence": {}, "processes": []}
        with patch.object(remote.shutil, "which", return_value="synthetic-tool"), \
             patch.object(remote, "ManagedProcess", Managed), patch.object(remote, "run_process", side_effect=copy_command), \
             patch.object(remote.time, "sleep"), \
             patch.object(remote.shutil, "disk_usage", return_value=SimpleNamespace(free=0 if low_disk else 10 * 1024**3)):
            if cancel:
                with self.assertRaises(KeyboardInterrupt):
                    remote.run_remote_nodes(self.profile, self.context, result)
            elif unavailable:
                with self.assertRaises(remote.RemoteUnavailable):
                    remote.run_remote_nodes(self.profile, self.context, result)
            elif not cleanup or receipt_missing or truncated or low_disk or duplicate_machine or preflight_mismatch:
                with self.assertRaises(ValueError):
                    remote.run_remote_nodes(self.profile, self.context, result)
            else:
                receipts = remote.run_remote_nodes(self.profile, self.context, result)
                self.assertEqual([r["node_index"] for r in receipts], [0, 1])
                self.assertTrue(all(Path(r["local_probe_path"]).is_file() for r in receipts))
        self.assertTrue(all(p.stopped for p in instances))
        return result

    def test_mocked_nodes_verify_bytes_collect_and_qualify_without_network(self):
        result = self._execute_mocked()
        self.assertTrue(result["assertions"][-1]["passed"])

    def test_unknown_remote_cleanup_halts_global_resources(self):
        for options in ({"cleanup": None}, {"receipt_missing": True}):
            result = self._execute_mocked(**options)
            self.assertTrue(result["containment_failure"])
            self.assertTrue(any(r["teardown"]["verified"] is not True for r in result["evidence"]["remote_nodes"]))

    def test_cancellation_preserves_verified_receipts_then_reraises(self):
        result = self._execute_mocked(cancel=True)
        self.assertEqual(len(result["evidence"]["remote_nodes"]), 2)
        self.assertTrue(all(r["teardown"]["verified"] for r in result["evidence"]["remote_nodes"]))

    def test_truncated_remote_log_never_qualifies(self):
        result = self._execute_mocked(truncated=True)
        self.assertFalse(result["assertions"])
        self.assertTrue(all("collection_error" in r for r in result["evidence"]["remote_nodes"]))

    def test_low_local_disk_prevents_bundle_transfer_and_extraction(self):
        result = self._execute_mocked(low_disk=True)
        self.assertTrue(all("insufficient local disk" in r["collection_error"] for r in result["evidence"]["remote_nodes"]))

    def test_safe_prelaunch_unavailable_is_distinct_from_execution_failure(self):
        result = self._execute_mocked(unavailable=True)
        self.assertFalse(result.get("containment_failure", False))
        self.assertFalse(result["assertions"])

    def test_node_aliases_cannot_claim_multiple_machines(self):
        result = self._execute_mocked(duplicate_machine=True)
        self.assertFalse(result["assertions"])

    def test_prelaunch_pin_mismatch_stdout_preserves_hard_failure_without_unknown_cleanup(self):
        result = self._execute_mocked(preflight_mismatch=True)
        self.assertFalse(result.get("containment_failure", False))
        self.assertTrue(all("machine identity hash mismatch" in r["error"] for r in result["evidence"]["remote_nodes"]))
        self.assertTrue(all(r["unavailable"] is False for r in result["evidence"]["remote_nodes"]))

    def test_fallback_mixed_backend_oom_schema_and_nonfinite_probe_are_rejected(self):
        files = self._probe_files(1)
        folder = self.root / "negative-node"; (folder / "probe").mkdir(parents=True)
        path = folder / "probe/probe.json"
        for suffix in ("Backend: Cpu\n", "falling back to CPU\n", "CUDA out of memory\n"):
            path.write_bytes(files["probe/probe.json"])
            (folder / "process.log").write_bytes(files["process.log"] + suffix.encode())
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                remote.qualify_node_probe(path, self.profile, 1, list(range(1, 9)), digest((self.root / "tokens.json").read_bytes()))
        (folder / "process.log").write_bytes(files["process.log"])
        for data in ({**json.loads(files["probe/probe.json"]), "schema_version": 2},
                     {**json.loads(files["probe/probe.json"]), "load_seconds": float("nan")},
                     {**json.loads(files["probe/probe.json"]), "vocab_size": 2}):
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                remote.qualify_node_probe(path, self.profile, 1, list(range(1, 9)), digest((self.root / "tokens.json").read_bytes()))


class RemoteSupervisorContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = {"__name__": "synthetic_remote_unit"}
        exec(compile(remote.REMOTE_SOURCE, "<fixed-remote-supervisor>", "exec"), self.ns)

    def test_anchor_retains_group_and_has_independent_eof_and_timeout(self):
        compile(remote.ANCHOR_SOURCE, "<fixed-group-anchor>", "exec")
        self.assertIn("signal.SIG_IGN", remote.ANCHOR_SOURCE)
        self.assertIn("lost.is_set() or time.monotonic()>=deadline", remote.ANCHOR_SOURCE)
        self.assertIn("os.killpg(os.getpgrp(),signal.SIGKILL)", remote.ANCHOR_SOURCE)
        self.assertIn("sys.stdin.buffer.read(1)", remote.ANCHOR_SOURCE)
        self.assertIn("complete=not reader.is_alive()", remote.ANCHOR_SOURCE)
        self.assertIn("'log_truncated':truncated[0] or not complete", remote.ANCHOR_SOURCE)

    def test_group_cleanup_escalates_even_after_probe_parent_exited(self):
        clock = iter([0, 1, 3, 4, 5])
        fake_signal = SimpleNamespace(SIGTERM=15, SIGKILL=9, Signals=lambda n: SimpleNamespace(name=str(n)))
        anchor = SimpleNamespace(pid=4321, wait=Mock(return_value=-9))
        members = [[{"pid": 4322}], []]
        with patch.dict(self.ns, {"signal": fake_signal, "process_identity": lambda pid: {"start_ticks": 10, "pgrp": pid, "session": pid},
                                  "group_members": Mock(side_effect=members),
                                  "time": SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None),
                                  "os": SimpleNamespace(killpg=Mock())}):
            proof = self.ns["stop_owned"](anchor, {"start_ticks": 10})
            self.assertTrue(proof["verified"])
            self.assertEqual([call.args for call in self.ns["os"].killpg.call_args_list], [(4321, 15), (4321, 9)])

    def test_reused_pid_is_never_signalled_and_live_group_is_unverified(self):
        kill = Mock()
        with patch.dict(self.ns, {"process_identity": lambda pid: {"start_ticks": 99, "pgrp": pid, "session": pid},
                                  "group_members": lambda _: [{"pid": 4321}], "os": SimpleNamespace(killpg=kill)}):
            proof = self.ns["stop_owned"](SimpleNamespace(pid=4321), {"start_ticks": 10})
        self.assertFalse(proof["verified"])
        kill.assert_not_called()

    def test_remote_memory_budget_checks_selected_devices_without_pooling(self):
        content = "MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n"
        config = {"timeout_seconds": 10, "devices": [0, 1],
                  "resources": {"ram_required_gib": 8, "per_gpu_required_gib": {"0": 8, "1": 8}}}
        with patch.object(Path, "read_text", return_value=content), patch.object(Path, "is_file", return_value=False), \
             patch.object(self.ns["subprocess"], "run", return_value=SimpleNamespace(returncode=0, stdout="0, 14000, 16000\n1, 2000, 16000\n")):
            budget = self.ns["hardware_budget"](config)
        self.assertFalse(budget["eligible"])
        self.assertTrue(budget["devices"][0]["eligible"])
        self.assertFalse(budget["devices"][1]["eligible"])

    def _prelaunch_failure(self, hash_mismatch=True, occupied=False, machine_mismatch=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, probe, dependency = root / "model.gguf", root / "DeepSeek41DistributedProbe.dll", root / "TensorSharp.Models.dll"
            model.write_bytes(b"small unit marker")
            probe.write_bytes(b"small unit probe marker")
            dependency.write_bytes(b"small unit dependency marker")
            token_bytes = b"[1,2,3,4]\n"
            config = {"run_id": "a" * 32, "node_index": 0, "timeout_seconds": 10,
                      "tokens": [1, 2, 3, 4], "tokens_base64": base64.b64encode(token_bytes).decode(), "tokens_sha256": digest(token_bytes),
                      "context_tokens": 256, "devices": [0], "mode": "tp", "local_degree": 1,
                      "peer_endpoints": ["node0:1", "node1:2"], "chunk": 2, "decode_count": 2,
                      "backend": "ggml_cuda", "weights_id": "synthetic",
                      "node": {"root": str(root), "probe_path": str(probe), "dotnet": "/opt/dotnet/dotnet", "model_path": str(model),
                               "expected_machine_id_sha256": digest(b"different machine\n" if machine_mismatch else b"synthetic-node-id\n"),
                               "model_files": [{"path": str(model), "sha256": "f" * 64 if hash_mismatch else digest(model.read_bytes())}],
                               "runtime_files": [{"path": str(probe), "sha256": digest(probe.read_bytes())},
                                                 {"path": str(dependency), "sha256": digest(dependency.read_bytes())}],
                               "source_files": []}}
            fake_sys = SimpleNamespace(platform="linux", stdin=SimpleNamespace(buffer=io.BytesIO((json.dumps(config) + "\n").encode())))
            fake_signal = SimpleNamespace(SIGHUP=1, SIGTERM=15, SIGINT=2, signal=Mock())
            process = Mock()
            budget = Mock(side_effect=[{"eligible": True}, {"eligible": False}])
            if occupied:
                existing = root / (".tensorsharp-regression-" + "a" * 32 + "-node0")
                existing.mkdir()
                (existing / "receipt.json").write_text("existing user evidence")
            original_is_file = Path.is_file
            original_read_bytes = Path.read_bytes
            with patch.dict(self.ns, {"sys": fake_sys, "signal": fake_signal,
                                     "hardware_budget": budget}), \
                 patch.object(self.ns["threading"], "Thread"), \
                 patch.object(self.ns["subprocess"], "Popen", process), \
                 patch.object(self.ns["shutil"], "disk_usage", return_value=SimpleNamespace(free=1024**3)), \
                 patch.object(Path, "is_file", lambda path: True if str(path).replace("\\", "/") in ("/proc/self/stat", "/etc/machine-id") else original_is_file(path)), \
                 patch.object(Path, "read_bytes", lambda path: b"synthetic-node-id\n" if str(path).replace("\\", "/") == "/etc/machine-id" else original_read_bytes(path)), \
                 patch.object(self.ns["os"], "uname", return_value=SimpleNamespace(nodename="synthetic-node"), create=True), \
                 patch("builtins.print") as emitted:
                code = self.ns["main"]()
            process.assert_not_called()
            self.assertEqual(code, 1)
            if machine_mismatch:
                budget.assert_not_called()
                self.assertFalse(list(root.glob(".tensorsharp-regression-*")))
                receipt = json.loads(emitted.call_args.args[0])
                self.assertIn("machine identity hash mismatch", receipt["error"])
                self.assertFalse(receipt["unavailable"])
                self.assertEqual(receipt["teardown"], {"attempted": False, "verified": True})
                self.assertEqual(receipt["node_identity"]["machine_id_sha256"], digest(b"synthetic-node-id\n"))
                return receipt
            if occupied:
                self.assertEqual((existing / "receipt.json").read_text(), "existing user evidence")
                budget.assert_not_called()
                return None
            receipt = json.loads(next(root.glob(".tensorsharp-regression-*/receipt.json")).read_text())
            self.assertEqual(receipt["teardown"], {"attempted": False, "verified": True})
            return receipt

    def test_remote_hash_mismatch_never_launches_probe_and_records_safe_failure(self):
        receipt = self._prelaunch_failure()
        self.assertIn("Hash mismatch", receipt["error"])
        self.assertFalse(receipt["unavailable"])

    def test_remote_capacity_is_refreshed_after_hashing_before_model_launch(self):
        receipt = self._prelaunch_failure(hash_mismatch=False)
        self.assertIn("resources became insufficient", receipt["error"])
        self.assertTrue(receipt["hardware_budget_before_hash"]["eligible"])
        self.assertFalse(receipt["hardware_budget"]["eligible"])
        self.assertTrue(receipt["unavailable"])

    def test_existing_remote_run_directory_is_never_modified(self):
        self._prelaunch_failure(occupied=True)

    def test_alias_resolving_to_another_pinned_node_never_launches_or_hashes_models(self):
        # Two aliases may target these same raw machine-id bytes. Distinct pins
        # mean the wrongly bound alias fails before any model/resource work.
        receipt = self._prelaunch_failure(machine_mismatch=True)
        self.assertEqual(receipt["stage"], "preflight")

    def test_ssh_worker_transmits_exact_bounded_payload_and_keeps_stdin_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launch.json"
            payload = {"node": {"host": "node.invalid", "user": "fixture", "port": 22}, "tokens_base64": "WyAxLCAyIF0K"}
            path.write_text(json.dumps({"payload": payload, "known_hosts": "pinned", "ssh_config": "empty", "ssh": "mock-ssh"}))
            pipe = io.BytesIO()
            observed = []
            def wait():
                self.assertFalse(pipe.closed)
                observed.append(json.loads(pipe.getvalue()))
                return 0
            child = SimpleNamespace(stdin=pipe, wait=wait)
            with patch.object(remote.subprocess, "Popen", return_value=child) as process:
                self.assertEqual(remote._ssh_worker(path), 0)
            self.assertEqual(observed, [payload])
            self.assertTrue(pipe.closed)
            command = process.call_args.args[0]
            self.assertTrue(command[-1].startswith("python3 -I -c "))
            self.assertNotIn(payload["tokens_base64"], command[-1])


if __name__ == "__main__":
    unittest.main()
