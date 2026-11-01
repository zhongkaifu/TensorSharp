"""Synthetic agent events prove that prose and final arithmetic cannot fake delegation."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import agent_regression as agents


def delegated():
    children = [{"agent_id": child, "parent_id": "/root", "agent_type": "reviewer",
                 "permissions": "read-only", "depends_on": [], "workspace_id": "workspace-" + label,
                 "status": "completed", "tool": None, "result": json.dumps(agents._EXPECTED[label]), "error": None}
                for child, label in agents._CHILDREN.items()]
    return {"content": json.dumps(agents._EXPECTED), "done": True, "chunks": [
        {"skill_step": "spawn_agent", "agent_id": "/root", "ok": True, "files": None},
        {"skill_step": "spawn_agent", "agent_id": "/root", "ok": True, "files": None},
        {"skill_step": "wait_agent", "agent_id": "/root", "ok": True, "files": None},
        {"tool_progress": "finished", "tool": "wait_agent", "agents": children},
        {"token": json.dumps(agents._EXPECTED)}, {"done": True, "aborted": False, "error": None, "truncated": False}]}


def single():
    return {"content": "42", "done": True, "chunks": [{"token": "42"}, {"done": True}]}


class FakeHttp:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def stream(self, route, payload, **kwargs):
        self.calls.append((route, payload, kwargs))
        return self.responses.pop(0)


class AgentRegressionTests(unittest.TestCase):
    def setUp(self):
        self.result = {"assertions": [], "inputs": {}, "evidence": {}}

    def validate(self, stream):
        agents.validate_delegation(stream, self.result)

    def test_two_completed_children_and_exact_semantics_pass(self):
        self.validate(delegated())
        self.assertTrue(all(item["passed"] for item in self.result["assertions"]))
        self.assertEqual(set(self.result["evidence"]["agents"]["observed_children"]), set(agents._CHILDREN))

    def test_final_answer_and_claimed_delegation_text_without_events_fail(self):
        stream = delegated()
        stream["chunks"] = [{"token": "I spawned two agents and waited successfully."}, {"done": True}]
        with self.assertRaisesRegex(ValueError, "two root children"):
            self.validate(stream)

    def test_failed_spawn_or_child_invocation_is_not_success(self):
        for change in ({"ok": False}, {"agent_id": "/root/proposal_a"}):
            stream = delegated()
            stream["chunks"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(stream)

    def test_wait_before_both_admissions_fails(self):
        stream = delegated()
        stream["chunks"][1], stream["chunks"][2] = stream["chunks"][2], stream["chunks"][1]
        with self.assertRaisesRegex(ValueError, "admitted before"):
            self.validate(stream)

    def test_unapproved_tools_and_file_exports_fail(self):
        for change in ({"skill_step": "shell"}, {"skill_step": "read_file"}, {"files": [{"name": "unexpected.txt"}]}):
            stream = delegated()
            stream["chunks"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(stream)
        stream = delegated()
        stream["chunks"].insert(0, {"tool_progress": "running", "tool": "shell"})
        with self.assertRaisesRegex(ValueError, "permitted coordination"):
            self.validate(stream)

    def test_missing_finished_wait_cannot_be_inferred_from_final_answer(self):
        for change in ({"tool_progress": "running"}, {"agents": []}, {"agents": {}}):
            stream = delegated()
            stream["chunks"][3].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(stream)

    def test_pending_failed_cancelled_or_cross_request_child_fails(self):
        for change in ({"status": "running"}, {"status": "cancelled"}, {"error": "failure"},
                       {"agent_id": "/root/unknown"}, {"parent_id": "/root/other"}):
            stream = delegated()
            stream["chunks"][3]["agents"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(stream)

    def test_mutable_role_dependencies_or_child_tools_fail(self):
        for change in ({"permissions": "workspace-write"}, {"agent_type": "worker"},
                       {"depends_on": ["/root/proposal_b"]}, {"tool": "list_agents"}):
            stream = delegated()
            stream["chunks"][3]["agents"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(stream)

    def test_distinct_logical_workspace_evidence_required(self):
        for workspace in (None, "", "workspace-B"):
            stream = delegated()
            stream["chunks"][3]["agents"][0]["workspace_id"] = workspace
            with self.subTest(workspace=workspace), self.assertRaises(ValueError):
                self.validate(stream)

    def test_child_bad_arithmetic_cannot_be_masked_by_correct_parent_result(self):
        stream = delegated()
        answer = dict(agents._EXPECTED["A"], profit=1300)
        stream["chunks"][3]["agents"][0]["result"] = json.dumps(answer)
        with self.assertRaisesRegex(ValueError, "semantic arithmetic result"):
            self.validate(stream)

    def test_final_output_requires_exact_integer_structure(self):
        for value in ({"A": agents._EXPECTED["A"]},
                      {**agents._EXPECTED, "C": {}},
                      {**agents._EXPECTED, "A": dict(agents._EXPECTED["A"], revenue="3000")}):
            stream = delegated()
            stream["content"] = json.dumps(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(stream)

    def test_duplicate_and_nonfinite_report_numbers_rejected(self):
        for value in ('{"profit": 1300, "profit": 1140}', '{"profit":NaN}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                agents._json_object(value)

    def test_terminal_must_be_unique_success(self):
        for terminal in ([], [{"done": True, "aborted": True}], [{"done": True}, {"done": True}]):
            stream = delegated()
            stream["chunks"] = stream["chunks"][:-1] + terminal
            with self.subTest(terminal=terminal), self.assertRaisesRegex(ValueError, "normal completion"):
                self.validate(stream)

    def test_run_has_separate_optout_and_no_client_tools_or_attachments(self):
        http = FakeHttp([delegated(), single()])
        agents.run_agents(http, "model", {"timeout_seconds": 600}, self.result)
        self.assertEqual(len(http.calls), 2)
        self.assertTrue(http.calls[0][1]["multi_agent"])
        self.assertFalse(http.calls[1][1]["multi_agent"])
        self.assertTrue(self.result["evidence"]["agents"]["opt_out_verified"])
        for route, payload, kwargs in http.calls:
            self.assertEqual(route, "/api/chat")
            self.assertEqual(kwargs, {"web_ui": True})
            self.assertFalse(set(payload) & {"tools", "skills", "files", "images", "knowledge_collections"})

    def test_optout_tools_or_wrong_answer_fail(self):
        for response in ({**single(), "content": "41"}, {**single(), "chunks": [{"skill_step": "spawn_agent", "ok": True}, {"done": True}]}):
            result = {"assertions": [], "inputs": {}, "evidence": {}}
            with self.subTest(response=response), self.assertRaises(ValueError):
                agents.run_agents(FakeHttp([delegated(), response]), "model", {}, result)

    def test_host_limits_never_enable_worker_or_code_capabilities(self):
        flags = agents.host_arguments(600)
        self.assertEqual(flags[flags.index("--agents-timeout") + 1], "120")
        self.assertEqual(flags[flags.index("--agents-max-count") + 1], "2")
        self.assertEqual(flags[flags.index("--agents-max-depth") + 1], "1")
        self.assertFalse(set(flags) & {"--code-exec", "--agents-allow-worker-tools", "--skill", "--skills-dir"})
        for value in (0, -1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                agents.host_arguments(value)


if __name__ == "__main__":
    unittest.main()
