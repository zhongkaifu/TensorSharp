"""Fail-closed attribution gates for the three-chat native acceptance evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "unified-three-chat-regression.py"
spec = importlib.util.spec_from_file_location("three_chat", path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def fixture():
    rows = [dict(request_id=f"http-{i}", session=dict(sessionId=f"session-{i}")) for i in range(3)]
    log = [dict(template="generation.sequence.bound", scope=dict(RequestId=f"http-{i}"),
                props=dict(InternalRequestId=f"seq-{i}", SessionId=f"session-{i}")) for i in range(3)]
    def batch(eid, members, position):
        return dict(template="batch.fused.accepted", props=dict(
            SequenceRequestIdsJson=members, PositionsBeforeJson=[position] * len(members),
            PositionsAfterJson=[position + 1] * len(members), ActiveSequenceCount=len(members),
            ManagedPublishCompleted=True, NativeAcceptedExecutionId=str(eid), NativeNodeCount=10,
            NativeGraphCount=1, NativeHostSegmentCount=4, NativeExecutionKind=1))
    log += [batch(1, [f"seq-{i}" for i in range(3)], 100),
            batch(2, [f"seq-{i}" for i in range(3)], 101),
            dict(template="generation.timings", scope=dict(RequestId="http-0"), props=dict(FinishReason="cancelled")),
            batch(3, ["seq-1", "seq-2"], 102), batch(4, ["seq-1", "seq-2"], 103)]
    return rows, log


class ThreeChatEvidenceTests(unittest.TestCase):
    def test_actual_three_then_two_accepts(self):
        rows, log = fixture()
        result = tool.native_proof(rows, log)
        self.assertTrue(result["passed"], result)
        self.assertEqual((2, 2), (result["n3_steps"], result["n2_sibling_steps"]))

    def test_closed_socket_without_server_cancellation_refuses(self):
        rows, log = fixture()
        del log[5]
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_two_before_cancel_does_not_qualify_survival(self):
        rows, log = fixture()
        log.insert(7, log.pop(5))
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_other_sessions_do_not_qualify(self):
        rows, log = fixture()
        log[1]["props"]["SessionId"] = "another-session"
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_partial_native_failure_refuses(self):
        rows, log = fixture()
        log[3]["props"]["ManagedPublishCompleted"] = False
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_duplicate_native_id_refuses(self):
        rows, log = fixture()
        log[4]["props"]["NativeAcceptedExecutionId"] = "1"
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_replayed_position_refuses(self):
        rows, log = fixture()
        log[-1]["props"]["PositionsBeforeJson"] = [101, 101]
        log[-1]["props"]["PositionsAfterJson"] = [102, 102]
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_multiple_graphs_are_not_one_joint_graph(self):
        rows, log = fixture()
        log[3]["props"]["NativeGraphCount"] = 3
        self.assertFalse(tool.native_proof(rows, log)["passed"])

    def test_serial_width_does_not_qualify_three(self):
        rows, log = fixture()
        for record in log[3:5]:
            p = record["props"]
            p["SequenceRequestIdsJson"] = ["seq-0", "seq-1"]
            p["PositionsBeforeJson"] = p["PositionsBeforeJson"][:2]
            p["PositionsAfterJson"] = p["PositionsAfterJson"][:2]
            p["ActiveSequenceCount"] = 2
        self.assertFalse(tool.native_proof(rows, log)["passed"])


if __name__ == "__main__": unittest.main()
