"""summarize-bench-suite.py and run-multinode-case.py --kind suite, on log text in the CLI's own format.

The warm excerpt is a real TensorSharp.Cli suite log (DeepSeek V4.1 synthetic fixture, --backend cpu, 2026-10-30);
the cold rows add the eviction lines a two-node run prints (one per node on the driver, one per worker log).
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("summarize_bench_suite", HERE.parent / "summarize-bench-suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
spec = importlib.util.spec_from_file_location("run_multinode_case", HERE.parent / "run-multinode-case.py")
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)

P = "20:17:43 info: TensorSharp.Cli[1710] "
WARM = [
    "bench suite starting: workloads={n} evictMeasure=sampled hostMappedTables=1",
    "bench suite workload 1/{n}: label=pp100-a source=synthetic prefill=100 decode=8 runs=2 mode=random evict=none chunked=0 seed=20261001 check=1",
    "inference benchmark starting: prefillTokens=100 decodeTokens=8 runs=2 chunked=False decodeMode=random-inference",
    "benchmark run 1/2: prefillMs=2946 prefillTps=33.9 decodeMs=2029 decodeTps=3.9 msPerTok=253.6 decodeModelMs=2029 decodeModelTps=3.9 greedySampleMs=0.0 decodeMode=random-inference",
    "benchmark run 2/2: prefillMs=2532 prefillTps=39.5 decodeMs=2280 decodeTps=3.5 msPerTok=285.0 decodeModelMs=2280 decodeModelTps=3.5 greedySampleMs=0.0 decodeMode=random-inference",
    "benchmark sampled tokens (untimed correctness): prefillTopToken=234 decode=179,136,121,86,175,212,213,130",
    "bench suite workload 1/{n} done: label=pp100-a wallMs=14012 coldVerified=na",
    "bench suite workload 2/{n}: label=pp64-d0-warm source=synthetic prefill=64 decode=0 runs=3 mode=fixed evict=none chunked=0 seed=20261001 check=1",
    "inference benchmark starting: prefillTokens=64 decodeTokens=0 runs=3 chunked=False decodeMode=fixed-inference",
    "benchmark run 1/3: prefillMs=1355 prefillTps=47.2 decodeMs=0 decodeTps=NaN msPerTok=NaN decodeModelMs=0 decodeModelTps=NaN greedySampleMs=0.0 decodeMode=fixed-inference",
    "benchmark run 2/3: prefillMs=1007 prefillTps=63.5 decodeMs=0 decodeTps=NaN msPerTok=NaN decodeModelMs=0 decodeModelTps=NaN greedySampleMs=0.0 decodeMode=fixed-inference",
    "benchmark run 3/3: prefillMs=958 prefillTps=66.8 decodeMs=0 decodeTps=NaN msPerTok=NaN decodeModelMs=0 decodeModelTps=NaN greedySampleMs=0.0 decodeMode=fixed-inference",
    "benchmark sampled tokens (untimed correctness): prefillTopToken=65 decode=",
    "bench suite workload 2/{n} done: label=pp64-d0-warm wallMs=4914 coldVerified=na",
]


def evict(label, before, node, verified="true", after="0.0"):
    return (f"bench evict: label={label} beforeRun={before} node={node} supported=true ranges=1 bytesMiB=128901.6 "
            f"residentBeforeMiB=61234.0 residentAfterMiB={after} verified={verified} advised=2 skipped=0 errors=0 "
            f"elapsedMs=812 evictMs=301")


def cold_row(index, n, label, evict_mode="row", runs=2, verified=("true", "true"), done="true"):
    lines = [f"bench suite workload {index}/{n}: label={label} source=synthetic prefill=2048 decode=128 runs={runs} "
             f"mode=random evict={evict_mode} chunked=0 seed=20261001 check=1",
             f"inference benchmark starting: prefillTokens=2048 decodeTokens=128 runs={runs} chunked=False decodeMode=random-inference"]
    for r in range(1, runs + 1):
        if r == 1 or evict_mode == "run":
            lines += [evict(label, r, 0, verified[0]), evict(label, r, 1, verified[1], "0.0" if verified[1] == "true" else "512.0")]
        lines.append(f"benchmark run {r}/{runs}: prefillMs={14000 + r} prefillTps=146.3 decodeMs={12800 + r} decodeTps=10.0 "
                     f"msPerTok=100.0 decodeModelMs={12800 + r} decodeModelTps=10.0 greedySampleMs=0.0 decodeMode=random-inference")
    lines.append(f"bench suite workload {index}/{n} done: label={label} wallMs=60000 coldVerified={done}")
    return lines


def log(lines):
    return "\n".join(P + line for line in lines) + "\n"


class BenchSuiteSummaryTests(unittest.TestCase):
    def test_a_real_warm_suite_log_measures_every_row(self):
        lines = [line.format(n=2) for line in WARM] + ["bench suite complete: workloads=2 wallMs=31958 json=x.json"]
        report = suite.summarize(log(lines), discard=1)
        self.assertEqual("measured", report["status"], report["failures"])
        a, b = report["workloads"]
        self.assertEqual(("pp100-a", [2], 100 * 1000 / 2532, 8 * 1000 / 2280), (a["label"], a["measured_runs"], a["median_prefill_tps"], a["median_decode_tps"]))
        self.assertEqual(100 * 1000 / 2946, a["run1_prefill_tps"])
        self.assertEqual([2, 3], b["measured_runs"])
        self.assertIsNone(b["median_decode_tps"])
        self.assertEqual([179, 136, 121, 86, 175, 212, 213, 130], a["sampled_tokens"]["decode_token_ids"])
        self.assertIsNone(a["cold_verified"])

    def test_the_json_supplies_exact_milliseconds_and_timestamps(self):
        lines = [line.format(n=2) for line in WARM] + ["bench suite complete: workloads=2 wallMs=31958"]
        exact = {"status": "completed", "workloads": [
            {"label": "pp100-a", "samples": [{"prefill_ms": 2945.6577, "decode_ms": 2028.9927, "prefill_start_utc": "t0"},
                                             {"prefill_ms": 2531.5, "decode_ms": 2280.25}]},
            {"label": "pp64-d0-warm", "samples": [{"prefill_ms": 1355.1}, {"prefill_ms": 1007.2}, {"prefill_ms": 958.3}]}]}
        report = suite.summarize(log(lines), exact, discard=1)
        self.assertEqual("measured", report["status"], report["failures"])
        a = report["workloads"][0]
        self.assertEqual(100 * 1000 / 2531.5, a["median_prefill_tps"])
        self.assertEqual("t0", a["samples"][0]["prefill_start_utc"])
        exact["workloads"][0]["samples"].pop()
        self.assertIn("pp100-a: the JSON has 1 runs, the log 2", suite.summarize(log(lines), exact)["failures"])
        exact["status"] = "failed"
        self.assertIn("suite JSON status is 'failed', not 'completed'", suite.summarize(log(lines), exact)["failures"])

    def test_cold_rows_need_a_verified_eviction_on_every_node(self):
        lines = [line.format(n=4) for line in WARM]
        lines += cold_row(3, 4, "pp2048-cold") + cold_row(4, 4, "pp2048-each", evict_mode="run", runs=3)
        lines.append("bench suite complete: workloads=4 wallMs=1")
        worker = log(["host-table eviction: node=1 supported=true ranges=1 bytesMiB=128901.6 residentBeforeMiB=61234.0 "
                      "residentAfterMiB=0.0 verified=true advised=2 skipped=0 errors=0 elapsedMs=812 evictMs=301"] * 4)
        report = suite.summarize(log(lines), worker_logs={"1": worker}, discard=1)
        self.assertEqual("measured", report["status"], report["failures"])
        cold, each = report["workloads"][2:]
        self.assertTrue(cold["cold_verified"])
        self.assertEqual([1], sorted({e["before_run"] for e in cold["evictions"]}))
        self.assertEqual([0, 1], [e["node"] for e in cold["evictions"]])
        self.assertEqual([1, 2, 3], sorted({e["before_run"] for e in each["evictions"]}))
        self.assertEqual(61234.0, cold["evictions"][1]["residentBeforeMiB"])
        # The worker logged fewer evictions than the driver gathered from it.
        short = suite.summarize(log(lines), worker_logs={"1": worker.split("\n", 1)[1]})
        self.assertTrue(any("node 1 logged 3 evictions" in f for f in short["failures"]), short["failures"])

    def test_an_unverified_or_missing_eviction_fails_the_row(self):
        for row, why in ((cold_row(3, 3, "x", verified=("true", "false"), done="false"), "node 1 verified=False"),
                         (cold_row(3, 3, "x", verified=("true", "na"), done="na"), "node 1 verified=None"),
                         ([l for l in cold_row(3, 3, "x") if "bench evict" not in l], "evictions before runs []")):
            lines = [line.format(n=3) for line in WARM] + row + ["bench suite complete: workloads=3 wallMs=1"]
            report = suite.summarize(log(lines))
            self.assertEqual("failed", report["status"])
            self.assertFalse(report["workloads"][2]["cold_verified"])
            self.assertTrue(any(why in f for f in report["failures"]), (why, report["failures"]))

    def test_an_incomplete_suite_fails(self):
        lines = [line.format(n=3) for line in WARM]
        report = suite.summarize(log(lines))
        self.assertEqual("failed", report["status"])
        self.assertIn("2 of 3 workloads started", report["failures"])
        self.assertTrue(any("stopped early" in f for f in report["failures"]))
        lines = [line.format(n=2) for line in WARM if "benchmark run 3/3" not in line] + ["bench suite complete: workloads=2 wallMs=1"]
        self.assertTrue(any("runs [1, 2] instead of 1..3" in f for f in suite.summarize(log(lines))["failures"]))


class SuiteKindTests(unittest.TestCase):
    BASE = ["--node", "root@203.0.113.10:22195", "--node", "root@203.0.113.20:22080", "--model", "/root/m-00001-of-00006.gguf",
            "--mode", "layer", "--local-degree", "2", "--peers", "127.0.0.1:19500,127.0.0.1:19501", "--label", "suite-unit"]

    def args(self, *extra):
        out = Path(tempfile.mkdtemp()) / "out"
        return case.parse_args(self.BASE + ["--output", str(out), *extra])

    def test_rows_reach_every_node_and_the_driver_writes_the_json(self):
        a = self.args("--kind", "suite", "--suite-workload", "label=c,prefill=2048,decode=128,runs=4,mode=random,cold",
                      "--suite-workload", "label=w,prefill=2048", "--suite-evict-measure", "full", "--bench-decode", "64")
        plan = case.build_plan(a)
        for p in plan:
            argv = p["argv"]
            self.assertEqual(["label=c,prefill=2048,decode=128,runs=4,mode=random,cold", "label=w,prefill=2048"],
                             [argv[i + 1] for i, x in enumerate(argv) if x == "--bench-workload"])
            self.assertIn("--bench-random-tokens", argv)
            self.assertEqual("64", argv[argv.index("--bench-decode") + 1])
            self.assertEqual("512", argv[argv.index("--bench-prefill") + 1])   # the default for rows without prefill=
            self.assertEqual("full", argv[argv.index("--bench-evict-measure") + 1])
            self.assertEqual(p["run"] + "/bench-suite.json", argv[argv.index("--bench-json") + 1])
            self.assertNotIn("--benchmark", argv)
            self.assertIn("--bench-evict-strict", argv)   # an unverified cold row stops the suite by default
            # Each row is one word of the launcher's command line.
            self.assertIn(" --bench-workload label=c,prefill=2048,decode=128,runs=4,mode=random,cold ", p["launcher"])

    def test_allow_unverified_drops_the_strict_flag(self):
        a = self.args("--kind", "suite", "--suite-workload", "label=w,prefill=8", "--suite-allow-unverified")
        self.assertNotIn("--bench-evict-strict", case.build_plan(a)[0]["argv"])

    def test_a_suite_file_adds_rows_and_bad_rows_are_refused(self):
        rows = Path(tempfile.mkdtemp()) / "rows.json"
        rows.write_text(json.dumps([{"label": "r8k", "text": "/root/s7/r8k.txt", "decode": 32, "cold": True}]), encoding="utf-8")
        a = self.args("--kind", "suite", "--suite-workload", "label=w,prefill=8", "--suite-file", str(rows))
        self.assertEqual(["label=w,prefill=8", "label=r8k,text=/root/s7/r8k.txt,decode=32,cold=1"], a.suite_rows)
        text = rows.with_suffix(".txt")
        text.write_text("# rows\nlabel=a,prefill=8,cold\n\nlabel=b,prefill=8\n", encoding="utf-8")
        self.assertEqual(["label=a,prefill=8,cold", "label=b,prefill=8"], case.suite_file_rows(text))
        for extra in (["--kind", "suite"], ["--kind", "bench", "--suite-workload", "label=x"],
                      ["--kind", "suite", "--suite-workload", "text=C:/prompts/r8k.txt"]):
            with self.assertRaises(SystemExit):
                self.args(*extra)


if __name__ == "__main__":
    unittest.main()
