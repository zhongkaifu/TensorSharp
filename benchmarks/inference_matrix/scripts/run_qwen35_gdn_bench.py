#!/usr/bin/env python3
"""
Qwen3.5 / Qwen3.6 GatedDeltaNet chunked-vs-per-token prefill benchmark.

Runs the TensorSharp CLI in synthetic prefill benchmark mode for a configurable
list of prefill lengths, with the chunked GDN path enabled (default) and
disabled (GDN_DISABLE_CHUNKED_PREFILL=1). Optionally runs a verification pass
(GDN_VERIFY_CHUNKED=1) that internally compares both paths and reports the max
absolute / relative drift between them.

The output is one JSON file containing per-prefill-length entries:
  - chunked_ms, chunked_tps, chunked_gdn_ms_per_call
  - per_token_ms, per_token_tps, per_token_gdn_ms_per_call
  - speedup (per_token_ms / chunked_ms)
  - prefill_top_token_chunked, prefill_top_token_per_token (sanity check)
  - verify_max_out_abs, verify_max_state_abs (when --verify is passed)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TENSORSHARP_BIN = REPO / "TensorSharp.Cli" / "bin" / "TensorSharp.Cli"

SUMMARY_RE = re.compile(
    r"benchmark summary:\s*"
    r"bestPrefillMs=(?P<pms>[0-9.]+)\s*"
    r"bestPrefillTps=(?P<pps>[0-9.]+)\s*"
    r"bestDecodeMs=(?P<dms>[0-9.]+)\s*"
    r"bestDecodeTps=(?P<dps>[0-9.]+)"
)
TOPTOK_RE = re.compile(
    r"benchmark sampled tokens \(run1\): prefillTopToken=(?P<tok>\-?\d+)"
)
CHUNKED_LINE = re.compile(
    r"chunked path:\s+(?P<calls>\d+) calls,\s+(?P<ms>[0-9.]+) ms total"
    r"(?:,\s+(?P<msc>[0-9.]+) ms/call)?"
)
PER_TOKEN_LINE = re.compile(
    r"per-token path:\s+(?P<calls>\d+) prefill calls,\s+(?P<ms>[0-9.]+) ms total"
    r"(?:,\s+(?P<msc>[0-9.]+) ms/call)?"
)
VERIFY_LINE = re.compile(
    r"verification:\s+(?P<calls>\d+) calls,\s+"
    r"max output \|.\|=(?P<oa>[0-9.eE+-]+)\s+\(rel\s+(?P<orel>[0-9.eE+-]+)\),\s+"
    r"max state \|.\|=(?P<sa>[0-9.eE+-]+)\s+\(rel\s+(?P<srel>[0-9.eE+-]+)\),\s+"
    r"warnings=(?P<warn>\d+)"
)


def run_one(model_path: str, prefill: int, runs: int, chunked: bool,
            verify: bool, kv_dtype: str) -> dict:
    """Run one benchmark configuration and return parsed metrics."""
    env = os.environ.copy()
    if not chunked:
        env["GDN_DISABLE_CHUNKED_PREFILL"] = "1"
    if verify:
        env["GDN_VERIFY_CHUNKED"] = "1"

    cmd = [
        str(TENSORSHARP_BIN),
        "--model", model_path,
        "--backend", "ggml_metal",
        "--benchmark",
        "--bench-prefill", str(prefill),
        "--bench-decode", "1",
        "--bench-runs", str(runs),
        "--kv-cache-dtype", kv_dtype,
        "--log-level", "info",
        "--log-file", "off",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    wall_ms = (time.monotonic() - t0) * 1000
    combined = (proc.stderr or "") + "\n" + (proc.stdout or "")

    out: dict = {
        "ok": proc.returncode == 0,
        "wall_ms": wall_ms,
        "exit_code": proc.returncode,
    }
    m = SUMMARY_RE.search(combined)
    if m:
        out["best_prefill_ms"] = float(m.group("pms"))
        out["best_prefill_tps"] = float(m.group("pps"))
        out["best_decode_ms"] = float(m.group("dms"))
        out["best_decode_tps"] = float(m.group("dps"))
    m = TOPTOK_RE.search(combined)
    if m:
        out["prefill_top_token"] = int(m.group("tok"))
    m = CHUNKED_LINE.search(combined)
    if m:
        out["chunked_calls"] = int(m.group("calls"))
        out["chunked_ms_total"] = float(m.group("ms"))
        out["chunked_ms_per_call"] = float(m.group("msc")) if m.group("msc") else None
    m = PER_TOKEN_LINE.search(combined)
    if m:
        out["per_token_calls"] = int(m.group("calls"))
        out["per_token_ms_total"] = float(m.group("ms"))
        out["per_token_ms_per_call"] = float(m.group("msc")) if m.group("msc") else None
    m = VERIFY_LINE.search(combined)
    if m:
        out["verify_calls"] = int(m.group("calls"))
        out["verify_max_out_abs"] = float(m.group("oa"))
        out["verify_max_state_abs"] = float(m.group("sa"))
        out["verify_warnings"] = int(m.group("warn"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to a Qwen3.5/Qwen3.6 GGUF (e.g. Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf)")
    ap.add_argument("--prefill-lengths", default="128,512,2048",
                    help="Comma-separated prefill token counts to benchmark")
    ap.add_argument("--runs", type=int, default=3,
                    help="Number of benchmark runs per configuration")
    ap.add_argument("--kv-cache-dtype", default="f16")
    ap.add_argument("--verify", action="store_true",
                    help="Also run with GDN_VERIFY_CHUNKED=1 to capture max diff metrics")
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "inference_matrix" /
                                          "results" / "qwen35__gdn_chunked.json"))
    args = ap.parse_args()

    pp_lengths = [int(s.strip()) for s in args.prefill_lengths.split(",") if s.strip()]

    summary: dict = {
        "model_path": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "runs_per_config": args.runs,
        "configurations": [],
    }

    for pp in pp_lengths:
        print(f"=== prefill={pp} (chunked) ===", flush=True)
        chunked = run_one(args.model, pp, args.runs, chunked=True,
                          verify=False, kv_dtype=args.kv_cache_dtype)
        print(f"  prefill_tps={chunked.get('best_prefill_tps','?')} "
              f"chunked_ms_per_call={chunked.get('chunked_ms_per_call','?')} "
              f"top_token={chunked.get('prefill_top_token','?')}", flush=True)

        print(f"=== prefill={pp} (per-token) ===", flush=True)
        per_token = run_one(args.model, pp, args.runs, chunked=False,
                            verify=False, kv_dtype=args.kv_cache_dtype)
        print(f"  prefill_tps={per_token.get('best_prefill_tps','?')} "
              f"per_token_ms_per_call={per_token.get('per_token_ms_per_call','?')} "
              f"top_token={per_token.get('prefill_top_token','?')}", flush=True)

        verify: dict = {}
        if args.verify:
            print(f"=== prefill={pp} (verify) ===", flush=True)
            verify = run_one(args.model, pp, 1, chunked=True,
                             verify=True, kv_dtype=args.kv_cache_dtype)
            print(f"  verify_max_out_abs={verify.get('verify_max_out_abs','?')} "
                  f"warnings={verify.get('verify_warnings','?')}", flush=True)

        # Compute speedup. We compare GDN-block time only since that's the
        # contribution actually saved by the chunked kernel; total prefill
        # also includes MoE / matmul that this change does not touch.
        speedup_gdn = None
        if chunked.get("chunked_ms_per_call") and per_token.get("per_token_ms_per_call"):
            speedup_gdn = per_token["per_token_ms_per_call"] / chunked["chunked_ms_per_call"]
        speedup_ttft = None
        if chunked.get("best_prefill_ms") and per_token.get("best_prefill_ms"):
            speedup_ttft = per_token["best_prefill_ms"] / chunked["best_prefill_ms"]

        cfg = {
            "prefill_tokens": pp,
            "chunked": chunked,
            "per_token": per_token,
            "speedup_gdn_block": speedup_gdn,
            "speedup_total_prefill": speedup_ttft,
            "top_token_match": (chunked.get("prefill_top_token") ==
                                per_token.get("prefill_top_token")),
        }
        if verify:
            cfg["verify"] = verify
        summary["configurations"].append(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    print("\n=== Speedup summary (GDN block) ===")
    print(f"{'prefill':>8} {'chunked ms/call':>16} {'per-tok ms/call':>16} "
          f"{'gdn speedup':>12} {'TTFT speedup':>13} {'tokens match':>13}")
    for cfg in summary["configurations"]:
        c = cfg["chunked"]
        p = cfg["per_token"]
        print(f"{cfg['prefill_tokens']:>8} "
              f"{c.get('chunked_ms_per_call', float('nan')):>16.2f} "
              f"{p.get('per_token_ms_per_call', float('nan')):>16.2f} "
              f"{cfg['speedup_gdn_block'] or float('nan'):>12.2f} "
              f"{cfg['speedup_total_prefill'] or float('nan'):>13.2f} "
              f"{str(cfg['top_token_match']):>13}")


if __name__ == "__main__":
    main()
