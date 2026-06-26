#!/usr/bin/env python3
"""
Cross-engine inference benchmark driver.

Compares TensorSharp, llama.cpp and vLLM on the same GGUF files across text /
image / audio / video / single-turn / multi-turn / function-call / structured
scenarios on GPU and CPU backends. One server instance is launched per
(engine, backend, model) group, all of that group's scenarios run against it,
then it is torn down — so per-scenario timings exclude model-load cost.

Each cell writes one JSON file to the results directory:
    results/{engine}__{backend}__{model}__{scenario}.json

Two extra axes are optional and default to the single baseline point so old
invocations are unchanged:
    --mtp off|on|off,on    MTP/NextN speculative decoding (TensorSharp only;
                           relaunches the server per mode)
    --concurrency 1,4,8    fire N identical requests in parallel per cell and
                           record system-wide aggregate decode throughput

Examples
--------
# Smoke test one cheap cell
python run_matrix.py --engines tensorsharp --backends gpu \
    --models gemma4-12b --scenarios text_short,multi_turn

# MTP on vs off (TensorSharp), single-stream
python run_matrix.py --engines tensorsharp --backends gpu \
    --models qwen36-35b-a3b,gemma4-12b --scenarios text_short --mtp off,on

# Parallel-request scaling (aggregate throughput under load)
python run_matrix.py --engines tensorsharp,llamacpp --backends gpu \
    --models gemma4-12b --scenarios text_short --concurrency 1,4,8

# Full matrix (engines auto-skip when a binary / endpoint is missing)
python run_matrix.py --engines tensorsharp,llamacpp,vllm --backends gpu,cpu \
    --models gemma4-12b,qwen36-35b-a3b,diffusiongemma --mtp off,on --concurrency 1,8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from itertools import groupby
from pathlib import Path


def _peek_config_arg(argv) -> str | None:
    """`--config PATH` must be honored before `config` imports + loads its JSON,
    so peek at argv here (argparse runs too late)."""
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


_cfg_arg = _peek_config_arg(sys.argv[1:])
if _cfg_arg:
    os.environ["BENCH_CONFIG"] = _cfg_arg

import config
import engines
import scenarios as scen


def _result_path(results_dir: Path, engine, backend, model_id, scenario,
                 mtp: bool = False, concurrency: int = 1) -> Path:
    # Baseline cells (no MTP, single request) keep their historical filename so
    # prior results stay valid; extra axes only add a suffix when non-default.
    name = f"{engine}__{backend}__{model_id}__{scenario}"
    if mtp:
        name += "__mtp"
    if concurrency and concurrency > 1:
        name += f"__c{concurrency}"
    return results_dir / f"{name}.json"


def _write(results_dir: Path, res: engines.BenchResult):
    p = _result_path(results_dir, res.engine, res.backend, res.model, res.scenario,
                     res.mtp, res.concurrency)
    p.write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")


def _run_cell(server, engine_id, backend, model, scenario_id, max_tokens,
              mtp=False, concurrency=1) -> engines.BenchResult:
    res = engines.BenchResult(engine=engine_id, backend=backend,
                              model=model.short_id, scenario=scenario_id,
                              mtp=mtp, concurrency=concurrency)
    sc = config.SCENARIOS[scenario_id]
    try:
        req = scen.build_request(scenario_id, engine_id, model)
    except Exception as ex:  # asset / decode error
        res.status = "fail"
        res.detail = f"build error: {ex}"
        return res
    if req.get("messages") is None:
        res.status = "skipped"
        res.detail = req.get("detail", "no request payload")
        return res

    model_name = engines.served_model_name(engine_id, server, model)
    eff_max_tokens = sc.max_tokens if sc.max_tokens else max_tokens
    try:
        if concurrency > 1:
            m = engines.run_openai_chat_parallel(
                server.base_url, model_name, req["messages"],
                concurrency=concurrency,
                tools=req.get("tools"),
                response_format=req.get("response_format"),
                max_tokens=eff_max_tokens,
                stream=not model.is_diffusion)
        else:
            m = engines.run_openai_chat(
                server.base_url, model_name, req["messages"],
                tools=req.get("tools"),
                response_format=req.get("response_format"),
                max_tokens=eff_max_tokens,
                stream=not model.is_diffusion)
    except Exception as ex:
        res.status = "fail"
        res.detail = f"request error: {ex}"
        return res

    res.status = "ok"
    res.prompt_tokens = m["prompt_tokens"]
    res.completion_tokens = m["completion_tokens"]
    res.ttft_ms = round(m["ttft_ms"], 1)
    res.prefill_tps = round(m["prefill_tps"], 1)
    res.decode_tps = round(m["decode_tps"], 1)
    res.aggregate_decode_tps = round(m.get("aggregate_decode_tps", m["decode_tps"]), 1)
    res.requests_ok = int(m.get("requests_ok", 1))
    res.total_wall_ms = round(m["total_wall_ms"], 1)
    res.finish_reason = m["finish_reason"]
    res.output_preview = (m.get("output_text") or "")[:300]
    if concurrency > 1 and res.requests_ok < concurrency:
        res.detail = f"{res.requests_ok}/{concurrency} parallel requests ok"
    checker = req.get("checker")
    if checker is not None:
        try:
            res.tool_call_ok = bool(checker(m))
        except Exception:
            res.tool_call_ok = None
    return res


def _warmup(server, engine_id, model):
    try:
        engines.run_openai_chat(
            server.base_url, engines.served_model_name(engine_id, server, model),
            [{"role": "user", "content": "Hello"}],
            max_tokens=8, stream=not model.is_diffusion, timeout_s=600)
    except Exception:
        pass


def _parse_mtp(s: str) -> list:
    """'off' | 'on' | 'off,on' -> [False] | [True] | [False, True]."""
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("off", "0", "false", "no", "none"):
            if False not in out:
                out.append(False)
        elif tok in ("on", "1", "true", "yes", "mtp"):
            if True not in out:
                out.append(True)
        elif tok in ("both", "off,on"):
            out = [False, True]
        else:
            raise SystemExit(f"--mtp: unrecognized value '{tok}' (use off | on | off,on)")
    return out or list(config.DEFAULT_MTP_MODES)


def _parse_concurrency(s: str) -> list:
    """'1,4,8' -> [1, 4, 8] (deduped, positive, ordered)."""
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError:
            raise SystemExit(f"--concurrency: '{tok}' is not an integer")
        if n < 1:
            raise SystemExit(f"--concurrency: '{tok}' must be >= 1")
        if n not in out:
            out.append(n)
    return out or list(config.DEFAULT_CONCURRENCY)


def _csv(s: str) -> list:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main():
    # Every default below comes from the config file (selected via --config /
    # BENCH_CONFIG); a command-line flag, when given, overrides that default.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="JSON benchmark config file (default: benchmark_config.json, "
                         "or set BENCH_CONFIG). All defaults below are read from it.")
    ap.add_argument("--engines", default=None,
                    help=f"comma list (default from config: {','.join(config.DEFAULT_ENGINES)})")
    ap.add_argument("--backends", default=None,
                    help=f"comma list (default from config: {','.join(config.BACKENDS)})")
    ap.add_argument("--models", default=None,
                    help=f"comma list (default from config: {','.join(config.DEFAULT_MODELS)})")
    ap.add_argument("--scenarios", default=None,
                    help=f"comma list (default from config: {','.join(config.DEFAULT_SCENARIOS)})")
    ap.add_argument("--mtp", default=None,
                    help="MTP/NextN speculative decoding modes: off | on | off,on "
                         "(TensorSharp only; relaunches the server per mode)")
    ap.add_argument("--concurrency", default=None,
                    help="parallel identical requests per cell, e.g. 1 or 1,4,8 "
                         "(measures aggregate decode throughput under load)")
    ap.add_argument("--results", default=None,
                    help="results directory (default from config)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"per-request max tokens (default from config: {config.DEFAULT_MAX_TOKENS})")
    ap.add_argument("--warmup", type=int, default=None,
                    help=f"warmup requests per server, 0 to disable (default from config: {config.DEFAULT_WARMUP})")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    engine_ids = _csv(args.engines) or list(config.DEFAULT_ENGINES)
    backend_ids = _csv(args.backends) or list(config.BACKENDS)
    model_ids = _csv(args.models) or list(config.DEFAULT_MODELS)
    scenario_ids = _csv(args.scenarios) or list(config.DEFAULT_SCENARIOS)
    mtp_modes = _parse_mtp(args.mtp) if args.mtp else list(config.DEFAULT_MTP_MODES)
    concurrency_levels = _parse_concurrency(args.concurrency) if args.concurrency else list(config.DEFAULT_CONCURRENCY)
    max_tokens = args.max_tokens if args.max_tokens is not None else config.DEFAULT_MAX_TOKENS
    warmup = args.warmup if args.warmup is not None else config.DEFAULT_WARMUP

    results_dir = Path(args.results) if args.results else config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    media = config.resolve_media()
    print("# engine-comparison benchmark")
    print(f"config     : {config.CONFIG_PATH}")
    print(f"engines    : {engine_ids}")
    print(f"backends   : {backend_ids}")
    print(f"models     : {model_ids}")
    print(f"scenarios  : {scenario_ids}")
    print(f"mtp        : {['on' if m else 'off' for m in mtp_modes]}")
    print(f"concurrency: {concurrency_levels}")
    print(f"results    : {results_dir}")
    print(f"media      : image={media['image']} audio={media['audio']} video={media['video']}")
    print()

    # Build the full plan, splitting into gated-skips and runnable cells. The
    # (engine, backend, model, mtp) tuple fixes how the server is launched; the
    # concurrency axis is applied per-cell against that one server.
    plan = []   # (engine, backend, model_id, scenario_id, mtp, applies, reason)
    for engine_id in engine_ids:
        if engine_id not in config.ENGINES:
            print(f"  unknown engine '{engine_id}', skipping")
            continue
        for backend in backend_ids:
            for model_id in model_ids:
                model = config.MODELS[model_id]
                for mtp in mtp_modes:
                    for scenario_id in scenario_ids:
                        sc = config.SCENARIOS[scenario_id]
                        ok, why = config.applies(engine_id, backend, model, sc, mtp=mtp)
                        plan.append((engine_id, backend, model_id, scenario_id, mtp, ok, why))

    runnable = [p for p in plan if p[5]]
    gated = [p for p in plan if not p[5]]

    # Record gated cells as skips (one per concurrency level) so the matrix is complete.
    for engine_id, backend, model_id, scenario_id, mtp, _, why in gated:
        for conc in concurrency_levels:
            res = engines.BenchResult(engine=engine_id, backend=backend, model=model_id,
                                      scenario=scenario_id, status="skipped", detail=why,
                                      mtp=mtp, concurrency=conc)
            _write(results_dir, res)

    n_runnable_cells = len(runnable) * len(concurrency_levels)
    print(f"{n_runnable_cells} runnable cells "
          f"({len(runnable)} scenario-points x {len(concurrency_levels)} concurrency), "
          f"{len(gated) * len(concurrency_levels)} gated (recorded as skipped)\n")

    # Group runnable cells by (engine, backend, model, mtp) so each server starts once.
    runnable.sort(key=lambda p: (p[0], p[1], p[2], p[4]))
    groups = [(k, list(g)) for k, g in groupby(runnable, key=lambda p: (p[0], p[1], p[2], p[4]))]

    for (engine_id, backend, model_id, mtp), cells in groups:
        model = config.MODELS[model_id]
        scen_ids = [c[3] for c in cells]
        mtp_tag = " mtp=on" if mtp else ""
        header = f"[{engine_id}/{backend}/{model_id}{mtp_tag}]"
        print(f"=== {header}  scenarios={scen_ids}  concurrency={concurrency_levels} ===", flush=True)

        # Pre-flight: model file and engine binary present?
        missing = None
        if not model.gguf.exists():
            missing = f"model file not found: {model.gguf}"
        elif engine_id == "tensorsharp" and not config.TENSORSHARP_SERVER_DLL.exists():
            missing = f"TensorSharp.Server.dll not found: {config.TENSORSHARP_SERVER_DLL}"
        elif engine_id == "llamacpp" and not config.LLAMA_SERVER_EXE.exists():
            missing = f"llama-server.exe not found: {config.LLAMA_SERVER_EXE}"
        if missing:
            print(f"    SKIP group: {missing}")
            for c in cells:
                for conc in concurrency_levels:
                    _write(results_dir, engines.BenchResult(
                        engine=engine_id, backend=backend, model=model_id,
                        scenario=c[3], status="skipped", detail=missing,
                        mtp=mtp, concurrency=conc))
            continue

        # Separate log per (engine, backend, model, mtp) so MTP-on / -off don't clobber.
        log_name = f"{engine_id}__{backend}__{model_id}" + ("__mtp" if mtp else "")
        log_path = results_dir / "logs" / f"{log_name}.log"
        # Server max-tokens must cover the busiest cell: concurrency doesn't raise
        # per-request length, but keep the configured headroom.
        server = engines.make_server(engine_id, model, backend, log_path,
                                     max_tokens=config.SERVER_MAX_TOKENS, mtp=mtp)

        t0 = time.monotonic()
        server.start()
        timeout = config.READY_TIMEOUT_S[model.size_class]
        ready = server.wait_ready(timeout)
        load_s = time.monotonic() - t0
        if not ready:
            status = "skipped" if engine_id == "vllm" else "fail"
            detail = (f"engine unavailable at {server.base_url}"
                      if engine_id == "vllm"
                      else f"server not ready in {timeout:.0f}s; log tail:\n{server.tail_log()}")
            print(f"    {status.upper()} group: server not ready ({load_s:.0f}s)")
            for c in cells:
                for conc in concurrency_levels:
                    _write(results_dir, engines.BenchResult(
                        engine=engine_id, backend=backend, model=model_id,
                        scenario=c[3], status=status, detail=detail,
                        mtp=mtp, concurrency=conc))
            server.stop()
            continue
        print(f"    server ready in {load_s:.0f}s", flush=True)

        if warmup > 0:
            for _ in range(warmup):
                _warmup(server, engine_id, model)

        for c in cells:
            scenario_id = c[3]
            for conc in concurrency_levels:
                out_file = _result_path(results_dir, engine_id, backend, model_id,
                                        scenario_id, mtp, conc)
                if args.skip_existing and out_file.exists():
                    try:
                        prev = json.loads(out_file.read_text(encoding="utf-8"))
                        if prev.get("status") == "ok":
                            print(f"    {scenario_id:14s} c={conc:<3d} cached")
                            continue
                    except Exception:
                        pass
                t = time.monotonic()
                res = _run_cell(server, engine_id, backend, model, scenario_id,
                                max_tokens, mtp=mtp, concurrency=conc)
                _write(results_dir, res)
                wall = time.monotonic() - t
                extra = ""
                if res.tool_call_ok is not None:
                    extra += f"  tool_ok={res.tool_call_ok}"
                agg = (f"  agg={res.aggregate_decode_tps:7.1f} t/s" if conc > 1 else "")
                print(f"    {scenario_id:14s} c={conc:<3d} {res.status:7s}  "
                      f"prefill={res.prefill_tps:7.1f} t/s  decode={res.decode_tps:6.1f} t/s{agg}  "
                      f"ttft={res.ttft_ms:7.0f}ms  wall={wall:5.1f}s{extra}  {res.detail[:50]}",
                      flush=True)

        server.stop()
        print()

    print("done. Generate the report with:  python report.py")


if __name__ == "__main__":
    main()
