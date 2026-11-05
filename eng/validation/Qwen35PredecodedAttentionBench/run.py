#!/usr/bin/env python3
"""Explicit future GPU entry point. Never called by build or CPU tests."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("predecoded_attention_build_for_run",HERE/"build.py")
build=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(build)


def verify_build(directory):
    m=json.loads((directory/"manifest.json").read_text())
    p=json.loads((directory/build.PERMIT).read_text())
    if(m.get("schema")!="q35-predecoded-attention-build/1" or m.get("complete") is not True
       or p.get("schema")!="q35-predecoded-attention-permission/1"
       or p.get("manifest_sha256")!=build.digest(directory/"manifest.json")):
        raise ValueError("missing/rejected/stale independent predecoded-attention build")
    if not isinstance(m.get("resources"),dict) or set(m["resources"])!=set(build.SYMBOLS):
        raise ValueError("incomplete actual resource evidence")
    if not isinstance(m.get("artifacts"),dict) or set(m["artifacts"])!=set(build.ARTIFACTS) or p.get("artifacts")!=m["artifacts"]:
        raise ValueError("artifact authorization differs")
    for n,h in m["artifacts"].items():
        if Path(n).name!=n or build.digest(directory/n)!=h:raise ValueError("artifact changed: "+n)
    if m.get("tools")!={n:build.digest(HERE/n) for n in build.TOOLS}:
        raise ValueError("owned tool changed; rebuild required")
    if m.get("frozen_tools")!={n:build.digest(build.FROZEN/n) for n in ("build.py","bench.cu","checks.h","attention64.cuh")}:
        raise ValueError("sealed fixture/helper changed")
    if build.digest(m["source"])!=m["source_sha256"]:raise ValueError("production source changed")
    return m


def validated_result(r,mode,long_cases,no_second_barrier=False):
    if r.get("complete") is not True:return False
    variants=[0,1,2] if no_second_barrier else [0,1]
    if r.get("no_second_barrier_requested") is not no_second_barrier:return False
    if mode=="--resources":return(r.get("validation")=="resources_only_not_quality"
        and r.get("active_variants")==variants and r.get("device_resource_rejected_variants")==[])
    shapes=81 if long_cases else 79
    decoded=shapes+((6 if long_cases else 4) if mode=="--bench" else 0)
    return(r.get("strict_shapes")==shapes and r.get("strict_candidate_comparisons")==shapes*3*(len(variants)-1)
        and r.get("decoded_fixture_checks")==decoded and r.get("decoded_boundary_checks")==73 and r.get("active_variants")==variants
        and r.get("device_resource_rejected_variants")==[]
        and r.get("long_cases") is long_cases and r.get("timed") is(mode=="--bench"))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build",type=Path,required=True);ap.add_argument("--name",required=True)
    modes=ap.add_mutually_exclusive_group();modes.add_argument("--bench",action="store_true");modes.add_argument("--resources",action="store_true")
    ap.add_argument("--long",action="store_true");ap.add_argument("--with-no-second-barrier",action="store_true");args=ap.parse_args(argv)
    directory=build.evidence_dir(args.build)
    if not args.name or Path(args.name).name!=args.name or args.name in(".",".."):ap.error("name must be one NEW subdirectory")
    out=directory/args.name;out.mkdir(parents=True,exist_ok=False)
    state=dict(schema="q35-predecoded-attention-run/1",complete=False,gpu_invoked=False,
        started_utc=datetime.now(timezone.utc).isoformat(),hardware_counters="NOT_RUN")
    persist=lambda:build.save(out/"run.json",state)
    persist()
    try:
        manifest=verify_build(directory)
        mode="--resources" if args.resources else "--bench" if args.bench else "--strict"
        command=[str(directory/build.RUNNER),str(directory),str(out),mode]+(["--long"] if args.long else [])+(["--with-no-second-barrier"] if args.with_no_second_barrier else [])
        state.update(command=command,build_manifest_sha256=build.digest(directory/"manifest.json"),build_manifest=manifest,gpu_invoked=True)
        persist()
        with(out/"runner.jsonl").open("w")as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        observed=json.loads((out/"result.json").read_text())if(out/"result.json").exists()else{}
        passed=result.returncode==0 and validated_result(observed,mode,args.long,args.with_no_second_barrier)
        state.update(complete=passed,exit_code=result.returncode,result=observed,
            runner_log_sha256=build.digest(out/"runner.jsonl"),
            status=("DRIVER_RESOURCES_ONLY_NOT_QUALITY" if mode=="--resources" else "FULL_HALF_OUTPUT_PARTIAL_BITS_PASS")if passed else"REJECTED_NOT_VALIDATED")
        persist();print(json.dumps(state,indent=2));return 0 if passed else 1
    except(OSError,ValueError,subprocess.SubprocessError)as error:
        state.update(complete=False,status="REJECTED_NOT_VALIDATED",error=str(error));persist();print(json.dumps(state));return 1


if __name__=="__main__":raise SystemExit(main())
