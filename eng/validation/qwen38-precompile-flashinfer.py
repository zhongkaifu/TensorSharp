#!/usr/bin/env python3
"""CPU-only official FlashInfer JitSpec.build(), without loading kernels or a model.

Use the same Python environment as the server. This preserves the installed
package and arithmetic; it only fills its normal on-disk compilation cache.
Concurrent CPU compilation must be recorded and excluded from timed benchmarks.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--arch", default="12.0f")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--temp-dir", type=Path, default=Path("/dev/shm/qwen38-vllm-runtime"))
    args = parser.parse_args()
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", FLASHINFER_CUDA_ARCH_LIST=args.arch,
                      MAX_JOBS=str(args.max_jobs), TMPDIR=str(args.temp_dir))
    import torch
    import flashinfer
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4
    assert not torch.cuda.is_initialized() and torch.cuda.device_count() == 0
    started = time.monotonic()
    spec = gen_gemm_sm120_module_cutlass_fp4()
    record = {"phase": "cpu_only_compiling", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "module": spec.name, "pid": os.getpid(), "torch": torch.__version__,
              "flashinfer": importlib.metadata.version("flashinfer-python"), "cuda_initialized": False,
              "environment": {key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES", "FLASHINFER_CUDA_ARCH_LIST", "MAX_JOBS", "TMPDIR")},
              "helper_sha256": digest(__file__), "official_api": "flashinfer.jit.gemm.gen_gemm_sm120_module_cutlass_fp4().build(need_lock=True)",
              "measurement_exclusion": "CPU compilation may overlap correctness gates only; exclude all overlapping timing observations."}
    args.status_json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.status_json.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    save()
    try:
        spec.build(need_lock=True)
        assert not torch.cuda.is_initialized()
        library = spec.get_library_path()
        record.update(phase="cpu_only_compilation_succeeded", cuda_initialized=torch.cuda.is_initialized(),
                      elapsed_seconds=time.monotonic() - started, library=str(library),
                      library_bytes=library.stat().st_size, library_sha256=digest(library))
    except BaseException as error:
        record.update(phase="cpu_only_compilation_failed", elapsed_seconds=time.monotonic() - started,
                      cuda_initialized=torch.cuda.is_initialized(), error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()


if __name__ == "__main__":
    main()
