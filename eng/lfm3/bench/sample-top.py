#!/usr/bin/env python3
"""Leaf-function breakdown of a macOS `sample` profile (its "Sort by top of stack" section), grouped by what a ggml-cpu
graph spends time on: products (the quantized dots, the repacked GEMV/GEMM kernels, the activation quantizers),
flash attention, the other ggml ops, thread-pool waits and barriers, and the rest (the host side: .NET, libsystem).

usage: sample-top.py SAMPLE.txt [--top N]
  The counts are 1 ms samples summed over every thread, idle ones included; the groups are also given as shares of the
  samples not blocked in the kernel (running or spinning threads).
Take the profile with `sample PID SECONDS 1 -mayDie -file SAMPLE.txt` while the run is in the phase of interest.
"""
import argparse
import collections
import re
import sys

GROUPS = [
    ("products: repacked GEMV/GEMM", re.compile(r"ggml_gem[mv]_|repack|forward_mul_mat.*tensor_traits")),
    ("products: dot kernels", re.compile(r"ggml_vec_dot_(q|iq|nvfp4|mxfp4|tq|bf16)|vec_dot_q|ggml_vec_dot_f16_unroll")),
    ("products: llamafile sgemm", re.compile(r"tinyBLAS|llamafile")),
    ("products: activation quantize/convert", re.compile(r"quantize_row|quantize_mat|ggml_cpu_fp32_to|fp32_to_bf16|fp32_to_fp16|from_float")),
    ("products: BLAS (Accelerate)", re.compile(r"cblas|sgemm|APL|BNNS|vDSP_mmul|libBLAS")),
    ("flash attention", re.compile(r"flash_attn|ggml_vec_dot_f16$|ggml_vec_mad_f16|ggml_vec_mad1_f16|ggml_vec_scale_f16|ggml_vec_dot_f16")),
    ("matmul driver", re.compile(r"ggml_compute_forward_mul_mat")),
    ("blocked in the kernel (idle threads)", re.compile(r"^(__psynch_cvwait|kevent|__workq_kernreturn|mach_msg2?_trap|__poll_nocancel|__semwait_signal|__select|__read_nocancel|__open_nocancel|__ulock_wait2?|semaphore_wait_trap|__wait4|__sigsuspend)")),
    ("thread pool: spin/barrier", re.compile(r"ggml_barrier|graph_compute_poll|check_for_work|thread_sync|ggml_graph_compute_thread|sched_yield|swtch_pri|ggml_threadpool|CpuWorkerPool|SpinWait|SpinOnce")),
    ("other ggml ops", re.compile(r"ggml_compute_forward_|ggml_vec_|ggml_cpu_|ggml_fp16_to|ggml_bf16_to|rope|rms_norm|ssm_conv|get_rows|set_rows|softmax|silu|swiglu")),
    ("memcpy/memset", re.compile(r"memmove|memcpy|memset|bzero|platform_memmove")),
    (".NET runtime / JIT code", re.compile(r"^\?\?\?|libcoreclr|libclrjit|libSystem.Native|System\.|Microsoft\.|TensorSharp\.")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    text = open(args.sample, errors="replace").read()
    at = text.find("Sort by top of stack")
    if at < 0:
        sys.exit("no 'Sort by top of stack' section")
    rows = []
    for line in text[at:].splitlines()[1:]:
        m = re.match(r"^\s+(.+?)\s+\(in (.+?)\)\s+(\d+)\s*$", line)
        if not m:
            m2 = re.match(r"^\s+(\?\?\?.*?)\s+(\d+)\s*$", line)
            if m2:
                rows.append((m2.group(1), "???", int(m2.group(2))))
            elif line.strip() == "" and rows:
                break
            continue
        rows.append((m.group(1), m.group(2), int(m.group(3))))
    total = sum(n for _, _, n in rows)
    dur = re.search(r"Duration:\s+([0-9.]+)", text)
    print(f"{args.sample}: {total} samples" + (f" over {dur.group(1)} s" if dur else ""))
    groups = collections.Counter()
    for sym, lib, n in rows:
        for name, rx in GROUPS:
            if rx.search(sym) or (name.startswith(".NET") and rx.search(lib)):
                groups[name] += n
                break
        else:
            groups["other: " + lib] += n
    busy = total - groups.get("blocked in the kernel (idle threads)", 0)
    print(f"  ({busy} samples not blocked in the kernel; second column: share of those)")
    for name, n in groups.most_common():
        share = "" if name.startswith("blocked") else f" {100.0 * n / max(1, busy):6.1f}%"
        print(f"  {name:45s} {n:8d} {100.0 * n / total:6.1f}%{share}")
    print("  top functions:")
    for sym, lib, n in sorted(rows, key=lambda r: -r[2])[:args.top]:
        print(f"    {n:8d} {100.0 * n / total:6.1f}%  {sym[:90]}  ({lib})")


if __name__ == "__main__":
    sys.exit(main())
