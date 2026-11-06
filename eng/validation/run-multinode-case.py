#!/usr/bin/env python3
"""Drive one multi-node TensorSharp case over ssh and collect its evidence.

Run from an operator machine (Windows or Linux) with ssh/scp on PATH. Node 0,
the first --node, is the driver. Other nodes run TensorSharp.Cli, or the
numeric probe, as workers with the same model, placement and flags. Each node
gets a fresh <remote-out>/<label>-node<K> directory and a generated launch.sh.
The script records the hostname, GPUs, runtime/native hashes before and after
the run (native-loaded.sha256 names the libGgmlOps.so the process loads), and a
git-free source fingerprint. It runs the command with output in
process.log and writes exit.json. Workers start first, detached with setsid and
nohup. The driver starts once each worker logs mesh progress or listens on its
peer port. Every run directory is copied back to OUTPUT/node<K>/.
OUTPUT/summary.json records the exact commands, exit codes, parsed metrics and
an explicit status. "passed" requires every process to exit 0 and the expected
metrics to parse. --dry-run prints the exact commands without running anything.

This tool does not build or sync sources, open rank tunnels or judge numerics.
Compare probe vectors with compare-deepseek41-distributed.py against a local
control, and measure the rank route with parallel-network-probe.py.

Example: node 0 (203.0.113.10, ssh port 22139) drives and node 1
(203.0.113.20, ssh port 22004) works. Rank 0 only dials out, so two nodes need a
single forward, opened on node 0 to node 1's rank port. Keep it low-latency:
OpenSSH >= 9.5 sets TCP_NODELAY on both ends of a '-tt ... -o
ObscureKeystrokeTiming=no' session. A '-N' forward-only session does not, and
Nagle batching then delays every small collective or activation frame.

  # On node 0, before each case (node 0 needs its own key for node 1):
  nohup ssh -tt -o ObscureKeystrokeTiming=no -o ExitOnForwardFailure=yes \\
      -o ServerAliveInterval=15 -L 127.0.0.1:19501:127.0.0.1:19501 \\
      -p 22004 root@203.0.113.20 'sleep infinity' </dev/null >/root/tunnel.log 2>&1 &

  # On the operator machine. In Git Bash, export MSYS_NO_PATHCONV=1 first:
  # otherwise MSYS rewrites /root/... arguments into C:/Program Files/Git/root/...
  python eng/validation/run-multinode-case.py \\
      --node root@203.0.113.10:22139 --node root@203.0.113.20:22004 \\
      --identity ~/.ssh/id_ed25519 --model /root/models/qwen38/model-00001-of-00003.gguf \\
      --mode layer --local-degree 2 --peers 127.0.0.1:19500,127.0.0.1:19501 \\
      --context 4096 --kind bench --label qwen38-layer-2x2-bench \\
      --output artifacts/multinode/qwen38-layer-2x2-bench

  python eng/validation/run-multinode-case.py \\
      --node root@203.0.113.10:22139 --node root@203.0.113.20:22004 \\
      --identity ~/.ssh/id_ed25519 --model /root/models/dsv41/model-00001-of-00004.gguf \\
      --mode tp --local-degree 2 --peers 127.0.0.1:19500,127.0.0.1:19501 \\
      --context 4096 --kind probe --probe-text /root/prompts/probe.txt \\
      --weights-id REPO@REVISION --label dsv41-tp-2x2-probe \\
      --output artifacts/multinode/dsv41-tp-2x2-probe

--kind suite runs many benchmark rows in ONE process per node (TensorSharp.Cli --bench-workload), so a block
pays one model load instead of one per row. Give the rows with --suite-workload SPEC (repeatable) and/or
--suite-file FILE (one spec per line, or a JSON array; read on this machine). A cold row ('cold' or
evict=row|run) drops the model's host-mapped tables (DeepSeek V4.1's Engram) from the page cache on EVERY node
first, inside the processes (the driver tells the workers; no restart, no evict.py). The driver writes
bench-suite.json next to its log; summarize-bench-suite.py turns both into OUTPUT/suite-summary.json:

  python eng/validation/run-multinode-case.py ... --kind suite --label dsv41-v1-cuda --output OUT \\
      --suite-workload label=pp2048-cold,prefill=2048,decode=128,runs=4,mode=random,cold \\
      --suite-workload label=pp2048-warm,prefill=2048,decode=128,runs=4,mode=random \\
      --suite-workload label=r8k-cold,text=/root/s7/r8k.txt,decode=32,runs=2,cold

Arguments after '--' go to every node's command, so model flags stay identical.
Use --driver-arg=--flag (with '=') for driver-only flags, such as server options.
A worker accepts its lower ranks only until TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS
(default 120 s) expires; raise it with --env when the driver starts slowly.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time

SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE", re.I)
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WINDOWS_PATH = re.compile(r"(^|[=,;])[A-Za-z]:[\\/]")
SAFE_REMOTE_DIR = re.compile(r"/[A-Za-z0-9._+/-]*")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
NODE = re.compile(r"(?:([A-Za-z0-9._-]+)@)?(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+)(?::(\d{1,5}))?")
BINARIES = {"cli": "TensorSharp.Cli/bin/TensorSharp.Cli.dll",
            "server": "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll",
            "probe": "eng/validation/DeepSeek41DistributedProbe/bin/Release/net10.0/DeepSeek41DistributedProbe.dll"}
# Sources hashed into each node's source fingerprint. TensorSharp.Backends.Cuda holds the direct CUDA engine
# that --backend cuda runs (Dsv4CudaEngine, its .cu kernels, and the committed PTX that a build without nvcc
# loads); the rest of that engine is in TensorSharp.Models, TensorSharp.Distributed and TensorSharp.Core.
SOURCE_DIRS = ("TensorSharp.GGML.Native", "TensorSharp.Core", "TensorSharp.Runtime", "TensorSharp.Models",
               "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda", "TensorSharp.Distributed",
               "TensorSharp.Cli", "TensorSharp.Chat", "TensorSharp.Server", "TensorSharp.Server.Host")
SOURCE_PATTERNS = ("*.cs", "*.csproj", "*.props", "*.targets", "*.cpp", "*.c", "*.h", "*.hpp",
                   "*.cu", "*.cuh", "*.ptx", "*.cmake", "CMakeLists.txt")
PROGRESS = ("bench suite workload|bench evict|inference benchmark starting|benchmark run|benchmark summary|"
            "Loaded model|Kernel warmup|"
            "TcpCommunicator|Distributed (tensor|layer)|Now listening on|Model load refused|Unhandled exception")
NOTABLE = re.compile(r"Distributed (tensor parallelism|layer splitting):|\[TcpCommunicator\] Rank \d+/\d+ connected|"
                     r"MoE CPU offload active|Context length: using|inference benchmark starting|benchmark summary:|"
                     r"bench suite (starting|complete)")
CRASH_MARKERS = ("GGML_ASSERT(", "GGML_ABORT", "Unhandled exception.", "Model load refused", "CUDA error:")
# Shell function: is anything listening on TCP port $1 on this host? (2 = cannot tell)
LISTEN_FN = ("listening() { if command -v ss >/dev/null 2>&1; then t=$(ss -ltn 2>/dev/null); "
             "elif command -v netstat >/dev/null 2>&1; then t=$(netstat -ltn 2>/dev/null); else return 2; fi; "
             "printf '%s\\n' \"$t\" | awk '{print $4}' | grep -q -E \"[:.]$1\\$\"; }")


class CaseFailure(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def number(value):
    try:
        result = float(value.rstrip(","))
    except ValueError:
        return value
    return result if math.isfinite(result) else None  # NaN is "not measured"; keep JSON strict


class Node:
    def __init__(self, index, spec):
        match = NODE.fullmatch(spec)
        if not match or match.group(2).startswith("-"):
            raise ValueError(f"--node must be user@host[:port], got {spec!r}")
        user, host, port = match.groups()
        self.index, self.port = index, int(port or 22)
        if not 0 < self.port < 65536:
            raise ValueError(f"Invalid ssh port in {spec!r}")
        self.host = host.strip("[]")
        prefix = user + "@" if user else ""
        self.dest = prefix + self.host
        self.scp_dest = prefix + ("[" + self.host + "]" if ":" in self.host else self.host)
        self.role = "driver" if index == 0 else "worker"


def windows_identity(path):
    path = os.path.expanduser(path)
    match = re.fullmatch(r"/([A-Za-z])/(.*)", path)
    if os.name == "nt" and match:
        return f"{match[1].upper()}:/{match[2]}"
    # scp runs from inside --output, so a relative key path must be made absolute.
    return path if os.path.isabs(path) else os.path.abspath(path)


def suite_file_rows(path):
    """Rows of a suite file, as TensorSharp.Cli --bench-suite reads them: a JSON array of objects (key: value),
    or one spec per line with blank lines and '#' comments skipped."""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = []
        for entry in json.loads(text):
            if not isinstance(entry, dict):
                raise ValueError("every JSON suite entry must be an object")
            parts = []
            for key, val in entry.items():
                val = ("1" if val else "0") if isinstance(val, bool) else str(val)
                if "," in val:
                    raise ValueError(f"{key!r} may not contain a comma")
                parts.append(f"{key}={val}")
            rows.append(",".join(parts))
        return rows
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def parse_args(argv):
    extra = []
    if "--" in argv:
        index = argv.index("--")
        argv, extra = argv[:index], argv[index + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", action="append", required=True, help="user@host[:sshport]; repeat; the first is the driver")
    p.add_argument("--identity", help="ssh private key path")
    p.add_argument("--ssh-option", action="append", default=[], help="Extra ssh/scp -o option (repeatable; wins over defaults)")
    p.add_argument("--ssh", default="ssh")
    p.add_argument("--scp", default="scp")
    p.add_argument("--remote-root", default="/root/ts", help="TensorSharp source tree on every node")
    p.add_argument("--dotnet", default="/opt/dotnet/dotnet")
    p.add_argument("--cli-dll", default=BINARIES["cli"], help="Relative to --remote-root unless absolute")
    p.add_argument("--server-dll", default=BINARIES["server"])
    p.add_argument("--probe-dll", default=BINARIES["probe"])
    p.add_argument("--native-dir", help="Prepend this node directory to LD_LIBRARY_PATH and hash its libGgmlOps.so")
    p.add_argument("--model", required=True, help="Model path on the nodes (identical on every node)")
    p.add_argument("--mode", choices=("layer", "tp"), required=True, help="--layer-split N or --tp N on every node")
    p.add_argument("--local-degree", type=int, required=True, help="GPUs per node (N)")
    p.add_argument("--peers", required=True, help="Comma-separated rank endpoints, node 0 first, as each node dials them")
    p.add_argument("--cuda-devices", default="0,1", help="CUDA_VISIBLE_DEVICES on every node ('none' leaves it unset)")
    p.add_argument("--context", type=int, help="MAX_CONTEXT on every node")
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="Environment on every node (repeatable)")
    p.add_argument("--backend", default="ggml_cuda")
    p.add_argument("--kv-cache-dtype", default="f16", help="'none' omits it")
    p.add_argument("--n-cpu-moe", default="0", help="'none' omits it")
    p.add_argument("--driver-arg", action="append", default=[], help="Driver-only argument; write --driver-arg=--flag")
    p.add_argument("--kind", choices=("bench", "probe", "server", "suite"), required=True)
    p.add_argument("--bench-prefill", type=int, default=512)
    p.add_argument("--bench-decode", type=int, default=128)
    p.add_argument("--bench-runs", type=int, default=4)
    p.add_argument("--bench-random-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--discard-runs", type=int, default=1, help="Initial benchmark runs excluded from medians")
    p.add_argument("--suite-workload", action="append", default=[], metavar="SPEC",
                   help="--kind suite: one TensorSharp.Cli --bench-workload row (repeatable; rows run in this order)")
    p.add_argument("--suite-file", type=Path, help="--kind suite: rows from a local file (one spec per line, '#' comments, "
                                                   "or a JSON array of objects), after the --suite-workload rows")
    p.add_argument("--suite-evict-measure", choices=("none", "sampled", "full"), default="sampled",
                   help="--kind suite: how each node checks that a cold row's tables left its page cache")
    p.add_argument("--suite-allow-unverified", action="store_true",
                   help="--kind suite: keep running after a cold row's eviction is not verified on every node "
                        "(default: the CLI's --bench-evict-strict stops the suite there)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--probe-text", help="UTF-8 text file on the nodes")
    source.add_argument("--probe-tokens", help="JSON token-id file on the nodes")
    p.add_argument("--probe-chunk", type=int, default=3)
    p.add_argument("--probe-decode-count", type=int, default=4)
    p.add_argument("--weights-id", help="Immutable checkpoint identity recorded by the probe")
    p.add_argument("--server-port", type=int, default=5100)
    p.add_argument("--server-command", help="Local command template run once the server is ready; {url} {output} {label} "
                   "are substituted. A JSON array avoids shell-style splitting (use it for Windows paths)")
    p.add_argument("--server-local-port", type=int, default=0, help="Local end of the ssh forward for {url} (0 = any free port)")
    p.add_argument("--server-forward", choices=("auto", "nodelay", "plain"), default="auto",
                   help="nodelay = '-tt -o ObscureKeystrokeTiming=no' (OpenSSH >= 9.5), plain = '-N'")
    p.add_argument("--server-hold", type=int, default=0, help="Seconds to keep a ready server up when no command is given")
    p.add_argument("--stop-grace", type=int, default=60, help="Seconds after SIGINT, then after SIGTERM, before escalating")
    p.add_argument("--remote-out", default="/root/runs", help="Base for per-node run directories")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True, help="Fresh local directory")
    p.add_argument("--timeout", type=int, default=7200, help="Seconds for the driver run (bench/probe) or server command")
    p.add_argument("--startup-timeout", type=int, default=1800, help="Seconds for the server driver to become ready")
    p.add_argument("--worker-ready-timeout", type=int, default=60,
                   help="Seconds to wait for a worker mesh signal; a live worker is then assumed ready")
    p.add_argument("--shutdown-timeout", type=int, default=180, help="Seconds for workers to exit after the driver")
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--ssh-retries", type=int, default=6, help="Consecutive failed status polls tolerated per node")
    p.add_argument("--collect-timeout", type=int, default=1800)
    p.add_argument("--collect-exclude", action="append", default=[],
                   help="File name pattern left on the node (moved to <run>.kept/, hashes recorded in "
                        "kept-files.sha256), e.g. '*.f32' for large probe vectors (repeatable)")
    p.add_argument("--gpu-telemetry-ms", type=int, default=1000, help="nvidia-smi sampling into gpu.csv; 0 disables")
    p.add_argument("--no-source-fingerprint", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the exact commands; run nothing")
    a = p.parse_args(argv)
    a.extra = extra
    try:
        a.nodes = [Node(i, spec) for i, spec in enumerate(a.node)]
    except ValueError as error:
        p.error(str(error))
    a.peer_list = [item.strip() for item in a.peers.split(",")]
    if len(a.nodes) < 2:
        p.error("A multi-node case needs at least two --node values")
    if len(a.peer_list) != len(a.nodes):
        p.error(f"--peers lists {len(a.peer_list)} endpoints for {len(a.nodes)} nodes")
    a.peer_ports = []
    for endpoint in a.peer_list:
        host, sep, port = endpoint.rpartition(":")
        if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
            p.error(f"Peer endpoint must be host:port, got {endpoint!r}")
        a.peer_ports.append(int(port))
    # summarize-parallel-cli-benchmark.py rejects runs without decode timings, so require them.
    if a.local_degree < 1 or a.bench_runs < 1 or a.bench_prefill < 1 or a.bench_decode < 1:
        p.error("Need a positive --local-degree, --bench-runs, --bench-prefill and --bench-decode")
    if not LABEL.fullmatch(a.label):
        p.error("--label may contain only letters, digits, '.', '_' and '-'")
    if a.kind == "probe" and (not a.weights_id or not (a.probe_text or a.probe_tokens)):
        p.error("--kind probe requires --weights-id and one of --probe-text/--probe-tokens")
    if a.kind != "server" and (a.server_command or a.server_hold):
        p.error("--server-command and --server-hold require --kind server")
    if a.kind == "bench" and a.discard_runs >= a.bench_runs:
        p.error("--discard-runs must leave at least one measured run")
    a.suite_rows = list(a.suite_workload)
    if a.suite_file:
        try:
            a.suite_rows += suite_file_rows(a.suite_file)
        except (OSError, ValueError) as error:
            p.error(f"--suite-file: {error}")
    if a.kind == "suite" and not a.suite_rows:
        p.error("--kind suite needs at least one --suite-workload or a --suite-file")
    if a.kind != "suite" and a.suite_rows:
        p.error("--suite-workload/--suite-file require --kind suite")
    for row in a.suite_rows:
        if not row.strip() or "\n" in row:
            p.error(f"Invalid suite row {row!r}")
        if WINDOWS_PATH.search(row):
            p.error(f"Suite row {row!r} names a Windows path; text=/tokens= files are read on the driver node.")
    remote_paths = {"--remote-root": a.remote_root, "--dotnet": a.dotnet, "--remote-out": a.remote_out, "--model": a.model}
    for name in ("cli_dll", "server_dll", "probe_dll", "native_dir", "probe_text", "probe_tokens"):
        if getattr(a, name):
            remote_paths["--" + name.replace("_", "-")] = getattr(a, name)
    hint = " From Git Bash, export MSYS_NO_PATHCONV=1: MSYS rewrites /paths passed to native Windows programs."
    for name, value in remote_paths.items():
        if WINDOWS_PATH.search(value) or "\n" in value:
            p.error(f"{name} must be a POSIX path on the nodes, got {value!r}.{hint}")
    for name in ("--remote-root", "--dotnet", "--remote-out", "--model"):
        if not remote_paths[name].startswith("/"):
            p.error(f"{name} must be absolute on the nodes")
    if not SAFE_REMOTE_DIR.fullmatch(a.remote_out.rstrip("/") or "/"):
        p.error("--remote-out may contain only letters, digits and . _ + / - (it is copied back with scp)")
    a.remote_out = a.remote_out.rstrip("/") or "/"
    a.env_pairs = {}
    for item in a.env:
        key, sep, value = item.partition("=")
        if not sep or not ENV_NAME.fullmatch(key):
            p.error(f"--env must be NAME=VALUE, got {item!r}")
        if WINDOWS_PATH.search(value):
            p.error(f"--env {key} looks like a Windows path.{hint}")
        a.env_pairs[key] = value
    for value in a.extra + a.driver_arg:
        if WINDOWS_PATH.search(value):
            p.error(f"Argument {value!r} looks like a Windows path.{hint}")
    if a.identity:
        a.identity = windows_identity(a.identity)
    a.output = a.output.resolve()
    if a.output.exists():
        p.error("--output must be fresh; preserve earlier evidence")
    return a


# ---------------------------------------------------------------- remote scripts

def ssh_options(a):
    # ssh keeps the first value it reads for an option, so operator options win.
    options = []
    for option in a.ssh_option:
        options += ["-o", option]
    options += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    if a.identity:
        options += ["-i", a.identity]
    return options


def ssh_argv(a, node, script):
    return [a.ssh, *ssh_options(a), "-p", str(node.port), "-T", node.dest, "sh -c " + shlex.quote(script)]


def scp_argv(a, node, remote, local):
    return [a.scp, *ssh_options(a), "-P", str(node.port), "-r", "-q", f"{node.scp_dest}:{remote}", local]


def stage_script(a, run, port):
    q = shlex.quote
    lines = ["set -u", f"mkdir -p {q(a.remote_out)} || exit 5",
             f"if [ -e {q(run)} ]; then echo {q('remote run directory already exists: ' + run)} >&2; exit 3; fi",
             LISTEN_FN,
             f"if listening {port}; then echo {q(f'TCP port {port} is already listening on this node (leftover rank?)')} >&2; exit 4; fi",
             f"mkdir {q(run)} || exit 5",
             f"cat > {q(run + '/launch.sh')} && chmod 700 {q(run + '/launch.sh')}"]
    return "\n".join(lines)


def secret_stage_script(run):
    return f"umask 077 && cat > {shlex.quote(run + '/secret-env.sh')}"


def start_script(run):
    # 'setsid -f' runs the launcher in its own session and process group with
    # default SIGINT handling ('cmd &' in a script would start it ignored).
    return (f"cd {shlex.quote(run)} && if setsid --help 2>&1 | grep -q -e --fork; then "
            "setsid -f nohup sh launch.sh </dev/null >launcher.log 2>&1; "
            "else (setsid nohup sh launch.sh </dev/null >launcher.log 2>&1 &); fi")


def poll_script(run, port=None, health=None):
    q = shlex.quote
    lines = [f"cd {q(run)} 2>/dev/null || {{ echo state=missing; exit 0; }}",
             "[ -f launcher.pid ] && echo pidfile=1",
             "pid=$(cat launcher.pid 2>/dev/null)",
             'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo alive=1; else echo alive=0; fi',
             "if [ -f exit.json ]; then printf 'exit=%s\\n' \"$(tr -d '\\n' < exit.json)\"; fi",
             "grep -E -q 'waiting for rank|connected to all peers' process.log 2>/dev/null && echo mesh=1",
             "grep -E -q 'Now listening on|Application started' process.log 2>/dev/null && echo serving=1"]
    if port is not None:
        lines += [LISTEN_FN, f"listening {port} && echo listen=1"]
    if health is not None:
        url = f"http://127.0.0.1:{health}/health"
        lines.append(f"(curl -fsS -m 3 -o /dev/null {url} || wget -q -T 3 -O /dev/null {url}) >/dev/null 2>&1 && echo health=1")
    lines += [f"printf 'progress=%s\\n' \"$(grep -E {q(PROGRESS)} process.log 2>/dev/null | tail -n 1 | cut -c1-200)\"",
              "exit 0"]
    return "\n".join(lines)


def signal_script(run, sig):
    # The launcher leads its own process group; signal the whole group.
    return (f"pid=$(cat {shlex.quote(run + '/launcher.pid')} 2>/dev/null); [ -n \"$pid\" ] || exit 0; "
            f"kill -s {sig} -- \"-$pid\" 2>/dev/null || kill -{sig} \"-$pid\" 2>/dev/null; exit 0")


def node_command(a, k, run):
    root = a.remote_root
    place = ["--layer-split" if a.mode == "layer" else "--tp", str(a.local_degree),
             "--tp-node-id", str(k), "--tp-peers", a.peers]
    if a.kind == "probe":
        source = ["--text", a.probe_text] if a.probe_text else ["--tokens", a.probe_tokens]
        dll = posixpath.join(root, a.probe_dll)
        argv = [a.dotnet, dll, "--model", a.model, *source, "--weights-id", a.weights_id,
                "--output", run + "/probe", *place, "--chunk", str(a.probe_chunk),
                "--decode-count", str(a.probe_decode_count), "--backend", a.backend]
    else:
        server = a.kind == "server" and k == 0
        dll = posixpath.join(root, a.server_dll if server else a.cli_dll)
        argv = [a.dotnet, dll, "--model", a.model, "--backend", a.backend]
        if a.kv_cache_dtype != "none":
            argv += ["--kv-cache-dtype", a.kv_cache_dtype]
        if a.n_cpu_moe != "none":
            argv += ["--n-cpu-moe", a.n_cpu_moe]
        argv += place
        if server:
            argv += ["--host", "127.0.0.1", "--port", str(a.server_port)]
        elif a.kind == "bench":
            # Workers receive the same flags; they enter the mirror loop before benchmarking.
            argv += ["--benchmark", "--bench-prefill", str(a.bench_prefill), "--bench-decode", str(a.bench_decode),
                     "--bench-runs", str(a.bench_runs)] + (["--bench-random-tokens"] if a.bench_random_tokens else [])
        elif a.kind == "suite":
            # Every node parses the rows. A worker never runs them and never reads their prompt files: it enters
            # the mirror loop, where the driver's cold rows reach it as an eviction op.
            argv += ["--bench-prefill", str(a.bench_prefill), "--bench-decode", str(a.bench_decode), "--bench-runs", str(a.bench_runs)]
            argv += ["--bench-random-tokens"] if a.bench_random_tokens else []
            for row in a.suite_rows:
                argv += ["--bench-workload", row]
            argv += ["--bench-evict-measure", a.suite_evict_measure, "--bench-json", run + "/bench-suite.json"]
            argv += [] if a.suite_allow_unverified else ["--bench-evict-strict"]
    return argv + a.extra + (a.driver_arg if k == 0 else []), dll


def case_environment(a):
    env = {"DOTNET_ROOT": posixpath.dirname(a.dotnet), "TENSORSHARP_GGML_NO_UPDATE": "1"}
    if a.cuda_devices != "none":
        env["CUDA_VISIBLE_DEVICES"] = a.cuda_devices
    if a.context:
        env["MAX_CONTEXT"] = str(a.context)
    if a.kind == "probe":
        # The probe has no --kv-cache-dtype/--n-cpu-moe flags; it reads these variables.
        if a.kv_cache_dtype != "none":
            env["KV_CACHE_DTYPE"] = a.kv_cache_dtype
        if a.n_cpu_moe != "none":
            env["TS_N_CPU_MOE"] = a.n_cpu_moe
    env.update(a.env_pairs)
    public = {k: v for k, v in env.items() if not SECRET_NAME.search(k)}
    secret = {k: v for k, v in env.items() if SECRET_NAME.search(k)}
    return public, secret


def redacted(a):
    public, secret = case_environment(a)
    return {**public, **{k: "<redacted>" for k in secret}}


def launcher_text(a, node, run, argv, dll, public_env):
    q = shlex.quote
    bin_dir, main = posixpath.split(dll)
    lines = ["#!/bin/sh",
             f"# Generated by eng/validation/run-multinode-case.py: {a.label} node {node.index} ({node.role}).",
             "# Signals reach the command; this shell waits for it, then writes exit.json.",
             f"run={q(run)} root={q(a.remote_root)} dotnet={q(a.dotnet)} bin={q(bin_dir)} main={q(main)}",
             'cd "$run" || exit 90',
             "echo $$ > launcher.pid",
             "trap 'echo \"launcher: signal at $(date -u +%Y-%m-%dT%H:%M:%SZ); waiting for the command\"' INT TERM HUP"]
    if a.native_dir:
        lines.append(f"export LD_LIBRARY_PATH={q(a.native_dir)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}")
    lines += [f"export {key}={q(value)}" for key, value in public_env.items()]
    lines += ["if [ -f secret-env.sh ]; then . ./secret-env.sh; rm -f secret-env.sh; fi",
              "now() { d=$(date +%s.%N); case $d in *N) date +%s ;; *) echo \"$d\" ;; esac; }",
              "t0=$(now) t1=null",
              "finish() { printf '{\"exit_code\": %s, \"phase\": \"%s\", \"launcher_started_unix\": %s, "
              "\"command_started_unix\": %s, \"ended_unix\": %s}\\n' \"$1\" \"$2\" \"$t0\" \"$t1\" \"$(now)\" "
              "> exit.json.tmp && mv exit.json.tmp exit.json; }",
              # Every path GgmlNative.ImportResolver tries for libGgmlOps.so, in its order (GetCandidatePaths in
              # TensorSharp.Backends.GGML/GgmlNative.cs): the app directory (the host resolves symlinks), then
              # TensorSharp.GGML.Native/{build,build/Release,build-windows,build-windows/Release} under each repo
              # root from the app directory upwards (a directory holding TensorSharp.slnx, TensorSharp.sln,
              # Seq2SeqSharp.sln or a .git directory), then the runtime's default probing via LD_LIBRARY_PATH.
              "ggml_candidates() ( d=$(cd \"$bin\" 2>/dev/null && pwd -P) || d=$bin; echo \"${d%/}/libGgmlOps.so\"; "
              "while :; do if [ -f \"$d/TensorSharp.slnx\" ] || [ -f \"$d/TensorSharp.sln\" ] || "
              "[ -f \"$d/Seq2SeqSharp.sln\" ] || [ -d \"$d/.git\" ]; then for s in build build/Release build-windows "
              "build-windows/Release; do echo \"${d%/}/TensorSharp.GGML.Native/$s/libGgmlOps.so\"; done; fi; "
              "p=$(dirname \"$d\"); [ \"$p\" = \"$d\" ] && break; d=$p; done; "
              "IFS=:; for d in ${LD_LIBRARY_PATH:-}; do [ -n \"$d\" ] && echo \"$d/libGgmlOps.so\"; done; exit 0 )",
              "runtime() { { for f in \"$bin/$main\" \"$bin\"/TensorSharp*.dll; do [ -f \"$f\" ] && sha256sum \"$f\"; done; "
              "ggml_candidates | while IFS= read -r f; do [ -f \"$f\" ] && sha256sum \"$f\"; done; } | LC_ALL=C sort -u -k2; }",
              # The library the process loads: the first candidate that exists. The resolver also skips a
              # candidate that dlopen rejects; this check cannot see that case.
              "loaded_native() { ggml_candidates | while IFS= read -r f; do [ -f \"$f\" ] && { sha256sum \"$f\"; break; }; done; return 0; }",
              "hostname > hostname.txt 2>&1; nvidia-smi -L > nvidia-smi-L.txt 2>&1",
              "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv > gpus.csv 2>&1",
              "\"$dotnet\" --version > dotnet-version.txt 2>&1",
              "env | grep -E '^(CUDA_|MAX_CONTEXT|KV_CACHE|LD_LIBRARY_PATH|DOTNET_|NCCL_|GGML_|TS_|TENSORSHARP_)' "
              "| grep -v -i -E '^[^=]*(TOKEN|SECRET|PASSW|CREDENTIAL|KEY)' | LC_ALL=C sort > effective-env.txt",
              "runtime > runtime-before.sha256", "loaded_native > native-loaded.sha256"]
    shard = re.search(r"-\d{5}-of-(\d{5})\.gguf$", a.model)
    model_files = q(a.model[:shard.start()]) + f"-*-of-{shard.group(1)}.gguf" if shard else q(a.model)
    lines.append(f"stat -L -c '%s %Y %n' {model_files} > model-files.txt 2>&1")
    if not a.no_source_fingerprint:
        dirs = SOURCE_DIRS + (("eng/validation/DeepSeek41DistributedProbe",) if a.kind == "probe" else ())
        names = " -o ".join("-name " + q(pattern) for pattern in SOURCE_PATTERNS)
        lines += [f"( cd \"$root\" || exit 0; set --; for d in {' '.join(dirs)}; do [ -d \"$d\" ] && set -- \"$@\" \"$d\"; done",
                  "  [ $# -gt 0 ] && find \"$@\" -type d \\( -name bin -o -name obj -o -name 'build*' -o -name node_modules "
                  f"-o -name .git \\) -prune -o -type f \\( {names} \\) -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum",
                  "  for f in Directory.Build.props Directory.Build.targets Directory.Packages.props global.json; do "
                  "[ -f \"$f\" ] && sha256sum \"$f\"; done ) > source-files.sha256 2>/dev/null",
                  "sha256sum < source-files.sha256 | cut -d' ' -f1 > source-fingerprint.txt",
                  "if command -v git >/dev/null 2>&1; then ( cd \"$root\" && git rev-parse HEAD; "
                  "git -C ExternalProjects/ggml rev-parse HEAD; "
                  "git -C ExternalProjects/ggml status --porcelain --untracked-files=all | head -n 50 ) > git-identity.txt 2>&1; fi"]
    lines.append('cd "$root" || { finish 91 cd-root; exit 91; }')
    if a.gpu_telemetry_ms > 0:
        lines.append("tele=; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi "
                     "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm --format=csv "
                     f"-lms {a.gpu_telemetry_ms} > \"$run/gpu.csv\" 2>&1 & tele=$!; fi")
    lines += ["t1=$(now)",
              " ".join(q(arg) for arg in argv) + ' > "$run/process.log" 2>&1 < /dev/null',
              "code=$?"]
    if a.gpu_telemetry_ms > 0:
        lines.append('[ -n "$tele" ] && kill "$tele" 2>/dev/null')
    lines += ['cd "$run" && runtime > runtime-after.sha256', 'finish "$code" exited', 'exit "$code"']
    return "\n".join(lines) + "\n"


def build_plan(a):
    public, secret = case_environment(a)
    plan = []
    for node in a.nodes:
        run = f"{a.remote_out}/{a.label}-node{node.index}"
        argv, dll = node_command(a, node.index, run)
        launcher = launcher_text(a, node, run, argv, dll, public)
        port = a.peer_ports[node.index]
        plan.append({"node": node, "run": run, "argv": argv, "dll": dll, "launcher": launcher, "port": port,
                     "secret": "".join(f"export {k}={shlex.quote(v)}\n" for k, v in secret.items()),
                     "stage_argv": ssh_argv(a, node, stage_script(a, run, port)),
                     "secret_argv": ssh_argv(a, node, secret_stage_script(run)) if secret else None,
                     "start_argv": ssh_argv(a, node, start_script(run)),
                     "poll_argv": ssh_argv(a, node, poll_script(run, port if node.index else None,
                                                                a.server_port if a.kind == "server" and not node.index else None)),
                     "collect_argv": scp_argv(a, node, run, f"node{node.index}")})
    return plan


# ---------------------------------------------------------------- log parsing

def log_evidence(log):
    evidence = {"bytes": len(log.encode("utf-8", "replace"))}
    loads = re.findall(r"Loaded model (\S+) architecture=(\S+) contextLength=(\d+) kvCacheDtype=(\S+) elapsedMs=([0-9.]+)", log)
    if loads:
        name, architecture, context, dtype, ms = loads[-1]
        evidence["model_load"] = {"file": name, "architecture": architecture, "context_length": int(context),
                                  "kv_cache_dtype": dtype, "elapsed_ms": float(ms), "source": "cli"}
    else:
        server = re.findall(r"Loaded model .*? in ([0-9.]+) ms", log)
        if server:
            evidence["model_load"] = {"elapsed_ms": float(server[-1]), "source": "server"}
    evidence["kernel_warmup_ms"] = [float(x) for x in re.findall(r"Kernel warmup completed in ([0-9.]+) ms", log)]
    lines = log.splitlines()
    evidence["notable_lines"] = [line.strip()[:400] for line in lines if NOTABLE.search(line)][:40]
    evidence["crash_lines"] = [line.strip()[:400] for line in lines
                               if any(marker in line for marker in CRASH_MARKERS)][:20]
    return evidence


def logged_runs(log):
    rows = []
    for match in re.finditer(r"benchmark run (\d+)/(\d+): ([^\r\n]*)", log):
        row = {"run": int(match[1]), "runs": int(match[2])}
        row.update({key: number(value) for key, value in re.findall(r"(\w+)=(\S+)", match[3])})
        rows.append(row)
    return rows


def read_text(path, limit=8000):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[:limit]
    except OSError:
        return None


def hash_lines(path):
    text = read_text(path, 10 ** 6)
    if text is None:
        return None
    result = {}
    for line in text.splitlines():
        digest, _, name = line.partition("  ")
        if name:
            result[name.strip()] = digest
    return result


def loaded_native(directory):
    """The libGgmlOps.so a node's process loads, from native-loaded.sha256: {"path", "sha256"} or None."""
    for name, digest in (hash_lines(Path(directory) / "native-loaded.sha256") or {}).items():
        return {"path": name, "sha256": digest}
    return None


def load_summarizer():
    """Reuse the existing CLI benchmark summarizer rather than a second parser."""
    path = Path(__file__).with_name("summarize-parallel-cli-benchmark.py")
    spec = importlib.util.spec_from_file_location("summarize_parallel_cli_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def ssh_client_version(a):
    try:
        result = subprocess.run([a.ssh, "-V"], capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    text = (result.stderr + result.stdout).strip()
    match = re.search(r"OpenSSH_(?:for_Windows_)?(\d+)\.(\d+)", text)
    return text.splitlines()[0] if text else None, (int(match[1]), int(match[2])) if match else None


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def state_done(state):
    return bool(state) and ("exit" in state or (state.get("pidfile") == "1" and state.get("alive") == "0"))


def parse_state(text):
    state = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            state[key.strip()] = value
    if "exit" in state:
        try:
            state["exit"] = json.loads(state["exit"])
        except ValueError:
            state["exit"] = {"unparsed": state["exit"]}
    return state


# ---------------------------------------------------------------- execution

class Case:
    def __init__(self, a, plan):
        self.a, self.plan, self.output = a, plan, a.output
        self.by_index = {p["node"].index: p for p in plan}
        self.state = {p["node"].index: {"staged": False, "started": False, "done": False} for p in plan}
        self.failures, self.warnings, self.metrics = [], [], {}
        version_text, self.ssh_version = ssh_client_version(a)
        public, secret = case_environment(a)
        self.summary = {
            "schema_version": 1, "status": "running", "label": a.label, "kind": a.kind, "mode": a.mode,
            "placement_flag": "--layer-split" if a.mode == "layer" else "--tp", "local_degree": a.local_degree,
            "node_count": len(plan), "total_gpus": len(plan) * a.local_degree, "peers": a.peers,
            "model": a.model, "remote_root": a.remote_root, "dotnet": a.dotnet, "backend": a.backend,
            "environment": redacted(a), "secret_environment_names": sorted(secret),
            "extra_arguments": a.extra, "driver_arguments": a.driver_arg,
            "request": self.request(), "ssh_client": version_text, "started_utc": utc_now(),
            "tool": {"path": "eng/validation/run-multinode-case.py", "sha256": sha(__file__)},
            "nodes": [{"index": p["node"].index, "role": p["node"].role, "ssh_destination": p["node"].dest,
                       "ssh_port": p["node"].port, "peer_endpoint": a.peer_list[p["node"].index],
                       "remote_run_dir": p["run"], "local_dir": f"node{p['node'].index}",
                       "command": p["argv"], "command_shell": shlex.join(p["argv"]),
                       "launcher_sha256": hashlib.sha256(p["launcher"].encode()).hexdigest(),
                       "start_command": p["start_argv"], "collect_command": p["collect_argv"]} for p in plan],
            "limitations": [
                "Model identity is the size/mtime of the model shards on each node (model-files.txt), not a content hash.",
                "Worker readiness is inferred from mesh log lines or the rank port listening; it is not a handshake.",
                "Benchmark medians exclude the first --discard-runs runs; that does not establish steady state.",
                "Rates include whatever route --peers selects (e.g. an SSH forward); record and measure it separately.",
                "Probe vectors are not judged here; compare them with a local control using compare-deepseek41-distributed.py.",
                "A server command reaches node 0 through an ssh forward from this machine; its HTTP timing includes that path."]}

    def request(self):
        a = self.a
        if a.kind == "bench":
            return {"prefill": a.bench_prefill, "decode": a.bench_decode, "runs": a.bench_runs,
                    "random_tokens": a.bench_random_tokens, "discard_runs": a.discard_runs}
        if a.kind == "probe":
            return {"text": a.probe_text, "tokens": a.probe_tokens, "chunk": a.probe_chunk,
                    "decode_count": a.probe_decode_count, "weights_id": a.weights_id}
        return {"port": a.server_port, "command_template": a.server_command, "hold_seconds": a.server_hold}

    def log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {self.a.label}: {message}", flush=True)

    def fail(self, message):
        if message not in self.failures:
            self.failures.append(message)
            self.log("FAIL " + message)

    def save(self):
        self.summary.update(failures=self.failures, warnings=self.warnings, metrics=self.metrics,
                            node_states={f"node{k}": {key: value for key, value in state.items()
                                                      if key != "last" and not key.endswith("_monotonic")}
                                         for k, state in self.state.items()})
        (self.output / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")

    def remote(self, k, script, data=None, timeout=120):
        argv = ssh_argv(self.a, self.by_index[k]["node"], script)
        stdin = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
        try:
            result = subprocess.run(argv, capture_output=True, timeout=timeout, **stdin)
        except subprocess.TimeoutExpired:
            return 255, "", "ssh timed out"
        return (result.returncode, result.stdout.decode("utf-8", "replace"),
                result.stderr.decode("utf-8", "replace"))

    def poll(self, k, port=None, health=None):
        error = ""
        for attempt in range(self.a.ssh_retries + 1):
            code, out, error = self.remote(k, poll_script(self.by_index[k]["run"], port, health), timeout=60)
            if code == 0:
                state = parse_state(out)
                self.state[k]["last"] = state
                if state_done(state):
                    self.state[k]["done"] = True
                return state
            time.sleep(min(5.0, self.a.poll_interval))
        raise CaseFailure(f"node{k}: {self.a.ssh_retries + 1} consecutive status polls failed: {error.strip()[-300:]}")

    def stop(self, k, sequence):
        run = self.by_index[k]["run"]
        record, state = {"signals": [], "forced": False}, self.state[k].get("last")
        for sig, wait in sequence:
            if state_done(state):
                break
            self.log(f"node{k}: sending SIG{sig} to the launcher process group")
            self.remote(k, signal_script(run, sig), timeout=60)
            record["signals"].append(sig)
            record["forced"] |= sig == "KILL"
            deadline = time.monotonic() + wait
            while True:
                state = self.poll(k)
                if state_done(state) or time.monotonic() >= deadline:
                    break
                time.sleep(2)
        record["final_state"] = {key: value for key, value in (state or {}).items() if key != "progress"}
        self.state[k]["stop"] = record
        if record["forced"]:
            self.fail(f"node{k} needed SIGKILL")
        elif not state_done(state):
            self.fail(f"node{k} still running after {record['signals']}")
        return record

    def stage(self):
        for p in self.plan:
            k = p["node"].index
            code, _, error = self.remote(k, stage_script(self.a, p["run"], p["port"]), data=p["launcher"].encode())
            if code in (3, 4, 5, 255):
                raise CaseFailure(f"node{k}: staging refused (exit {code}): {error.strip()[-400:]}")
            self.state[k]["staged"] = True
            if code != 0:
                raise CaseFailure(f"node{k}: staging failed (exit {code}): {error.strip()[-400:]}")
            if p["secret"]:
                code, _, error = self.remote(k, secret_stage_script(p["run"]), data=p["secret"].encode())
                if code != 0:
                    raise CaseFailure(f"node{k}: secret environment staging failed (exit {code})")
        self.log(f"staged {len(self.plan)} run directories under {self.a.remote_out}")

    def start(self, k):
        code, _, error = self.remote(k, start_script(self.by_index[k]["run"]))
        if code != 0:
            raise CaseFailure(f"node{k}: start failed (exit {code}): {error.strip()[-400:]}")
        self.state[k].update(started=True, started_utc=utc_now(), started_monotonic=time.monotonic())
        node = self.by_index[k]["node"]
        self.log(f"node{k} ({node.role}) started on {node.dest}")

    def start_workers(self):
        workers = sorted((k for k in self.by_index if k), reverse=True)  # higher ranks listen first
        for k in workers:
            self.start(k)
        for k in workers:
            port, deadline = self.by_index[k]["port"], time.monotonic() + self.a.worker_ready_timeout
            while True:
                state = self.poll(k, port=port)
                if state_done(state):
                    raise CaseFailure(f"node{k} worker exited before the driver started: {state.get('exit')}")
                if state.get("mesh") == "1" or state.get("listen") == "1":
                    readiness = "mesh_log" if state.get("mesh") == "1" else f"listening:{port}"
                    break
                if time.monotonic() >= deadline:
                    if state.get("alive") != "1":
                        raise CaseFailure(f"node{k} worker launcher is not running; see node{k}/launcher.log")
                    readiness = "assumed_alive_after_timeout"
                    self.warnings.append(f"node{k}: no mesh signal within {self.a.worker_ready_timeout}s; started the driver anyway")
                    break
                time.sleep(min(2.0, self.a.poll_interval))
            self.state[k].update(readiness=readiness,
                                 ready_seconds=round(time.monotonic() - self.state[k]["started_monotonic"], 1))
            self.log(f"node{k} ready ({readiness})")

    def check_workers(self):
        # The driver's Dispose broadcasts shutdown, so a healthy worker can exit before the
        # driver process does. analyze() judges each exit code; callers decide on timing.
        for k in self.by_index:
            if k and not self.state[k]["done"]:
                state = self.poll(k)
                if state_done(state):
                    self.log(f"node{k} worker exited: {state.get('exit')}")
                    self.state[k]["early_exit_monotonic"] = time.monotonic()
        early = [s["early_exit_monotonic"] for s in self.state.values() if "early_exit_monotonic" in s]
        return min(early) if early else None

    def wait_driver(self):
        deadline, last = time.monotonic() + self.a.timeout, None
        while True:
            state = self.poll(0)
            if state_done(state):
                self.log(f"driver finished: {state.get('exit')}")
                return
            if state.get("progress") and state["progress"] != last:
                last = state["progress"]
                self.log("driver: " + last)
            early = self.check_workers()
            if early is not None and time.monotonic() - early > self.a.shutdown_timeout:
                self.fail("driver kept running after a worker exited; stopping it")
                self.stop(0, (("TERM", 30), ("KILL", 15)))
                return
            if time.monotonic() >= deadline:
                self.fail(f"driver did not finish within --timeout {self.a.timeout}s; stopping it")
                self.stop(0, (("INT", 30), ("TERM", 30), ("KILL", 15)))
                return
            time.sleep(self.a.poll_interval)

    def serve(self):
        a, server = self.a, self.metrics.setdefault("server", {})
        deadline, last = time.monotonic() + a.startup_timeout, None
        while True:
            state = self.poll(0, health=a.server_port)
            if state_done(state):
                raise CaseFailure(f"server driver exited before it became ready: {state.get('exit')}")
            if state.get("health") == "1" or state.get("serving") == "1":
                server.update(ready=True, readiness="health" if state.get("health") == "1" else "log",
                              ready_seconds=round(time.monotonic() - self.state[0]["started_monotonic"], 1))
                self.log(f"server ready after {server['ready_seconds']}s ({server['readiness']})")
                break
            if state.get("progress") and state["progress"] != last:
                last = state["progress"]
                self.log("driver: " + last)
            if self.check_workers() is not None:
                raise CaseFailure("a worker exited while the server was starting")
            if time.monotonic() >= deadline:
                raise CaseFailure(f"server not ready within --startup-timeout {a.startup_timeout}s")
            time.sleep(a.poll_interval)
        if a.server_command:
            self.run_server_command(server)
        elif a.server_hold:
            end = time.monotonic() + a.server_hold
            while time.monotonic() < end:
                if state_done(self.poll(0)):
                    raise CaseFailure("server exited during --server-hold")
                time.sleep(min(a.poll_interval, max(0.0, end - time.monotonic())))
        self.log("stopping the server driver (SIGINT, then SIGTERM)")
        record = self.stop(0, (("INT", a.stop_grace), ("TERM", a.stop_grace), ("KILL", 15)))
        server["stop_signals"] = record["signals"]

    def forward_argv(self, local_port):
        a, node = self.a, self.by_index[0]["node"]
        mode = a.server_forward
        if mode == "auto":
            mode = "nodelay" if os.name != "nt" and self.ssh_version and self.ssh_version >= (9, 5) else "plain"
        spec = f"127.0.0.1:{local_port}:127.0.0.1:{a.server_port}"
        base = [a.ssh, *ssh_options(a), "-p", str(node.port), "-o", "ExitOnForwardFailure=yes", "-L", spec]
        if mode == "nodelay":
            return mode, base[:1] + ["-tt", "-o", "ObscureKeystrokeTiming=no"] + base[1:] + [node.dest, "exec sleep 2147483647"]
        return mode, base + ["-N", node.dest]

    def run_server_command(self, server):
        a = self.a
        local_port = a.server_local_port or free_port()
        mode, forward = self.forward_argv(local_port)
        url = f"http://127.0.0.1:{local_port}"
        template = json.loads(a.server_command) if a.server_command.lstrip().startswith("[") else shlex.split(a.server_command)
        argv = [str(part).replace("{url}", url).replace("{output}", str(self.output)).replace("{label}", a.label)
                for part in template]
        record = server["command"] = {"argv": argv, "forward": forward, "forward_mode": mode, "url": url}
        with (self.output / "server-forward.log").open("wb") as forward_log:
            tunnel = subprocess.Popen(forward, stdin=subprocess.DEVNULL, stdout=forward_log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while True:
                    if tunnel.poll() is not None:
                        raise CaseFailure(f"ssh forward exited ({tunnel.returncode}); see server-forward.log")
                    try:
                        socket.create_connection(("127.0.0.1", local_port), timeout=2).close()
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise CaseFailure("ssh forward to the server did not open within 30s")
                        time.sleep(0.5)
                self.log("running server command: " + shlex.join(argv))
                started = time.monotonic()
                with (self.output / "server-command.stdout.txt").open("wb") as out, \
                        (self.output / "server-command.stderr.txt").open("wb") as err:
                    try:
                        code = subprocess.run(argv, cwd=self.output, stdin=subprocess.DEVNULL, stdout=out,
                                              stderr=err, timeout=a.timeout).returncode
                    except subprocess.TimeoutExpired:
                        code = None
                        self.fail(f"server command exceeded --timeout {a.timeout}s")
                    except OSError as error:
                        code = None
                        self.fail(f"server command could not start: {error}")
                record.update(exit_code=code, seconds=round(time.monotonic() - started, 2))
                if code not in (0, None):
                    self.fail(f"server command exited with {code}")
            finally:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    tunnel.kill()

    def wait_workers(self):
        pending = [k for k in self.by_index if k and not self.state[k]["done"]]
        deadline = time.monotonic() + self.a.shutdown_timeout
        while pending and time.monotonic() < deadline:
            pending = [k for k in pending if not state_done(self.poll(k))]
            if pending:
                time.sleep(min(5.0, self.a.poll_interval))
        for k in pending:
            self.fail(f"node{k} worker still running {self.a.shutdown_timeout}s after the driver; stopping it")
            self.stop(k, (("TERM", 15), ("KILL", 10)))

    def cleanup(self):
        for k in sorted(self.by_index):  # the driver first: workers follow it out
            if self.state[k]["started"] and not self.state[k]["done"]:
                try:
                    if not state_done(self.poll(k)):
                        self.fail(f"node{k} was still running at cleanup; stopping it")
                        self.stop(k, (("TERM", 15), ("KILL", 10)))
                except CaseFailure as error:
                    self.fail(str(error))

    def collect(self):
        for p in self.plan:
            k = p["node"].index
            if not self.state[k]["staged"]:
                continue
            if p["secret"]:
                # A launcher that never ran leaves its secret file behind; never copy it back.
                self.remote(k, f"rm -f {shlex.quote(p['run'] + '/secret-env.sh')}", timeout=60)
            if self.a.collect_exclude:
                # Large outputs stay on the node; their hashes travel with the evidence.
                run, kept = shlex.quote(p["run"]), shlex.quote(p["run"] + ".kept")
                names = " -o ".join(f"-name {shlex.quote(x)}" for x in self.a.collect_exclude)
                code, _, error = self.remote(
                    k, f"cd {run} && find . -type f \\( {names} \\) | sort | while read -r f; do "
                       f"sha256sum \"$f\" >> kept-files.sha256 && mkdir -p {kept}/\"$(dirname \"$f\")\" && "
                       f"mv \"$f\" {kept}/\"$f\" || exit 1; done", timeout=self.a.collect_timeout)
                if code != 0:
                    self.fail(f"node{k}: moving --collect-exclude files aside failed ({code}): {error.strip()[-300:]}")
            try:
                result = subprocess.run(p["collect_argv"], cwd=self.output, stdin=subprocess.DEVNULL,
                                        capture_output=True, timeout=self.a.collect_timeout)
                code, error = result.returncode, result.stderr.decode("utf-8", "replace")
            except (OSError, subprocess.TimeoutExpired) as exception:
                code, error = None, str(exception)
            self.state[k]["collect_exit_code"] = code
            if code != 0 or not (self.output / f"node{k}").is_dir():
                self.fail(f"node{k}: collecting {p['run']} failed ({code}): {error.strip()[-300:]}")
        self.log(f"collected run directories into {self.output}")

    def analyze(self):
        nodes, exit_codes = {}, {}
        for p in self.plan:
            k, directory = p["node"].index, self.output / f"node{p['node'].index}"
            if not self.state[k]["started"]:
                if self.state[k]["staged"]:
                    self.fail(f"node{k} was staged but never started")
                continue
            evidence = {"hostname": read_text(directory / "hostname.txt"), "gpus": read_text(directory / "nvidia-smi-L.txt"),
                        "dotnet": read_text(directory / "dotnet-version.txt"),
                        "source_fingerprint": read_text(directory / "source-fingerprint.txt"),
                        "git_identity": read_text(directory / "git-identity.txt"),
                        "model_files": read_text(directory / "model-files.txt")}
            try:
                evidence["exit"] = json.loads((directory / "exit.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                evidence["exit"] = None
            code = evidence["exit"].get("exit_code") if isinstance(evidence["exit"], dict) else None
            exit_codes[f"node{k}"] = code
            if code is None:
                self.fail(f"node{k}: no exit code (exit.json missing: killed, still running or not collected)")
            elif code != 0:
                self.fail(f"node{k}: command exited with {code}; see node{k}/process.log")
            before, after = hash_lines(directory / "runtime-before.sha256"), hash_lines(directory / "runtime-after.sha256")
            identity = "passed" if before and before == after and any(n.endswith("/" + posixpath.basename(p["dll"])) for n in before) else "failed"
            loaded = loaded_native(directory)
            evidence["runtime_identity"] = {"status": identity, "files": before,
                                            "native_hashed": bool(before and loaded) and before.get(loaded["path"]) == loaded["sha256"],
                                            "native_loaded": loaded,
                                            "scope": "sha256 of the main DLL, TensorSharp*.dll and every libGgmlOps.so the GGML "
                                                     "import resolver can load (app directory, repo-root native builds, "
                                                     "LD_LIBRARY_PATH) before and after the command; native_loaded is the "
                                                     "first of them that exists, the one the process loads"}
            if identity != "passed" and code is not None:
                self.fail(f"node{k}: runtime files missing or changed during the run")
            log_path = directory / "process.log"
            if log_path.is_file():
                evidence["process_log_sha256"] = sha(log_path)
                evidence["log"] = log_evidence(log_path.read_text(encoding="utf-8", errors="replace"))
                for line in evidence["log"]["crash_lines"][:3]:
                    self.fail(f"node{k}: {line[:240]}")
            else:
                self.fail(f"node{k}: process.log missing")
            nodes[f"node{k}"] = evidence
        fingerprints = {e.get("source_fingerprint") for e in nodes.values()}
        if len(fingerprints) > 1:
            self.warnings.append("Source fingerprints differ across nodes: " + json.dumps(sorted(map(str, fingerprints))))
        # Keyed by full path: every node shares --remote-root, and one node may hold several libGgmlOps.so copies.
        natives = {}
        for e in nodes.values():
            for name, digest in (e["runtime_identity"]["files"] or {}).items():
                natives.setdefault(name, set()).add(digest)
        differing = sorted(name for name, digests in natives.items() if len(digests) > 1)
        if differing:
            self.warnings.append("Runtime files differ across nodes: " + ", ".join(differing))
        loaded = {k: (e["runtime_identity"]["native_loaded"] or {}).get("sha256") for k, e in nodes.items()}
        if len(set(loaded.values())) > 1:
            self.warnings.append("The libGgmlOps.so each node loads differs across nodes: " + json.dumps(loaded, sort_keys=True))
        self.summary["exit_codes"], self.summary["node_evidence"] = exit_codes, nodes
        if self.state[0]["started"] and exit_codes.get("node0") is not None:
            try:
                {"bench": self.bench_metrics, "probe": self.probe_metrics, "server": self.server_metrics,
                 "suite": self.suite_metrics}[self.a.kind]()
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.fail(f"Could not parse {self.a.kind} metrics: {error!r}")
        elif self.state[0]["started"]:
            self.fail(f"No {self.a.kind} metrics: the driver has no exit code")

    def bench_metrics(self):
        a, driver = self.a, self.output / "node0"
        module, summarizer = load_summarizer()
        adapter = self.output / "driver-benchmark"
        adapter.mkdir()
        shutil.copyfile(driver / "process.log", adapter / "process.log")
        node = self.summary["node_evidence"]["node0"]
        stop = self.state[0].get("stop") or {}
        run = {"adapter_for": summarizer.name, "source": "run-multinode-case.py node0 (driver)",
               "status": "completed" if node["exit"]["exit_code"] == 0 and not stop.get("signals") else "failed",
               "command": self.by_index[0]["argv"], "environment": redacted(a),
               "shutdown": {"started": True, "requested": bool(stop.get("signals")), "forced": bool(stop.get("forced")),
                            "exit_code": node["exit"]["exit_code"]},
               "runtime_file_identity_check": {"status": node["runtime_identity"]["status"],
                                               "scope": node["runtime_identity"]["scope"]}}
        (adapter / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        report = module.summarize(adapter, a.discard_runs)
        (adapter / "cli-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log = (driver / "process.log").read_text(encoding="utf-8", errors="replace")
        logged = {row["run"]: row for row in logged_runs(log)}
        runs = []
        for sample in report.get("samples", []):
            row = logged.get(sample["run"], {})
            runs.append({"run": sample["run"], "prefill_ms": sample["prefill_ms"], "decode_ms": sample["decode_ms"],
                         "prefill_tps": sample.get("prefill_tokens_per_second"),
                         "decode_tps": sample.get("decode_tokens_per_second"),
                         "logged_prefill_tps": row.get("prefillTps"), "logged_decode_tps": row.get("decodeTps"),
                         "logged_ms_per_token": row.get("msPerTok"), "logged_decode_model_tps": row.get("decodeModelTps"),
                         "decode_mode": sample.get("mode")})
        measured = [r for r in runs if r["run"] > a.discard_runs]
        def median(field):
            values = [r[field] for r in measured if isinstance(r[field], float)]
            return statistics.median(values) if values else None
        metrics = {"status": report["status"], "failures": report["failures"],
                   "configuration": report.get("configuration"), "runs": runs,
                   "measured_runs": [r["run"] for r in measured],
                   "median_excluding_discarded": {"prefill_tps": median("prefill_tps"), "decode_tps": median("decode_tps"),
                                                  "logged_prefill_tps": median("logged_prefill_tps"),
                                                  "logged_decode_tps": median("logged_decode_tps")},
                   "summarizer": report.get("summary"), "summarizer_report": "driver-benchmark/cli-summary.json",
                   "summary_line": next(iter(re.findall(r"benchmark summary: ([^\r\n]*)", log)[-1:]), None)}
        sampled = re.findall(r"benchmark sampled tokens \((untimed correctness|run1)\): prefillTopToken=(-?\d+) decode=([0-9,]*)", log)
        if sampled:
            source, top, decode = sampled[-1]
            metrics["sampled_tokens"] = {"source": source, "prefill_top_token": int(top),
                                         "decode_token_ids": [int(x) for x in decode.split(",") if x]}
        elif a.bench_decode > 0:
            self.fail("benchmark: no 'benchmark sampled tokens' line in the driver log")
        if report["status"] != "measured":
            self.fail("benchmark: " + "; ".join(report["failures"]))
        self.metrics["benchmark"] = metrics

    def suite_metrics(self):
        """Parse the driver's suite log and bench-suite.json, and every worker's eviction lines (summarize-bench-suite.py)."""
        path = Path(__file__).with_name("summarize-bench-suite.py")
        spec = importlib.util.spec_from_file_location("summarize_bench_suite", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        driver = self.output / "node0"
        log = (driver / "process.log").read_text(encoding="utf-8", errors="replace")
        suite_json = None
        if (driver / "bench-suite.json").is_file():
            suite_json = json.loads((driver / "bench-suite.json").read_text(encoding="utf-8"))
        else:
            self.fail("suite: node0/bench-suite.json missing")
        workers = {}
        for p in self.plan[1:]:
            k = p["node"].index
            worker_log = self.output / f"node{k}" / "process.log"
            if worker_log.is_file():
                workers[str(k)] = worker_log.read_text(encoding="utf-8", errors="replace")
        report = module.summarize(log, suite_json, workers, self.a.discard_runs)
        report["summarizer"] = path.name
        (self.output / "suite-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for failure in report["failures"]:
            self.fail("suite: " + failure)
        keys = ("label", "evict", "cold_verified", "median_prefill_tps", "median_decode_tps", "run1_prefill_tps",
                "run1_decode_tps", "measured_runs", "status")
        self.metrics["suite"] = {"status": report["status"], "report": "suite-summary.json",
                                 "workloads": [{key: w[key] for key in keys} for w in report["workloads"]]}

    def probe_metrics(self):
        result = {}
        for p in self.plan:
            k = p["node"].index
            path = self.output / f"node{k}" / "probe" / "probe.json"
            if not path.is_file():
                self.fail(f"node{k}: probe.json missing")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = "completed" if k == 0 else "worker_loop_returned"
            steps = []
            kept_hashes = {}
            kept_list = self.output / f"node{k}" / "kept-files.sha256"
            if kept_list.is_file():
                for line in kept_list.read_text(encoding="utf-8").splitlines():
                    digest, _, name = line.partition("  ")
                    if name:
                        kept_hashes[name[2:] if name.startswith("./") else name] = digest
            for step in data.get("steps", []):
                row = {key: step.get(key) for key in ("operation", "argmax", "seconds", "count", "logits")}
                row["token_count"] = len(step.get("tokens") or [])
                if step.get("logits"):
                    file = path.parent / step["logits"]
                    if not file.is_file() and self.a.collect_exclude:
                        # Left on the node by --collect-exclude: trust its recorded hash.
                        kept = kept_hashes.get(posixpath.join(path.parent.name, step["logits"]))
                        row.update(logits_bytes=None, logits_sha256=kept, logits_kept_on_node=True)
                        if kept is None:
                            self.fail(f"node{k}: logits file {step['logits']} neither collected nor kept on the node")
                        steps.append(row)
                        continue
                    row["logits_bytes"] = file.stat().st_size if file.is_file() else None
                    row["logits_sha256"] = sha(file) if file.is_file() else None
                    if row["logits_bytes"] != 4 * (data.get("vocab_size") or -1):
                        self.fail(f"node{k}: logits file {step['logits']} missing or not vocab_size F32 values")
                steps.append(row)
            result[f"node{k}"] = {"status": data.get("status"), "expected_status": expected,
                                  "probe_json_sha256": sha(path), "weights_id": data.get("weights_id"),
                                  "topology": data.get("topology"), "load_seconds": data.get("load_seconds"),
                                  "vocab_size": data.get("vocab_size"), "token_count": len(data.get("tokens") or []),
                                  "tokens_sha256": data.get("tokens_sha256"), "text_sha256": data.get("text_sha256"),
                                  "cpu_offload": data.get("cpu_offload"), "error": (data.get("error") or "")[:2000] or None,
                                  "steps": steps if k == 0 else len(steps)}
            if data.get("status") != expected:
                self.fail(f"node{k}: probe status {data.get('status')!r}, expected {expected!r}")
            if k == 0 and not any(row.get("logits") for row in steps):
                self.fail("node0: probe saved no logit vectors")
        self.metrics["probe"] = result

    def server_metrics(self):
        server = self.metrics.setdefault("server", {})
        server["model_load"] = (self.summary["node_evidence"].get("node0", {}).get("log") or {}).get("model_load")
        if not server.get("ready"):
            self.fail("server never reported ready")
        if self.a.server_command and (server.get("command") or {}).get("exit_code") != 0:
            self.fail("server command did not complete with exit code 0")

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()
        interrupted = False
        try:
            self.stage()
            self.start_workers()
            self.start(0)
            if self.a.kind == "server":
                self.serve()
            else:
                self.wait_driver()
            self.wait_workers()
        except CaseFailure as error:
            self.fail(str(error))
        except KeyboardInterrupt:
            interrupted = True
            self.fail("Interrupted by the operator")
        except Exception as error:  # an orchestrator bug must not leave a "passed" summary
            self.fail(f"Orchestrator error: {error!r}")
        finally:
            for step in (self.cleanup, self.collect, self.analyze):
                try:
                    step()
                except CaseFailure as error:
                    self.fail(str(error))
                except KeyboardInterrupt:
                    interrupted = True
                    self.fail(f"Interrupted during {step.__name__}")
                except Exception as error:
                    self.fail(f"Orchestrator error during {step.__name__}: {error!r}")
            self.summary["ended_utc"] = utc_now()
            self.summary["status"] = "interrupted" if interrupted else "failed" if self.failures else "passed"
            self.save()
        headline = {key: self.summary[key] for key in ("status", "failures", "warnings", "exit_codes") if key in self.summary}
        if "benchmark" in self.metrics:
            headline["benchmark_median"] = self.metrics["benchmark"]["median_excluding_discarded"]
        if "suite" in self.metrics:
            headline["suite"] = {w["label"]: [w["median_prefill_tps"], w["median_decode_tps"], w["cold_verified"]]
                                 for w in self.metrics["suite"]["workloads"]}
        print(json.dumps(headline, indent=2))
        return 0 if self.summary["status"] == "passed" else 1


def show_ssh(title, argv, note=""):
    """Print an ssh call as its exact local prefix plus the script it passes to 'sh -c'."""
    script = shlex.split(argv[-1][len("sh -c "):])[0]
    print(f"{title}: {shlex.join(argv[:-1])} 'sh -c SCRIPT'{note}")
    for line in script.splitlines():
        print("  | " + line)


def print_dry_run(a, plan):
    public, secret = case_environment(a)
    print(f"# dry run: label={a.label} kind={a.kind} mode={a.mode} local_degree={a.local_degree} "
          f"nodes={len(plan)}; nothing is executed")
    print("# each ssh call's last argument is 'sh -c ' + shlex.quote(SCRIPT); SCRIPT lines are prefixed '  | '")
    print("# environment on every node: " + json.dumps(redacted(a)))
    # Workers first, highest rank first (so it listens before lower ranks dial it), then the driver.
    order = sorted((p for p in plan if p["node"].index), key=lambda item: -item["node"].index) + [plan[0]]
    for p in order:
        node = p["node"]
        print(f"\n## node{node.index} ({node.role}) {node.dest} ssh port {node.port} -> {p['run']}")
        show_ssh("stage", p["stage_argv"], " with stdin = launch.sh:")
        print(f"launch.sh ({p['run']}/launch.sh):")
        for line in p["launcher"].splitlines():
            print("  > " + line)
        if p["secret_argv"]:
            show_ssh("stage secrets", p["secret_argv"], f" with stdin = <{len(secret)} exports redacted>")
        show_ssh("start", p["start_argv"])
        if node.index:
            when = f"until mesh progress is logged or port {p['port']} listens (up to {a.worker_ready_timeout}s)"
        elif a.kind == "server":
            when = f"until GET /health on {a.server_port} answers or 'Now listening on' is logged (up to {a.startup_timeout}s)"
        else:
            when = f"until exit.json exists (up to {a.timeout}s)"
        show_ssh("poll " + when, p["poll_argv"])
    driver = plan[0]
    print("\n## stop: the server driver at the end, or any node on timeout; signals go to the launcher's process group")
    show_ssh("signal", ssh_argv(a, driver["node"], signal_script(driver["run"], "INT")),
             " (server: INT, then TERM after --stop-grace, then KILL; on timeout: TERM, then KILL)")
    if a.kind == "server" and a.server_command:
        print(f"server command template (through an ssh -L forward from this machine): {a.server_command}")
    print(f"\n## collect (cwd {a.output}, created at run time)")
    for p in plan:
        print("collect: " + shlex.join(p["collect_argv"]))
    print(f"\n# then: parse node0/process.log (or node*/probe/probe.json) and write {a.output / 'summary.json'}")


def main(argv=None):
    # Windows Python writes cp1252 to a pipe (Git Bash): remote log text must not crash it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    a = parse_args(sys.argv[1:] if argv is None else argv)
    plan = build_plan(a)
    if a.dry_run:
        print_dry_run(a, plan)
        return 0
    return Case(a, plan).run()


if __name__ == "__main__":
    sys.exit(main())
