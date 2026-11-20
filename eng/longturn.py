#!/usr/bin/env python3
"""Long-context agent turns against an OpenAI-compatible server (the reproducer of issue #376).

For each document size the client sends three requests, one at a time, and waits for each answer:
  A  a system turn and the document as one user turn, max_tokens 1: a fresh prefill of a long prompt
  B  A + an assistant "OK" + a new user turn of about 1K tokens, max_tokens 1: a short suffix over the cached document
  C  B's prompt again with --max-tokens (192): the whole prompt reused, then decoding (speculative where the server drafts)
The document is TensorSharp's own TensorSharp.Runtime/**/*.cs, as in the issue: about 3.2 characters a token, so the
default sizes 8192 / 32768 / 65536 / 98304 give 6.6K / 24K / 47K / 69K-token documents.

  python3 eng/longturn.py 127.0.0.1:5000                       # the issue's four sizes, server-side sampling
  python3 eng/longturn.py 127.0.0.1:5000 --sizes 65536 --rounds 2 --temperature 0.3 --top-k 20 --top-p 0.8
  <server command> 2>&1 | python3 -u eng/longturn.py stamp > <out>/server.log   # a UTC stamp on every server line
  python3 eng/longturn-report.py <out>

Run the server with TS_SCHED_STEP_LOG=1 (one line per scheduler step) and its output stamped into <out>/server.log;
eng/longturn-report.py then reads <out>/server.log and <out>/client.jsonl and prints time to first token, A's tail after
its first token, B's admission gap, the first prompt step, decode and speculative per-step times and the reused tokens.

The corpus comes from git (--corpus-rev, default 4c819ed2: the files the issue's numbers came from), so every server
tree under test gets the same prompts. --corpus-rev worktree reads the checkout's files from disk the way the issue's
script did: a built tree then adds its obj/**/*.cs files, which changes B's turn (1,044 tokens on the issue's
validation VM against the issue's 969).

Without --temperature / --top-k / --top-p / --seed the requests carry no sampling fields and the server's defaults
apply, as in the issue's script (acceptance then varies from run to run); the issue's replay pinned 0.3 / 20 / 0.8.

Writes <out>/run.json (arguments, model, corpus revision and sha256) and <out>/client.jsonl (one record a request: size,
round, step, UTC send and receive times, HTTP status, response id, prompt and completion tokens, reply text, error), and prints one
line a request. --out defaults to artifacts/longturn/<UTC time>/ in this checkout (ignored by git). A request that fails
ends its size (B and C need A); the next size runs unless --stop-on-error.
Standard library only.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SIZES = (8192, 32768, 65536, 98304)
DEFAULT_CORPUS_REV = "4c819ed2"
WORKTREE = "worktree"
SUFFIX_CHARS = 4300
CHARS_PER_TOKEN = 3.2
SYSTEM = "You are a code assistant. Answer briefly."
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t: datetime.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def stamp(stream_in=None, stream_out=None) -> None:
    """Prefix every line of stream_in with a UTC wall-clock stamp (HH:MM:SS.mmm) and flush each line."""
    stream_in = stream_in or sys.stdin
    stream_out = stream_out or sys.stdout
    for line in stream_in:
        stream_out.write(utc_now().strftime("%H:%M:%S.%f")[:-3] + " " + line)
        stream_out.flush()


def _universal_newlines(text: str) -> str:
    # open() in text mode (the issue's script) turns \r\n and \r into \n; blobs read from git get the same.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def corpus_files(checkout: str, rev: str) -> list[tuple[str, str]]:
    """(path relative to TensorSharp.Runtime, text) of every TensorSharp.Runtime/**/*.cs, in the issue script's order."""
    root = os.path.join(checkout, "TensorSharp.Runtime")
    if rev == WORKTREE:
        files = sorted(glob.glob(os.path.join(root, "**", "*.cs"), recursive=True))
        out = []
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                out.append((os.path.relpath(f, root), fh.read()))
        return out
    try:
        listing = subprocess.run(["git", "-C", checkout, "ls-tree", "-r", "--name-only", rev, "--", "TensorSharp.Runtime"],
                                 check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", "") or str(e)
        raise SystemExit(f"longturn: cannot list TensorSharp.Runtime at {rev} in {checkout}: {detail.strip()}\n"
                         f"(pass --corpus-rev {WORKTREE} to read the files on disk instead)")
    prefix = "TensorSharp.Runtime/"
    # glob's "**/*.cs" skips hidden files and directories.
    rels = sorted(p[len(prefix):] for p in listing
                  if p.startswith(prefix) and p.endswith(".cs") and not any(c.startswith(".") for c in p.split("/")))
    out = []
    for rel in rels:
        blob = subprocess.run(["git", "-C", checkout, "show", f"{rev}:{prefix}{rel}"], check=True, capture_output=True).stdout
        out.append((rel.replace("/", os.sep), _universal_newlines(blob.decode("utf-8", errors="replace"))))
    return out


def build_corpus(files: list[tuple[str, str]]) -> str:
    return "".join(f"// FILE {rel}\n" + text for rel, text in files)


def conversation(corpus: str, size: int, round_index: int, fresh_rounds: bool,
                 suffix_chars: int = SUFFIX_CHARS) -> tuple[list, list]:
    """The A and B message lists of one size (C sends B's). B's file is the corpus's last suffix_chars characters; the
    document is always the corpus less its last SUFFIX_CHARS, so A's prompt does not depend on B's file."""
    suffix, doc = corpus[-suffix_chars:], corpus[:-SUFFIX_CHARS]
    cut = doc[: int(size * CHARS_PER_TOKEN)]
    system = SYSTEM if not (fresh_rounds and round_index > 0) else f"{SYSTEM} (Round {round_index + 1}.)"
    a = [{"role": "system", "content": system},
         {"role": "user", "content": f"Here is part of a C# codebase:\n```csharp\n{cut}\n```\nReply with OK."}]
    b = a + [{"role": "assistant", "content": "OK"},
             {"role": "user", "content": f"Now read this file too:\n```csharp\n{suffix}\n```\n"
                                          "Rewrite its last method with every local variable renamed to snake_case."}]
    return a, b


def sampling_fields(args) -> dict:
    fields = {}
    if args.temperature is not None: fields["temperature"] = args.temperature
    if args.top_k is not None: fields["top_k"] = args.top_k
    if args.top_p is not None: fields["top_p"] = args.top_p
    if args.seed is not None: fields["seed"] = args.seed
    return fields


def base_url(host: str) -> str:
    host = host.rstrip("/")
    return host if host.startswith(("http://", "https://")) else "http://" + host


def get_model(base: str, timeout: float) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as r:
        return json.load(r)["data"][0]["id"]


def chat(base: str, body: dict, timeout: float) -> dict:
    """One non-streamed request; a record with its timings, status, usage and error."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data, {"Content-Type": "application/json"})
    sent = utc_now()
    t0 = time.monotonic()
    rec = {"sent": iso(sent), "status": None, "id": None, "prompt_tokens": None, "completion_tokens": None,
           "cached_tokens": None, "finish_reason": None, "error": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rec["status"] = resp.status
            r = json.load(resp)
        usage = r.get("usage") or {}
        rec["id"] = r.get("id")
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        choices = r.get("choices") or [{}]
        rec["finish_reason"] = choices[0].get("finish_reason")
        # The reply's text, so runs of two trees compare token for token where nothing should change (greedy runs).
        rec["content"] = (choices[0].get("message") or {}).get("content")
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException):
            detail = ""
        rec["error"] = f"HTTP {e.code}: {detail.strip()[:300]}"
    # http.client.HTTPException: a server that dies mid-answer leaves IncompleteRead, which is not an OSError.
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    rec["wall_s"] = round(time.monotonic() - t0, 3)
    rec["done"] = iso(utc_now())
    return rec


def run(args) -> int:
    base = base_url(args.host)
    checkout = os.path.abspath(args.corpus)
    files = corpus_files(checkout, args.corpus_rev)
    if not files:
        raise SystemExit(f"longturn: no TensorSharp.Runtime/**/*.cs under {checkout} at {args.corpus_rev}")
    corpus = build_corpus(files)
    longest = max(args.sizes)
    if int(longest * CHARS_PER_TOKEN) > len(corpus) - SUFFIX_CHARS:
        print(f"longturn: the corpus has {len(corpus) - SUFFIX_CHARS} document characters; size {longest} wants "
              f"{int(longest * CHARS_PER_TOKEN)}, so its document is the whole corpus", file=sys.stderr)
    model = args.model or get_model(base, args.timeout)
    out = args.out or os.path.join(REPO, "artifacts", "longturn", utc_now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    meta = {
        "started": iso(utc_now()), "url": base, "model": model, "sizes": args.sizes, "rounds": args.rounds,
        "fresh_rounds": args.fresh_rounds, "max_tokens_c": args.max_tokens, "sampling": sampling_fields(args) or None,
        "suffix_chars": args.suffix_chars,
        "corpus_checkout": checkout, "corpus_rev": args.corpus_rev, "corpus_files": len(files),
        "corpus_chars": len(corpus), "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
    }
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"# longturn {base} model={model} sizes={args.sizes} rounds={args.rounds} sampling={meta['sampling'] or 'server'} "
          f"corpus={args.corpus_rev} ({len(files)} files, sha256 {meta['corpus_sha256'][:12]}) out={out}", flush=True)
    failed = False
    with open(os.path.join(out, "client.jsonl"), "a", encoding="utf-8") as log:
        for size in args.sizes:
            for rnd in range(args.rounds):
                a, b = conversation(corpus, size, rnd, args.fresh_rounds, args.suffix_chars)
                broken = False
                for step, msgs, max_tokens in (("A", a, 1), ("B", b, 1), ("C", b, args.max_tokens)):
                    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, **sampling_fields(args)}
                    rec = {"size": size, "round": rnd, "step": step, "max_tokens": max_tokens, **chat(base, body, args.timeout)}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    tail = f"  {rec['error']}" if rec["error"] else ""
                    print(f"{rec['done'][11:23]} {size} r{rnd} {step} {rec['prompt_tokens']} {rec['wall_s']:.2f} s{tail}", flush=True)
                    if rec["error"]:
                        broken = failed = True
                        break
                if broken:
                    break
            if failed and args.stop_on_error:
                break
    print(f"OUT={out}", flush=True)
    return 1 if failed else 0


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="the server: host:port or a base URL")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                    help="document sizes, in tokens at 3.2 characters a token (default: %(default)s)")
    ap.add_argument("--rounds", type=int, default=1, help="A/B/C sequences per size, one after another (default: 1)")
    ap.add_argument("--fresh-rounds", action="store_true",
                    help="rounds after the first change the system turn, so their A is a fresh prefill instead of a reuse")
    ap.add_argument("--max-tokens", type=int, default=192, help="C's max_tokens (A and B take 1; default: 192)")
    ap.add_argument("--suffix-chars", type=int, default=SUFFIX_CHARS,
                    help="characters of B's file, the corpus's last ones (default: %(default)s; 3800 gives issue #383's "
                         "961-token B turn, under the prefix cache's 1,024-token prompt-tail minimum)")
    ap.add_argument("--temperature", type=float, help="sampling temperature (default: not sent, the server's default)")
    ap.add_argument("--top-k", type=int, help="top-k (default: not sent)")
    ap.add_argument("--top-p", type=float, help="top-p (default: not sent)")
    ap.add_argument("--seed", type=int, help="sampling seed (default: not sent)")
    ap.add_argument("--model", help="the model id to request (default: the first of /v1/models)")
    ap.add_argument("--corpus", default=REPO, help="the TensorSharp checkout the corpus comes from (default: this one)")
    ap.add_argument("--corpus-rev", default=DEFAULT_CORPUS_REV,
                    help=f"the git revision of the corpus files, or '{WORKTREE}' for the files on disk (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds a request may take (default: 3600)")
    ap.add_argument("--out", help="output directory (default: artifacts/longturn/<UTC time>/ in this checkout)")
    ap.add_argument("--stop-on-error", action="store_true", help="stop at the first failed request instead of the next size")
    args = ap.parse_args(argv)
    if args.rounds < 1 or any(s <= 0 for s in args.sizes) or args.max_tokens < 1 or not 0 < args.suffix_chars <= SUFFIX_CHARS:
        ap.error(f"--rounds, --sizes and --max-tokens must be positive, --suffix-chars 1-{SUFFIX_CHARS}")
    return args


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["stamp"]:
        stamp()
        return 0
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
