#!/usr/bin/env python3
"""Long-context quality of an arithmetic DEFAULT CHANGE, between two arms (standard library only).

Three steps, the middle one in eng/LongContextQualityProbe (one loaded model, arms as in-process seams or per process):

  cases     build a case file and its pinned corpora under an ignored directory (artifacts/...):
              ppl       prompt-path NLL of the last --score-tokens of corpus windows (prose: Moby-Dick, Project
                        Gutenberg #2701; code: TensorSharp.Runtime/**/*.cs at a pinned revision, the issue #376 corpus)
              continue  the same windows, the scored tokens fed one decode row at a time (the served situation)
              layers    the 8-bit prompt attention against the 16-bit kernel on the same rows of a real prompt
              greedy    256 greedy tokens after code prompts cut at method boundaries (identity between arms)
              chat      retrieval (five vault records at fixed depths: codes, document order, neighbours, 8K-128K),
                        GSM8K and HumanEval subsets behind a long shared reference (pinned upstream files)
  (probe)   dotnet eng/LongContextQualityProbe/bin/Release/net10.0/LongContextQualityProbe.dll --model M --cases
            CASES/cases.jsonl --output RUN.jsonl --arms f16,8bit         (or one arm a process: --kv-cache-dtype q8_0)
  report    pair a reference arm with a candidate arm (any two RUN.jsonl:ARM), grade, and check pre-registered gates
  estimate  GPU time of a case file, extrapolated from a run's measured prefill, scoring and decode times

Outputs belong in ignored directories (artifacts/ or docs/validation/); the tool refuses other paths inside the repository.
Downloads are pinned by sha256 and cached under artifacts/quality/downloads.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "long-context-quality/1"

# Public, pinned inputs. Moby-Dick: the Project Gutenberg #2701 plain text as its mirrors serve it (public domain in the
# USA); the three mirrors serve identical bytes.
PROSE = {
    "name": "Moby-Dick; or, The Whale (Herman Melville), Project Gutenberg #2701",
    "license": "public domain in the USA (Project Gutenberg license header/footer stripped)",
    "urls": ["https://gutenberg.pglaf.org/2/7/0/2701/2701-0.txt", "http://aleph.gutenberg.org/2/7/0/2701/2701-0.txt",
             "https://mirrors.xmission.com/gutenberg/2/7/0/2701/2701-0.txt"],
    "sha256": "f12eb530502755ef2b8f535e2ecef3e95802eab584e60ac7992ccf856256fdbe",
    "start": "*** START OF THE PROJECT GUTENBERG EBOOK 2701 ***",
    "end": "*** END OF THE PROJECT GUTENBERG EBOOK 2701 ***",
}
GSM8K = {
    "name": "GSM8K test split (openai/grade-school-math)", "license": "MIT",
    "urls": ["https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/"
             "grade_school_math/data/test.jsonl"],
    "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
    "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
HUMANEVAL = {
    "name": "HumanEval (openai/human-eval data/HumanEval.jsonl.gz)", "license": "MIT",
    "urls": ["https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz"],
    "revision": "b9453e8f1865df0a0057a1383767caa400acfbcf",
    "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
}
CODE_REVISION = "4c819ed2"

CODE_WORDS_A = ["AMBER", "COBALT", "VERMILION", "SAFFRON", "INDIGO", "CERULEAN", "OCHRE", "VIRIDIAN", "MAGENTA", "UMBER"]
CODE_WORDS_B = ["OTTER", "HERON", "LYNX", "MARTEN", "PETREL", "IBEX", "GECKO", "BITTERN", "TAPIR", "WOMBAT"]
VAULTS = ("alpha", "beta", "gamma", "delta", "epsilon")
DEPTHS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Presets: "full" is the protocol of design t2-attn8 sec. 12.2 / t3 sec. 3 (b2); "smoke" exercises every kind small, with
# two calibration points (32K, 64K) for `estimate`.
PRESETS = {
    "full": {
        "ppl": {"corpora": ["prose", "code"], "sizes": {8: 6, 16: 4, 32: 3, 64: 2, 96: 1}, "score": 2048},
        "continue": {"corpora": ["prose", "code"], "sizes": {8: 6, 16: 4, 32: 3, 64: 2, 96: 1}, "score": 1024},
        "layers": {"corpus": "prose", "sizes": [32]},
        "greedy": {"sizes": [8, 32, 64], "prompts": 8, "new_tokens": 256},
        "retrieval": {"sizes": [8, 16, 32, 64, 96, 128], "repeats": 3, "max_new": 192},
        "gsm8k": {"count": 50, "prefix": 16, "max_new": 512},
        "humaneval": {"count": 40, "prefix": 16, "max_new": 768},
    },
    "smoke": {
        "ppl": {"corpora": ["prose"], "sizes": {8: 1, 32: 1}, "score": 512},
        "continue": {"corpora": ["prose"], "sizes": {8: 1, 64: 1}, "score": 64},
        "layers": {"corpus": "prose", "sizes": [8]},
        "greedy": {"sizes": [8], "prompts": 2, "new_tokens": 64},
        "retrieval": {"sizes": [8], "repeats": 1, "max_new": 192},
        "gsm8k": {"count": 3, "prefix": 4, "max_new": 384},
        "humaneval": {"count": 3, "prefix": 4, "max_new": 512},
    },
}
KINDS = ["ppl", "continue", "layers", "greedy", "retrieval", "gsm8k", "humaneval"]


# --------------------------------------------------------------------------- paths and downloads

def ignored_output(path):
    """Refuse an output directory inside the repository unless it is under the ignored artifacts/ or docs/validation/."""
    path = Path(path).resolve()
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return path
    if rel.parts[:1] == ("artifacts",) or rel.parts[:2] == ("docs", "validation"):
        return path
    raise SystemExit(f"{path}: outputs go under the ignored artifacts/ or docs/validation/, not into the repository")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fetch(spec, downloads):
    """A pinned file: from the download cache, else from its first URL that serves the pinned sha256."""
    downloads.mkdir(parents=True, exist_ok=True)
    cached = downloads / spec["sha256"]
    if cached.exists() and sha256_bytes(cached.read_bytes()) == spec["sha256"]:
        return cached.read_bytes()
    errors = []
    for url in spec["urls"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "long-context-quality/1"})
            data = urllib.request.urlopen(req, timeout=120).read()
        except OSError as error:
            errors.append(f"{url}: {error}")
            continue
        got = sha256_bytes(data)
        if got != spec["sha256"]:
            errors.append(f"{url}: sha256 {got}, pinned {spec['sha256']}")
            continue
        cached.write_bytes(data)
        return data
    raise SystemExit(f"{spec['name']}: no source served the pinned file:\n  " + "\n  ".join(errors))


def prose_corpus(downloads):
    """Moby-Dick between the Gutenberg markers, paragraphs whitespace-collapsed, joined by blank lines."""
    text = fetch(PROSE, downloads).decode("utf-8").replace("\r\n", "\n")
    start, end = text.index(PROSE["start"]) + len(PROSE["start"]), text.index(PROSE["end"])
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text[start:end])]
    return "\n\n".join(p for p in paragraphs if p) + "\n"


def code_corpus(repo, revision):
    """The issue's corpus: TensorSharp.Runtime/**/*.cs, sorted, each after a '// FILE <relpath>' line, at a git revision
    (text-mode newlines, as the issue's script read the files)."""
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", revision, "--", "TensorSharp.Runtime"],
                           check=True, capture_output=True, text=True).stdout.split("\n")
    files = sorted(n for n in names if n.endswith(".cs"))
    if not files:
        raise SystemExit(f"no TensorSharp.Runtime/**/*.cs at {revision} in {repo}")
    parts = []
    for name in files:
        data = subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{name}"], check=True, capture_output=True).stdout
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        parts.append(f"// FILE {os.path.relpath(name, 'TensorSharp.Runtime')}\n" + text)
    return "".join(parts)


MEMBER_START = r"(?:///|\[|public |private |internal |protected |static |override |async |unsafe |readonly |sealed |virtual |abstract |partial )"


def method_ends(code):
    """Character offsets just past a line that closes a member body: a '}' at a member's indentation (4 or 8 spaces), a blank
    line, then the next member's declaration, attribute or doc comment at the same indentation."""
    return [m.end() for m in re.finditer(r"\n( {4}| {8})\}\n(?=\n\1" + MEMBER_START + ")", code)]


# --------------------------------------------------------------------------- cases

def nonce(seed, *parts):
    h = hashlib.sha256(":".join(str(p) for p in (seed,) + parts).encode()).hexdigest()
    return int(h[:8], 16) % 900000 + 100000


def reference_messages(seed, user):
    return [{"role": "system", "content": f"Deployment {nonce(seed, 'deployment')}. The product reference below is background for every "
                                          "conversation.\n\n<reference>\n{{document}}\n</reference>\n\nYou are a helpful assistant. "
                                          "Use the reference only when the request needs it; otherwise follow the request exactly."},
            {"role": "user", "content": user}]


def build_cases(args):
    preset = json.loads(json.dumps(PRESETS[args.preset]))
    kinds = args.kinds.split(",") if args.kinds else KINDS
    for k in kinds:
        if k not in KINDS:
            raise SystemExit(f"--kinds: {k} is not one of {', '.join(KINDS)}")
    out = ignored_output(args.out)
    downloads = ignored_output(args.downloads)
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    seed = args.seed

    prose = prose_corpus(downloads)
    corpora = {"prose": {"file": "corpus/prose.txt", "text": prose, "source": {k: PROSE[k] for k in ("name", "license", "urls", "sha256")}}}
    if any(k in kinds for k in ("ppl", "continue", "greedy")) and \
            ("code" in preset["ppl"]["corpora"] + preset["continue"]["corpora"] or "greedy" in kinds):
        code = code_corpus(args.repo, args.code_revision)
        corpora["code"] = {"file": "corpus/code.txt", "text": code,
                           "source": {"name": "TensorSharp.Runtime/**/*.cs (issue #376 corpus)", "repository": "zhongkaifu/TensorSharp",
                                      "revision": args.code_revision, "license": "BSD-3-Clause"}}
    header = {"type": "header", "schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "preset": args.preset, "seed": seed, "kinds": kinds, "think": args.think, "corpora": {}, "datasets": {}}
    for name, c in corpora.items():
        data = c["text"].encode("utf-8")
        (out / c["file"]).write_bytes(data)
        header["corpora"][name] = {"file": c["file"], "sha256": sha256_bytes(data), "chars": len(c["text"]), "source": c["source"]}

    cases = []
    for kind in ("ppl", "continue"):
        if kind not in kinds:
            continue
        p = preset[kind]
        for corpus in p["corpora"]:
            for k, windows in sorted((int(size), n) for size, n in p["sizes"].items()):
                for w in range(windows):
                    cases.append({"type": "case", "id": f"{kind}/{corpus}/{k}k/w{w}", "kind": kind, "corpus": corpus,
                                  "tokens": k * 1024, "start_token": w * k * 1024, "score_tokens": p["score"]})
    if "layers" in kinds:
        for k in preset["layers"]["sizes"]:
            cases.append({"type": "case", "id": f"layers/{preset['layers']['corpus']}/{k}k", "kind": "layers",
                          "corpus": preset["layers"]["corpus"], "tokens": k * 1024, "start_token": 0})
    if "greedy" in kinds:
        g = preset["greedy"]
        code = corpora["code"]["text"]
        ends = method_ends(code)
        for k in g["sizes"]:
            lo = k * 1024 * 6   # room for k K tokens before the cut at ~3.3 characters a token
            pool = [e for e in ends if e >= lo]
            if len(pool) < g["prompts"]:
                raise SystemExit(f"greedy {k}K: only {len(pool)} method ends past character {lo}")
            for i in range(g["prompts"]):
                target = lo + (i + 0.5) * (len(code) - lo) / g["prompts"]
                end = next(e for e in pool if e >= target)
                cases.append({"type": "case", "id": f"greedy/code/{k}k/p{i}", "kind": "greedy", "corpus": "code",
                              "tokens": k * 1024, "end_char": end, "new_tokens": g["new_tokens"]})
    if "retrieval" in kinds:
        r = preset["retrieval"]
        for k in r["sizes"]:
            for rep in range(r["repeats"]):
                cases.append(retrieval_case(seed, k, rep, r["max_new"], prose, args.think))
    if "gsm8k" in kinds:
        g = preset["gsm8k"]
        rows = [json.loads(line) for line in fetch(GSM8K, downloads).decode("utf-8").splitlines() if line.strip()]
        picks = sorted(random.Random(seed).sample(range(len(rows)), g["count"]))
        header["datasets"]["gsm8k"] = {k: GSM8K[k] for k in ("name", "license", "urls", "revision", "sha256")}
        header["datasets"]["gsm8k"]["indices_zero_based"] = picks
        for i in picks:
            answer = rows[i]["answer"].split("####")[-1].strip().replace(",", "")
            user = (rows[i]["question"] + "\n\nSolve the problem step by step, briefly. End with a final line of the form "
                    "'Answer: <number>'.")
            cases.append({"type": "case", "id": f"gsm8k/test-{i}", "kind": "chat", "group": "gsm8k",
                          "fit_tokens": g["prefix"] * 1024, "document": {"corpus": "prose", "start_paragraph": 211},
                          "messages": reference_messages(seed, user), "think": args.think, "max_new_tokens": g["max_new"],
                          "grade": {"type": "gsm8k", "answer": answer}})
    if "humaneval" in kinds:
        h = preset["humaneval"]
        rows = [json.loads(line) for line in gzip.decompress(fetch(HUMANEVAL, downloads)).decode("utf-8").splitlines() if line.strip()]
        picks = sorted(random.Random(seed).sample(range(len(rows)), h["count"]))
        header["datasets"]["humaneval"] = {k: HUMANEVAL[k] for k in ("name", "license", "urls", "revision", "sha256")}
        header["datasets"]["humaneval"]["task_ids"] = [rows[i]["task_id"] for i in picks]
        for i in picks:
            row = rows[i]
            user = ("Complete the following Python function. Reply with the complete function, including its signature and any "
                    "imports it needs, in one ```python code block.\n\n```python\n" + row["prompt"] + "```")
            cases.append({"type": "case", "id": f"humaneval/{row['task_id'].replace('/', '-')}", "kind": "chat", "group": "humaneval",
                          "fit_tokens": h["prefix"] * 1024, "document": {"corpus": "prose", "start_paragraph": 211},
                          "messages": reference_messages(seed, user), "think": args.think, "max_new_tokens": h["max_new"],
                          "grade": {"type": "humaneval", "task_id": row["task_id"], "prompt": row["prompt"], "test": row["test"],
                                    "entry_point": row["entry_point"]}})
    path = out / "cases.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for c in cases:
            f.write(json.dumps(c) + "\n")
    counts = {}
    for c in cases:
        key = c.get("group") or c["kind"]
        counts[key] = counts.get(key, 0) + 1
    print(f"{path}: {len(cases)} cases ({', '.join(f'{k} {v}' for k, v in counts.items())}); corpora "
          + ", ".join(f"{n} {h['chars']} chars sha256 {h['sha256'][:12]}" for n, h in header["corpora"].items()))
    return 0


def retrieval_case(seed, k, rep, max_new, prose, think):
    """Five vault records at fixed depth fractions of a Moby-Dick haystack, their physical order shuffled; the answer gives
    every code, the document order and gamma's neighbours (eng/qwen38-engine-bench.py --long-order-quality's checks)."""
    rng = random.Random(f"{seed}:retrieval:{k}:{rep}")
    codes = {key: f"{rng.choice(CODE_WORDS_A)}-{rng.choice(CODE_WORDS_B)}-{rng.randint(1000, 9999)}" for key in VAULTS}
    if any(code in prose for code in codes.values()):
        raise SystemExit("a vault code occurs in the haystack")
    order = list(VAULTS)
    rng.shuffle(order)
    inserts = [{"depth": d, "text": f"\n\nArchive record: the {key} vault code is {codes[key]}.\n\n"} for key, d in zip(order, DEPTHS)]
    question = ('Return a JSON object with the complete vault codes in keys "alpha", "beta", "gamma", "delta", "epsilon"; "order" '
                'must be an array of the five vault names in the order their records appear from the beginning to the end of the '
                'document; "before_gamma" and "after_gamma" must be the vault names whose records appear immediately before and '
                'after the gamma record, or null when there is no such record. Base the order on document positions, not '
                'alphabetical order. Output JSON only.')
    gamma = order.index("gamma")
    return {"type": "case", "id": f"retrieval/{k}k/r{rep}", "kind": "chat", "group": "retrieval",
            # The whole rendered prompt plus the answer stays within k * 1024 positions (a power-of-two cache at 128K).
            "fit_tokens": k * 1024 - max_new - 64,
            "document": {"corpus": "prose", "start_paragraph": 97 * k + 13 * rep, "inserts": inserts},
            "messages": [{"role": "system", "content": f"You are a helpful assistant. Session {nonce(seed, 'retrieval', k, rep)}."},
                         {"role": "user", "content": "Read the following document carefully.\n\n<document>\n{{document}}\n</document>\n\n"
                                                     + question}],
            "think": think, "max_new_tokens": max_new,
            "grade": {"type": "retrieval", "codes": codes, "order": order,
                      "before_gamma": order[gamma - 1] if gamma > 0 else None,
                      "after_gamma": order[gamma + 1] if gamma + 1 < len(order) else None}}


# --------------------------------------------------------------------------- reading runs

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass   # a line cut short by a killed run
    return out


def load_arm(spec):
    """FILE:ARM -> (run headers, {case id: record}); the last record of a case wins."""
    path, _, arm = spec.rpartition(":")
    if not path or not arm:
        raise SystemExit(f"{spec}: RUN.jsonl:ARM")
    rows = read_jsonl(path)
    runs = [r for r in rows if r.get("type") == "run"]
    records = {r["case"]: r for r in rows if r.get("type") == "record" and r.get("arm") == arm}
    if not records:
        arms = sorted({r.get("arm") for r in rows if r.get("type") == "record"})
        failed = [r.get("error") for r in runs if r.get("status") == "load_failed"]
        raise SystemExit(f"{spec}: no records of arm '{arm}' (arms there: {', '.join(arms) or 'none'})"
                         + (f"; the model did not load: {failed[-1]}" if failed else ""))
    return {"spec": spec, "path": path, "arm": arm, "runs": runs, "records": records}


def load_cases(path):
    rows = read_jsonl(path)
    header = next((r for r in rows if r.get("type") == "header"), None)
    return header, {r["id"]: r for r in rows if r.get("type") == "case"}


# --------------------------------------------------------------------------- grading

def strip_thinking(text):
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text


def strip_fence(text):
    m = re.search(r"```(?:json|python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1), True) if m else (text, False)


def grade_retrieval(text, grade):
    body, _ = strip_fence(strip_thinking(text).strip())
    start, end = body.find("{"), body.rfind("}")
    try:
        actual = json.loads(body[start:end + 1]) if start >= 0 and end > start else None
    except ValueError:
        actual = None
    ok = isinstance(actual, dict)
    checks = {f"needle_{k}": ok and actual.get(k) == v for k, v in grade["codes"].items()}
    for key in ("order", "before_gamma", "after_gamma"):
        expected = grade[key]
        checks[f"position_{key}"] = ok and key in actual and actual[key] == expected
    return checks


ANSWER = re.compile(r"Answer:\s*\**\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I)


def grade_gsm8k(text, grade):
    found = ANSWER.findall(strip_thinking(text))
    if not found:
        return False
    try:
        return Decimal(found[-1].replace(",", "")) == Decimal(grade["answer"])
    except InvalidOperation:
        return False


HUMANEVAL_TIMEOUT = 10
HUMANEVAL_LIMITS = {"RLIMIT_CPU": 10, "RLIMIT_AS": 2 << 30, "RLIMIT_FSIZE": 1 << 20, "RLIMIT_CORE": 0, "RLIMIT_NOFILE": 64}


def humaneval_program(text, grade):
    """The candidate's code block (else its whole answer), the prompt prepended when the block does not define the entry
    point itself (a body completion), then the dataset's test and its check call."""
    code, _ = strip_fence(strip_thinking(text))
    if not re.search(r"^\s*def\s+" + re.escape(grade["entry_point"]) + r"\s*\(", code, re.M):
        code = grade["prompt"] + code
    return code + "\n\n" + grade["test"] + f"\n\ncheck({grade['entry_point']})\n"


def humaneval_worker(path):
    """In the child: resource limits that must hold (else no execution), common destructive calls removed, then the program."""
    import builtins
    import resource
    for name, value in HUMANEVAL_LIMITS.items():
        resource.setrlimit(getattr(resource, name), (value, value))
    source = Path(path).read_text(encoding="utf-8")
    import shutil as _shutil
    import subprocess as _subprocess
    for mod, names in ((os, ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
                             "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
                             "chroot", "lchown", "chdir")),
                       (_shutil, ("rmtree", "move", "chown")), (_subprocess, ("Popen",))):
        for n in names:
            if hasattr(mod, n):
                setattr(mod, n, None)
    builtins.exit = builtins.quit = None
    exec(compile(source, "<humaneval-candidate>", "exec"), {"__name__": "__main__"})
    return 0


def run_humaneval(program):
    """PASS / FAIL / TIMEOUT / LIMITS_UNAVAILABLE for one program, in a fresh directory and session."""
    with tempfile.TemporaryDirectory(prefix="he-") as d:
        src = Path(d) / "candidate.py"
        src.write_text(program, encoding="utf-8")
        child = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).resolve()), "humaneval-worker", str(src)],
                                 cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                 env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"})
        try:
            _, err = child.communicate(timeout=HUMANEVAL_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()   # reap it and close its pipes
            return "TIMEOUT"
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child.returncode == 0:
            return "PASS"
        if b"setrlimit" in err or b"not allowed" in err and b"resource" in err:
            return "LIMITS_UNAVAILABLE"
        return "FAIL"


def mcnemar_p(b, c):
    """Exact two-sided McNemar p: the discordant pairs as a fair coin."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- report

GATES = {
    "t2-attn8": {"ppl": ("delta_ppl_le", 0.5, "nll_se_or", 0.005), "continue": ("delta_ppl_le", 0.3), "layers": (0.03, 0.06),
                 "greedy_gap": 1.0, "retrieval": "no_regression", "tasks": (1, 0.1)},
    "t3-kvq8": {"ppl": ("abs_delta_ppl_le", 0.3), "continue": ("abs_delta_ppl_le", 0.3), "layers": None,
                "greedy_gap": None, "retrieval": "all_pass", "tasks": (1, 0.1)},
    "none": {"ppl": None, "continue": None, "layers": None, "greedy_gap": None, "retrieval": None, "tasks": None},
}
GATE_TEXT = {
    "t2-attn8": "design t2-attn8 sec. 12.3: ppl every L dPPL <= +0.5% and mean dNLL within 2 SE or <= +0.005 nats; continue dPPL <= "
                "+0.3%; layers median rel-RMS <= 3%, max <= 6%; greedy divergences only at reference top-2 gaps < 1.0 logit; "
                "retrieval: the candidate passes every check the reference passes; tasks: net loss <= 1 problem and McNemar p > 0.1",
    "t3-kvq8": "design t3 sec. 3 (b2): |dPPL| <= 0.3% for ppl and continue; retrieval 100% for the candidate; greedy identity reported; "
               "tasks: net loss <= 1 problem and McNemar p > 0.1 (t2's task gate, borrowed: t3 names the tasks, not a threshold)",
    "none": "no gates",
}


def block_se(diff_lists, block=128):
    means = []
    for diffs in diff_lists:
        for i in range(0, len(diffs) - block + 1, block):
            means.append(sum(diffs[i:i + block]) / block)
    return statistics.stdev(means) / math.sqrt(len(means)) if len(means) >= 2 else float("nan")


def topk_of(rec):
    k, data = rec.get("topk_k"), rec.get("topk")
    if not k or not data:
        return None
    raw = base64.b64decode(data)
    n = len(raw) // 8
    return list(struct.unpack(f"<{n}i", raw[:4 * n]))[::k]   # each row's argmax id


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}%"


class Report:
    def __init__(self, ref, cand, cases, gates, execute_humaneval, floor=None):
        self.ref, self.cand, self.cases, self.gates, self.floor = ref, cand, cases, GATES[gates], floor
        self.gate_name, self.execute = gates, execute_humaneval
        self.lines, self.summary, self.verdicts = [], {"reference": ref["spec"], "candidate": cand["spec"], "gates": gates}, []

    def out(self, line=""):
        self.lines.append(line)

    def verdict(self, name, ok, detail):
        state = "NOT RUN" if ok is None else ("PASS" if ok else "FAIL")
        self.verdicts.append((name, state, detail))

    def pairs(self, predicate, cand=None):
        """(case, ref record, cand record) of the cases both arms ran; the others are listed as not paired."""
        both, unpaired = [], []
        cand = cand or self.cand
        for cid, case in self.cases.items():
            if not predicate(case):
                continue
            r, c = self.ref["records"].get(cid), cand["records"].get(cid)
            if r and c and r.get("status") == "ok" and c.get("status") == "ok":
                key = "input_sha256" if "input_sha256" in r else "prompt_sha256"
                if r.get(key) != c.get(key):
                    unpaired.append((cid, "the arms' inputs differ"))
                else:
                    both.append((case, r, c))
            else:
                state = lambda x: "absent" if not x else x.get("status") + (f" ({x.get('reason') or x.get('error')})" if x.get("status") != "ok" else "")
                unpaired.append((cid, f"reference {state(r)}, candidate {state(c)}"))
        return both, unpaired

    def not_paired(self, unpaired):
        for cid, why in unpaired:
            self.out(f"- not paired: `{cid}`: {why}")

    def run(self):
        self.header()
        for kind in ("ppl", "continue"):
            self.scored(kind)
        self.layers()
        self.greedy()
        self.retrieval()
        for group in ("gsm8k", "humaneval"):
            self.task(group)
        if self.floor:
            self.out(f"## Noise floor: `{self.floor['spec']}` against the same reference (no gates)")
            self.out()
            self.out("A numerics difference already shipped (e.g. another backend over the same model and inputs): the candidate's "
                     "deltas read against it.")
            self.out()
            for kind in ("ppl", "continue"):
                self.scored(kind, cand=self.floor)
            self.greedy(cand=self.floor)
        self.evidence()
        self.out("## Gates")
        self.out(f"Pre-registered: {GATE_TEXT[self.gate_name]}.")
        self.out()
        self.out("| gate | result | detail |")
        self.out("|---|---|---|")
        for name, state, detail in self.verdicts:
            self.out(f"| {name} | {state} | {detail} |")
        self.summary["gates_result"] = [{"gate": n, "result": s, "detail": d} for n, s, d in self.verdicts]
        return "\n".join(self.lines) + "\n"

    def header(self):
        self.out("# Long-context quality: reference vs candidate")
        self.out()
        for role, arm in (("reference", self.ref), ("candidate", self.cand)):
            run = arm["runs"][-1] if arm["runs"] else {}
            kinds = {}
            for r in arm["records"].values():
                kinds.setdefault(r.get("status"), 0)
                kinds[r.get("status")] += 1
            self.out(f"- **{role}** `{arm['spec']}`: {run.get('architecture')} on {run.get('backend')} ({run.get('device')}), KV "
                     f"{run.get('kv_cache_dtype')}, prompt attention {next((a['prompt_attention'] for a in run.get('arms', []) if a['name'] == arm['arm']), '?')}, "
                     f"records {', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}; label {run.get('label')!r}")
        self.out()

    def scored(self, kind, cand=None):
        floor = cand is not None
        both, unpaired = self.pairs(lambda c: c["kind"] == kind, cand)
        if not both and not unpaired:
            return
        title = "prompt path (scored prompt chunks)" if kind == "ppl" else "decode path (scored decode rows after a prompt-path prefix)"
        self.out(f"{'### noise floor,' if floor else '##'} {kind}: {title}")
        self.out()
        self.out("| corpus | L | windows | tokens | PPL ref | PPL cand | dPPL | mean dNLL (SE) | KL mean | KL p99 | top-1 agree |")
        self.out("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        groups = {}
        for case, r, c in both:
            groups.setdefault((case["corpus"], case["tokens"]), []).append((r, c))
        gate, ok_all, rows = None if floor else self.gates[kind], True, []
        for (corpus, L), items in sorted(groups.items()):
            diffs = [[b - a for a, b in zip(r["nll"], c["nll"])] for r, c in items]
            nr = [x for r, _ in items for x in r["nll"]]
            nc = [x for _, c in items for x in c["nll"]]
            n = len(nr)
            mean = sum(sum(d) for d in diffs) / n
            se = block_se(diffs)
            dppl = (math.exp(sum(nc) / n - sum(nr) / n) - 1) * 100
            kls = [x for _, c in items for x in (c.get("kl") or [])]
            kl_ok = all(c.get("kl_reference") and (c["kl_reference"].endswith(":" + self.ref["arm"]) or
                                                   c["kl_reference"].startswith(self.ref["arm"] + " ")) for _, c in items)
            klm = sum(kls) / len(kls) if kls and kl_ok else None
            klp = sorted(kls)[int(0.99 * (len(kls) - 1))] if kls and kl_ok else None
            agree = sum(a == b for r, c in items for a, b in zip(r["top1"], c["top1"])) / n
            row = {"corpus": corpus, "tokens": L, "windows": len(items), "scored": n, "ppl_ref": math.exp(sum(nr) / n),
                   "ppl_cand": math.exp(sum(nc) / n), "delta_ppl_pct": dppl, "mean_dnll": mean, "se": se, "kl_mean": klm, "kl_p99": klp,
                   "top1_agree": agree}
            rows.append(row)
            self.out(f"| {corpus} | {L // 1024}K | {len(items)} | {n} | {row['ppl_ref']:.4f} | {row['ppl_cand']:.4f} | {fmt_pct(dppl)} | "
                     f"{mean:+.5f} ({se:.5f}) | {'n/a' if klm is None else f'{klm:.2e}'} | {'n/a' if klp is None else f'{klp:.2e}'} | "
                     f"{agree * 100:.2f}% |")
            if gate:
                if gate[0] == "delta_ppl_le":
                    ok = dppl <= gate[1] and (len(gate) < 4 or abs(mean) <= 2 * se or mean <= gate[3])
                else:
                    ok = abs(dppl) <= gate[1]
                ok_all &= ok
        self.not_paired(unpaired)
        self.out()
        self.summary[("floor_" if floor else "") + kind] = rows
        if gate:
            self.verdict(f"{kind} (every corpus and L run)", ok_all if rows else None,
                         f"{len(rows)} groups; worst dPPL {max((r['delta_ppl_pct'] for r in rows), default=float('nan')):+.3f}%")

    def layers(self):
        recs = [r for cid, r in self.cand["records"].items() if r.get("kind") == "layers" and r.get("status") == "ok"]
        recs += [r for cid, r in self.ref["records"].items() if r.get("kind") == "layers" and r.get("status") == "ok"
                 and cid not in self.cand["records"]]
        if not recs:
            return
        self.out("## layers: 8-bit prompt attention vs the 16-bit kernel on the same rows (real activations)")
        self.out()
        rels = [x[1] for r in recs for x in r.get("checks", [])]
        for r in recs:
            self.out(f"- `{r['case']}` ({r['arm']}): {len(r.get('checks', []))} launches, rel RMS median {r.get('median_rel_rms', float('nan')) * 100:.2f}%, "
                     f"max {r.get('max_rel_rms', float('nan')) * 100:.2f}%, max |d| {r.get('max_abs')}")
        self.out()
        if rels:
            med, mx = statistics.median(rels), max(rels)
            self.summary["layers"] = {"launches": len(rels), "median_rel_rms": med, "max_rel_rms": mx}
            if self.gates["layers"]:
                self.verdict("layers", med <= self.gates["layers"][0] and mx <= self.gates["layers"][1],
                             f"median {med * 100:.2f}%, max {mx * 100:.2f}% over {len(rels)} launches")

    def greedy(self, cand=None):
        floor = cand is not None
        both, unpaired = self.pairs(lambda c: c["kind"] == "greedy", cand)
        if not both and not unpaired:
            return
        self.out(f"{'### noise floor,' if floor else '##'} greedy: identity of the generated tokens")
        self.out()
        self.out("| case | tokens | identical prefix | identical | divergence: ref gap / cand gap |")
        self.out("|---|---:|---:|---|---|")
        gaps_at, identical, rows = [], 0, []
        for case, r, c in both:
            a, b = r["generated"], c["generated"]
            prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            same = a == b
            identical += same
            detail = "-"
            if not same and prefix < min(len(a), len(b)):
                gr, gc = r["gaps"][prefix], c["gaps"][prefix]
                gaps_at.append(gr)
                detail = f"at {prefix}: {gr:.3f} / {gc:.3f}"
            rows.append({"case": case["id"], "prefix": prefix, "identical": same, "ref_gap": gaps_at[-1] if detail != "-" else None})
            self.out(f"| {case['id']} | {case['tokens'] // 1024}K | {prefix}/{len(a)} | {'yes' if same else 'no'} | {detail} |")
        self.not_paired(unpaired)
        self.out()
        self.out(f"Identical: {identical}/{len(both)}; divergences at reference top-2 gaps {', '.join(f'{g:.3f}' for g in gaps_at) or 'none'}.")
        self.out()
        self.summary["floor_greedy" if floor else "greedy"] = {"identical": identical, "cases": len(both), "rows": rows}
        if self.gates["greedy_gap"] is not None and not floor:
            self.verdict("greedy (divergences at near-ties only)", all(g < self.gates["greedy_gap"] for g in gaps_at) if both else None,
                         f"{identical}/{len(both)} identical; max ref gap at a divergence {max(gaps_at, default=0):.3f}")

    def chat_pairs(self, group):
        return self.pairs(lambda c: c["kind"] == "chat" and c.get("group") == group)

    def retrieval(self):
        both, unpaired = self.chat_pairs("retrieval")
        if not both and not unpaired:
            return
        self.out("## retrieval: five vault records (codes, document order, gamma's neighbours)")
        self.out()
        self.out("| case | prompt tokens | ref checks | cand checks | regressions | identical text |")
        self.out("|---|---:|---:|---:|---|---|")
        regressions, cand_all, rows = 0, True, []
        for case, r, c in both:
            gr, gc = grade_retrieval(r.get("text"), case["grade"]), grade_retrieval(c.get("text"), case["grade"])
            lost = [k for k in gr if gr[k] and not gc[k]]
            regressions += len(lost)
            cand_all &= all(gc.values())
            rows.append({"case": case["id"], "prompt_tokens": c["prompt_tokens"], "ref": gr, "cand": gc})
            self.out(f"| {case['id']} | {c['prompt_tokens']} | {sum(gr.values())}/{len(gr)} | {sum(gc.values())}/{len(gc)} | "
                     f"{', '.join(lost) or '-'} | {'yes' if r.get('generated') == c.get('generated') else 'no'} |")
        self.not_paired(unpaired)
        self.out()
        self.summary["retrieval"] = rows
        rule = self.gates["retrieval"]
        if rule == "no_regression":
            self.verdict("retrieval (no check the reference passes is lost)", regressions == 0 if both else None, f"{regressions} lost checks")
        elif rule == "all_pass":
            self.verdict("retrieval (the candidate passes every check)", cand_all if both else None,
                         f"{sum(all(x['cand'].values()) for x in rows)}/{len(rows)} cases fully passed")

    def task(self, group):
        both, unpaired = self.chat_pairs(group)
        if not both and not unpaired:
            return
        self.out(f"## {group}")
        self.out()
        if group == "humaneval" and not self.execute:
            self.out(f"{len(both)} paired completions, not graded: pass --execute-humaneval to run the generated code (restricted subprocess).")
            self.not_paired(unpaired)
            self.out()
            self.verdict(group, None, "not graded (--execute-humaneval not given)")
            return
        grades, cache = [], {}
        for case, r, c in both:
            pair = []
            for rec in (r, c):
                if group == "gsm8k":
                    pair.append(grade_gsm8k(rec.get("text"), case["grade"]))
                else:
                    program = humaneval_program(rec.get("text"), case["grade"])
                    key = sha256_bytes(program.encode())
                    if key not in cache:
                        cache[key] = run_humaneval(program)
                    if cache[key] == "LIMITS_UNAVAILABLE":
                        raise SystemExit("HumanEval: the resource limits cannot be installed here; no program was counted")
                    pair.append(cache[key] == "PASS")
            grades.append((case["id"], pair[0], pair[1], r.get("generated") == c.get("generated")))
        ref_ok, cand_ok = sum(g[1] for g in grades), sum(g[2] for g in grades)
        b = sum(1 for g in grades if g[1] and not g[2])
        c_ = sum(1 for g in grades if g[2] and not g[1])
        p = mcnemar_p(b, c_)
        same = sum(g[3] for g in grades)
        self.out(f"Reference {ref_ok}/{len(grades)}, candidate {cand_ok}/{len(grades)}; lost {b}, gained {c_} (net {c_ - b:+d}); "
                 f"McNemar exact p = {p:.3f}; identical outputs {same}/{len(grades)}.")
        lost = [g[0] for g in grades if g[1] and not g[2]]
        if lost:
            self.out(f"Lost: {', '.join(lost)}.")
        self.not_paired(unpaired)
        self.out()
        self.summary[group] = {"ref": ref_ok, "cand": cand_ok, "n": len(grades), "lost": b, "gained": c_, "p": p, "identical": same}
        if self.gates["tasks"]:
            loss, pmin = self.gates["tasks"]
            self.verdict(group, (b - c_ <= loss and (p > pmin or c_ >= b)) if grades else None, f"net {c_ - b:+d}, p {p:.3f}")

    def evidence(self):
        self.out("## Arms' evidence")
        self.out()
        for role, arm in (("reference", self.ref), ("candidate", self.cand)):
            recs = [r for r in arm["records"].values() if r.get("status") == "ok"]
            launches = sum(r.get("attn8_launches", 0) for r in recs)
            refusals = sum(r.get("attn8_band_refusals", 0) for r in recs)
            marked = sum(r.get("marked_prompt_attentions", 0) for r in recs)
            skipped = [r for r in arm["records"].values() if r.get("status") != "ok"]
            self.out(f"- {role}: 8-bit prompt-attention launches {launches}, band refusals {refusals}"
                     f"{' (those chunks ran the 16-bit kernel)' if refusals else ''}, marked prompt-chunk attentions {marked}; "
                     f"not ok: {len(skipped)}")
            for r in skipped:
                self.out(f"  - `{r['case']}`: {r.get('status')}: {r.get('reason') or r.get('error')}")
        self.out()


def report(args):
    ref, cand = load_arm(args.ref), load_arm(args.cand)
    cases_path = args.cases or next((r.get("cases_file") for r in ref["runs"][::-1] if r.get("cases_file")), None)
    if not cases_path or not Path(cases_path).exists():
        raise SystemExit("--cases CASES.jsonl (the run's cases_file is not readable here)")
    _, cases = load_cases(cases_path)
    rep = Report(ref, cand, cases, args.gates, args.execute_humaneval, load_arm(args.floor) if args.floor else None)
    text = rep.run()
    sys.stdout.write(text)
    if args.out:
        Path(ignored_output(Path(args.out).parent), Path(args.out).name).write_text(text, encoding="utf-8")
    if args.json:
        Path(ignored_output(Path(args.json).parent), Path(args.json).name).write_text(json.dumps(rep.summary, indent=1) + "\n", encoding="utf-8")
    return 1 if any(state == "FAIL" for _, state, _ in rep.verdicts) else 0


# --------------------------------------------------------------------------- estimate

def fit_linear(xs, ys):
    """Least squares y = a + b x (b >= 0); a constant when x does not vary."""
    n = len(xs)
    if n == 0:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = max(0.0, sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx > 0 else 0.0
    return my - b * mx, b


def fit_prefill(points):
    """Prefill seconds = a L + b L^2 (L in thousands of tokens) by least squares through the origin."""
    s11 = sum(L ** 2 for L, _ in points)
    s12 = sum(L ** 3 for L, _ in points)
    s22 = sum(L ** 4 for L, _ in points)
    t1 = sum(L * t for L, t in points)
    t2 = sum(L * L * t for L, t in points)
    det = s11 * s22 - s12 * s12
    if len({round(L) for L, _ in points}) < 2 or det <= 0:
        return sum(t / L for L, t in points) / len(points), 0.0
    a, b = (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det
    if b < 0:
        return t1 / s11, 0.0
    return a, b


def arm_timings(recs):
    """One arm's measured costs: prefill(L) = a L + b L^2 seconds (L in K tokens, the prompt-path tokens), decode and scored-row
    seconds a token as linear functions of the context, the 8-bit check's cost a launch (layers), mean generated tokens a chat
    group."""
    prefill, decode, score, layers, generated = [], [], [], [], {}
    for r in recs:
        tokens = r.get("input_tokens") or r.get("prompt_tokens")
        if not tokens or r.get("prefill_s") is None:
            continue
        if r["kind"] == "layers":
            layers.append(r)
            continue
        prefill.append(((tokens - (r.get("score_tokens") or 0)) / 1024, r["prefill_s"]))
        if r["kind"] == "ppl":
            score.append((tokens / 1024, r["score_s"] / r["score_tokens"]))
        elif r["kind"] == "continue":
            decode.append((tokens / 1024, r["score_s"] / r["score_tokens"]))
        elif r.get("decode_s") is not None and r.get("generated"):
            decode.append(((tokens + len(r["generated"]) / 2) / 1024, r["decode_s"] / len(r["generated"])))
            if r["kind"] == "chat":
                generated.setdefault(r.get("group"), []).append(len(r["generated"]))
    if not prefill or not decode:
        return None
    t = {"prefill": fit_prefill(prefill), "decode": fit_linear([x for x, _ in decode], [y for _, y in decode]),
         "score": fit_linear([x for x, _ in score], [y for _, y in score]) if score else None,
         "generated": {g: statistics.mean(v) for g, v in generated.items()},
         "points": (len(prefill), min(x for x, _ in prefill), max(x for x, _ in prefill), len(decode))}
    a, b = t["prefill"]
    per_launch = [(r["prefill_s"] - (a * r["input_tokens"] / 1024 + b * (r["input_tokens"] / 1024) ** 2)) / len(r["checks"])
                  for r in layers if r.get("checks")]
    t["check_s"] = max(0.0, statistics.mean(per_launch)) if per_launch else None
    t["launches_per_k"] = statistics.mean(len(r["checks"]) / (r["input_tokens"] / 1024) for r in layers if r.get("checks")) if per_launch else None
    return t


def predict(case, t):
    a, b = t["prefill"]
    pre = lambda k: a * k + b * k * k
    dec = lambda k: t["decode"][0] + t["decode"][1] * k
    if case["kind"] in ("ppl", "continue"):
        L, S = case["tokens"] / 1024, case["score_tokens"]
        if case["kind"] == "ppl":
            per = (t["score"][0] + t["score"][1] * L) if t["score"] else (a + 2 * b * L) / 1024
            return pre(L - S / 1024) + per * S
        return pre(L - S / 1024) + dec(L) * S
    if case["kind"] == "layers":
        L = case["tokens"] / 1024
        return pre(L) + (t["check_s"] * t["launches_per_k"] * L if t["check_s"] is not None else pre(L))
    if case["kind"] == "greedy":
        L, n = case["tokens"] / 1024, case["new_tokens"]
        return pre(L) + dec(L + n / 2048) * n
    L = case["fit_tokens"] / 1024
    n = min(case["max_new_tokens"], t["generated"].get(case.get("group"), case["max_new_tokens"]))
    return pre(L) + dec(L + n / 2048) * n


def estimate(args):
    rows = read_jsonl(args.run)
    run = next((r for r in rows if r.get("type") == "run"), {})
    _, cases = load_cases(args.cases)
    arms = sorted({r["arm"] for r in rows if r.get("type") == "record" and r.get("status") == "ok"})
    if args.arm:
        arms = [a for a in arms if a in args.arm.split(",")]
    if not arms:
        raise SystemExit(f"{args.run}: no ok records" + (f" of arms {args.arm}" if args.arm else ""))
    by_kind, grand = {}, 0.0
    print(f"from {args.run} ({run.get('device')}, KV {run.get('kv_cache_dtype')}), extrapolated to {args.cases} ({len(cases)} cases):")
    for arm in arms:
        t = arm_timings([r for r in rows if r.get("type") == "record" and r.get("status") == "ok" and r["arm"] == arm])
        if t is None:
            print(f"- arm {arm}: no prefill and decode timings")
            continue
        a, b = t["prefill"]
        n, lo, hi, nd = t["points"]
        print(f"- arm {arm}: prefill(L) = {a:.4f} L + {b:.6f} L^2 s (L in K tokens; {n} points, {lo:.1f}K-{hi:.1f}K); decode "
              f"{t['decode'][0] * 1000:.1f} + {t['decode'][1] * 1000:.3f} L ms/token ({nd} points)"
              + (f"; scored rows {t['score'][0] * 1000:.3f} + {t['score'][1] * 1000:.4f} L ms/row" if t["score"] else "")
              + (f"; 8-bit check {t['check_s'] * 1000:.0f} ms/launch" if t["check_s"] is not None else "")
              + (f"; chat tokens generated {', '.join(f'{g} {v:.0f}' for g, v in t['generated'].items())}" if t["generated"] else ""))
        for case in cases.values():
            if case["kind"] == "layers" and t["check_s"] is None:
                continue   # the layers case runs in an 8bit arm only
            kind = case["kind"] if case["kind"] != "chat" else case.get("group", "chat")
            sec = predict(case, t)
            by_kind[kind] = by_kind.get(kind, 0.0) + sec
            grand += sec
    load = (run.get("load_s", 0) or 0) * args.processes
    print("\n| kind | GPU hours |")
    print("|---|---:|")
    for kind, sec in sorted(by_kind.items()):
        print(f"| {kind} | {sec / 3600:.2f} |")
    print(f"| model load x {args.processes} | {load / 3600:.2f} |")
    print(f"| **total** | **{(grand + load) / 3600:.2f}** |")
    return 0


# --------------------------------------------------------------------------- main

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["humaneval-worker"]:
        return humaneval_worker(argv[1])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cases", help="build a case file and its pinned corpora")
    c.add_argument("--out", required=True, help="output directory (under artifacts/ or docs/validation/, or outside the repository)")
    c.add_argument("--preset", choices=sorted(PRESETS), default="full")
    c.add_argument("--kinds", help=f"comma list of {', '.join(KINDS)} (default all)")
    c.add_argument("--seed", type=int, default=376)
    c.add_argument("--think", action="store_true", help="chat cases render with thinking enabled (default off)")
    c.add_argument("--repo", default=str(ROOT), help="git checkout holding the code corpus revision")
    c.add_argument("--code-revision", default=CODE_REVISION)
    c.add_argument("--downloads", default=str(ROOT / "artifacts" / "quality" / "downloads"))
    r = sub.add_parser("report", help="pair two arms, grade and check gates")
    r.add_argument("--ref", required=True, help="RUN.jsonl:ARM of the reference")
    r.add_argument("--cand", required=True, help="RUN.jsonl:ARM of the candidate")
    r.add_argument("--cases", help="the case file (default: the reference run's cases_file)")
    r.add_argument("--gates", choices=sorted(GATES), default="none")
    r.add_argument("--floor", help="RUN.jsonl:ARM of a noise-floor arm (a shipped numerics difference) shown against the reference")
    r.add_argument("--execute-humaneval", action="store_true", help="run generated HumanEval code in a restricted subprocess")
    r.add_argument("--out", help="write the Markdown report here (ignored directories only)")
    r.add_argument("--json", help="write the summary JSON here (ignored directories only)")
    e = sub.add_parser("estimate", help="GPU time of a case file from a run's measured timings")
    e.add_argument("--run", required=True)
    e.add_argument("--cases", required=True)
    e.add_argument("--arm", help="comma list of the run's arms to extrapolate (default every arm in the run, summed)")
    e.add_argument("--processes", type=int, default=1, help="model loads the protocol needs (one per process)")
    args = ap.parse_args(argv)
    return {"cases": build_cases, "report": report, "estimate": estimate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
