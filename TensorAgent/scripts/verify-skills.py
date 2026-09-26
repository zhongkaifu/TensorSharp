#!/usr/bin/env python3
"""Decide which skills can actually run inside the TensorAgent app.

A skill on a phone is not the same proposition as a skill on a workstation. There
is no pip at run time, no browser to drive, no child process to spawn, and the
only interpreter is the CPython that was staged into the app bundle. A skill whose
instructions are excellent but whose scripts import a package that is not there
will fail on the user's first attempt, which is worse than not offering it.

So this script answers one question per skill, by inspection rather than by
optimism:

  * every script it ships is in a language the app can run;
  * every module those scripts import is either in the staged standard library,
    in the staged site packages, or is a sibling file of the skill itself;
  * no shell script it ships runs a command the in-app shell does not have (a Node
    package manager or bundler);
  * nothing it ships reaches for a capability iOS does not have.

Run it against the staged runtime:

    python3 TensorAgent/scripts/verify-skills.py \\
        --skills <a checkout of github.com/anthropics/skills>/skills \\
        --runtime TensorAgent/python-runtime/simulator

It prints a verdict per skill and exits non-zero if a skill named with --require
does not pass, which is what makes it usable from CI.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

# Capabilities the app does not have, and the module that would reach for each.
# A skill that imports one of these cannot work here no matter what else is true.
UNAVAILABLE = {
    "subprocess": "there is no way to start a process on iOS",
    "multiprocessing": "there is no way to start a process on iOS",
    "webbrowser": "there is no external browser to open",
    "playwright": "a browser engine cannot be bundled or driven",
    "selenium": "a browser engine cannot be bundled or driven",
    "pdf2image": "it shells out to poppler, which cannot be bundled",
    "mcp": "an MCP server would need a process and a socket",
    "anthropic": "it needs the network, which is off unless the user allows it",
    "tkinter": "there is no desktop toolkit",
    "ctypes": "loading arbitrary native code would walk around every check",
}

# Interpreters the in-process shell can dispatch to.
RUNNABLE_SUFFIXES = {".py": "python", ".sh": "sh", ".bash": "sh", ".js": "node", ".mjs": "node"}

# Commands a shell script may run that the app's in-process shell does not have. The
# shell has a JavaScriptCore `node` and no package manager (ShellMissingCommand.cs says
# npm and npx are missing on the device, and there is no pnpm or yarn builtin), so a
# script that installs or runs a Node tool fails at its first line on the phone. These
# are what the Python import check above could never see: a shell wrapper has no imports.
UNAVAILABLE_COMMANDS = {
    "npm": "there is no Node package manager in the app",
    "npx": "there is no Node package manager in the app",
    "pnpm": "there is no Node package manager in the app",
    "yarn": "there is no Node package manager in the app",
    "parcel": "a Node bundler cannot be installed or run in the app",
    "vite": "a Node bundler cannot be installed or run in the app",
}


@dataclass
class Verdict:
    name: str
    ok: bool
    kind: str
    scripts: int = 0
    reasons: list[str] = field(default_factory=list)
    missing: set[str] = field(default_factory=set)

    def line(self) -> str:
        mark = "PASS" if self.ok else "SKIP"
        detail = f"{self.kind}, {self.scripts} script(s)"
        if self.reasons:
            detail += " — " + "; ".join(sorted(set(self.reasons)))
        return f"  [{mark}] {self.name:<24} {detail}"


def staged_modules(runtime: Path) -> set[str]:
    """Every top-level module name the bundled interpreter can import."""
    names: set[str] = set(sys.builtin_module_names)

    stdlib = runtime / "python" / "lib" / "python3.13"
    packages = runtime / "python" / "app_packages"
    for root in (stdlib, packages):
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if entry.name.startswith("_") and entry.name not in {"__future__"}:
                # Private modules are still importable; keep them.
                names.add(entry.stem)
            if entry.is_dir() and (entry / "__init__.py").exists():
                names.add(entry.name)
            elif entry.suffix == ".py":
                names.add(entry.stem)

    # Compiled extensions are staged as one framework per dotted module name.
    frameworks = runtime / "Frameworks"
    if frameworks.is_dir():
        for entry in frameworks.iterdir():
            if entry.suffix == ".framework":
                names.add(entry.stem.split(".")[0])

    return names


def imports_of(path: Path) -> set[str]:
    """Top-level module names a Python file imports, absolute imports only."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # A relative import resolves inside the skill and needs nothing staged.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def commands_of(path: Path) -> set[str]:
    """Names of UNAVAILABLE_COMMANDS a shell script runs, comments ignored.

    A command counts where one can start: at the beginning of a line or after ;, &, |,
    (, `, $( or a keyword such as `then`/`do`, and as the argument of `exec`, `command`
    or `command -v` (a script that checks for pnpm and installs it with npm when it is
    missing is reaching for both).
    """
    found: set[str] = set()
    names = "|".join(sorted(UNAVAILABLE_COMMANDS))
    pattern = re.compile(
        r"(?:^|[;&|(`]|\$\(|\b(?:then|do|else|exec|command(?:\s+-v)?|xargs|env|sudo)\s)\s*(" + names + r")\b")
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0] if not raw.lstrip().startswith("#!") else ""
        for match in pattern.finditer(line):
            found.add(match.group(1))
    return found


def inspect(skill: Path, available: set[str]) -> Verdict:
    name = skill.name
    if not (skill / "SKILL.md").is_file():
        return Verdict(name, False, "not a skill", reasons=["no SKILL.md"])

    scripts = [p for p in skill.rglob("*") if p.is_file() and p.suffix in RUNNABLE_SUFFIXES]
    python_files = [p for p in scripts if p.suffix == ".py"]

    if not scripts:
        # Instructions only. Nothing to run means nothing to fail: these work here
        # exactly as well as they work anywhere.
        return Verdict(name, True, "instructions only")

    verdict = Verdict(name, True, "scripts", scripts=len(scripts))

    # A sibling module inside the skill satisfies its own imports.
    # A module that ships with the skill satisfies its own import. Both spellings
    # count: a package with an __init__.py, and the implicit namespace package a
    # bare directory becomes when the script beside it is run.
    local = {p.stem for p in skill.rglob("*.py")}
    local |= {d.name for d in skill.rglob("*") if d.is_dir() and not d.name.startswith(".")}

    for path in python_files:
        for module in imports_of(path):
            # The unavailable list is checked FIRST and deliberately. subprocess and
            # webbrowser are both in the staged standard library, so an
            # "is it importable" test passes them and the skill fails later, at the
            # call, in front of the user. Importable is not the same as usable.
            if module in UNAVAILABLE:
                verdict.ok = False
                verdict.reasons.append(f"{module}: {UNAVAILABLE[module]}")
                continue
            if module in local or module in available:
                continue
            verdict.ok = False
            verdict.missing.add(module)

    if verdict.missing:
        verdict.reasons.append("not in the bundled runtime: " + ", ".join(sorted(verdict.missing)))

    # Shell scripts have no imports to check, so check what they run. Every one of these
    # used to pass unexamined, which is how playwright (npx) and web-artifacts-builder
    # (pnpm, npm, parcel) were marked as working in the app.
    for path in scripts:
        if path.suffix not in (".sh", ".bash"):
            continue
        for command in sorted(commands_of(path)):
            verdict.ok = False
            verdict.reasons.append(f"{path.relative_to(skill)} runs {command}: {UNAVAILABLE_COMMANDS[command]}")

    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skills", required=True, type=Path, help="directory of skill folders")
    parser.add_argument("--runtime", required=True, type=Path, help="a staged python-runtime slice")
    parser.add_argument("--json", type=Path, help="write the verdicts here as JSON")
    parser.add_argument("--copy-to", type=Path, help="copy every passing skill into this directory")
    parser.add_argument("--require", nargs="*", default=[], help="fail if one of these does not pass")
    args = parser.parse_args()

    available = staged_modules(args.runtime)
    if not available:
        print(f"error: no staged modules found under {args.runtime}; run prepare-python.sh first", file=sys.stderr)
        return 2

    print(f"Bundled runtime offers {len(available)} importable top-level modules.\n")

    verdicts = [inspect(p, available) for p in sorted(args.skills.iterdir()) if p.is_dir()]
    passed = [v for v in verdicts if v.ok]

    print("Skills that work end to end in the app:")
    for verdict in verdicts:
        if verdict.ok:
            print(verdict.line())
    print("\nSkills that do not, and why:")
    for verdict in verdicts:
        if not verdict.ok:
            print(verdict.line())

    print(f"\n{len(passed)} of {len(verdicts)} skills pass.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([
            {"name": v.name, "ok": v.ok, "kind": v.kind, "scripts": v.scripts, "reasons": sorted(set(v.reasons))}
            for v in verdicts
        ], indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    if args.copy_to:
        import shutil
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for verdict in passed:
            source = args.skills / verdict.name
            target = args.copy_to / verdict.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        print(f"Copied {len(passed)} skills into {args.copy_to}")

    missing_required = [name for name in args.require if name not in {v.name for v in passed}]
    if missing_required:
        print(f"\nerror: required skill(s) did not pass: {', '.join(missing_required)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
