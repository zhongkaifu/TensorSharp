#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""The prompt-step windows of every fresh prompt in a stamped server log (issue #383).

  python3 eng/validation/prefill-windows.py server.log [--min-tokens 5000]

A fresh prompt is a request whose "Radix prompt reuse" line reads "0/N tokens" with N >= --min-tokens. Its window runs
from the start of its first prompt step ([step] line of TS_SCHED_STEP_LOG=1: the step's end stamp t= minus its ms=) to the
end of its last one (the steps whose work names the request, "<id4>:P<rows>"). Prints one LABEL=HH:MM:SS.mmm-HH:MM:SS.mmm
argument a fresh prompt (the label is its token count), for eng/validation/nsys-prefill-classes.py, and on stderr the
steps' count, rows and summed milliseconds. Standard library only.
"""
import re
import sys
from datetime import datetime, timedelta

args = sys.argv[1:]
min_tokens = 5000
if "--min-tokens" in args:
    i = args.index("--min-tokens"); min_tokens = int(args[i + 1]); del args[i:i + 2]
lines = open(args[0], errors="replace").read().splitlines()
fmt = "%H:%M:%S.%f"
out = []
for i, line in enumerate(lines):
    m = re.search(r"Radix prompt reuse for chat-([0-9a-f]+): 0/(\d+) tokens", line)
    if not m or int(m.group(2)) < min_tokens:
        continue
    rid, tokens = m.group(1)[-4:], int(m.group(2))
    steps = []
    for later in lines[i + 1:]:
        s = re.match(r"\S+ \[step\] t=(\S+) gap=[\d.]+ ms=([\d.]+) \S+ (.*)", later)
        if not s:
            continue
        work = re.findall(rf"{rid}:([PD])(\d+)", s.group(3))
        if not work:
            if steps:
                continue   # another request's step in between
            continue
        if work[0][0] != "P":
            break
        steps.append((s.group(1), float(s.group(2)), int(work[0][1])))
    if not steps:
        continue
    t_end = datetime.strptime(steps[-1][0], fmt)
    t0 = datetime.strptime(steps[0][0], fmt) - timedelta(milliseconds=steps[0][1])
    out.append(f"{tokens}={t0.strftime(fmt)[:-3]}-{t_end.strftime(fmt)[:-3]}")
    print(f"{tokens}: {len(steps)} prompt steps, rows {[r for _, _, r in steps]}, {sum(ms for _, ms, _ in steps):.0f} ms summed",
          file=sys.stderr)
print(" ".join(out))
