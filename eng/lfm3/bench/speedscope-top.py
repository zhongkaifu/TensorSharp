#!/usr/bin/env python3
"""Self and inclusive time per frame of a dotnet-trace speedscope file (`dotnet-trace collect --profile
dotnet-sampled-thread-time --format Speedscope`), summed over every thread: where the managed CPU lane's time goes.

usage: speedscope-top.py FILE.speedscope.json [--top N] [--match REGEX]
  --match  also print the inclusive time of the frames whose names match (e.g. 'VecDot|Dequantize|Attention')
Sampled thread time counts waiting threads too: a spinning or parked pool worker shows under its wait frame.
"""
import argparse
import collections
import json
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--match", default=None)
    args = ap.parse_args()
    data = json.load(open(args.file))
    frames = [f["name"] for f in data["shared"]["frames"]]
    # dotnet-trace's sampled thread time ends each stack in a pseudo frame (CPU_TIME, UNMANAGED_CODE_TIME, ...): the self
    # time goes to the real frame under it, and the pseudo frame names the kind.
    pseudo = {i for i, n in enumerate(frames) if n.isupper() and " " not in n}
    kind_t = collections.Counter()
    self_t = collections.Counter()
    incl_t = collections.Counter()
    total = 0.0
    for prof in data["profiles"]:
        if prof.get("type") != "evented":
            continue
        stack, last = [], None
        for ev in prof["events"]:
            at = ev["at"]
            if stack and last is not None:
                dt = at - last
                real = [f for f in stack if f not in pseudo]
                if real:
                    self_t[real[-1]] += dt
                if stack[-1] in pseudo:
                    kind_t[frames[stack[-1]]] += dt
                for f in set(stack):
                    incl_t[f] += dt
                total += dt
            last = at
            if ev["type"] == "O":
                stack.append(ev["frame"])
            elif ev["type"] == "C" and stack:
                stack.pop()
    unit = data["profiles"][0].get("unit", "?") if data["profiles"] else "?"
    print(f"{args.file}: {total:.0f} {unit} of thread time over {len(data['profiles'])} threads")
    print("  " + ", ".join(f"{k} {100 * t / total:.1f}%" for k, t in kind_t.most_common()))
    print("  self time:")
    for f, t in self_t.most_common(args.top):
        print(f"    {t:10.0f} {100 * t / total:6.1f}%  {frames[f][:140]}")
    if args.match:
        rx = re.compile(args.match)
        print(f"  inclusive time of frames matching {args.match!r}:")
        for f, t in sorted(((f, t) for f, t in incl_t.items() if rx.search(frames[f])), key=lambda x: -x[1])[:args.top]:
            print(f"    {t:10.0f} {100 * t / total:6.1f}%  {frames[f][:140]}")


if __name__ == "__main__":
    sys.exit(main())
