#!/usr/bin/env python3
"""Compare two MuseGlimmerKvGrowProbe reports position by position.

    compare.py REFERENCE.json CANDIDATE.json [--logits REF.f16 CAND.f16] [--mark 2048 4096]

Reports the first step whose argmax differs, argmax agreement before and after each
--mark position, and the cosine similarity of the full logit vectors in the same
windows: from the two logits files (--logits), or the cosines the candidate recorded
against the reference's logits file (the probe's --logits-ref; Steps[].Cos). Exit
status 1 when any compared step's argmax differs from the reference, or any cosine
drops below --min-cos. --margins also prints, for every argmax difference, the
reference's and the candidate's top-two gaps (the near-tie evidence).
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--logits", nargs=2, metavar=("REF", "CAND"))
    ap.add_argument("--mark", nargs="*", type=int, default=[2048])
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--margins", action="store_true")
    ap.add_argument("--cos-field", default="Cos", help="the candidate's recorded cosine to read: Cos (--logits-ref) or Cos2 (--logits-ref2)")
    args = ap.parse_args()

    ref, cand = load(args.reference), load(args.candidate)
    if ref["PromptTokens"] != cand["PromptTokens"]:
        print("prompts differ", file=sys.stderr)
        return 2
    rs, cs = ref["Steps"], cand["Steps"]
    n = min(len(rs), len(cs))
    forced = any(c["Chosen"] != c["Argmax"] for c in cs)
    first_diff = next((i for i in range(n) if rs[i]["Argmax"] != cs[i]["Argmax"]), None)
    print(f"reference {args.reference}: scenario={ref['Scenario']} capacity->{ref['FinalCapacity']} env={ref['Environment']}")
    print(f"candidate {args.candidate}: scenario={cand['Scenario']} capacity->{cand['FinalCapacity']} env={cand['Environment']} teacher-forced={forced}")
    print(f"prompt tokens {len(ref['PromptTokens'])}, compared steps {n}")
    if first_diff is None:
        print("argmax identical at every step")
    else:
        print(f"first argmax difference at step {first_diff} (position {cs[first_diff]['Position']}): "
              f"ref {rs[first_diff]['Argmax']} cand {cs[first_diff]['Argmax']}")

    marks = sorted(args.mark)
    edges = [-1] + marks + [1 << 30]
    windows = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def window_of(pos):
        for lo, hi in windows:
            if lo <= pos < hi:
                return (lo, hi)
        return windows[-1]

    agree = {w: [0, 0] for w in windows}
    for i in range(n):
        w = window_of(cs[i]["Position"])
        agree[w][1] += 1
        agree[w][0] += rs[i]["Argmax"] == cs[i]["Argmax"]
    for w in windows:
        a, t = agree[w]
        if t:
            print(f"  positions [{max(w[0], 0)}, {w[1] if w[1] < (1 << 30) else 'end'}): argmax agreement {a}/{t} = {a / t:.3f}")

    failed = False
    if first_diff is not None and cs[first_diff]["Position"] < marks[0]:
        # Before the first mark nothing about the grow has happened yet; any
        # difference there is a different bug (or nondeterminism) worth knowing.
        print("  NOTE: divergence precedes the first mark")

    cos = None
    if args.logits:
        import numpy as np
        vocab = args.vocab or None
        a = np.fromfile(args.logits[0], dtype=np.float16)
        b = np.fromfile(args.logits[1], dtype=np.float16)
        if vocab is None:
            vocab = a.size // len(rs)
        a = a.reshape(-1, vocab)[:n].astype(np.float32)
        b = b.reshape(-1, vocab)[:n].astype(np.float32)
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = list(num / np.maximum(den, 1e-30))
    elif n and all(c.get(args.cos_field) is not None for c in cs[:n]):
        # The candidate's own cosines against the reference's logits file (--logits-ref / --logits-ref2).
        ref_file = cand.get("LogitsReference" if args.cos_field == "Cos" else "LogitsReference2")
        print(f"  cosines recorded by the candidate ({args.cos_field}) against {ref_file}")
        cos = [c[args.cos_field] for c in cs[:n]]
    if cos is not None:
        for w in windows:
            idx = [i for i in range(n) if window_of(cs[i]["Position"]) == w]
            if idx:
                c = [cos[i] for i in idx]
                print(f"  positions [{max(w[0], 0)}, {w[1] if w[1] < (1 << 30) else 'end'}): cosine min {min(c):.6f} "
                      f"mean {sum(c) / len(c):.6f} (steps {len(idx)})")
        worst = min(range(n), key=lambda i: cos[i])
        print(f"  worst cosine {cos[worst]:.6f} at step {worst} position {cs[worst]['Position']}")
        if cos[worst] < args.min_cos:
            failed = True
    if first_diff is not None:
        failed = True
        if forced:
            # Teacher-forced: both runs saw identical inputs, so argmax differences
            # are attributable to the forward pass alone.
            mism = sum(1 for i in range(n) if rs[i]["Argmax"] != cs[i]["Argmax"])
            print(f"  teacher-forced argmax mismatches: {mism}/{n}")
        if args.margins:
            def gap(step):
                top = step["Top"]
                return top[0]["Logit"] - top[1]["Logit"] if len(top) > 1 else float("inf")
            for i in range(n):
                if rs[i]["Argmax"] != cs[i]["Argmax"]:
                    print(f"  step {i} position {cs[i]['Position']}: ref {rs[i]['Argmax']} (top-two gap {gap(rs[i]):.4f}), "
                          f"cand {cs[i]['Argmax']} (gap {gap(cs[i]):.4f})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
