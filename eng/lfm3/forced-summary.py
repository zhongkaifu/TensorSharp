#!/usr/bin/env python3
# Summarize eng/ForcedLogitProbe runs (each directory's metrics.json, written with --reference): one line per run with the
# top-1 mismatches over every teacher-forced row, the median / max relative L2 of the logits and the largest logit
# difference, then one line per case. --markdown prints a table row per run instead.
#
#   eng/lfm3/forced-summary.py [--markdown] RUN_DIR [RUN_DIR ...]
import json
import sys


def main(argv):
    markdown = "--markdown" in argv
    dirs = [a for a in argv if a != "--markdown"]
    if markdown:
        print("| run | complete | rows | top-1 mismatches | median relL2 | max relL2 | max abs dlogit | max KL |")
        print("|---|---|---|---|---|---|---|---|")
    for d in dirs:
        try:
            with open(d + "/metrics.json") as f:
                m = json.load(f)
        except (OSError, ValueError) as e:
            print(f"{d}: missing ({e})")
            continue
        s = m["summary"]
        dmax = max((r["max_abs"] for c in m["cases"] for r in c["metrics"]), default=float("nan"))
        if markdown:
            print(f"| {d} | {m['complete']} | {s['rows']} | {s['top1_mismatches']} | {s['median_relative_l2']:.2e} | "
                  f"{s['max_relative_l2']:.2e} | {dmax:.3f} | {s['max_kl']:.2e} |")
            continue
        print(f"== {d} complete={m['complete']} rows={s['rows']} top1_mismatch={s['top1_mismatches']} "
              f"med_relL2={s['median_relative_l2']:.2e} max_relL2={s['max_relative_l2']:.2e} max|dlogit|={dmax:.3f} "
              f"max_kl={s['max_kl']:.2e}")
        for c in m["cases"]:
            cs = c["summary"]
            first = c["first_divergence"]["step"] if c.get("first_divergence") else None
            print(f"   {c['name']:14s} prompt={c['prompt_tokens']:5d} top1_mismatch={cs['top1_mismatches']} "
                  f"max_relL2={cs['max_relative_l2']:.2e} max|dlogit|={max(r['max_abs'] for r in c['metrics']):.4f} first_div={first}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
