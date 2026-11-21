#!/usr/bin/env python3
"""Per-step kernel breakdowns of a serving run profiled with Nsight Systems (an `nsys export --type=sqlite` of a capture
taken with `--trace=cuda --cuda-graph-trace=node`), as used for issue #391 (concurrent DFlash2 speculation on direct CUDA).

  units    <db>                      every launch unit (a graph launch, or one kernel launch): per graph id its kernels,
                                     launches, median GPU span and kernel time, sorted by total time
  graph    <db> <ids> [grid] [top]   per-kernel time per launch of the given graph ids (comma separated); with `grid` each
                                     kernel's grid and block shape apart
  timeline <db> <from> <to>          the launch units between two unit indices (from `units`): host call, GPU span, the gap
                                     before each graph launch, the kernels launched one by one in between
  phases   <db> [steps] [first]      a run whose kernels are launched one by one (no graphs: 038fb303's direct engine), cut into
                                     steps and phases by kernel name: draft (embed .. df_select), verify (.. argmax), commit
                                     (.. rows_by_index), catch-up (.. the next embed); per-phase spans and kernels, and gaps

The units of a database are cached beside it (<db>.units.pkl)."""
import collections
import os
import pickle
import sqlite3
import statistics
import sys


def load_units(db):
    cache = db + ".units.pkl"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(db):
        with open(cache, "rb") as f:
            return pickle.load(f)
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    api = {}
    for start, end, corr, nid in c.execute("select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[corr] = (names.get(nid, "?"), start, end)
    kern = collections.defaultdict(list)
    for start, end, corr, sn, gid in c.execute("select start, end, correlationId, shortName, graphId from CUPTI_ACTIVITY_KIND_KERNEL"):
        kern[corr].append((start, end, names.get(sn, "?"), gid))
    units = []
    for corr, ks in kern.items():
        a = api.get(corr, ("?", 0, 0))
        units.append(dict(corr=corr, api=a[0], hs=a[1], he=a[2], gs=min(k[0] for k in ks), ge=max(k[1] for k in ks), n=len(ks),
                          ksum=sum(k[1] - k[0] for k in ks), gid=ks[0][3], names=[k[2] for k in sorted(ks)]))
    units.sort(key=lambda u: u["gs"])
    with open(cache, "wb") as f:
        pickle.dump(units, f)
    return units


def cmd_units(db):
    units = load_units(db)
    print("units", len(units), "kernels", sum(u["n"] for u in units))
    by = collections.defaultdict(list)
    for u in units:
        graph = "Graph" in u["api"]
        by[(u["api"], u["gid"] if graph else None, u["n"] if graph else None)].append(u)
    rows = sorted(((sum(u["ksum"] for u in us), key, len(us), statistics.median([u["ge"] - u["gs"] for u in us]) / 1e3,
                    statistics.median([u["ksum"] for u in us]) / 1e3) for key, us in by.items()), reverse=True)
    print(f"{'api':24} {'graph':>6} {'kern':>5} {'launches':>8} {'span us':>9} {'ksum us':>9} {'total ms':>9}")
    for tot, key, cnt, span, ks in rows[:40]:
        print(f"{key[0][:24]:24} {str(key[1]):>6} {str(key[2]):>5} {cnt:8d} {span:9.1f} {ks:9.1f} {tot / 1e6:9.1f}")


def cmd_graph(db, ids, grid=False, top=30):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    for gid in ids:
        launches, agg = set(), collections.defaultdict(lambda: [0, 0])
        for corr, sn, s, e, gx, gy, gz, bx in c.execute(
                "select correlationId, shortName, start, end, gridX, gridY, gridZ, blockX from CUPTI_ACTIVITY_KIND_KERNEL where graphId=?", (gid,)):
            launches.add(corr)
            key = names.get(sn, "?") + (f" g{gx}x{gy}x{gz} b{bx}" if grid else "")
            agg[key][0] += 1
            agg[key][1] += e - s
        n = len(launches)
        if n == 0:
            print(f"== graph {gid}: no launches")
            continue
        print(f"== graph {gid}: {n} launches, {sum(v[0] for v in agg.values()) / n:.0f} kernels, "
              f"{sum(v[1] for v in agg.values()) / n / 1e3:.1f} us per launch")
        for k, (cnt, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
            print(f"  {t / n / 1e3:9.1f} us {cnt / n:6.1f}x  {t / cnt / 1e3:7.1f} us/each  {k[:110]}")


def cmd_timeline(db, a, b):
    units = load_units(db)
    t0, prev, run = units[a]["gs"], None, None

    def flush(r):
        if r:
            print(f"      ... {r[0]} kernels launched one by one, gpu from {r[1] / 1e3:9.1f} us, sum {r[2] / 1e3:7.1f} us  [{', '.join(sorted(set(r[3])))[:150]}]")
    for i in range(a, min(b, len(units))):
        u = units[i]
        if "Graph" not in u["api"]:
            run = run or [0, u["gs"] - t0, 0, []]
            run[0] += 1
            run[2] += u["ksum"]
            run[3] += u["names"]
            continue
        flush(run)
        run = None
        gap = (u["gs"] - prev) / 1e3 if prev else 0
        print(f"{i:6d} host {(u['hs'] - t0) / 1e6:9.3f} ms  gpu {(u['gs'] - t0) / 1e6:9.3f}..{(u['ge'] - t0) / 1e6:9.3f} ms  "
              f"gap-before {gap:8.1f} us  graph {u['gid']} n={u['n']} ksum {u['ksum'] / 1e6:7.3f} ms")
        prev = u["ge"]
    flush(run)


def cmd_phases(db, nsteps=80, first=1):
    units = load_units(db)
    ks = sorted(((x["gs"], x["ge"], x["names"][0]) for x in units), key=lambda k: k[0])
    steps, cur, phase = [], None, None
    for s, e, n in ks:
        if phase is None:
            if n.startswith("ts_q35_df_select"):
                phase, cur = "verify", collections.defaultdict(list)
            continue
        if phase == "verify" and cur["verify"] and cur["verify"][-1][2].startswith("ts_q35_argmax"):
            phase = "commit"
        if phase == "commit" and n.startswith("ts_q35_rows_by_index"):
            phase = "catchup"
        if phase in ("commit", "catchup") and n.startswith("ts_q35_embed_q4_0"):
            steps.append(cur)
            cur, phase = collections.defaultdict(list), "draft"
        cur[phase].append((s, e, n))
        if phase == "draft" and n.startswith("ts_q35_df_select"):
            phase = "verify"
    steps = [st for st in steps if st["draft"]][first:first + nsteps]
    print(f"{len(steps)} steps")
    period = [steps[i + 1]["draft"][0][0] - steps[i]["draft"][0][0] for i in range(len(steps) - 1)]
    print(f"step period median {statistics.median(period) / 1e6:.2f} ms")
    for ph in ["draft", "verify", "commit", "catchup"]:
        have = [st for st in steps if st[ph]]
        if not have:
            continue
        spans = [(st[ph][-1][1] - st[ph][0][0]) / 1e6 for st in have]
        sums = [sum(e - s for s, e, _ in st[ph]) / 1e6 for st in have]
        print(f"== {ph}: span {statistics.median(spans):.3f} ms, kernel sum {statistics.median(sums):.3f} ms, "
              f"{statistics.median([len(st[ph]) for st in have]):.0f} kernels")
        agg = collections.defaultdict(lambda: [0, 0])
        for st in steps:
            for s, e, n in st[ph]:
                agg[n][0] += 1
                agg[n][1] += e - s
        for n, (cnt, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:16]:
            print(f"  {t / len(steps) / 1e3:9.1f} us {cnt / len(steps):6.1f}x  {t / cnt / 1e3:7.1f} us/each  {n[:90]}")
    for a, b in [("draft", "verify"), ("verify", "commit"), ("commit", "catchup")]:
        g = [(st[b][0][0] - st[a][-1][1]) / 1e3 for st in steps if st[a] and st[b]]
        if g:
            print(f"gap {a}->{b} median {statistics.median(g):.1f} us")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    cmd, db, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if cmd == "units":
        cmd_units(db)
    elif cmd == "graph":
        cmd_graph(db, [int(g) for g in rest[0].split(",")], grid=len(rest) > 1 and rest[1] == "grid",
                  top=int(rest[2]) if len(rest) > 2 else 30)
    elif cmd == "timeline":
        cmd_timeline(db, int(rest[0]), int(rest[1]))
    elif cmd == "phases":
        cmd_phases(db, int(rest[0]) if rest else 80, int(rest[1]) if len(rest) > 1 else 1)
    else:
        print(__doc__)
        sys.exit(2)
