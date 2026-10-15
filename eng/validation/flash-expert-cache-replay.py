#!/usr/bin/env python3
"""Replay real routed expert IDs at fixed per-layer capacities.

Capture with TS_HOST_MOE_ROUTE_TRACE=1 in a diagnostic probe process, never a
throughput run. LRU must reproduce native counters before interpreting another
policy. Replay estimates misses only: it cannot establish speed or quality.
"""
import argparse
import json
from pathlib import Path
import re


CREATE = re.compile(r'^\[HOSTMOE-ROUTE-CREATE\] layer=(\d+) slots=(\d+) experts=(\d+)$')
ROUTE = re.compile(r'^\[HOSTMOE-ROUTE\] layer=(\d+) ids=(\d+(?:,\d+)*)$')


class Cache:
    def __init__(self, capacity, experts, decay):
        if not 0 < capacity <= experts:
            raise ValueError('Invalid expert capacity')
        self.slots = [-1] * capacity
        self.age = [0] * capacity
        self.counts = [0] * experts
        self.decay = decay
        self.clock = self.rows = 0

    def access(self, ids):
        if any(i < 0 or i >= len(self.counts) for i in ids) or len(set(ids)) > len(self.slots):
            raise ValueError('Invalid selected expert IDs')
        self.rows += 1
        if self.decay and self.rows % self.decay == 0:
            self.counts = [x >> 1 for x in self.counts]
        for i in ids:
            self.counts[i] = min(65535, self.counts[i] + 1)
        remapped = [self.slots.index(i) if i in self.slots else -1 for i in ids]
        protected = {s for s in remapped if s >= 0}
        hits = len(protected) if len(ids) == len(set(ids)) else sum(s >= 0 for s in remapped)
        misses = 0
        for k, expert in enumerate(ids):
            if remapped[k] >= 0:
                continue
            if expert in ids[:k]:
                remapped[k] = remapped[ids.index(expert)]
                hits += 1
                continue
            def score(s):
                if not self.decay:
                    return (self.age[s], s)
                return (self.counts[self.slots[s]] if self.slots[s] >= 0 else -1, self.age[s], s)
            slot = min((s for s in range(len(self.slots)) if s not in protected), key=score)
            protected.add(slot)
            remapped[k] = slot
            self.slots[slot] = expert
            misses += 1
        for slot in remapped:
            self.clock += 1
            self.age[slot] = self.clock
        return hits, misses


def replay(lines, decay):
    caches = {}
    totals = dict(calls=0, hits=0, misses=0, creations=0)
    rows = []
    for line in lines:
        if match := CREATE.fullmatch(line):
            layer, slots, experts = map(int, match.groups())
            caches[layer] = Cache(slots, experts, decay)
            totals['creations'] += 1
        elif match := ROUTE.fullmatch(line):
            layer = int(match[1])
            if layer not in caches:
                raise ValueError('Route without a cache creation record')
            hits, misses = caches[layer].access([int(x) for x in match[2].split(',')])
            totals['calls'] += 1
            totals['hits'] += hits
            totals['misses'] += misses
            rows.append((hits, misses))
        elif line.startswith('[HOSTMOE-ROUTE'):
            raise ValueError('Malformed or truncated route trace')
    if not rows:
        raise ValueError('No route records')
    totals['hit_rate'] = totals['hits'] / (totals['hits'] + totals['misses'])
    return totals, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not any(output.is_relative_to(root / p) for p in ('artifacts', 'docs/validation')):
        parser.error('Output must stay in an ignored validation directory')
    probe = json.loads(args.probe.read_text(encoding='utf-8-sig'))
    if not probe.get('passed') or not probe.get('run_complete'):
        raise ValueError('Incomplete model run')
    lines = args.log.read_text(encoding='utf-8', errors='strict').splitlines()
    baseline, rows = replay(lines, 0)
    for key in ('Calls', 'Hits', 'Misses'):
        expected = probe['cache_stats'][key]
        if baseline[key.lower()] != expected:
            raise ValueError(f'LRU replay differs from native {key}: {baseline[key.lower()]} vs {expected}')
    results = {}
    for period in (0, 32, 64, 128, 256, 512, 1024):
        total, records = (baseline, rows) if period == 0 else replay(lines, period)
        slices = []
        before = probe['cache_stats_before_timed_work']['Calls']
        for run in probe['runs']:
            after_prefill = run['cache_stats_after_prefill']['Calls']
            after_decode = run['cache_stats_after_decode']['Calls']
            for phase, start, end in (('prefill', before, after_prefill), ('decode', after_prefill, after_decode)):
                if end > start:
                    hits = sum(r[0] for r in records[start:end])
                    misses = sum(r[1] for r in records[start:end])
                    slices.append(dict(first=run['warmup'], iteration=run['iteration'], phase=phase,
                        calls=end-start, hits=hits, misses=misses, hit_rate=hits/(hits+misses)))
            before = after_decode
        results['lru' if period == 0 else f'lfu_decay_{period}'] = dict(total=total, phases=slices)
    report = dict(native_lru_reproduced=True, log=str(args.log.resolve()), probe=str(args.probe.resolve()),
                  identity={k:probe[k] for k in ('native_sha256','checkpoint_identity','managed_assemblies_sha256')},
                  policies=results, limitation='Fixed recorded routing and slot capacities; no throughput, RAM/VRAM or quality claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v['total'] for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
