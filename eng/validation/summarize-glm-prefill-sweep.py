#!/usr/bin/env python3
"""Summarize a completed paired GLM pilot without running or hashing a model."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    execution = read(args.matrix)
    if execution['Status'] != 'complete' or not execution['Cases'] \
            or not all(r.get('Passed') is True for r in execution['Cases']):
        raise ValueError('Only a complete independently qualified matrix can be summarized.')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in execution['Cases']:
        folder = Path(row['OutputDirectory'])
        if row['Engine'] == 'tensorsharp':
            plan = read(row['Plan']['Path'])
            wrapper = read(Path(plan['WrapperOutput']) / 'execution.json')
            if not wrapper['telemetry']['passed']:
                raise ValueError('Missing complete runtime identity/resource telemetry.')
            memory = wrapper['telemetry']['memory']
            perf = row['Performance']
            metrics = dict(PrefillMs=perf['PrefillAttributedMs'], DecodeMs=perf['DecodeAttributedMs'],
                TtftMs=perf['FirstPublishedTokenWallMs'], RequestMs=perf['RequestWallMs'])
        else:
            samples = [json.loads(line) for line in (folder/'maya-run/telemetry.jsonl').read_text().splitlines()]
            if not samples:
                raise ValueError('No baseline telemetry.')
            memory = dict(samples=len(samples), peak_rss_bytes=max(r['memory']['rss'] for r in samples),
                peak_private_bytes=max(r['memory']['private'] for r in samples),
                peak_board_mib=max(g['used_mib'] for r in samples for g in r['gpu']))
            perf = row['Performance']
            metrics = dict(PrefillMs=perf['PrefillDoneMs'], DecodeMs=perf['DecodeDoneMs'],
                TtftMs=perf['FirstTokenWallMs'], RequestMs=perf['RequestWallMs'])
        cases.append(dict(Task=row['Task'], PrefillRows=row['PrefillRows'], Engine=row['Engine'],
            Label=row['Label'], PromptTokens=len(row['RawPromptTokens']), OutputTokens=len(row['RawOutputTokens']),
            OutputForwards=perf['OutputForwardCount'], NaturalEos=row['NaturalEos'],
            Metrics=metrics, ObservedMemory=memory, OutputDirectory=str(folder)))
    result = dict(Scope=execution['Scope'], PerformanceScope=execution['PerfScope'],
        MatrixSha256=hashlib.sha256(args.matrix.read_bytes()).hexdigest(), Cases=cases,
        PairComparability=execution['PairComparability'],
        Limits=['A single pilot observation per task/shape/engine is not a statistical conclusion.',
            'TS forwards P+O including EOS; Maya P+O-1. Different raw output IDs/counts forbid equal-work decode/whole-request ratios.',
            'Prefill attribution differs: TS forward intervals vs Maya engine DONE; TTFT starts at each client request.',
            'Sampled memory covers the whole process lifetime and can miss peaks; board VRAM includes other apps.',
            'Payload ceilings differ from full physical ownership/hard limits; this is not stock Host or N3/32K qualification.'])
    (args.output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    header = '<tr><th>Task</th><th>PP cap</th><th>Engine</th><th>PP s</th><th>TTFT s</th><th>Decode s</th><th>Request s</th><th>P/O/forward</th><th>RSS GiB</th><th>Private GiB</th><th>Board MiB</th></tr>'
    rows = []
    for row in sorted(cases, key=lambda r:(r['Task'], r['PrefillRows'], r['Engine'], r['Label'])):
        m, p = row['ObservedMemory'], row['Metrics']
        cells = [row['Task'], str(row['PrefillRows']), row['Engine']+' '+row['Label']]
        cells += [f'{p[k]/1000:.3f}' for k in ['PrefillMs','TtftMs','DecodeMs','RequestMs']]
        cells += [f"{row['PromptTokens']}/{row['OutputTokens']}/{row['OutputForwards']}",
            f"{m['peak_rss_bytes']/2**30:.3f}", f"{m['peak_private_bytes']/2**30:.3f}", str(m['peak_board_mib'])]
        rows.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
    page = '<!doctype html><meta charset="utf-8"><title>GLM prefill pilot</title><style>body{font:16px system-ui;margin:2em;max-width:1500px}table{border-collapse:collapse}td,th{padding:.55em;border:1px solid #ccc;text-align:right}th{background:#eee}li{margin:.5em}</style><h1>GLM 5.3 Flash: paired prefill pilot</h1><p>All listed tasks passed independent semantics and natural EOS. Performance target is not certified.</p>'
    page += '<p>'+html.escape(result['Scope'])+'</p><table>'+header+''.join(rows)+'</table><ul>'
    page += ''.join('<li>'+html.escape(line)+'</li>' for line in result['Limits'])+'</ul>'
    (args.output/'report.html').write_text(page, 'utf-8')
    print(json.dumps(dict(Completed=True, Cases=len(cases), Output=str(args.output))))


if __name__ == '__main__':
    main()
