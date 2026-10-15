#!/usr/bin/env python3
"""Write independent fixed-task sidecars without changing original Maya reports.

This rechecks recorded production-parser content; it neither reruns the model
nor decodes tokens again. Original conclusions and complete source-file pins
remain separate from the new versioned answer oracle.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('maya_rescore_report', HERE / 'default-readiness-report.py')
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def audit_report(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('plan'), dict) or 'rows' not in data:
        raise ValueError('Require an original Maya report with plan and rows')
    summary = reporter.maya_summary(data)
    tools = [Path(__file__).resolve(), Path(reporter.__file__).resolve(),
             Path(reporter.fixed_tasks.__file__).resolve()]
    pins = {str(tool): sha_bytes(tool.read_bytes()) for tool in tools}
    original = dict(harness_sha256=data.get('harness_sha256'),
        policy=summary['recorded_runner_policy'], quality_passed=data.get('quality_passed'),
        completed=data.get('completed'), process_success=data.get('process_success'),
        rows=[dict(quality=row.get('quality'), semantic=row.get('semantic')) for row in data['rows']])
    if raw != path.read_bytes():
        raise ValueError('Original report changed while rescoring')
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        source_report=dict(path=str(path), sha256=sha_bytes(raw), bytes=len(raw)),
        checker_pins=pins, original=original, independent=summary,
        scope='Offline recheck of recorded production-parser content with fixed known questions/answers; no model rerun or token re-decode, no original report mutation.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = reporter.bench.evidence_directory(args.output)
    names = [path.parent.name + '-' + path.stem + '.rescore.json' for path in args.reports]
    if len(set(names)) != len(names):
        parser.error('Report names collide; select distinct source directories')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for path, name in zip(args.reports, names):
        record = audit_report(path)
        (output / name).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        records.append(record)
    rows = [dict(scenario=Path(r['source_report']['path']).parent.name,
                 status='offline recheck complete',
                 quality=f"Original={r['original']['quality_passed']}; independent={r['independent']['independent_quality_passed']}",
                 performance='No model or performance rerun', resources='No model allocation',
                 scope=f"Original policy={r['original']['policy']}; independent={r['independent']['independent_policy']}; original report preserved") for r in records]
    (output / 'summary.json').write_text(json.dumps(dict(reports=records, display_rows=rows), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
