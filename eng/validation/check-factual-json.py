#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
"""Grade finished factual JSON observations without treating another model as truth.

Never launches a model. The two FFVIII fixtures are validation inputs, not
production answer repair. A Maya observation needs its original --plan because
the observation records only that plan's SHA, not its source revision/path.
Maya parser/tokenizer code must still be at that clean revision. Only small
tokenizer/evidence files are reread; large pack/checkpoint bookends remain the
historical observation's explicitly identified evidence, not fresh hashes.
TensorSharp task=unknown/exit1 is intentionally not a semantic failure here.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>'
SUFFIX = '<|assistant|><think>'
FIXTURES = {
    'ff8-credits-en': 'Who directed the original 1999 PlayStation game Final Fantasy VIII, and who designed its characters? Reply only with JSON containing exactly the keys "director" and "character_designer". Use the full official English names.',
    'ff8-credits-evidence-en': 'Use only this supplied evidence for FINAL FANTASY VIII: Yoshinori Kitase (FINAL FANTASY VIII Director). CHARACTER DESIGN: TETSUYA NOMURA. Reply only with JSON containing exactly the keys "director" and "character_designer", using the full English names.',
}
EXPECTED = {'director': 'Yoshinori Kitase', 'character_designer': 'Tetsuya Nomura'}
SOURCES = ['https://na.finalfantasy.com/news/1195', 'https://eu.finalfantasy.com/copyrights']
SHA = re.compile(r'[0-9a-fA-F]{64}\Z')
MAX_JSON = 32 << 20


def unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError('Duplicate JSON member')
    return value


def strict_json(raw):
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON number')
        return result
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=bad, parse_float=floating)


def load(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_JSON:
        raise ValueError('JSON evidence outside bounded size')
    raw = path.read_bytes()
    if len(raw) > MAX_JSON:
        raise ValueError('JSON evidence grew beyond bound')
    return strict_json(raw.decode('utf-8-sig'))


def load_token_file(path):
    """Read pinned IDs, including the public probe's strict CSV/whitespace form."""
    path = Path(path)
    maximum = 4_194_304
    if not 0 < path.stat().st_size <= maximum:
        raise ValueError('Token evidence outside bounded size')
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > maximum:
        raise ValueError('Token evidence grew beyond bound')
    raw = raw_bytes.decode('utf-8-sig')
    if raw.lstrip(' \t\r\n').startswith('['):
        result = strict_json(raw)
    else:
        if not re.fullmatch(r'[ \t\r\n]*[0-9]+(?:(?:[ \t\r\n]*,[ \t\r\n]*|[ \t\r\n]+)[0-9]+)*[ \t\r\n]*', raw):
            raise ValueError('Malformed raw token separators')
        words = re.split(r'[, \t\r\n]+', raw.strip(' \t\r\n'))
        if any(len(word) > 10 or not re.fullmatch(r'(?:0|[1-9][0-9]*)', word) for word in words):
            raise ValueError('Invalid canonical nonnegative int32 token')
        result = [int(word) for word in words]
    if not ids(result):
        raise ValueError('Invalid nonnegative int32 token array')
    return result


def digest(path, maximum=MAX_JSON):
    path = Path(path)
    if not 0 < path.stat().st_size <= maximum:
        raise ValueError('Small identity evidence outside bounded size: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path):
    value = str(path).replace('\\', '/')
    return value.casefold() if re.match(r'^[A-Za-z]:/', value) else value


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2147483647


def ids(value):
    return isinstance(value, list) and bool(value) and len(value) <= 1048576 and all(integer(v) for v in value)


def identities(rows):
    if not isinstance(rows, list) or not rows:
        return False
    paths = set()
    for row in rows:
        if not isinstance(row, dict): return False
        path = row.get('Path', row.get('path'))
        size = row.get('Bytes', row.get('bytes'))
        sha = row.get('Sha256', row.get('sha256'))
        if not isinstance(path, str) or not path or canonical(path) in paths \
                or type(size) is not int or size <= 0 or not isinstance(sha, str) or not SHA.fullmatch(sha):
            return False
        paths.add(canonical(path))
    return True


def bookend(report, name):
    before, after = report.get(name + 'Before'), report.get(name + 'After')
    return identities(before) and before == after and report.get(name + 'Unchanged') is True


def thinking_closed(prompt, raw):
    if not isinstance(prompt, str) or not isinstance(raw, str): return False
    opened = prompt.endswith('<think>')
    for match in re.finditer(r'</?think>', raw):
        opened = match[0] == '<think>'
    return not opened


def semantic(content):
    try:
        return isinstance(content, str) and len(content) <= 65536 and strict_json(content) == EXPECTED
    except (ValueError, TypeError, RecursionError):
        return False


def content_diagnostics(content):
    """Report embedded values separately; wrappers never change semantic()."""
    value, observed = None, None
    if isinstance(content, str) and len(content) <= 65536:
        try:
            value = strict_json(content)
        except (ValueError, TypeError, RecursionError):
            pass
        candidate = value
        if candidate is None:
            # A bounded, duplicate-rejecting read of the first JSON object is
            # diagnostic only. Never feed this object back to semantic().
            start = content.find('{')
            if start >= 0:
                try:
                    candidate, _ = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(content[start:])
                except (ValueError, TypeError, RecursionError):
                    pass
        if isinstance(candidate, dict) and all(isinstance(candidate.get(k), str) for k in EXPECTED):
            observed = {key: candidate[key] for key in EXPECTED}
    strict = isinstance(value, dict) and set(value) == set(EXPECTED) and all(isinstance(v, str) for v in value.values())
    return {'StrictFormatPassed': strict, 'FactualValuesObserved': observed,
        'FactualValuesMatch': observed == EXPECTED if observed is not None else None,
        'FactualDiagnosticScope': 'Values in the first duplicate-rejecting JSON object, possibly embedded in wrappers/prose. Observation only; no wrapper removal, answer repair, or contribution to QualityPassed.'}


def natural_eos(output, eos):
    return ids(output) and ids(eos) and len(set(eos)) == len(eos) \
        and output[-1] in eos and not any(t in eos for t in output[:-1])


def ts_round(row, report, rendered):
    prompt, output, published = report.get('PromptTokens'), row.get('OutputTokens'), row.get('PublishedTokens')
    done = row.get('Completion', {})
    raw, content = row.get('RawText'), row.get('Content')
    valid_ids = ids(prompt) and ids(output)
    counts = valid_ids and isinstance(done, dict) and done.get('PromptTokenCount') == len(prompt) \
        and done.get('OutputTokenCount') == len(output) and type(done.get('PromptTokenCount')) is int \
        and type(done.get('OutputTokenCount')) is int and published == output[:-1] \
        and row.get('ComputedTokens') == len(prompt) + len(output) and type(row.get('ComputedTokens')) is int
    eos = natural_eos(output, report.get('CheckpointEosIds'))
    checks = {
        'Counts': bool(counts),
        'NaturalEos': bool(eos and done.get('Status') == 'FinishedStopped' and done.get('FinishReason') == 'eos'
            and row.get('NaturalEos') is True and row.get('StatusBeforeDispose') == 'FinishedStopped'),
        'NoReuse': done.get('PrefixCacheReusedTokens') == 0 and type(done.get('PrefixCacheReusedTokens')) is int,
        'ThinkingClosed': row.get('ThinkingClosed') is True and thinking_closed(rendered, raw),
        'NoTools': row.get('ToolCalls') in (None, []) and isinstance(raw, str) and '<tool_call>' not in raw,
        'ProductionContentBound': isinstance(content, str) and isinstance(raw, str) and content in raw,
        'FiniteIfObserved': row.get('FinalLogitsFinite') is not False,
    }
    return {'ExecutionChecks': checks, 'ExecutionPassed': all(checks.values()),
            'SemanticPassed': semantic(content), 'Content': content} | content_diagnostics(content)


def model_link(report):
    """Recompute the captured historical manifest/stat link without model I/O."""
    manifest = report.get('HistoricalModelManifest', {})
    files = manifest.get('files') if isinstance(manifest, dict) else None
    stats = report.get('ModelStatsBefore')
    if not isinstance(files, list) or not files or not isinstance(stats, dict) or not stats:
        return False
    entrypoint = report.get('Arguments', {}).get('model')
    if not isinstance(entrypoint, str) or not entrypoint:
        return False
    entrypoint = canonical(entrypoint)
    match = re.fullmatch(r'(.*)-([0-9]{5})-of-([0-9]{5})\.gguf', entrypoint, re.IGNORECASE)
    if match:
        count = int(match[3])
        if match[2] != '00001' or not 1 <= count <= 4096:
            return False
        required = {f'{match[1]}-{i:05d}-of-{count:05d}.gguf' for i in range(1, count + 1)}
    else:
        required = {entrypoint}
    expected = {}
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get('path'), str) \
                or type(row.get('bytes')) is not int or row['bytes'] <= 0 \
                or type(row.get('mtime_ns')) is not int or not isinstance(row.get('sha256'), str) \
                or not SHA.fullmatch(row['sha256']) or canonical(row['path']) in expected:
            return False
        expected[canonical(row['path'])] = row
    actual = set()
    for path, row in stats.items():
        if not isinstance(path, str) or not isinstance(row, dict) or canonical(path) not in expected \
                or canonical(row.get('Path', '')) != canonical(path) or canonical(path) in actual:
            return False
        historical = expected[canonical(path)]
        ticks = row.get('LastWriteTimeUtcTicks')
        if type(row.get('Bytes')) is not int or row['Bytes'] != historical['bytes'] \
                or type(ticks) is not int or (ticks - 621355968000000000) * 100 != historical['mtime_ns']:
            return False
        actual.add(canonical(path))
    # A full historical identity bundle may also cover an unused mmproj.
    # Stats must cover every and only the loaded entrypoint's shards, while
    # each loaded shard still matches the full historical file identity.
    return actual == required and required <= set(expected)


def inspect_tensorsharp(report, fixture):
    rendered = PREFIX + FIXTURES[fixture] + SUFFIX
    arguments = report.get('Arguments', {})
    rows = report.get('Rounds')
    valid_rounds = isinstance(rows, list) and 1 <= len(rows) <= 8 and all(isinstance(r, dict) for r in rows)
    requested = report.get('RequestedRepetitions')
    try:
        max_new = int(arguments.get('max-new', ''))
    except (ValueError, TypeError): max_new = 0
    source = report.get('SourceToolsBefore', [])
    source_names = {canonical(r.get('Path', '')).rsplit('/', 1)[-1] for r in source if isinstance(r, dict)} if isinstance(source, list) else set()
    managed = report.get('ManagedBefore', [])
    managed_names = {canonical(r.get('Path', '')).rsplit('/', 1)[-1].casefold() for r in managed if isinstance(r, dict)} if isinstance(managed, list) else set()
    native, native_after = report.get('IntendedNative'), report.get('IntendedNativeAfter')
    counters = report.get('EngineCountersBeforeDispose', {})
    ledger = report.get('BudgetAfterSessionDisposed', {}).get('Current', [])
    checks = {
        'NoErrors': report.get('Errors') == [] and report.get('SequenceError') is None,
        'Completed': report.get('EngineCompletionObserved') is True and report.get('Status') == 'completed-quality-unknown',
        'UnknownTaskAllowed': arguments.get('task') == 'unknown',
        'UnforcedPlainGeneration': arguments.get('capture-token-id') is None and report.get('DiagnosticForcedCaptureToken') is None
            and report.get('Scheduler', {}).get('Enabled') is False and report.get('Scheduler', {}).get('EnablePrefixCaching') is False,
        'FixtureBound': report.get('DecodedPrompt') == rendered and ids(report.get('PromptTokens')),
        'SourceIdentity': bookend(report, 'SourceTools') and {'program.cs', 'probepolicy.cs', 'glmunifiedsessionprobe.csproj'} <= {n.casefold() for n in source_names},
        'ManagedIdentity': bookend(report, 'Managed') and {'tensorsharp.core.dll', 'tensorsharp.memory.dll',
            'tensorsharp.models.dll', 'tensorsharp.runtime.dll', 'tensorsharp.backends.ggml.dll',
            'tensorsharp.backends.cuda.dll', 'glmunifiedsessionprobe.dll'} <= managed_names,
        'NativeIdentity': identities([native]) and native == native_after and report.get('NativeUnchanged') is True
            and report.get('LoadedNativeAfterLoad') == [native] and report.get('LoadedNativeAfter') == [native]
            and arguments.get('native-sha256', '').lower() == native.get('Sha256', '').lower(),
        'InputIdentity': identities([report.get('InputFile')]) and report.get('InputFile') == report.get('InputFileAfter')
            and report.get('InputFileUnchanged') is True,
        'ModelIdentity': report.get('ModelIdentityLinked') is True and report.get('ModelStatsUnchanged') is True
            and bool(report.get('ModelStatsBefore')) and report.get('ModelStatsBefore') == report.get('ModelStatsAfter')
            and identities([report.get('HistoricalModelIdentity')]) and report.get('HistoricalModelIdentity') == report.get('HistoricalModelIdentityAfter')
            and report.get('HistoricalModelIdentityUnchanged') is True and model_link(report),
        'Disposal': report.get('EngineDisposed') is True and report.get('SessionDisposed') is True,
        'LedgerZero': report.get('LedgerZeroAfterCleanup') is True and isinstance(ledger, list) and bool(ledger)
            and all(isinstance(p, dict) and type(p.get('Reserved')) is int and p['Reserved'] == 0
                and type(p.get('Committed')) is int and p['Committed'] == 0 for p in ledger),
        'NoPending': isinstance(counters, dict) and all(type(counters.get(k)) is int and counters[k] == 0
            for k in ('RunningCount', 'WaitingCount', 'PendingRequestCount')),
        'AllRequestedRounds': valid_rounds and type(requested) is int and requested == len(rows)
            and type(report.get('CompletedRepetitions')) is int and report['CompletedRepetitions'] == requested
            and type(counters.get('TotalSubmitted')) is int and counters['TotalSubmitted'] == requested
            and type(counters.get('TotalCompleted')) is int and counters['TotalCompleted'] == requested,
        'DistinctRequests': valid_rounds and all(type(r.get('Round')) is int and r['Round'] == i
            and isinstance(r.get('RequestId'), str) and bool(r['RequestId']) for i, r in enumerate(rows, 1))
            and len({r.get('RequestId') for r in rows}) == len(rows),
    }
    results = [ts_round(row, report, rendered) for row in rows] if valid_rounds else []
    checks['AllRoundExecution'] = bool(results) and all(r['ExecutionPassed'] for r in results)
    checks['OutputLimit'] = 1 <= max_new <= 2048 and bool(results) and all(ids(r.get('OutputTokens')) and len(r['OutputTokens']) <= max_new for r in rows)
    checks['CleanupTrajectory'] = valid_rounds and report.get('OutputTokensAfterCleanup') == rows[-1].get('OutputTokens') \
        and report.get('ComputedTokensAfterCleanup') == rows[-1].get('ComputedTokens') \
        and report.get('RawTextAfterCleanup') == rows[-1].get('RawText') \
        and report.get('NaturalEosAfterCleanup') is True and report.get('SequenceStatusAfterCleanup') == 'FinishedStopped' \
        and report.get('FinishReasonAfterCleanup') == 'eos'
    return finish('tensorsharp', fixture, checks, results)


def inspect_maya(report, fixture, tokenizer, parser_class, eos, identity_checks):
    rendered = PREFIX + FIXTURES[fixture] + SUFFIX
    prompt, output = report.get('Tokens'), report.get('RawOutputIds')
    done = report.get('Done', {})
    valid_ids = ids(prompt) and ids(output)
    decoded_prompt = tokenizer.decode(prompt) if ids(prompt) else None
    raw = tokenizer.decode(output) if ids(output) else None
    natural = natural_eos(output, eos) and done.get('finish') == 'stop'
    parser_tokens = output[:-1] if natural else output
    content, calls, closed = None, None, False
    if raw is not None:
        parser = parser_class(thinking=True, tools=None, stream_tools=False)
        # Maya serve/server.py retains the sampled stop ID in raw_ids, but
        # breaks before detok.push()/parser.feed(). Match that exact boundary.
        # Keep the original raw/full IDs for independent identity and counts.
        events = parser.feed(tokenizer.decode(parser_tokens)) + parser.finish()
        content = ''.join(e.text for e in events if e.kind == 'content')
        calls = [e for e in events if e.kind == 'tool_call']
        closed = thinking_closed(rendered, raw)
    times = report.get('RawTokenWallMs')
    max_new = report.get('MaxNewTokens')
    checks = dict(identity_checks) | {
        'Completed': report.get('ExecutionComplete') is True and report.get('ExitCode') == 0 and type(report.get('ExitCode')) is int,
        'Cleanup': report.get('Errors') == [] and report.get('CleanupErrors') == [] and report.get('TimedOut') is False,
        'FixtureBound': decoded_prompt == rendered and report.get('DecodedPrompt') == rendered and tokenizer.encode(rendered, parse_special=True) == prompt,
        'RawOutputBound': raw is not None and raw == report.get('RawText'),
        'Counts': valid_ids and type(done.get('generated')) is int and done['generated'] == len(output)
            and type(done.get('prompt_tokens')) is int and done['prompt_tokens'] == len(prompt),
        'NaturalEos': natural,
        'NoReuseOrDraft': all(type(done.get(k)) is int and done[k] == 0 for k in ('reused', 'drafts_offered', 'drafts_accepted')),
        'OutputLimit': integer(max_new, 1) and max_new <= 2048 and ids(output) and len(output) <= max_new,
        'WireTokens': isinstance(times, list) and ids(output) and len(times) == len(output)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in times) and times == sorted(times),
        'ThinkingClosed': closed,
        'NoTools': calls == [] and raw is not None and '<tool_call>' not in raw,
    }
    result = {'ExecutionPassed': all(checks.values()), 'SemanticPassed': semantic(content), 'Content': content,
        'ParserInputTokenCount': len(parser_tokens) if ids(output) else None,
        'TerminalEosExcludedFromParser': bool(natural),
        'ParserInputScope': 'Exclude exactly one verified terminal checkpoint EOS, as Maya production serve/server.py does before detokenization. RawText and full RawOutputIds are unchanged.'} | content_diagnostics(content)
    return finish('maya', fixture, checks, [result])


def finish(engine, fixture, checks, rows):
    execution = all(v is True for v in checks.values())
    semantics = all(r['SemanticPassed'] is True for r in rows) if rows else None
    return {'Schema': 'tensorsharp.factual-json-qualification.v1', 'Engine': engine, 'Fixture': fixture,
        'ExecutionPassed': execution, 'SemanticPassed': semantics, 'QualityPassed': execution and semantics is True,
        'StrictFormatPassed': all(r.get('StrictFormatPassed') is True for r in rows) if rows else None,
        'ExecutionChecks': checks, 'Rounds': rows, 'IndependentExpected': EXPECTED, 'Sources': SOURCES,
        'Scope': 'Exact two-role factual JSON only; evidence fixture measures supplied-fact extraction, not stored knowledge. No general factual, equal-resource performance or default-readiness claim.'}


def maya_identity(report, plan_path, source=None, pack=None):
    plan = load(plan_path)
    source = Path(source or plan['source_path']).resolve()
    pack = Path(pack or plan['pack']).resolve()
    checks = {
        'PlanIdentity': digest(plan_path) == report.get('PlanSha256'),
        'ObservationToolIdentity': digest(HERE/'maya-token-observation.py') == report.get('ToolSha256'),
        'SourcePath': canonical(source) == canonical(Path(plan['source_path']).resolve()),
        'PackPath': canonical(pack) == canonical(Path(plan['pack']).resolve()),
    }
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, timeout=20).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True, timeout=20).strip()
    checks['SourceIdentity'] = revision == plan.get('source_revision') and not dirty
    before, after = report.get('VerifiedPackBefore'), report.get('VerifiedPackAfter')
    checks['PackBookends'] = identities(before) and before == after
    command = report.get('Command', [])
    checks['CommandIdentity'] = isinstance(command, list) and len(command) == 8 \
        and canonical(command[0]) == canonical(plan.get('exe', '')) and command[1] == '--glm-pack' \
        and canonical(command[2]) == canonical(pack) and command[3] == '--max-context' \
        and command[4] == str(report.get('ContextTokens')) and command[5] == '--prefill' \
        and command[6] == str(report.get('PrefillRows')) and command[7] == '--serve'
    evidence = load(plan['pack_evidence'])
    checks['PackEvidenceIdentity'] = digest(plan['pack_evidence']) == plan.get('pack_evidence_sha256') \
        and evidence.get('exit_code') == 0 and evidence.get('original_checkpoint_stat_unchanged') is True \
        and evidence.get('source_unchanged') is True
    checks['HarnessIdentity'] = isinstance(plan.get('current_harness_pins'), dict) and bool(plan.get('current_harness_pins')) \
        and all(isinstance(pin, dict) and digest(pin['path']) == pin.get('sha256') for pin in plan.get('current_harness_pins', {}).values())
    names = ('chat_template.jinja', 'merges.txt', 'token_type.json', 'tokenizer.json', 'vocab.json')
    rows = {canonical(r.get('path', '')): r for r in before or [] if isinstance(r, dict)}
    checks['TokenizerIdentity'] = all(canonical(pack/'tokenizer'/name) in rows \
        and (pack/'tokenizer'/name).stat().st_size == rows[canonical(pack/'tokenizer'/name)]['bytes'] \
        and digest(pack/'tokenizer'/name, 8 << 20) == rows[canonical(pack/'tokenizer'/name)]['sha256'] for name in names)
    if not all(checks.values()):
        raise ValueError('Maya identity prerequisites failed: ' + ', '.join(k for k, v in checks.items() if not v))
    spec = importlib.util.spec_from_file_location('factual_json_maya_base', HERE/'maya-token-baseline.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tokenizer, parser = module.load_maya_text(source, pack)
    eos = [tokenizer.special_ids.get('tokenizer.ggml.' + key + '_token_id') for key in ('eos', 'eot', 'eom')]
    if not ids(plan['eos_ids']) or len(set(plan['eos_ids'])) != len(plan['eos_ids']) \
            or not all(integer(token) for token in eos) or set(eos) != set(plan['eos_ids']):
        raise ValueError('Checkpoint EOS identity differs')
    return tokenizer, parser, plan['eos_ids'], checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--engine', required=True, choices=('tensorsharp', 'maya'))
    parser.add_argument('--fixture', required=True, choices=tuple(FIXTURES))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', type=Path, help='Original Maya observation plan, required for Maya source identity.')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--pack', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    workspace = HERE.parents[1]
    if not any(output.is_relative_to(workspace/path) for path in ('artifacts', 'docs/validation')):
        parser.error('--output must be in ignored artifacts/ or docs/validation/.')
    if output.exists(): parser.error('--output must be a new file; preserve previous evidence.')
    if args.engine == 'maya' and args.plan is None: parser.error('Maya requires its original --plan.')
    if args.engine == 'tensorsharp' and any(v is not None for v in (args.plan, args.source, args.pack)):
        parser.error('--plan/--source/--pack apply only to Maya.')
    try:
        report = load(args.report)
        if not isinstance(report, dict): raise ValueError('Report must be a JSON object')
        if args.engine == 'tensorsharp':
            result = inspect_tensorsharp(report, args.fixture)
            captured = report.get('InputFile', {})
            historical = report.get('HistoricalModelIdentity', {})
            external = {
                'OriginalInputFile': digest(captured['Path']) == captured['Sha256'] and load_token_file(captured['Path']) == report.get('PromptTokens'),
                'OriginalModelManifest': digest(historical['Path']) == historical['Sha256'] and load(historical['Path']) == report.get('HistoricalModelManifest'),
            }
            result['ExecutionChecks'].update(external)
            result['ExecutionPassed'] = all(v is True for v in result['ExecutionChecks'].values())
            result['QualityPassed'] = result['ExecutionPassed'] and result['SemanticPassed'] is True
        else:
            result = inspect_maya(report, args.fixture, *maya_identity(report, args.plan, args.source, args.pack))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError, subprocess.SubprocessError, RecursionError) as error:
        result = finish(args.engine, args.fixture, {'EvidenceValid': False}, [])
        result['Errors'] = [str(error)]
    result['Report'] = {'Path': str(args.report.resolve()), 'Sha256': digest(args.report)}
    result['ToolSha256'] = digest(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('ExecutionPassed', 'SemanticPassed', 'QualityPassed')}))
    return 0 if result['QualityPassed'] else 1


if __name__ == '__main__':
    sys.exit(main())
