#!/usr/bin/env python3
"""Owned, bounded llama.cpp image/history control. Never builds or patches llama.

The second request replays the ORIGINAL TensorSharp history, not this baseline's
new first answer. At twenty nonempty content-token records the client disconnects
and records a partial watchdog observation, not a complete quality pass.
"""
import argparse
import base64
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import tempfile
import time
import psutil

HERE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


bench = module('image_baseline_bench', 'multimodel-quality-bench.py')
owned = module('image_baseline_cleanup', 'llama-planned-chat.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def meaningful(text):
    return bool(re.search(r'[\w\u4e00-\u9fff]', text, re.UNICODE))


def frozen_requests(prefix, stdout, image):
    reference = json.loads(prefix.read_text(encoding='utf-8-sig'))
    log = stdout.read_text(encoding='utf-8')
    image_hash = sha(image)
    if image_hash != '10145e33904f59e8f4e61cdd035cebbf45d7e5591d8ded6cdedd5aa82e022b18':
        raise ValueError('Original image identity changed')
    selected = [r for r in reference['runs'] if r['scenario'] == 'image' and r['turn'] in ('turn-1','turn-2')]
    if len(selected) != 2:
        raise ValueError('Exactly two original image requests required')
    traces = re.findall(r'generation.validation.input internal=([^ ]+)[^\r\n]*?tokens=([\d,]+)', log)
    result = []
    data = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()
    for index, original in enumerate(selected):
        messages = []
        for msg in original['request']['messages']:
            converted = {'role': msg['role'], 'content': msg['content']}
            if msg.get('imagePaths'):
                if msg['imagePaths'] != ['sha256:' + image_hash]:
                    raise ValueError('Unexpected media in original history')
                # TensorSharp's imagePaths renderer prepends media to Content;
                # the original 317/353 IDs have vision_start at index 3.
                converted['content'] = [{'type': 'image_url', 'image_url': {'url': data}},
                                        {'type': 'text', 'text': msg['content']}]
            messages.append(converted)
        expected = 317 if index == 0 else 353
        candidates = [(rid, list(map(int, ids.split(',')))) for rid, ids in traces
                      if len(ids.split(',')) == expected]
        if len(candidates) != 1:
            raise ValueError('Ambiguous original input trace for ' + str(expected))
        result.append({'Turn': index + 1, 'OriginalRequestSha256': original['request_sha256'],
                       'OriginalRequest': original['request'], 'ExpectedRawInputIds': candidates[0][1],
                       'OriginalInternalRequestId': candidates[0][0],
                       'Body': {'messages': messages, 'stream': True, 'temperature': 0,
                                'top_k': 0, 'top_p': 1, 'min_p': 0, 'repeat_penalty': 1,
                                'presence_penalty': 0, 'frequency_penalty': 0, 'seed': 17,
                                'max_tokens': 2000, 'cache_prompt': False, 'logprobs': True,
                                'return_tokens': True, 'verbose': True,
                                'top_logprobs': 1, 'stream_options': {'include_usage': True},
                                'chat_template_kwargs': {'enable_thinking': False}}})
    return result


def json_request(port, path, body, remaining):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=max(.1, remaining()))
    try:
        connection.request('GET' if body is None else 'POST', path,
                           None if body is None else json.dumps(body, ensure_ascii=False).encode(),
                           {} if body is None else {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {raw[:1000]}')
        return json.loads(raw)
    finally:
        connection.close()


def stream_request(port, body, remaining, output, cap):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=max(.1, remaining()))
    started = time.monotonic()
    row = {'Content': '', 'ContentTokenRecords': 0, 'WatchdogPartial': False,
           'NaturalFinishReason': None, 'TtftMs': None, 'Usage': None, 'Timings': None,
           'RawOutputIdsFromProbabilityRecords': [], 'NativeVerbose': None}
    try:
        connection.request('POST', '/v1/chat/completions', json.dumps(body, ensure_ascii=False).encode(),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        row['HttpStatus'] = response.status
        if response.status != 200:
            raise RuntimeError(response.read().decode()[:1000])
        with output.open('x', encoding='utf-8') as raw:
            while True:
                if remaining() <= 0:
                    raise TimeoutError('Total owned lifetime exhausted')
                line = response.readline()
                if not line:
                    break
                raw.write(line.decode('utf-8', errors='replace')); raw.flush()
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if event.get('__verbose') is not None: row['NativeVerbose'] = event['__verbose']
                if event.get('usage') is not None: row['Usage'] = event['usage']
                if event.get('timings') is not None: row['Timings'] = event['timings']
                for choice in event.get('choices', []):
                    content = (choice.get('delta') or {}).get('content') or ''
                    if content:
                        if row['TtftMs'] is None: row['TtftMs'] = (time.monotonic()-started)*1000
                        row['Content'] += content
                    records = (choice.get('logprobs') or {}).get('content') or []
                    row['RawOutputIdsFromProbabilityRecords'].extend(x['id'] for x in records if type(x.get('id')) is int)
                    row['ContentTokenRecords'] += sum(bool(str(x.get('token', '')).strip()) for x in records)
                    if choice.get('finish_reason'): row['NaturalFinishReason'] = choice['finish_reason']
                if row['ContentTokenRecords'] >= cap:
                    row['WatchdogPartial'] = True
                    break
            if not row['WatchdogPartial'] and row['NaturalFinishReason'] is None:
                raise RuntimeError('Stream ended without a natural finish event or the reviewed content watchdog')
    finally:
        connection.close()
        row['WallMs'] = (time.monotonic()-started)*1000
        row['MeaningfulTextObserved'] = meaningful(row['Content'])
        row['EmptyOrPunctuationOnlyNaturalEnd'] = bool(row['NaturalFinishReason'] and not row['MeaningfulTextObserved'])
        row['ProtocolCompletionObserved'] = row['WatchdogPartial'] or row['NaturalFinishReason'] is not None
        row['CompleteQualityPassed'] = False
        row['QualityScope'] = 'Partial content watchdog observation; requires independent review. Natural empty/punctuation-only stop fails quality.'
        output.with_suffix('.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    return row


def prepare(args):
    requests = frozen_requests(args.prefix, args.stdout, args.image)
    command = [str(args.server.resolve()), '--model', str(args.model.resolve()), '--mmproj', str(args.mmproj.resolve()),
               '--host', '127.0.0.1', '--port', str(args.port), '--ctx-size', '32768', '--parallel', '1',
               '--n-gpu-layers', 'all', '--cpu-moe', '--threads', '8', '--threads-batch', '8',
               '--batch-size', '512', '--ubatch-size', '512', '--flash-attn', 'on',
               '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--jinja', '--no-warmup',
               '--log-prompts-dir', str(args.output / 'rendered-prompts')]
    return {'Schema': 'qwen-image-llama-control-plan.v1', 'Status': 'prepared-not-run', 'Command': command,
            'SourceRevision': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'SourceStatus': subprocess.check_output(['git', '-C', str(args.repo), 'status', '--short', '--untracked-files=no'], text=True).splitlines(),
            'ImageSha256': sha(args.image), 'OriginalPrefixSha256': sha(args.prefix),
            'OriginalTraceSha256': sha(args.stdout), 'Requests': requests,
            'TotalLifetimeSeconds': args.timeout, 'ContentTokenWatchdog': args.content_tokens,
            'ChildEnvironment': {'LLAMA_MEDIA_MARKER': '<__media__>'},
            'BinaryMetadataOnly': [{'Path': str(p), 'Bytes': p.stat().st_size, 'MtimeNs': p.stat().st_mtime_ns}
                                   for p in args.server.parent.glob('*') if p.suffix in ('.exe', '.dll')],
            'ModelMetadataOnly': [{'Path': str(p), 'Bytes': p.stat().st_size, 'MtimeNs': p.stat().st_mtime_ns}
                                  for p in [*bench.checkpoint_files(args.model), args.mmproj]],
            'Limitations': ['Same canonical messages/image; actual expanded IDs must be independently observed before exact-input claims.',
                            'llama mtmd merger uses tanh GELU, unlike independent HF reference erf merger.',
                            'No equal physical hard-limit or matched-throughput claim. Memory samples are observations.',
                            'Readiness/load failure means baseline generation not run, never TensorSharp quality pass.']}


def main():
    if sys.argv[1:] == ['--self-test']:
        assert not meaningful('') and not meaningful('\n\n…') and not meaningful(' .!? ')
        assert meaningful('4821') and meaningful('红色矩形') and meaningful('The picture')
        assert canonical({'b':2,'a':1}) == canonical({'a':1,'b':2})
        passed = 7
        original_connection = http.client.HTTPConnection
        def event(choice): return b'data: '+json.dumps({'choices':[choice]}).encode()+b'\n\n'
        class FakeConnection:
            closed = False
            def __init__(self,*args,**kwargs): self.status=200; self.raw=io.BytesIO(payload)
            def request(self,*args,**kwargs): pass
            def getresponse(self): return self
            def readline(self): return self.raw.readline()
            def close(self): FakeConnection.closed=True
        http.client.HTTPConnection = FakeConnection
        try:
            with tempfile.TemporaryDirectory(prefix='qwen-image-reader-') as tmp:
                for index,payload in enumerate((b'',b'data: [DONE]\n\n')):
                    try: stream_request(1,{},lambda:1,Path(tmp)/f'eof-{index}.sse',20)
                    except RuntimeError: pass
                    else: raise AssertionError('A clean EOF without finish must fail')
                    row=json.loads((Path(tmp)/f'eof-{index}.json').read_text())
                    assert not row['ProtocolCompletionObserved'] and FakeConnection.closed; passed+=1
                payload=event({'delta':{'content':'\n\n'},'finish_reason':'stop'})+b'data: [DONE]\n\n'
                row=stream_request(1,{},lambda:1,Path(tmp)/'empty-natural.sse',20)
                assert row['ProtocolCompletionObserved'] and row['EmptyOrPunctuationOnlyNaturalEnd'] and not row['CompleteQualityPassed'];passed+=1
                payload=event({'delta':{'content':'useful partial'},'logprobs':{'content':[{'id':1,'token':'a'}]*20},'finish_reason':None})
                row=stream_request(1,{},lambda:1,Path(tmp)/'watchdog.sse',20)
                assert row['ProtocolCompletionObserved'] and row['WatchdogPartial'] and row['NaturalFinishReason'] is None and not row['CompleteQualityPassed'];passed+=1
        finally:
            http.client.HTTPConnection = original_connection
        print(json.dumps({'Schema':'qwen-image-llama-control-selftest.v2','Passed':passed,'NativeModelCalls':0}))
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--server', type=Path, required=True); ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True); ap.add_argument('--mmproj', type=Path, required=True)
    ap.add_argument('--prefix', type=Path, required=True); ap.add_argument('--stdout', type=Path, required=True)
    ap.add_argument('--image', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--port', type=int, default=52371); ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--content-tokens', type=int, default=20); ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if args.timeout not in (180, 360) or args.content_tokens != 20:
        ap.error('Use a bounded 180 or 360 seconds /20 nonempty content-token records')
    out = bench.evidence_directory(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'rendered-prompts').mkdir()
    plan = prepare(args)
    save = lambda name, value: (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('plan.json', plan)
    if args.prepare_only:
        return 0
    report = {'Completed': False, 'Passed': False, 'GenerationRun': False, 'Plan': plan, 'Runs': [], 'TimedOut': False}
    server = sampler = timer = None
    try:
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            name = (process.info['name'] or '').lower()
            command = ' '.join(process.info['cmdline'] or []).lower()
            if (any(x in name for x in ('llama-server', 'tensorsharp.server.host', 'strata-generate', 'qwenimagelogitprobe'))
                    or name in ('dotnet.exe','dotnet') and any(x in command for x in
                        ('glmunifiedsessionprobe.dll','qwenunifiedprefillprobe.dll','qwenimagelogitprobe.dll','directcudamodelprobe.dll'))):
                raise RuntimeError('Existing model/Host left untouched: ' + str(process.info['pid']))
        if any(k.startswith(('TS_', 'TENSORSHARP_', 'LLAMA_ARG_', 'GGML_')) and v for k,v in os.environ.items()):
            raise RuntimeError('Inherited model override')
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', args.port)) == 0: raise RuntimeError('Port occupied')
        report['BinariesBefore'] = {str(p): sha(p) for p in args.server.parent.iterdir() if p.suffix in ('.dll','.exe')}
        started = time.monotonic(); deadline = started + args.timeout
        remaining = lambda: deadline-time.monotonic()
        with (out/'server.log').open('x', encoding='utf-8') as log:
            server = subprocess.Popen(plan['Command'], stdout=log, stderr=subprocess.STDOUT,
                                      cwd=str(args.server.parent), creationflags=bench.PROCESS_FLAGS,
                                      env={**os.environ, **plan['ChildEnvironment']})
            process = psutil.Process(server.pid)
            report['Owned'] = {'Pid': server.pid, 'CreateTime': process.create_time(), 'Executable': process.exe()}
            save('execution.json', report)
            def expired():
                report['TimedOut'] = True
                if server.poll() is None: server.kill()
            timer = threading.Timer(max(.1,remaining()), expired); timer.start()
            sampler = bench.Sampler(out/'memory.jsonl'); sampler.process = process; sampler.thread.start()
            while True:
                if server.poll() is not None: raise RuntimeError('Server exited during loading: ' + str(server.returncode))
                if remaining() <= 0: raise TimeoutError('Readiness deadline')
                try:
                    status = json_request(args.port,'/health',None,remaining)
                    if status.get('status') == 'ok': break
                except (OSError, RuntimeError): time.sleep(.3)
            report['ReadySeconds'] = time.monotonic()-started
            listeners = [x for x in process.net_connections(kind='tcp') if x.status == psutil.CONN_LISTEN and x.laddr.port == args.port]
            if not listeners: raise RuntimeError('Owned process does not own listener')
            report['OwnedListener'] = [str(x) for x in listeners]
            report['LoadedModulePaths'] = sorted({m.path for m in process.memory_maps() if m.path.lower().endswith('.dll')})
            for request in plan['Requests']:
                turn = request['Turn']; body = request['Body']; row = {'Turn': turn, 'OriginalRequestSha256': request['OriginalRequestSha256'], 'RequestSha256': canonical(body)}
                report['Runs'].append(row); save(f'request-{turn}.json', request)
                row['AppliedTemplate'] = json_request(args.port,'/apply-template',body,remaining)
                row['InputTokenCount'] = json_request(args.port,'/v1/chat/completions/input_tokens',body,remaining)
                prompt = row['AppliedTemplate'].get('prompt')
                if isinstance(prompt,str) and prompt.count('<__media__>') == 1:
                    parts = prompt.split('<__media__>')
                    text_ids = [json_request(args.port,'/tokenize',{'content':p,'add_special':index==0,'parse_special':True},remaining)['tokens']
                                for index,p in enumerate(parts)]
                    boundaries = [json_request(args.port,'/tokenize',{'content':p,'add_special':False,'parse_special':True},remaining)['tokens']
                                  for p in ('<|vision_start|>','<|vision_end|>')]
                    rows = row['InputTokenCount']['input_tokens']-sum(map(len,text_ids))-sum(map(len,boundaries))
                    row['InferredImageRowsFromMtmdCount'] = rows
                    row['TextSegmentInputIds'] = text_ids
                    if rows >= 0 and boundaries == [[248053],[248054]]:
                        reconstructed = text_ids[0]+boundaries[0]+[248056]*rows+boundaries[1]+text_ids[1]
                        row['ReconstructedExpandedInputIds'] = reconstructed
                        row['ReconstructedOriginalExpandedInputIdsEqual'] = reconstructed == request['ExpectedRawInputIds']
                    row['ImageRowInferenceScope'] = 'Actual MTMD placeholder count minus separately tokenized native text/boundaries; not a direct vision tensor export.'
                row['ActualExpandedInputIds'] = None
                row['ExactOriginalExpandedInputIdsVerified'] = False
                row['IdentityScope'] = 'Canonical messages/image equal. Native HTTP API exposes MTMD count/rendering, not all expanded image placeholder IDs.'
                report['GenerationRun'] = True
                row['Generation'] = stream_request(args.port,body,remaining,out/f'turn-{turn}.sse',args.content_tokens)
                save('execution.json', report)
                if row['Generation']['EmptyOrPunctuationOnlyNaturalEnd']:
                    row['QualityFailed'] = True
            report['Completed'] = True
            report['Passed'] = False
            report['QualityStatus'] = 'needs-independent-review' if all(x['Generation']['MeaningfulTextObserved'] for x in report['Runs']) else 'failed-empty-or-punctuation-continuation'
    except Exception as error:
        report['Error'] = repr(error)
    finally:
        report['Cleanup'] = owned.cleanup_owned(server,timer,sampler)
        report['BinariesAfter'] = {str(p): sha(p) for p in args.server.parent.iterdir() if p.suffix in ('.dll','.exe')}
        report['BinariesUnchanged'] = report.get('BinariesBefore') == report['BinariesAfter']
        report['ModelMetadataAfter'] = [{'Path': p['Path'], 'Bytes': Path(p['Path']).stat().st_size,
                                        'MtimeNs': Path(p['Path']).stat().st_mtime_ns}
                                       for p in plan['ModelMetadataOnly']]
        report['ModelMetadataUnchanged'] = plan['ModelMetadataOnly'] == report['ModelMetadataAfter']
        report['ImageSha256After'] = sha(args.image)
        report['ImageUnchanged'] = report['ImageSha256After'] == plan['ImageSha256']
        if sampler is not None: report['SampledMemory'] = bench.sampled_memory_summary(sampler.rows,sampler.errors)
        report['ImageAndModelIdentityScope'] = 'Original small image SHA; model path/size/mtime metadata, not a new 74.5GB hash.'
        log_path = out/'server.log'
        report['ActualImageRows'] = None
        report['MediaLogObservations'] = ([line for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines()
                                            if re.search(r'image|vision|encode|mtmd',line,re.I)] if log_path.exists() else [])
        report['ActualImageRowsScope'] = 'Raw native media log observations are preserved. No tensor row count is invented when the native log omits it.'
        save('execution.json',report)
    return 0 if (report['Completed'] and not report['TimedOut'] and report['Cleanup']['owned_server_exited']
                 and not report['Cleanup']['cleanup_errors'] and report['BinariesUnchanged']
                 and report['ModelMetadataUnchanged'] and report['ImageUnchanged']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
