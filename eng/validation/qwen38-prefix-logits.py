#!/usr/bin/env python3
"""Capture two same-history Qwen predictions in three fresh, pinned processes.

TS consumes the original prompt once and one supplied scalar token. llama consumes
the same prompt and must independently select that scalar before its second row.
HTTP full pre-sampling log probabilities are compared after a common-token offset;
they are never called raw logits. Completion of this diagnostic is not EOS,
semantic quality, numeric equivalence, or quiet-throughput qualification.
"""
from __future__ import annotations
import argparse
import array
import copy
import heapq
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


raw = module('prefix_raw_helpers', 'llama-raw-completion.py')
capture = module('prefix_capture', 'qwen38_capture_evidence.py')
teacher = module('prefix_teacher', 'qwen38-llama-teacher.py')
compare = raw.semantic
bench = raw.bench
require = capture.require


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load(path):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), 'Duplicate JSON object key')
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def finite_values(values, size):
    require(len(values) == size and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            'Missing, boolean or nonfinite vocabulary values')


def row_summary(values, candidates):
    top = heapq.nlargest(min(32, len(values)), range(len(values)), key=values.__getitem__)
    return dict(argmax=top[0], top32=[dict(id=i, value=values[i]) for i in top],
                candidates={str(i): dict(value=values[i], rank=1 + sum(x > values[i] for x in values)) for i in candidates},
                candidate_margin=values[candidates[0]] - values[candidates[1]])


def relative_difference(left, right):
    require(len(left) == len(right) and len(left) > 0, 'Unequal vocabulary sizes')
    reference = max(range(len(right)), key=right.__getitem__)
    a = [x - left[reference] for x in left]
    b = [x - right[reference] for x in right]
    delta = [x-y for x,y in zip(a,b)]
    return dict(reference_token=reference,
        relative_l2_after_common_token_offset=math.sqrt(math.fsum(x*x for x in delta)) /
            max(math.sqrt(math.fsum(x*x for x in b)), 1e-150),
        max_abs_after_common_token_offset=max(abs(x) for x in delta),
        max_pairwise_margin_error=max(delta)-min(delta),
        argmax=[max(range(len(left)), key=left.__getitem__), reference])


def validate_plan(plan):
    diagnostic = plan['diagnostic']
    vocab, first = diagnostic['vocabulary'], diagnostic['first_token']
    require(type(vocab) is int and vocab > 1 and type(first) is int and 0 <= first < vocab,
            'Invalid diagnostic vocabulary or scalar token')
    require(type(diagnostic['context']) is int and diagnostic['context'] > 0, 'Invalid context')
    candidates = diagnostic['candidate_tokens']
    require(isinstance(candidates, list) and len(candidates) == 2 and candidates[0] != candidates[1]
            and all(type(x) is int and 0 <= x < vocab for x in candidates), 'Invalid candidate pair')
    llama = plan['llama']
    body = llama['futureRequest']['body']
    require(type(body.get('n_predict')) is int and body['n_predict'] == 2
            and type(body.get('n_probs')) is int and body['n_probs'] == vocab
            and body.get('post_sampling_probs') is False, 'Require exactly two complete pre-sampling distributions')
    # Reuse strict endpoint/sampling checks without changing the semantic runner.
    control = copy.deepcopy(llama)
    control['futureRequest']['body']['n_probs'] = 0
    control['futureRequest']['body'].pop('post_sampling_probs')
    endpoint = raw.validate_plan(control, control['futureRequest']['body'])
    prompt = body['prompt']
    require(all(x < vocab for x in prompt) and first not in llama['expected_checkpoint_eos_ids'],
            'Invalid prompt ID or EOS scalar')
    cmd = llama['futureCommand']
    require(int(cmd[cmd.index('--ctx-size') + 1]) == diagnostic['context'], 'llama context differs')
    require(diagnostic['expected_backend_sampling'] is False, 'Pre-sampling capture requires the baseline CPU sampler')
    baseline=diagnostic['baseline_llama_tokens']
    require(isinstance(baseline,list) and len(baseline)==2 and baseline[0]==first
            and all(type(x) is int and 0 <= x < vocab for x in baseline), 'Invalid baseline prediction IDs')
    require(plan['tensorsharp']['cache_budgets_mb'] == [0, 8192], 'Require exactly the reviewed cache0/cache8G arms')
    require(len(prompt) + 2 <= diagnostic['context'], 'Diagnostic exceeds context')
    return endpoint


def full_distributions(response, plan):
    spec = plan['diagnostic']; body = plan['llama']['futureRequest']['body']
    tokens = response.get('tokens')
    require(isinstance(tokens, list) and len(tokens) == 2
            and all(type(x) is int and 0 <= x < spec['vocabulary'] for x in tokens), 'Invalid prediction IDs')
    require(tokens[0] == spec['first_token'], 'First generated token differs; second-row history is not matched')
    require(type(response.get('tokens_evaluated')) is int and response['tokens_evaluated'] == len(body['prompt'])
            and type(response.get('tokens_predicted')) is int and response['tokens_predicted'] == 2,
            'Prompt or prediction count differs')
    require(response.get('stop_type') == 'limit' and response.get('truncated') is False
            and not any(t in plan['llama']['expected_checkpoint_eos_ids'] for t in tokens),
            'Expected a two-prediction diagnostic cap without EOS/context truncation')
    settings = response.get('generation_settings', {})
    require(settings.get('backend_sampling') is False and settings.get('post_sampling_probs') is False
            and type(settings.get('n_probs')) is int and settings['n_probs'] == spec['vocabulary'],
            'Actual distribution/sampling mode differs')
    for name in ('temperature','samplers','top_k','top_p','min_p','repeat_penalty','repeat_last_n',
                 'presence_penalty','frequency_penalty','dry_multiplier','ignore_eos','stop','n_predict','stream'):
        actual,expected=settings.get(name),body[name]
        require(actual==expected and (type(actual) is bool if type(expected) is bool else
            type(actual) in (int,float) if type(expected) in (int,float) else True),
            'Actual sampling setting differs: '+name)
    require(settings.get('grammar')=='' and settings.get('logit_bias')==[]
            and isinstance(settings.get('speculative.types'),str)
            and set(settings['speculative.types'].split(','))=={'none'}, 'Actual grammar/bias/speculation differs')
    rows = response.get('completion_probabilities')
    require(isinstance(rows, list) and len(rows) == 2, 'Missing complete prediction rows')
    values = []
    for index, row in enumerate(rows):
        require(type(row.get('id')) is int and row['id'] == tokens[index], 'Probability row token differs')
        probabilities = row.get('top_logprobs')
        require(isinstance(probabilities, list) and len(probabilities) == spec['vocabulary'], 'Incomplete vocabulary')
        current = [None] * spec['vocabulary']
        for item in probabilities:
            token, value = item.get('id'), item.get('logprob')
            require(type(token) is int and 0 <= token < len(current) and current[token] is None,
                    'Duplicate or invalid vocabulary ID')
            require(type(value) in (int, float) and math.isfinite(value) and -1e20 < value <= 0,
                    'Nonfinite, clipped, boolean or positive log probability')
            current[token] = value
        require(current[tokens[index]] == max(current), 'Selected token is not pre-sampling greedy argmax')
        values.append(current)
    return values


def checkpoint_state(llama):
    identity = llama['checkpoint']
    require(bench.sha(identity['integrity_manifest']) == identity['integrity_manifest_sha256'],
            'Checkpoint integrity manifest changed')
    shards = identity['shards']
    require(bool(shards), 'Missing checkpoint shards')
    for row in shards:
        stat = Path(row['path']).stat()
        require(stat.st_size == row['bytes'] and stat.st_mtime_ns == row['mtime_ns'], 'Checkpoint metadata changed')
    return copy.deepcopy(shards)


def environment(controls):
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(tuple(controls['clear_prefixes']))}
    env.update(controls['set'])
    return env


def run_llama(plan, directory, timeout):
    endpoint = validate_plan(plan); llama = plan['llama']
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, diagnostic_valid=False, semantic_quality_validated=False,
                  eos_validated=False, timeout_seconds=timeout, command=llama['futureCommand'])
    server = sampler = timer = deadline = None
    timed_out = threading.Event(); watchdog_errors = []
    save(directory/'execution.json', result)
    try:
        with socket.socket() as probe: probe.bind(('127.0.0.1', endpoint.port))
        source = llama['source']
        require(subprocess.check_output(['git','-C',source['path'],'rev-parse','HEAD'], text=True).strip() == source['revision']
                and not subprocess.check_output(['git','-C',source['path'],'status','--porcelain'], text=True).strip(),
                'Upstream source revision/clean state differs')
        result['deployment_before'] = raw.deployment_hashes(llama)
        result['checkpoint_before'] = checkpoint_state(llama)
        command = llama['futureCommand']
        require(Path(command[0]).resolve() == Path(llama['deployment']['files'][Path(command[0]).name]['path']).resolve(),
                'Server executable path differs')
        require(command.count('--model') == 1 and Path(command[command.index('--model')+1]).resolve()
                == Path(llama['checkpoint']['shards'][0]['path']).resolve(), 'Server model path differs')
        body = llama['futureRequest']['body']
        save(directory/'request.json', body)
        result['request_sha256'] = bench.sha(directory/'request.json')
        env = environment(llama['environment'])
        result['environment'] = {k:v for k,v in env.items() if k.upper().startswith(
            ('LLAMA_','GGML_','TS_','TENSORSHARP_','STRATA_','CUDA_','OMP_','GOMP_','KMP_','MKL_','OPENBLAS_'))}
        with (directory/'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(command, cwd=Path(command[0]).parent, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, creationflags=bench.PROCESS_FLAGS)
            deadline = time.monotonic() + timeout
            result['server_pid'] = server.pid
            timer = raw.start_watchdog(server, timeout, timed_out, watchdog_errors)
            sampler = bench.Sampler(directory/'memory.jsonl')
            sampler.process = bench.psutil.Process(server.pid); sampler.thread.start()
            save(directory/'execution.json', result)
            def remaining():
                left = deadline-time.monotonic()
                if left <= 0 or timed_out.is_set():
                    timed_out.set(); raise TimeoutError('Total distribution capture deadline exceeded')
                require(server.poll() is None, 'Owned server exited before completed capture')
                return left
            while True:
                try:
                    raw.owned.request(endpoint.port, '/health', None, min(2, remaining())); break
                except (OSError, TimeoutError, RuntimeError): time.sleep(min(.5, remaining()))
            listeners = [c for c in sampler.process.net_connections(kind='tcp') if c.status == bench.psutil.CONN_LISTEN
                         and c.laddr.port == endpoint.port and c.laddr.ip == '127.0.0.1']
            require(len(listeners) == 1, 'Listener does not belong to owned server')
            result['owned_listener_observed'] = dict(pid=server.pid, port=endpoint.port)
            result['loaded_modules'] = raw.observe_modules(server.pid, llama, remaining())
            save(directory/'execution.json', result)
            connection = http.client.HTTPConnection('127.0.0.1', endpoint.port, timeout=remaining())
            try:
                connection.request('POST','/completion',(directory/'request.json').read_bytes(),{'Content-Type':'application/json'})
                response = connection.getresponse(); result['http_status'] = response.status
                # Preserve partial bytes on deadline/HTTP/parser failures without an unbounded RAM read.
                with (directory/'response.raw.json').open('wb') as stream:
                    while True:
                        block = response.read(1024*1024)
                        if not block: break
                        stream.write(block)
                        remaining()
                require(response.status == 200, 'Distribution HTTP status differs from200')
                remaining()
                answer = load(directory/'response.raw.json')
                values = full_distributions(answer, plan)
                write_distributions(directory, values, body['prompt'], plan['diagnostic']['first_token'])
                result.update(complete=True, diagnostic_valid=True, generated_tokens=answer['tokens'],
                    stop_type=answer['stop_type'], expected_cap=True,
                    baseline_prefix_reproduced=answer['tokens'] == plan['diagnostic']['baseline_llama_tokens'],
                    rows=[row_summary(row,plan['diagnostic']['candidate_tokens']) for row in values])
            finally: connection.close()
            remaining()
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        if not isinstance(error, Exception): raise
    finally:
        result['timed_out'] = timed_out.is_set() or (deadline is not None and time.monotonic() >= deadline)
        result.update(raw.owned.cleanup_owned(server, timer, sampler))
        result['watchdog_errors'] = watchdog_errors
        try:
            result['deployment_after'] = raw.deployment_hashes(llama)
            result['checkpoint_after'] = checkpoint_state(llama)
        except Exception as error: result['identity_after_error'] = repr(error)
        result['passed'] = bool(result['diagnostic_valid'] and result['complete'] and result['owned_server_exited']
            and not result['timed_out'] and not result.get('error') and not result['cleanup_errors']
            and not watchdog_errors and not result.get('identity_after_error'))
        save(directory/'execution.json',result)
    return result


def write_distributions(directory, values, prompt, first):
    # Store F64 to retain the JSON API values exactly; these are normalized logprobs.
    data = directory/'logprobs.f64'; rows=[]
    with data.open('wb') as stream:
        for i,row in enumerate(values):
            payload=array.array('d',row)
            if sys.byteorder != 'little': payload.byteswap()
            block=payload.tobytes(); offset=stream.tell(); stream.write(block)
            import hashlib
            rows.append(dict(input_tokens=prompt+([first] if i else []), byte_offset=offset,
                             elements=len(row), sha256=hashlib.sha256(block).hexdigest()))
    save(directory/'rows.json',dict(format='normalized-logprob-f64le',data_path=str(data.resolve()),rows=rows))


def validate_ts(directory, plan, budget):
    report_path=directory/'probe.json'; index_path=directory/'logits/rows.json'
    identity=capture.validate(index_path,report_path)
    data=load(report_path); spec=plan['diagnostic']; config=plan['tensorsharp']
    require(identity['decode_mode']=='teacher-forced' and identity['validated_rows']==2
            and identity['prompt_tokens']==plan['llama']['prompt']['tokens']
            and data['forced_tokens']==[spec['first_token']], 'TS did not consume the original prompt then one scalar')
    require(identity['geometry']['context_limit']==spec['context']
            and identity['geometry']['vocabulary']==spec['vocabulary']
            and identity['geometry']['architecture']=='qwen4exp'
            and data.get('backend')=='GgmlCuda' and data.get('placement')=='host', 'TS geometry/backend differs')
    require(identity['checkpoint']==[(x['sha256'],x['bytes']) for x in plan['llama']['checkpoint']['shards']],
            'TS checkpoint differs from llama')
    opts=data['requested_options']
    require(opts.get('warmup')=='0' and opts.get('iterations')=='1' and opts.get('decode-tokens')=='1'
            and opts.get('prefill-chunk','0')=='0', 'TS invocation geometry changed')
    require(data['environment'].get('TS_HOST_MOE_EXPERT_CACHE_MB')==str(budget), 'TS cache budget differs')
    # This two-row diagnostic has one cold scalar decode: reuse is not expected.
    # Prove activation/ownership independently instead of the probe's warm-hit gate.
    require(opts.get('require-cache')=='0', 'Diagnostic must not require a warm cache hit')
    stats=data.get('cache_stats', {})
    require(all(type(stats.get(key)) is int and stats[key]>=0 for key in
        ('Calls','Hits','Misses','ReservedBytes','BudgetBytes')), 'Missing or invalid cache counters')
    require(stats['BudgetBytes']==budget*(1<<20) and stats['ReservedBytes']<=stats['BudgetBytes'],
            'Cache ownership differs from the diagnostic budget')
    require((stats['Calls']>=identity['geometry']['layers'] and stats['Misses']>0 and stats['ReservedBytes']>0)
            if budget else stats['Calls']==stats['Hits']==stats['Misses']==stats['ReservedBytes']==0,
            'Cold cache activation or disabled-cache isolation is not proven')
    files=config['deployment']['files']
    require(Path(data['native_path']).resolve()==Path(files['GgmlOps.dll']['path']).resolve()
            and identity['native_sha256']==files['GgmlOps.dll']['sha256'], 'Loaded TS native identity differs')
    required={name for name in files if name.startswith('TensorSharp.') and name.endswith('.dll')}
    managed=data['managed_assemblies_sha256']
    require({Path(p).name for p in managed}==required, 'Loaded TS managed set differs')
    for path,digest in managed.items():
        expected=files[Path(path).name]
        require(Path(path).resolve()==Path(expected['path']).resolve() and digest==expected['sha256'], 'Loaded TS managed identity differs')
    index=load(index_path)
    values=[teacher.read_row(Path(index['data_path']),row) for row in index['rows']]
    require(max(range(len(values[0])),key=values[0].__getitem__)==spec['first_token'],
            'TS prefill does not predict the recorded common first token')
    return identity,values


def run_ts(plan,directory,budget,timeout):
    config=plan['tensorsharp']; spec=plan['diagnostic']; llama=plan['llama']
    before=raw.deployment_hashes(config); checkpoint_state(llama)
    prompt=directory.parent/'prompt.ids'; forced=directory.parent/'forced.ids'
    command=['dotnet',config['deployment']['files']['Qwen4ExpExpertCacheProbe.dll']['path'],
        '--model',llama['checkpoint']['shards'][0]['path'],'--output',str(directory/'probe.json'),
        '--placement','host','--backend','ggml_cuda','--tokens-file',str(prompt),
        '--forced-tokens-file',str(forced),'--generation','teacher-forced','--decode-tokens','1',
        '--max-context',str(spec['context']),'--warmup','0','--iterations','1',
        '--require-cache','0','--logits-dir',str(directory/'logits'),
        '--model-identity-report',llama['checkpoint']['integrity_manifest']]
    env=environment(config['environment']); env['TS_HOST_MOE_EXPERT_CACHE_MB']=str(budget)
    execution=compare.run_process(command,Path(command[1]).parent,env,directory,timeout)
    result=dict(execution=execution,deployment_before=before,passed=False)
    try:
        result['deployment_after']=raw.deployment_hashes(config); checkpoint_state(llama)
        require(execution['passed'] is True,'TS process did not complete')
        identity,values=validate_ts(directory,plan,budget)
        result.update(passed=True,identity=identity,rows=[row_summary(row,spec['candidate_tokens']) for row in values])
    except Exception as error: result['error']=f'{type(error).__name__}: {error}'
    save(directory/'diagnostic.json',result)
    return result


def run(plan, output, timeout):
    validate_plan(plan)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'plan.json',plan)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    prompt=plan['llama']['prompt']['tokens']; spec=plan['diagnostic']
    (output/'prompt.ids').write_text(' '.join(map(str,prompt))+'\n',encoding='utf-8')
    (output/'forced.ids').write_text(str(spec['first_token'])+'\n',encoding='utf-8')
    report=dict(complete=False,passed=False,semantic_quality_validated=False,eos_validated=False,
                numerical_equivalence_claimed=False,quiet_performance_validated=False,runs={},
                harness_sha256=bench.sha(__file__),helper_sha256={p:bench.sha(HERE/p) for p in (
                    'llama-raw-completion.py','llama-planned-chat.py','qwen38_capture_evidence.py',
                    'qwen38-llama-teacher.py','qwen38-strata-compare.py','multimodel-quality-bench.py')})
    save(output/'summary.json',report)
    try:
        for budget in plan['tensorsharp']['cache_budgets_mb']:
            name=f'ts-cache{budget}'
            report['runs'][name]=run_ts(plan,output/name,budget,timeout)
            save(output/'summary.json',report)
            require(report['runs'][name]['passed'],'TS capture invalid; refusing further GPU runs')
        report['runs']['llama']=run_llama(plan,output/'llama',timeout)
        save(output/'summary.json',report)
        require(report['runs']['llama']['passed'],'llama complete-distribution capture invalid')
        answer=load(output/'llama/response.raw.json'); reference=full_distributions(answer,plan)
        ts={budget:validate_ts(output/f'ts-cache{budget}',plan,budget)[1] for budget in (0,8192)}
        report['comparisons']=[]
        for step in range(2):
            report['comparisons'].append(dict(stage='prefill' if step==0 else 'decode',
                input_tokens=prompt+([spec['first_token']] if step else []),
                ts_cache0_vs_llama=relative_difference(ts[0][step],reference[step]),
                ts_cache8192_vs_llama=relative_difference(ts[8192][step],reference[step]),
                ts_cache0_vs_cache8192=relative_difference(ts[0][step],ts[8192][step]),
                ts_cache_bitexact=ts[0][step].tobytes()==ts[8192][step].tobytes()))
        report.update(complete=True,passed=True,
            scope='Three successful diagnostic captures with exact conditioning. HTTP values include F32 softmax/log rounding; additive offsets removed. No EOS, semantics, arithmetic equivalence or throughput pass.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        if not isinstance(error,Exception): raise
    finally: save(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=300,help='Each owned model process lifetime; preflight/postprocess excluded')
    parser.add_argument('--prepare-only',action='store_true',help='Validate plan and print the fixed three-process geometry; no GPU/model process')
    args=parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout<=0: parser.error('Require a positive finite timeout')
    plan=load(args.plan);validate_plan(plan)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,processes=3,prompt_count=plan['llama']['prompt']['count'],
                             predictions=2,scalar=plan['diagnostic']['first_token'],context=plan['diagnostic']['context'])))
        return 0
    return 0 if run(plan,bench.evidence_directory(args.output).resolve(),args.timeout)['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
