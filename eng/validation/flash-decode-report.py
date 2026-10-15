#!/usr/bin/env python3
"""Render annotated Flash benchmark evidence without conflating correctness and speed.

The input JSON supplies title, notes, groups (name and comparison directories),
and optional semantic_report. Paths resolve relative to that JSON. Each group is
revalidated by the benchmark comparator and must use identical binary/checkpoint
identities. First requests remain separate from repeated requests. Output is
restricted to ignored validation directories. No semantic failures are waived.
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def distribution(values):
    return dict(median=statistics.median(values), minimum=min(values),
                maximum=max(values), count=len(values))


def identities(model):
    return dict(native=model['native_sha256'], checkpoint=model['checkpoint_identity'],
                managed={PureWindowsPath(k).name: v for k, v in model['managed_assemblies_sha256'].items()})


def aggregate(directories):
    spec = importlib.util.spec_from_file_location('flash_comparator', Path(__file__).with_name('compare-flash-decode-runs.py'))
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    rows = []
    reference = None
    intervals = []
    chains = set()
    for directory in directories:
        saved = read(directory / 'comparison.json')
        executions = [Path(p) for p in saved['reports']]
        first_model = read(executions[0].parent / 'model.json')
        second_model = read(executions[1].parent / 'model.json')
        file_comparison = first_model['environment'].get('TS_HOST_MOE_FILE_READ') != second_model['environment'].get('TS_HOST_MOE_FILE_READ')
        lfu_comparison = first_model['environment'].get('TS_HOST_MOE_EXPERT_CACHE_LFU') != second_model['environment'].get('TS_HOST_MOE_EXPERT_CACHE_LFU')
        checked = comparator.compare(executions, saved['execution_order'][0] == 'candidate', file_comparison, lfu_comparison)
        if not checked['passed']:
            raise ValueError(f'{directory}: {checked["failures"]}')
        for arm, execution_path in zip(checked['execution_order'], executions):
            model = read(execution_path.parent / 'model.json')
            execution = read(execution_path)
            key = dict(identities=identities(model), prompt=model['prompt_tokens'],
                       geometry=model['model_geometry'], mode=model['decode_mode'],
                       experiment='lfu' if lfu_comparison else 'file' if file_comparison else 'prefetch',
                       expert_cache_bytes=model['cache_stats']['ReservedBytes'],
                       environment={k:v for k,v in model['environment'].items() if k != (
                           'TS_HOST_MOE_EXPERT_CACHE_LFU' if lfu_comparison else 'TS_HOST_MOE_FILE_READ' if file_comparison else 'TS_HOST_MOE_EXPERT_CACHE_PREFETCH')},
                       options={k:v for k,v in model['requested_options'].items() if k not in ('output','logits-dir')})
            if reference is None:
                reference = key
            if key != reference:
                raise ValueError('Cannot pool different binaries, checkpoint or request identities')
            intervals.append((execution['started_unix'], execution['started_unix'] + execution['wall_seconds']))
            for run in model['runs']:
                chains.add(run['full_logit_chain_sha256'])
            rows.append(dict(arm=arm, execution=str(execution_path), runs=model['runs'],
                peak_working_set_bytes=execution['windows_process_memory']['peak_rss_bytes'],
                sampled_board_peak_mib=max(p['peak_memory_used_mib'] for p in execution['gpu_sampling']['peaks']),
                expert_cache_bytes=model['cache_stats']['ReservedBytes']))
    intervals.sort()
    if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
        raise ValueError('Cannot pool overlapping benchmark processes')
    if len(chains) != 1:
        raise ValueError('Vocabulary histories differ across comparison rounds')
    arms = {}
    for arm in ('control', 'candidate'):
        selected = [r for r in rows if r['arm'] == arm]
        result = {}
        for phase, first in (('first_request', True), ('warm_request', False)):
            requests = [r for p in selected for r in p['runs'] if r['warmup'] is first]
            result[phase] = {metric: distribution([r[metric] for r in requests]) for metric in ('prefill_tps', 'decode_tps')}
            result[phase]['decode_expert_misses'] = distribution([
                r['cache_stats_after_decode']['Misses'] - r['cache_stats_after_prefill']['Misses']
                for r in requests])
        result.update(peak_working_set_bytes=max(p['peak_working_set_bytes'] for p in selected),
                      sampled_board_peak_mib=max(p['sampled_board_peak_mib'] for p in selected),
                      expert_cache_bytes=sorted({p['expert_cache_bytes'] for p in selected}))
        arms[arm] = result
    return dict(identity=reference, full_logit_chain_sha256=chains.pop(), arms=arms, executions=rows,
                warm_candidate_to_control={k: arms['candidate']['warm_request'][k]['median'] /
                    arms['control']['warm_request'][k]['median'] for k in ('prefill_tps', 'decode_tps')})


def escape(value):
    return html.escape(str(value))


def qualify_semantics(group, case):
    spec = importlib.util.spec_from_file_location('flash_quality', Path(__file__).with_name('qwen38-strata-compare.py'))
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    if group['identity']['options'].get('prompt') != quality.CASES[case]:
        raise ValueError('Semantic case does not match the measured prompt')
    requests = [r for execution in group['executions'] for r in execution['runs']]
    if not requests:
        raise ValueError('No measured requests to qualify')
    checks = [quality.semantic_check(case, r['selected_text'], r['finish_reason'] == 'eos') for r in requests]
    return dict(case=case, passed=all(c['passed'] for c in checks), requests=len(checks),
                passed_requests=sum(c['passed'] for c in checks), example_text=requests[0]['selected_text'],
                limitation='Deterministic task check, not broad language quality or an executed agent loop.')


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def render(report):
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(report['title'])+'</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;color:#202c3b;background:#f3f5f8;margin:auto;max-width:1200px;padding:32px}h1{font-size:30px}h2{margin-top:36px}section{background:white;border:1px solid #dce3ec;border-radius:10px;padding:24px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3ec}th{background:#eef3f7;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace;background:#f4f6f8;padding:12px}a{color:#005eac}.notice{border-left:5px solid #b35800}li{margin:8px 0}</style>',
        '<h1>'+escape(report['title'])+'</h1><section class="notice"><ul>'+''.join('<li>'+escape(n)+'</li>' for n in report['notes'])+'</ul></section>']
    for name, group in report['groups'].items():
        parts.append('<section><h2>'+escape(name)+'</h2>')
        rows = []
        for arm, data in group['arms'].items():
            for phase in ('first_request', 'warm_request'):
                metrics = data[phase]
                fmt = lambda k: f'{metrics[k]["median"]:.2f} ({metrics[k]["minimum"]:.2f}–{metrics[k]["maximum"]:.2f})'
                rows.append([arm, phase, metrics['decode_tps']['count'], fmt('prefill_tps'), fmt('decode_tps'),
                    f'{metrics["decode_expert_misses"]["median"]:g}',
                    f'{data["peak_working_set_bytes"]/2**30:.2f}', f'{data["sampled_board_peak_mib"]/1024:.2f}'])
        parts.append(table(['路径','请求阶段','样本数','Prefill tok/s：中位数（范围）','Decode tok/s：中位数（范围）','Decode 专家未命中/请求：中位数','进程峰值 WS GiB¹','全设备峰值 GiB¹'],rows))
        labels = 'control=LRU，candidate=衰减频次淘汰，两侧均为有界文件读取。' if group['identity']['experiment'] == 'lfu' else 'control=mmap，candidate=有界文件读取。' if group['identity']['experiment'] == 'file' else 'control=关闭预取，candidate=自适应预取。'
        parts.append('<p>'+labels+'¹每条路径全部进程的峰值，不能拆成阶段峰值；全设备含桌面。首次请求没有清空 OS 缓存。热请求同进程重复相同提示与 greedy 历史。</p>')
        parts.append('<p>热请求变化：'+escape(', '.join(f'{k} {(v-1)*100:+.2f}%' for k,v in group['warm_candidate_to_control'].items()))+'</p>')
        if group.get('quality'):
            q = group['quality']
            parts.append('<p>独立任务检查：'+('通过' if q['passed'] else '失败')+
                         f'（{q["passed_requests"]}/{q["requests"]} 请求满足严格条件）。不剥除代码围栏或修改原输出。</p>'+
                         '<details><summary>原始生成示例</summary><pre>'+escape(q['example_text'])+'</pre></details>')
        parts.append('<details><summary>二进制身份与完整词表校验</summary><pre>'+escape(json.dumps(dict(identity=group['identity'], full_logit_chain_sha256=group['full_logit_chain_sha256']),ensure_ascii=False,indent=2))+'</pre></details></section>')
    if report.get('semantic'):
        semantic=report['semantic']
        parts.append('<section><h2>独立 Strata 对照与严格语义结果</h2><p>各行均为新进程首个请求。两引擎计时范围和分母不同；不是相同 RAM+VRAM 上限的性能验收。Strata 使用同源兼容包，其中 dense 权重经 BF16 转换。原始语义失败保持失败。</p>')
        rows=[]
        outputs=[]
        for case, data in semantic['cases'].items():
            for run in data['runs']:
                for arm, result in run['arms'].items():
                    rows.append([case,run['repetition'],arm,f'{result["prefill_tps"]:.2f}',f'{result["decode_tps"]:.2f}',
                        f'{result["time_to_first_token_ms"]/1000:.2f}',f'{result["whole_process_seconds"]:.2f}',
                        f'{result["sampled_peak_working_set_bytes"]/2**30:.2f}',
                        f'{result["sampled_board_peak_mib"]/1024:.2f}' if result.get('sampled_board_peak_mib') is not None else '不可用',
                        '通过' if result['semantic_check']['passed'] else '失败'])
                    outputs.append('<details><summary>'+escape(f'{case} / {run["repetition"]} / {arm}')+'</summary><pre>'+escape(result['selected_text'])+'</pre></details>')
        parts.append(table(['用例','轮次','路径','Prefill tok/s','Decode tok/s','TTFT s','进程 s','采样 WS GiB','全设备 GiB','严格语义'],rows))
        parts.extend(outputs)
        parts.append('</section>')
    for section in report.get('sections',[]):
        parts.append('<section><h2>'+escape(section['title'])+'</h2><ul>'+''.join('<li>'+escape(n)+'</li>' for n in section['notes'])+'</ul></section>')
    parts.append('<p>完整数据：<a href="summary.json">summary.json</a>。生成脚本只聚合证据，不把比较器 passed 解释为性能或语义验收通过。</p></html>')
    return '\n'.join(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    output=args.output.resolve()
    if not any(output.is_relative_to(root / p) for p in ('artifacts','docs/validation')):
        parser.error('Output must stay in an ignored validation directory')
    config=read(args.input)
    resolve=lambda p:(args.input.resolve().parent / p).resolve()
    result={k:v for k,v in config.items() if k not in ('groups','semantic_report')}
    result['groups']={}
    for group in config['groups']:
        data = aggregate([resolve(p) for p in group['directories']])
        if group.get('semantic_case'):
            data['quality'] = qualify_semantics(data, group['semantic_case'])
        result['groups'][group['name']] = data
    result['semantic']=read(resolve(config['semantic_report'])) if config.get('semantic_report') else None
    if result['semantic']:
        semantic_root=resolve(config['semantic_report']).parent
        for case, data in result['semantic']['cases'].items():
            for run in data['runs']:
                for arm, record in run['arms'].items():
                    gpu_path=semantic_root/case/str(run['repetition'])/arm/'gpu.json'
                    samples=read(gpu_path) if gpu_path.is_file() else []
                    board=[]
                    for sample in samples:
                        if sample.get('exit_code') != 0:
                            continue
                        for row in csv.reader(sample.get('csv','').splitlines(),skipinitialspace=True):
                            try:
                                if len(row)==7:
                                    board.append(float(row[3]))
                            except ValueError:
                                pass
                    record['sampled_board_peak_mib']=max(board) if board else None
    result['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'report.html').write_text(render(result),encoding='utf-8')
    print(output/'report.html')


if __name__=='__main__':
    main()
