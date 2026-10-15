"""Synthetic, model-free metadata for historical-plan unit tests.

These inputs exercise the original schemas, 113 request identities and nine-suite
ordering. They are not archived benchmark results or verified model artifacts.
Production tools retain their historical SHA pins; tests override pins only in
their isolated imported module/subprocess after writing these temporary inputs.
"""
import copy
import hashlib
import json
from pathlib import Path


ENGINE = Path(__file__).with_name('request_body_builder.py.fixture')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


def bundle(root):
    root=Path(root);reports=root/'reports';archive=reports/'sparse-attention-candidate/deepseek41-sparse-attention'
    outputs=['quality.json','structured-tool-results.json','serial-tool-workflows.json','long-context.json',
             'decode-benchmark.json','tool-policies.json','media.json','invalid-images.json','audio-rejection.json']
    suites=[['fixture-validation.py','--profile','synthetic-only','--output','fixture/'+name] for name in outputs]
    suites[3]+=['--scenarios','long_8k,long_32k,long_64k'];suites[4]+=['--concurrency','1,4']
    sparse={'id':'synthetic-sparse','model':'/synthetic-only/checkpoint-00001-of-00011.gguf',
            'extra_args':['--backend','ggml_cuda','--tp','7','--cpu-moe','12','--no-skills'],
            'env':{'TS_DSV4_FA':'1','TS_DSV41_SPARSE_FA':'1','TS_DSV41_TP':'0','MAX_CONTEXT':'65536'},
            'suites':suites,'before_suites':[['fixture-prerequisite.py']], 'before_suites_quiet_seconds':2}
    sparse_path=write(reports/'sparse-attention-candidate/sparse-attention-build/full-profile.json',sparse)
    reference=copy.deepcopy(sparse);reference['id']='synthetic-reference';reference['env']['TS_DSV4_FA']='0'
    for command in reference['suites']:command[command.index('--profile')+1]=reference['id']
    reference_path=write(reports/'decomposed-f32-reference/raw/profile.json',{'profile':reference})
    model=write(reports/'deepseek41-q4km.json',{'status':'verified','repository':'synthetic-only/no-model',
               'revision':'synthetic-metadata','directory':'/synthetic-only',
               'files':[{'path':f'fixture-{index:02d}.gguf','sha256':f'{index:064x}'} for index in range(11)]})
    tensor=write(reports/'tp7-checkpoint-v5/actual-q4km-routed-types.json',{'scope':'synthetic schema only','types':[]})
    pins={};global_index=0
    # 30*2 +25*1 +5*5 +3*1 =113 requests from63 cases.
    for name,count,turns in [('quality.json',30,2),('structured-tool-results.json',25,1),
                             ('serial-tool-workflows.json',5,5),('long-context.json',3,1)]:
        cases=[]
        for case_index in range(count):
            case_turns=[]
            for turn_index in range(turns):
                format_={'type':'json_object'} if global_index<20 else \
                        {'type':'json_schema','json_schema':{'name':'fixture','schema':{'type':'object'}}} if global_index<25 else None
                request={'messages':[{'role':'user','content':f'fixture request {global_index}: 北京 🚀'}],
                         'tools':[{'type':'function','function':{'name':'fixture_tool','parameters':{'type':'object','properties':{}}}}],
                         'response_format':format_,'extra_body':{'think':False,'tool_choice':'auto'},
                         'max_tokens':128+turn_index,'stream':True}
                case_turns.append({'request':request,'metrics':{'assistant_message':{'role':'assistant','content':'synthetic fixture only'}}})
                global_index+=1
            cases.append({'tag':f'fixture-c{case_index}','scenario':'fixture','concurrency':1,
                          'status':'fail','turns':case_turns})
        path=write(archive/name,{'weights_id':'58d8ac86298fdf85a2440defee08b1abcad32e45-Q4_K_M',
                               'thinking':False,'run_complete':True,'execution_plan':{'expected_cases':count},'cases':cases})
        pins[name]=digest(path)
    inventory=[]
    def entry(name,files):
        inventory.append({'id':name,'status':'downloaded-and-verified',
                          'files':[{'path':'/synthetic-only/'+name+'/'+file,'sha256':'0'*64} for file in files]})
    entry('minilm',['fixture.gguf']);entry('snowflake',['fixture.gguf'])
    for name,file in [('gemma4-e4b','gemma-4-E4B-it-Q8_0.gguf'),('gemma4-12b','gemma-4-12B-it-qat-UD-Q4_K_XL.gguf'),
                      ('gemma4-26b-qat','gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'),
                      ('gemma4-26b','gemma-4-26B-A4B-it-Q4_0.gguf'),('gemma4-31b','gemma-4-31B-it-Q4_0.gguf')]:
        entry(name,[file,'mmproj-fixture.gguf'])
    entry('minimax-h3',['minimax_h3_fl2va_pruned-Q4_K.gguf','minimax_h3_ref2va_pruned-Q4_K.gguf',
                       'qwen3vl_32b_minimax_h3-Q4_K_M.gguf','minimax_h3_video_vae_fp16.safetensors',
                       'minimax_h3_audio_vae_fp32.safetensors'])
    entry('minimax-h3-tokenizer',['vocab.json','merges.txt','tokenizer_config.json'])
    for name in ('wan-turbo','wan-base','wan21-13b','wan21-14b'):entry(name,['fixture.gguf'])
    entry('wan-a14b',['fixture_high_noise.gguf','fixture_low_noise.gguf'])
    for name in ('wan22-t2v-a14b','wan22-i2v-a14b-base'):entry(name,['fixture_HighNoise.gguf','fixture_LowNoise.gguf'])
    entry('umt5-encoder',['fixture.gguf']);entry('wan-vae',['fixture.safetensors']);entry('wan-a14b-vae',['fixture.safetensors'])
    inventory_path=write(root/'inventory.json',{'scope':'synthetic schema only; no files downloaded/verified','models':inventory})
    return dict(root=root,reports=reports,archive=archive,pinned_reports=pins,sparse=sparse_path,reference=reference_path,
                model=model,tensor=tensor,inventory=inventory_path,engine=ENGINE)
