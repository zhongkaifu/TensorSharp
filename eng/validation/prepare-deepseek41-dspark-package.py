#!/usr/bin/env python3
"""Prepare a pinned official DSpark download/conversion package locally only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/validation/ggml-no-patch-2026-09-15'
REVISION='dba1be0a40aa45a94ad051997016db3960a90277'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def expected_gguf(capture,config,index):
    headers={name:value for shard in capture['shards'] for name,value in shard['tensor_metadata'].items()}
    assert len(headers)==2401
    c=config['text_config']
    assert (c['num_nextn_predict_layers'],c['dspark_block_size'],c['dspark_noise_token_id'],
            c['dspark_markov_rank'],c['dspark_n_routed_experts'],c['dspark_num_experts_per_tok'])==(3,5,128799,256,128,3)
    assert c['dspark_target_layer_ids']==[37,38,39]
    result={}; used=set()
    def origin(name):
        value=headers[name]; used.add(name)
        scale=name.rsplit('.',1)[0]+'.scale'
        if value['dtype']=='F8_E4M3':
            factor=headers[scale];used.add(scale)
            block=config['quantization_config']['weight_block_size']
            assert factor['shape']==[(value['shape'][i]+block[i]-1)//block[i] for i in range(2)]
        return value
    def add(src,dst,kind):
        value=origin(src);shape=list(reversed(value['shape']))
        block,size={0:(1,4),8:(32,34),30:(1,2),39:(32,17)}[kind]
        assert shape[0]%block==0
        result[dst]={'shape':shape,'type':kind,'bytes':math.prod(shape)//block*size,'sources':[src]}
    q8={'attn.wq_a.weight':'attn_q_a.weight','attn.wq_b.weight':'attn_q_b.weight',
        'attn.wkv.weight':'attn_kv.weight','attn.wo_a.weight':'attn_output_a.weight',
        'attn.wo_b.weight':'attn_output_b.weight','ffn.shared_experts.w1.weight':'ffn_gate_shexp.weight',
        'ffn.shared_experts.w2.weight':'ffn_down_shexp.weight','ffn.shared_experts.w3.weight':'ffn_up_shexp.weight'}
    f32={'attn.q_norm.weight':'attn_q_a_norm.weight','attn.kv_norm.weight':'attn_kv_a_norm.weight',
         'attn.attn_sink':'attn_sinks.weight','attn_norm.weight':'attn_norm.weight','ffn_norm.weight':'ffn_norm.weight',
         **{name:name+'.weight' for name in ['hc_attn_fn','hc_attn_scale','hc_attn_base','hc_ffn_fn','hc_ffn_scale','hc_ffn_base']},
         'ffn.gate.weight':'ffn_gate_inp.weight','ffn.gate.bias':'exp_probs_b.bias','ffn.gate.bias_vl':'exp_probs_b_vl.bias'}
    for stage in range(3):
        prefix=f'mtp.{stage}.'
        for source,target in q8.items(): add(prefix+source,prefix+target,8)
        for source,target in f32.items(): add(prefix+source,prefix+target,0)
        for source,target in [('w1','ffn_gate_exps.weight'),('w3','ffn_up_exps.weight'),('w2','ffn_down_exps.weight')]:
            shape=None; names=[]
            for expert in range(128):
                name=prefix+f'ffn.experts.{expert}.{source}.weight';value=origin(name)
                assert value['dtype']=='I8'
                current=[value['shape'][1]*2,value['shape'][0],128]
                assert shape is None or shape==current
                shape=current;names.append(name)
                scale=name.rsplit('.',1)[0]+'.scale';scale_value=headers[scale];used.add(scale)
                assert scale_value['dtype']=='F8_E8M0' and scale_value['shape']==[shape[1],shape[0]//32]
            result[prefix+target]={'shape':shape,'type':39,'bytes':math.prod(shape)//32*17,'sources':names}
    add('mtp.0.main_norm.weight','mtp.0.main_norm.weight',0)
    add('mtp.0.main_proj.weight','mtp.0.main_proj.weight',8)
    add('mtp.2.norm.weight','mtp.2.norm.weight',0)
    add('mtp.2.confidence_head.proj.weight','mtp.2.confidence_head.proj.weight',0)
    add('mtp.2.markov_head.embed.weight','mtp.2.markov_head.markov_w1.weight',30)
    add('mtp.2.markov_head.head.weight','mtp.2.markov_head.markov_w2.weight',8)
    assert used==set(headers), sorted(set(headers)-used)
    assert all(index['weight_map'][name] in {s['file'] for s in capture['shards']} for name in used)
    metadata={'general.architecture':'deepseek41-dspark','dspark.target_architecture':'deepseek41',
        'dspark.expert_count':128,'dspark.expert_used_count':3,'general.name':'DeepSeek-V4.1-Flash DSpark',
        'general.alignment':32,'dspark.n_layers':3,'dspark.stage_count':3,'dspark.block_size':5,
        'dspark.markov_rank':256,'dspark.noise_token_id':128799,'dspark.target_layer_ids':[37,38,39]}
    header=24
    for key,value in metadata.items():
        payload=8+len(value.encode()) if isinstance(value,str) else 12+4*len(value) if isinstance(value,list) else 4
        header+=8+len(key.encode())+4+payload
    for name,tensor in result.items(): header+=8+len(name.encode())+4+8*len(tensor['shape'])+4+8
    size=(header+31)//32*32+sum((row['bytes']+31)//32*32 for row in result.values())
    return {'metadata':metadata,'tensors':result,'estimated_output_bytes':size,'official_source_tensors_covered':len(used)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    capture_path=DOC/'deepseek41-dspark-discovery/headers/capture.json'
    capture=json.loads(capture_path.read_text())
    assert capture['revision']==REVISION and capture['status']=='captured-header-only'
    metadata=ROOT/'TestResults/ggml-no-patch-2026-09-15/deepseek41-draft-metadata'
    config=metadata/'deepseek-ai--DeepSeek-V4.1-Flash.config.json'
    index=metadata/'deepseek-ai--DeepSeek-V4.1-Flash.model.safetensors.index.json'
    assert sha(config)=='8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'
    assert sha(index)=='74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8'
    expected=expected_gguf(capture,json.loads(config.read_text()),json.loads(index.read_text()))
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'metadata').mkdir();(out/'headers').mkdir()
    for source,destination in [(config,out/'metadata/config.json'),(index,out/'metadata/model.safetensors.index.json'),
        (metadata/'deepseek-ai--DeepSeek-V4.1-Flash.api.json',out/'metadata/repository-api.json'),
        (capture_path,out/'headers/capture.json'),(Path(__file__),out/Path(__file__).name),
        (ROOT/'eng/validation/download-convert-deepseek41-dspark.py',out/'download-convert-deepseek41-dspark.py'),
        (ROOT/'eng/validation/tests/test_deepseek41_dspark_preparation.py',out/'test_deepseek41_dspark_preparation.py'),
        (ROOT/'eng/dsv4-dspark-to-gguf.py',out/'dsv4-dspark-to-gguf.py'),
        (ROOT/'eng/tests/test_dspark_converter.py',out/'test_dspark_converter.py')]: shutil.copy2(source,destination)
    for shard in capture['shards']:
        prefix=capture_path.parent/(shard['file']+'.header-prefix')
        assert sha(prefix)==shard['header_prefix_sha256']
        shutil.copy2(prefix,out/'headers'/prefix.name)
    save(out/'expected-gguf.json',expected)
    shards=[{key:row[key] for key in ['file','file_bytes','publisher_sha256','origin_url','header_prefix_bytes','header_prefix_sha256']}
            for row in capture['shards']]
    save(out/'plan.json',{'status':'prepared-not-executed','release_qualified':False,'revision':REVISION,
        'destination':'/workspace/models/deepseek41-dspark','shards':shards,
        'total_source_bytes':sum(s['file_bytes'] for s in shards),'estimated_output_bytes':expected['estimated_output_bytes'],
        'converter_sha256':sha(out/'dsv4-dspark-to-gguf.py'),'converter_test_count':16,'expert_type':'mxfp4',
        'memory_guard_bytes':24*1024**3,'disk_margin_bytes':4*1024**3,
        'resource_limit':'Point-in-time checks require root to reserve the complete exclusive preparation window; they do not prevent another unrelated owner from starting a model later.'})
    save(out/'manifest.json',{'status':'prepared-not-executed','release_qualified':False,
        'files':{str(p.relative_to(out)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'package':str(out),'manifest_sha256':sha(out/'manifest.json'),'output_bytes':expected['estimated_output_bytes'],
                      'output_tensors':len(expected['tensors']),'source_tensors':2401}))


if __name__=='__main__': main()
