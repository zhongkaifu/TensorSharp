import hashlib,json,pathlib,struct,urllib.request
research=pathlib.Path(__file__).resolve().parent
summary=json.loads((research/'summary.json').read_text())
header=json.loads((research/'audio-header.json').read_text())
config=json.loads((research/'config.json').read_text())
root=pathlib.Path('/workspace/models/nemotron-omni/audio-'+summary['revision'][:7]);root.mkdir(exist_ok=True)
weights=sorted((n,v) for n,v in header['weights'].items() if v['dtype']=='BF16')
ranges=sorted((v['data_offsets'][0],v['data_offsets'][1]) for n,v in weights)
assert all(ranges[i-1][1]==ranges[i][0] for i in range(1,len(ranges)))
start,end=ranges[0][0],ranges[-1][1]
raw=root/'official-audio-bf16.bin'
url='https://huggingface.co/'+summary['model']+'/resolve/'+summary['revision']+'/'+summary['shards'][0]
if not raw.exists():
    partial=raw.with_suffix('.partial')
    with partial.open('wb') as output:
        for offset in range(start,end,16*1024*1024):
            stop=min(end,offset+16*1024*1024)
            absolute=8+header['header_length']+offset
            request=urllib.request.Request(url+'?audio_chunk='+str(offset),headers={'Range':f'bytes={absolute}-{absolute+stop-offset-1}'})
            with urllib.request.urlopen(request,timeout=120) as response:
                if response.status!=206: raise RuntimeError('Range ignored')
                data=response.read(stop-offset+1)
                if len(data)!=stop-offset: raise RuntimeError('Wrong range length')
                output.write(data)
            print(f'{stop-start}/{end-start}',flush=True)
    partial.rename(raw)
assert raw.stat().st_size==end-start
def string(value):
    b=value.encode();return struct.pack('<Q',len(b))+b
metadata={'general.architecture':'nemotron_audio','general.name':'NVIDIA Nemotron Omni Parakeet audio companion','nemotron.audio.source_revision':summary['revision'],'nemotron.audio.source_model':summary['model'],'nemotron.audio.compute_bf16':False,'nemotron.audio.projection_dim':config['llm_config']['hidden_size']}
metadata.update({'nemotron.audio.'+k:v for k,v in config['sound_config'].items() if isinstance(v,(str,int,bool))})
out=root/'mmproj-audio-bf16-f32compute.gguf'
with out.open('wb') as f:
    f.write(b'GGUF'+struct.pack('<IQQ',3,len(weights),len(metadata)))
    for k,v in metadata.items():
        f.write(string(k))
        if isinstance(v,bool):f.write(struct.pack('<I?',7,v))
        elif isinstance(v,int):f.write(struct.pack('<II',4,v))
        else:f.write(struct.pack('<I',8)+string(v))
    position=0;offsets=[]
    for n,v in weights:
        f.write(string(n)+struct.pack('<I',len(v['shape'])))
        f.write(struct.pack('<'+'Q'*len(v['shape']),*reversed(v['shape'])))
        f.write(struct.pack('<IQ',30,position));offsets.append(position)
        position+=(v['data_offsets'][1]-v['data_offsets'][0]+31)//32*32
    f.write(bytes((-f.tell())%32));base=f.tell()
    with raw.open('rb') as source:
        for (n,v),offset in zip(weights,offsets):
            f.write(bytes(base+offset-f.tell()))
            a,b=v['data_offsets'];source.seek(a-start)
            remaining=b-a
            while remaining:
                block=source.read(min(remaining,4*1024*1024));f.write(block);remaining-=len(block)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest={'source':summary['model'],'revision':summary['revision'],'original_shard':summary['shards'][0],'source_range_start':start,'source_range_end_exclusive':end,'tensor_count':len(weights),'discarded_tensors':'24 batch-normalization num_batches_tracked counters (I64, unused for inference)','raw_sha256':sha(raw),'gguf_sha256':sha(out),'gguf_bytes':out.stat().st_size}
(root/'manifest-f32compute.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
