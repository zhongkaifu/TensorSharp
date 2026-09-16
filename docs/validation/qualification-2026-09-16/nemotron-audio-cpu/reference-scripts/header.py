import json,pathlib,struct,urllib.request
root=pathlib.Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
shard=summary['shards'][0]
base='https://huggingface.co/'+summary['model']+'/resolve/'+summary['revision']+'/'+shard
def read_range(start,end):
    req=urllib.request.Request(base+'?audio_metadata_range='+str(start)+'_'+str(end),headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=90) as r:
        if r.status!=206: raise RuntimeError('Server did not honour Range: '+str(r.status))
        data=r.read(end-start+2)
        if len(data)!=end-start+1: raise RuntimeError('Wrong range length: '+str(len(data)))
        return data
length=struct.unpack('<Q',read_range(0,7))[0]
header=json.loads(read_range(8,7+length))
(root/'shard-header.json').write_text(json.dumps(header,indent=2))
audio={k:header[k] for k in summary['weights']}
byte_count=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in audio.values())
(root/'audio-header.json').write_text(json.dumps({'header_length':length,'weights':audio,'audio_bytes':byte_count},indent=2))
print(json.dumps({'header_length':length,'audio_bytes':byte_count,'dtypes':sorted(set(v['dtype'] for v in audio.values())),'first_weights':dict(list(sorted(audio.items()))[:15])},indent=2))
