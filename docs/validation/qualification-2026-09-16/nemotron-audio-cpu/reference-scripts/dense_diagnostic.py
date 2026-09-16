import json,mmap,pathlib,torch
root=pathlib.Path('/workspace/ts-nemotron-audio-research');asset=pathlib.Path('/workspace/models/nemotron-omni/audio-e5e9932')
d=json.loads((root/'trained_details.json').read_text());diff=json.loads(pathlib.Path('/workspace/ts-nemotron-audio-derived-r6/trained-diffs.json').read_text())
p='sound_encoder.encoder.layers.0.conv.pointwise_conv1';x=torch.tensor(d['sound_encoder.encoder.layers.0.norm_conv']).reshape(3,1024)
h=json.loads((root/'audio-header.json').read_text())['weights'][p+'.weight'];start=json.loads((asset/'manifest.json').read_text())['source_range_start'];a,b=h['data_offsets']
with (asset/'official-audio-bf16.bin').open('rb') as f:
 mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);w=torch.frombuffer(mm,dtype=torch.bfloat16,count=(b-a)//2,offset=a-start).reshape(2048,1024).clone()
y=(x.double()@w.double().T);yb=y.bfloat16().float().flatten();expected=torch.tensor(d[p]);actual=torch.tensor(next(r for r in diff if r['name']==p)['actual'])
for i in torch.where(actual!=expected)[0].tolist():print('index',i,'expected',float(expected[i]),'actual',float(actual[i]),'fp64',float(y.flatten()[i]),'fp64bf16',float(yb[i]))
print('double differs official',int((yb!=expected).sum()),'double differs native',int((yb!=actual).sum()))
