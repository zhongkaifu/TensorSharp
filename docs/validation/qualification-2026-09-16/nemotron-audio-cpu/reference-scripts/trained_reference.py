import json,mmap,pathlib,runpy,time,resource
import torch
from torch import nn
root=pathlib.Path(__file__).resolve().parent
reference=runpy.run_path(str(root/'reference.py'))
ns=reference['ns']
config=json.loads((root/'config.json').read_text())
cfg=reference['config']
for key,value in config['sound_config'].items():setattr(cfg,key,value)
cfg.num_key_value_heads=cfg.num_attention_heads
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling=ns['ParakeetEncoderSubsamplingConv2D'](cfg)
        self.layers=nn.ModuleList(ns['ParakeetEncoderBlock'](cfg,i) for i in range(cfg.num_hidden_layers))
        self.project=ns['SoundProjection'](cfg.hidden_size,cfg.projection_hidden_size,config['llm_config']['hidden_size'],bias=cfg.projection_bias)
    def forward(self,x,valid):
        mask=torch.arange(x.shape[1])[None,:]<valid
        x=self.subsampling(x,mask)
        pos=self.encode_positions(x)
        mask=torch.arange(x.shape[1])[None,:]<((valid+7)//8)
        attention=mask[:,None,None,:]&mask[:,None,:,None]
        for i,layer in enumerate(self.layers):
            x=layer(x,attention_mask=attention,position_embeddings=pos)
            print('layer',i,flush=True)
        return self.project(x)
with torch.device('meta'):model=FullModel().eval()
header=json.loads((root/'audio-header.json').read_text())['weights']
asset=pathlib.Path('/workspace/models/nemotron-omni/audio-e5e9932')
start=json.loads((asset/'manifest.json').read_text())['source_range_start']
state={}
with (asset/'official-audio-bf16.bin').open('rb') as file:
    mm=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
    for name,meta in header.items():
        if '.feature_extractor.' in name:continue
        key=('project.'+name[len('sound_projection.'):]) if name.startswith('sound_projection.') else name[len('sound_encoder.encoder.'):]
        if meta['dtype']=='I64':state[key]=torch.tensor(0,dtype=torch.int64);continue
        a,b=meta['data_offsets'];state[key]=torch.frombuffer(mm,dtype=torch.bfloat16,count=(b-a)//2,offset=a-start).reshape(meta['shape']).clone()
model.load_state_dict(state,strict=True,assign=True)
del state
model.encode_positions=ns['ParakeetEncoderRelPositionalEncoding'](cfg).to(torch.bfloat16)
torch.set_num_threads(4)
frames,valid=21,20
x=torch.sin(torch.arange(frames*128,dtype=torch.float32)*.0137).reshape(1,frames,128)
x[:,valid:]=0
begin=time.monotonic()
with torch.no_grad():output=model(x.to(torch.bfloat16),valid)
report={'frames':frames,'valid_frames':valid,'input':x.flatten().tolist(),'expected':output.float().flatten().tolist(),'shape':list(output.shape),'elapsed_seconds':time.monotonic()-begin,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gguf_sha256':json.loads((asset/'manifest.json').read_text())['gguf_sha256'],'reference':'pinned official Parakeet modules and NVIDIA sound projector'}
(root/'trained_reference.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print('trained_reference',report['shape'],report['elapsed_seconds'],report['max_rss_kib'])
