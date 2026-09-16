"""Diagnostic FP64 arithmetic, retaining the model's explicit BF16/FP8 cache boundaries.

This does not replace or relax the independent F32 acceptance oracle.
"""
from pathlib import Path
import importlib.util,inspect,textwrap,json
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
repo=Path('/workspace/ts-codex-20260916-r3/repo')
spec=importlib.util.spec_from_file_location('dsref',repo/'eng/dsv41-dspark-reference.py');ds=importlib.util.module_from_spec(spec);spec.loader.exec_module(ds)
r=ds.reference
original_rows=r.GgufWeights.rows
r.GgufWeights.rows=lambda self,name,rows:original_rows(self,name,rows).double()
def mm(self,name,x,expert=None,out_features=None):
    shape=self.shape(name);out_features=out_features or (shape[1] if len(shape)>1 else 1)
    offset=0 if expert is None else int(expert)*out_features
    return torch.nn.functional.linear(x.double(),self.rows(name,slice(offset,offset+out_features)))
r.GgufWeights.mm=mm
def rms(x,epsilon,weight=None):
    out=x.double()*torch.rsqrt(x.double().square().mean(-1,keepdim=True)+epsilon)
    return out if weight is None else out*weight.double()
r.rms=rms
original_cache=r.Reference.cache
r.Reference.cache=lambda self,x,kind='raw':original_cache(self,x,kind).double()
ns={}
exec(textwrap.dedent(inspect.getsource(r.Reference.rope)).replace('.float()', '.double()'),r.__dict__,ns)
r.Reference.rope=ns['rope']
fixture=Path('/workspace/ts-fix-dspark/repo/fixtures/text-f32-small')
config=json.loads((fixture/'deepseek41.config.json').read_text())['config']
w=r.GgufWeights(fixture/'deepseek41-fixture.gguf');engram=r.load_engram(fixture/'deepseek41.engram.bin')
corpus=np.random.default_rng(159410).integers(3,240,620,dtype=np.int32)
result={}
for prefix,run_dir in [(5,Path('/workspace/ts-numerical-r3-dspark-cuda2-20260916')),(11,Path('/workspace/ts-numerical-r3-prefix11-trace'))]:
    target=ds.TargetWithDraftFeatures(w,config,engram,'model',target_layers=[1,3,4])
    target.forward(corpus[:prefix].tolist())
    recorded=np.load(run_dir/f'oracle-prefix{prefix}.npz')
    anchor=int(recorded['target_logits'][-1].argmax())
    if prefix==5:
        dw=r.GgufWeights(run_dir/'deepseek41-dspark-fixture.gguf')
        settings=json.loads((run_dir/'draft-config.json').read_text())
        head=ds.DSparkReference(dw,w,config,**{k:v for k,v in settings.items() if k!='target_layers'})
        head.commit_features(target.draft_features())
        tokens,confidence,trace=head.draft(anchor)
        result['prefix5']={'stage1_token2_channel35_prequant':float(head.committed_prequant[1][2,35]),
          'stage1_token2_channel35_cache':float(head.keys[1][2,35]),'tokens':tokens,'confidence':confidence.tolist(),
          'original_confidence':recorded['confidence'].tolist()}
    else:
        logits=target.forward([anchor]+recorded['tokens'].tolist()).numpy()
        result['prefix11']={'layer2_token16_channel19_cache':float(target.states[2]['raw'][16,19]),
          'max_abs_from_original_f32_logits':float(np.max(np.abs(logits-recorded['target_logits'][-1]))) if False else None}
print('DIAGNOSTIC_FP64 '+json.dumps(result))
