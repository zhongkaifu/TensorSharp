import ast, contextlib, hashlib, json, math, pathlib, types, urllib.request
import torch
from torch import nn
torch.set_num_threads(2)
root=pathlib.Path(__file__).resolve().parent
path='src/transformers/models/parakeet/modeling_parakeet.py'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TensorSharp-validation'}),timeout=60) as r:return r.read()
rev='da7234ac435f6d7c75d8b88d1ac32f53fb1f19a6'
source=get('https://raw.githubusercontent.com/huggingface/transformers/'+rev+'/'+path).decode()
(root/'modeling_parakeet.py').write_text(source)
class RemoveDecorators(ast.NodeTransformer):
    def visit_ClassDef(self,node):node.decorator_list=[];return self.generic_visit(node)
    def visit_FunctionDef(self,node):node.decorator_list=[d for d in node.decorator_list if isinstance(d,ast.Name) and d.id in ("staticmethod","classmethod","property")];return self.generic_visit(node)
def load_classes(source,names,ns):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    module=ast.fix_missing_locations(RemoveDecorators().visit(module))
    exec(compile(module,'official-reference.py','exec'),ns)
def sdpa(module,query,key,value,attention_mask,scaling,dropout,**kwargs):
    out=torch.nn.functional.scaled_dot_product_attention(query,key,value,attention_mask,dropout,scale=scaling)
    return out.transpose(1,2).contiguous(),None
ns={'torch':torch,'nn':nn,'math':math,'GradientCheckpointingLayer':nn.Module,'ACT2FN':{'silu':torch.nn.functional.silu},'maybe_autocast':lambda **kwargs:contextlib.nullcontext(),'ALL_ATTENTION_FUNCTIONS':types.SimpleNamespace(get_interface=lambda *a:sdpa),'eager_attention_forward':sdpa}
names=['ParakeetEncoderRelPositionalEncoding','ParakeetEncoderFeedForward','ParakeetEncoderConvolutionModule','ParakeetEncoderAttention','ParakeetEncoderSubsamplingConv2D','ParakeetEncoderBlock']
load_classes(source,names,ns)
load_classes((root/'audio_model.py').read_text(),['SquaredReLU','RMSNorm','SoundProjection'],ns)
config=types.SimpleNamespace(hidden_size=8,num_attention_heads=2,num_key_value_heads=2,num_hidden_layers=2,intermediate_size=16,conv_kernel_size=3,convolution_bias=False,attention_bias=False,hidden_act='silu',activation_dropout=0.,attention_dropout=0.,subsampling_conv_channels=4,subsampling_conv_kernel_size=3,subsampling_conv_stride=2,subsampling_factor=8,num_mel_bins=8,max_position_embeddings=512,_attn_implementation='sdpa')
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling=ns['ParakeetEncoderSubsamplingConv2D'](config)
        self.encode_positions=ns['ParakeetEncoderRelPositionalEncoding'](config)
        self.layers=nn.ModuleList(ns['ParakeetEncoderBlock'](config,i) for i in range(2))
        self.project=ns['SoundProjection'](8,12,6,bias=False)
    def forward(self,x,valid):
        mask=torch.arange(x.shape[1])[None,:]<valid
        x=self.subsampling(x,mask)
        pos=self.encode_positions(x)
        mask=torch.arange(x.shape[1])[None,:]<((valid+7)//8)
        attention=mask[:,None,None,:]&mask[:,None,:,None]
        trace=[x.detach().float().flatten().tolist()]
        for layer in self.layers:x=layer(x,attention_mask=attention,position_embeddings=pos);trace.append(x.detach().float().flatten().tolist())
        return self.project(x),trace
torch.manual_seed(160916)
model=Model().eval()
# Nontrivial normalization buffers expose mean/variance and epsilon mistakes.
for layer in model.layers:
    layer.conv.norm.running_mean.copy_(torch.linspace(-.3,.2,8))
    layer.conv.norm.running_var.copy_(torch.linspace(.6,1.2,8))
base=model.state_dict()
fixtures=[]
for bf16 in [False,True]:
    model=Model().eval();model.load_state_dict(base)
    if bf16:model=model.to(torch.bfloat16)
    weights=[]
    for name,tensor in model.state_dict().items():
        if name.endswith('num_batches_tracked'):continue
        name=('sound_projection.'+name[len('project.'):]) if name.startswith('project.') else 'sound_encoder.encoder.'+name
        weights.append({'name':name,'shape':list(tensor.shape),'values':tensor.float().flatten().tolist()})
    details={}
    def hook(name,module,args,out):
        if isinstance(out,tuple):out=out[0]
        if isinstance(module,(nn.Conv1d,nn.BatchNorm1d)):out=out.transpose(1,2)
        if isinstance(module,nn.Conv2d):out=out.permute(0,2,3,1)
        key=('sound_projection.'+name[len('project.'):]) if name.startswith('project.') else 'sound_encoder.encoder.'+name
        details[key]=out.detach().float().flatten().tolist()
    for name,module in model.named_modules():
        if name:module.register_forward_hook(lambda m,args,out,name=name:hook(name,m,args,out))
    for i,layer in enumerate(model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(lambda m,args,i=i:details.__setitem__('sound_encoder.encoder.layers.'+str(i)+'.self_attn.context',args[0].detach().float().flatten().tolist()))
    cases=[]
    for frames,valid in [(1,1),(8,7),(9,8),(17,16),(18,17),(33,31)]:
        x=torch.sin(torch.arange(frames*8,dtype=torch.float32)*.137).reshape(1,frames,8)
        x[:,valid:]=0
        with torch.no_grad():out,trace=model(x.to(torch.bfloat16 if bf16 else torch.float32),valid)
        cases.append({'frames':frames,'valid_frames':valid,'input':x.flatten().tolist(),'expected':out.float().flatten().tolist(),'trace':trace,'details':dict(details)})
    fixtures.append({'bf16':bf16,'weights':weights,'cases':cases})
result={'transformers_revision':rev,'parakeet_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'nvidia_revision':json.loads((root/'summary.json').read_text())['revision'],'torch_version':torch.__version__,'config':vars(config),'projection_hidden_size':12,'projection_dim':6,'fixtures':fixtures}
(root/'reference.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('fixtures','config')},indent=2));print('reference_bytes',(root/'reference.json').stat().st_size)
