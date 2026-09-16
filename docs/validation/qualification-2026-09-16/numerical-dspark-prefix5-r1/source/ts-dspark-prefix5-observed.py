from pathlib import Path
import importlib.util,json
import numpy as np
import torch
root=Path("/workspace/ts-numerical-r3-prefix5-cuda-trace-20260916")
repo=Path("/workspace/ts-codex-20260916-r3/repo")
spec=importlib.util.spec_from_file_location("ds",repo/"eng/dsv41-dspark-reference.py");ds=importlib.util.module_from_spec(spec);spec.loader.exec_module(ds)
fixture=Path("/workspace/ts-main/repo/fixtures/text-f32-small")
c=json.loads((fixture/"deepseek41.config.json").read_text())["config"]
w=ds.reference.GgufWeights(root/"deepseek41-dspark-fixture.gguf")
tw=ds.reference.GgufWeights(fixture/"deepseek41-fixture.gguf")
epsilon=np.longdouble(c["text_config"]["rms_norm_eps"])
o=np.load(root/"committed-inputs-prefix5.npz")
values={"native_features":np.fromfile(root/"native-trace/p000000_v41.00.dspark_features.f32",np.float32).reshape(o["features"].shape), "oracle_features":o["features"],"native_main":np.fromfile(root/"native-trace/p000000_v41.00.dspark_main_normalized.f32",np.float32).reshape(o["main"].shape),"oracle_main":o["main"]}
math=ds.reference.Reference(tw,c,{},"model")
def linear_norm(x,projection,norm):
    weights=w.weight(projection).numpy().astype(np.longdouble)
    linear=(weights*x[:,None,:].astype(np.longdouble)).sum(-1,dtype=np.longdouble)
    return linear/np.sqrt((linear*linear).mean(-1,keepdims=True,dtype=np.longdouble)+epsilon)*w.weight(norm).numpy().reshape(-1).astype(np.longdouble)
def kv(x):
    z=linear_norm(x,"mtp.1.attn_kv.weight","mtp.1.attn_kv_a_norm.weight")
    dim=c["text_config"]["qk_rope_head_dim"];base=np.longdouble(c["text_config"]["rope_theta"])
    angles=np.arange(len(z),dtype=np.longdouble)[:,None]/(base**(np.arange(0,dim,2,dtype=np.longdouble)/dim))
    pairs=z[:,-dim:].reshape(len(z),dim//2,2).copy()
    z[:,-dim::2]=pairs[:,:,0]*np.cos(angles)-pairs[:,:,1]*np.sin(angles)
    z[:,-dim+1::2]=pairs[:,:,0]*np.sin(angles)+pairs[:,:,1]*np.cos(angles)
    return z
def details(x):
    v=x[2,35];t=torch.tensor(np.asarray(x,np.float32))
    return {"prequant":str(v),"bf16":float(t.bfloat16()[2,35]),"cache":float(math.cache(t)[2,35])}
result={}
for name,value in list(values.items()):
    if name.endswith("features"):
        main=linear_norm(value,"mtp.0.main_proj.weight","mtp.0.main_norm.weight")
        result[name+"_main_error"]={"native":float(np.max(np.abs(main-values["native_main"]))),"oracle":float(np.max(np.abs(main-values["oracle_main"])))}
        result[name+"_longdouble_main_kv"]=details(kv(main))
    else:result[name+"_longdouble_kv"]=details(kv(value))
native=np.fromfile(root/"native-trace/p000000_v41.01.dspark_committed_prequant.f32",np.float32).reshape(o["stage1"].shape)
result["recorded_native"]=details(native);result["recorded_oracle"]=details(o["stage1"])
result["max_features_abs"]=float(np.max(np.abs(values["native_features"]-values["oracle_features"])))
result["max_main_abs"]=float(np.max(np.abs(values["native_main"]-values["oracle_main"])))
print(json.dumps(result,indent=2))
