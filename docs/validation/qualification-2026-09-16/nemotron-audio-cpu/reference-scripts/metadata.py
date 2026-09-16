import json, pathlib, urllib.request
root=pathlib.Path(__file__).resolve().parent
model='nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16'
def get(url):
    with urllib.request.urlopen(url, timeout=60) as r: return r.read()
info=json.loads(get('https://huggingface.co/api/models/'+model))
revision=info['sha']
for name in ['config.json','model.safetensors.index.json','audio_model.py','modeling.py','processing.py']:
    try: (root/name).write_bytes(get('https://huggingface.co/'+model+'/resolve/'+revision+'/'+name))
    except Exception as exc: print(name,str(exc),flush=True)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
weights={k:v for k,v in index.items() if k.startswith(('sound_encoder.','sound_projection.'))}
summary={'model':model,'revision':revision,'weights':weights,'shards':sorted(set(weights.values()))}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'revision':revision,'audio_weight_count':len(weights),'shards':summary['shards'],'first_names':list(weights)[:30]},indent=2))
