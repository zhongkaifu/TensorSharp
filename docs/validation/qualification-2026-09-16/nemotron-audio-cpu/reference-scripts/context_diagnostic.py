import json,pathlib,torch,math
r=json.loads(pathlib.Path('/workspace/ts-nemotron-audio-research/reference.json').read_text())['fixtures'][1]
c=next(c for c in r['cases'] if c['frames']==17)
p='sound_encoder.encoder.layers.0.self_attn.'
d=c['details'];w={v['name']:torch.tensor(v['values']).reshape(v['shape']) for v in r['weights']}
q=torch.tensor(d[p+'q_proj']).reshape(3,2,4);k=torch.tensor(d[p+'k_proj']).reshape(3,2,4);v=torch.tensor(d[p+'v_proj']).reshape(3,2,4);rk=torch.tensor(d[p+'relative_k_proj']).reshape(5,2,4)
bf=lambda x:x.to(torch.bfloat16).float()
qu=bf(q+w[p+'bias_u']);qb=bf(q+w[p+'bias_v'])
ctx=torch.zeros_like(q)
for t in range(2):
 for h in range(2):
  scores=[]
  for s in range(2):
   content=torch.sum(qu[t,h]*k[s,h]);rel=torch.sum(qb[t,h]*rk[2-t+s,h]);scores.append(content*.5+bf(bf(rel)*.5))
  probs=torch.softmax(torch.stack(scores),0)
  ctx[t,h]=bf(probs@v[:2,h])
  print('t/head',t,h,'scores',scores,'probs',probs.tolist())
print('expected',d[p+'context']);print('manual',ctx.flatten().tolist())
print('u',w[p+'bias_u']);print('v',w[p+'bias_v'])

qv=qb.permute(1,0,2)[None].bfloat16();rr=rk.permute(1,2,0)[None].bfloat16()
bd=qv@rr
bd=torch.nn.functional.pad(bd,(1,0)).view(1,2,-1,3)[:,:,1:].view(1,2,3,5)[...,:3]*.5
print("shift bd",bd.float().tolist())
mask=torch.tensor([1,1,0],dtype=torch.bool);bd=bd.masked_fill(~(mask[None,None,None,:]&mask[None,None,:,None]),-math.inf)
actual=torch.nn.functional.scaled_dot_product_attention(qu.permute(1,0,2)[None].bfloat16(),k.permute(1,0,2)[None].bfloat16(),v.permute(1,0,2)[None].bfloat16(),bd,scale=.5)
print("sdpa",actual.transpose(1,2).float().flatten().tolist())
content=(qu.permute(1,0,2)[None]@k.permute(1,2,0)[None])*.5
probs=torch.softmax(content+bd.float(),-1).nan_to_num(0)
plain=bf(probs@v.permute(1,0,2)[None]);rounded=(probs.bfloat16()@v.permute(1,0,2)[None].bfloat16())
print('content',content.tolist());print('probs',probs.tolist());print('plainfull',plain.transpose(1,2).flatten().tolist());print('rounded',rounded.transpose(1,2).float().flatten().tolist())
scores=content+bd.float();num=torch.exp(scores-scores.max(-1,keepdim=True).values).nan_to_num(0)
flash=bf((bf(num)@v.permute(1,0,2)[None])/num.sum(-1,keepdim=True)).nan_to_num(0)
print('flashsimulate',flash.transpose(1,2).flatten().tolist())
