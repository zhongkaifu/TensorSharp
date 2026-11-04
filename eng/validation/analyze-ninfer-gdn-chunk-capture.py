#!/usr/bin/env python3
"""Read-only first-sparse GDN stage analysis; never changes acceptance scores."""
import argparse, hashlib, json, pathlib
import numpy as np


def bf16(x):
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def raw_tf32(x):
    # Diagnostic model of raw FP32 bit operands consumed as TF32. This is an
    # explicitly labelled hypothesis, not a promise about arbitrary SDK/device semantics.
    return (np.asarray(x, dtype=np.float32).view(np.uint32) & np.uint32(0xffffe000)).view(np.float32)


def stats(actual, reference):
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite or inconsistent stage arrays')
    d = a-r
    return {'count': int(a.size), 'relative_l2': float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(r.ravel()), 1e-30)),
            'max_absolute_error': float(np.max(np.abs(d))), 'max_absolute_reference': float(np.max(np.abs(r)))}


def exact_bits(actual, expected):
    a, e = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    return {'numeric_different': int(np.count_nonzero(a != e)),
            'bit_different': int(np.count_nonzero(a.view(np.uint32) != e.view(np.uint32)))}


def analyze(folder):
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'ninfer-first-sparse-stage-capture/1' or manifest['rows'] != 64:
        raise ValueError('Expected the first 64-token sparse capture')
    metadata = {x['name']: x for x in manifest['files']}
    for name, item in metadata.items():
        path = folder / name
        if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Capture SHA/size mismatch: '+name)
    def read(name):
        item=metadata[name]
        dtype='<f4' if item['dtype']=='float32-le' else '<u2'
        result=np.fromfile(folder/name,dtype=dtype).reshape(item['shape'])
        return (result.astype(np.uint32)<<16).view(np.float32) if dtype=='<u2' else result
    q,k,v,U,Wphysical,Vnew,H,gc=[read(x) for x in ['query.bf16','key.bf16','value.bf16','U.bf16',
        'W-original-layout.bf16','v_new.bf16','h_chunk.bf16','g_cumsum.f32']]
    qkv,g,beta,initial,final,output=[read(x) for x in ['input-qkv.f32','decay.f32','beta.f32','initial-state.f32','final-state.f32','output.f32']]
    t,hv,d=64,48,128
    # Recover the producer's private groups-of-eight W order. The XOR swizzle
    # belongs to shared-memory staging, not to this contiguous global payload.
    W=Wphysical.reshape(t,hv,d//8,8)[...,[0,2,4,6,1,3,5,7]].reshape(t,hv,d)
    expected_q=np.zeros_like(q);expected_k=np.zeros_like(k)
    for head in range(16): expected_q[:,head,head]=1;expected_k[:,head,head]=1
    checks={'query_onehot':exact_bits(q,expected_q),'key_onehot':exact_bits(k,expected_k),
        'value_bridge':exact_bits(v,bf16(qkv[:,4096:].reshape(t,hv,d))),
        'h_chunk_initial_bf16':exact_bits(H[0],bf16(initial)),
        'cumulative_decay':exact_bits(gc,np.cumsum(g,dtype=np.float32,axis=0))}
    if any(x['numeric_different'] for x in checks.values()):
        raise ValueError('Captured representation does not match the intended sparse fixture: '+str(checks))
    expected_beta=np.array([.25+(h%3)*.125 for h in range(hv)],np.float32)
    if not np.array_equal(g,np.full_like(g,-.03125)) or not np.array_equal(beta,np.broadcast_to(expected_beta,beta.shape)):
        raise ValueError('Unexpected sparse gate inputs')
    logical_initial=np.zeros((hv,d),np.float64)
    for head in range(hv):logical_initial[head]=initial[head,:,head//3]
    masked=initial.copy()
    for head in range(hv):masked[head,:,head//3]=0
    if np.count_nonzero(masked):raise ValueError('Sparse reference requires zero other initial key coordinates')
    scale=np.float64(np.float32(.08838834764831845))
    real_output=np.zeros((t,hv,d),np.float64)
    exact_u=np.zeros((t,hv,d),np.float64)
    exact_w=np.zeros((t,hv,d),np.float64)
    exact_vnew=np.zeros((t,hv,d),np.float64)
    s=logical_initial.copy();zero_state=np.zeros_like(s)
    for token in range(t):
        decay=np.exp(g[token].astype(np.float64))[:,None];b=beta[token].astype(np.float64)[:,None]
        delta=b*(v[token].astype(np.float64)-decay*s)
        s=decay*s+delta;real_output[token]=scale*s;exact_vnew[token]=delta
        u=b*(v[token].astype(np.float64)-decay*zero_state)
        zero_state=decay*zero_state+u;exact_u[token]=u
        for head in range(hv):
            exact_w[token,head,head//3]=float(beta[token,head])*np.exp(float(gc[token,head]))*(1-float(beta[token,head]))**token
    true_state=np.zeros_like(initial,dtype=np.float64)
    for head in range(hv):true_state[head,:,head//3]=s[head]
    w_logical=np.empty((t,hv),np.float32)
    outside_w=W.copy()
    for head in range(hv):w_logical[:,head]=W[:,head,head//3];outside_w[:,head,head//3]=0
    # In the first chunk the one-hot fixture reduces W @ state to exactly one
    # represented product. These initial state values are already TF32-exact.
    vraw=np.asarray(U-w_logical[:,:,None]*raw_tf32(logical_initial)[None,:,:],dtype=np.float32)
    stage_checks={'U_vs_fp64_delta_zero_initial':stats(U,exact_u),
        'U_vs_BF16_fp64_delta_zero_initial':stats(U,bf16(exact_u)),
        'W_vs_fp64_transform':stats(W,exact_w),
        'W_vs_BF16_fp64_transform':stats(W,bf16(exact_w)),
        'W_outside_expected_onehot_coordinate':int(np.count_nonzero(outside_w)),
        'Vnew_vs_fp64_delta':stats(Vnew,exact_vnew),
        'Vnew_vs_reconstructed_captured_UW':exact_bits(Vnew,bf16(vraw))}
    reconstructed_state=np.zeros_like(initial,dtype=np.float64)
    reconstructed_state_tf32=np.zeros_like(initial,dtype=np.float64)
    reconstructed_output=np.zeros_like(real_output)
    reconstructed_output_tf32=np.zeros_like(real_output)
    exp_gc=np.exp(gc.astype(np.float64))
    for head in range(hv):
        weights=np.exp(float(gc[-1,head])-gc[:,head].astype(np.float64))
        pred=exp_gc[-1,head]*logical_initial[head]+np.einsum('t,td->d',weights,vraw[:,head].astype(np.float64))
        vd=np.asarray(np.asarray(weights,dtype=np.float32)[:,None]*vraw[:,head],dtype=np.float32)
        pred_tf=exp_gc[-1,head]*logical_initial[head]+np.sum(raw_tf32(vd).astype(np.float64),axis=0)
        reconstructed_state[head,:,head//3]=pred
        reconstructed_state_tf32[head,:,head//3]=pred_tf
        for token in range(t):
            attn=np.exp(float(gc[token,head])-gc[:token+1,head].astype(np.float64))
            prior=exp_gc[token,head]*H[0,head,:,head//3].astype(np.float64)
            pred=scale*(prior+np.einsum('t,td->d',attn,Vnew[:token+1,head].astype(np.float64)))
            pred_tf=scale*(prior+np.einsum('t,td->d',raw_tf32(attn).astype(np.float64),Vnew[:token+1,head].astype(np.float64)))
            reconstructed_output[token,head]=bf16(pred)
            reconstructed_output_tf32[token,head]=bf16(pred_tf)
    # Diagnostic counterfactuals preserve the captured input. They do not alter
    # any original component scores and do not model an implemented kernel.
    def downstream(U_,W_,v_round,attention_mode,state_mode):
        wi=np.empty((t,hv),np.float32)
        for head in range(hv):wi[:,head]=W_[:,head,head//3]
        raw=np.asarray(U_-wi[:,:,None]*logical_initial.astype(np.float32)[None,:,:],dtype=np.float32)
        vo=bf16(raw) if v_round else raw
        out=np.zeros_like(real_output);st=np.zeros_like(true_state)
        def operand(x,mode):
            x=np.asarray(x,np.float32)
            if mode=='raw_tf32':return raw_tf32(x)
            if mode=='rn_tf32':
                u=x.view(np.uint32)
                return ((u+np.uint32(4095)+((u>>13)&1)) & np.uint32(0xffffe000)).view(np.float32)
            return x
        for head in range(hv):
            weights=np.exp(float(gc[-1,head])-gc[:,head].astype(np.float64))
            vd=np.asarray(weights.astype(np.float32)[:,None]*raw[:,head],np.float32)
            st[head,:,head//3]=exp_gc[-1,head]*logical_initial[head]+np.sum(operand(vd,state_mode).astype(np.float64),axis=0)
            for token in range(t):
                attn=np.exp(float(gc[token,head])-gc[:token+1,head].astype(np.float64))
                prior=exp_gc[token,head]*logical_initial[head]
                val=prior+np.einsum('t,td->d',operand(attn,attention_mode).astype(np.float64),vo[:token+1,head].astype(np.float64))
                out[token,head]=bf16(scale*val)
        return {'output_vs_true':stats(out,real_output),'state_vs_true':stats(st,true_state)}
    counterfactual={
        'captured_UW_BF16vnew_rawTF32_downstream':downstream(U,W,True,'raw_tf32','raw_tf32'),
        'captured_UW_BF16vnew_RN_TF32_downstream':downstream(U,W,True,'rn_tf32','rn_tf32'),
        'captured_UW_F32vnew_F32_downstream':downstream(U,W,False,'f32','f32'),
        'exact_transform_rounded_BF16_original_downstream':downstream(bf16(exact_u),bf16(exact_w),True,'raw_tf32','raw_tf32'),
        'exact_transform_rounded_F32_BF16vnew_F32_downstream':downstream(exact_u.astype(np.float32),exact_w.astype(np.float32),True,'f32','f32'),
        'exact_transform_rounded_F32_F32vnew_F32_downstream':downstream(exact_u.astype(np.float32),exact_w.astype(np.float32),False,'f32','f32')}
    return {'schema':'ninfer-first-sparse-stage-analysis/1',
        'status':'DIAGNOSTIC_ONLY; original acceptance results are unchanged',
        'capture_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'capture_module_sha256':manifest['moduleImageSha256'],'capture_files_sha_verified':len(metadata),
        'represented_inputs':checks,'stages':stage_checks,
        'actual_original_failed_outputs':stats(output,real_output),
        'actual_original_failed_state':stats(final,true_state),
        'state_reconstruction':{'from_captured_UW_real_weighted_accum':stats(final,reconstructed_state),
            'from_captured_UW_raw_tf32_vdecay_hypothesis':stats(final,reconstructed_state_tf32)},
        'diagnostic_counterfactuals_not_implemented':counterfactual,
        'output_reconstruction':{'from_captured_BF16_vnew_real_attention':stats(output,reconstructed_output),
            'from_captured_BF16_vnew_raw_tf32_attention_hypothesis':stats(output,reconstructed_output_tf32),
            'raw_tf32_predicted_bf16_bits':exact_bits(output,reconstructed_output_tf32)},
        'limits':['Closed-form analysis applies only to this captured one-hot first chunk.',
            'Raw TF32 masking is a diagnostic operand hypothesis, not an actual rerun or general device guarantee.',
            'exp2_approx and exact tensor-core accumulation order are not reproduced by this CPU script.',
            'This script does not promote, rescore or waive any original failed component or model-quality gate.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('capture',type=pathlib.Path);parser.add_argument('--output',required=True,type=pathlib.Path)
    args=parser.parse_args();result=analyze(args.capture)
    if args.output.exists():raise FileExistsError('Refusing to overwrite an earlier analysis')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
