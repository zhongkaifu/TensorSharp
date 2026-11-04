#!/usr/bin/env python3
"""CPU NumPy FP64 full-output oracle; streams original BF16 weights, never stores them."""
import argparse,datetime,hashlib,json,os,pathlib,sys
# Bound BLAS thread resources before importing NumPy. No torch/CUDA import.
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='4'
import numpy as np
U32=2.0**-24

def need(b,m):
 if not b:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def pin(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def check(p):need(pin(pathlib.Path(p['path']))==p,'Pinned prepared data changed')
def source_row(r):return r//512*256+r%256+(7168 if r%512>=256 else 0)if r<12288 else 6144+r-12288 if r<13312 else r
def synthetic(kind,r,k):
 rr=r[:,None];kk=np.arange(k,dtype=np.int64)[None,:]
 if kind=='dense':v=((rr*17+kk*13)%31-15).astype(np.float32)/np.float32(32)
 elif kind=='cancellation':
  v=((rr*17+(kk//2)*13)%31-15).astype(np.float32)/np.float32(32);v[r==0,:]=np.float32(-0.)
 else:
  v=np.zeros((len(r),k),dtype=np.float32);sg=np.where(r%2==0,1.,-1.).astype(np.float32);v[:,0]=sg;v[:,1]=sg*(np.float32(2**-8)+np.where(r%3==0,0.,np.where(r%3==1,2**-15,-2**-15)).astype(np.float32));v[:,-1]=np.float32(2**16);v[:,-2]=np.float32(-2**-16)
 bits=v.view(np.uint32);need(np.all((bits&65535)==0),'Synthetic weight must be exact BF16');return(bits>>16).astype('<u2')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepared',type=pathlib.Path,required=True);ap.add_argument('--expect-prepared-sha',required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args();need(sha(a.prepared)==a.expect_prepared_sha,'Prepared manifest SHA differs');prepared_pin=pin(a.prepared);p=json.loads(a.prepared.read_bytes());need(p['schema']=='ninfer-bf16-small-t-prepared/1'and p['generatorPolicy']=='bf16-smallt-fixtures-v1','Unknown source');out=a.out;need(out.is_absolute()and not out.exists(),'Fresh absolute oracle output required');need('artifacts'in out.parts or any(out.parts[i:i+2]==('docs','validation')for i in range(len(out.parts)-1)),'Ignored output required')
 artifact=pathlib.Path(p['artifact']);st=artifact.stat();need(st.st_size==p['artifactStat']['bytes']and st.st_mtime_ns//100+621355968000000000==p['artifactStat']['mtimeTicks'],'Original artifact stat changed');out.mkdir();cases=[]
 with artifact.open('rb',buffering=0)as f:
  for c in p['cases']:
   for role in ('input','seed','norm'):check(c[role])
   n,k=(14336,5120)if c['shape']=='input'else(5120,6144);x=np.fromfile(c['input']['path'],dtype='<f4').reshape(8,k);need(np.isfinite(x).all()and np.all((x.view(np.uint32)&65535)==0),'Complete represented BF16 inputs required');xd=x.astype(np.float64);ideal=np.empty((8,n),np.float64);l1=np.empty_like(ideal);mapped=hashlib.sha256();original=hashlib.sha256()
   if c['kind']=='real':
    f.seek(c['absoluteOffset']);remaining=c['weightBytes']
    while remaining:
     b=f.read(min(8<<20,remaining));need(bool(b),'Truncated original tensor');original.update(b);remaining-=len(b)
   for first in range(0,n,256):
    stop=min(n,first+256);r=np.arange(first,stop,dtype=np.int64)
    if c['kind']=='real':
     src=source_row(first)if c['permuted']else first;f.seek(c['absoluteOffset']+src*k*2);raw=f.read((stop-first)*k*2);need(len(raw)==(stop-first)*k*2,'Truncated BF16 rows');words=np.frombuffer(raw,dtype='<u2').reshape(stop-first,k)
    else:words=synthetic(c['kind'],r,k);raw=words.tobytes()
    mapped.update(raw);w=(words.astype(np.uint32)<<16).view(np.float32);need(np.isfinite(w).all(),'Nonfinite BF16 weights');wd=w.astype(np.float64);ideal[:,first:stop]=xd@wd.T;l1[:,first:stop]=np.abs(xd)@np.abs(wd).T
   need(np.isfinite(ideal).all()and np.isfinite(l1).all(),'Nonfinite FP64 oracle');ip=out/(c['id']+'-ideal.f64');lp=out/(c['id']+'-l1.f64');ideal.astype('<f8').tofile(ip);l1.astype('<f8').tofile(lp);cases.append(dict(id=c['id'],weightSha256=mapped.hexdigest(),originalWeightSha256=original.hexdigest()if c['kind']=='real'else mapped.hexdigest(),ideal=pin(ip),l1=pin(lp)));print(json.dumps(dict(prepared=c['id'],n=n,k=k,fullOutputValues=8*n)),flush=True)
 now=artifact.stat();need(now.st_size==st.st_size and now.st_mtime_ns==st.st_mtime_ns and now.st_ino==st.st_ino,'Artifact changed during CPU oracle preparation');check(prepared_pin)
 manifest=dict(schema='ninfer-bf16-small-t-oracle/1',prepared=prepared_pin,cases=cases,numpyVersion=np.__version__,python=sys.version,threadEnvironment={k:os.environ[k]for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},bound='gamma(K+64,u32)*sumAbs + 32*K*u64*sumAbs + K*2^-149; full unrounded F32 before consumer. No tuning to GPU results.',GPU='NOT_RUN',sourceSha256=sha(pathlib.Path(__file__)),generatorPolicy=p['generatorPolicy'],created=datetime.datetime.now(datetime.timezone.utc).isoformat());q=out/'oracle.json';q.write_text(json.dumps(manifest,indent=2)+'\n');q.with_name(q.name+'.sha256').write_text(sha(q)+'  oracle.json\n');print(json.dumps(pin(q)))
if __name__=='__main__':main()
