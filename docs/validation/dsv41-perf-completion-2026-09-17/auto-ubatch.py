import ctypes as C, json, os, sys
import numpy as np
os.environ.update(TS_DSV41_TP='0', TS_DSV41_ENGRAM_THREADS='2', TS_DSV41_ENGRAM_WARM='0', TS_DSV41_RETAINED_CACHE='0', TS_DSV41_REWIND_CHECKPOINT='1')
lib=C.CDLL(sys.argv[1])
load=lib.TSGgml_Dsv4LoadModelDspark
load.argtypes=[C.c_char_p]+[C.c_int]*4+[C.c_char_p,C.c_int,C.c_char_p]
load.restype=C.c_void_p
for name in ['UBatch','NPast']:
 fn=getattr(lib,'TSGgml_Dsv4'+name); fn.argtypes=[C.c_void_p]; fn.restype=C.c_int
for name in ['Forward','ForwardSpec']:
 fn=getattr(lib,'TSGgml_Dsv4'+name); fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]; fn.restype=C.c_int
lib.TSGgml_Dsv4Free.argtypes=[C.c_void_p]
lib.TSGgml_Dsv4Free.restype=None
checks=[]
for backend,requested,expected in [(b'CPU',-1,256),(b'CUDA',-1,1024),(b'CUDA',32,32)]:
 h=load(b'/workspace/ts-main/fixtures-managed/dsv41/deepseek41-fixture.gguf',1,1024,requested,2,b'/workspace/ts-main/fixtures-managed/deepseek41-dspark-fixture.gguf',0,backend)
 assert h, (backend,requested,'load failed')
 try:
  actual=lib.TSGgml_Dsv4UBatch(h)
  assert actual==expected,(backend,actual,expected)
  seed=np.array([0,15,32,64,128],dtype=np.int32); output=np.empty(256,dtype=np.float32)
  assert lib.TSGgml_Dsv4Forward(h,seed.ctypes.data,len(seed),output.ctypes.data)==0
  verify=np.array([41,43,47],dtype=np.int32); rows=np.empty(3*256,dtype=np.float32)
  assert lib.TSGgml_Dsv4ForwardSpec(h,verify.ctypes.data,len(verify),rows.ctypes.data)==1
  assert np.isfinite(rows).all() and lib.TSGgml_Dsv4NPast(h)==8
  checks.append(dict(backend=backend.decode(),requested=requested,actual=actual,verify_rows=3,head=8,passed=True))
 finally: lib.TSGgml_Dsv4Free(h)
print(json.dumps(checks,indent=2))
