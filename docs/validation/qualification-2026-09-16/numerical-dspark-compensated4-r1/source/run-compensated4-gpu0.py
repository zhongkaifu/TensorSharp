from pathlib import Path
import subprocess,os,json,hashlib,datetime
root=Path('/workspace/ts-dspark-sum-variants-20260916')
repo=Path('/workspace/ts-codex-20260916-r3/repo')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/workspace/tensorsharp-no-patch-20260915/fixture-python-deps')
variant='compensated4'
commands={
 'prefix1-17':['/usr/local/bin/python',str(repo/'eng/tests/dsv41-dspark-inference.py'),'/workspace/ts-main/repo/fixtures/text-f32-small','--out',str(root/(variant+'-prefix1-17')),'--library',str(root/(variant+'.so')),'--backend','CUDA','--gpus','1','--reference-only','--reference-prefixes',','.join(map(str,range(1,18))),'--keep-going'],
 'precision':[str(root/(variant+'-precision-test'))],
 'benchmark':[str(root/(variant+'-benchmark'))]}
results={}
for name,cmd in commands.items():
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (root/(variant+'-'+name+'-r1.log')).open('w') as log:
  code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 results[name]={'exit_code':code,'command':cmd,'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 print(json.dumps({name:results[name]}),flush=True)
 if name=='prefix1-17':
  report=json.loads((root/(variant+'-prefix1-17')/'report.json').read_text())
  print(json.dumps({k:report.get(k) for k in ('status','native_sha256','failed_checks')}),flush=True)
(root/(variant+'-run.json')).write_text(json.dumps(results,indent=2)+'\n')
