from pathlib import Path
import subprocess,json,os,hashlib,datetime
root=Path("/workspace/ts-tp-chosen-prototype-20260916")
rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
free={int(a):int(b) for a,b in (x.split(",") for x in rows.splitlines())}
assert free[0]>=8192 and free[6]>=8192,free
exe=root/"integrated-test"
record={"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"physical_gpus":[0,6],"free_mib_before":free,"binary_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),"shared_load":True,"performance_qualified":False}
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="0,6"
env.pop("GGML_CUDA_DISABLE_FUSION",None);env.pop("GGML_CUDA_DISABLE_GRAPHS",None)
command=[str(exe),"--cuda","2","--checkpoint-shape"]
with (root/"physical-cuda2-r1.log").open("w") as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(command=command,exit_code=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/"physical-cuda2-r1.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record));print("\n".join((root/"physical-cuda2-r1.log").read_text().splitlines()[-5:]))
raise SystemExit(result.returncode)
