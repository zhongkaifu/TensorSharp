from pathlib import Path
import subprocess,os,json,hashlib
root=Path("/workspace/ts-dspark-sum-variants-20260916")
repo=Path("/workspace/ts-codex-20260916-r3/repo")
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="6",PYTHONPATH="/workspace/tensorsharp-no-patch-20260915/fixture-python-deps")
results={}
for name in ["indexed_f64","vector_f64","compensated"]:
 output=root/(name+"-prefix5")
 command=["/usr/local/bin/python",str(repo/"eng/tests/dsv41-dspark-inference.py"),"/workspace/ts-main/repo/fixtures/text-f32-small","--out",str(output),"--library",str(root/(name+".so")),"--backend","CUDA","--gpus","1","--reference-only","--reference-prefixes","5","--trace-native","--native-ring-control","--keep-going"]
 with (root/(name+"-prefix5.log")).open("w") as log:code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 report=json.loads((output/"report.json").read_text())
 results[name]={"exit_code":code,"status":report["status"],"failed_checks":report.get("failed_checks",[]),"native_sha256":report["native_sha256"],"command":command}
 print(json.dumps(results[name]|{"variant":name}),flush=True)
(root/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
