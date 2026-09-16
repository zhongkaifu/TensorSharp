from pathlib import Path
import shutil,json,hashlib,datetime
root=Path("/workspace/ts-dspark-sum-variants-20260916")
out=root/"artifacts-pair4-r1";out.mkdir(exist_ok=False)
(out/"source").mkdir()
for p in root.iterdir():
 if p.suffix in (".cpp",".cu",".py"):shutil.copy2(p,out/"source"/p.name)
 if p.suffix==".log":shutil.copy2(p,out/p.name)
for name in ["pair4-prefix1-17"]:
 dest=out/name;dest.mkdir()
 for p in (root/name).iterdir():
  if p.suffix==".json" or p.name.startswith("check-"):shutil.copy2(p,dest/p.name)
repo=Path("/workspace/ts-codex-20260916-r3/repo")
shutil.copy2(repo/"TensorSharp.GGML.Native/ggml_ops_matmul_precision.cu",out/"source/original.cu")
identity={"recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_directory":str(root),"upstream_revision":"456172ec733a135778adcd32d00e576a58232e45","binaries_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix==".so" or p.name in ("pair4-benchmark","pair4-precision-test")},"release_qualified":False,"remaining_original_failure":"See retained failed_checks in pair4-prefix1-17/report.json"}
(out/"identity.json").write_text(json.dumps(identity,indent=2)+"\n")
print(str(out))
