import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/workspace/ts-codex-20260916-r4');out=root/'history-probe-r1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=out/'source';src.mkdir();shutil.copy2(root/'QwenHistoryProbe.cs',src/'Program.cs')
base=root/'qwen-trained-r1';app=base/'app';refs=''.join('<Reference Include="'+p.stem+'"><HintPath>'+str(p)+'</HintPath></Reference>' for p in sorted(app.glob('*.dll')) if p.stem.startswith(('TensorSharp.','AdvUtils')) and p.stem not in ('TensorSharp.Server.Host',))
(src/'Probe.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><OutputType>Exe</OutputType><AssemblyName>QwenHistoryProbe</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup>'+refs+'<FrameworkReference Include="Microsoft.AspNetCore.App" /></ItemGroup></Project>')
dotnet='/workspace/tensorsharp-no-patch-20260915/dotnet/dotnet'
with (out/'build.log').open('x') as log:subprocess.run([dotnet,'build',str(src/'Probe.csproj'),'-c','Release','-o',str(out/'build')],stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.copytree(app,out/'app')
for p in (out/'build').iterdir():
 if p.is_file() and p.name.startswith('QwenHistoryProbe'):shutil.copy2(p,out/'app'/p.name)
import sys
sys.path.insert(0,str(root/'repo/eng/validation'))
from release_application_identity import make_manifest,digest
managed=root/'results/build1/managed-build.json';manifest=out/'application.json';manifest.write_text(json.dumps(make_manifest(out/'app',managed,digest(managed)),indent=2)+'\n')
plan=json.loads((base/'mtp.plan.json').read_text());plan.update(scope='Successive committed target-block history versus scalar full-vocabulary teacher forcing for two trained mismatches. No scheduler/grammar replay; shared VM and no performance qualification.',application=str(out/'app'),application_manifest=str(manifest),application_manifest_sha256=sha(manifest),diagnostic_source_sha256=sha(src/'Program.cs'))
for name in ('plain.rows.json','plain.rows.json.inputs.json','mtp.rows.json'):plan['assets_sha256'][str(base/name)]=sha(base/name)
plan['argv']=[dotnet,str(out/'app/QwenHistoryProbe.dll'),plan['argv'][3],'/workspace/models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',str(base/'plain.rows.json'),str(base/'plain.rows.json.inputs.json'),str(base/'mtp.rows.json'),str(out/'probe')]
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(sha(out/'plan.json'))
