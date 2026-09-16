import hashlib, json, os, pathlib, shutil, subprocess

root = pathlib.Path(__file__).resolve().parent
frozen = pathlib.Path('/workspace/ts-codex-20260916-r4/repo/InferenceWeb.Tests/bin/Release/net10.0')
sdk = '/workspace/tensorsharp-no-patch-20260915/dotnet/dotnet'
app = root / 'app'
project = root / 'Derived.csproj'
sources = sorted(root.glob('*.cs'))
project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><AssemblyName>InferenceWeb.Tests</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>annotations</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AllowUnsafeBlocks>true</AllowUnsafeBlocks></PropertyGroup><ItemGroup><Using Include="Xunit"/><FrameworkReference Include="Microsoft.AspNetCore.App"/>' + ''.join('<Compile Include="'+p.name+'"/>' for p in sources) + ''.join('<Reference Include="'+p.stem+'"><HintPath>'+str(p)+'</HintPath><Private>false</Private></Reference>' for p in frozen.glob('*.dll') if p.name != 'InferenceWeb.Tests.dll') + '</ItemGroup></Project>')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', TS_TEST_GGML_BACKEND='cpu', TS_TEST_GLM_CUDA='1', TS_TEST_GLM_SNAPSHOT_BOUNDARY='1')
build = [sdk,'build',str(project),'-c','Release','-o',str(root/'build'),'-m:1','/nodeReuse:false']
with (root/'build.log').open('w') as log:
    status = subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
if status:
    print((root/'build.log').read_text()[-12000:]); raise SystemExit(status)
shutil.copytree(frozen, app, dirs_exist_ok=True)
shutil.copy2(root/'build'/'InferenceWeb.Tests.dll', app/'InferenceWeb.Tests.dll')
shutil.copy2(root/'build'/'InferenceWeb.Tests.pdb', app/'InferenceWeb.Tests.pdb')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(app/'libGgmlOps.so') == '7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7'
command = [sdk,'vstest',str(app/'InferenceWeb.Tests.dll'),'--Logger:trx','--ResultsDirectory:'+str(root/'results')]
with (root/'run.log').open('w') as log:
    status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
manifest = {'native_sha256':digest(app/'libGgmlOps.so'), 'build_command':build, 'run_command':command,'run_exit_code':status,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','TS_TEST_GGML_BACKEND','TS_TEST_GLM_CUDA','TS_TEST_GLM_SNAPSHOT_BOUNDARY']},'source_sha256':{p.name:digest(p) for p in [project,*sources]},'assembly_sha256':{p.name:digest(p) for p in app.glob('*.dll') if p.name.startswith(('InferenceWeb','TensorSharp','AdvUtils'))}}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print((root/'run.log').read_text()[-18000:])
raise SystemExit(status)
