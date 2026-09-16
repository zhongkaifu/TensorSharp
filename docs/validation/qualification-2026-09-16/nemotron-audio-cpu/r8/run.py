import hashlib,json,os,pathlib,shutil,subprocess
root=pathlib.Path(__file__).resolve().parent
base=pathlib.Path('/workspace/ts-codex-20260916-r4/repo/InferenceWeb.Tests/bin/Release/net10.0')
app=root/'app';shutil.copytree(base,app,dirs_exist_ok=True)
shutil.copy2(root/'TensorSharp.Models.dll',app/'TensorSharp.Models.dll')
project=root/'Derived.csproj'
project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><AssemblyName>InferenceWeb.Tests</AssemblyName><ImplicitUsings>enable</ImplicitUsings><Nullable>annotations</Nullable><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><Using Include="Xunit"/><FrameworkReference Include="Microsoft.AspNetCore.App"/><EmbeddedResource Include="reference.json" LogicalName="InferenceWeb.Tests.Fixtures.NemotronAudio.reference.json"/><EmbeddedResource Include="reference128.json" LogicalName="InferenceWeb.Tests.Fixtures.NemotronAudio.reference128.json"/>'+''.join('<Compile Include="'+p.name+'"/>' for p in root.glob('*.cs') if p.name != 'Probe.cs')+''.join('<Reference Include="'+p.stem+'"><HintPath>'+str(p)+'</HintPath><Private>false</Private></Reference>' for p in app.glob('*.dll') if p.name!='InferenceWeb.Tests.dll')+'</ItemGroup></Project>')
dotnet='/workspace/tensorsharp-no-patch-20260915/dotnet/dotnet'
with (root/'build.log').open('w') as f:status=subprocess.run([dotnet,'build',str(project),'-c','Release','-o',str(root/'build')],stdout=f,stderr=subprocess.STDOUT).returncode
if status:print((root/'build.log').read_text()[-6000:]);raise SystemExit(status)
shutil.copy2(root/'build'/'InferenceWeb.Tests.dll',app/'InferenceWeb.Tests.dll')
env=dict(os.environ,TS_TEST_GGML_BACKEND='cpu')
command=[dotnet,'vstest',str(app/'InferenceWeb.Tests.dll'),'--Logger:trx','--ResultsDirectory:'+str(root/'results')]
with (root/'run.log').open('w') as log:status=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'manifest.json').write_text(json.dumps({'command':command,'exit_code':status,'environment':{'TS_TEST_GGML_BACKEND':'cpu'},'app_sha256':{name:sha(app/name) for name in ('TensorSharp.Models.dll','InferenceWeb.Tests.dll','libGgmlOps.so')},'source_sha256':{p.name:sha(p) for p in root.glob('*.cs')}},indent=2))
print((root/'run.log').read_text()[-24000:]);raise SystemExit(status)
