import pathlib,subprocess,shutil,os
root=pathlib.Path(__file__).resolve().parent
base=root/'app'
project=root/'Probe.csproj'
project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><OutputType>Exe</OutputType><ImplicitUsings>enable</ImplicitUsings><EnableDefaultCompileItems>false</EnableDefaultCompileItems></PropertyGroup><ItemGroup><FrameworkReference Include="Microsoft.AspNetCore.App"/><Compile Include="Probe.cs"/>'+''.join('<Reference Include="'+p.stem+'"><HintPath>'+str(p)+'</HintPath><Private>false</Private></Reference>' for p in base.glob('*.dll') if p.name not in ('InferenceWeb.Tests.dll',))+'</ItemGroup></Project>')
dotnet='/workspace/tensorsharp-no-patch-20260915/dotnet/dotnet'
with (root/'probe-build.log').open('w') as f:status=subprocess.run([dotnet,'build',str(project),'-c','Release','-o',str(root/'probe-build')],stdout=f,stderr=subprocess.STDOUT).returncode
if status:print((root/'probe-build.log').read_text()[-6000:]);raise SystemExit(status)
for p in (root/'probe-build').glob('Probe*'):shutil.copy2(p,base/p.name)
# Use the already pinned r4 dependency closure; our probe only adds its entry assembly.
shutil.copy2(base/'InferenceWeb.Tests.deps.json',base/'Probe.deps.json')
with (root/'trained-run.log').open('w') as f:status=subprocess.run([dotnet,str(base/'Probe.dll')],cwd=root,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,TS_TEST_GGML_BACKEND='cpu')).returncode
print((root/'trained-run.log').read_text()[-7000:]);raise SystemExit(status)
