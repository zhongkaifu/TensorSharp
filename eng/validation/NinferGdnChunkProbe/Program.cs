// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Security.Cryptography;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

bool diagnose=args.Contains("--diagnose-sparse",StringComparer.Ordinal);
bool capture=args.Contains("--capture-first-sparse",StringComparer.Ordinal);
string[] positional=args.Where(x=>x!="--diagnose-sparse"&&x!="--capture-first-sparse").ToArray();
if(positional.Length is <2 or >3||args.Length!=positional.Length+(diagnose?1:0)+(capture?1:0))
    throw new ArgumentException("Usage: NinferGdnChunkProbe CHUNK_MODULE OUTPUT_JSON [SEQUENTIAL_HELPER_MODULE] [--diagnose-sparse|--capture-first-sparse]");
CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();
using var context=CudaContext.Create(0);
using var chunk=new CudaNinferGdnChunk(positional[0]);
using var sequential=positional.Length==3?new CudaNinferKernels(positional[2]):null;
var records=new List<object>();bool numericalFailure=false;
string? tensorManifest=null;
var sparse1024Outputs=new Dictionary<bool,float[]>();
try
{
    if(capture)Run(64,true);
    else if(diagnose)
    {
        foreach(int rows in new[]{64,128,192,512,896,960,1024})Run(rows,true);
        Run(1024,true,true);Run(1024,false);
    }
    else foreach(var (rows,sparse) in new[]{(64,false),(128,false),(192,false),(1024,true),(4096,true)})Run(rows,sparse);
    Save(!numericalFailure,null);
    Console.WriteLine((numericalFailure?"FAIL":"PASS")+" original-source GDN chunk component: "+records.Count+" records");
    if(numericalFailure)Environment.ExitCode=1;
}
catch(Exception error){Save(false,error.ToString());throw;}

void Save(bool passed,string? error)
{
    File.WriteAllText(positional[1],JsonSerializer.Serialize(new{schema="ninfer-original-gdn-chunk-probe/1",scope="synthetic-component-only",
        profile=CudaNinferGdnChunk.Profile,moduleSha256=chunk.ImageSha256,kernelSymbols=chunk.KernelSymbols,
        criteriaSource="gpillon/ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4 kernel/vendor/tests/ops/test_gated_delta_net.cpp:25-32 and ops/op_check.h:127-135",
        criteriaWereFixedBeforeDeviceRun=true,diagnosticMode=diagnose,captureFirstSparse=capture,tensorManifest,passed,error,records},new JsonSerializerOptions{WriteIndented=true}));
}
void Run(int rows,bool sparse,bool forceSingleOutput=false)
{
    string caseTag=forceSingleOutput?"_singleOutput":string.Empty;
    const int guard=128;using var memory=new DeviceMemory();
    var input=Fixture.Create(rows,sparse);var oracle=Fixture.Reference(input,sparse);
    var qkv=memory.Put(input.Qkv);var g=memory.Put(input.Decay);var beta=memory.Put(input.Beta);
    float[] statePoison=Enumerable.Repeat(123456f,input.State.Length+2*guard).ToArray();
    Array.Copy(input.State,0,statePoison,guard,input.State.Length);
    var stateBase=memory.Put(statePoison);var state=stateBase+guard*sizeof(float);
    float[] outputPoison=Enumerable.Repeat(123456f,oracle.Output.Length+2*guard).ToArray();
    var outputBase=memory.Put(outputPoison);var output=outputBase+guard*sizeof(float);
    long bytes=CudaNinferGdnChunk.WorkspaceBytes(rows);
    byte[] workspacePoison=Enumerable.Repeat((byte)0x3c,checked((int)bytes+512)).ToArray();
    var workspaceBase=memory.Put(workspacePoison);var workspace=workspaceBase+256;
    var wall=Stopwatch.StartNew();
    double candidateMs=EventTime(()=>chunk.RunNormalizedDiagnostic(qkv,rows,g,beta,state,output,workspace,bytes,IntPtr.Zero,forceSingleOutput));
    float[] actualOutput=memory.Read<float>(outputBase,outputPoison.Length),actualState=memory.Read<float>(stateBase,statePoison.Length);
    Guards("output",actualOutput,outputPoison,guard);Guards("state",actualState,statePoison,guard);
    byte[] workspaceAfter=memory.Read<byte>(workspaceBase,workspacePoison.Length);
    if(!workspaceAfter.AsSpan(0,256).SequenceEqual(workspacePoison.AsSpan(0,256))
        ||!workspaceAfter.AsSpan(256+(int)bytes,256).SequenceEqual(workspacePoison.AsSpan(256+(int)bytes,256)))throw new Exception("workspace guard changed");
    if(capture&&rows==64&&sparse&&!forceSingleOutput)
        CaptureFirstSparse(input,actualOutput.AsSpan(guard,oracle.Output.Length).ToArray(),
            actualState.AsSpan(guard,oracle.State.Length).ToArray(),workspaceAfter.AsSpan(256,(int)bytes).ToArray());
    CheckReduction($"chunk_T{rows}{caseTag}_output",actualOutput.AsSpan(guard,oracle.Output.Length),oracle.Output,0.0041,0.000005,0.0055);
    CheckReduction($"chunk_T{rows}{caseTag}_state",actualState.AsSpan(guard,oracle.State.Length),oracle.State,0.0027,0.00001,0.0039);
    if(diagnose)
    {
        records.Add(new{name="output_error_by_token",rows,sparse,forceSingleOutput,
            tokens=ErrorByToken(actualOutput.AsSpan(guard,oracle.Output.Length),oracle.Output,rows)});
        if(rows==1024&&sparse)
        {
            sparse1024Outputs[forceSingleOutput]=actualOutput.AsSpan(guard,oracle.Output.Length).ToArray();
            if(sparse1024Outputs.TryGetValue(false,out var multi)&&sparse1024Outputs.TryGetValue(true,out var single))
            {
                int different=0;double maximum=0;
                for(int i=0;i<multi.Length;i++)
                {
                    if(BitConverter.SingleToInt32Bits(multi[i])!=BitConverter.SingleToInt32Bits(single[i]))different++;
                    maximum=Math.Max(maximum,Math.Abs((double)multi[i]-single[i]));
                }
                records.Add(new{name="sparse_T1024_default_multi_vs_forced_original_single_output",count=multi.Length,different,maxAbsoluteDifference=maximum,exact=different==0,passed=different==0});
                if(different!=0)numericalFailure=true;
            }
        }
    }
    // Reinitialize state/output and poison scratch differently. Fully written
    // stage buffers must not depend on prior allocation contents.
    memory.Write(stateBase,statePoison);memory.Write(outputBase,outputPoison);
    Array.Fill(workspacePoison,(byte)0xa5);memory.Write(workspaceBase,workspacePoison);
    double repeatMs=EventTime(()=>chunk.RunNormalizedDiagnostic(qkv,rows,g,beta,state,output,workspace,bytes,IntPtr.Zero,forceSingleOutput));
    var repeatedOutput=memory.Read<float>(outputBase,outputPoison.Length);var repeatedState=memory.Read<float>(stateBase,statePoison.Length);
    Exact($"chunk_T{rows}_AA_output",actualOutput,repeatedOutput);Exact($"chunk_T{rows}_AA_state",actualState,repeatedState);
    Exact($"chunk_T{rows}_qkv_unchanged",memory.Read<float>(qkv,input.Qkv.Length),input.Qkv);
    Exact($"chunk_T{rows}_decay_unchanged",memory.Read<float>(g,input.Decay.Length),input.Decay);
    Exact($"chunk_T{rows}_beta_unchanged",memory.Read<float>(beta,input.Beta.Length),input.Beta);
    double? sequentialMs=null;
    if(sequential!=null)
    {
        memory.Write(stateBase,statePoison);memory.Write(outputBase,outputPoison);
        var stateTable=memory.Put(new[]{state});
        // The baseline receives the same represented BF16 normalized Q/K/V as
        // the original chunk stages. It is not allowed extra input precision.
        float[] represented=input.Qkv.Select(Fixture.Bf).ToArray();var baselineInput=memory.Put(represented);
        sequentialMs=EventTime(()=>sequential.GdnScan(baselineInput,10240,16,48,rows,1,g,beta,stateTable,output,false,IntPtr.Zero));
        var baselineOutput=memory.Read<float>(outputBase,outputPoison.Length);var baselineState=memory.Read<float>(stateBase,statePoison.Length);
        Guards("baseline output",baselineOutput,outputPoison,guard);Guards("baseline state",baselineState,statePoison,guard);
        CheckReduction($"sequential_T{rows}_output",baselineOutput.AsSpan(guard,oracle.Output.Length),oracle.Output,0.0041,0.000005,0.0055);
        CheckReduction($"sequential_T{rows}_state",baselineState.AsSpan(guard,oracle.State.Length),oracle.State,0.0027,0.00001,0.0039);
    }
    records.Add(new{name="component_timing",rows,forceSingleOutput,fixture=sparse?"onehot-structured":"dense-bounded-deterministic",workspaceBytes=bytes,
        candidateMs,repeatMs,sequentialMs,candidateIncludesPackAndUnpack=true,sequentialIncludesOnlyScan=true,
        timingStatus="diagnostic_single_case_not_e2e_or_statistical_speedup",wallSeconds=wall.Elapsed.TotalSeconds,passed=true});
    Console.WriteLine("COMPLETE T="+rows+" sparse="+sparse+" forceSingleOutput="+forceSingleOutput+" component candidate="+candidateMs.ToString("F3")+"ms");
}
void CaptureFirstSparse(Fixture fixture,float[] output,float[] finalState,byte[] workspace)
{
    string result=Path.GetFullPath(positional[1]);
    string folder=Path.Combine(Path.GetDirectoryName(result)!,Path.GetFileNameWithoutExtension(result)+"-first-sparse-tensors");
    if(Directory.Exists(folder))throw new IOException("Refusing to overwrite existing tensor capture: "+folder);
    Directory.CreateDirectory(folder);
    var files=new List<object>();
    void Write(string name,byte[] bytes,string dtype,int[] shape,string layout)
    {
        string path=Path.Combine(folder,name);File.WriteAllBytes(path,bytes);
        files.Add(new{name,bytes=bytes.Length,sha256=Convert.ToHexStringLower(SHA256.HashData(bytes)),dtype,shape,layout});
    }
    void F32(string name,float[] values,int[] shape,string layout)
        =>Write(name,MemoryMarshal.AsBytes(values.AsSpan()).ToArray(),"float32-le",shape,layout);
    var plan=CudaNinferGdnChunk.WorkspaceLayout(fixture.Rows);
    void Region(string name,long offset,int bytes,string dtype,int[] shape,string layout)
        =>Write(name,workspace.AsSpan(checked((int)offset),bytes).ToArray(),dtype,shape,layout);
    const int t=64,hk=16,hv=48,d=128;
    Region("query.bf16",plan.Query,t*hk*d*2,"bfloat16-le",new[]{t,hk,d},"token,query-key-head,dimension");
    Region("key.bf16",plan.Key,t*hk*d*2,"bfloat16-le",new[]{t,hk,d},"token,query-key-head,dimension");
    Region("value.bf16",plan.Value,t*hv*d*2,"bfloat16-le",new[]{t,hv,d},"token,value-head,dimension");
    Region("U.bf16",plan.U,t*hv*d*2,"bfloat16-le",new[]{t,hv,d},"token,value-head,dimension");
    Region("W-original-layout.bf16",plan.W,t*hv*d*2,"bfloat16-le",new[]{t,hv,d},"token,value-head,dimension; each group of 8 dimensions stored in original order {0,4,1,5,2,6,3,7}; not logical contiguous coordinates");
    Region("v_new.bf16",plan.VNew,t*hv*d*2,"bfloat16-le",new[]{t,hv,d},"token,value-head,dimension");
    Region("h_chunk.bf16",plan.ChunkStates,hv*d*d*2,"bfloat16-le",new[]{1,hv,d,d},"chunk,value-head,value-coordinate,key-coordinate");
    Region("g_cumsum.f32",plan.CumulativeDecay,t*hv*4,"float32-le",new[]{t,hv},"token,value-head");
    F32("input-qkv.f32",fixture.Qkv,new[]{t,10240},"token,packed q|k|v channel");
    F32("decay.f32",fixture.Decay,new[]{t,hv},"token,value-head");
    F32("beta.f32",fixture.Beta,new[]{t,hv},"token,value-head");
    F32("initial-state.f32",fixture.State,new[]{hv,d,d},"value-head,value-coordinate,key-coordinate");
    F32("final-state.f32",finalState,new[]{hv,d,d},"value-head,value-coordinate,key-coordinate");
    F32("output.f32",output,new[]{t,hv,d},"token,value-head,dimension; represented BF16 values promoted exactly");
    string probeAssembly=typeof(Fixture).Assembly.Location,backendAssembly=typeof(CudaNinferGdnChunk).Assembly.Location;
    tensorManifest=Path.Combine(folder,"manifest.json");
    File.WriteAllText(tensorManifest,JsonSerializer.Serialize(new{schema="ninfer-first-sparse-stage-capture/1",
        scope="diagnostic-first-candidate-before-AA-and-before-sequential-reference",rows=64,fixture="onehot-structured",
        moduleImageSha256=chunk.ImageSha256,profile=CudaNinferGdnChunk.Profile,workspaceBytes=workspace.Length,
        probeAssemblySha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(probeAssembly))),
        backendAssemblySha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(backendAssembly))),
        originalRevision="e600a7d9e0d5921d767a6cb464373583b3cdcaf4",criterionModified=false,files},new JsonSerializerOptions{WriteIndented=true}));
    records.Add(new{name="first_sparse_tensor_capture",manifest=tensorManifest,
        manifestSha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(tensorManifest))),
        fileCount=files.Count,interpretation="diagnostic data; original failing score remains authoritative",passed=true});
}
void CheckReduction(string name,ReadOnlySpan<float> actual,double[] reference,double relativeL2Limit,double grossAbsolute,double grossRelative)
{
    if(actual.Length!=reference.Length)throw new Exception(name+": size mismatch");
    double sumError=0,sumReference=0,maxError=0,maxReference=0;int nonfinite=0;
    for(int i=0;i<actual.Length;i++)
    {
        if(!float.IsFinite(actual[i])||!double.IsFinite(reference[i])){nonfinite++;continue;}
        double error=actual[i]-reference[i];sumError+=error*error;sumReference+=reference[i]*reference[i];
        maxError=Math.Max(maxError,Math.Abs(error));maxReference=Math.Max(maxReference,Math.Abs(reference[i]));
    }
    double relativeL2=Math.Sqrt(sumError)/Math.Max(Math.Sqrt(sumReference),1e-30),grossLimit=grossAbsolute+grossRelative*maxReference;
    bool passed=nonfinite==0&&relativeL2<=relativeL2Limit&&maxError<=grossLimit;
    records.Add(new{name,count=reference.Length,relativeL2,relativeL2Limit,maxAbsoluteError=maxError,maxReference,grossLimit,grossAbsolute,grossRelative,nonfinite,passed});
    if(!passed)
    {
        numericalFailure=true;
        if(!(diagnose||capture)||nonfinite>0)throw new Exception($"{name}: relL2={relativeL2}/{relativeL2Limit}, gross={maxError}/{grossLimit}, nonfinite={nonfinite}");
    }
}
void Exact(string name,float[] actual,float[] expected)
{
    if(actual.Length!=expected.Length)throw new Exception(name+": size mismatch");
    for(int i=0;i<actual.Length;i++)if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(expected[i]))throw new Exception(name+": mismatch "+i);
    records.Add(new{name,count=actual.Length,exact=true,passed=true});
}
static object[] ErrorByToken(ReadOnlySpan<float> actual,double[] reference,int rows)
{
    const int width=48*128;var result=new object[rows];
    for(int t=0;t<rows;t++)
    {
        double squaredError=0,squaredReference=0,maxError=0,maxReference=0;int maxIndex=-1;
        for(int d=0;d<width;d++)
        {
            int i=t*width+d;double delta=actual[i]-reference[i];squaredError+=delta*delta;squaredReference+=reference[i]*reference[i];
            maxReference=Math.Max(maxReference,Math.Abs(reference[i]));
            if(Math.Abs(delta)>maxError){maxError=Math.Abs(delta);maxIndex=d;}
        }
        result[t]=new{token=t,relativeL2=Math.Sqrt(squaredError)/Math.Max(Math.Sqrt(squaredReference),1e-30),maxError,maxReference,maxHead=maxIndex<0?-1:maxIndex/128,maxDimension=maxIndex<0?-1:maxIndex%128};
    }
    return result;
}
static void Guards(string name,float[] actual,float[] poison,int guard)
{
    for(int i=0;i<guard;i++)if(BitConverter.SingleToInt32Bits(actual[i])!=BitConverter.SingleToInt32Bits(poison[i])
        ||BitConverter.SingleToInt32Bits(actual[actual.Length-1-i])!=BitConverter.SingleToInt32Bits(poison[poison.Length-1-i]))throw new Exception(name+": guard changed");
}
static double EventTime(Action action)
{
    CudaDriverApi.cuEventCreate(out var start,0).ThrowOnError();CudaDriverApi.cuEventCreate(out var end,0).ThrowOnError();
    try
    {
        CudaDriverApi.cuEventRecord(start,IntPtr.Zero).ThrowOnError();action();CudaDriverApi.cuEventRecord(end,IntPtr.Zero).ThrowOnError();
        CudaDriverApi.cuEventSynchronize(end).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out float milliseconds,start,end).ThrowOnError();return milliseconds;
    }
    finally{CudaDriverApi.cuEventDestroy(start).ThrowOnError();CudaDriverApi.cuEventDestroy(end).ThrowOnError();}
}
internal sealed record Fixture(int Rows,float[] Qkv,float[] Decay,float[] Beta,float[] State)
{
    const int Hk=16,Hv=48,D=128,C=10240;
    public static float Bf(float x){uint bits=(uint)BitConverter.SingleToInt32Bits(x);return BitConverter.Int32BitsToSingle((int)((bits+0x7fffu+((bits>>16)&1))&0xffff0000u));}
    public static Fixture Create(int rows,bool sparse)
    {
        var random=new Random(367+rows);float Sample(float low,float high)=>low+(high-low)*(float)random.NextDouble();
        float[] qkv=new float[rows*C],g=new float[rows*Hv],beta=new float[rows*Hv],state=new float[Hv*D*D];
        for(int t=0;t<rows;t++)for(int h=0;h<Hk;h++)
        {
            for(int side=0;side<2;side++)
            {
                int offset=t*C+(side*Hk+h)*D;
                if(sparse){qkv[offset+h]=1;continue;}
                double ss=0;for(int d=0;d<D;d++){float x=Bf(Sample(-1,1));qkv[offset+d]=x;ss+=(double)x*x;}
                double inv=1/Math.Sqrt(ss+1e-6);
                for(int d=0;d<D;d++)qkv[offset+d]=(float)(qkv[offset+d]*inv);
            }
        }
        for(int t=0;t<rows;t++)for(int h=0;h<Hv;h++)
        {
            g[t*Hv+h]=sparse?-0.03125f:Sample(-0.1f,-0.005f);
            beta[t*Hv+h]=sparse?0.25f+(h%3)*0.125f:Sample(0.05f,0.95f);
            for(int d=0;d<D;d++)qkv[t*C+2*Hk*D+h*D+d]=Bf(sparse?((h*3+d*7+t*11)%31-15)/32f:Sample(-0.5f,0.5f));
        }
        if(sparse)
            for(int h=0;h<Hv;h++)for(int d=0;d<D;d++)state[(h*D+d)*D+h/3]=((h+d)%7-3)/256f;
        else for(int i=0;i<state.Length;i++)state[i]=Sample(-0.02f,0.02f);
        return new(rows,qkv,g,beta,state);
    }
    public static (double[] Output,double[] State) Reference(Fixture input,bool sparse)
    {
        var output=new double[input.Rows*Hv*D];var state=input.State.Select(x=>(double)x).ToArray();
        float[] qkv=input.Qkv.Select(Bf).ToArray();const double scale=(double)0.08838834764831845f;
        Parallel.For(0,Hv,h=>
        {
            int kh=h/3;double[] key=new double[D],query=new double[D];
            for(int t=0;t<input.Rows;t++)
            {
                double decay=Math.Exp(input.Decay[t*Hv+h]),beta=input.Beta[t*Hv+h];
                if(sparse)
                {
                    for(int v=0;v<D;v++)
                    {
                        int index=(h*D+v)*D+kh;double old=state[index],value=qkv[t*C+2*Hk*D+h*D+v];
                        double next=decay*old+beta*(value-decay*old);state[index]=next;output[(t*Hv+h)*D+v]=scale*next;
                    }
                    continue;
                }
                for(int k=0;k<D;k++){query[k]=qkv[t*C+kh*D+k];key[k]=qkv[t*C+(Hk+kh)*D+k];}
                for(int v=0;v<D;v++)
                {
                    int offset=(h*D+v)*D;double dot=0;for(int k=0;k<D;k++)dot+=state[offset+k]*key[k];
                    double delta=beta*(qkv[t*C+2*Hk*D+h*D+v]-decay*dot),readout=0;
                    for(int k=0;k<D;k++){double next=decay*state[offset+k]+delta*key[k];state[offset+k]=next;readout+=next*query[k];}
                    output[(t*Hv+h)*D+v]=readout*scale;
                }
            }
        });
        return(output,state);
    }
}
sealed unsafe class DeviceMemory:IDisposable
{
    readonly List<IntPtr> _allocations=new();
    public IntPtr Put<T>(T[] values)where T:unmanaged
    {CudaDriverApi.cuMemAlloc(out var p,(UIntPtr)checked((long)values.Length*sizeof(T))).ThrowOnError();_allocations.Add(p);Write(p,values);return p;}
    public void Write<T>(IntPtr p,T[] values)where T:unmanaged
    {fixed(T* v=values)CudaDriverApi.cuMemcpyHtoD(p,(IntPtr)v,(UIntPtr)checked((long)values.Length*sizeof(T))).ThrowOnError();}
    public T[] Read<T>(IntPtr p,int count)where T:unmanaged
    {T[] result=new T[count];fixed(T* v=result)CudaDriverApi.cuMemcpyDtoH((IntPtr)v,p,(UIntPtr)checked((long)count*sizeof(T))).ThrowOnError();return result;}
    public void Dispose(){foreach(var p in _allocations)CudaDriverApi.cuMemFree(p).ThrowOnError();}
}
