// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models;
using TensorSharp.Runtime;

if(args.Length<2)throw new ArgumentException("Usage: NinferRepackProbe MODULE OUTPUT [--artifact FILE] [--all-target-nvfp4]");
string? artifact=null;bool all=false;
for(int i=2;i<args.Length;i++)
    if(args[i]=="--artifact"&&i+1<args.Length)artifact=args[++i];
    else if(args[i]=="--all-target-nvfp4")all=true;
    else throw new ArgumentException("Unknown or incomplete probe option: "+args[i]);
if(all&&artifact==null)throw new ArgumentException("--all-target-nvfp4 requires --artifact");
CudaBackend.Register();CudaDriverApi.cuInit(0).ThrowOnError();using var context=CudaContext.Create(0);
using var module=new CudaNinferRepackModule(args[0]);var records=new List<object>();
try
{
    var synthetic=new[]{
        (128,64,CudaNinferOriginalRowMapping.Identity),(384,192,CudaNinferOriginalRowMapping.Identity),
        (128,65536,CudaNinferOriginalRowMapping.Identity),(5120,17408,CudaNinferOriginalRowMapping.Identity),
        (16384,5120,CudaNinferOriginalRowMapping.Identity),(256,64,CudaNinferOriginalRowMapping.GateUpRows8),
        (768,192,CudaNinferOriginalRowMapping.GateUpRows8),(34816,5120,CudaNinferOriginalRowMapping.GateUpRows8),
        (14336,64,CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue),
        (14336,192,CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue),
        (14336,5120,CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue)};
    int seed=31;
    foreach(var (rows,columns,mapping) in synthetic)
    {
        using var fixture=new Fixture(rows,columns,seed++);
        using var planes=new NinferNvfp4PlaneReader(fixture.Reader,fixture.Weights,mapping:ModelMapping(mapping));
        byte[] cpu=ReadCpuPlanes(planes),independent=fixture.IndependentExpected(mapping);
        Need(cpu.AsSpan().SequenceEqual(independent),"Independent logical writer differs from existing CPU reader");
        byte[] validatedRaw=ReadValidatedOriginal(fixture.Reader,fixture.Weights,mapping);
        Need(validatedRaw.AsSpan().SequenceEqual(fixture.Raw),"Validated raw adapter changed original fixture bytes");
        CheckCase("synthetic",validatedRaw,rows,columns,mapping,cpu,fixture.Identity());
    }
    if(artifact!=null)
    {
        using var reader=new NinferArtifactReader(artifact);var plan=NinferQwen38TargetPlan.Create(reader);
        var identity=new{path=reader.FullPath,reader.FileBytes,reader.DirectorySha256,reader.PrefixSha256,
            reader.FileLastWriteUtcTicks,fullArtifactSha256Recomputed=false};
        string[] selected={"text/layers/0/gdn/query_key_value_z","text/layers/0/mlp/gate_up",
            "text/layers/0/mlp/down","text/layers/27/attention/query_key_gate_value"};
        var tensors=plan.Tensors.Where(x=>x.Descriptor.Format==NinferTensorFormat.NvFp4&&(all||selected.Contains(x.Descriptor.Name,StringComparer.Ordinal))).ToArray();
        Need(tensors.Length==(all?247:4),"Unexpected number of selected actual NVFP4 target matrices");
        foreach(var binding in tensors)
        {
            using var planes=NinferNvfp4PlaneReader.FromTarget(reader,binding);
            var mapping=CudaMapping(planes.RowMapping);var geometry=CudaNinferRepackModule.ValidateGeometry(planes.Rows,planes.Columns,mapping);
            byte[] raw=ReadValidatedOriginal(reader,binding.Descriptor,mapping);
            Need(BinaryPrimitives.ReadUInt32LittleEndian(raw.AsSpan(raw.Length-4))==planes.WeightDivisorBits,"Actual original divisor differs");
            CheckCase(binding.Descriptor.Name,raw,planes.Rows,planes.Columns,mapping,ReadCpuPlanes(planes),identity);
        }
    }
    Save(true,null);Console.WriteLine("PASS lossless original NVFP4 GPU repack: "+records.Count+" matrices");
}
catch(Exception error){Save(false,error.ToString());throw;}

void Save(bool passed,string? error)
    =>File.WriteAllText(args[1],JsonSerializer.Serialize(new{schema="ninfer-lossless-gpu-repack-probe/1",
        moduleImageSha256=module.ImageSha256,scope="encoded-byte-conversion-only",actualArtifactRequested=artifact!=null,allTargetNvfp4Requested=all,
        inferenceOrQualityValidated=false,passed,error,matrices=records},new JsonSerializerOptions{WriteIndented=true}));
void CheckCase(string name,byte[] raw,int rows,int columns,CudaNinferOriginalRowMapping mapping,byte[] expected,object identity)
{
    const int guard=256;var geometry=CudaNinferRepackModule.ValidateGeometry(rows,columns,mapping);
    Need(raw.Length==geometry.OriginalBytes&&expected.Length==geometry.DestinationBytes,"Declared buffer bytes differ");
    using var memory=new DeviceMemory();
    byte[] sourceGuarded=Enumerable.Repeat((byte)0x91,raw.Length+guard*2).ToArray();raw.CopyTo(sourceGuarded,guard);
    var rawBase=memory.Put(sourceGuarded);var rawDevice=rawBase+guard;
    byte[] destinationPoison=Enumerable.Repeat((byte)0xa5,expected.Length+guard*2).ToArray();
    var destinationBase=memory.Put(destinationPoison);var destination=destinationBase+guard;
    // Both same-base and shifted overlap must fail before launch.
    Refuse(()=>module.Repack(rawDevice,rawDevice,rows,columns,mapping,IntPtr.Zero));
    Refuse(()=>module.Repack(rawDevice,rawDevice+8,rows,columns,mapping,IntPtr.Zero));
    double first=Time(()=>module.Repack(rawDevice,destination,rows,columns,mapping,IntPtr.Zero));
    byte[] firstOutput=memory.Read<byte>(destinationBase,destinationPoison.Length);
    Need(firstOutput.AsSpan(guard,expected.Length).SequenceEqual(expected),name+": first GPU planes differ from CPU bytes");
    Guards(firstOutput,destinationPoison,guard);byte[] inverse=Invert(firstOutput.AsSpan(guard,expected.Length),rows,columns,mapping);
    Need(inverse.AsSpan().SequenceEqual(raw.AsSpan(0,raw.Length-4)),name+": independent inverse differs from original encoded bytes");
    Array.Fill(destinationPoison,(byte)0x3c);memory.Write(destinationBase,destinationPoison);
    double repeat=Time(()=>module.Repack(rawDevice,destination,rows,columns,mapping,IntPtr.Zero));
    byte[] repeated=memory.Read<byte>(destinationBase,destinationPoison.Length);
    Need(repeated.AsSpan(guard,expected.Length).SequenceEqual(expected),name+": A/A GPU planes differ");Guards(repeated,destinationPoison,guard);
    Need(memory.Read<byte>(rawBase,sourceGuarded.Length).AsSpan().SequenceEqual(sourceGuarded),name+": original bytes, trailer or guard changed");
    records.Add(new{name,rows,columns,rowMapping=mapping.ToString(),geometry.OriginalBytes,geometry.DestinationBytes,
        identity,sourceSha256=Sha(raw),cpuPlaneSha256=Sha(expected),gpuPlaneSha256=Sha(firstOutput.AsSpan(guard,expected.Length)),
        independentInverseSha256=Sha(inverse),weightDivisorBits=BinaryPrimitives.ReadUInt32LittleEndian(raw.AsSpan(raw.Length-4)),
        exactComparedBytes=expected.Length,aaRepeats=2,outputGuardBytes=guard*2,sourceGuardAndTrailerChecked=true,
        overlapRefusals=2,productionRawAdapterValidatedBeforeLaunch=true,kernelEventMilliseconds=new[]{first,repeat},
        timingScope="diagnostic_conversion_only_no_transfer_or_startup_speed_claim",passed=true});
    Console.WriteLine("PASS "+name+" "+rows+"x"+columns+" "+mapping+" "+expected.Length+" exact bytes");
}
static void Refuse(Action operation){try{operation();}catch(ArgumentException){return;}throw new Exception("Overlapping conversion was not rejected");}
static void Need(bool value,string message){if(!value)throw new InvalidOperationException(message);}
static void Guards(byte[] actual,byte[] expected,int count)
{
    Need(actual.AsSpan(0,count).SequenceEqual(expected.AsSpan(0,count))
        &&actual.AsSpan(actual.Length-count,count).SequenceEqual(expected.AsSpan(expected.Length-count,count)),"Destination guard was modified");
}
static string Sha(ReadOnlySpan<byte> bytes)=>Convert.ToHexStringLower(SHA256.HashData(bytes));
static NinferNvfp4RowMapping ModelMapping(CudaNinferOriginalRowMapping m)=>(NinferNvfp4RowMapping)(int)m;
static CudaNinferOriginalRowMapping CudaMapping(NinferNvfp4RowMapping m)=>(CudaNinferOriginalRowMapping)(int)m;
static unsafe byte[] ReadValidatedOriginal(NinferArtifactReader reader,NinferTensorDescriptor tensor,CudaNinferOriginalRowMapping mapping)
{
    using var source=new NinferNvfp4WeightSource(new NinferNvfp4PlaneReader(reader,tensor,mapping:ModelMapping(mapping)));
    var descriptor=new CudaNinferProjectionDesc(source,0,source.Bytes,source.Rows,source.Columns,
        CudaNinferProjectionFormat.NvFp4Planes,weightScaleDivisor:tensor.WeightDivisor!.Value);
    var validated=new CudaNinferValidatedOriginalSource(descriptor);byte[] raw=new byte[checked((int)validated.Bytes)];
    fixed(byte* data=raw)validated.Read(0,(IntPtr)data,raw.Length);
    validated.RequireCompleteValidation();Need(validated.IsValidated,"Original source validation was not completed");return raw;
}
static byte[] ReadCpuPlanes(NinferNvfp4PlaneReader reader)
{
    int codes=checked(reader.Rows*reader.Columns/2);var result=new byte[checked(codes+reader.Rows*reader.Columns/16)];
    for(int t=0;t<reader.TileCount;t++)
    {
        var tile=reader.ReadTile(t);tile.Codes.Span.CopyTo(result.AsSpan(tile.FirstRow*(reader.Columns/2)));
        tile.Scales.Span.CopyTo(result.AsSpan(codes+tile.FirstRow*(reader.Columns/16)));
    }
    return result;
}
// Independent inverse direction: find each source row's destination, recover
// adjacent logical nibbles, then serialize scale bytes in five producer axes.
static int DestinationForSource(int source,int rows,CudaNinferOriginalRowMapping mapping)
    =>mapping switch
    {
        CudaNinferOriginalRowMapping.Identity=>source,
        CudaNinferOriginalRowMapping.GateUpRows8=>source%(rows/2)/8*16+source/(rows/2)*8+source%8,
        CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue when source<6144=>source/256*512+source%256,
        CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue when source<7168=>12288+source-6144,
        CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue when source<13312=>(source-7168)/256*512+256+(source-7168)%256,
        _=>source
    };
static byte CodeAt(ReadOnlySpan<byte> planes,int row,int column,int width)
{
    int within=column%16;byte packed=planes[row*(width/2)+(column/16)*8+within%8];
    return(byte)((within<8?packed:packed>>4)&15);
}
static byte[] Invert(ReadOnlySpan<byte> planes,int rows,int columns,CudaNinferOriginalRowMapping mapping)
{
    int codes=checked(rows*columns/2);var result=new byte[planes.Length];
    for(int source=0;source<rows;source++)
    {
        int destination=DestinationForSource(source,rows,mapping);
        for(int c=0;c<columns;c+=2)
            result[source*(columns/2)+c/2]=(byte)(CodeAt(planes,destination,c,columns)|(CodeAt(planes,destination,c+1,columns)<<4));
    }
    int at=codes;
    for(int mb=0;mb<rows/128;mb++)for(int kb=0;kb<columns/64;kb++)
        for(int mod=0;mod<32;mod++)for(int quarter=0;quarter<4;quarter++)for(int lane=0;lane<4;lane++)
        {
            int source=mb*128+quarter*32+mod,destination=DestinationForSource(source,rows,mapping);
            result[at++]=planes[codes+destination*(columns/16)+kb*4+lane];
        }
    return result;
}
static double Time(Action action)
{
    CudaDriverApi.cuEventCreate(out var start,0).ThrowOnError();CudaDriverApi.cuEventCreate(out var end,0).ThrowOnError();
    try{CudaDriverApi.cuEventRecord(start,IntPtr.Zero).ThrowOnError();action();CudaDriverApi.cuEventRecord(end,IntPtr.Zero).ThrowOnError();CudaDriverApi.cuEventSynchronize(end).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out float ms,start,end).ThrowOnError();return ms;}
    finally{CudaDriverApi.cuEventDestroy(start).ThrowOnError();CudaDriverApi.cuEventDestroy(end).ThrowOnError();}
}
sealed class Fixture:IDisposable
{
    readonly string _path=Path.GetTempFileName();readonly uint[] _rowHash,_columnHash;readonly int _seed;
    public int Rows{get;}public int Columns{get;}public byte[] Raw{get;}public NinferArtifactReader Reader{get;}
    public NinferTensorDescriptor Weights=>(NinferTensorDescriptor)Reader.GetObject("weights");
    public Fixture(int rows,int columns,int seed)
    {
        Rows=rows;Columns=columns;_seed=seed;_rowHash=Enumerable.Range(0,rows).Select(i=>Mix((uint)i^(uint)seed)).ToArray();
        _columnHash=Enumerable.Range(0,columns).Select(i=>Mix((uint)i+0x1934a1u)).ToArray();
        int codes=checked(rows*columns/2),scales=checked(rows*columns/16);Raw=new byte[codes+scales+4];
        for(int r=0;r<rows;r++)for(int c=0;c<columns;c+=2)Raw[r*(columns/2)+c/2]=(byte)(Code(r,c)|(Code(r,c+1)<<4));
        int at=codes;
        for(int mb=0;mb<rows/128;mb++)for(int kb=0;kb<columns/64;kb++)
            for(int mod=0;mod<32;mod++)for(int quarter=0;quarter<4;quarter++)for(int lane=0;lane<4;lane++)
                Raw[at++]=Scale(mb*128+quarter*32+mod,kb*4+lane);
        BinaryPrimitives.WriteUInt32LittleEndian(Raw.AsSpan(at),0x3f800001u);
        byte[] directory=JsonSerializer.SerializeToUtf8Bytes(new{identity=new{model_id="independent-repack-fixture",weights_id="not-a-model"},objects=new[]{new{name="weights",kind="tensor",shape=new[]{rows,columns},format="NVFP4",layout="blockscale-k16-m128x4-v1",offset=0,bytes=Raw.Length}}});
        using(var stream=File.Create(_path))
        {
            stream.Write("NINFER\0\x02"u8);Span<byte> size=stackalloc byte[8];BinaryPrimitives.WriteUInt64LittleEndian(size,(ulong)directory.Length);stream.Write(size);stream.Write(directory);
            long payload=(16L+directory.Length+4095)/4096*4096;stream.Position=payload;stream.Write(Raw);
        }
        Reader=new NinferArtifactReader(_path);
    }
    static uint Mix(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
    byte Code(int row,int column)=>(byte)(((_rowHash[row]>>(((column/16)&7)*4))^_columnHash[column])&15);
    byte Scale(int row,int group)=>(byte)(((long)row*19+(long)group*37+(row/128)*53+_seed)%127);
    public byte[] IndependentExpected(CudaNinferOriginalRowMapping mapping)
    {
        int codeBytes=Rows*Columns/2;var result=new byte[Raw.Length-4];
        for(int source=0;source<Rows;source++)
        {
            int destination=ProgramDestination(source,Rows,mapping);
            for(int group=0;group<Columns/16;group++)
            {
                for(int j=0;j<8;j++)result[destination*(Columns/2)+group*8+j]=(byte)(Code(source,group*16+j)|(Code(source,group*16+j+8)<<4));
                result[codeBytes+destination*(Columns/16)+group]=Scale(source,group);
            }
        }
        return result;
    }
    // Source-to-destination enumeration is intentionally opposite the CUDA
    // destination gather. This is independent of CPU reader internals.
    static int ProgramDestination(int s,int rows,CudaNinferOriginalRowMapping m)
    {
        if(m==CudaNinferOriginalRowMapping.GateUpRows8){int half=s/(rows/2),r=s%(rows/2);return r/8*16+half*8+r%8;}
        if(m==CudaNinferOriginalRowMapping.AttentionQueryGateKeyValue)
        {if(s<6144)return s/256*512+s%256;if(s<7168)return 12288+s-6144;if(s<13312){int gate=s-7168;return gate/256*512+256+gate%256;}}
        return s;
    }
    public object Identity()=>new{synthetic=true,_seed,Reader.DirectorySha256,Reader.PrefixSha256,divisorBits=0x3f800001u};
    public void Dispose(){Reader.Dispose();File.Delete(_path);}
}
sealed unsafe class DeviceMemory:IDisposable
{
    readonly List<IntPtr> _allocations=new();
    public IntPtr Put<T>(T[] data)where T:unmanaged
    {CudaDriverApi.cuMemAlloc(out var p,(UIntPtr)checked((long)data.Length*sizeof(T))).ThrowOnError();_allocations.Add(p);Write(p,data);return p;}
    public void Write<T>(IntPtr p,T[] data)where T:unmanaged
    {fixed(T* d=data)CudaDriverApi.cuMemcpyHtoD(p,(IntPtr)d,(UIntPtr)checked((long)data.Length*sizeof(T))).ThrowOnError();}
    public T[] Read<T>(IntPtr p,int count)where T:unmanaged
    {var r=new T[count];fixed(T* d=r)CudaDriverApi.cuMemcpyDtoH((IntPtr)d,p,(UIntPtr)checked((long)count*sizeof(T))).ThrowOnError();return r;}
    public void Dispose(){foreach(var p in _allocations)CudaDriverApi.cuMemFree(p).ThrowOnError();}
}
