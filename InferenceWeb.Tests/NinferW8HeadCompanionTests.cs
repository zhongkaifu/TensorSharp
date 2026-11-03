using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// CPU metadata and actual small-reader range tests, not a synthetic whole-model artifact.
public sealed unsafe class NinferW8HeadCompanionTests
{
    private sealed class Fixture : IDisposable
    {
        private readonly string _directory = Path.Combine(AppContext.BaseDirectory, "artifacts", "head-source-" + Guid.NewGuid().ToString("N"));
        internal string PathName { get; }
        internal string ModulePath { get; }
        internal long Payload { get; }
        internal long Bytes { get; }
        internal Fixture(int n = 3, int k = 129)
        {
            Directory.CreateDirectory(_directory); PathName = Path.Combine(_directory, "tiny.ninfer"); ModulePath = Path.Combine(_directory, "not-a-real-module.ptx");File.WriteAllText(ModulePath,"CPU-path-validation-only");
            long padded = ((long)k + 127) / 128 * 128;long codes = n*padded;long scale = (codes+255)/256*256;Bytes=scale+n*padded/16;
            byte[] metadata=JsonSerializer.SerializeToUtf8Bytes(new{identity=new{model_id="tiny",weights_id="fixture"},objects=new[]{new{name="tiny-head",kind="tensor",shape=new[]{n,k},format="W8G32_F16S",layout="row-split-k128-v1",offset=0,bytes=Bytes}}});
            Payload=(16L+metadata.Length+4095)/4096*4096;
            using var file=File.Create(PathName);file.Write("NINFER\0\x02"u8);Span<byte> length=stackalloc byte[8];BinaryPrimitives.WriteUInt64LittleEndian(length,(ulong)metadata.Length);file.Write(length);file.Write(metadata);file.Position=Payload;
            byte[] chunk=new byte[65536];for(long at=0;at<Bytes;at+=chunk.Length){int take=(int)Math.Min(chunk.Length,Bytes-at);for(int i=0;i<take;i++)chunk[i]=Pattern(at+i);file.Write(chunk,0,take);}
            file.Write(new byte[]{0x77,0x88,0x99}); // Outside the object must never be exposed.
        }
        internal static byte Pattern(long index)=>(byte)((index*7+(index>>8))%251);
        public void Dispose()=>Directory.Delete(_directory,true);
    }
    private static NinferTensorGeometry Geometry=>new(1_350_860_800,256,5120,32,1_271_398_400,1_271_398_400,0,1_271_398_400,79_462_400);
    private static void Validate(string model="qwen3.8-27b",string weights="nvfp4full",string name="text/output_head",NinferTensorFormat format=NinferTensorFormat.W8G32Float16Scale,NinferTensorLayout layout=NinferTensorLayout.RowSplitK128V1,ulong[]? shape=null,NinferTensorGeometry? geometry=null,ulong bytes=1_350_860_800)
        =>NinferHeadWeightSource.ValidateMetadata(new(model,weights),name,format,layout,shape??new ulong[]{248320,5120},geometry??Geometry,bytes);

    [Fact]
    public void UnsetOptionsAreOffWithoutTouchingAPath()=>Assert.Null(Qwen35CudaExecutor.ParseNinferHeadOptions(null,null));
    [Fact]
    public void NativeRouteGuardDoesNotChangeOffOrHideNativePairValidation()
    {
        Qwen35CudaExecutor.ValidateNinferHeadNativeRoute(false,null,null);
        Qwen35CudaExecutor.ValidateNinferHeadNativeRoute(true,"/head.ninfer",null);
        Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions("/head.ninfer",null));
    }
    [Theory]
    [InlineData("/head.ninfer",null)][InlineData(null,"/module.ptx")][InlineData("",null)][InlineData(null,"")][InlineData("/head.ninfer","/module.ptx")]
    public void EveryExplicitOptinMustRefuseANonNativeRouteBeforeModelConstruction(string? artifact,string? module)
        =>Assert.Throws<NotSupportedException>(()=>Qwen35CudaExecutor.ValidateNinferHeadNativeRoute(false,artifact,module));
    [Theory]
    [InlineData(null,"/module.ptx")][InlineData("/head.ninfer",null)][InlineData("","")][InlineData(" ","/module.ptx")][InlineData("/head.ninfer","")]
    public void IncompleteOrBlankOptinsNeverSilentlyFallBack(string? artifact,string? module)
        =>Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions(artifact,module));
    [Fact]
    public void OptionsOnlySnapshotNormalizedPathsWithoutOpeningThem()
    {
        string head=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"not-created.ninfer")),module=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"not-created.ptx"));
        Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions(head,module)); // No implicit shared-target recipe.
        var options=Qwen35CudaExecutor.ParseNinferHeadOptions(head,module,"draft")!;Assert.Equal(head,options.ArtifactPath);Assert.Equal(module,options.ModulePath);Assert.Equal(CudaW8HeadScope.DraftOnly,options.Scope);Assert.False(File.Exists(head));
        Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions("relative.ninfer",module,"draft"));
        Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions(head,Path.Combine(AppContext.BaseDirectory,"..","not-normalized.ptx"),"draft"));
    }
    [Fact]
    public void ExplicitDraftScopeSnapshotsTheSamePathsWithoutOpeningThem()
    {
        string head=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"draft-head-not-created.ninfer"));
        string module=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"draft-head-not-created.ptx"));
        var options=Qwen35CudaExecutor.ParseNinferHeadOptions(head,module,"draft")!;
        Assert.Equal(CudaW8HeadScope.DraftOnly,options.Scope);Assert.Equal(head,options.ArtifactPath);Assert.Equal(module,options.ModulePath);
        Assert.False(File.Exists(head));Assert.False(File.Exists(module));
        Assert.Equal("TS_Q35_NINFER_HEAD_SCOPE",Qwen35CudaExecutor.NinferHeadScopeEnvironment);
    }
    [Theory]
    [InlineData("")][InlineData(" ")][InlineData("Draft")][InlineData("both")]
    public void ConfiguredPathsRequireExactlyDraftScope(string scope)
        =>Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions("/head.ninfer","/module.ptx",scope));
    [Theory]
    [InlineData(null,null)][InlineData("/head.ninfer",null)][InlineData(null,"/module.ptx")]
    public void ScopeAloneCannotHideMissingCompanionPaths(string? artifact,string? module)
        =>Assert.Throws<ArgumentException>(()=>Qwen35CudaExecutor.ParseNinferHeadOptions(artifact,module,"draft"));
    [Theory]
    [InlineData("")][InlineData("draft")]
    public void NonNativeRouteRejectsEvenScopeOnlyOptins(string scope)
        =>Assert.Throws<NotSupportedException>(()=>Qwen35CudaExecutor.ValidateNinferHeadNativeRoute(false,null,null,scope));
    [Fact]
    public void ExactHeadContractDoesNotCreateReadableFakeDescriptors()
    {
        Validate(); // Pure metadata facts only. No reader/source/GPU descriptor is returned.
        Assert.Throws<NotSupportedException>(()=>Validate(model:"qwen3.5-27b"));Assert.Throws<NotSupportedException>(()=>Validate(weights:"hybrid"));Assert.Throws<NotSupportedException>(()=>Validate(name:"text/token_embedding"));
        Assert.Throws<NotSupportedException>(()=>Validate(format:NinferTensorFormat.BFloat16));Assert.Throws<NotSupportedException>(()=>Validate(layout:NinferTensorLayout.ContiguousLittleEndianV1));
        Assert.Throws<NotSupportedException>(()=>Validate(shape:new ulong[]{5120,248320}));Assert.Throws<NotSupportedException>(()=>Validate(shape:new ulong[]{248077,5120}));
        Assert.Throws<NotSupportedException>(()=>Validate(geometry:Geometry with{ScalePlaneOffset=1_271_398_401}));Assert.Throws<NotSupportedException>(()=>Validate(geometry:Geometry with{GroupSize=64}));Assert.Throws<NotSupportedException>(()=>Validate(geometry:Geometry with{DivisorOffset=1_350_860_796}));Assert.Throws<NotSupportedException>(()=>Validate(bytes:1_350_860_801));
    }
    [Fact]
    public void CompanionRejectsBodyGeometryMismatchBeforeCuda()
    {
        var m=new Q35CudaEngine.ModelDesc{NLayer=64,NEmbd=5120,NFf=17408,NVocab=248320,NHead=24,NKvHead=4,HeadDim=256,GdnKHeads=16,GdnVHeads=48,DConv=4};
        Qwen35CudaExecutor.ValidateNinferHeadBody(m);m.NVocab=248077;Assert.Throws<NotSupportedException>(()=>Qwen35CudaExecutor.ValidateNinferHeadBody(m));m.NVocab=248320;m.NEmbd=4096;Assert.Throws<NotSupportedException>(()=>Qwen35CudaExecutor.ValidateNinferHeadBody(m));
    }
    [Theory]
    [InlineData("Qwen3.5-27B",null)][InlineData(null,"qwen3_5")][InlineData("Qwen 3.5",null)][InlineData("Llama-27B",null)]
    public void ExplicitOtherFamilyMetadataIsRejected(string? name,string? basename)
    {
        var m=new Q35CudaEngine.ModelDesc{NLayer=64,NEmbd=5120,NFf=17408,NVocab=248320,NHead=24,NKvHead=4,HeadDim=256,GdnKHeads=16,GdnVHeads=48,DConv=4};
        Assert.Throws<NotSupportedException>(()=>Qwen35CudaExecutor.ValidateNinferHeadBody(m,name,basename));
    }
    [Fact]
    public void ActualOpaqueGeneralNameIsNotFalselyRequiredToContainAFamilyName()
    {
        var m=new Q35CudaEngine.ModelDesc{NLayer=64,NEmbd=5120,NFf=17408,NVocab=248320,NHead=24,NKvHead=4,HeadDim=256,GdnKHeads=16,GdnVHeads=48,DConv=4};
        Qwen35CudaExecutor.ValidateNinferHeadBody(m,"1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",null);
        Qwen35CudaExecutor.ValidateNinferHeadBody(m,"Qwen3.8-27B",null);Qwen35CudaExecutor.ValidateNinferHeadBody(m,null,null);
    }
    [Fact]
    public void RealRangeReadsPreserveDestinationGuardsAndRelativeOffsets()
    {
        using var f=new Fixture();var reader=new NinferArtifactReader(f.PathName);using var source=new NinferHeadWeightSource(reader,reader.Tensors[0]);
        byte[] actual=Enumerable.Repeat((byte)0xa5,80).ToArray();fixed(byte* dst=actual)source.Read(7,(IntPtr)(dst+16),31);
        Assert.All(actual[..16],b=>Assert.Equal(0xa5,b));Assert.Equal(Enumerable.Range(7,31).Select(i=>Fixture.Pattern(i)),actual[16..47]);Assert.All(actual[47..],b=>Assert.Equal(0xa5,b));
        source.Read(f.Bytes,IntPtr.Zero,0);Assert.Contains("fullArtifactSha256Recomputed\":false",source.Provenance);Assert.DoesNotContain("GgmlType",source.Provenance);
    }
    [Fact]
    public void RangeAndPointerErrorsCannotExposeAdjacentArtifactBytes()
    {
        using var f=new Fixture();var reader=new NinferArtifactReader(f.PathName);using var source=new NinferHeadWeightSource(reader,reader.Tensors[0]);
        foreach(var (offset,bytes)in new[]{(-1L,1L),(0L,-1L),(f.Bytes+1,0L),(f.Bytes-1,2L),(long.MaxValue,1L)})Assert.Throws<ArgumentOutOfRangeException>(()=>source.Read(offset,(IntPtr)1024,bytes));
        Assert.Throws<ArgumentException>(()=>source.Read(0,IntPtr.Zero,1));Assert.Throws<ArgumentException>(()=>source.Read(0,new IntPtr(long.MaxValue-1),4));
    }
    [Fact]
    public void AForeignOrDisposedReaderCannotDonateADescriptor()
    {
        using var a=new Fixture();using var b=new Fixture();using var readerA=new NinferArtifactReader(a.PathName);using var readerB=new NinferArtifactReader(b.PathName);
        Assert.Throws<ArgumentException>(()=>new NinferHeadWeightSource(readerA,readerB.Tensors[0]));readerB.Dispose();Assert.Throws<ObjectDisposedException>(()=>new NinferHeadWeightSource(readerB,readerB.Tensors[0]));
    }
    [Fact]
    public void ConcurrentSmallReadsHaveNoSharedCursor()
    {
        using var f=new Fixture();var reader=new NinferArtifactReader(f.PathName);using var source=new NinferHeadWeightSource(reader,reader.Tensors[0]);
        Parallel.For(0,16,i=>{byte[] bytes=new byte[32];fixed(byte* p=bytes)source.Read(i*17,(IntPtr)p,bytes.Length);Assert.Equal(Enumerable.Range(0,32).Select(j=>Fixture.Pattern(i*17+j)),bytes);});
    }
    [Theory][InlineData(false)][InlineData(true)]
    public void ActualLoadOwnershipSeamClosesReaderOnSuccessOrFailure(bool fail)
    {
        using var f=new Fixture();var reader=new NinferArtifactReader(f.PathName);var source=new NinferHeadWeightSource(reader,reader.Tensors[0]);
        int Load(){source.Read(0,IntPtr.Zero,0);if(fail)throw new InvalidOperationException("synthetic upload failure");return 42;}
        if(fail)Assert.Throws<InvalidOperationException>(()=>Qwen35CudaExecutor.LoadNinferHeadOwned(source,Load));else Assert.Equal(42,Qwen35CudaExecutor.LoadNinferHeadOwned(source,Load));
        Assert.Throws<ObjectDisposedException>(()=>source.Read(0,IntPtr.Zero,0));Assert.Throws<ObjectDisposedException>(()=>reader.ReadRange(0,Span<byte>.Empty));source.Dispose();
    }
    [Fact]
    public void LargeDeclaredReadIsSplitWithoutAFullHeadCpuArray()
    {
        using var f=new Fixture(1,8<<20);var reader=new NinferArtifactReader(f.PathName);using var source=new NinferHeadWeightSource(reader,reader.Tensors[0]);long calls=reader.ReadOperations,bytes=reader.TotalBytesRead;
        IntPtr memory=Marshal.AllocHGlobal(checked((int)f.Bytes));
        try{source.Read(0,memory,f.Bytes);Assert.Equal(f.Bytes,reader.TotalBytesRead-bytes);Assert.True(reader.ReadOperations-calls>=2);byte* p=(byte*)memory;foreach(int at in new[]{0,(8<<20)-1,8<<20,(int)f.Bytes-1})Assert.Equal(Fixture.Pattern(at),p[at]);}
        finally{Marshal.FreeHGlobal(memory);}
    }
    [Fact]
    public void UnsupportedRealSmallArtifactCannotBecomeACompanion()
    {
        using var f=new Fixture();Assert.Throws<NotSupportedException>(()=>NinferHeadWeightSource.Open(new(f.PathName,f.ModulePath)));
        Assert.Throws<IOException>(()=>NinferHeadWeightSource.Open(new(f.PathName,f.ModulePath+".missing")));
    }
}
