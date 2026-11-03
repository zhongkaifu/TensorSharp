// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// CPU descriptor, bounded chunks and existing bit-bridge wiring only; no GPU/model qualification.
using System.Buffers.Binary;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class Qwen35W8OutputHeadCompanionTests
{
    private sealed class UnreadSource : ICudaWeightSource
    {
        public int Reads;
        public void Read(long offset, IntPtr destination, long bytes)
        {
            Reads++;
            throw new InvalidOperationException("A metadata/default-path check must not read weights.");
        }
    }
    private static string ModulePath => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "not-opened-w8-module.ptx"));
    private static CudaW8HeadWeightDesc Desc(UnreadSource? source = null, long offset = 0, long bytes = 1_350_860_800,
        int rows = 248320, int columns = 5120, string? module = null, string provenance = "CPU-only declared source", string profile = "companion-v1", string? sha = null,
        CudaW8HeadScope scope = CudaW8HeadScope.TargetAndDraft)
        => new(source ?? new(), offset, bytes, rows, columns, module ?? ModulePath, provenance, profile, sha, scope);
    private static long XBytes(int rows) => (long)rows * 5120 * 4;
    private static long YBytes(int rows) => (long)rows * 248320 * 4;
    private static CudaWeightDesc OriginalQ6Head => new() { GgmlType=14, Ne0=5120, Ne1=248320, Ne2=1, RowBytes=4200 };

    [Fact]
    public void DescriptorSnapshotsExplicitMetadataWithoutReadingOrOpeningAnything()
    {
        var source = new UnreadSource();var d = Desc(source, 123, sha: new string('A',64));
        Assert.Same(source,d.Source);Assert.Equal(123,d.SourceOffset);Assert.Equal(1_350_860_800,d.Bytes);
        Assert.Equal(ModulePath,d.ModulePath);Assert.Equal(new string('a',64),d.ExpectedPayloadSha256);
        Assert.Equal(0,source.Reads);Assert.False(File.Exists(ModulePath));
    }
    [Fact]
    public void DescriptorRejectsForeignGeometryUnboundedRangesAndAmbiguousIdentity()
    {
        Assert.Throws<ArgumentNullException>(()=>new CudaW8HeadWeightDesc(null!,0,1_350_860_800,248320,5120,ModulePath,"source"));
        Assert.Throws<ArgumentException>(()=>Desc(offset:-1));Assert.Throws<ArgumentException>(()=>Desc(offset:long.MaxValue));
        Assert.Throws<ArgumentException>(()=>Desc(bytes:1_350_860_799));Assert.Throws<ArgumentException>(()=>Desc(rows:248077));Assert.Throws<ArgumentException>(()=>Desc(columns:4096));
        Assert.Throws<ArgumentException>(()=>Desc(module:"relative.ptx"));Assert.Throws<ArgumentException>(()=>Desc(module:Path.Combine(AppContext.BaseDirectory,"..","ambiguous.ptx")));
        Assert.Throws<ArgumentException>(()=>Desc(provenance:" "));Assert.Throws<ArgumentException>(()=>Desc(provenance:new string('x',8193)));
        Assert.Throws<ArgumentException>(()=>Desc(profile:"bad:profile"));Assert.Throws<ArgumentException>(()=>Desc(profile:new string('a',129)));
        Assert.Throws<ArgumentException>(()=>Desc(sha:new string('g',64)));Assert.Throws<ArgumentException>(()=>Desc(sha:new string('a',63)));
    }
    [Fact]
    public void ScopeDefaultsPreserveSharedRecipeAndDraftGetsItsOwnProfileWithoutReading()
    {
        var source=new UnreadSource();
        var shared=new CudaW8HeadWeightDesc(source,0,1_350_860_800,248320,5120,ModulePath,"source");
        var draft=new CudaW8HeadWeightDesc(source,0,1_350_860_800,248320,5120,ModulePath,"source",scope:CudaW8HeadScope.DraftOnly);
        Assert.Equal(CudaW8HeadScope.TargetAndDraft,shared.Scope);Assert.Equal(CudaW8HeadWeightDesc.SharedProfile,shared.Profile);
        Assert.Equal(CudaW8HeadScope.DraftOnly,draft.Scope);Assert.Equal(CudaW8HeadWeightDesc.DraftProfile,draft.Profile);
        Assert.Equal("custom-profile",Desc(profile:"custom-profile",scope:CudaW8HeadScope.DraftOnly).Profile);
        Assert.Equal(0,source.Reads);
    }
    [Theory]
    [InlineData(-1)][InlineData(2)]
    public void UnknownScopeCannotReachAnEngineOrReadWeights(int scope)
        =>Assert.Throws<ArgumentOutOfRangeException>(()=>Desc(scope:(CudaW8HeadScope)scope));
    [Theory]
    [InlineData(CudaW8HeadScope.TargetAndDraft,true)]
    [InlineData(CudaW8HeadScope.DraftOnly,false)]
    public void ActualEnginePolicyRetainsQ6TargetOnlyForDraftScope(CudaW8HeadScope scope,bool targetW8)
    {
        var source=new UnreadSource();var head=Desc(source,scope:scope);
        var engine=(Q35CudaEngine)RuntimeHelpers.GetUninitializedObject(typeof(Q35CudaEngine));
        typeof(Q35CudaEngine).GetField("_outputW8",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(engine,head);
        Assert.True(engine.OutputW8Enabled);Assert.True(engine.OutputW8DraftEnabled);
        Assert.Equal(targetW8,engine.OutputW8TargetEnabled);Assert.Equal(targetW8,Q35CudaEngine.UsesW8TargetHead(head));
        Q35CudaEngine.ValidateOutputW8Model(new(){NEmbd=5120,NVocab=248320,Output=OriginalQ6Head},head,false,false);
        Assert.Equal(0,source.Reads);
    }
    [Fact]
    public void ActualTargetHeadAndLoaderUseScopeWhileDraftSelectorQuantizationStaysUnconditional()
    {
        var constructor=typeof(Q35CudaEngine).GetConstructor(new[]{typeof(Q35CudaEngine.ModelDesc)})!;
        var head=typeof(Q35CudaEngine).GetMethod("Head",BindingFlags.Instance|BindingFlags.NonPublic)!;
        var headCode=Instructions(head);var constructorCode=Instructions(constructor);
        Assert.Contains(constructorCode,x=>x.Member?.Name=="get_OutputW8TargetEnabled");
        Assert.Contains(headCode,x=>x.Member?.Name=="get_OutputW8TargetEnabled");
        Assert.DoesNotContain(headCode,x=>x.Member?.Name=="_outputW8");
        Assert.Contains(headCode,x=>x.Member?.Name=="QuantizeQ81");Assert.Contains(headCode,x=>x.Member?.Name=="GemvQ6k");
        Assert.True(SelectorQuantizationDominatesHeadBranch(typeof(Q35CudaEngine).GetMethod("DraftPass",BindingFlags.Instance|BindingFlags.NonPublic)!));
    }
    [Fact]
    public void ScopeSeparatesCacheIdentityEvenWithIdenticalPayloadModuleAndCustomProfile()
    {
        string payload=new('a',64),module=new('b',64);
        string shared=Q35CudaEngine.OutputW8StateSuffix("p",payload,module);
        string draft=Q35CudaEngine.OutputW8StateSuffix("p",payload,module,CudaW8HeadScope.DraftOnly);
        Assert.NotEqual(shared,draft);Assert.StartsWith("|draft-head-w8-bf16-chunk33-v1:",draft);
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.OutputW8StateSuffix("p",payload,module,(CudaW8HeadScope)2));
    }
    [Theory]
    [InlineData(1,1)] [InlineData(33,1)] [InlineData(34,2)] [InlineData(64,2)] [InlineData(112,4)]
    public void CompleteTargetAndDraftRowsUseContiguousIndependent33RowChunks(int rows,int count)
    {
        int next=0;var chunks=new List<Q35CudaEngine.OutputW8Chunk>();
        while(next<rows)
        {
            var c=Q35CudaEngine.NextOutputW8Chunk(next,rows,XBytes(rows),YBytes(rows));
            Assert.Equal(next,c.FirstRow);Assert.InRange(c.Rows,1,33);
            Assert.Equal(XBytes(next),c.InputByteOffset);Assert.Equal(YBytes(next),c.OutputByteOffset);
            chunks.Add(c);next+=c.Rows;
        }
        Assert.Equal(rows,next);Assert.Equal(count,chunks.Count);Assert.Equal(rows,chunks.Sum(c=>c.Rows));
        Assert.Equal(337_920,Q35CudaEngine.OutputW8InputBytes);Assert.Equal(16_389_120,Q35CudaEngine.OutputW8LogitsBytes);
    }
    [Fact]
    public void EvenTheFirstChunkRequiresTheWholeGlobalF32DestinationCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.NextOutputW8Chunk(0,112,XBytes(112),YBytes(33)));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.NextOutputW8Chunk(0,112,XBytes(111),YBytes(112)));
        foreach(int n in new[]{0,-1})Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.NextOutputW8Chunk(0,n,0,0));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.NextOutputW8Chunk(34,34,XBytes(34),YBytes(34)));
    }
    [Fact]
    public void WholeF32BuffersRejectOverlapBadAlignmentAndPointerOverflowBeforeAnyLaunch()
    {
        IntPtr x=new(0x100000000L),y=new(0x200000000L);
        Q35CudaEngine.ValidateOutputW8Buffers(x,112,XBytes(112),y,YBytes(112));
        Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ValidateOutputW8Buffers(x,112,XBytes(112)-1,y,YBytes(112)));
        Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ValidateOutputW8Buffers(x,112,XBytes(112),y,YBytes(112)-1));
        Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ValidateOutputW8Buffers(x,1,XBytes(1),x+4,YBytes(1)));
        Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ValidateOutputW8Buffers(x+1,1,XBytes(1),y,YBytes(1)));
        Assert.Throws<ArgumentException>(()=>Q35CudaEngine.ValidateOutputW8Buffers(x,1,XBytes(1),new IntPtr(long.MaxValue-3),YBytes(1)));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.ValidateOutputW8Buffers(x,0,0,y,0));
        Assert.Equal(new IntPtr(y.ToInt64()+YBytes(99)),Q35CudaEngine.OutputW8RowPointer(y,YBytes(99)));
        Assert.Throws<OverflowException>(()=>Q35CudaEngine.OutputW8RowPointer(new IntPtr(long.MaxValue-3),4));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.OutputW8RowPointer(y,-4));
    }
    [Fact]
    public void ScaleValidationPreservesOriginalFiniteBitsIncludingSignedZeroAndNegativeScales()
    {
        ushort[] finite={0,0x8000,1,0x8001,0x03ff,0x3c00,0xbc00,0x7bff,0xfbff};
        byte[] bytes=new byte[finite.Length*2];for(int i=0;i<finite.Length;i++)BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i*2),finite[i]);
        Q35CudaEngine.ValidateOutputW8Scales(bytes,CudaW8G32HeadModule.ScalesOffset);
        Q35CudaEngine.ValidateOutputW8Scales(bytes,CudaW8G32HeadModule.WeightBytes-bytes.Length);
        Q35CudaEngine.ValidateOutputW8Scales(new byte[]{0xff,0x7f},0); // Signed codes are not half scales.
        Q35CudaEngine.ValidateOutputW8Scales(new byte[]{0xff,0x7f,0,0x3c},CudaW8G32HeadModule.ScalesOffset-2);
    }
    [Theory]
    [InlineData(0x7c00)] [InlineData(0xfc00)] [InlineData(0x7c01)] [InlineData(0x7fff)] [InlineData(0xffff)]
    public void NonfiniteScaleAtAnyChunkBoundaryIsRejected(int bits)
    {
        byte[] bytes=new byte[4];BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2),(ushort)bits);
        Assert.Throws<InvalidOperationException>(()=>Q35CudaEngine.ValidateOutputW8Scales(bytes,CudaW8G32HeadModule.ScalesOffset));
        Assert.Throws<InvalidOperationException>(()=>Q35CudaEngine.ValidateOutputW8Scales(bytes,CudaW8G32HeadModule.ScalesOffset-2));
        Assert.Throws<InvalidOperationException>(()=>Q35CudaEngine.ValidateOutputW8Scales(bytes,CudaW8G32HeadModule.WeightBytes-4));
    }
    [Fact]
    public void PartialHalvesOrRangesOutsideThePayloadCannotEvadeScaleValidation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.ValidateOutputW8Scales(new byte[1],0));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.ValidateOutputW8Scales(new byte[2],1));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.ValidateOutputW8Scales(new byte[2],CudaW8G32HeadModule.WeightBytes));
        Assert.Throws<ArgumentOutOfRangeException>(()=>Q35CudaEngine.ValidateOutputW8Scales(new byte[2],long.MaxValue-1));
    }
    [Fact]
    public void OffKeepsTheOriginalNamespaceAndDoesNotInspectACompanionOrGraphs()
    {
        Q35CudaEngine.ValidateOutputW8Model(null!,null!,true,true);
        var engine=(Q35CudaEngine)RuntimeHelpers.GetUninitializedObject(typeof(Q35CudaEngine));
        Assert.False(engine.OutputW8Enabled);Assert.Equal(string.Empty,engine.KVStateNamespaceSuffix);
        typeof(Q35CudaEngine).GetField("_verifyDecodeQ81",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(engine,true);
        Assert.Equal("|verify-decode=q8_1-v1",engine.KVStateNamespaceSuffix);
    }
    [Fact]
    public void OnRejectsIncompatibleBodyOrGraphPoliciesBeforeAnyCudaCall()
    {
        var source=new UnreadSource();var head=Desc(source);var m=new Q35CudaEngine.ModelDesc{NEmbd=5120,NVocab=248320,Output=OriginalQ6Head};
        Q35CudaEngine.ValidateOutputW8Model(m,head,false,false);
        Assert.Throws<NotSupportedException>(()=>Q35CudaEngine.ValidateOutputW8Model(m,head,true,false));
        Assert.Throws<NotSupportedException>(()=>Q35CudaEngine.ValidateOutputW8Model(m,head,false,true));
        m.Output=new CudaWeightDesc{GgmlType=2};Assert.Throws<NotSupportedException>(()=>Q35CudaEngine.ValidateOutputW8Model(m,head,false,false));
        m.Output=OriginalQ6Head;m.Output.Ne0=4096;Assert.Throws<NotSupportedException>(()=>Q35CudaEngine.ValidateOutputW8Model(m,head,false,false));
        m.Output=OriginalQ6Head;m.Output.RowBytes=4199;Assert.Throws<NotSupportedException>(()=>Q35CudaEngine.ValidateOutputW8Model(m,head,false,false));
        m.Output=OriginalQ6Head;m.NVocab=248077;Assert.Throws<NotSupportedException>(()=>Q35CudaEngine.ValidateOutputW8Model(m,head,false,false));
        Assert.Equal(0,source.Reads);
    }
    [Fact]
    public void NamespaceSeparatesActualPayloadModuleAndProfileWhileNormalizingOnlyHexCase()
    {
        string a=new('a',64),b=new('b',64),c=new('c',64);string expected="|output-head-w8-bf16-chunk33-v1:companion-v1:"+a+":"+b;
        Assert.Equal(expected,Q35CudaEngine.OutputW8StateSuffix("companion-v1",a.ToUpperInvariant(),b));
        Assert.NotEqual(expected,Q35CudaEngine.OutputW8StateSuffix("companion-v1",c,b));Assert.NotEqual(expected,Q35CudaEngine.OutputW8StateSuffix("companion-v2",a,b));Assert.NotEqual(expected,Q35CudaEngine.OutputW8StateSuffix("companion-v1",a,c));
        Assert.Throws<ArgumentException>(()=>Q35CudaEngine.OutputW8StateSuffix("bad:profile",a,b));Assert.Throws<ArgumentException>(()=>Q35CudaEngine.OutputW8StateSuffix("p",new string('g',64),b));
    }
    [Fact]
    public void BitBridgeResolvesAndLoadsOnlyTheExistingSingleFlatModuleFunction()
    {
        var calls=new List<string>();
        IntPtr function=CudaKernels.ResolveW8HeadBitBridge(name=>{calls.Add("find:"+name);return new IntPtr(42);},f=>calls.Add("load:"+f));
        Assert.Equal(new IntPtr(42),function);Assert.Equal(new[]{"find:ts_h3_weight_bf16_f32","load:42"},calls);
        bool loaded=false;Assert.Throws<NotSupportedException>(()=>CudaKernels.ResolveW8HeadBitBridge(_=>IntPtr.Zero,_=>loaded=true));Assert.False(loaded);
        Assert.Throws<InvalidOperationException>(()=>CudaKernels.ResolveW8HeadBitBridge(_=>new IntPtr(42),_=>throw new InvalidOperationException("load failed")));
    }

    [Fact]
    public void DraftSelectorInputIsQuantizedBeforeEitherHeadPolicyBranch()
    {
        // Inspect the actual production method: the later selector DMatMul consumes
        // the same Q81 buffers, even when the output head itself consumes F32.
        var method=typeof(Q35CudaEngine).GetMethod("DraftPass",BindingFlags.Instance|BindingFlags.NonPublic)!;
        Assert.True(SelectorQuantizationDominatesHeadBranch(method));
    }
    internal static bool SelectorQuantizationDominatesHeadBranch(MethodInfo method)
    {
        var code=Instructions(method);
        int gather=code.FindLastIndex(x=>x.Member?.Name=="DfGather");
        int policy=code.FindIndex(gather+1,x=>x.Member?.Name=="_outputW8");
        int quant=code.FindIndex(gather+1,x=>x.Member?.Name=="DQuant");
        int head=code.FindIndex(gather+1,x=>x.Member?.Name=="OutputW8Head");
        int selector=code.FindIndex(head+1,x=>x.Member?.Name=="DMatMul");
        return gather>=0 && gather<quant && quant<policy && policy<head && head<selector
            && !code.Skip(gather+1).Take(quant-gather-1).Any(x=>x.Code.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch or FlowControl.Return);
    }
    private static List<(OpCode Code,MemberInfo? Member)> Instructions(MethodBase method)
    {
        var opcodes=typeof(OpCodes).GetFields(BindingFlags.Public|BindingFlags.Static)
            .Where(f=>f.FieldType==typeof(OpCode)).Select(f=>(OpCode)f.GetValue(null)!).ToDictionary(x=>x.Value);
        byte[] il=method.GetMethodBody()!.GetILAsByteArray()!;
        var result=new List<(OpCode,MemberInfo?)>();
        for(int i=0;i<il.Length;)
        {
            short value=il[i++];if(value==0xfe)value=unchecked((short)(0xfe00|il[i++]));
            OpCode op=opcodes[value];MemberInfo? member=null;
            if(op.OperandType==OperandType.InlineMethod)member=method.Module.ResolveMethod(BitConverter.ToInt32(il,i));
            if(op.OperandType==OperandType.InlineField)member=method.Module.ResolveField(BitConverter.ToInt32(il,i));
            int bytes=op.OperandType switch
            {
                OperandType.InlineNone=>0,
                OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar=>1,
                OperandType.InlineVar=>2,
                OperandType.InlineI8 or OperandType.InlineR=>8,
                OperandType.InlineSwitch=>checked(4+4*BitConverter.ToInt32(il,i)),
                _=>4
            };
            result.Add((op,member));i=checked(i+bytes);
        }
        return result;
    }
}
