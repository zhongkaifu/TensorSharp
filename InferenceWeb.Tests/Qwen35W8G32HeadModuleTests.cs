// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// CPU ABI/dispatch tests only: no CUDA load, module or model qualification.
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

namespace InferenceWeb.Tests;
public sealed unsafe class Qwen35W8G32HeadModuleTests
{
    [Fact]
    public void DriverConfigAndOutputMatchTheDocumented64BitAbi()
    {
        CudaW8G32HeadModule.ValidateManagedAbi();
        Assert.Equal(16, sizeof(CudaW8G32HeadModule.OutputArgument));
        Assert.Equal(72, sizeof(CudaDriverApi.CudaLaunchAttribute));
        Assert.Equal(56, sizeof(CudaDriverApi.CudaLaunchConfig));
        var output = new CudaW8G32HeadModule.OutputArgument { Data = new IntPtr(0x1122334455667788L), LeadingDimension = 248320, ReservedPadding = 0 };
        ReadOnlySpan<byte> bytes = new(&output, sizeof(CudaW8G32HeadModule.OutputArgument));
        Assert.Equal(0x1122334455667788UL, MemoryMarshal.Read<ulong>(bytes));
        Assert.Equal(248320, MemoryMarshal.Read<int>(bytes[8..]));
        Assert.Equal(0, MemoryMarshal.Read<int>(bytes[12..]));
        var attr = new CudaDriverApi.CudaLaunchAttribute { Id = 6, ProgrammaticStreamSerializationAllowed = 1 };
        ReadOnlySpan<byte> attributeBytes = new(&attr, sizeof(CudaDriverApi.CudaLaunchAttribute));
        Assert.Equal(6, MemoryMarshal.Read<int>(attributeBytes)); Assert.Equal(1, MemoryMarshal.Read<int>(attributeBytes[8..]));
        Assert.All(attributeBytes[12..].ToArray(), x => Assert.Equal(0, x));
    }
    [Theory]
    [InlineData(1,8,8)] [InlineData(4,8,8)] [InlineData(5,8,8)] [InlineData(8,8,8)]
    [InlineData(9,16,8)] [InlineData(16,16,8)] [InlineData(17,24,8)] [InlineData(24,24,8)]
    [InlineData(25,32,8)] [InlineData(32,32,8)] [InlineData(33,40,4)]
    public void OriginalScheduleAndGridArePreserved(int n, int tile, int warps)
    {
        var p = CudaW8G32HeadModule.Plan(n);
        Assert.Equal(n,p.Rows); Assert.Equal(tile,p.TileTokens); Assert.Equal(warps,p.KWarps);
        Assert.Equal(warps*32,p.Threads); Assert.Equal(2,p.MinimumBlocks); Assert.Equal(15520u,p.GridX);
    }
    [Theory] [InlineData(0)] [InlineData(-1)] [InlineData(34)] [InlineData(int.MaxValue)]
    public void UnsupportedRowsHaveNoAlternateMathFallback(int n) => Assert.Throws<ArgumentOutOfRangeException>(() => CudaW8G32HeadModule.Plan(n));
    [Fact]
    public void NamesMatchEightActualNvccOriginalElfSymbols()
    {
        using var stream = typeof(Qwen35W8G32HeadModuleTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Qwen35W8Head.official-nvcc-symbols.json");
        Assert.NotNull(stream);
        using var fixture = JsonDocument.Parse(stream!);
        var ns = new HashSet<int>();
        foreach (var x in fixture.RootElement.GetProperty("symbols").EnumerateArray())
        {
            string name = x.GetString()!;
            Assert.True(CudaW8G32HeadModule.TryParseSymbol(name,out int n)); ns.Add(n);
            Assert.Equal(name,CudaW8G32HeadModule.ExpectedSymbol(n));
        }
        Assert.Equal(new[]{1,7,8,14,16,28,32,33},ns.Order().ToArray());
    }
    [Fact]
    public void NameParserRejectsWrongGeometrySchedulePolicyOrSuffix()
    {
        string valid = CudaW8G32HeadModule.ExpectedSymbol(7);
        foreach (string bad in new[]{valid.Replace("Li248320E","Li5120E"),valid.Replace("Li5120EEE","Li10240EEE"),valid.Replace("Li7ENS1","Li07ENS1"),valid.Replace("Li2ELNS1","Li3ELNS1"),valid.Replace("ScaleAccessE1","ScaleAccessE0"),valid+"junk",valid.Replace("ELb0EEE","ELb1EEE"),"ts_w8_tc_7"})
            Assert.False(CudaW8G32HeadModule.TryParseSymbol(bad,out _),bad);
    }
    private static CudaFunctionSymbol[] Symbols() => Enumerable.Range(0,34).Select(n => new CudaFunctionSymbol(n==0?CudaW8G32HeadModule.CastSymbol:CudaW8G32HeadModule.ExpectedSymbol(n),(IntPtr)(n+1))).ToArray();
    [Fact]
    public void BindingRequiresAll33UniqueFunctionsPlusCast()
    {
        var valid = Symbols(); Assert.Equal(valid,CudaW8G32HeadModule.ValidateSymbols(valid.Reverse().ToArray()));
        Assert.Throws<InvalidOperationException>(() => CudaW8G32HeadModule.ValidateSymbols(valid[..^1]));
        foreach (int mode in new[]{0,1,2})
        {
            var bad = Symbols();
            bad[33] = mode==0 ? bad[32] : mode==1 ? bad[33] with { Function=IntPtr.Zero } : bad[33] with { Name="unknown" };
            Assert.Throws<InvalidOperationException>(() => CudaW8G32HeadModule.ValidateSymbols(bad));
        }
    }
    [Fact]
    public void DeviceParameterLayoutIsIndependentOfManagedSizeAssertions()
    {
        ulong[] offset={0,8,16,24,40,41},size={8,8,8,16,1,1};
        CudaW8G32HeadModule.ValidateParameterLayout(offset,size,offset,size);
        for(int i=0;i<6;i++)
        {
            var o=(ulong[])offset.Clone();o[i]++;Assert.Throws<InvalidOperationException>(() => CudaW8G32HeadModule.ValidateParameterLayout(o,size,offset,size));
            var s=(ulong[])size.Clone();s[i]++;Assert.Throws<InvalidOperationException>(() => CudaW8G32HeadModule.ValidateParameterLayout(offset,s,offset,size));
        }
        Assert.Throws<InvalidOperationException>(() => CudaW8G32HeadModule.ValidateParameterLayout(offset[..^1],size[..^1],offset,size));
    }
    [Fact]
    public void FormatBytesAndCallerBoundsAreExactAndNoNegativeOrOverflowPointerIsAccepted()
    {
        Assert.Equal(248320L*5120,CudaW8G32HeadModule.CodesBytes);
        Assert.Equal(248320L*(5120/32)*2,CudaW8G32HeadModule.ScalesBytes);
        Assert.Equal(CudaW8G32HeadModule.CodesBytes+CudaW8G32HeadModule.ScalesBytes,CudaW8G32HeadModule.WeightBytes);
        Assert.Equal(0x100000000UL,CudaW8G32HeadModule.CheckBuffer(new IntPtr(0x100000000L),1350860800,1350860800,16));
        foreach(var (p,n) in new[]{(IntPtr.Zero,1350860800L),(new IntPtr(0x100000001L),1350860800L),(new IntPtr(0x100000000L),1350860799L),((IntPtr)(-16),1350860800L),(new IntPtr(long.MaxValue-15),1350860800L)})
            Assert.Throws<ArgumentException>(() => CudaW8G32HeadModule.CheckBuffer(p,n,1350860800,16));
        CudaW8G32HeadModule.Disjoint(1024,32,1056,64);
        Assert.Throws<ArgumentException>(() => CudaW8G32HeadModule.Disjoint(1024,33,1056,64));
        Assert.Throws<ArgumentException>(() => CudaW8G32HeadModule.Disjoint(1056,64,1024,33));
    }
    [Fact]
    public void EverySelectedFunctionIsLoadedBeforeInspection()
    {
        var observed = new List<(string Kind, IntPtr Handle)>();
        var loaded = new HashSet<IntPtr>();
        foreach (var symbol in Symbols())
        {
            int n = checked((int)symbol.Function - 1);
            int threads = n == 0 ? 256 : CudaW8G32HeadModule.Plan(n).Threads;
            var info = CudaW8G32HeadModule.PrepareFunction(n, symbol.Name, symbol.Function, threads,
                f => { observed.Add(("load", f)); Assert.True(loaded.Add(f)); },
                (row, name, f, block) =>
                {
                    Assert.Contains(f, loaded); observed.Add(("inspect", f));
                    Assert.Equal(n, row); Assert.Equal(symbol.Name, name); Assert.Equal(threads, block);
                    return new CudaW8G32HeadModule.KernelInfo(row, name, block, 48, 0, 0, 1);
                });
            Assert.Equal(n, info.Rows);
        }
        Assert.Equal(34, loaded.Count);
        Assert.Equal(68, observed.Count);
        for (int i = 0; i < observed.Count; i += 2)
        {
            Assert.Equal("load", observed[i].Kind); Assert.Equal("inspect", observed[i + 1].Kind);
            Assert.Equal(observed[i].Handle, observed[i + 1].Handle);
        }
    }
    [Fact]
    public void LoadFailurePropagatesWithoutInspectingOrUsingTheHandle()
    {
        bool inspected = false; int loads = 0; var expected = new InvalidOperationException("driver load failed");
        var actual = Assert.Throws<InvalidOperationException>(() => CudaW8G32HeadModule.PrepareFunction(7, "selected", (IntPtr)7, 256,
            _ => { loads++; throw expected; },
            (_, _, _, _) => { inspected = true; return default; }));
        Assert.Same(expected, actual); Assert.Equal(1, loads); Assert.False(inspected);
    }
    [Fact]
    public void NullHandleIsRejectedBeforeLoadingOrInspection()
    {
        int calls = 0;
        Assert.Throws<ArgumentException>(() => CudaW8G32HeadModule.PrepareFunction(0, "null", IntPtr.Zero, 256,
            _ => calls++, (_, _, _, _) => { calls++; return default; }));
        Assert.Equal(0, calls);
    }

}
