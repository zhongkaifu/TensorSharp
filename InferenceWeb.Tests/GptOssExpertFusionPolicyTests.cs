// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Memory;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class GptOssExpertFusionPolicyTests
{
    private const BindingFlags Fields = BindingFlags.Instance | BindingFlags.NonPublic;
    private const string FusionPolicy = "TS_WEIGHT_FUSION_COPIES";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DefaultStackedCudaPreservesEverySourceAndBiasIncludingMixedQuantTypes(bool mixedTypes)
    {
        using var environment = new EnvScope(); environment.Set(FusionPolicy, null);
        using var fixture = new FusionFixture(mixedTypes: mixedTypes);
        fixture.Fuse();
        fixture.AssertSourcesPreserved();
        Assert.Equal(!mixedTypes, Get<bool>(fixture.Model, typeof(GptOssModel), "_temporaryExpertGateUpFusion"));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("false")]
    public void ExplicitDeclineIsModelWideBeforeAnEarlierF32PairCanBeConsumed(string policy)
    {
        using var environment = new EnvScope(); environment.Set(FusionPolicy, policy);
        // Layer zero is F32; layer one is raw Q8. Deciding only at the first
        // quantized pair used to fuse/dispose layer zero before publishing split names.
        using var fixture = new FusionFixture(firstLayerFloat: true);
        fixture.Fuse();
        fixture.AssertSourcesPreserved();
        Assert.False(Get<bool>(fixture.Model, typeof(GptOssModel), "_temporaryExpertGateUpFusion"));
    }

    [Theory]
    [InlineData("1")]
    [InlineData("TrUe")]
    public void ExplicitCopiesKeepTheOriginalFusedRepresentation(string policy)
    {
        using var environment = new EnvScope(); environment.Set(FusionPolicy, policy);
        using var fixture = new FusionFixture();
        fixture.Fuse();
        Assert.False(Get<bool>(fixture.Model, typeof(GptOssModel), "_expertGateUpSplit"));
        for (int layer = 0; layer < 2; layer++)
        {
            string prefix = $"blk.{layer}.ffn_";
            Assert.False(fixture.Quant.ContainsKey(prefix + "gate_exps.0.weight"));
            Assert.False(fixture.Quant.ContainsKey(prefix + "up_exps.0.weight"));
            var fused = fixture.Quant[prefix + "gate_up_exps.0.weight"];
            Assert.Equal(4, fused.Ne1);
            Assert.Equal(136, fused.RawBytes);
            Assert.Equal(new[] { 1f + layer, 2f + layer, 3f + layer, 4f + layer },
                fixture.Float[prefix + "gate_up_exps.0.bias"].GetElementsAsFloat(4));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExplicitCopiesPreflightAllPairsBeforeConsumingCompatibleEarlierLayers(bool mixedRepresentation)
    {
        using var environment = new EnvScope(); environment.Set(FusionPolicy, "1");
        using var fixture = new FusionFixture(mixedTypes: !mixedRepresentation, mixedRepresentation: mixedRepresentation);
        fixture.Fuse();
        // Layer zero is fusible; the incompatible layer one must be detected
        // before any of layer zero's weights or biases are removed.
        fixture.AssertSourcesPreserved();
    }

    [Fact]
    public void ExplicitCopiesDoNotDiscardPerSourceScale()
    {
        using var environment = new EnvScope(); environment.Set(FusionPolicy, "1");
        using var fixture = new FusionFixture();
        fixture.Quant["blk.1.ffn_up_exps.0.weight"].Scale = 0.5f;
        fixture.Fuse();
        fixture.AssertSourcesPreserved();
        Assert.Equal(0.5f, fixture.Quant["blk.1.ffn_up_exps.0.weight"].Scale);
    }

    [Fact]
    public void TensorParallelRefusesIncompatiblePairsBeforeMutatingEarlierLayers()
    {
        using var environment = new EnvScope(); environment.Set(FusionPolicy, "1");
        using var fixture = new FusionFixture(mixedTypes: true);
        Set(fixture.Model, typeof(ModelBase), "_tpGroup", new ShapeOnlyTpGroup());
        var error = Assert.Throws<TargetInvocationException>(fixture.Fuse);
        var cause = Assert.IsType<NotSupportedException>(error.InnerException);
        Assert.Contains("layer 1, expert 0", cause.Message);
        fixture.AssertSourcesPreserved(split: false);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void SplitExpertProjectionRefusalReleasesIntermediatesAndAllowsRetry(int failAt)
        => ProjectionRefusalReleasesIntermediatesAndAllowsRetry(failAt, temporary: false);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void TemporaryFusedProjectionRefusalReleasesIntermediatesAndAllowsRetry(int failAt)
        => ProjectionRefusalReleasesIntermediatesAndAllowsRetry(failAt, temporary: true);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DownBiasReadFailureReleasesResultAndExpertIntermediates(bool temporary)
        => ProjectionRefusalReleasesIntermediatesAndAllowsRetry(int.MaxValue, temporary, failBiasRead: true);

    private static void ProjectionRefusalReleasesIntermediatesAndAllowsRetry(int failAt, bool temporary, bool failBiasRead = false)
    {
        var allocator = new ProjectionAllocator { FailAt = failAt };
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        using var input = new Tensor(cpu, DType.Float32, 1, 32);
        input.SetElementsAsFloat(Enumerable.Repeat(0.125f, 32).ToArray());
        using var gate = new QuantizedWeight(new byte[34 * 32], 8, 32, 32);
        using var up = new QuantizedWeight(new byte[34 * 32], 8, 32, 32);
        using var down = new QuantizedWeight(new byte[34 * 32], 8, 32, 32);
        var model = (GptOssModel)RuntimeHelpers.GetUninitializedObject(typeof(GptOssModel));
        GC.SuppressFinalize(model);
        Set(model, typeof(ModelBase), "_backend", BackendType.Cpu);
        Set(model, typeof(ModelBase), "_allocator", allocator);
        Set(model, typeof(ModelBase), "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
        Set(model, typeof(ModelBase), "_quantWeights", new Dictionary<string, QuantizedWeight>
            { ["gate"] = gate, ["up"] = up, ["down"] = down });
        var biasAllocator = new BiasReadAllocator();
        using var downBias = new Tensor(biasAllocator, DType.Float32, 1, 32);
        downBias.SetElementsAsFloat(new float[32]);
        Set(model, typeof(ModelBase), "_weights", failBiasRead
            ? new Dictionary<string, Tensor> { ["down.bias"] = downBias } : new Dictionary<string, Tensor>());
        biasAllocator.Fail = failBiasRead;
        using var budgetScope = new HostAllocationBudgetScope(new MemoryBudget([new("ram", 2176)]), ["ram"]);
        var method = typeof(GptOssModel).GetMethod(temporary ? "ExpertFFNTemporary" : "ExpertFFNSplit", Fields)!;
        // Two rows exercise the contiguous activation copy as well as down.
        using var twoRows = new Tensor(cpu, DType.Float32, 2, 32);
        twoRows.SetElementsAsFloat(Enumerable.Repeat(0.125f, 64).ToArray());
        Tensor Execute() => (Tensor)method.Invoke(model, [twoRows, "gate", "gate.bias", "up", "up.bias", "down", "down.bias", 2])!;

        var error = Assert.Throws<TargetInvocationException>(() => Execute());
        if (failBiasRead)
            Assert.Equal("Injected bias read failure.", Assert.IsType<IOException>(error.InnerException).Message);
        else
            Assert.Equal("Injected expert projection allocation refusal.", Assert.IsType<OutOfMemoryException>(error.InnerException).Message);
        Assert.Equal(failBiasRead ? 3 : failAt, allocator.Attempts);
        Assert.Equal(failBiasRead ? 3 : failAt - 1, allocator.Created.Count);
        Assert.All(allocator.Created, storage => Assert.Equal(IntPtr.Zero, storage.buffer));
        Assert.NotEqual(IntPtr.Zero, ((CpuStorage)input.Storage).buffer);
        Assert.All(new[] { gate, up, down }, weight => Assert.NotEqual(IntPtr.Zero, weight.Data));
        Assert.Null(Get<object>(model, typeof(GptOssModel), "_temporaryExpertGateUpOwner"));
        Assert.Equal(0, budgetScope.Usage.Bytes);

        allocator.FailAt = int.MaxValue;
        biasAllocator.Fail = false;
        using (Tensor result = Execute())
        {
            Assert.All(result.GetElementsAsFloat(32), value => Assert.Equal(0f, value));
            Assert.Single(allocator.Created.Where(storage => storage.buffer != IntPtr.Zero));
        }
        Assert.All(allocator.Created, storage => Assert.Equal(IntPtr.Zero, storage.buffer));
        Assert.Equal(0, budgetScope.Usage.Bytes);
    }

    [Fact]
    public void TemporaryWeightQuotaRefusalPreservesSourcesAndNoOwnerCredit()
    {
        using var gate = new QuantizedWeight(new byte[68], 8, 32, 2);
        using var up = new QuantizedWeight(new byte[68], 8, 32, 2);
        var budget = new MemoryBudget([new("ram", 192)]);
        budget.RefreshCapacities([new("ram", 191)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        Assert.Throws<MemoryPressureException>(() => new GptOssTemporaryExpertWeight(gate, up, false));
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(0, budget.Snapshot().Single().Committed);
        Assert.NotEqual(IntPtr.Zero, gate.Data); Assert.NotEqual(IntPtr.Zero, up.Data);
        budget.RefreshCapacities([new("ram", 192)]);
        using (var owner = new GptOssTemporaryExpertWeight(gate, up, false))
        { Assert.Equal(192, scope.Usage.Bytes); Assert.Equal(4, owner.Weight.Ne1); }
        Assert.Equal(0, scope.Usage.Bytes);
    }

    [Fact]
    public void TemporaryWeightDoesNotAliasAdjacentSourceCacheKeys()
    {
        byte[] raw = Enumerable.Range(0, 136).Select(i => (byte)i).ToArray();
        using var source = new QuantizedWeight(raw, 8, 32, 4);
        using var gate = QuantizedWeight.CreateExternalView(source.Data, 68, 8, 32, 2, source);
        using var up = QuantizedWeight.CreateExternalView(source.Data + 68, 68, 8, 32, 2, source);
        using var owner = new GptOssTemporaryExpertWeight(gate, up, false);
        Assert.NotEqual(gate.CacheKey, owner.Weight.CacheKey);
        Assert.NotEqual(up.CacheKey, owner.Weight.CacheKey);
        Assert.False(owner.Weight.HasExternalHostView);
        byte[] copied = new byte[raw.Length];
        Marshal.Copy(owner.Weight.Data, copied, 0, copied.Length);
        Assert.Equal(raw, copied);
        owner.Dispose();
        Assert.NotEqual(IntPtr.Zero, source.Data);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(9)]
    public void TemporaryFallbackMatchesFusedBiasActivationAndDownForAlternatingExperts(int rows)
    {
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        var model = (GptOssModel)RuntimeHelpers.GetUninitializedObject(typeof(GptOssModel));
        GC.SuppressFinalize(model);
        var quant = new Dictionary<string, QuantizedWeight>();
        var biases = new Dictionary<string, Tensor>();
        Set(model, typeof(ModelBase), "_backend", BackendType.Cpu);
        Set(model, typeof(ModelBase), "_allocator", cpu);
        Set(model, typeof(ModelBase), "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
        Set(model, typeof(ModelBase), "_quantWeights", quant);
        Set(model, typeof(ModelBase), "_weights", biases);
        Set(model, typeof(GptOssModel), "_expertGateUpSplit", true);
        Set(model, typeof(GptOssModel), "_temporaryExpertGateUpFusion", true);
        try
        {
            for (int expert = 0; expert < 2; expert++)
            {
                foreach (string part in new[] { "gate", "up", "down" })
                {
                    byte[] raw = new byte[34 * 32];
                    for (int r = 0; r < 32; r++)
                    {
                        BitConverter.TryWriteBytes(raw.AsSpan(r * 34, 2), (System.Half)(1f / 64));
                        for (int k = 0; k < 32; k++)
                            raw[r * 34 + 2 + k] = unchecked((byte)(sbyte)((k * 3 + r + expert * 5 + part.Length) % 17 - 8));
                    }
                    quant.Add($"{expert}.{part}", new QuantizedWeight(raw, 8, 32, 32));
                    var bias = new Tensor(cpu, DType.Float32, 1, 32);
                    bias.SetElementsAsFloat(Enumerable.Range(0, 32).Select(i => (i - 15 + expert * 3 + part.Length) / 32f).ToArray());
                    biases.Add($"{expert}.{part}.bias", bias);
                }
                quant.Add($"{expert}.fused", QuantizedWeight.ConcatOrCreateCopy(quant[$"{expert}.gate"], quant[$"{expert}.up"]));
                var fusedBias = new Tensor(cpu, DType.Float32, 1, 64);
                fusedBias.SetElementsAsFloat(biases[$"{expert}.gate.bias"].GetElementsAsFloat(32)
                    .Concat(biases[$"{expert}.up.bias"].GetElementsAsFloat(32)).ToArray());
                biases.Add($"{expert}.fused.bias", fusedBias);
            }
            using var input = new Tensor(cpu, DType.Float32, rows, 32);
            input.SetElementsAsFloat(Enumerable.Range(0, rows * 32).Select(i => (i % 31 - 15) / 16f).ToArray());
            var fusedMethod = typeof(GptOssModel).GetMethod("ExpertFFN", Fields)!;
            var forwardMethod = typeof(GptOssModel).GetMethod("ExpertForward", Fields)!;
            var budget = new MemoryBudget([new("ram", 2176)]);
            using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
            var references = new Dictionary<int, float[]>();
            foreach (int expert in new[] { 0, 1, 0, 1, 0 })
            {
                using var reference = (Tensor)fusedMethod.Invoke(model,
                    [input, $"{expert}.fused", $"{expert}.fused.bias", $"{expert}.down", $"{expert}.down.bias", rows])!;
                string[] names = [$"{expert}.gate", $"{expert}.gate.bias", $"{expert}.down", $"{expert}.down.bias", $"{expert}.up", $"{expert}.up.bias"];
                using var actual = (Tensor)forwardMethod.Invoke(model, [input, names, rows])!;
                float[] expected = reference.GetElementsAsFloat(rows * 32), observed = actual.GetElementsAsFloat(rows * 32);
                Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(observed.AsSpan())));
                references[expert] = expected;
                Assert.Null(Get<object>(model, typeof(GptOssModel), "_temporaryExpertGateUpOwner"));
                Assert.Equal(0, scope.Usage.Bytes);
            }
            Assert.False(references[0].SequenceEqual(references[1]));
            Assert.Equal(2176, budget.HighWatermarks().Single().Committed);
        }
        finally
        {
            foreach (var weight in quant.Values) weight.Dispose();
            foreach (var bias in biases.Values) bias.Dispose();
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TemporaryWeightRetainsExactOwnerAndQuotaUntilBarrierAndCacheRetirementSucceed(bool failBarrier)
    {
        using var gate = new QuantizedWeight(new byte[68], 8, 32, 2);
        using var up = new QuantizedWeight(new byte[68], 8, 32, 2);
        var budget = new MemoryBudget([new("ram", 192)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        bool fail = true;
        var events = new List<string>();
        IntPtr invalidated = IntPtr.Zero;
        using var owner = new GptOssTemporaryExpertWeight(gate, up, false,
            () => { events.Add("barrier"); if (fail && failBarrier) throw new IOException("barrier"); },
            key => { events.Add("invalidate"); invalidated = key; if (fail && !failBarrier) throw new IOException("invalidate"); });
        var weight = owner.Weight;
        // Exercise the cache identity differing from the allocation address.
        IntPtr key = weight.EnsureDeviceCacheKey();
        Assert.NotEqual(weight.Data, key);
        Assert.Throws<IOException>(owner.Dispose);
        Assert.Same(weight, owner.Weight);
        Assert.NotEqual(IntPtr.Zero, weight.Data);
        Assert.Equal(192, scope.Usage.Bytes);
        Assert.Equal(failBarrier ? new[] { "barrier" } : new[] { "barrier", "invalidate" }, events);
        fail = false;
        owner.Dispose();
        Assert.Equal(key, invalidated);
        Assert.Equal(IntPtr.Zero, weight.Data);
        Assert.Null(owner.Weight);
        Assert.Equal(0, scope.Usage.Bytes);
    }

    /// <summary>Explicit heavy model gate; run in a fresh GGML CUDA test process.
    /// This tests the unchanged stacked execution arithmetic, not the >1024-row
    /// per-expert fallback whose different projection shape needs its own oracle.</summary>
    [CudaFact("TS_TEST_GPTOSS_FUSION_MODEL", "gpt-oss|gpt_oss|gptoss", GgmlBackend = BackendType.GgmlCuda)]
    public void RealStackedPrefillAndDecodeAreBitExactWithoutUnusedExpertCopies()
    {
        string configured = Environment.GetEnvironmentVariable("TS_TEST_GPTOSS_FUSION_MODEL")!;
        string path = File.Exists(configured) ? configured
            : TestGates.FindGguf(configured, "gpt-oss|gpt_oss|gptoss")!;
        using var environment = new EnvScope();
        environment.ClearSpeculationVars();
        environment.Set("MAX_CONTEXT", "512");
        environment.Set(FusionPolicy, "1");
        var reference = Capture(path, true, null, null);
        environment.Set(FusionPolicy, null);
        var candidate = Capture(path, false, reference.Prompt, reference.Teacher);
        Assert.Equal(reference.Rows.Count, candidate.Rows.Count);
        for (int row = 0; row < reference.Rows.Count; row++)
            Assert.True(MemoryMarshal.AsBytes(reference.Rows[row].AsSpan()).SequenceEqual(
                MemoryMarshal.AsBytes(candidate.Rows[row].AsSpan())), $"Full vocabulary differs at row {row}.");
    }

    private static (int[] Prompt, int[] Teacher, List<float[]> Rows) Capture(
        string path, bool fused, int[]? fixedPrompt, int[]? fixedTeacher)
    {
        using ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda);
        Assert.IsType<GptOssModel>(model);
        var quant = Get<Dictionary<string, QuantizedWeight>>(model, typeof(ModelBase), "_quantWeights");
        var stacks = Get<Dictionary<string, StackedExpertWeights>>(model, typeof(ModelBase), "_stackedExpertWeights");
        var expertTypes = stacks.Where(p => p.Key.Contains(".ffn_gate_exps.") || p.Key.Contains(".ffn_up_exps."))
            .Select(p => p.Value.GgmlType).Distinct().Order().ToArray();
        Assert.NotEmpty(expertTypes);
        // The real gpt-oss-20b-Q8_0 filename has MXFP4 experts. Inspect tensors,
        // never infer expert quantization from the checkpoint's display name.
        Assert.All(expertTypes, type => Assert.Contains(type, new[] { 8 /* Q8_0 */, 39 /* MXFP4 */ }));
        Console.WriteLine($"GPT-OSS actual gate/up GGML types: {string.Join(",", expertTypes)}; eager fusion={fused}");
        int expectedPairs = checked(model.Config.NumLayers * model.Config.NumExperts);
        Assert.True(expectedPairs > 0);
        var copies = quant.Where(p => p.Key.Contains(".ffn_gate_up_exps.")).ToArray();
        Assert.Equal(fused ? expectedPairs : 0, copies.Length);
        if (fused) Assert.All(copies, p => Assert.True(p.Value.HasHostData && !p.Value.HasExternalHostView));
        else
        {
            Assert.True(Get<bool>(model, typeof(GptOssModel), "_expertGateUpSplit"));
            Assert.Equal(expectedPairs, quant.Keys.Count(n => n.Contains(".ffn_gate_exps.")));
            Assert.Equal(expectedPairs, quant.Keys.Count(n => n.Contains(".ffn_up_exps.")));
        }
        int[] seed = model.Tokenizer.Encode("Explain why the sky looks blue during a clear day. ", addSpecial: false).ToArray();
        Assert.NotEmpty(seed);
        int[] prompt = fixedPrompt ?? Enumerable.Range(0, 128).Select(i => seed[i % seed.Length]).ToArray();
        var rows = new List<float[]> { (float[])model.Forward(prompt).Clone() };
        int[] teacher = fixedTeacher ?? new int[4];
        for (int step = 0; step < teacher.Length; step++)
        {
            if (fixedTeacher == null) teacher[step] = ArgMax(rows[^1]);
            rows.Add((float[])model.Forward([teacher[step]]).Clone());
        }
        Assert.All(rows, row => { Assert.Equal(model.Config.VocabSize, row.Length); Assert.All(row, v => Assert.True(float.IsFinite(v))); });
        return (prompt, teacher, rows);
    }

    private static int ArgMax(float[] row)
    { int best = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i; return best; }
    private static T Get<T>(object value, Type type, string name) => (T)type.GetField(name, Fields)!.GetValue(value)!;
    private static void Set(object value, Type type, string name, object data) => type.GetField(name, Fields)!.SetValue(value, data);

    private sealed class FusionFixture : IDisposable
    {
        internal GptOssModel Model { get; } = (GptOssModel)RuntimeHelpers.GetUninitializedObject(typeof(GptOssModel));
        internal Dictionary<string, QuantizedWeight> Quant { get; } = new();
        internal Dictionary<string, Tensor> Float { get; } = new();
        private readonly Dictionary<string, QuantizedWeight> _originalQuant = new();
        private readonly Dictionary<string, Tensor> _originalFloat = new();

        internal FusionFixture(bool mixedTypes = false, bool firstLayerFloat = false, bool mixedRepresentation = false)
        {
            GC.SuppressFinalize(Model); // This fixture owns only the explicitly initialized payloads.
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            var stacked = new Dictionary<string, StackedExpertWeights>();
            Set(Model, typeof(ModelBase), "_backend", BackendType.GgmlCuda);
            Set(Model, typeof(ModelBase), "_allocator", allocator);
            Set(Model, typeof(ModelBase), "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            Set(Model, typeof(ModelBase), "<Config>k__BackingField", new ModelConfig { NumLayers = 2, HiddenSize = 32, NumExperts = 1 });
            Set(Model, typeof(ModelBase), "_quantWeights", Quant);
            Set(Model, typeof(ModelBase), "_weights", Float);
            Set(Model, typeof(ModelBase), "_stackedExpertWeights", stacked);
            Set(Model, typeof(GptOssModel), "_numExperts", 1);
            for (int layer = 0; layer < 2; layer++)
            foreach (string part in new[] { "gate", "up", "down" })
            {
                string prefix = $"blk.{layer}.ffn_{part}_exps";
                if ((firstLayerFloat && layer == 0) || (mixedRepresentation && layer == 1 && part == "up"))
                {
                    var value = new Tensor(allocator, DType.Float32, 2, 32);
                    value.SetElementsAsFloat(Enumerable.Repeat(0.125f, 64).ToArray());
                    Float.Add(prefix + ".0.weight", value);
                }
                else
                {
                    int type = mixedTypes && layer == 1 && part == "up" ? 2 : 8;
                    var value = new QuantizedWeight(new byte[(type == 8 ? 34 : 18) * 2], type, 32, 2);
                    Quant.Add(prefix + ".0.weight", value);
                    stacked.Add(prefix + ".weight", new(value.Data, type, 32, 2, 1, value.RawBytes,
                        isExternalView: true, ownerToken: value, ownedBuffer: IntPtr.Zero));
                }
                var bias = new Tensor(allocator, DType.Float32, 1, 2);
                bias.SetElementsAsFloat(part == "up" ? [3f + layer, 4f + layer] : [1f + layer, 2f + layer]);
                Float.Add(prefix + ".0.bias", bias);
            }
            foreach (var pair in Quant) _originalQuant.Add(pair.Key, pair.Value);
            foreach (var pair in Float) _originalFloat.Add(pair.Key, pair.Value);
        }

        internal void Fuse() => typeof(GptOssModel).GetMethod("FuseExpertGateUpWeights", Fields)!.Invoke(Model, null);
        internal void AssertSourcesPreserved(bool split = true)
        {
            Assert.Equal(split, Get<bool>(Model, typeof(GptOssModel), "_expertGateUpSplit"));
            Assert.Equal(_originalQuant.Count, Quant.Count);
            Assert.Equal(_originalFloat.Count, Float.Count);
            foreach (var pair in _originalQuant)
            { Assert.Same(pair.Value, Quant[pair.Key]); Assert.NotEqual(IntPtr.Zero, pair.Value.Data); }
            foreach (var pair in _originalFloat)
            { Assert.Same(pair.Value, Float[pair.Key]); Assert.NotEqual(IntPtr.Zero, ((CpuStorage)pair.Value.Storage).buffer); }
            Assert.DoesNotContain(Quant.Keys.Concat(Float.Keys), name => name.Contains("gate_up_exps"));
        }
        public void Dispose()
        {
            foreach (var value in Quant.Values.Concat(_originalQuant.Values).Distinct()) value.Dispose();
            foreach (var value in Float.Values.Concat(_originalFloat.Values).Distinct()) value.Dispose();
        }
    }

    private sealed class ShapeOnlyTpGroup : ITensorParallelGroup
    {
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }

    private sealed class ProjectionAllocator : IAllocator
    {
        internal int FailAt;
        internal int Attempts;
        internal List<CpuStorage> Created { get; } = [];
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType type, long count)
        {
            if (++Attempts == FailAt) throw new OutOfMemoryException("Injected expert projection allocation refusal.");
            var storage = new CpuStorage(this, type, count);
            Created.Add(storage);
            return storage;
        }
    }

    private sealed class BiasReadAllocator : IAllocator
    {
        internal bool Fail;
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        public Storage Allocate(DType type, long count) => new BiasReadStorage(this, type, count);
        private sealed class BiasReadStorage(BiasReadAllocator allocator, DType type, long count)
            : CpuStorage(allocator, type, count)
        {
            public override void EnsureHostReadable()
            { if (allocator.Fail) throw new IOException("Injected bias read failure."); }
        }
    }
}
