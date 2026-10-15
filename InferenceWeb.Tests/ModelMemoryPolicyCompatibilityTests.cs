// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using TensorSharp;
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

public sealed class ModelMemoryPolicyCompatibilityTests
{
    [Fact]
    public void OriginalClrConstructorSignaturesAndOptionalDefaultsRemainAvailable()
    {
        AssertConstructor(typeof(ModelBase), BindingFlags.NonPublic,
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup), typeof(int), typeof(WeightStreamingOptions)],
            [1, null, 1, null]);
        AssertConstructor(typeof(Gemma4Model), BindingFlags.Public,
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup), typeof(WeightStreamingOptions), typeof(string)],
            [1, null, null, null]);
        AssertConstructor(typeof(Qwen35Model), BindingFlags.Public,
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup), typeof(string), typeof(WeightStreamingOptions)],
            [1, null, null, null]);
        AssertConstructor(typeof(ModelCreateContext), BindingFlags.Public,
            [typeof(string), typeof(BackendType), typeof(GgufFile), typeof(int), typeof(ITensorParallelGroup), typeof(string), typeof(int), typeof(WeightStreamingOptions)],
            [1, null, null, 1, null]);
    }

    [Fact]
    public void OriginalFactoryBindsAnExistingSevenArgumentDelegate()
    {
        var method = typeof(ModelBase).GetMethod(nameof(ModelBase.Create),
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup), typeof(string), typeof(int), typeof(WeightStreamingOptions)]);
        Assert.NotNull(method);
        Assert.NotNull(method.CreateDelegate<LegacyFactory>());
        AssertDefaults(method.GetParameters().Skip(2).ToArray(), [1, null, null, 1, null]);
        var policyMethod = typeof(ModelBase).GetMethod(nameof(ModelBase.Create),
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup), typeof(string), typeof(int), typeof(WeightStreamingOptions), typeof(ModelMemoryPolicy)]);
        Assert.NotNull(policyMethod);
        Assert.All(policyMethod.GetParameters(), parameter => Assert.False(parameter.IsOptional));
        Assert.NotNull(typeof(ModelCreateContext).GetMethod(nameof(ModelCreateContext.With),
            [typeof(int), typeof(ITensorParallelGroup), typeof(int)]));
    }

    [Fact]
    public void ExistingShortAndUntypedDefaultCallsRemainUnambiguousAlongsidePolicyCalls()
    {
        var policy = new ModelMemoryPolicy(128, 32);
        // These delegates deliberately remain uninvoked: compiling the call sites
        // verifies overload resolution independently of a checkpoint or backend.
        Func<ModelBase>[] calls =
        [
            () => ModelBase.Create("fixture.gguf", BackendType.Cpu),
            () => new Gemma4Model("fixture.gguf", BackendType.Cpu),
            () => new Qwen35Model("fixture.gguf", BackendType.Cpu),
            () => new DerivedModel("fixture.gguf"),
            () => ModelBase.Create("fixture.gguf", BackendType.Cpu, default),
            () => new Gemma4Model("fixture.gguf", BackendType.Cpu, default),
            () => new Qwen35Model("fixture.gguf", BackendType.Cpu, default),
            () => ModelBase.Create("fixture.gguf", BackendType.GgmlCuda, 1, null, null, 1, null, memoryPolicy: policy),
            () => new Gemma4Model("fixture.gguf", BackendType.GgmlCuda, 1, null, null, null, memoryPolicy: policy),
            () => new Qwen35Model("fixture.gguf", BackendType.GgmlCuda, 1, null, null, null, memoryPolicy: policy),
            () => new DerivedModel("fixture.gguf", policy)
        ];
        Assert.All(calls, call => Assert.NotNull(call));
        Func<ModelCreateContext> context = () => new("fixture.gguf", BackendType.Cpu, null, default);
        Assert.NotNull(context);
    }

    [Fact]
    public void ContextWithRetainsPolicyAndLegacyConstructionRetainsDefaults()
    {
        var legacy = new ModelCreateContext("fixture.gguf", BackendType.Cpu, null);
        Assert.Null(legacy.MemoryPolicy);
        Assert.Null(legacy.With(2, null, 1).MemoryPolicy);
        Assert.Equal(1, legacy.TpDegree);
        Assert.Equal(1, legacy.LayerSplitDegree);

        var policy = new ModelMemoryPolicy(128, 32);
        var context = new ModelCreateContext("fixture.gguf", BackendType.GgmlCuda, null,
            1, null, "draft.gguf", 1, null, memoryPolicy: policy);
        var revised = context.With(2, null, 3);
        Assert.Same(policy, revised.MemoryPolicy);
        Assert.Equal(context.GgufPath, revised.GgufPath);
        Assert.Equal(context.DraftModelPath, revised.DraftModelPath);
        Assert.Equal(context.Backend, revised.Backend);
        Assert.Equal(2, revised.TpDegree);
        Assert.Equal(3, revised.LayerSplitDegree);
        Assert.Equal(1, context.TpDegree);
    }

    private delegate ModelBase LegacyFactory(string path, BackendType backend, int tpDegree,
        ITensorParallelGroup group, string draftPath, int layerSplitDegree, WeightStreamingOptions streaming);

    private static void AssertConstructor(Type type, BindingFlags visibility, Type[] parameters, object?[] defaults)
    {
        var ctor = type.GetConstructor(visibility | BindingFlags.Instance, null, parameters, null);
        Assert.NotNull(ctor);
        AssertDefaults(ctor.GetParameters().TakeLast(defaults.Length).ToArray(), defaults);
        var policyCtor = type.GetConstructor(visibility | BindingFlags.Instance, null,
            [.. parameters, typeof(ModelMemoryPolicy)], null);
        Assert.NotNull(policyCtor);
        Assert.All(policyCtor.GetParameters(), parameter => Assert.False(parameter.IsOptional));
    }

    private static void AssertDefaults(ParameterInfo[] parameters, object?[] defaults)
    {
        Assert.Equal(defaults.Length, parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(parameters[i].IsOptional);
            Assert.Equal(defaults[i], parameters[i].DefaultValue);
        }
    }

    private sealed class DerivedModel : ModelBase
    {
        public DerivedModel(string path) : base(path, BackendType.Cpu) { }
        public DerivedModel(string path, ModelMemoryPolicy policy)
            : base(path, BackendType.Cpu, 1, null, 1, null, memoryPolicy: policy) { }
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }
}
