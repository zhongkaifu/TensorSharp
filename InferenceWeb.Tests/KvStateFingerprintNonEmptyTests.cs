// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// M0a (radix design DEC-37, requirement K1a): every engine-served family reports a
// non-empty KV-state fingerprint that names the shape of the state it resumes from.
//
// Why it matters: InferenceEngineHost rebuilds the engine when the fingerprint changes,
// the paged prefix cache salts every block hash with it, and persisted shared-prefix
// checkpoints are keyed by it. Qwen4Exp, DeepSeek V4/V4.1, GLM (glm-dsa and glm5next)
// and Hunyuan dense all inherited ModelBase's "" - so any two of them, or two different
// geometries of one of them, were indistinguishable to all three.
//
// These run without any GGUF: the geometry fields are set on uninitialized instances,
// which is exactly what the property reads (it must not depend on anything a load does
// later, because the host re-reads it on every request).
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class KvStateFingerprintNonEmptyTests
{
    /// <summary>Families the engine serves today, named so the discovery below cannot
    /// silently find nothing.</summary>
    private static readonly Type[] KnownEngineServedFamilies =
    {
        typeof(Qwen3Model), typeof(Qwen35Model), typeof(Qwen4ExpModel), typeof(Gemma4Model),
        typeof(GptOssModel), typeof(Mistral3Model), typeof(NemotronModel), typeof(MuseGlimmerModel),
        typeof(GlmDsaModel), typeof(DeepSeek4Model), typeof(DeepSeek41Model),
    };

    /// <summary>A family can be served by the engine when it snapshots K/V or runs the
    /// batched paged contract (InferenceEngineHost.TryGetEngine). Hunyuan dense is added
    /// explicitly: the Hunyuan engine branch makes it engine-served, and its fingerprint
    /// must exist before that lands (radix design §6.2.4).</summary>
    private static IEnumerable<Type> EngineServedFamilies()
    {
        var found = typeof(ModelBase).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && typeof(ModelBase).IsAssignableFrom(t))
            .Where(t => typeof(IBatchedPagedModel).IsAssignableFrom(t)
                || DeclaredBelowModelBase(t, nameof(ModelBase.SupportsKVStateSnapshot)))
            .ToHashSet();
        found.Add(typeof(HunyuanDenseModel));
        return found;
    }

    private static bool DeclaredBelowModelBase(Type t, string property)
        => t.GetProperty(property, BindingFlags.Public | BindingFlags.Instance)!.GetGetMethod()!.DeclaringType != typeof(ModelBase);

    [Fact]
    public void EveryEngineServedFamily_OverridesTheFingerprint()
    {
        var families = EngineServedFamilies().ToList();
        Assert.All(KnownEngineServedFamilies, known => Assert.Contains(known, families));

        var missing = families
            .Where(t => !DeclaredBelowModelBase(t, nameof(ModelBase.KVStateFingerprint)))
            .Select(t => t.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();
        Assert.True(missing.Count == 0,
            "Engine-served families still inheriting ModelBase's empty KVStateFingerprint: " + string.Join(", ", missing));
    }

    public static IEnumerable<object[]> ConfigDrivenFamilies() => new[]
    {
        // The four M0a families first; the rest pin that the pre-existing overrides keep
        // the same two properties.
        new object[] { typeof(Qwen4ExpModel) },
        new object[] { typeof(DeepSeek4Model) },
        new object[] { typeof(DeepSeek41Model) },
        new object[] { typeof(HunyuanDenseModel) },
        new object[] { typeof(Qwen3Model) },
        new object[] { typeof(Qwen35Model) },
        new object[] { typeof(Gemma4Model) },
        new object[] { typeof(GptOssModel) },
        new object[] { typeof(Mistral3Model) },
        new object[] { typeof(NemotronModel) },
        new object[] { typeof(MuseGlimmerModel) },
    };

    [Theory]
    [MemberData(nameof(ConfigDrivenFamilies))]
    public void Fingerprint_IsNonEmpty_AndVariesWithLayerCountAndKvDtype(Type family)
    {
        string baseline = Fingerprint(family, layers: 4, KvCacheDtype.F16);
        Assert.False(string.IsNullOrWhiteSpace(baseline), $"{family.Name} reports an empty fingerprint");

        Assert.Equal(baseline, Fingerprint(family, layers: 4, KvCacheDtype.F16));
        Assert.NotEqual(baseline, Fingerprint(family, layers: 6, KvCacheDtype.F16));
        Assert.NotEqual(baseline, Fingerprint(family, layers: 4, KvCacheDtype.Q8_0));
    }

    [Fact]
    public void Glm_IsNonEmpty_AndVariesWithTrunkDepthArchitectureAndCacheDtype()
    {
        // GLM's cache rows are not sized by the process-wide KV dtype: the native
        // executor allocates F16 and the per-op path F32, so the dtype component follows
        // the executor that holds the rows.
        string dsa = GlmFingerprint("glm-dsa", trunkLayers: 4, native: false);
        Assert.False(string.IsNullOrWhiteSpace(dsa));
        Assert.Contains("dtype=f32", dsa);
        Assert.Equal(dsa, GlmFingerprint("glm-dsa", trunkLayers: 4, native: false));
        Assert.NotEqual(dsa, GlmFingerprint("glm-dsa", trunkLayers: 6, native: false));
        Assert.NotEqual(dsa, GlmFingerprint("glm5next", trunkLayers: 4, native: false));

        string nativeDsa = GlmFingerprint("glm-dsa", trunkLayers: 4, native: true);
        Assert.Contains("dtype=f16", nativeDsa);
        Assert.NotEqual(dsa, nativeDsa);
    }

    [Fact]
    public void Glm_SyntheticCheckpoints_ReportDistinctNonEmptyFingerprints()
    {
        // End to end through the real constructors: the tiny glm-dsa and glm5next
        // fixtures on the per-op path, and glm-dsa again on the native executor.
        string dir = Path.Combine(Path.GetTempPath(), "ts-fp-glm-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        using var env = new EnvScope();
        env.ClearSpeculationVars();
        env.Set("MAX_CONTEXT", "256");
        env.Set("TS_GLM_NATIVE", null);
        try
        {
            string dsaPath = GlmDsaSyntheticModelBuilder.Write(Path.Combine(dir, "dsa.gguf"));
            string nextPath = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
                Path.Combine(dir, "next.gguf"), numHeads: 4, quantizeAttentionOutput: false);

            string managedDsa, managedNext, nativeDsa;
            using (var m = ModelBase.Create(dsaPath, BackendType.Cpu)) managedDsa = m.KVStateFingerprint;
            using (var m = ModelBase.Create(nextPath, BackendType.Cpu)) managedNext = m.KVStateFingerprint;
            using (var m = ModelBase.Create(dsaPath, BackendType.GgmlCpu)) nativeDsa = m.KVStateFingerprint;

            Assert.All(new[] { managedDsa, managedNext, nativeDsa },
                fp => Assert.False(string.IsNullOrWhiteSpace(fp)));
            Assert.Equal(3, new[] { managedDsa, managedNext, nativeDsa }.Distinct(StringComparer.Ordinal).Count());
            Assert.Contains("arch=glm-dsa", managedDsa);
            Assert.Contains("arch=glm5next", managedNext);
            Assert.Contains("exec=native", nativeDsa);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }

    // ------------------------------------------------------------------ helpers

    private static string Fingerprint(Type family, int layers, KvCacheDtype dtype)
    {
        var model = (ModelBase)RuntimeHelpers.GetUninitializedObject(family);
        SetField(typeof(ModelBase), model, "<Config>k__BackingField", new ModelConfig
        {
            Architecture = family.Name,
            NumLayers = layers,
            NumHeads = 8,
            NumKVHeads = 2,
            HiddenSize = 512,
        });
        SetField(typeof(ModelBase), model, "_kvCacheDtype", dtype);
        return model.KVStateFingerprint;
    }

    private static string GlmFingerprint(string arch, int trunkLayers, bool native)
    {
        var model = (GlmDsaModel)RuntimeHelpers.GetUninitializedObject(typeof(GlmDsaModel));
        SetField(typeof(ModelBase), model, "<Config>k__BackingField", new ModelConfig
        {
            Architecture = arch,
            NumLayers = trunkLayers + 1,
            NumHeads = 8,
        });
        SetField(typeof(GlmDsaModel), model, "_numTrunkLayers", trunkLayers);
        SetField(typeof(GlmDsaModel), model, "_numNextnLayers", 1);
        SetField(typeof(GlmDsaModel), model, "_mtpLayer", -1);
        // Only the fingerprint reads this, and the instance is never disposed, so a
        // non-zero sentinel never reaches the native side.
        SetField(typeof(GlmDsaModel), model, "_native", native ? new IntPtr(1) : IntPtr.Zero);
        try
        {
            return model.KVStateFingerprint;
        }
        finally
        {
            SetField(typeof(GlmDsaModel), model, "_native", IntPtr.Zero);
        }
    }

    private static void SetField(Type owner, object target, string name, object value)
    {
        var field = owner.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            ?? throw new MissingFieldException(owner.Name, name);
        field.SetValue(target, value);
    }
}
