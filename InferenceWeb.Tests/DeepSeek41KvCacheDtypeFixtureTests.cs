// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41TinyFixtureFactAttribute : FactAttribute
{
    public DeepSeek41TinyFixtureFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_DSV41_TARGET")))
            Skip = "Requires TS_TEST_DSV41_TARGET: the tiny DeepSeek V4.1 fixture GGUF (with its Engram sidecar beside it).";
    }
}

[CollectionDefinition("DeepSeek V4.1 KV cache dtype", DisableParallelization = true)]
public sealed class DeepSeek41KvCacheDtypeCollection { }

/// <summary>
/// What <c>KV_CACHE_DTYPE</c> does to a real DeepSeek V4.1 load. Before the
/// refusal, a q8_0 request loaded the native graph with F16 caches and
/// <see cref="ModelBase.KvCacheDtype"/> reported q8_0: a silent downgrade
/// with a wrong banner. Now the request is refused before the native loader
/// runs, and the float requests report the F16 the executor really uses.
/// </summary>
[Collection("DeepSeek V4.1 KV cache dtype")]
[Trait("Requires", "Models")]
public sealed class DeepSeek41KvCacheDtypeFixtureTests : IDisposable
{
    private static readonly int[] Prompt = [0, 15, 32, 64, 128];
    private readonly ITestOutputHelper _output;
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly Dictionary<string, string?> _restoreEnv = new();

    public DeepSeek41KvCacheDtypeFixtureTests(ITestOutputHelper output)
    {
        _output = output;
        // The tiny-fixture environment the other V4.1 fixture tests run under;
        // the native loader reads these at load time.
        foreach (var (key, value) in new Dictionary<string, string>
        {
            ["MAX_CONTEXT"] = "1024", ["TS_DSV4_UBATCH"] = "32", ["TS_DSV4_THREADS"] = "2",
            ["TS_DSV41_TP"] = "0", ["TS_DSV41_ENGRAM_THREADS"] = "2", ["TS_DSV41_ENGRAM_WARM"] = "0",
            ["TS_DSV41_RETAINED_CACHE"] = "0", ["TS_DSV41_REWIND_CHECKPOINT"] = "1",
        })
        {
            _restoreEnv[key] = Environment.GetEnvironmentVariable(key);
            Environment.SetEnvironmentVariable(key, value);
        }
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        foreach (var (key, value) in _restoreEnv)
            Environment.SetEnvironmentVariable(key, value);
    }

    private static string Target => Environment.GetEnvironmentVariable("TS_TEST_DSV41_TARGET")!;

    [DeepSeek41TinyFixtureFact]
    public void BlockQuantizedRequestIsRefusedBeforeTheNativeLoad()
    {
        foreach (KvCacheDtype dtype in new[] { KvCacheDtype.Q8_0, KvCacheDtype.Q4_0 })
        {
            KvCacheDtypeConfig.Set(dtype);
            var error = Assert.Throws<NotSupportedException>(() =>
            {
                using var model = new DeepSeek4Model(Target, BackendType.GgmlCpu);
                _output.WriteLine($"UNEXPECTED: loaded with kvCacheDtype={model.KvCacheDtype.ToShortString()}");
            });
            _output.WriteLine($"{dtype.ToShortString()}: {error.Message}");
            Assert.StartsWith($"KV_CACHE_DTYPE={dtype.ToShortString()} is not supported by DeepSeek V4.1 Flash", error.Message);
        }
    }

    [DeepSeek41TinyFixtureFact]
    public void FloatRequestsLoadAndReportTheF16TheExecutorUses()
    {
        float[]? reference = null;
        foreach (KvCacheDtype dtype in new[] { KvCacheDtype.F16, KvCacheDtype.F32 })
        {
            KvCacheDtypeConfig.Set(dtype);
            var stderr = new StringWriter();
            TextWriter original = Console.Error;
            Console.SetError(stderr);
            DeepSeek4Model model;
            try { model = new DeepSeek4Model(Target, BackendType.GgmlCpu); }
            finally { Console.SetError(original); }
            using (model)
            {
                _output.WriteLine($"{dtype.ToShortString()} requested -> reported {model.KvCacheDtype.ToShortString()}");
                Assert.Equal(KvCacheDtype.F16, model.KvCacheDtype);
                string notice = stderr.ToString();
                if (dtype == KvCacheDtype.F32)
                    Assert.Contains("KV_CACHE_DTYPE=f32 requested, but DeepSeek V4.1 Flash keeps its attention caches in F16", notice);
                else
                    Assert.DoesNotContain("KV_CACHE_DTYPE=f32 requested", notice);

                float[] logits = (float[])model.ForwardRefill(Prompt).Clone();
                Assert.All(logits, x => Assert.True(float.IsFinite(x)));
                // Both requests run the same F16 graph, so the logits are identical.
                if (reference == null) reference = logits;
                else Assert.Equal(reference, logits);
            }
        }
    }
}
