// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The fused whole-model graphs and a block-quantized K/V cache.
///
/// <para>
/// A quantized cache is the only lever that shrinks the one thing which grows with
/// the conversation, and on Metal it used to cost an order of magnitude: the fused
/// graph refused any block-quant dtype on every backend except CUDA, so selecting
/// q8_0 silently dropped to the per-op path. MEASURED on ggml_metal with
/// Qwen3.6-35B-A3B: decode 3.7 tok/s that way against 75.6 on f16, and the engine
/// said so itself ("KV cache dtype Q8_0 unsupported by fused graph on GgmlMetal;
/// using per-op decode"). Nothing was actually missing -- ggml-metal instantiates
/// flash_attn_ext_q8_0/_q4_0 for exactly the same dk/dv set as f16, and the cache
/// write has kernel_cpy_f32_q8_0 -- so the gate now admits Metal too.
/// </para>
/// </summary>
public class QuantizedKvFusedGraphTests
{
    private readonly ITestOutputHelper _output;
    public QuantizedKvFusedGraphTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void Q8_0_MapsToExpectedGgmlTypeAndDType()
    {
        // GGML_TYPE_Q8_0 == 8 in ggml.h. The fused graphs pass this id straight
        // through as the native kernels' kv_cache_type argument, so a wrong value
        // here builds the cache with the wrong element type.
        Assert.Equal(8, KvCacheDtype.Q8_0.GgmlType());
        Assert.Equal(DType.Q8_0, KvCacheDtype.Q8_0.ToDType());
        Assert.Equal("q8_0", KvCacheDtype.Q8_0.ToShortString());
    }

    [Fact]
    public void BothBlockQuantTiersRouteTogether()
    {
        // Q8_0 and Q4_0 must agree on IsBlockQuantized: every guard that sends a
        // cache down the native flash path keys on it, and a tier that answered
        // differently would take half the routing and none of the rest.
        Assert.True(KvCacheDtype.Q8_0.IsBlockQuantized());
        Assert.True(KvCacheDtype.Q4_0.IsBlockQuantized());
        Assert.False(KvCacheDtype.F16.IsBlockQuantized());
        Assert.False(KvCacheDtype.F32.IsBlockQuantized());
    }

    [Theory]
    [InlineData("q8_0")]
    [InlineData("Q8_0")]
    [InlineData("q8")]
    public void TryParse_AcceptsQ8Aliases(string value)
    {
        Assert.True(KvCacheDtypeConfig.TryParse(value, out KvCacheDtype dtype));
        Assert.Equal(KvCacheDtype.Q8_0, dtype);
    }

    /// <summary>
    /// The memory claim, in arithmetic rather than prose: a block-quantized cache
    /// has to be materially smaller per element than F16, or none of this is worth
    /// the numerical drift. MEASURED end to end on Qwen3.6-35B-A3B at ctx 32768:
    /// 41.8 KiB/token at f16, 22.4 at q8_0, 11.8 at q4_0.
    /// </summary>
    [Fact]
    public void QuantizedTiersAreMateriallySmallerPerElement()
    {
        const long elements = 4096;
        long f16 = KvCacheDtype.F16.ByteLengthFor(elements);
        long q8 = KvCacheDtype.Q8_0.ByteLengthFor(elements);
        long q4 = KvCacheDtype.Q4_0.ByteLengthFor(elements);

        Assert.True(q8 < f16, $"q8_0 ({q8}) must be smaller than f16 ({f16})");
        Assert.True(q4 < q8, $"q4_0 ({q4}) must be smaller than q8_0 ({q8})");
        // ~1.0625 B/elem for Q8_0 and ~0.5625 for Q4_0 against 2 for F16.
        Assert.InRange(q8 / (double)f16, 0.50, 0.56);
        Assert.InRange(q4 / (double)f16, 0.26, 0.30);
    }

    /// <summary>
    /// End to end on real weights: a quantized cache must produce the SAME answer as
    /// f16, not merely run. Corruption from a mis-strided block write does not throw,
    /// it produces fluent nonsense, so this compares greedy continuations rather than
    /// checking for an exception.
    ///
    /// <para>
    /// TS_KVQ_MODEL=&lt;gguf&gt; selects the weights, TS_KVQ_BACKEND the backend
    /// (default: the process's pinned GGML backend, TS_TEST_GGML_BACKEND; <c>direct_cuda</c> is TensorSharp's own CUDA
    /// backend, whose graphs take q8_0 alone of the block tiers). Skipped when unset.
    /// </para>
    /// </summary>
    [ModelFact("TS_KVQ_MODEL")]
    public void QuantizedKvCacheAgreesWithF16OnRealWeights()
    {
        string modelPath = Environment.GetEnvironmentVariable("TS_KVQ_MODEL");
        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
        {
            _output.WriteLine("[kv-quant] TS_KVQ_MODEL unset; skipping");
            return;
        }

        BackendType backend = Environment.GetEnvironmentVariable("TS_KVQ_BACKEND")?.ToLowerInvariant() switch
        {
            "ggml_metal" or "metal" => BackendType.GgmlMetal,
            "ggml_cuda" or "cuda" => BackendType.GgmlCuda,
            "ggml_cpu" or "cpu" => BackendType.GgmlCpu,
            "direct_cuda" => BackendType.Cuda,
            _ => TestGates.PinnedGgmlBackend,
        };
        const int steps = 12;

        KvCacheDtype restore = KvCacheDtypeConfig.Current;
        bool restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            int[] reference;
            string arch;
            try
            {
                reference = GreedyContinuation(modelPath, backend, KvCacheDtype.F16, steps, out arch);
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("already initialized", StringComparison.Ordinal))
            {
                // GGML binds ONE backend per process, so whichever test touched it
                // first wins. That is a property of the host, not of this scenario:
                // skip rather than report a failure that says nothing about the
                // quantized cache. Run this alone (or in its own job) to exercise it.
                _output.WriteLine($"[kv-quant] {backend} unavailable in this process ({ex.Message}); skipping");
                return;
            }
            _output.WriteLine($"[kv-quant] arch={arch} backend={backend} f16={string.Join(",", reference)}");

            // Direct CUDA's executor takes q8_0 caches in its flash kernels and no q4_0 (that tier keeps a model off its graphs).
            KvCacheDtype[] tiers = backend == BackendType.Cuda ? new[] { KvCacheDtype.Q8_0 } : new[] { KvCacheDtype.Q8_0, KvCacheDtype.Q4_0 };
            foreach (KvCacheDtype tier in tiers)
            {
                int[] got = GreedyContinuation(modelPath, backend, tier, steps, out _);
                _output.WriteLine($"[kv-quant] {tier.ToShortString()}={string.Join(",", got)}");

                int agree = reference.Zip(got, (a, b) => a == b ? 1 : 0).Sum();
                // Not exact equality: a quantized cache is a lossy cache and the
                // greedy path can legitimately diverge once it does. Wholesale
                // disagreement is what a corrupt cache looks like.
                Assert.True(agree >= (int)Math.Ceiling(steps * 0.66),
                    $"{tier.ToShortString()} agreed with f16 on only {agree}/{steps} greedy steps, "
                    + "which is what a mis-written block cache produces");
            }
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(restore, restoreExplicit);
        }
    }

    private static int[] GreedyContinuation(
        string modelPath, BackendType backend, KvCacheDtype dtype, int steps, out string architecture)
    {
        KvCacheDtypeConfig.Set(dtype);
        using ModelBase model = ModelBase.Create(modelPath, backend);
        architecture = model.Config.Architecture;

        // The model may refuse the tier (Gemma 4 does, by design, until its managed
        // fallback can read a block cache); the run is then an f16 run and the
        // comparison is trivially satisfied rather than falsely failing.
        int vocab = model.Config.VocabSize;
        int[] prompt = model.Tokenizer.Encode("The capital of France is", addSpecial: true).ToArray();

        var produced = new int[steps];
        float[] logits = model.Forward(prompt);
        for (int i = 0; i < steps; i++)
        {
            int next = ArgMax(logits, vocab);
            produced[i] = next;
            logits = model.Forward(new[] { next });
        }
        return produced;
    }

    private static int ArgMax(float[] logits, int vocab)
    {
        int best = 0;
        float bestValue = float.NegativeInfinity;
        for (int i = 0; i < vocab && i < logits.Length; i++)
        {
            if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
        }
        return best;
    }
}
