// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Muse-Glimmer's KV cache while the full layers' capacity is at or below the
// sliding-window ring - which on ggml_metal is every conversation shorter than
// 4352 tokens, because the full layers start at 2048 rows and grow on demand while
// the ring is allocated at its final 4352 rows up front.
//
// Two bugs lived here, and together they made the server's answers collapse into a
// one-token loop ("（（（（", "respond respond", "```\n\n```") the moment prompt plus
// output crossed 2048 tokens, and made radix prefix reuse hand the model a
// half-empty cache:
//
//  1. LAYOUT. The native kernel treated the sliding-window layers as a ring only
//     while the ring was SMALLER than the full layers' capacity. Below that it bound
//     the 4352-row ring buffers as [cache_size]-row flat caches: head 1 was written
//     inside head 0's rows, moved whenever the capacity grew, and was not where the
//     host-side ring code (KvBlockTransfer, truncation) looked for it. The size
//     mismatch also made every device->host sync of those layers a silent no-op, so
//     captured prefix blocks carried zero K/V for them.
//  2. GROW. EnsureCacheCapacity copied the full layers' rows from the host copy
//     without first pulling back the rows the fused kernel had written on the
//     device, so the grown layers restarted from the zero fill.
//
// These tests run a tiny synthetic Muse-Glimmer (MuseGlimmerSyntheticModelBuilder)
// on the pinned GGML backend. On a GPU backend (TS_TEST_GGML_BACKEND=metal, cuda or
// vulkan) its sliding-window layers get the ring exactly like the 30B's and the K/V
// lives in device copies, which is where both bugs lived; on ggml_cpu there is no
// ring and the K/V is the host memory itself, so the tests run there but cannot
// fail. A cache that grows mid-sequence must be bit-identical to one that was big
// enough from the start: the kernel reads the same windows either way, so any
// difference is lost or misplaced K/V.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[CollectionDefinition("Muse-Glimmer KV grow", DisableParallelization = true)]
public sealed class MuseGlimmerKvGrowCollection { }

[Collection("Muse-Glimmer KV grow")]
public sealed class MuseGlimmerKvGrowTests : IDisposable
{
    /// <summary>Matches SchedulerConfig.BlockSize, the size CaptureNewlyFullBlocks uses.</summary>
    private const int BlockSize = 256;

    private readonly ITestOutputHelper _output;
    private readonly string _dir;
    private readonly string _path;

    public MuseGlimmerKvGrowTests(ITestOutputHelper output)
    {
        _output = output;
        _dir = Path.Combine(Path.GetTempPath(), "ts-mg-kvgrow-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        _path = new MuseGlimmerSyntheticModelBuilder().Write(Path.Combine(_dir, "muse-glimmer-synthetic.gguf"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    /// <summary>
    /// Grow the full layers three times mid-sequence (256 -> 512 during a prefill
    /// that already has 200 rows to keep, 512 -> 1024 on a decode step, 1024 -> 4096
    /// on a prefill that also carries the sequence past the 2304-row ring), then
    /// decode on the wrapped ring. Every forward must return exactly the logits of a
    /// cache that never had to grow.
    /// </summary>
    [Fact]
    public void SyntheticMuseGlimmer_GrowingTheCacheMidSequence_MatchesAPresizedCache()
    {
        int[] tokens = Tokens(2600);
        var plan = new List<int[]>
        {
            tokens[..200],          // [0, 200)      fits the initial 256 rows
            tokens[200..400],       // [200, 400)    grows 256 -> 512, keeping 200 rows
        };
        for (int p = 400; p < 550; p++)
            plan.Add(new[] { tokens[p] });   // decode: grows 512 -> 1024 at position 512
        plan.Add(tokens[550..2450]);          // grows 1024 -> 4096; passes the 2304-row ring
        for (int p = 2450; p < 2510; p++)
            plan.Add(new[] { tokens[p] });   // decode on a wrapped ring

        List<float[]> presized = Run("4096", plan, out int presizedRing, out int presizedCap);
        List<float[]> grown = Run("256", plan, out int grownRing, out int grownCap);

        _output.WriteLine($"backend {TestGates.PinnedGgmlBackend}: ring rows {grownRing}, " +
            $"capacity 256 -> {grownCap} (presized {presizedCap})");
        Assert.Equal(presizedRing, grownRing);
        if (TestGates.PinnedGgmlBackend != BackendType.GgmlCpu)
            Assert.True(grownRing > 0, "A GPU backend must give the sliding-window layers a ring; this test would not cover it.");
        Assert.Equal(4096, grownCap);

        int position = 0;
        for (int step = 0; step < plan.Count; step++)
        {
            position += plan[step].Length;
            float maxAbs = MaxAbsDiff(presized[step], grown[step]);
            Assert.True(maxAbs == 0f,
                $"Forward {step} (ending at position {position}) differs from the presized cache by {maxAbs:E3}: " +
                "growing the full layers lost or moved K/V rows.");
        }
    }

    /// <summary>
    /// Capture prefix blocks and restore them into a fresh cache while the full
    /// layers are still below the ring - what a pooled radix hit does for every
    /// ordinary chat on ggml_metal. The next token's logits must be exactly the live
    /// ones: before the fix the sliding-window layers came back as zeros.
    /// </summary>
    [Fact]
    public void SyntheticMuseGlimmer_RestoringAPrefixBelowTheRingCapacity_ReproducesTheLiveLogits()
    {
        using var env = new EnvScope();
        env.Set("TS_KV_INITIAL_TOKENS", "1024");
        using var model = MuseGlimmerArchitecture.CreateLegacyModel(_path, TestGates.PinnedGgmlBackend);

        int[] tokens = Tokens(2 * BlockSize + 1);
        int prefixLen = 2 * BlockSize;
        int next = tokens[prefixLen];

        model.ResetKVCache();
        model.Forward(tokens[..prefixLen]);
        long blockBytes = model.ComputeKVBlockByteSize(BlockSize);
        var blocks = new byte[prefixLen / BlockSize][];
        for (int b = 0; b < blocks.Length; b++)
        {
            blocks[b] = new byte[blockBytes];
            Assert.True(model.TryExtractKVBlock(b * BlockSize, BlockSize, blocks[b]), $"capture of block {b} refused");
        }
        float[] live = (float[])model.Forward(new[] { next }).Clone();

        ScrambleCache(model, prefixLen + 1);
        for (int b = 0; b < blocks.Length; b++)
            Assert.True(model.TryInjectKVBlock(b * BlockSize, BlockSize, blocks[b]), $"restore of block {b} refused");
        float[] restored = (float[])model.Forward(new[] { next }).Clone();

        float maxAbs = MaxAbsDiff(live, restored);
        _output.WriteLine($"restored {prefixLen} tokens below the ring (ring rows {RingRows(model)}): max |delta| {maxAbs:E3}");
        Assert.True(maxAbs == 0f, $"Restored logits differ from the live ones by {maxAbs:E3}; the snapshot is lossy.");
    }

    /// <summary>
    /// Rewind the live cache (what the scheduler does when a new turn diverges a few
    /// tokens before the head) while the full layers are below the ring, and re-run
    /// the tail. It must agree with a cache that never went past the rewind point.
    /// </summary>
    [Fact]
    public void SyntheticMuseGlimmer_TruncatingBelowTheRingCapacity_KeepsTheSlidingWindowLayers()
    {
        using var env = new EnvScope();
        env.Set("TS_KV_INITIAL_TOKENS", "512");
        using var model = MuseGlimmerArchitecture.CreateLegacyModel(_path, TestGates.PinnedGgmlBackend);

        int[] tokens = Tokens(300);

        model.ResetKVCache();
        model.Forward(tokens[..290]);
        float[] direct = (float[])model.Forward(tokens[290..300]).Clone();

        model.ResetKVCache();
        model.Forward(tokens[..300]);
        Assert.True(model.TryTruncateKVCache(290), "rewind to 290 refused");
        float[] rewound = (float[])model.Forward(tokens[290..300]).Clone();

        float maxAbs = MaxAbsDiff(direct, rewound);
        _output.WriteLine($"rewind 300 -> 290 below the ring (ring rows {RingRows(model)}): max |delta| {maxAbs:E3}");
        Assert.Equal(ArgMax(direct), ArgMax(rewound));
        Assert.True(maxAbs <= 1e-3f,
            $"Logits after the rewind differ from a cache that never passed it by {maxAbs:E3}; " +
            "the rewind lost rows it had to keep.");
    }

    // ------------------------------------------------------------------

    private List<float[]> Run(string initialTokens, List<int[]> plan, out int ringRows, out int finalCapacity)
    {
        using var env = new EnvScope();
        env.Set("TS_KV_INITIAL_TOKENS", initialTokens);
        using var model = MuseGlimmerArchitecture.CreateLegacyModel(_path, TestGates.PinnedGgmlBackend);
        model.ResetKVCache();
        var outputs = new List<float[]>(plan.Count);
        foreach (int[] chunk in plan)
            outputs.Add((float[])model.Forward(chunk).Clone());
        ringRows = RingRows(model);
        finalCapacity = Field(model, "_kvCacheCapacity");
        return outputs;
    }

    /// <summary>
    /// Overwrite the first <paramref name="rows"/> positions - host rows AND their
    /// device copies - with unrelated K/V, then empty the cache again. A reset alone
    /// keeps the device copies on the GPU backends, so without this a restore that
    /// wrote or refreshed nothing would still read back the live prefill's rows.
    /// </summary>
    private static void ScrambleCache(ModelBase model, int rows)
    {
        model.ResetKVCache();
        model.Forward(Tokens(rows, seed: 0x2545F491u));
        model.ResetKVCache();
    }

    /// <summary>Deterministic token ids over the byte vocabulary.</summary>
    private static int[] Tokens(int count, uint seed = 0x9E3779B9u)
    {
        var ids = new int[count];
        uint state = seed;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ids[i] = (int)(state % MuseGlimmerSyntheticModelBuilder.ByteTokens);
        }
        return ids;
    }

    private static int RingRows(ModelBase model) => Field(model, "_kvSwaRows");

    private static int Field(ModelBase model, string name)
        => (int)typeof(MuseGlimmerModel)
            .GetField(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(model)!;

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = Math.Abs(a[i] - b[i]);
            if (float.IsNaN(d)) return float.NaN;
            if (d > max) max = d;
        }
        return max;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
