// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>
/// The F32 reference tools' procedure (eng/validation/MuseGlimmerF32Reference, eng/validation/Gemma4F32Reference) on the
/// managed executor with exact products (<see cref="StandardDecoderModel.FloatProductsForTests"/>): a prompt, then scored
/// prompt chunks of one size one after another. Every chunk after the first of its size runs as a stream of the first one's
/// recorded program (TensorGraphExecutor.Stream), and must keep the exact products the recorded program ran; a stream had
/// run the managed kernels' own block-format products from the second chunk on, so a reference made of several sequences of
/// one length (or one scored in several chunks) was partly not exact.
/// </summary>
public sealed class StreamedExactProductsTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-stream-exact-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < b.Length; i++)
        {
            double e = (double)a[i] - b[i];
            num += e * e;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>Scored chunks of 32 rows against one-row steps over the same tokens: a Qwen 3.5 synthetic file with Q8_0
    /// projections (the managed kernels' own Q8_0 product quantizes the activation rows; an exact product dequantizes the
    /// weight), every row within 1e-4 relative L2. Without the stream keeping the recorded program's setting, the second
    /// chunk on was the quantized product's.</summary>
    [Fact]
    public void ScoredPromptChunksOfOneSize_AllKeepExactProducts()
    {
        Directory.CreateDirectory(_dir);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 4, UntiedOutput = true, WeightType = GgmlType.Q8_0 }
            .Write(Path.Combine(_dir, "q35-q8.gguf"));
        const int prefix = 31, chunk = 32;
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        using var model = new Qwen35RuntimeModel(path, BackendType.Cpu);
        model.FloatProductsForTests = true;
        int vocab = model.Config.VocabSize;
        int[] all = Enumerable.Range(0, 160).Select(i => 3 + (i * 37) % (vocab - 5)).ToArray();

        var one = new List<float[]> { (float[])model.ForwardRefill(all[..prefix]).Clone() };
        for (int i = prefix; i < all.Length - 1; i++) one.Add((float[])model.Forward(new[] { all[i] }).Clone());

        model.ResetKVCache();
        model.Forward(all[..(prefix - 1)]);
        int[] scored = all[(prefix - 1)..^1];
        var rows = new List<float[]>();
        for (int start = 0; start < scored.Length; start += chunk)
        {
            int[] c = scored[start..Math.Min(scored.Length, start + chunk)];
            float[] logits = model.ForwardPromptRowsForTests(c);
            for (int r = 0; r < c.Length; r++) rows.Add(logits.AsSpan(r * vocab, vocab).ToArray());
        }
        double[] d = Enumerable.Range(0, one.Count).Select(i => RelL2(rows[i], one[i])).ToArray();
        output.WriteLine($"{one.Count} rows in chunks of {chunk}: " + string.Join(" ",
            Enumerable.Range(0, (d.Length + chunk - 1) / chunk).Select(b => d.Skip(b * chunk).Take(chunk).Max().ToString("G2"))));
        Assert.True(d.Max() <= 1e-4, $"scored chunk row {Array.IndexOf(d, d.Max())} is {d.Max():G3} from the one-row steps");
    }
}
