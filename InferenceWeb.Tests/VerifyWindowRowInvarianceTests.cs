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
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A verify window's rows on direct CUDA are the decode rows at their positions, bit for bit, for every width the shared
/// runtime verifies (2 to 16 rows, StandardDecoderModel.SmallChunkRows), every row kept or a prefix kept: each row's logits,
/// and the state the windows leave (the logits of the plain steps after them). Speculative decoding then answers as plain
/// greedy decoding whatever drafts it was given - however the accepted tokens fall into windows. The windows run across the
/// 256-key bucket, where the decode rows' attention takes other key splits. Three things had made a window's rows round
/// apart: a decode row's residual add and norm ran unfused (the stream [hidden] beside the block's [1, hidden] output), the
/// window's rows split their keys as its last row's bucket, and 9-16 row windows took the multi-row attention (and, on NVFP4
/// weights, the FP4 GEMMs). Qwen3.8-27B served with its DFlash2 drafter answered differently from run to run.
/// </summary>
public sealed class VerifyWindowRowInvarianceTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("window-rows").FullName;
    private readonly ITestOutputHelper _output;

    public VerifyWindowRowInvarianceTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const int PromptLength = 250, Tokens = 40, Follow = 2;

    /// <summary>Windows of a width keeping every row (Keep 0) or their first Keep rows.</summary>
    private static readonly (int Width, int Keep)[] Windows =
        { (2, 0), (3, 0), (4, 0), (5, 0), (7, 0), (8, 0), (9, 0), (12, 0), (16, 0), (4, 3), (8, 5), (16, 11) };

    /// <remarks>Over a q8_0 cache too (the CUDA executor's _q8 flash kernels and q8_0 row writes): a row's bytes depend on its
    /// values alone, and a window's rows read the dequantized tiles the decode rows read. With the attention prep region off
    /// ("ops"), every row is written by the set_rows ops, whose recorded appends re-read their write row on every replay:
    /// baked at the capture, a q8_0 window's replay at a later position had rewritten the capture's rows.</remarks>
    [CudaTheory]
    [InlineData("qwen35", "f16")]
    [InlineData("qwen35", "q8_0")]
    [InlineData("qwen35", "q8_0-ops")]
    public void WindowRowsAreTheDecodeRowsBitForBit(string family, string kv)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(kv.StartsWith("q8_0", StringComparison.Ordinal) ? KvCacheDtype.Q8_0 : KvCacheDtype.F16);
        // Rows of 1,024: the row norm's reductions (256 threads) and the fused add and norm's (512) sum other partials there.
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, Width = 1024 }
            .Write(Path.Combine(_dir, family + ".gguf"));
        using var model = new Qwen35RuntimeModel(path, BackendType.Cuda) { AttentionPrepRegionDisabledForTests = kv.EndsWith("-ops", StringComparison.Ordinal) };
        Assert.Equal(kv.Split('-')[0], model.KvCacheDtype.ToShortString());
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        var batched = Assert.IsAssignableFrom<IBatchedSpeculativeTarget>(model);
        int vocab = model.Config.VocabSize;
        int[] text = Enumerable.Range(0, PromptLength + Tokens + Follow).Select(i => 3 + (i * 37 + i / 7) % (vocab - 6)).ToArray();

        // Plain decoding: the prompt, then a token a step.
        (Dictionary<int, float[]> plain, List<float[]> plainAfter) = Run("plain", 1, 0);
        int failures = 0;
        foreach ((int width, int keep) in Windows)
        {
            (Dictionary<int, float[]> rows, List<float[]> after) = Run($"w{width}k{keep}", width, keep);
            int differ = rows.Count(r => !r.Value.AsSpan().SequenceEqual(plain[r.Key]));
            int afterDiffer = after.Where((a, i) => !a.AsSpan().SequenceEqual(plainAfter[i])).Count();
            int first = rows.Where(r => !r.Value.AsSpan().SequenceEqual(plain[r.Key])).Select(r => r.Key).DefaultIfEmpty(-1).Min();
            _output.WriteLine($"{family} ({kv}) {width}-row windows{(keep > 0 ? $" keeping {keep}" : "")}: {rows.Count - differ} of {rows.Count} rows " +
                $"the decode rows' bits{(first >= 0 ? $" (first apart at position {first})" : "")}; {Follow - afterDiffer} of {Follow} steps after them the same");
            if (differ > 0 || afterDiffer > 0) failures++;
        }
        Assert.Equal(0, failures);

        // Every row a window computed (a row past a kept prefix is computed again by the next window: both are compared) and
        // the plain steps after the last window.
        (Dictionary<int, float[]> Rows, List<float[]> After) Run(string id, int width, int keep)
        {
            unified.EnsureSequenceSlot(id, PromptLength + Tokens + 32);
            var last = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { text[..PromptLength] }, new[] { 0 }, last));
            var rows = new Dictionary<int, float[]>();
            int pos = PromptLength;
            while (pos < PromptLength + Tokens)
            {
                int n = Math.Min(width, PromptLength + Tokens - pos);
                if (width == 1)
                {
                    Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { text[pos] } }, new[] { pos }, last));
                    rows[pos++] = (float[])last[0].Clone();
                    continue;
                }
                var all = new float[n * vocab];
                Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { id }, new[] { pos }, n, text[pos..(pos + n)], all, null), $"{id}: verify at {pos}");
                for (int r = 0; r < n; r++)
                {
                    float[] row = all.AsSpan(r * vocab, vocab).ToArray();
                    if (rows.TryGetValue(pos + r, out float[] earlier))
                        Assert.True(earlier.AsSpan().SequenceEqual(row), $"{id}: position {pos + r} computed again rounds apart");
                    rows[pos + r] = row;
                }
                int kept = keep > 0 && n > keep ? keep : n;
                batched.CommitBatchedSpecVerify(new[] { id }, new[] { pos }, new[] { kept });
                pos += kept;
            }
            var after = new List<float[]>();
            for (int i = 0; i < Follow; i++, pos++)
            {
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { text[pos] } }, new[] { pos }, last));
                after.Add((float[])last[0].Clone());
            }
            model.OnSequenceReleased(id);
            return (rows, after);
        }
    }

    /// <summary>
    /// A prompt step that hands a DFlash drafter its last row (the drafter's rings follow the trunk in the step's graph, tapping
    /// the stream at its target layers) leaves the trunk as a plain prompt step does: the last row's logits and the plain steps
    /// after it, bit for bit. The taps read the stream before the next block's norm, which then ran the row-norm op's reduction
    /// instead of the fused add and norm's, so a drafter loaded beside Qwen3.8-27B changed the trunk's own answers (served at
    /// temperature 0, DFlash2 parted from plain decoding at a later turn).
    /// </summary>
    [CudaFact]
    public void APromptFollowedByTheDrafterLeavesTheTrunkAsAPlainPrompt()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        const int width = 1024;
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, Width = width }.Write(Path.Combine(_dir, "trunk.gguf"));
        string draft = Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(_dir, "dflash.gguf"), hidden: width);
        using var model = new Qwen35RuntimeModel(path, BackendType.Cuda, draftModelPath: draft);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        int feature = Assert.IsAssignableFrom<ISpeculativeTarget>(model).SpecFeatureSize, vocab = model.Config.VocabSize;
        foreach (int length in new[] { 5, 17, 41 })
        {
            int[] prompt = Enumerable.Range(0, length).Select(i => 3 + (i * 37 + i / 5) % (vocab - 6)).ToArray();
            List<float[]> plain = Run("plain", null), followed = Run("followed", new[] { new float[feature] });
            for (int i = 0; i < plain.Count; i++)
                Assert.True(plain[i].AsSpan().SequenceEqual(followed[i]),
                    $"{length}-token prompt: {(i == 0 ? "its last row" : $"plain step {i} after it")} rounds apart when the drafter follows the prompt");

            List<float[]> Run(string id, float[][] features)
            {
                unified.EnsureSequenceSlot(id, length + 16);
                var last = new float[1][];
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt }, new[] { 0 }, last, features));
                var rows = new List<float[]> { (float[])last[0].Clone() };
                for (int t = 0; t < 2; t++)
                {
                    Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { 7 + t } }, new[] { length + t }, last));
                    rows.Add((float[])last[0].Clone());
                }
                model.OnSequenceReleased(id);
                return rows;
            }
        }
    }
}
