// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The glm-dsa token-batched decode against per-sequence decode on a native layer
/// split. forward_batched_decode used to write the RoPE positions only on the device
/// that embeds the tokens, so every layer on a later device rotated q/k by stale
/// positions and every concurrent GLM-5.2 stream degenerated on a 6-GPU split.
/// Rows past the visible GPU count degenerate to fewer devices (nGpu is a cap), so
/// the multi-device rows need at least that many GPUs to cover the defect.
/// </summary>
public sealed class GlmDsaNativeBatchedDecodeLayerSplitTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-glm-bd-split-" + Guid.NewGuid().ToString("N"));

    public GlmDsaNativeBatchedDecodeLayerSplitTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static int ArgMax(float[] v, int offset, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++) if (v[offset + i] > v[offset + best]) best = i;
        return best;
    }

    [GlmNativeCudaTheory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void BatchedDecode_MatchesPerSequenceDecode(int nGpu)
    {
        string path = GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "tiny-glm-dsa.gguf"));
        IntPtr handle = GgmlGlmNative.LoadModel(path, nGpu, 256, 16, 2, backendName: "CUDA", ctxIsHardLimit: true);
        Assert.NotEqual(IntPtr.Zero, handle);
        try
        {
            int vocab = GgmlGlmNative.VocabSize(handle);
            // Different lengths, so a position taken from the wrong row or never written
            // cannot line up by accident; both are past indexer top_k (sparse path).
            int[][] prompts =
            {
                Enumerable.Range(0, 24).Select(i => 65 + (i * 7) % 50).ToArray(),
                Enumerable.Range(0, 37).Select(i => 3 + (i * 11) % 90).ToArray(),
            };
            // Slots 0/1 decode batched, slots 2/3 decode the same tokens one at a time.
            int[] batched = { 0, GgmlGlmNative.SlotAlloc(handle) };
            int[] serial = { GgmlGlmNative.SlotAlloc(handle), GgmlGlmNative.SlotAlloc(handle) };
            Assert.All(batched.Concat(serial), s => Assert.True(s >= 0));

            var next = new int[2];
            var pos = new int[2];
            var logits = new float[vocab];
            for (int s = 0; s < 2; s++)
            {
                foreach (int slot in new[] { batched[s], serial[s] })
                {
                    Assert.True(GgmlGlmNative.SetActiveSlot(handle, slot));
                    Assert.True(GgmlGlmNative.Forward(handle, prompts[s], logits));
                }
                next[s] = ArgMax(logits, 0, vocab);
                pos[s] = prompts[s].Length;
            }

            const int steps = 6;
            var batchLogits = new float[2 * vocab];
            double worst = 0;
            for (int step = 0; step < steps; step++)
            {
                Assert.True(GgmlGlmNative.ForwardBatchedDecode(handle, batched, next, pos, batchLogits),
                    $"the native side declined the batched decode at step {step} (nGpu={nGpu})");
                for (int s = 0; s < 2; s++)
                {
                    Assert.True(GgmlGlmNative.SetActiveSlot(handle, serial[s]));
                    Assert.True(GgmlGlmNative.Forward(handle, new[] { next[s] }, logits));
                    double maxAbs = 0;
                    for (int v = 0; v < vocab; v++)
                        maxAbs = Math.Max(maxAbs, Math.Abs(batchLogits[s * vocab + v] - logits[v]));
                    worst = Math.Max(worst, maxAbs);
                    // Measured on 1-3 A40s: 0 with the fix; stale positions gave 0.028 at step 0.
                    Assert.True(maxAbs < 1e-3,
                        $"nGpu={nGpu} step {step} sequence {s}: batched logits differ from per-sequence by {maxAbs}");
                    Assert.Equal(ArgMax(logits, 0, vocab), ArgMax(batchLogits, s * vocab, vocab));
                    // Teacher-force the per-sequence choice so both paths keep seeing the same tokens.
                    next[s] = ArgMax(logits, 0, vocab);
                    pos[s]++;
                }
            }
            _output.WriteLine($"nGpu={nGpu}: worst max|batched - serial| over {steps} steps = {worst:G3}");
        }
        finally
        {
            GgmlGlmNative.Free(handle);
        }
    }
}
