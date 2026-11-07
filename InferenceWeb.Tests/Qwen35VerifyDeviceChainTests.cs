// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using TensorSharp;
using TensorSharp.Models;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;
using System.Reflection;
using System.Runtime.InteropServices;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen35VerifyDeviceChainTests
{
    private readonly ITestOutputHelper _output;
    public Qwen35VerifyDeviceChainTests(ITestOutputHelper output) => _output = output;

    private const int Hidden = 32, Head = 32, Conv = 3 * Head, Rows = 4, Vocab = 17;

    [Fact]
    public void AllRecurrentPackedVerify_InitialAndReplayMatchFreshGraphs()
    {
        // No checkpoint is needed. Two tiny recurrent layers keep the native
        // persistent path enabled while deliberately omitting every attention
        // input from the graph. Before the upload guard, Metal packing aborted
        // on the first upload to the unallocated position tensor.
        using var env = new NativeEnvScope();
        GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
        bool metal = backend == GgmlBackendType.Metal;
        env.Set("TS_Q35_VERIFY_PACK", "0");
        env.Set("TS_Q35_GPU_STATE_COMMIT", "0");
        var reference = RunAllRecurrent(backend, rebuild: true, new[] { 0, 2, 1 });
        env.Set("TS_Q35_VERIFY_PACK", "1");
        env.Set("TS_Q35_GPU_STATE_COMMIT", metal ? "2" : "0"); // Metal must execute its snapshot-copy graph.
        var actual = RunAllRecurrent(backend, rebuild: false, new[] { 0, 2, 1 });
        Assert.Equal(reference.Count, actual.Count);
        for (int step = 0; step < reference.Count; step++)
        {
            double error = 0, norm = 0, maxError = 0;
            for (int i = 0; i < reference[step].Length; i++)
            {
                Assert.True(float.IsFinite(actual[step][i]));
                double delta = actual[step][i] - reference[step][i];
                error += delta * delta;
                norm += (double)reference[step][i] * reference[step][i];
                maxError = Math.Max(maxError, Math.Abs(delta));
            }
            double relative = Math.Sqrt(error / Math.Max(1e-30, norm));
            _output.WriteLine($"All-recurrent step {step}: relative L2={relative:E3}, max absolute={maxError:E3}");
            Assert.True(relative < 1e-4 && maxError < 1e-4);
        }
    }

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaReplayCommit_LeavesTheStateASnapshotCommitLeaves()
    {
        // ggml_cuda keeps no recurrent state per verify row: the verify keeps what
        // each recurrence consumed and the commit replays the kept prefix with the
        // rejected rows' gate and beta zeroed. Every acceptance of a window - none
        // kept (slot 3), part (2, 1), all (0) - must commit exactly the state a
        // per-row snapshot holds, and the next window must see it.
        int[] slots = { 3, 0, 2, 1, 3, 1 };
        using var env = new NativeEnvScope();
        env.Set("TS_Q35_VERIFY_REPLAY", "0");
        var snapshots = RunAllRecurrent(GgmlBackendType.Cuda, rebuild: false, slots);
        env.Set("TS_Q35_VERIFY_REPLAY", "1");
        var replay = RunAllRecurrent(GgmlBackendType.Cuda, rebuild: false, slots, expectNoSnapshots: true);
        Assert.Equal(snapshots.Count, replay.Count);
        for (int step = 0; step < snapshots.Count; step++)
            Assert.Equal(snapshots[step], replay[step]);
    }

    // A two-layer all-recurrent trunk verified in 4-row windows, one window per
    // entry of slots, each committed at that slot (0 keeps the whole window).
    // Returns, per step, the logits, the normed rows and the committed state.
    // expectNoSnapshots asserts that each verify kept no per-row state slots (the
    // host fetch of a slot declines, leaving the pending verify untouched).
    private static List<float[]> RunAllRecurrent(GgmlBackendType backend, bool rebuild, int[] slots,
        bool expectNoSnapshots = false)
    {
        GgmlBasicOps.EnsureBackendAvailable(backend);
        long owner = Qwen35Model.AllocateVerifyOwnerId();
        var buffers = new List<IntPtr>();
        var random = new Random(72819);
        IntPtr Make(int count, float scale = 0.025f, float? constant = null)
        {
            IntPtr pointer = GgmlBasicOps.AlignedAlloc(count * sizeof(float));
            Assert.NotEqual(IntPtr.Zero, pointer);
            buffers.Add(pointer);
            float[] values = Enumerable.Range(0, count)
                .Select(_ => constant ?? (float)(random.NextDouble() * 2 - 1) * scale).ToArray();
            Marshal.Copy(values, 0, pointer, count);
            return pointer;
        }
        float[] Read(IntPtr pointer, int count)
        {
            var values = new float[count];
            Marshal.Copy(pointer, values, 0, count);
            return values;
        }
        try
        {
            var layers = new Qwen35LayerDecodeArgs[2];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new Qwen35LayerDecodeArgs
                {
                    StructBytes = Marshal.SizeOf<Qwen35LayerDecodeArgs>(), IsRecurrent = 1, FfDense = Hidden,
                    AttnNormW = Make(Hidden, constant: 1), PostAttnNormW = Make(Hidden, constant: 1),
                    GdnQkvW = Make(Hidden * Conv), GdnQkvNe0 = Hidden, GdnQkvNe1 = Conv,
                    GdnQkvBytes = Hidden * Conv * sizeof(float),
                    GdnGateW = Make(Hidden * Head), GdnGateNe0 = Hidden, GdnGateNe1 = Head,
                    GdnGateBytes = Hidden * Head * sizeof(float),
                    SsmBetaW = Make(Hidden), SsmBetaNe0 = Hidden, SsmBetaNe1 = 1, SsmBetaBytes = Hidden * sizeof(float),
                    SsmAlphaW = Make(Hidden), SsmAlphaNe0 = Hidden, SsmAlphaNe1 = 1, SsmAlphaBytes = Hidden * sizeof(float),
                    Conv1dW = Make(4 * Conv), SsmDtW = Make(1), SsmAW = Make(1, constant: -0.15f),
                    SsmNormW = Make(Head, constant: 1),
                    SsmOutW = Make(Head * Hidden), SsmOutNe0 = Head, SsmOutNe1 = Hidden,
                    SsmOutBytes = Head * Hidden * sizeof(float),
                    ConvStateIn = Make(3 * Conv, scale: 0.002f), DeltaStateIn = Make(Head * Head, scale: 0.002f),
                    ConvStateOut = Make(3 * Conv), DeltaStateOut = Make(Head * Head),
                    GuW = Make(Hidden * 2 * Hidden), GuNe0 = Hidden, GuNe1 = 2 * Hidden,
                    GuBytes = Hidden * 2 * Hidden * sizeof(float),
                    DownW = Make(Hidden * Hidden), DownNe0 = Hidden, DownNe1 = Hidden,
                    DownBytes = Hidden * Hidden * sizeof(float),
                    // GGML_TYPE_F32 is zero, matching all default type fields.
                };
            IntPtr input = Make(Hidden * Rows), logits = Make(Vocab * Rows), normed = Make(Hidden * Rows);
            IntPtr lmHead = Make(Hidden * Vocab), finalNorm = Make(Hidden, constant: 1), used = Make(1);
            var result = new List<float[]>();
            int step = 0;
            foreach (int slot in slots)
            {
                if (rebuild) GgmlBasicOps.Qwen35ResetVerifyCache(owner);
                float[] values = Enumerable.Range(0, Hidden * Rows)
                    .Select(i => 0.1f * MathF.Sin(i * 0.13f + step)).ToArray();
                Marshal.Copy(values, 0, input, values.Length);
                Assert.True(GgmlBasicOps.Qwen35ModelVerify(layers, layers.Length,
                    input, Hidden, step * Rows, Rows, 1, 1, Head, 256, Head, 0, 0,
                    4, Head, Head, 1, 1, 1e-5f, 10000, 1, 0, 0, 0, 0, 0, 1,
                    logits, Vocab, lmHead, 0, Hidden, Vocab, Hidden * Vocab * sizeof(float),
                    finalNorm, normed, stateSnapshots: Rows, stateSnapshotsUsed: used,
                    deferStateDownload: true, ownerId: owner));
                Assert.Equal(Rows, Marshal.ReadInt32(used)); // Persistent deferred mode is required.
                if (expectNoSnapshots)
                    Assert.False(GgmlBasicOps.Qwen35FetchStateSnapshot(1,
                        layers.Select(_ => Make(3 * Conv)).ToArray(),
                        layers.Select(_ => Make(Head * Head)).ToArray(), layers.Length, owner));
                Assert.True(GgmlBasicOps.Qwen35CommitStateSnapshot(slot, layers.Length, owner));
                // Fetch committed states into the next call's input buffers.
                // This tests the retained context-only views after compute,
                // and makes the next replay depend on the selected prefix.
                Assert.True(GgmlBasicOps.Qwen35DrainDeviceState(
                    layers.Select(l => l.ConvStateIn).ToArray(),
                    layers.Select(l => l.DeltaStateIn).ToArray(), layers.Length, owner));
                var observed = Read(logits, Vocab * Rows).Concat(Read(normed, Hidden * Rows)).ToList();
                foreach (var layer in layers)
                {
                    observed.AddRange(Read(layer.ConvStateIn, 3 * Conv));
                    observed.AddRange(Read(layer.DeltaStateIn, Head * Head));
                }
                result.Add(observed.ToArray());
                step++;
            }
            return result;
        }
        finally
        {
            GgmlBasicOps.Qwen35ReleaseVerifyOwner(owner);
            foreach (IntPtr pointer in buffers)
            {
                GgmlBasicOps.InvalidateHostBuffer(pointer);
                GgmlBasicOps.AlignedFree(pointer);
            }
        }
    }

    // Set to a real Qwen3.5 checkpoint (for example Qwen3.5-9B-IQ4_XS.gguf),
    // together with TS_TEST_GGML_BACKEND=metal. Missing opt-in, or another pinned
    // backend, is a visible skip.
    [ModelFact("TS_Q35_STATE_MODEL", GgmlBackend = BackendType.GgmlMetal)]
    public void DeviceStateChain_MatchesHostDrainsAcrossCommitsRollbacksAndSingleRows()
    {
        string path = Environment.GetEnvironmentVariable("TS_Q35_STATE_MODEL");
        Assert.True(File.Exists(path), "TS_Q35_STATE_MODEL must name a Qwen3.5 GGUF file.");
        using ModelBase loaded = ModelBase.Create(path, BackendType.GgmlMetal);
        Qwen35Model model = Assert.IsType<Qwen35Model>(loaded);
        var target = (ISpeculativeTarget)model;
        int[] prompt = model.Tokenizer.Encode(
            "The lighthouse keeper recorded seventeen ships on Monday and twelve on Tuesday. " +
            "On Wednesday there were no ships because of the storm. Repeat the record:\n",
            addSpecial: true).ToArray();
        int[] input = prompt.TakeLast(13).ToArray();
        // Cross the small-matvec/GEMM boundary, reuse a thirteen-row graph, and
        // select full/partial prefixes with single-row commits between them.
        (int rows, int accepted)[] steps =
            { (4, 3), (1, 0), (13, 12), (1, 0), (13, 10), (13, 3), (2, 0), (1, 0), (4, 3) };
        using var env = new NativeEnvScope();
        env.Set("TS_Q35_GPU_STATE_COMMIT", "0");
        env.Set("TS_Q35_VERIFY_PACK", "0");
        var host = Run(forceDrain: true);
        // Strict mode rejects a native CPU fallback. The device-state assertion
        // below also rejects the managed snapshot-fetch fallback.
        env.Set("TS_Q35_GPU_STATE_COMMIT", "2");
        env.Set("TS_Q35_VERIFY_PACK", "1");
        var device = Run(forceDrain: false);
        Assert.Equal(host.Count, device.Count);
        for (int step = 0; step < host.Count; step++)
        {
            AssertClose(host[step].hidden, device[step].hidden, $"step {step} hidden");
            AssertClose(host[step].logits, device[step].logits, $"step {step} logits");
            int vocab = model.Config.VocabSize;
            for (int row = 0; row < steps[step].rows; row++)
                Assert.Equal(Argmax(host[step].logits.AsSpan(row * vocab, vocab)),
                    Argmax(device[step].logits.AsSpan(row * vocab, vocab)));
        }

        List<(float[] hidden, float[] logits)> Run(bool forceDrain)
        {
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            var results = new List<(float[], float[])>();
            foreach (var (rows, accepted) in steps)
            {
                if (forceDrain) model.DrainDeviceRecurrentState();
                int position = target.CacheSeqLen;
                target.SpecEnsureCapacity(position + rows);
                target.SpecSnapshotRecurrentState();
                var hidden = new float[rows * model.Config.HiddenSize];
                var logits = new float[rows * model.Config.VocabSize];
                target.SpecForward(input.Take(rows).ToArray(), hidden, logits, allLogitsRows: true);
                Assert.Equal(position + rows, target.CacheSeqLen);
                if (rows > 1)
                {
                    target.SpecOnVerifyAccepted(accepted, rows - 1);
                    bool retained = rows <= 8 || rows - 1 - accepted < 3;
                    Assert.Equal(retained, target.SpecVerifyPersistsAcceptedKv);
                    if (retained)
                        target.SpecRewindCache(position + accepted + 1);
                    else
                    {
                        // An early rejection in a wide window deliberately has
                        // no retained snapshot. Exercise the executor's actual
                        // restore-and-reforward path and compare the NEXT step too.
                        target.SpecRestoreRecurrentState();
                        target.SpecRewindCache(position);
                        target.SpecForward(input.Take(accepted + 1).ToArray(), null,
                            new float[model.Config.VocabSize], allLogitsRows: false);
                    }
                    Assert.Equal(position + accepted + 1, target.CacheSeqLen);
                }
                // A missing wide-window snapshot intentionally restores and
                // re-forwards through the normal host-state prefill path.
                if (!forceDrain && (rows <= 8 || rows - 1 - accepted < 3))
                {
                    var current = typeof(Qwen35Model).GetField("_fvDeviceStateCurrent",
                        BindingFlags.Instance | BindingFlags.NonPublic);
                    Assert.NotNull(current);
                    Assert.True((bool)current.GetValue(model),
                        $"Rows={rows}, accepted={accepted}: the required GPU commit fell back to downloading a recurrent-state snapshot.");
                }
                results.Add((hidden, logits));
            }
            // Also exercise the explicit host handoff after the final device
            // commit; the next run's ResetKVCache must not inherit this authority.
            model.DrainDeviceRecurrentState();
            return results;
        }
    }

    private void AssertClose(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        double error = 0, norm = 0, maxError = 0, scale = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (!float.IsFinite(reference[i]) || !float.IsFinite(actual[i]))
                Assert.Fail($"{label}: non-finite value at {i}");
            double delta = (double)actual[i] - reference[i];
            error += delta * delta;
            norm += (double)reference[i] * reference[i];
            maxError = Math.Max(maxError, Math.Abs(delta));
            scale = Math.Max(scale, Math.Abs(reference[i]));
        }
        double relativeL2 = Math.Sqrt(error / Math.Max(1e-30, norm));
        _output.WriteLine($"{label}: relative L2={relativeL2:E3}, max absolute error={maxError:E3}");
        Assert.True(relativeL2 <= 1e-5, $"{label}: relative L2={relativeL2:E3}");
        Assert.True(maxError <= 1e-4 * Math.Max(1, scale), $"{label}: max absolute error={maxError:E3}");
        Assert.True(MemoryMarshal.AsBytes(reference.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())),
            $"{label}: copying recurrent state on the GPU changed a hidden/logit bit");
    }

    private static int Argmax(ReadOnlySpan<float> values)
    {
        int index = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[index]) index = i;
        return index;
    }
}
