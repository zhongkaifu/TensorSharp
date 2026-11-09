// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A speculative verify's recurrent-state commit at every acceptance of a window, against the plain answer: the state
/// a window of only the accepted prefix leaves. Windows of 4, 9, 12 and 16 rows on the pinned GGML backend. On
/// ggml_metal a window wider than eight rows replays the accepted prefix at commit (as ggml_cuda does) instead of
/// keeping a recurrent state per row - which on Qwen3.8-27B is 151 MB a row - so a block drafter's 16-row window
/// commits any acceptance without the restore-and-reforward the old three-slot cap fell back to.
/// </summary>
public sealed class Qwen35VerifyWideWindowTests
{
    private readonly ITestOutputHelper _output;
    public Qwen35VerifyWideWindowTests(ITestOutputHelper output) => _output = output;

    private const int Hidden = 32, Head = 32, Conv = 3 * Head, Vocab = 17, MaxRows = 16;

    [Theory]
    [InlineData(4)]
    [InlineData(9)]
    [InlineData(12)]
    [InlineData(16)]
    public void EveryAcceptance_CommitsTheStateOfTheAcceptedPrefixAlone(int rows)
    {
        GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
        if (backend is not (GgmlBackendType.Cpu or GgmlBackendType.Metal))
            return;   // the CUDA lane covers replay in Qwen35VerifyDeviceChainTests
        using var h = new Harness(backend);
        bool replay = backend == GgmlBackendType.Metal && rows > 8;
        double worst = 0;
        for (int accepted = 0; accepted < rows; accepted++)
        {
            float[] wide = h.Run(rows, rows - 1 - accepted, expectReplay: replay);
            float[] alone = h.Run(accepted + 1, 0, expectReplay: backend == GgmlBackendType.Metal && accepted + 1 > 8);
            double err = 0, norm = 0;
            for (int i = 0; i < wide.Length; i++)
            {
                Assert.True(float.IsFinite(wide[i]));
                err += (double)(wide[i] - alone[i]) * (wide[i] - alone[i]);
                norm += (double)alone[i] * alone[i];
            }
            double rel = Math.Sqrt(err / Math.Max(1e-30, norm));
            worst = Math.Max(worst, rel);
            Assert.True(rel < 1e-5, $"window {rows}, accepted {accepted}: relative L2 {rel:E3}");
        }
        _output.WriteLine($"{backend} window {rows} ({(replay ? "replay" : "snapshots")}): worst relative L2 {worst:E3} over {rows} acceptances");
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void WithReplayOff_AWideMetalWindowKeepsOnlyItsLastThreeStates()
    {
        // TS_Q35_VERIFY_REPLAY=0 restores the old wide-window protocol: three per-row states, and an earlier
        // rejection declines the commit (the caller restores and re-forwards).
        using var env = new NativeEnvScope();
        env.Set("TS_Q35_VERIFY_REPLAY", "0");
        using var h = new Harness(GgmlBackendType.Metal);
        Assert.NotNull(h.Run(12, 2, expectReplay: false));
        Assert.Null(h.Run(12, 5, expectReplay: false, commitMayFail: true));
    }

    /// <summary>A two-layer all-recurrent trunk (no attention), as in Qwen35VerifyDeviceChainTests: each run starts
    /// from the same recurrent state and verifies the first n rows of the same inputs.</summary>
    private sealed class Harness : IDisposable
    {
        private readonly List<IntPtr> _buffers = new();
        private readonly Random _random = new(9137);
        private readonly long _owner = Qwen35Model.AllocateVerifyOwnerId();
        private readonly Qwen35LayerDecodeArgs[] _layers = new Qwen35LayerDecodeArgs[2];
        private readonly float[][] _conv0, _delta0;
        private readonly IntPtr _input, _logits, _normed, _lmHead, _finalNorm, _used;

        public Harness(GgmlBackendType backend)
        {
            GgmlBasicOps.EnsureBackendAvailable(backend);
            _conv0 = new float[_layers.Length][];
            _delta0 = new float[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Qwen35LayerDecodeArgs
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
                };
                _conv0[i] = Read(_layers[i].ConvStateIn, 3 * Conv);
                _delta0[i] = Read(_layers[i].DeltaStateIn, Head * Head);
            }
            _input = Make(Hidden * MaxRows);
            float[] values = Enumerable.Range(0, Hidden * MaxRows).Select(i => 0.1f * MathF.Sin(i * 0.13f)).ToArray();
            Marshal.Copy(values, 0, _input, values.Length);
            _logits = Make(Vocab * MaxRows);
            _normed = Make(Hidden * MaxRows);
            _lmHead = Make(Hidden * Vocab);
            _finalNorm = Make(Hidden, constant: 1);
            _used = Make(1);
        }

        private IntPtr Make(int count, float scale = 0.025f, float? constant = null)
        {
            IntPtr pointer = GgmlBasicOps.AlignedAlloc(count * sizeof(float));
            Assert.NotEqual(IntPtr.Zero, pointer);
            _buffers.Add(pointer);
            float[] values = Enumerable.Range(0, count)
                .Select(_ => constant ?? (float)(_random.NextDouble() * 2 - 1) * scale).ToArray();
            Marshal.Copy(values, 0, pointer, count);
            return pointer;
        }

        private static float[] Read(IntPtr pointer, int count)
        {
            var values = new float[count];
            Marshal.Copy(pointer, values, 0, count);
            return values;
        }

        /// <summary>Verifies the first n rows from the initial state, commits `slot` (counted back from the window's
        /// end; 0 keeps every row) and returns the committed recurrent state, or null when the commit declined.</summary>
        public float[] Run(int n, int slot, bool expectReplay, bool commitMayFail = false)
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                Marshal.Copy(_conv0[i], 0, _layers[i].ConvStateIn, _conv0[i].Length);
                Marshal.Copy(_delta0[i], 0, _layers[i].DeltaStateIn, _delta0[i].Length);
                GgmlBasicOps.InvalidateHostBuffer(_layers[i].ConvStateIn);
                GgmlBasicOps.InvalidateHostBuffer(_layers[i].DeltaStateIn);
            }
            Assert.True(GgmlBasicOps.Qwen35ModelVerify(_layers, _layers.Length,
                _input, Hidden, 0, n, 1, 1, Head, 256, Head, 0, 0,
                4, Head, Head, 1, 1, 1e-5f, 10000, 1, 0, 0, 0, 0, 0, 1,
                _logits, Vocab, _lmHead, 0, Hidden, Vocab, Hidden * Vocab * sizeof(float),
                _finalNorm, _normed, stateSnapshots: n, stateSnapshotsUsed: _used,
                deferStateDownload: true, ownerId: _owner), GgmlBasicOps.LastNativeError());
            int used = Marshal.ReadInt32(_used);
            if (n > 1)
                Assert.True(used > 1, $"the verify of {n} rows kept no commit slots ({used})");
            if (expectReplay)
            {
                // A replay verify answers n like a full snapshot set but keeps no state slot: the host fetch of one
                // declines and leaves the pending verify for the commit. (A fetch that succeeds consumes it, so
                // snapshot windows are not probed.)
                Assert.False(GgmlBasicOps.Qwen35FetchStateSnapshot(1,
                    _layers.Select(_ => Make(3 * Conv)).ToArray(), _layers.Select(_ => Make(Head * Head)).ToArray(),
                    _layers.Length, _owner));
            }
            bool committed = GgmlBasicOps.Qwen35CommitStateSnapshot(n > 1 ? slot : -1, _layers.Length, _owner);
            if (!committed)
            {
                Assert.True(commitMayFail, $"the commit of slot {slot} of a {n}-row window declined");
                return null;
            }
            Assert.True(GgmlBasicOps.Qwen35DrainDeviceState(
                _layers.Select(l => l.ConvStateIn).ToArray(), _layers.Select(l => l.DeltaStateIn).ToArray(),
                _layers.Length, _owner));
            var state = new List<float>();
            foreach (var layer in _layers)
            {
                state.AddRange(Read(layer.ConvStateIn, 3 * Conv));
                state.AddRange(Read(layer.DeltaStateIn, Head * Head));
            }
            return state.ToArray();
        }

        public void Dispose()
        {
            GgmlBasicOps.Qwen35ReleaseVerifyOwner(_owner);
            foreach (IntPtr pointer in _buffers)
            {
                GgmlBasicOps.InvalidateHostBuffer(pointer);
                GgmlBasicOps.AlignedFree(pointer);
            }
        }
    }
}
