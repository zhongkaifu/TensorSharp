// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Adaptive residency's swaps overlapped with the decode (Strata's asynchronous adaptive tier, STRATA_ADAPT_OVERLAP).
// A blocking swap copies its two experts with the GPU idle: 8 swaps a round were affordable, Strata's 96 were not
// (TS_MOE_TIER_ADAPT_SWAPS=96 halved the CPU's experts after long prompts and gave the time back in copies). Here a
// round's copies run on a stream of their own, in three steps a forward boundary apart, so the tables a window reads
// never name bytes in flight:
//   1. copy each hot host expert into a staging slot and each cold VRAM expert into another (the window that runs
//      meanwhile still reads the hot expert on the host and the cold one in its slot - both only read);
//   2. point both experts' tables at their staging copies (the hot one now computed on the GPU), then copy the hot
//      expert into the cold one's slot and the cold one into the hot one's packed host place (neither is named by a
//      table any more);
//   3. point the hot expert at its slot and the cold one back at the host.
// The staging slots are the prompt staging's, idle in decode; a prompt or a wide step first finishes the round.
// TS_MOE_TIER_ADAPT_ASYNC=0 keeps the blocking swaps.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _asyncSwapsEnv = Environment.GetEnvironmentVariable("TS_MOE_TIER_ADAPT_ASYNC") != "0";
        private IntPtr _swapStream, _swapEvent;
        private int _swapStep;

        private sealed class PendingSwap
        {
            public int Layer, Hot, Cold, Index;
            /// <summary>The staging bases of the hot expert's copy and the cold expert's.</summary>
            public IntPtr HotCopy, ColdCopy;
            /// <summary>The cold expert's slot, per matrix (gate, up, down).</summary>
            public ulong[] Slot = new ulong[3];
        }

        private readonly List<PendingSwap> _swaps = new List<PendingSwap>();

        /// <summary>Whether rounds run overlapped: residency with adaptation and host experts, and the staging to borrow.</summary>
        private bool AsyncSwapsUsable => _asyncSwapsEnv && _residency?.Tab != null && _expertStage != null && _hostServer != null;

        /// <summary>Swaps one overlapped round can carry: two staging slots each, short of the last ones the GPU's share of
        /// a decode step's missed experts goes through (TS_Q4E_HYBRID_MISSES).</summary>
        private int AsyncSwapCapacity => _expertStage == null ? 0 : (_expertStage.Capacity - (_hybridStage?.Capacity ?? 0)) / 2;

        private static long[] ExpertMatrixBytes(LayerDesc l, int nExpert)
            => new[] { l.GateExps.TotalBytes / nExpert, l.UpExps.TotalBytes / nExpert, l.DownExps.TotalBytes / nExpert };

        private static long MatrixOffset(in CudaExpertStageLayer s, int k) => k == 0 ? 0 : k == 1 ? s.OffUp : s.OffDown;

        /// <summary>
        /// Up to <paramref name="budget"/> trades, largest gain first across the layers: per layer the most-selected host
        /// experts paired in order with the least-selected VRAM experts, while the first was selected more than twice as often
        /// (and at least twice) - <see cref="BestTrade"/> repeated without applying the trades in between.
        /// </summary>
        private List<(int Layer, int Hot, int Cold)> SelectTrades(int budget)
        {
            var r = _residency;
            var all = new List<(float Gain, int Layer, int Hot, int Cold)>();
            for (int il = 0; il < _m.NLayer; il++)
            {
                int[] index = r.HostIndex[il];
                if (index == null)
                    continue;
                float[] c = _adaptCounts[il];
                var hosts = new List<int>();
                var slots = new List<int>();
                for (int e = 0; e < _m.NExpert; e++)
                    (index[e] >= 0 ? hosts : slots).Add(e);
                hosts.Sort((a, b) => c[b].CompareTo(c[a]));
                slots.Sort((a, b) => c[a].CompareTo(c[b]));
                for (int j = 0; j < Math.Min(hosts.Count, slots.Count); j++)
                {
                    int hot = hosts[j], cold = slots[j];
                    if (c[hot] < 2f || c[hot] <= 2f * c[cold] + 1f)
                        break;
                    all.Add((c[hot] - c[cold], il, hot, cold));
                }
            }
            all.Sort((a, b) => b.Gain.CompareTo(a.Gain));
            var picked = new List<(int, int, int)>();
            for (int i = 0; i < all.Count && picked.Count < budget; i++)
                picked.Add((all[i].Layer, all[i].Hot, all[i].Cold));
            return picked;
        }

        /// <summary>Step 1 of an overlapped round: both experts of each trade copied into staging on the swap stream.</summary>
        private void BeginAsyncSwaps(List<(int Layer, int Hot, int Cold)> trades)
        {
            var r = _residency;
            var dev = _devs[0];
            dev.MakeCurrent();
            if (_swapStream == IntPtr.Zero)
            {
                CudaDriverApi.cuStreamCreate(out _swapStream, 0x1 /*NON_BLOCKING*/).ThrowOnError();
                CudaDriverApi.cuEventCreate(out _swapEvent, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
            }
            _swaps.Clear();
            int n = Math.Min(trades.Count, AsyncSwapCapacity);
            for (int j = 0; j < n; j++)
            {
                var (il, hot, cold) = trades[j];
                var stage = _layers[il].Stage;
                var s = new PendingSwap
                {
                    Layer = il, Hot = hot, Cold = cold, Index = r.HostIndex[il][hot],
                    HotCopy = _expertStage.Staging + (nint)(2L * j * _expertStage.Stride),
                    ColdCopy = _expertStage.Staging + (nint)((2L * j + 1) * _expertStage.Stride),
                };
                long[] bytes = ExpertMatrixBytes(_m.Layers[il], _m.NExpert);
                for (int k = 0; k < 3; k++)
                {
                    long eb = bytes[k], off = MatrixOffset(stage, k);
                    s.Slot[k] = r.Tab[il][k][cold];
                    CudaDriverApi.cuMemcpyHtoDAsync(s.HotCopy + (nint)off, r.Host[il][k] + (nint)(s.Index * eb), new UIntPtr((ulong)eb), _swapStream).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoDAsync(s.ColdCopy + (nint)off, (IntPtr)(long)s.Slot[k], new UIntPtr((ulong)eb), _swapStream).ThrowOnError();
                }
                _swaps.Add(s);
            }
            if (n == 0)
                return;
            CudaDriverApi.cuEventRecord(_swapEvent, _swapStream).ThrowOnError();
            _swapStep = 1;
        }

        /// <summary>
        /// At a forward boundary: the round's next step once its copies have landed (or, with <paramref name="force"/>,
        /// after waiting for them), until the round is done.
        /// </summary>
        private void AdvanceAsyncSwaps(bool force)
        {
            while (_swapStep != 0)
            {
                _devs[0].MakeCurrent();
                if (!force && CudaDriverApi.cuEventQuery(_swapEvent) != 0)
                    return;
                CudaDriverApi.cuEventSynchronize(_swapEvent).ThrowOnError();
                if (_swapStep == 1)
                    SwapStepToStaging();
                else
                    SwapStepToFinal();
                if (!force)
                    return;
            }
        }

        /// <summary>The round done before staging is used for anything else (a prompt, a wide step).</summary>
        private void FlushAsyncSwaps() => AdvanceAsyncSwaps(force: true);

        /// <summary>Step 2: tables on the staging copies, then the copies into their final places.</summary>
        private void SwapStepToStaging()
        {
            var r = _residency;
            var changed = new HashSet<int>();
            foreach (var s in _swaps)
            {
                var stage = _layers[s.Layer].Stage;
                for (int k = 0; k < 3; k++)
                {
                    long off = MatrixOffset(stage, k);
                    r.Tab[s.Layer][k][s.Hot] = (ulong)(long)(s.HotCopy + (nint)off);
                    r.Tab[s.Layer][k][s.Cold] = (ulong)(long)(s.ColdCopy + (nint)off);
                }
                // The hot expert is the GPU's from here; its packed host place is free.
                r.IsHostFlags[s.Layer][s.Hot] = 0;
                r.HostIndex[s.Layer][s.Hot] = -1;
                changed.Add(s.Layer);
            }
            foreach (int il in changed)
                UploadLayerTables(il);
            foreach (var s in _swaps)
            {
                long[] bytes = ExpertMatrixBytes(_m.Layers[s.Layer], _m.NExpert);
                var stage = _layers[s.Layer].Stage;
                for (int k = 0; k < 3; k++)
                {
                    long eb = bytes[k], off = MatrixOffset(stage, k);
                    CudaDriverApi.cuMemcpyDtoDAsync((IntPtr)(long)s.Slot[k], s.HotCopy + (nint)off, new UIntPtr((ulong)eb), _swapStream).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoHAsync(r.Host[s.Layer][k] + (nint)(s.Index * eb), s.ColdCopy + (nint)off, new UIntPtr((ulong)eb), _swapStream).ThrowOnError();
                }
            }
            CudaDriverApi.cuEventRecord(_swapEvent, _swapStream).ThrowOnError();
            _swapStep = 2;
        }

        /// <summary>Step 3: the hot experts on their slots, the cold ones back on the host.</summary>
        private void SwapStepToFinal()
        {
            var r = _residency;
            var changed = new HashSet<int>();
            foreach (var s in _swaps)
            {
                for (int k = 0; k < 3; k++)
                {
                    r.Tab[s.Layer][k][s.Hot] = s.Slot[k];
                    r.Tab[s.Layer][k][s.Cold] = (ulong)(long)_zeroExpert;
                    r.Src[s.Layer][k][s.Cold] = r.Src[s.Layer][k][s.Hot];
                    r.Src[s.Layer][k][s.Hot] = 0;
                }
                r.IsHostFlags[s.Layer][s.Cold] = 1;
                r.HostIndex[s.Layer][s.Cold] = s.Index;
                changed.Add(s.Layer);
                _adaptSwaps++;
                CudaMoeOffloadCounters.AddSwap();
            }
            foreach (int il in changed)
                UploadLayerTables(il);
            _swaps.Clear();
            _swapStep = 0;
        }

        private void DestroyAsyncSwaps()
        {
            if (_swapStream == IntPtr.Zero)
                return;
            try
            {
                _devs[0].MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(_swapStream);
                CudaDriverApi.cuEventDestroy(_swapEvent);
                CudaDriverApi.cuStreamDestroy(_swapStream);
            }
            catch
            {
            }
            _swapStream = _swapEvent = IntPtr.Zero;
        }
    }
}
