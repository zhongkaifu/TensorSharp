// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// A long prompt chunk's MoE layers with their host experts staged ahead. A chunk of a few hundred rows or more selects
    /// most of a layer's experts, so its host experts (the ones the device tier does not hold) were copied in after the
    /// router's routes came back to the host, on the step's stream: the device waited for the routes, then for the copies
    /// (about 195 MB a layer of Qwen3.8 Flash Next UD-Q2_K_XL, 8 ms over an A40's PCIe 4 link). Here the staging arena is
    /// two halves; while one layer's experts run from one half, the copy engines bring every host expert of the next MoE
    /// layer into the other half (from the model file's page-locked mapping), and that layer's kernels wait for those
    /// copies. The routes stay on the device: the grouped MoE kernels read the router's ids through per-layer tables that
    /// point at the tier's copies and the half's slots, and a counting kernel keeps the routes for the routing profile and
    /// the adaptive tiers (handed to the expert host at the next run). The removed direct Qwen3.8 engines staged their prompt
    /// groups the same way; the bytes the kernels read are those the routed staging copies, so the answers are the same.
    /// </summary>
    public sealed partial class CudaGraphExecutor
    {
        /// <summary>Rows from which a prompt chunk's MoE layers take the staged-ahead path: below, copying only the selected
        /// host experts (after the routes return) moves fewer bytes than every host expert of the layer.</summary>
        internal const int DefaultPrefetchMinRows = 1024;

        /// <summary>The threshold this executor uses (a test lowers it on its own instance to drive the path with a small
        /// fixture).</summary>
        internal int PrefetchMinRows { get; set; } = DefaultPrefetchMinRows;

        /// <summary>Layers that ran the staged-ahead path, and of those, layers whose experts were copied while the previous
        /// layer ran (tests check that the path engaged).</summary>
        internal long PrefetchedLayers { get; private set; }
        internal long PrefetchedAhead { get; private set; }

        private IntPtr _prefetchArena, _copyStream;
        private readonly IntPtr[] _halfReady = new IntPtr[2], _halfFree = new IntPtr[2];
        private readonly int[] _halfLayer = { -1, -1 };
        private readonly long[] _halfRun = new long[2];
        private long _halfBytes, _arenaGeneration, _runOrdinal;
        private int _lastHalf = 1;

        /// <summary>MoE layers seen (any step), by index: the next one after a layer is staged while it runs.</summary>
        private readonly SortedDictionary<int, HostedTables> _moeLayers = new();

        /// <summary>Layers whose routes a prompt chunk counted on the device, with the chunk's rows (the adaptive weight).</summary>
        private readonly Dictionary<HostedTables, int> _promptCounted = new(ReferenceEqualityComparer.Instance);

        private bool TryPrefetchedExperts(Tensor x, Value selected, Value weightValue, IGraphHostedExperts gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int layer, int rows, int k, out Tensor result)
        {
            result = null;
            var cuda = (CudaAllocator)Allocator;
            Tensor sel = selected.Tensor, w = weightValue.Tensor;
            if (rows < PrefetchMinRows || Kernels == null || CudaGraphCapture.IsCapturing(cuda) || x.Storage is not CudaStorage
                || x.ElementType != DType.Float32 || !x.IsContiguous() || sel?.Storage is not CudaStorage || w?.Storage is not CudaStorage
                || sel.ElementType != DType.Int32 || w.ElementType != DType.Float32 || !sel.IsContiguous() || !w.IsContiguous()
                || gate.Format != up.Format)
                return false;
            int hidden = (int)gate.Ne0, ff = (int)gate.Ne1, experts = (int)gate.Ne2;
            HostedTables t = Tables(gate, up, down, layer);
            if (t.Sets == null || !GroupedTakes(t.Sets, rows, hidden, ff) || !PrepareHostList(t)
                || !EnsurePrefetchArena(t.HostList.Length * t.PerExpert))
                return false;

            // The half the previous layer staged this one into, else this one staged now (its copies then wait in line).
            int h = Array.IndexOf(_halfLayer, layer);
            bool ahead = h >= 0 && _halfRun[h] == _runOrdinal;
            if (!ahead)
            {
                h = 1 - _lastHalf;
                StageLayer(t, h);
            }
            _lastHalf = h;
            Tensor[] tables = HalfTables(t, h);

            cuda.Context.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(cuda.Stream.Handle, _halfReady[h], 0).ThrowOnError();
            using Tensor x2 = x.View(rows, hidden);
            var output = new Tensor(Allocator, DType.Float32, rows, hidden);
            Ops.Fill(output, 0f);
            if (!TryGroupedMoe(x2, output, sel, w, t.Sets, tables, rows, k, hidden, ff, experts))
            {
                output.Dispose();
                return false;
            }
            cuda.Context.MakeCurrent();
            CudaDriverApi.cuEventRecord(_halfFree[h], cuda.Stream.Handle).ThrowOnError();
            // The next MoE layer's host experts into the other half while this one computes (after the half's last reader,
            // the previous layer's kernels; issued once this layer's kernels are queued, so the device has work meanwhile).
            HostedTables next = NextMoeLayer(layer);
            if (next != null && PrepareHostList(next) && next.HostList.Length * next.PerExpert <= _halfBytes)
                StageLayer(next, 1 - h);
            CountPromptRoutes(t, sel, rows, k);
            PrefetchedLayers++;
            if (ahead) PrefetchedAhead++;
            result = output;
            return true;
        }

        private Tensor Zeroed(int count)
        {
            var t = new Tensor(Allocator, DType.Int32, count);
            t.SetElementsAsInt(new int[count]);
            ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            return t;
        }

        /// <summary>Page locks this executor took for experts an adaptive trade moved to the host after the load locked the
        /// host's share (released with the executor, before the model's mappings go).</summary>
        private CudaHostPins _tradedPins;

        /// <summary>The layer's host experts in ascending order (staging slot j of a half holds HostList[j]) and their
        /// matrices' bytes, each page-locked (already, unless a trade moved it to the host since the load); false where an
        /// expert has no host bytes to copy from.</summary>
        private bool PrepareHostList(HostedTables t)
        {
            if (t.HostList != null) return true;
            if (t.Sets == null) return false;
            var bytes = new long[3];
            for (int m = 0; m < 3; m++)
            {
                try { bytes[m] = t.Sets[m].Expert(0).Bytes; }
                catch (NotSupportedException) { return false; }
            }
            var list = new List<int>();
            for (int e = 0; e < t.Experts; e++)
            {
                if (t.Mask[e] == 1) continue;   // (2: a host expert the device may also read in place)
                for (int m = 0; m < 3; m++)
                    if (HostBytes(t, m, e) == IntPtr.Zero) return false;
                list.Add(e);
            }
            t.MemberBytes = bytes;
            // A slot per expert, 256-byte aligned (the kernels read the stacked rows in aligned words).
            t.PerExpert = (bytes[0] + bytes[1] + bytes[2] + 255) / 256 * 256;
            t.HostList = list.ToArray();
            // A copy from pageable pages blocks this thread while the driver bounces it: lock what the load did not (only where
            // the host experts are read from the mapping: the packed copies are pinned already).
            ((CudaAllocator)Allocator).Context.MakeCurrent();
            foreach (int e in t.HostList)
                for (int m = 0; m < 3; m++)
                    if (t.Copies?.PlaceOf(e) is not >= 0)
                        (_tradedPins ??= new CudaHostPins()).Pin(HostBytes(t.Sets[m], e), bytes[m]);
            return true;
        }

        /// <summary>An expert's matrix in its stack's host mapping (a member's own host copy may be released).</summary>
        private static IntPtr HostBytes(IGraphExpertWeights set, int e)
            => set.Data != IntPtr.Zero ? set.Data + (nint)(set.Bytes / set.Ne2 * e) : set.Expert(e).Data;

        /// <summary>Where a host expert's matrix is read from: its place in the layer's packed host copies (pinned), else its
        /// place in the stack's mapping.</summary>
        private static IntPtr HostBytes(HostedTables t, int m, int e)
            => t.Copies?.Bytes(m, e) is { } packed && packed != IntPtr.Zero ? packed : HostBytes(t.Sets[m], e);

        private HostedTables NextMoeLayer(int layer)
        {
            foreach (KeyValuePair<int, HostedTables> kv in _moeLayers)
                if (kv.Key > layer) return kv.Value;
            return null;
        }

        /// <summary>Every host expert of <paramref name="t"/>'s layer copied into half <paramref name="h"/> on the copy
        /// stream, after the half's last reader; the half's ready event follows the copies.</summary>
        private void StageLayer(HostedTables t, int h)
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(_copyStream, _halfFree[h], 0).ThrowOnError();
            IntPtr half = _prefetchArena + (nint)(h * _halfBytes);
            for (int j = 0; j < t.HostList.Length; j++)
            {
                int e = t.HostList[j];
                long offset = j * t.PerExpert;
                for (int m = 0; m < 3; m++)
                {
                    int rc = CudaHostPins.CopyToDevice(half + (nint)offset, HostBytes(t, m, e), t.MemberBytes[m], _copyStream);
                    if (rc != 0)
                        throw new InvalidOperationException($"layer {t.Layer}: staging host expert {e} matrix {m} ahead failed (CUDA error {rc})");
                    offset += t.MemberBytes[m];
                }
            }
            CudaDriverApi.cuEventRecord(_halfReady[h], _copyStream).ThrowOnError();
            _halfLayer[h] = t.Layer;
            _halfRun[h] = _runOrdinal;
        }

        /// <summary>The layer's pointer tables for half <paramref name="h"/>: the tier's copies, and each host expert at its
        /// slot of the half (rebuilt after a trade or a new arena; uploaded on the step's stream).</summary>
        private Tensor[] HalfTables(HostedTables t, int h)
        {
            t.Half ??= new Tensor[2][];
            t.HalfGeneration ??= new long[] { -1, -1 };
            if (t.Half[h] != null && t.HalfGeneration[h] == _arenaGeneration) return t.Half[h];
            IntPtr half = _prefetchArena + (nint)(h * _halfBytes);
            t.Half[h] ??= new[]
            {
                new Tensor(Allocator, DType.Int32, 2L * t.Experts), new Tensor(Allocator, DType.Int32, 2L * t.Experts),
                new Tensor(Allocator, DType.Int32, 2L * t.Experts),
            };
            long memberOffset = 0;
            for (int m = 0; m < 3; m++)
            {
                var words = (int[])t.Words[m].Clone();
                for (int j = 0; j < t.HostList.Length; j++)
                {
                    long p = (half + (nint)(j * t.PerExpert + memberOffset)).ToInt64();
                    int e = t.HostList[j];
                    words[2 * e] = unchecked((int)p);
                    words[2 * e + 1] = unchecked((int)(p >> 32));
                }
                t.Half[h][m].SetElementsAsInt(words);
                ((CudaStorage)t.Half[h][m].Storage).EnsureDeviceCurrent();
                memberOffset += t.MemberBytes[m];
            }
            t.HalfGeneration[h] = _arenaGeneration;
            return t.Half[h];
        }

        private bool _arenasSized;

        /// <summary>
        /// Both staging arenas at their full size at the first expert step (the load's warm-up), while the device's memory
        /// is still free: sized by the most experts the host serves in a layer (<see cref="IGraphExpertHost.MostHostExperts"/>),
        /// so no prompt grows them mid-request (a growth synchronizes and releases the memory pool's blocks, and every
        /// buffer of that chunk was then allocated anew: the first 2,048 rows of a 4K prompt ran in 1.78 s against 0.90).
        /// </summary>
        private void SizeStagingArenas(IGraphHostedExperts gate, IGraphExpertWeights up, IGraphExpertWeights down)
        {
            if (_arenasSized || CudaGraphCapture.IsCapturing((CudaAllocator)Allocator)) return;
            _arenasSized = true;
            int most = gate.Host.MostHostExperts;
            if (most <= 0) return;
            long[] member = new long[3];
            try { (member[0], member[1], member[2]) = (gate.Expert(0).Bytes, up.Expert(0).Bytes, down.Expert(0).Bytes); }
            catch (NotSupportedException) { return; }
            long bytes = member[0] + member[1] + member[2];
            long half = most * ((bytes + 255) / 256 * 256);
            bool ahead = EnsurePrefetchArena(half);
            // In the device plan's slots (each matrix from a 256-byte boundary), which the host-planned copies fit too.
            EnsureStagingArena(TradeStride(member), most);
            Console.Error.WriteLine(ahead
                ? $"[cuda] long prompt chunks stage each next layer's host experts ahead: two halves of {_halfBytes >> 20} MiB "
                  + $"(up to {most} host experts a layer), and {_stagingBytes >> 20} MiB for shorter chunks."
                : $"[cuda] long prompt chunks stage a layer's host experts as its routes select them ({_stagingBytes >> 20} MiB of "
                  + $"staging): two halves of {half >> 20} MiB for {most} host experts a layer do not fit the device's free memory.");
        }

        /// <summary>Two halves of at least <paramref name="halfNeed"/> bytes each (a quarter more, for layers with a few
        /// more host experts), within half the staging cap and the device's free memory less the floor; the pool's
        /// cached blocks are released first where they hold the room.</summary>
        private bool EnsurePrefetchArena(long halfNeed)
        {
            if (halfNeed <= 0) halfNeed = 1;
            if (halfNeed <= _halfBytes) return true;
            if (PrefetchArenaRefusedForTests) return false;
            long want = Math.Min(StagingArenaCapBytes / 2, halfNeed + halfNeed / 4);
            if (want < halfNeed) return false;
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            cuda.Stream.Synchronize();
            if (_copyStream != IntPtr.Zero) CudaDriverApi.cuStreamSynchronize(_copyStream).ThrowOnError();
            if (_prefetchArena != IntPtr.Zero) CudaDriverApi.cuMemFree(_prefetchArena);
            _prefetchArena = IntPtr.Zero;
            _halfBytes = 0;
            _halfLayer[0] = _halfLayer[1] = -1;
            _arenaGeneration++;
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out _).ThrowOnError();
            long shortfall = 2 * want - ((long)free.ToUInt64() - StagingFreeFloorBytes);
            if (shortfall > 0)
            {
                // Only what the halves lack goes back, the least recently used cached blocks first.
                cuda.TrimLargeCache(shortfall);
                cuda.Context.MakeCurrent();
                CudaDriverApi.cuMemGetInfo(out free, out _).ThrowOnError();
            }
            if ((long)free.ToUInt64() - StagingFreeFloorBytes < 2 * want
                || CudaDriverApi.cuMemAlloc(out IntPtr arena, new UIntPtr((ulong)(2 * want))) != 0)
                return false;
            _prefetchArena = arena;
            _halfBytes = want;
            if (_copyStream == IntPtr.Zero)
            {
                CudaDriverApi.cuStreamCreate(out _copyStream, 0x1 /* NON_BLOCKING */).ThrowOnError();
                for (int h = 0; h < 2; h++)
                {
                    CudaDriverApi.cuEventCreate(out _halfReady[h], 0x2 /* DISABLE_TIMING */).ThrowOnError();
                    CudaDriverApi.cuEventCreate(out _halfFree[h], 0x2).ThrowOnError();
                }
            }
            return true;
        }

        /// <summary>The routes prompt chunks counted on the device (<see cref="CountPromptRoutes"/>), handed to the expert hosts
        /// (weighted as the chunk's rows weigh in the adaptive tiers) and zeroed, every layer's in one wait; never inside a
        /// recording. A step's start harvests the step before it, so each chunk's routes carry its own weight.</summary>
        private void HarvestPromptCounts()
        {
            if (_promptCounted.Count == 0) return;
            var cuda = (CudaAllocator)Allocator;
            if (CudaGraphCapture.IsCapturing(cuda)) return;
            var layers = new List<(HostedTables T, int Rows)>(_promptCounted.Count);
            foreach ((HostedTables t, int rows) in _promptCounted) layers.Add((t, rows));
            _promptCounted.Clear();
            var tensors = new Tensor[layers.Count];
            var experts = new int[layers.Count];
            for (int i = 0; i < layers.Count; i++)
            {
                tensors[i] = layers[i].T.PromptCounts;
                experts[i] = layers[i].T.Experts;
            }
            int[][] counts = TakeCounts(tensors, experts);
            var ids = new List<int>();
            for (int i = 0; i < layers.Count; i++)
                RecordCounts(layers[i].T, counts[i], GraphExpertAdapt.Weight(layers[i].Rows), ids);
        }

        /// <summary>A prompt chunk's routes counted on the device for the layer's expert host (the routing profile and the
        /// adaptive tiers), handed over at the next step's start (<see cref="HarvestPromptCounts"/>).</summary>
        private void CountPromptRoutes(HostedTables t, Tensor sel, int rows, int k)
        {
            t.PromptCounts ??= Zeroed(t.Experts);
            Launch(new[] { sel }, t.PromptCounts, (kn, stream) => kn.LaunchCountRoutes(Ptr(sel), rows * k, Ptr(t.PromptCounts), t.Experts, stream));
            _promptCounted[t] = Math.Max(rows, _promptCounted.TryGetValue(t, out int counted) ? counted : 0);
        }

        /// <summary>A trade moved the layer's tier: its host list and half tables are rebuilt when next used.</summary>
        private static void TierMoved(HostedTables t)
        {
            t.HostList = null;
            if (t.HalfGeneration != null) t.HalfGeneration[0] = t.HalfGeneration[1] = -1;
        }

        private void DisposePrefetch()
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            if (_copyStream != IntPtr.Zero)
            {
                CudaDriverApi.cuStreamSynchronize(_copyStream);
                CudaDriverApi.cuStreamDestroy(_copyStream);
                _copyStream = IntPtr.Zero;
            }
            for (int h = 0; h < 2; h++)
            {
                if (_halfReady[h] != IntPtr.Zero) CudaDriverApi.cuEventDestroy(_halfReady[h]);
                if (_halfFree[h] != IntPtr.Zero) CudaDriverApi.cuEventDestroy(_halfFree[h]);
                _halfReady[h] = _halfFree[h] = IntPtr.Zero;
            }
            if (_prefetchArena != IntPtr.Zero)
            {
                cuda.Stream.Synchronize();
                CudaDriverApi.cuMemFree(_prefetchArena);
            }
            _prefetchArena = IntPtr.Zero;
            _halfBytes = 0;
            _halfLayer[0] = _halfLayer[1] = -1;
            foreach (HostedTables t in _moeLayers.Values)
            {
                if (t.Half != null)
                    foreach (Tensor[] tables in t.Half)
                        if (tables != null) foreach (Tensor table in tables) table.Dispose();
                t.Half = null;
                t.PromptCounts?.Dispose();
                t.PromptCounts = null;
            }
            _moeLayers.Clear();
            _promptCounted.Clear();
            _tradedPins?.Dispose();
            _tradedPins = null;
        }
    }
}
