// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Each offloaded layer's host experts packed in page-locked host memory the device reads (null: none, or
        /// the host could not pin them and their file ranges are locked instead).</summary>
        private HostExpertCopies[] _hostCopies;

        /// <summary>The copies' allocations, freed with the model once its executors are gone.</summary>
        private readonly List<IntPtr> _hostCopyBlocks = new();

        private long _hostCopyBytes;

        /// <summary>What the copies were allocated from (freed through it).</summary>
        private IMappedHostAllocator _hostCopyAllocator;

        /// <summary>
        /// The host experts packed, once, at load, as the direct engines keep them (Q4eCudaEngine.PinHostExperts): per offloaded
        /// layer and matrix one block of page-locked host memory the device addresses, each host expert at a place, in expert
        /// order, read from the model file's mapping by parallel runs. The device stages and reads its misses there and the
        /// host computes there, and an adaptive trade only moves bytes between places: nothing is page-locked while the model
        /// serves. Locking the experts' file ranges instead had registered the pages of every expert a trade moved to the host,
        /// on a worker beside the decode: Qwen3.8 Flash Next UD-Q2_K_XL on an A40 decoded 57.1 / 54.4 tok/s (short / after a
        /// 4K prompt) against the engine's 62.3 / 60.2. False, with nothing allocated, where the allocator cannot map host
        /// memory, a stack has no host bytes, or the copies and <see cref="HostPinReserve"/> do not fit the host's available
        /// memory (<paramref name="available"/>, 0 unknown), as the engine decides (its HostPinnedNeed): the file ranges are
        /// locked instead.
        /// </summary>
        /// <param name="held">Whether expert e of layer l is a tier member (no place).</param>
        /// <param name="allocator">The memory (the executor's allocator's, where it maps host memory for its device).</param>
        private bool TryPackHostExperts(long available, Func<int, int, bool> held, IMappedHostAllocator allocator = null)
        {
            IMappedHostAllocator alloc = allocator ?? _allocator as IMappedHostAllocator;
            if (IsGgmlBackend || alloc == null) return false;
            var layers = new List<(int Layer, StackedExpertWeights[] Stacks, int[] Place, int Count)>();
            long need = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!IsExpertLayerOffloaded(l) || !TryGetLayerExperts(l, out var gate, out var up, out var down)) continue;
                var stacks = new[] { gate, up, down };
                foreach (StackedExpertWeights w in stacks)
                    if (w == null || w.Data == IntPtr.Zero || w.NumExperts != _numExperts) return false;
                var place = new int[_numExperts];
                int n = 0;
                for (int e = 0; e < _numExperts; e++) place[e] = held(l, e) ? -1 : n++;
                if (n == 0) continue;
                layers.Add((l, stacks, place, n));
                foreach (StackedExpertWeights w in stacks) need += n * w.PerExpertRawBytes;
            }
            if (layers.Count == 0) return false;
            static string Gib(long b) => (b / (double)(1L << 30)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            if (available > 0 && need + HostPinReserve > available)
            {
                Console.WriteLine($"[moe-offload] {Config.Architecture}: packing the host experts into pinned host memory takes {Gib(need)} GiB, " +
                    $"the host has {Gib(available)} GiB available ({Gib(HostPinReserve)} GiB kept free): their file ranges are page-locked instead.");
                return false;
            }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var copies = new HostExpertCopies[Config.NumLayers];
            var blocks = new List<IntPtr>();
            var reads = new List<(IntPtr Dst, IntPtr Src, long Bytes)>();
            foreach (var (l, stacks, place, n) in layers)
            {
                var hosts = new IntPtr[3];
                var devices = new IntPtr[3];
                var bytes = new long[3];
                for (int m = 0; m < 3; m++)
                {
                    long per = stacks[m].PerExpertRawBytes;
                    IntPtr host = alloc.AllocateMappedHost(n * per, out IntPtr device);
                    if (host == IntPtr.Zero)
                    {
                        foreach (IntPtr b in blocks) alloc.FreeMappedHost(b);
                        Console.WriteLine($"[moe-offload] {Config.Architecture}: pinning {Gib(need)} GiB of host experts ran out of host memory: " +
                            "their file ranges are page-locked instead.");
                        return false;
                    }
                    blocks.Add(host);
                    (hosts[m], devices[m], bytes[m]) = (host, device, per);
                    // One read a run of consecutive host experts (at most 64 MiB), into consecutive places.
                    int maxRun = (int)Math.Max(1, (64L << 20) / Math.Max(1, per));
                    for (int e = 0; e < _numExperts;)
                    {
                        if (place[e] < 0) { e++; continue; }
                        int end = e + 1;
                        while (end < _numExperts && end - e < maxRun && place[end] >= 0) end++;
                        reads.Add((host + (nint)(place[e] * per), stacks[m].Data + (nint)(e * per), (end - e) * per));
                        e = end;
                    }
                }
                copies[l] = new HostExpertCopies(hosts, devices, bytes, place);
            }
            // In parallel: on network storage the reads are the slow part.
            Parallel.ForEach(reads, new ParallelOptions { MaxDegreeOfParallelism = GgufFile.DefaultWarmThreads() }, r =>
            {
                unsafe { Buffer.MemoryCopy((void*)r.Src, (void*)r.Dst, r.Bytes, r.Bytes); }
            });
            _hostCopies = copies;
            _hostCopyAllocator = alloc;
            _hostCopyBlocks.AddRange(blocks);
            _hostCopyBytes = need;
            Console.WriteLine($"[moe-offload] {Config.Architecture}: packed {Gib(need)} GiB of host experts into pinned host memory in " +
                $"{sw.Elapsed.TotalSeconds:F1}s (the device reads them there, the host computes them there; nothing is page-locked while serving).");
            return true;
        }

        /// <summary>The copies freed: after the executors (their tables and doorbells read them) are disposed.</summary>
        private void FreeHostExpertCopies()
        {
            foreach (IntPtr b in _hostCopyBlocks) _hostCopyAllocator?.FreeMappedHost(b);
            _hostCopyBlocks.Clear();
            _hostCopyAllocator = null;
            _hostCopies = null;
            _hostCopyBytes = 0;
        }

        /// <summary>Tests: the host experts packed again for the tier the test named (as a planned tier's load packs them), before
        /// the first step; false where they cannot be.</summary>
        internal bool RepackHostExpertsForTests()
        {
            FreeHostExpertCopies();
            return TryPackHostExperts(MappedTableWarm.HostMemoryAvailable(),
                (l, e) => HostFor(l)?.IsResident(l, e) == true);
        }

        /// <summary>Tests: the host experts packed for the tier the test named into <paramref name="allocator"/>'s memory (a backend
        /// that reads host memory in place, or process memory where the host alone reads them), before the first step.</summary>
        internal bool PackHostExpertsForTests(IMappedHostAllocator allocator)
        {
            FreeHostExpertCopies();
            return TryPackHostExperts(0, (l, e) => HostFor(l)?.IsResident(l, e) == true, allocator);
        }

        /// <summary>Tests: the copies dropped and the host experts' file ranges page-locked instead (the load's way where the host
        /// cannot pin the copies), before the first step.</summary>
        internal void DropHostExpertCopiesForTests()
        {
            FreeHostExpertCopies();
            if (_hostPins == null && _allocator is IHostRangePinner pinner)
                _hostPins = pinner.PinHostRanges(HostExpertRuns(), 0, default);
        }

        /// <summary>Tests: the bytes of the packed copies (0: none).</summary>
        internal long HostExpertCopyBytesForTests => _hostCopyBytes;

        /// <summary>Tests: whether the load page-locked the host experts' file ranges (the way without packed copies).</summary>
        internal bool HostRangesLockedForTests => _hostPins != null;
    }
}
