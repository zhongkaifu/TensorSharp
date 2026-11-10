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
using System.Collections.Generic;

namespace TensorSharp.Models
{
    /// <summary>
    /// Rows of the host paged pool and where they sit in the batched fused decode's device pool:
    /// <c>Rows</c> rows from host slot <c>HostSlot</c> and device slot <c>DeviceSlot</c> on.
    /// </summary>
    internal readonly record struct PoolRowRange(long HostSlot, long DeviceSlot, int Rows);

    /// <summary>The transfers <see cref="BatchedDecodePool"/> orders, per attention layer K and V alike.</summary>
    internal interface IBatchedDecodePoolTransfers
    {
        /// <summary>Rows the device pool holds now.</summary>
        int DeviceSlots { get; }

        /// <summary>A new device pool of <paramref name="deviceSlots"/> rows (the old one and its rows dropped).
        /// False when it could not be allocated (<see cref="LastError"/>).</summary>
        bool Resize(int deviceSlots);

        /// <summary>Host pool rows into the device pool; <paramref name="zeroFirst"/> clears the whole device pool
        /// first. False on failure (<see cref="LastError"/>), the device pool then in an unknown state.</summary>
        bool Upload(IReadOnlyList<PoolRowRange> ranges, bool zeroFirst);

        /// <summary>Device pool rows back into the host pool. False on failure (<see cref="LastError"/>).</summary>
        bool Download(IReadOnlyList<PoolRowRange> ranges);

        /// <summary>The device pool still holds what was uploaded to it and written in it (its device copies are
        /// resident).</summary>
        bool Resident { get; }

        string LastError { get; }
    }

    /// <summary>
    /// Which device block of the batched fused decode's pool holds which block of the host paged pool. The device
    /// pool used to be indexed by host block id, so it was as large as the host pool (256 blocks: 8 GiB of F32 K/V
    /// for Qwen3.8-27B's 16 attention layers) however few blocks the batch used; the block pool hands out ids
    /// first-in-first-out, so even one short request reached the top ids. Here a host block takes a device block
    /// when a step first needs it and keeps it while it lives; a device block whose host block the block pool has
    /// freed goes to the next block that needs one; and only when none is free does the pool grow, by
    /// <see cref="GrowthSlots"/> rows at a time.
    /// </summary>
    internal sealed class BatchedDecodeBlockMap
    {
        /// <summary>Rows the device pool grows by (whole blocks).</summary>
        internal const int GrowthSlots = 2048;

        private readonly int _blockSize;
        private readonly int _growthBlocks;
        private int[] _deviceOf = Array.Empty<int>();   // host block -> device block, -1 unmapped
        private int[] _hostOf = Array.Empty<int>();     // device block -> host block, -1 free

        public BatchedDecodeBlockMap(int blockSize, int growthSlots = GrowthSlots)
        {
            if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
            if (growthSlots <= 0) throw new ArgumentOutOfRangeException(nameof(growthSlots));
            _blockSize = blockSize;
            _growthBlocks = Math.Max(1, (growthSlots + blockSize - 1) / blockSize);
        }

        public int BlockSize => _blockSize;
        public int DeviceBlocks => _hostOf.Length;
        public int DeviceSlots => checked(_hostOf.Length * _blockSize);

        /// <summary>The device block holding <paramref name="hostBlock"/>, or -1.</summary>
        public int DeviceBlockOf(int hostBlock)
            => hostBlock >= 0 && hostBlock < _deviceOf.Length ? _deviceOf[hostBlock] : -1;

        /// <summary>The host block held by <paramref name="deviceBlock"/>, or -1 when it is free.</summary>
        public int HostBlockOf(int deviceBlock) => _hostOf[deviceBlock];

        /// <summary>The device slot of a mapped host slot.</summary>
        public long DeviceSlotOf(long hostSlot)
        {
            int device = DeviceBlockOf(checked((int)(hostSlot / _blockSize)));
            if (device < 0)
                throw new InvalidOperationException($"Host slot {hostSlot} is in no mapped block.");
            return (long)device * _blockSize + hostSlot % _blockSize;
        }

        /// <summary>Every mapped (host block, device block) pair.</summary>
        public IEnumerable<(int Host, int Device)> Mapped()
        {
            for (int d = 0; d < _hostOf.Length; d++)
                if (_hostOf[d] >= 0) yield return (_hostOf[d], d);
        }

        /// <summary>
        /// Map the host blocks one step reads or writes. A mapped block keeps its device block. A new one takes a
        /// free device block, else one whose host block <paramref name="isLive"/> reports freed (and not needed by
        /// this step), else the pool grows by whole <see cref="GrowthSlots"/> steps. Returns the device blocks that
        /// were assigned (their host rows belong in the device pool before the step); <paramref name="grew"/> when
        /// the pool must be reallocated larger.
        /// </summary>
        public List<int> Map(IReadOnlyList<int> stepHostBlocks, Func<int, bool> isLive, out bool grew)
        {
            grew = false;
            var joined = new List<int>();
            var needed = new HashSet<int>(stepHostBlocks);
            bool swept = false;
            foreach (int host in stepHostBlocks)
            {
                if (host < 0) throw new ArgumentOutOfRangeException(nameof(stepHostBlocks), $"Block id {host}.");
                if (host >= _deviceOf.Length)
                {
                    int old = _deviceOf.Length;
                    Array.Resize(ref _deviceOf, Math.Max(host + 1, old * 2));
                    Array.Fill(_deviceOf, -1, old, _deviceOf.Length - old);
                }
                if (_deviceOf[host] >= 0)
                    continue;
                int device = Array.IndexOf(_hostOf, -1);
                if (device < 0 && !swept)
                {
                    swept = true;
                    for (int d = 0; d < _hostOf.Length; d++)
                        if (_hostOf[d] >= 0 && !needed.Contains(_hostOf[d]) && !isLive(_hostOf[d]))
                        {
                            _deviceOf[_hostOf[d]] = -1;
                            _hostOf[d] = -1;
                        }
                    device = Array.IndexOf(_hostOf, -1);
                }
                if (device < 0)
                {
                    int old = _hostOf.Length;
                    Array.Resize(ref _hostOf, old + _growthBlocks);
                    Array.Fill(_hostOf, -1, old, _growthBlocks);
                    device = old;
                    grew = true;
                }
                _hostOf[device] = host;
                _deviceOf[host] = device;
                joined.Add(device);
            }
            return joined;
        }

        /// <summary>Forget every mapping and the pool's size.</summary>
        public void Clear()
        {
            _deviceOf = Array.Empty<int>();
            _hostOf = Array.Empty<int>();
        }
    }

    /// <summary>
    /// The batched fused decode's device K/V pool kept coherent with the host paged pool, which stays the source of
    /// truth for the op-by-op steps, the prefix cache and every reseed.
    /// <para>
    /// The native graph appends each step's rows in the device pool. The pool used to be kept in step through the
    /// pool tensors' HOST bytes: a seed wrote them and the mirror read them back. On ggml_cuda the pool is a device
    /// copy, so neither reached it: the mirror copied stale host bytes over the rows the step had written, and a
    /// reseed (after any op-by-op step, such as a request joining) left the device pool without the rows that step
    /// wrote. Every transfer here is an explicit host/device row copy, in this order:
    /// <list type="number">
    /// <item>after a step, the rows it wrote are downloaded into the host pool before anything else touches either
    /// pool (<see cref="CompleteStep"/>, and again before any reseed or invalidation, should a step have thrown
    /// between its compute and its download);</item>
    /// <item>before a step, the device pool is sized to the blocks the batch maps, and seeded from the host pool:
    /// every mapped block after a resize, an invalidation or a lost device copy, else only the blocks that
    /// joined.</item>
    /// </list>
    /// A failed allocation or upload defers the step (the caller runs it op-by-op) and leaves the pool to be
    /// reseeded; it is never taken as the kernel being unsupported.
    /// </para>
    /// </summary>
    internal sealed class BatchedDecodePool
    {
        private readonly BatchedDecodeBlockMap _map;
        private readonly IBatchedDecodePoolTransfers _io;
        private bool _seeded;
        private List<PoolRowRange> _pending;   // rows a step wrote that have not reached the host pool

        public BatchedDecodePool(int blockSize, IBatchedDecodePoolTransfers io, int growthSlots = BatchedDecodeBlockMap.GrowthSlots)
        {
            _map = new BatchedDecodeBlockMap(blockSize, growthSlots);
            _io = io ?? throw new ArgumentNullException(nameof(io));
        }

        public int BlockSize => _map.BlockSize;
        public int DeviceSlots => _map.DeviceSlots;
        public bool Seeded => _seeded;
        public BatchedDecodeBlockMap Map => _map;

        public long DeviceSlotOf(long hostSlot) => _map.DeviceSlotOf(hostSlot);

        /// <summary>The host pool was written under the device pool (an op-by-op step, a restore): the next step
        /// reseeds. A step's rows still pending reach the host pool first, so the writer sees them.</summary>
        public void InvalidateSeed()
        {
            CompletePendingDownload();
            _seeded = false;
        }

        /// <summary>
        /// Map and seed for a step over <paramref name="stepHostBlocks"/>. False defers the step:
        /// <paramref name="reason"/> says why (no memory for the pool, a failed upload), and the next step starts
        /// from a full reseed.
        /// </summary>
        public bool TryPrepare(IReadOnlyList<int> stepHostBlocks, Func<int, bool> isLive, out string reason)
        {
            reason = null;
            CompletePendingDownload();
            List<int> joined = _map.Map(stepHostBlocks, isLive, out _);
            if (_io.DeviceSlots != _map.DeviceSlots)
            {
                _seeded = false;
                if (!_io.Resize(_map.DeviceSlots))
                {
                    reason = $"no memory for a {_map.DeviceSlots}-row device pool ({_io.LastError})";
                    return false;
                }
            }
            if (_seeded && !_io.Resident)
                _seeded = false;
            bool ok = _seeded ? joined.Count == 0 || _io.Upload(RangesOf(joined), zeroFirst: false)
                              : _io.Upload(RangesOf(null), zeroFirst: true);
            if (!ok)
            {
                _seeded = false;
                reason = $"seeding the device pool failed ({_io.LastError})";
                return false;
            }
            _seeded = true;
            return true;
        }

        /// <summary>
        /// Before the step's compute: the host slots it will write (at their device slots). They are owed to the host
        /// pool from here on, so a step that throws after its compute still has them downloaded before the next
        /// reseed or invalidation. (A step that declines before computing downloads rows it did not write; the
        /// op-by-op step that replaces it rewrites them on the host.)
        /// </summary>
        public void BeginStep(ReadOnlySpan<int> hostSlots)
        {
            CompletePendingDownload();
            var rows = new List<PoolRowRange>(hostSlots.Length);
            foreach (int slot in hostSlots)
                rows.Add(new PoolRowRange(slot, _map.DeviceSlotOf(slot), 1));
            _pending = rows;
        }

        /// <summary>After the step's compute: download the rows it wrote into the host pool.</summary>
        public void CompleteStep() => CompletePendingDownload();

        /// <summary>
        /// The step declined (its graph did not build, allocate or run): nothing it owes the host pool is worth
        /// downloading, since the op-by-op step that replaces it computes those rows on the host, and that step's
        /// <see cref="InvalidateSeed"/> reseeds the device pool over whatever a failed compute left in it.
        /// </summary>
        public void AbandonStep()
        {
            _pending = null;
            _seeded = false;
        }

        private void CompletePendingDownload()
        {
            if (_pending == null)
                return;
            var rows = _pending;
            if (!_io.Download(rows))
                throw new InvalidOperationException(
                    "The batched fused decode could not copy the K/V rows it wrote back to the host paged pool " +
                    $"({_io.LastError}); the host pool would miss them.");
            _pending = null;
        }

        private List<PoolRowRange> RangesOf(List<int> deviceBlocks)
        {
            int bs = _map.BlockSize;
            var ranges = new List<PoolRowRange>();
            if (deviceBlocks == null)
            {
                foreach (var (host, device) in _map.Mapped())
                    ranges.Add(new PoolRowRange((long)host * bs, (long)device * bs, bs));
            }
            else
            {
                foreach (int device in deviceBlocks)
                    ranges.Add(new PoolRowRange((long)_map.HostBlockOf(device) * bs, (long)device * bs, bs));
            }
            return ranges;
        }
    }
}
