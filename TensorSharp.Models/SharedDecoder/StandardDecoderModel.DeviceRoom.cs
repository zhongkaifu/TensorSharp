// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    /// <summary>
    /// Device memory on one direct-CUDA device (<see cref="DeviceRoomManaged"/>), which the model accounts and reclaims itself.
    /// Before it allocates a slot, copies one or runs a lone prompt's step, it makes room in a fixed order, the cheapest loss
    /// first (<see cref="EnsureDeviceRoom"/>): finished requests' pooled slots, parked recordings, the window buffers retained
    /// slots keep, then finished conversations the prefix cache keeps, the oldest first, and for a prompt step the cached graphs
    /// last. A new request whose slot cannot fit waits beside running ones and is refused, with the numbers, when it runs
    /// alone; a running request's growth that cannot fit fails with what holds the device. The prefix cache sees the device's
    /// deficit below its headroom (<see cref="DeviceMemoryDeficitBytes"/>), and the allocator asks the model for memory when
    /// an allocation no site sized finds the device full (<see cref="ReclaimForAllocator"/>). Before, room checks counted every
    /// block the allocator cached as free, a prompt step's first reaction to pressure was to drop every cached graph, and a
    /// 47K-token request on a 32 GB RTX PRO 4500 failed part way with CUDA error 2 (HTTP 500) where releasing a finished
    /// conversation would have made room (issue #376).
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Device memory this model accounts and reclaims itself: one direct-CUDA device, or a test's stand-in
        /// (<see cref="DeviceRoomForTests"/>). A split or tensor-parallel mesh keeps today's checks.</summary>
        private bool DeviceRoomManaged => DeviceRoomForTests != null
            || (_backend == BackendType.Cuda && _allocator is TensorSharp.Cuda.CudaAllocator
                && _mesh != null && _mesh.Stages == 1 && _mesh.TensorParallel == 1 && LayerSplitGroup == null);

        /// <summary>Tests stand in for the device: its free bytes (with the blocks its allocator frees on demand) and total,
        /// read at every question.</summary>
        internal Func<(long Free, long Total)> DeviceRoomForTests { get; set; }

        /// <summary>How often each stage of <see cref="EnsureDeviceRoom"/> (and the allocator's reclaimer) freed something,
        /// by stage: 1 cached blocks trimmed, 2 pooled slots, 3 parked recordings, 4 retained slots' window buffers,
        /// 5 retained conversations, 6 the cached graphs.</summary>
        internal int[] DeviceRoomStages { get; } = new int[7];

        /// <summary>Tests: room for <paramref name="bytes"/> of recorded steps beside the running requests, as a step that could
        /// not record asks for it (<see cref="RecordingRoomBytes"/>), at a size the test chooses; with <paramref name="mayEvict"/>
        /// false, as a check that releases no conversation asks.</summary>
        internal bool EnsureRecordingRoomForTests(long bytes, bool mayEvict = true)
            => EnsureDeviceRoom(bytes, RunningSlots(), RoomUse.Recording, mayEvict, "recorded steps (test)");

        /// <summary>New requests refused because their slot does not fit the device alone (tests check the route).</summary>
        internal int SlotRefusals { get; private set; }

        /// <summary>New requests that wait for a running one to finish before their slot fits (tests check the route).</summary>
        internal int SlotWaits { get; private set; }

        private enum RoomUse
        {
            /// <summary>A slot's caches and states (made, copied or grown): beside the headroom and room for a prompt step or
            /// a request's recordings, whichever is larger.</summary>
            Slot,
            /// <summary>A lone prompt's step: its rows' activations and scratch, as the step was sized.</summary>
            PromptStep,
            /// <summary>Recorded steps (<see cref="RecordingRoomBytes"/>); the parked recordings are not given up for them.</summary>
            Recording,
        }

        // ---------------------------------------------------------------- the device

        // The last reading of the device, and when it was taken (Stopwatch ticks; 0: none since this model allocated or freed).
        private (long Free, long Total) _deviceReading;
        private long _deviceReadAt;
        private bool _deviceRoomLogged;

        private static readonly long DeviceReadTicks = Stopwatch.Frequency / 10;

        /// <summary>
        /// The device's free bytes, with the large blocks the CUDA allocator caches (it frees them on demand; the shards' small
        /// blocks are not counted), and its total; (long.MaxValue, 0) where not known. Read again when <paramref name="fresh"/>,
        /// after this model allocated or freed a slot, and 100 ms after the last reading otherwise: cuMemGetInfo took about
        /// 3 ms on an RTX PRO 4000.
        /// </summary>
        private (long Free, long Total) DeviceMemory(bool fresh = false)
        {
            if (DeviceRoomForTests is { } stand) return stand();
            long now = Stopwatch.GetTimestamp();
            if (!fresh && _deviceReadAt != 0 && now - _deviceReadAt < DeviceReadTicks) return _deviceReading;
            if (_allocator is not TensorSharp.Cuda.CudaAllocator cuda || !cuda.TryGetDeviceMemory(out long free, out long total))
                return (long.MaxValue, 0);
            if (!_deviceRoomLogged)
            {
                _deviceRoomLogged = true;
                cuda.LogDeviceRoom($"{Config.Architecture} device room");
            }
            _deviceReading = (free, total);
            _deviceReadAt = now;
            return _deviceReading;
        }

        /// <summary>This model allocated or freed device memory: the next question reads the device anew.</summary>
        private void NoteDeviceMemoryChanged() => _deviceReadAt = 0;

        /// <summary>What a use's room is measured against, read anew: a lone prompt's step and recordings by what tests stand in
        /// for it (<see cref="PromptFreeBytesForTests"/>), else the device.</summary>
        private long RoomFree(RoomUse use)
        {
            long device = DeviceMemory(fresh: true).Free;
            return use != RoomUse.Slot && PromptFreeBytesForTests is { } prompt ? prompt() : device;
        }

        /// <summary>The narrowest lone-prompt step on direct CUDA (256 rows and its scratch), which a prompt narrows to when
        /// the device is short (<see cref="CudaPromptStepTokens"/>); 0 elsewhere.</summary>
        private long NarrowestPromptStepBytes => _backend == BackendType.Cuda ? 256 * CudaPromptRowBytes() + CudaPromptReserveBytes : 0;

        /// <summary>
        /// The free bytes <paramref name="bytes"/> of <paramref name="use"/> want: a slot's beside the headroom and the larger of
        /// a lone prompt's step and a request's recordings (a slot that took them made the next prompt narrower and its decode
        /// unrecorded); a prompt step's and the recordings' as 4c819ed2 sized them, without the headroom (a step that narrows
        /// against more than that would split prompts where it did not, which changes their arithmetic).
        /// </summary>
        private long RoomNeed(long bytes, long total, RoomUse use)
            => use == RoomUse.Slot ? bytes + DeviceHeadroom(total) + Math.Max(PromptStepBytes, RecordingRoomBytes) : bytes;

        /// <summary>Whether <paramref name="bytes"/> fit beside the headroom and the narrowest prompt step: a slot that does
        /// runs its request (narrower prompt steps, decode steps that record only where they find room), one that does not
        /// would fail part way with the device out of memory.</summary>
        private bool FitsBesideHeadroom(long bytes)
        {
            var (free, total) = DeviceMemory(fresh: true);
            return total <= 0 || free >= bytes + DeviceHeadroom(total) + NarrowestPromptStepBytes;
        }

        // ---------------------------------------------------------------- making room

        /// <summary>
        /// Make room for <paramref name="bytes"/> of <paramref name="use"/> (<see cref="RoomNeed"/>), the cheapest loss first,
        /// each stage only while the device is still short:
        /// <list type="number">
        /// <item>(last, once there is room) the allocator gives back the cached blocks the allocation needs at the driver, so
        /// its tensors do not each fail a cuMemAlloc and climb (nothing is lost);</item>
        /// <item>pooled slots, the largest first (they belong to no conversation);</item>
        /// <item>parked recordings that bind none of <paramref name="keep"/> (a later step records again), not for recordings;</item>
        /// <item>the window buffers the conversations the prefix cache keeps hold (<see cref="Retained"/>), the oldest first, never
        /// one of <paramref name="keep"/> (the next verify on such a slot allocates them again and records its windows anew);</item>
        /// <item>with <paramref name="mayEvict"/>, the conversations the prefix cache keeps, the oldest first, each reported to
        /// it as reclaimed (their next turns prefill again), never one of <paramref name="keep"/>; then, still short, the drafter
        /// rings no request holds, with the contexts they keep for finished requests' next turns
        /// (ModelBase.ReleaseIdleDFlashRings), the least recently kept first (counted as stage 4);</item>
        /// <item>for a prompt step, the cached graphs and every recording, which had been the first reaction.</item>
        /// </list>
        /// <paramref name="keep"/> holds the slots a step or a copy is about to use. False when even these made no room. One
        /// line goes to stderr for every call that released a conversation, naming what for.
        /// <para>The answer is the room: a caller does not read the device again to second-guess it. The window buffers stage 4
        /// frees wait in the CUDA allocator's small-block cache, which no reading of the device counts (an allocation that needs
        /// them takes them back at the reclaim ladder's second rung), so a reading after a true answer can still look short, and
        /// acting on it lost a conversation or narrowed a prompt step for room that had been made.</para>
        /// </summary>
        /// <param name="blockBytes">The size of the tensors the allocation is made of, whose cached blocks stay.</param>
        private bool EnsureDeviceRoom(long bytes, ReadOnlySpan<Slot> keep, RoomUse use, bool mayEvict, string what, long blockBytes = 0)
            => EnsureDeviceRoom(bytes, keep, use, mayEvict, what, out _, blockBytes);

        /// <summary>Make room as above, and say how much there is.</summary>
        /// <param name="room">The free bytes the stages reached, with the window buffers stage 4 freed (what a caller sizes its
        /// allocation from); long.MaxValue where the model does not manage the device or <paramref name="bytes"/> are none.</param>
        private bool EnsureDeviceRoom(long bytes, ReadOnlySpan<Slot> keep, RoomUse use, bool mayEvict, string what, out long room, long blockBytes = 0)
        {
            room = long.MaxValue;
            if (!DeviceRoomManaged || bytes <= 0) return true;
            _slotMutationDepth++;
            try
            {
                long free = RoomFree(use), before = free;
                room = free;
                long total = _deviceReading.Total;
                if (DeviceRoomForTests is { } stand) total = stand().Total;
                if (total <= 0) return true;   // the device does not say
                long need = RoomNeed(bytes, total, use);
                if (free < need) (_allocator as TensorSharp.Cuda.CudaAllocator)?.LogDeviceRoom($"{Config.Architecture} {what}: {Mib(need - free)} MiB short");

                // 2. Pooled slots, the largest first.
                bool freed = false;
                while (free < need && _slotPool is { Count: > 0 })
                {
                    Slot largest = _slotPool[0];
                    foreach (Slot pooled in _slotPool)
                        if (pooled.Capacity > largest.Capacity) largest = pooled;
                    _slotPool.Remove(largest);
                    DisposeSlot(largest);
                    freed = true;
                    free = RoomFree(use);
                }
                if (freed) DeviceRoomStages[2]++;

                // 3. Parked recordings no slot of keep binds.
                if (free < need && use != RoomUse.Recording && DisposeParkedRecordingsNotBinding(keep) > 0)
                {
                    DeviceRoomStages[3]++;
                    free = RoomFree(use);
                }

                // 4. The window buffers the conversations the prefix cache keeps hold, the oldest first, where they cover the
                // shortfall (short of it, conversations go next, and the buffers of those that stay would be lost for nothing).
                // They count as freed: each is under the CUDA allocator's large-block size, so a freed one waits in its
                // small-block cache, which the reading leaves out and an allocation that needs it takes back before it asks
                // anyone for memory (the reclaim ladder's second rung). Re-read only, the reading did not move: every retained
                // slot gave its buffers up, and the conversations went anyway.
                if (free < need && WindowBufferBytes(keep) >= need - free)
                {
                    bool buffers = false;
                    long counted = free;
                    foreach (Slot slot in _retainedSlots.Values.OrderBy(s => s.RetainedAt).ToList())
                    {
                        if (free >= need) break;
                        if (slot.K == null || Holds(keep, slot)) continue;
                        long windowBytes = ReleaseWindowBuffers(slot);
                        if (windowBytes <= 0) continue;
                        buffers = true;
                        counted += windowBytes;
                        NoteDeviceMemoryChanged();
                        free = Math.Max(RoomFree(use), counted);   // a reading that sees them (frees past the pool) counts them once
                    }
                    if (buffers) DeviceRoomStages[4]++;
                }

                // 5. The conversations the prefix cache keeps, the oldest first.
                int released = 0;
                long releasedBytes = 0;
                if (free < need && mayEvict && _retainedSlots is { Count: > 0 })
                {
                    foreach (var (key, slot) in _retainedSlots.OrderBy(kv => kv.Value.RetainedAt).ToList())
                    {
                        if (free >= need) break;
                        if (slot.K == null || Holds(keep, slot)) continue;
                        releasedBytes += SlotStorageBytes(slot);
                        _retainedSlots.Remove(key);
                        DisposeSlot(slot);
                        _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
                        released++;
                        free = RoomFree(use);
                    }
                    if (released > 0)
                    {
                        DeviceRoomStages[5]++;
                        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                            $"[{Config.Architecture}] device memory short by {Mib(need - before)} MiB for {what}: released {released} retained " +
                            $"conversation{(released == 1 ? "" : "s")} ({Mib(releasedBytes)} MiB; their next turns prefill again)."));
                    }
                }

                // 5b. Still short: the drafter rings finished requests keep for their conversations' next turns (a turn whose
                // ring went drafts without its context until its rows come back), with the graphs over them, the least recently
                // kept first and only as many as needed. They go after the conversations: an 81 MB ring is little room beside a
                // conversation, and is worth a reused turn's drafts (on a 32 GB RTX PRO 4500 at 131K context, rings given back
                // ahead of the conversations - every idle one at each new long prompt, or the few that covered a small
                // shortfall - left the next reused turn of a conversation the prefix cache still kept drafting without its own:
                // 53-58% acceptance against 98%). Only where conversations may go (mayEvict): a check that releases none (a
                // slot's roomier target, SlotTarget) keeps them too.
                if (free < need && mayEvict)
                    free = ReleaseRingsForRoom(need - free, free, use);

                // 6. The cached graphs and every recording (a prompt step only).
                if (free < need && use == RoomUse.PromptStep)
                {
                    ClearGraphCache();
                    DeviceRoomStages[6]++;
                    free = RoomFree(use);
                }

                room = free;
                if (free < need) return false;
                // 1. The allocation's room at the driver: the free bytes counted the cached large blocks (the last reading's).
                if (DeviceRoomForTests == null && _allocator is TensorSharp.Cuda.CudaAllocator cuda
                    && cuda.TrimForAllocation(bytes, blockBytes, _deviceReading.Free - cuda.ReclaimableBytes) > 0)
                {
                    DeviceRoomStages[1]++;
                    NoteDeviceMemoryChanged();
                }
                return true;
            }
            finally
            {
                _slotMutationDepth--;
            }
        }

        private static long Mib(long bytes) => Math.Max(0, bytes) >> 20;

        /// <summary>Gives back idle drafter rings for <paramref name="shortBy"/> bytes of room (ModelBase.ReleaseIdleDFlashRings,
        /// counted as stage 4) and returns the free bytes after them.</summary>
        private long ReleaseRingsForRoom(long shortBy, long free, RoomUse use)
        {
            long ringBytes = ReleaseIdleDFlashRings(shortBy);
            if (ringBytes <= 0) return free;
            DeviceRoomStages[4]++;
            NoteDeviceMemoryChanged();
            return Math.Max(RoomFree(use), free + ringBytes);
        }

        /// <summary>A drafter ring made to keep a finished request's context takes only room a slot would leave (its headroom
        /// and a prompt step's or recordings' room) on a device this model manages, or one ggml_cuda or Metal device; anywhere
        /// else the pool's bound alone limits it.</summary>
        private protected override bool RoomForDFlashRing(long bytes)
        {
            if (DeviceRoomManaged)
            {
                var (free, total) = DeviceMemory();
                return total <= 0 || free >= RoomNeed(bytes, total, RoomUse.Slot);
            }
            if (AdmitsSlotsByDeviceMemory && DeviceMemoryProbe() is { } memory)
                return memory.Free >= bytes + DeviceHeadroom(memory.Total);
            return true;
        }

        /// <summary>Whether <paramref name="keep"/> holds <paramref name="slot"/> or a slot over its caches.</summary>
        private static bool Holds(ReadOnlySpan<Slot> keep, Slot slot)
        {
            foreach (Slot k in keep)
                if (k != null && (ReferenceEquals(k, slot) || (slot.K != null && ReferenceEquals(k.K, slot.K))))
                    return true;
            return false;
        }

        /// <summary>The slots requests run on (the bound one's newest fields among them) and <paramref name="also"/>: what
        /// making room never touches.</summary>
        private Slot[] RunningSlots(Slot also = null)
        {
            var running = new List<Slot>();
            if (_slots != null)
                foreach (Slot slot in _slots.Values)
                    if (slot?.K != null) running.Add(slot);
            if (_activeKey != null) running.Add(SnapshotActiveFields());
            if (also != null) running.Add(also);
            return running.ToArray();
        }

        /// <summary>A slot's caches' and states' device bytes.</summary>
        private long SlotStorageBytes(Slot slot)
        {
            long bytes = RecurrentStateBytes(slot);
            foreach (Tensor t in AllCacheTensors(slot.K)) bytes += t.Storage.ByteLength;
            foreach (Tensor t in AllCacheTensors(slot.V)) bytes += t.Storage.ByteLength;
            return bytes;
        }

        /// <summary>The verify buffers the retained slots outside <paramref name="keep"/> kept (<see cref="Retained"/>): what stage 4
        /// of <see cref="EnsureDeviceRoom"/> can free.</summary>
        private long WindowBufferBytes(ReadOnlySpan<Slot> keep)
        {
            if (_retainedSlots is not { Count: > 0 }) return 0;
            long bytes = 0;
            foreach (Slot slot in _retainedSlots.Values)
                if (slot.K != null && slot.Recurrent != null && !Holds(keep, slot))
                    foreach (RecurrentState[] layer in slot.Recurrent)
                        if (layer != null)
                            foreach (RecurrentState state in layer)
                                if (state != null) bytes += state.SpecBufferBytes;
            return bytes;
        }

        /// <summary>Parked recordings that bind none of <paramref name="keep"/>'s caches or states go (their pool blocks back to
        /// the allocator); how many.</summary>
        private int DisposeParkedRecordingsNotBinding(ReadOnlySpan<Slot> keep)
        {
            if (_parkedRecordings.Count == 0) return 0;
            var kept = new HashSet<object>(ReferenceEqualityComparer.Instance);
            foreach (Slot slot in keep)
            {
                if (slot == null) continue;
                object[] bound = BoundStorages(new[] { slot });
                for (int i = 0; i < bound.Length; i += 2) kept.Add(bound[i]);
            }
            var drop = new List<(ulong, ulong)>();
            foreach (var (signature, parked) in _parkedRecordings)
            {
                bool binds = false;
                for (int i = 0; i < parked.Bound.Length && !binds; i += 2) binds = kept.Contains(parked.Bound[i]);
                if (!binds) drop.Add(signature);
            }
            foreach (var signature in drop) DisposeParked(signature);
            return drop.Count;
        }

        /// <summary>A slot's bytes at <paramref name="capacity"/> cells with its margin (<see cref="SlotDeviceBytes"/>); before
        /// any slot was measured, the primary cache's per-cell and state bytes stand for a slot's.</summary>
        private long SlotBytesEstimate(int capacity)
        {
            if (_slotStateBytes >= 0) return SlotDeviceBytes(capacity);
            Slot primary = _activeKey == null ? SnapshotActiveFields() : _primary;
            if (primary?.K == null || primary.Capacity <= 0) return 0;
            long cache = 0;
            foreach (Tensor t in AllCacheTensors(primary.K)) cache += t.Storage.ByteLength;
            foreach (Tensor t in AllCacheTensors(primary.V)) cache += t.Storage.ByteLength;
            return cache / primary.Capacity * capacity + RecurrentStateBytes(primary) + SlotExtraBytes;
        }

        /// <summary>The caches' bytes per cell (the primary's before any slot was measured).</summary>
        private long CacheBytesPerCell()
        {
            if (_slotBytesPerToken >= 0) return _slotBytesPerToken;
            Slot primary = _activeKey == null ? SnapshotActiveFields() : _primary;
            if (primary?.K == null || primary.Capacity <= 0) return 0;
            long cache = 0;
            foreach (Tensor t in AllCacheTensors(primary.K)) cache += t.Storage.ByteLength;
            foreach (Tensor t in AllCacheTensors(primary.V)) cache += t.Storage.ByteLength;
            return cache / primary.Capacity;
        }

        /// <summary>The pool block one attention layer's K cache of <paramref name="capacity"/> cells takes (0 where unknown):
        /// a slot's caches are made of tensors of about this size, and cached blocks of it serve them.</summary>
        private long CacheBlockBytes(int capacity)
        {
            if (_allocator is not TensorSharp.Cuda.CudaAllocator cuda) return 0;
            Tensor[][] k = _activeKey == null ? _cacheK : _primary?.K;
            int cells = _activeKey == null ? _cacheCapacity : _primary?.Capacity ?? 0;
            if (k == null || cells <= 0) return 0;
            foreach (Tensor t in AllCacheTensors(k))
                return cuda.AllocationBlockBytes(t.Storage.ByteLength / cells * capacity);
            return 0;
        }

        // ---------------------------------------------------------------- slots made with room

        /// <summary>
        /// A new request's slot on a device this model manages, made only where it fits: a pooled slot that holds the request,
        /// else room for a fresh one beside the headroom and a prompt's step or its recordings, made from pooled slots, parked
        /// recordings and retained slots' window buffers; short of that, the oldest retained slot that holds the request runs it
        /// (<see cref="ReclaimRetainedSlot"/>: nothing is allocated; the recordings its request parked there went with the
        /// parked recordings making room gave up), then finished conversations are released for a fresh one. Room made is room
        /// had: no reading of the device second-guesses it. A request that still finds no room waits beside running ones
        /// (<see cref="SequenceSlotUnavailableException"/>); alone, it runs where its slot fits beside the headroom and the
        /// narrowest prompt step, and is refused with the numbers where it does not, before any of its prompt runs (it had
        /// failed part way with CUDA error 2).
        /// </summary>
        /// <param name="required">The cells the request's first step needs (its prompt, its first token and the verify window);
        /// where slots are sized to the device's room (<see cref="SizesSlotsToRoom"/>) the slot is <see cref="SlotTarget"/>'s:
        /// the class of the reservation with headroom where memory allows, else the largest that fits, and only for the required
        /// cells are finished conversations released.</param>
        private Slot NewSlotWithRoom(string requestId, int reserve, int required)
        {
            int capacity = NewSlotCapacity(reserve);
            if (_slotPool?.Exists(p => p.Capacity >= capacity) == true)
                return ReclaimRetainedSlot(reserve) ?? NewSlot(reserve);
            Slot[] keep = RunningSlots();
            if (SizesSlotsToRoom)
            {
                int least = NewSlotCapacity(required);
                int roomy = RoomySlotTarget(least, reserve, keep, "a new request's slot", 0);
                if (roomy > 0) return Logged(FreshSlot(Math.Max(roomy, least)));
                if (_slotPool?.Exists(p => p.Capacity >= least) == true) return Logged(NewSlot(least));
                if (ReclaimRetainedSlot(least, roomShort: true) is { } taken) return Logged(taken);
                int tight = TightSlotTarget(least, keep, "a new request's slot", 0);
                if (tight > 0) return Logged(FreshSlot(Math.Max(tight, least)));
                capacity = least;

                Slot Logged(Slot slot)
                {
                    (_allocator as TensorSharp.Cuda.CudaAllocator)?.LogDeviceRoom(string.Create(CultureInfo.InvariantCulture,
                        $"{Config.Architecture} {requestId}'s slot: {slot.Capacity:N0} cells (reserved {reserve:N0}, required {required:N0})"));
                    return slot;
                }
            }
            long bytes = SlotBytesEstimate(capacity), block = CacheBlockBytes(capacity);
            // Room made is room had: a retained slot is taken over only where none could be made (ReclaimRetainedSlot's own
            // check reads the device, which leaves out the window buffers stage 4 freed, and took the oldest conversation over
            // for nothing where they had made the room).
            if (EnsureDeviceRoom(bytes, keep, RoomUse.Slot, mayEvict: false, "a new request's slot", block))
                return NewSlot(reserve);
            if (ReclaimRetainedSlot(reserve, roomShort: true) is { } reclaimed)
                return reclaimed;
            if (EnsureDeviceRoom(bytes, keep, RoomUse.Slot, mayEvict: true, "a new request's slot", block))
                return NewSlot(reserve);
            if (keep.Length > 0)
            {
                SlotWaits++;
                var (free, total) = DeviceMemory(fresh: true);
                throw new SequenceSlotUnavailableException(string.Create(CultureInfo.InvariantCulture,
                    $"{Config.Architecture}: no device memory for another request's slot ({capacity:N0} cells, {Mib(bytes)} MiB, " +
                    $"{Mib(RoomNeed(bytes, total, RoomUse.Slot) - free)} MiB short beside the headroom and a prompt step's or recordings' room); " +
                    $"{requestId} waits for one of the {keep.Length} running requests to finish."));
            }
            if (FitsBesideHeadroom(bytes))
                return SizesSlotsToRoom ? FreshSlot(capacity) : NewSlot(reserve);
            SlotRefusals++;
            throw SlotRefusal(requestId, capacity, bytes);
        }

        /// <summary>The cells of a new request's slot for a reservation of <paramref name="reserve"/> tokens.</summary>
        private int NewSlotCapacity(int reserve)
            => AlignedSlotCapacity(Math.Max(Math.Max(1, reserve), Math.Min(FreshSlotTokens, Math.Max(1, _maxContextLength))));

        /// <summary>Tests: the free bytes a new request's slot for <paramref name="reserve"/> tokens wants (<see cref="RoomNeed"/>),
        /// against the device's total as the model reads it.</summary>
        internal long NewSlotRoomNeedForTests(int reserve)
            => RoomNeed(SlotBytesEstimate(NewSlotCapacity(reserve)), DeviceMemory(fresh: true).Total, RoomUse.Slot);

        /// <summary>Tests: the bytes of a lone prompt's widest step on direct CUDA (its rows' activations and scratch).</summary>
        internal long PromptStepBytesForTests => PromptStepBytes;

        /// <summary>The refusal of a request whose slot does not fit the device alone, with the numbers.</summary>
        private InvalidOperationException SlotRefusal(string requestId, int capacity, long bytes)
        {
            var (free, total) = DeviceMemory(fresh: true);
            long kv = CacheBytesPerCell() * capacity, states = Math.Max(0, bytes - kv - SlotExtraBytes);
            long headroom = DeviceHeadroom(total), room = Math.Max(0, free - headroom - NarrowestPromptStepBytes);
            return new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
                $"{Config.Architecture}: {requestId} needs a slot of {capacity:N0} cells (its prompt, its generation budget and the " +
                $"verify window): {kv / 1e9:F2} GB of K/V and {states / 1e9:F2} GB of states, and the device holds {room / 1e9:F2} GB " +
                $"beside the weights, its {headroom / 1e9:F2} GB headroom and a prompt step, with no finished conversation left to " +
                $"release. A shorter prompt or a smaller max_tokens fits; the request was refused before any of it ran."));
        }

        /// <summary>
        /// Room for <paramref name="slot"/>'s caches grown to <paramref name="capacity"/> cells (the old ones live until their rows
        /// are copied), finished conversations released for it where needed. A growth that still does not fit beside the
        /// headroom and the narrowest prompt step fails with what holds the device instead of running out of memory part way.
        /// </summary>
        private void EnsureRoomToGrow(Slot slot, int capacity)
        {
            long bytes = CacheBytesPerCell() * capacity;
            Slot[] keep = RunningSlots(slot);
            if (EnsureDeviceRoom(bytes, keep, RoomUse.Slot, mayEvict: true, "a slot's growth", CacheBlockBytes(capacity))
                || FitsBesideHeadroom(bytes))
                return;
            throw GrowthExhausted(slot, capacity, keep, bytes);
        }

        /// <summary>A running request's growth that found no room even with every finished conversation released: what holds
        /// the device. <paramref name="bytes"/>: what the growth needs (default: the added cells and one tensor, as a slot
        /// grown a tensor at a time does, <see cref="GrowSlotRolling"/>).</summary>
        private InvalidOperationException GrowthExhausted(Slot slot, int capacity, Slot[] keep, long bytes = -1)
        {
            if (bytes < 0) bytes = SlotTargetBytes(capacity, slot.Capacity);
            long held = 0;
            var seen = new HashSet<Tensor[][]>(ReferenceEqualityComparer.Instance);
            foreach (Slot running in keep)
                if (running?.K != null && seen.Add(running.K)) held += SlotStorageBytes(running);
            var (free, total) = DeviceMemory(fresh: true);
            return new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
                $"{Config.Architecture}: device memory exhausted by running requests: {seen.Count} slots hold {held / 1e9:F2} GB, and " +
                $"growing a slot from {slot.Capacity:N0} to {capacity:N0} cells needs {bytes / 1e9:F2} GB beside the " +
                $"{DeviceHeadroom(total) / 1e9:F2} GB headroom and a prompt step, where {Math.Max(0, free) / 1e9:F2} GB are free."));
        }

        /// <summary>
        /// Room for a copy of <paramref name="source"/> sized for <paramref name="capacityTokens"/> (<see cref="CopySlot"/>): a
        /// pooled slot that holds it, or a fresh one's room made as for a new slot (conversations released only with
        /// <paramref name="mayEvict"/>; never the source). A <paramref name="needed"/> copy (a request adopts it) is made wherever
        /// it fits beside the headroom and the narrowest prompt step; an optional one (a checkpoint) only with
        /// the whole room.
        /// </summary>
        private bool RoomForSlotCopy(Slot source, int capacityTokens, bool mayEvict, bool needed, string what)
            => RoomForSlotCopy(source, capacityTokens, mayEvict, needed, what, out _);

        /// <summary>Room for a copy as above, and the cells to make it with (<see cref="CopySlot"/>): where slots are sized to the
        /// device's room (<see cref="SizesSlotsToRoom"/>) a <paramref name="needed"/> copy - a request runs on it - is
        /// <see cref="SlotTarget"/>'s, with headroom where memory allows; a checkpoint stays as long as its rows (nothing runs on it
        /// yet, and the request it was taken from runs beside it).</summary>
        private bool RoomForSlotCopy(Slot source, int capacityTokens, bool mayEvict, bool needed, string what, out int copyCapacity)
        {
            copyCapacity = capacityTokens;
            if (!DeviceRoomManaged) return true;
            int capacity = AlignedSlotCapacity(Math.Max(FreshSlotTokens, capacityTokens));
            if (_slotPool?.Exists(p => p.Capacity >= capacity) == true) return true;
            long bytes = SlotBytesEstimate(capacity);
            if (SizesSlotsToRoom && needed)
            {
                int target = SlotTarget(capacity, capacity, RunningSlots(source), what, mayEvict: mayEvict);
                if (target > 0)
                {
                    copyCapacity = target;
                    return true;
                }
                return FitsBesideHeadroom(bytes);
            }
            return EnsureDeviceRoom(bytes, RunningSlots(source), RoomUse.Slot, mayEvict, what, CacheBlockBytes(capacity))
                || (needed && FitsBesideHeadroom(bytes));
        }

        // ---------------------------------------------------------------- the allocator's reclaimer

        /// <summary>Slots are made, copied, grown or freed: the allocator's reclaimer gives up nothing meanwhile (what it would
        /// release may be what the section reads, and the section made its own room).</summary>
        private int _slotMutationDepth;

        // The thread the model's steps run on (the latest step entry's: EnsureSequenceSlot, TryForwardSlots): the only one the
        // reclaimer acts on. Not the first one's: a host warms the kernels on its loading thread (WarmPooledSlot makes a slot
        // there), and the engine's own worker runs every later step, where a reclaimer bound to the loader never acted.
        private int _engineThread;

        /// <summary>The calling thread runs the model's steps from here on (<see cref="ReclaimForAllocator"/> acts only there).</summary>
        private void NoteEngineThread() => _engineThread = Environment.CurrentManagedThreadId;

        private bool _reclaiming;
        private DeviceRoomReclaimer _reclaimer;

        // Slots the reclaimer freed: their steps and recordings are forgotten at the model's next entry.
        private List<(Tensor[][] K, RecurrentState[][] Recurrent, object[] Bound)> _deferredForgets;

        /// <summary>Tests stand in for whether a CUDA graph is being captured on the allocator.</summary>
        internal Func<bool> GraphCaptureForTests { get; set; }

        /// <summary>Tests: the reclaimer's view while slots are being changed.</summary>
        internal IDisposable MutatingSlotsForTests()
        {
            _slotMutationDepth++;
            return new SlotChangeScope(() => _slotMutationDepth--);
        }

        private sealed class SlotChangeScope : IDisposable
        {
            private Action _end;
            internal SlotChangeScope(Action end) => _end = end;
            public void Dispose()
            {
                _end?.Invoke();
                _end = null;
            }
        }

        /// <summary>What the CUDA allocator asks before its last resort (a full GC and a drain of every cached block) when an
        /// allocation no site sized finds the device full: step activations, recordings, verify buffers, drafter rings.</summary>
        private sealed class DeviceRoomReclaimer : TensorSharp.Cuda.IDeviceMemoryReclaimer
        {
            private readonly StandardDecoderModel _model;
            internal DeviceRoomReclaimer(StandardDecoderModel model) => _model = model;
            public long Reclaim(long bytes) => _model.ReclaimForAllocator(bytes);
        }

        /// <summary>Register the model's reclaimer with its CUDA allocator (once attached to a prefix cache, which keeps the
        /// conversations it releases).</summary>
        private void RegisterDeviceRoomReclaimer()
        {
            if (DeviceRoomForTests != null || !DeviceRoomManaged || _allocator is not TensorSharp.Cuda.CudaAllocator cuda) return;
            _reclaimer ??= new DeviceRoomReclaimer(this);
            cuda.RegisterReclaimer(_reclaimer);
        }

        private void UnregisterDeviceRoomReclaimer()
        {
            if (_reclaimer != null && _allocator is TensorSharp.Cuda.CudaAllocator cuda) cuda.UnregisterReclaimer(_reclaimer);
        }

        private bool GraphCapturing() => GraphCaptureForTests?.Invoke()
            ?? (_allocator is TensorSharp.Cuda.CudaAllocator cuda && cuda.CapturesGraph);

        /// <summary>
        /// The allocator's rung 3: release towards <paramref name="bytes"/> what no running request uses - pooled slots, the
        /// largest first, the window buffers retained slots keep, then the conversations the prefix cache keeps, the oldest first
        /// (each reported to it as reclaimed) -
        /// and return the bytes released. Only on the engine's thread, never while a graph is captured, never re-entered and
        /// never while slots are being changed (0 then): the slots' caches go at once, the steps and recordings that bind them
        /// at the model's next entry (<see cref="ForgetDeferredGraphs"/>), since an allocation may run inside a step that walks
        /// them.
        /// </summary>
        internal long ReclaimForAllocator(long bytes)
        {
            if (bytes <= 0 || _engineThread == 0 || Environment.CurrentManagedThreadId != _engineThread
                || _reclaiming || _slotMutationDepth > 0 || GraphCapturing())
                return 0;
            _reclaiming = true;
            try
            {
                Slot[] keep = RunningSlots();
                long released = 0;
                bool pooled = false;
                while (released < bytes && _slotPool is { Count: > 0 })
                {
                    Slot largest = _slotPool[0];
                    foreach (Slot slot in _slotPool)
                        if (slot.Capacity > largest.Capacity) largest = slot;
                    _slotPool.Remove(largest);
                    released += DisposeSlotLater(largest);
                    pooled = true;
                }
                if (pooled) DeviceRoomStages[2]++;
                // The window buffers retained slots keep, where they cover what is asked: no step binds them (a retained slot's
                // kept steps went when it was retained), and a parked recording over them is adopted only by their identity, which
                // no new buffer has, or moved with every state pointer patched (TakeRelocated).
                if (released < bytes && WindowBufferBytes(keep) >= bytes - released)
                {
                    long buffers = 0;
                    foreach (Slot slot in _retainedSlots.Values.OrderBy(s => s.RetainedAt).ToList())
                    {
                        if (released + buffers >= bytes) break;
                        if (slot.K != null && !Holds(keep, slot)) buffers += ReleaseWindowBuffers(slot);
                    }
                    if (buffers > 0)
                    {
                        released += buffers;
                        DeviceRoomStages[4]++;
                        NoteDeviceMemoryChanged();
                    }
                }
                int conversations = 0;
                long conversationBytes = 0;
                if (released < bytes && _retainedSlots is { Count: > 0 })
                {
                    foreach (var (key, slot) in _retainedSlots.OrderBy(kv => kv.Value.RetainedAt).ToList())
                    {
                        if (released >= bytes) break;
                        if (slot.K == null || Holds(keep, slot)) continue;
                        _retainedSlots.Remove(key);
                        long b = DisposeSlotLater(slot);
                        released += b;
                        conversationBytes += b;
                        conversations++;
                        _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
                    }
                }
                if (conversations > 0)
                {
                    DeviceRoomStages[5]++;
                    Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"[{Config.Architecture}] device memory ran out for a {Mib(bytes)} MiB allocation: released {conversations} retained " +
                        $"conversation{(conversations == 1 ? "" : "s")} ({Mib(conversationBytes)} MiB; their next turns prefill again)."));
                }
                return released;
            }
            finally
            {
                _reclaiming = false;
            }
        }

        /// <summary><see cref="DisposeSlot"/> for the reclaimer: the caches and states go now, the steps and recordings that
        /// bind them at the model's next entry. The bytes freed.</summary>
        private long DisposeSlotLater(Slot slot)
        {
            long bytes = SlotStorageBytes(slot);
            (_deferredForgets ??= new()).Add((slot.K, slot.Recurrent, BoundStorages(new[] { slot })));
            _idleSpareRefused = false;
            foreach (Tensor t in AllCacheTensors(slot.K)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
            foreach (Tensor t in AllCacheTensors(slot.V)) { InvalidateTensorDeviceCache(t); t.Dispose(); }
            slot.K = slot.V = null;
            DisposeStates(slot.Recurrent);
            slot.Recurrent = null;
            slot.DraftCarry?.Dispose();
            slot.DraftCarry = null;
            NoteDeviceMemoryChanged();
            return bytes;
        }

        /// <summary>The steps and recordings over slots the reclaimer freed go (<see cref="ForgetGraphsOf"/>), at an entry where
        /// nothing walks them.</summary>
        private void ForgetDeferredGraphs()
        {
            if (_deferredForgets is not { Count: > 0 } due) return;
            _deferredForgets = null;
            if (!GraphsKeyedByIdentity)
            {
                ClearGraphCache();
                return;
            }
            foreach (var (k, recurrent, bound) in due)
            {
                DropReplaysOf(k, recurrent, park: false);
                DisposeParkedBinding(bound);
            }
            ForgetProgramsKeyedOtherwise();
        }
    }
}
