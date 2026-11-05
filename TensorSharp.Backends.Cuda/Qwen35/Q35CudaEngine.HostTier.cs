// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Host copies of Qwen3.5 / Qwen3.8 direct-CUDA engine slots.
//
// A slot that device memory can no longer hold (a retained prompt checkpoint or finished
// conversation) moves to pinned host memory and comes back when a request reuses it: copying a
// 17K-token checkpoint back costs about 60 ms, computing it again seconds. Its attention blocks are
// shared by content: every device block carries a key that changes whenever rows are written into
// it, a host copy is made once per key however many host slots hold it, and a slot coming back
// shares the device block that still carries the key instead of uploading the copy.
//
// Host memory comes in units of one attention block; a slot's recurrent states and drafter ring
// are packed into units too. Pinning is slow (about 2.5 GB/s), so a background thread keeps a
// reserve of pinned units ahead of the copies.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        /// <summary>A slot held in pinned host memory: its position, recurrent states, drafter ring
        /// and attention blocks (shared by content with other host slots).</summary>
        public sealed class HostSlot
        {
            public int NPast { get; internal set; }
            /// <summary>The slot's M-RoPE delta (see <see cref="Slot.RopeDelta"/>).</summary>
            public int RopeDelta { get; internal set; }
            internal HostBlock[] Blocks;
            internal IntPtr[] StateUnits;
            internal object Owner;
            internal string StateNamespace, StateProfile;
            internal int StatePartCount;
            internal NinferDraftSlotSnapshot NinferDraft;
            internal long DraftMetadataBytes;
        }

        internal sealed class HostBlock
        {
            internal IntPtr Data;
            internal long Key;
            internal int Refs;
        }

        // Content keys of live device blocks, and the device block holding each key.
        private readonly Dictionary<IntPtr, long> _blockKey = new Dictionary<IntPtr, long>();
        private readonly Dictionary<long, IntPtr> _deviceByKey = new Dictionary<long, IntPtr>();
        private long _nextBlockKey = 1;
        private readonly Dictionary<long, HostBlock> _hostBlocks = new Dictionary<long, HostBlock>();
        private readonly object _hostSlotOwner = new object();

        // Pinned units (one block's bytes), and the pinner that tops them up. Guarded by _hostLock.
        private readonly object _hostLock = new object();
        private readonly Stack<IntPtr> _freeUnits = new Stack<IntPtr>();
        private readonly List<IntPtr> _hostChunks = new List<IntPtr>();
        private long _pinnedBytes, _hostLimitBytes, _hostReserveBytes;
        private Thread _pinner;
        private bool _pinnerStop;
        /// <summary>Units one pinned allocation carves.</summary>
        private const int UnitsPerChunk = 16;

        // Where each recurrent-state part of a slot lands (unit, offset), and the units one slot's state takes.
        private (int Unit, long Offset)[] _stateLayout;
        private int _stateUnits;
        private string _stateLayoutProfile;

        /// <summary>Host bytes held by slots: pinned units plus owned original-draft
        /// snapshot metadata (units kept for reuse excluded).</summary>
        public long HostBytesInUse { get; private set; }
        /// <summary>Owned native draft metadata charged within HostBytesInUse:
        /// 64 bytes per capsule plus its optional 51,200-byte raw BF16 anchor.</summary>
        public long HostDraftMetadataBytesInUse { get; private set; }
        /// <summary>Milliseconds the engine thread spent pinning and copying to host (diagnostics).</summary>
        public double HostPinMs { get; private set; }
        public double HostCopyMs { get; private set; }

        /// <summary>Host bytes a slot's recurrent states and drafter ring take.</summary>
        public long HostStateBytes => StateLayout() * KvBlockBytes;

        /// <summary>Lets the background pinner keep <paramref name="reserveBytes"/> of pinned units
        /// free, pinning at most <paramref name="limitBytes"/> in all.</summary>
        public void ConfigureHostPool(long limitBytes, long reserveBytes)
        {
            lock (_hostLock)
            {
                _hostLimitBytes = Math.Max(0, limitBytes);
                _hostReserveBytes = Math.Max(0, Math.Min(reserveBytes, _hostLimitBytes));
                if (_hostLimitBytes > 0 && _pinner == null)
                {
                    _pinner = new Thread(PinLoop) { IsBackground = true, Name = "q35-host-pinner" };
                    _pinner.Start();
                }
                Monitor.PulseAll(_hostLock);
            }
        }

        private void PinLoop()
        {
            _alloc.Context.MakeCurrent();
            while (true)
            {
                lock (_hostLock)
                {
                    while (!_pinnerStop && (_freeUnits.Count * KvBlockBytes >= _hostReserveBytes
                        || _pinnedBytes + UnitsPerChunk * KvBlockBytes > _hostLimitBytes))
                        Monitor.Wait(_hostLock);
                    if (_pinnerStop)
                        return;
                }
                if (CudaDriverApi.cuMemHostAlloc(out IntPtr chunk, new UIntPtr((ulong)(KvBlockBytes * UnitsPerChunk)), 0) != 0)
                {
                    lock (_hostLock)
                        _hostLimitBytes = _pinnedBytes;   // the system refused: pin no more
                    continue;
                }
                lock (_hostLock)
                    AddChunk(chunk);
            }
        }

        private void AddChunk(IntPtr chunk)
        {
            _hostChunks.Add(chunk);
            _pinnedBytes += KvBlockBytes * UnitsPerChunk;
            for (int i = UnitsPerChunk - 1; i >= 0; i--)
                _freeUnits.Push(chunk + (nint)(i * KvBlockBytes));
        }

        private IntPtr TakeUnit()
        {
            lock (_hostLock)
            {
                if (_freeUnits.Count == 0)
                {
                    // The pinner fell behind: pin here.
                    long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    CudaDriverApi.cuMemHostAlloc(out IntPtr chunk, new UIntPtr((ulong)(KvBlockBytes * UnitsPerChunk)), 0).ThrowOnError();
                    HostPinMs += System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    AddChunk(chunk);
                }
                IntPtr u = _freeUnits.Pop();
                if (_freeUnits.Count * KvBlockBytes < _hostReserveBytes)
                    Monitor.PulseAll(_hostLock);
                return u;
            }
        }

        private void ReturnUnit(IntPtr u)
        {
            lock (_hostLock)
                _freeUnits.Push(u);
        }

        private void SetBlockKey(IntPtr b, long key)
        {
            if (_blockKey.TryGetValue(b, out long old) && _deviceByKey.TryGetValue(old, out IntPtr holder) && holder == b)
                _deviceByKey.Remove(old);
            _blockKey[b] = key;
            _deviceByKey[key] = b;
        }

        private void DropBlockKey(IntPtr b)
        {
            if (_blockKey.Remove(b, out long key) && _deviceByKey.TryGetValue(key, out IntPtr holder) && holder == b)
                _deviceByKey.Remove(key);
        }

        /// <summary>Rows are about to be written into <paramref name="b"/>: host copies of its old
        /// content no longer describe it.</summary>
        private void RekeyBlock(IntPtr b) => SetBlockKey(b, _nextBlockKey++);

        /// <summary>Recurrent states and drafter ring of a slot, in the order a host slot stores them.</summary>
        private IEnumerable<(IntPtr Device, long Bytes)> StateParts(Slot s)
        {
            if (s == null || s.Arena == IntPtr.Zero || s.Conv == null || s.State == null
                || s.Conv.Length != _nGdn || s.State.Length != _nGdn)
                throw new InvalidOperationException("Incomplete native slot recurrent-state layout.");
            for (int i = 0; i < _nGdn; i++)
            {
                if (s.Conv[i] == IntPtr.Zero || s.State[i] == IntPtr.Zero)
                    throw new InvalidOperationException("Missing native recurrent-state allocation.");
                yield return (s.Conv[i], ConvBytes);
                yield return (s.State[i], StateBytes);
            }
            if (HasDrafter || HasNinferDrafter)
            {
                if (RingLayers < 1 || s.RingK == null || s.RingV == null || s.RingK.Length != RingLayers
                    || s.RingV.Length != RingLayers || s.RingBase == IntPtr.Zero || s.RingPos == IntPtr.Zero)
                    throw new InvalidOperationException("The slot lacks its complete attached-drafter ring layout.");
                for (int l = 0; l < RingLayers; l++)
                {
                    if (s.RingK[l] == IntPtr.Zero || s.RingV[l] == IntPtr.Zero)
                        throw new InvalidOperationException("Missing attached-drafter K/V allocation.");
                    yield return (s.RingK[l], RingLayerBytes);
                    yield return (s.RingV[l], RingLayerBytes);
                }
                // Native DFlash uses the typed frontier, not these compatibility
                // tags, but the common arena owns them and copies them as bytes.
                yield return (s.RingPos, 4L * RingRows);
            }
        }

        private string HostStateNamespace => KVStateNamespaceSuffix + "|host-native-draft=" + NinferDraftNamespaceSuffix;
        private string HostStateProfile => $"host-slot-v2:{_nGdn}:{ConvBytes}:{StateBytes}:{KvBlockBytes}:{HasDrafter || HasNinferDrafter}:{RingLayers}:{RingRows}:{RingLayerBytes}";
        private static long HostDraftMetadataCharge(NinferDraftSlotSnapshot snapshot) => snapshot == null ? 0 : checked(64L + snapshot.Pending.Length);

        /// <summary>Attachment must not change the layout of existing host slots.</summary>
        private void InvalidateHostStateLayout()
        {
            if (HostBytesInUse != 0) throw new InvalidOperationException("Cannot change native state layout while host snapshots are retained.");
            _stateLayout = null; _stateUnits = 0; _stateLayoutProfile = null;
        }

        /// <summary>Packs the state parts into units in order; returns the units a slot's state takes.</summary>
        private int StateLayout()
        {
            string profile = HostStateProfile;
            if (_stateLayout != null && _stateLayoutProfile != profile) InvalidateHostStateLayout();
            if (_stateLayout != null)
                return _stateUnits;
            var layout = new List<(int, long)>();
            int unit = 0;
            long at = 0;
            foreach (var (_, bytes) in StateParts(_active))
            {
                if (bytes > KvBlockBytes)
                    throw new NotSupportedException($"[q35-cuda] a {bytes}-byte state part exceeds a host unit");
                if (at + bytes > KvBlockBytes)
                {
                    unit++;
                    at = 0;
                }
                layout.Add((unit, at));
                at += bytes;
            }
            _stateLayout = layout.ToArray();
            _stateUnits = unit + 1;
            _stateLayoutProfile = profile;
            return _stateUnits;
        }

        /// <summary>Pure preflight, deliberately called before destination blocks,
        /// metadata or device allocations are changed by a host restore.</summary>
        internal static void ValidateHostSlotRestoreMetadata(HostSlot host, object owner, string stateNamespace, string profile,
            int maximumContext, int units, int parts, bool nativeDraft, string draftNamespace)
        {
            ArgumentNullException.ThrowIfNull(host);
            if (owner == null || maximumContext < 0 || units < 1 || parts < 0 || string.IsNullOrEmpty(profile)
                || !ReferenceEquals(host.Owner, owner) || string.IsNullOrEmpty(stateNamespace)
                || host.StateNamespace != stateNamespace || host.StateProfile != profile || host.NPast < 0 || host.NPast > maximumContext
                || host.StateUnits == null || host.StateUnits.Length != units || host.StatePartCount != parts || host.Blocks == null
                || host.Blocks.Length != (host.NPast + Q35Kernels.KvBlockTokens - 1L) / Q35Kernels.KvBlockTokens)
                throw new InvalidOperationException("Host slot owner, numerical namespace, position or state geometry differs from this engine.");
            foreach (IntPtr unit in host.StateUnits) if (unit == IntPtr.Zero) throw new InvalidOperationException("Incomplete host state units.");
            foreach (HostBlock block in host.Blocks)
                if (block == null || block.Data == IntPtr.Zero || block.Refs < 1)
                    throw new InvalidOperationException("Incomplete owned host attention blocks.");
            if (nativeDraft)
            {
                if (host.RopeDelta != 0) throw new InvalidOperationException("Original native draft host restore requires text positions.");
                ValidateNinferDraftSnapshot(host.NinferDraft, host.NPast, draftNamespace);
            }
            else if (host.NinferDraft != null) throw new InvalidOperationException("Host snapshot carries an original drafter that this engine does not have.");
            if (host.DraftMetadataBytes != HostDraftMetadataCharge(host.NinferDraft))
                throw new InvalidOperationException("Host draft metadata byte ownership differs.");
        }

        /// <summary>A host copy of <paramref name="s"/> (the slot itself is unchanged; free it after).</summary>
        public HostSlot CopyToHost(Slot s)
        {
            MakeCurrent();
            Sync();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            double pinBefore = HostPinMs;
            HostSlot h = IssueCopyToHost(s, Stream);
            Sync();
            HostCopyMs += System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds - (HostPinMs - pinBefore);
            return h;
        }

        private IntPtr _copyStream;
        private readonly Stack<IntPtr> _freeEvents = new Stack<IntPtr>();

        /// <summary>Starts a host copy of <paramref name="s"/> on a copy stream, behind every kernel
        /// issued so far: the slot must not change until <paramref name="done"/> completes
        /// (<see cref="CopyDone"/>, <see cref="WaitCopy"/>), and the host slot is valid only then.</summary>
        public HostSlot BeginCopyToHost(Slot s, out IntPtr done)
        {
            MakeCurrent();
            if (_copyStream == IntPtr.Zero)
                CudaDriverApi.cuStreamCreate(out _copyStream, 1 /* CU_STREAM_NON_BLOCKING */).ThrowOnError();
            IntPtr ready = TakeEvent();
            CudaDriverApi.cuEventRecord(ready, Stream).ThrowOnError();
            CudaDriverApi.cuStreamWaitEvent(_copyStream, ready, 0).ThrowOnError();
            _freeEvents.Push(ready);   // reusable once recorded again: the wait already captured it
            HostSlot h = IssueCopyToHost(s, _copyStream);
            done = TakeEvent();
            CudaDriverApi.cuEventRecord(done, _copyStream).ThrowOnError();
            return h;
        }

        /// <summary>True once the copy <paramref name="done"/> marks has finished (the event is then released).</summary>
        public bool CopyDone(IntPtr done)
        {
            int r = CudaDriverApi.cuEventQuery(done);
            if (r == 600 /* CUDA_ERROR_NOT_READY */)
                return false;
            r.ThrowOnError();
            _freeEvents.Push(done);
            return true;
        }

        /// <summary>Waits for the copy <paramref name="done"/> marks (the event is then released).</summary>
        public void WaitCopy(IntPtr done)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            CudaDriverApi.cuEventSynchronize(done).ThrowOnError();
            HostCopyMs += System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            _freeEvents.Push(done);
        }

        private IntPtr TakeEvent()
        {
            if (_freeEvents.Count > 0)
                return _freeEvents.Pop();
            CudaDriverApi.cuEventCreate(out IntPtr e, 2 /* CU_EVENT_DISABLE_TIMING */).ThrowOnError();
            return e;
        }

        private HostSlot IssueCopyToHost(Slot s, IntPtr stream)
        {
            // Capture immutable metadata before starting any copy. A native
            // ring without its frontier/pending anchor is never a valid snapshot.
            var snapshot = CaptureNinferDraftSlotState(s);
            var parts = new List<(IntPtr Device, long Bytes)>(StateParts(s));
            int units = StateLayout();
            if (parts.Count != _stateLayout.Length || s.NPast < 0 || s.NPast > _m.NCtx)
                throw new InvalidOperationException("Native host capture state profile differs from its current layout.");
            var h = new HostSlot { NPast = s.NPast, RopeDelta = s.RopeDelta, Owner = _hostSlotOwner,
                StateNamespace = HostStateNamespace, StateProfile = HostStateProfile, StatePartCount = parts.Count, NinferDraft = snapshot };
            int nb = (s.NPast + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens;
            if (s.Blocks.Count < nb) throw new InvalidOperationException("The source lacks its committed attention blocks.");
            h.Blocks = new HostBlock[nb];
            h.StateUnits = new IntPtr[units];
            try
            {
                for (int i = 0; i < nb; i++)
                {
                    IntPtr b = s.Blocks[i];
                    if (!_blockKey.TryGetValue(b, out long key))
                    {
                        key = _nextBlockKey++;
                        SetBlockKey(b, key);
                    }
                    bool copy = !_hostBlocks.TryGetValue(key, out HostBlock hb);
                    if (copy)
                    {
                        hb = new HostBlock { Key = key, Data = TakeUnit() };
                        _hostBlocks[key] = hb;
                        HostBytesInUse += KvBlockBytes;
                    }
                    hb.Refs++; h.Blocks[i] = hb;
                    if (copy) CudaDriverApi.cuMemcpyDtoHAsync(hb.Data, b, new UIntPtr((ulong)KvBlockBytes), stream).ThrowOnError();
                }
                for (int u = 0; u < h.StateUnits.Length; u++)
                {
                    h.StateUnits[u] = TakeUnit();
                    HostBytesInUse += KvBlockBytes;
                }
                for (int p = 0; p < parts.Count; p++)
                {
                    var (dev, bytes) = parts[p]; var (unit, offset) = _stateLayout[p];
                    CudaDriverApi.cuMemcpyDtoHAsync(h.StateUnits[unit] + (nint)offset, dev, new UIntPtr((ulong)bytes), stream).ThrowOnError();
                }
                h.DraftMetadataBytes = HostDraftMetadataCharge(snapshot);
                HostDraftMetadataBytesInUse += h.DraftMetadataBytes; HostBytesInUse += h.DraftMetadataBytes;
                return h;
            }
            catch
            {
                // Do not return a unit to the pool while an earlier enqueued
                // transfer can still write it, even if a later transfer failed.
                CudaDriverApi.cuStreamSynchronize(stream);
                ReleaseHostSlotUnits(h); throw;
            }
        }

        /// <summary>Makes <paramref name="dst"/> the slot <paramref name="h"/> holds: blocks whose
        /// content is still on the device are shared, the others uploaded; states and ring uploaded.</summary>
        public void CopyFromHost(HostSlot h, Slot dst)
        {
            ArgumentNullException.ThrowIfNull(dst);
            int units = StateLayout();
            ValidateHostSlotRestoreMetadata(h, _hostSlotOwner, HostStateNamespace, HostStateProfile, _m.NCtx,
                units, _stateLayout.Length, HasNinferDrafter, _ninferDraftNamespace);
            var parts = new List<(IntPtr Device, long Bytes)>(StateParts(dst));
            if (parts.Count != _stateLayout.Length) throw new InvalidOperationException("Destination native state geometry differs.");
            foreach (HostBlock block in h.Blocks)
                if (!_hostBlocks.TryGetValue(block.Key, out HostBlock owned) || !ReferenceEquals(owned, block))
                    throw new InvalidOperationException("A host block no longer belongs to the live host pool.");
            // All owner/namespace/position/frontier/layout validation above is
            // complete before destination blocks or any device bytes can change.
            MakeCurrent();
            // Host blocks another slot's copy is still filling must land before they are read.
            if (_copyStream != IntPtr.Zero)
                CudaDriverApi.cuStreamSynchronize(_copyStream).ThrowOnError();
            ForgetNinferVerification(dst);
            ReleaseBlocks(dst);
            foreach (HostBlock hb in h.Blocks)
            {
                if (_deviceByKey.TryGetValue(hb.Key, out IntPtr b))
                    _blockRefs[b]++;
                else
                {
                    b = AllocBlock();
                    SetBlockKey(b, hb.Key);
                    CudaDriverApi.cuMemcpyHtoDAsync(b, hb.Data, new UIntPtr((ulong)KvBlockBytes), Stream).ThrowOnError();
                }
                dst.Blocks.Add(b);
            }
            dst.Version++;
            for (int p = 0; p < parts.Count; p++)
            {
                var (dev, bytes) = parts[p]; var (unit, offset) = _stateLayout[p];
                CudaDriverApi.cuMemcpyHtoDAsync(dev, h.StateUnits[unit] + (nint)offset, new UIntPtr((ulong)bytes), Stream).ThrowOnError();
            }
            dst.NPast = h.NPast;
            dst.RopeDelta = h.RopeDelta;
            Sync();
            RestoreNinferDraftSlotState(dst, h.NinferDraft);
        }

        /// <summary>Releases a host slot; returns the pinned bytes that freed.</summary>
        public long FreeHostSlot(HostSlot h)
        {
            if (h == null || h.StateUnits == null)
                return 0;
            if (!ReferenceEquals(h.Owner, _hostSlotOwner)) throw new InvalidOperationException("Cannot release another engine's host slot.");
            return ReleaseHostSlotUnits(h);
        }

        private long ReleaseHostSlotUnits(HostSlot h)
        {
            long freed = 0;
            foreach (HostBlock hb in h.Blocks)
            {
                if (hb == null) continue; // A failed capture may own a prefix only.
                if (--hb.Refs > 0)
                    continue;
                _hostBlocks.Remove(hb.Key);
                ReturnUnit(hb.Data);
                freed += KvBlockBytes;
            }
            foreach (IntPtr u in h.StateUnits)
                if (u != IntPtr.Zero) { ReturnUnit(u); freed += KvBlockBytes; }
            freed += h.DraftMetadataBytes; HostDraftMetadataBytesInUse -= h.DraftMetadataBytes;
            h.DraftMetadataBytes = 0; h.NinferDraft = null;
            h.StateUnits = null;
            h.Blocks = Array.Empty<HostBlock>();
            HostBytesInUse -= freed;
            return freed;
        }

        /// <summary>The content key of the last full block below <paramref name="tokens"/> (0 when there is none):
        /// two slots with the same key there share every row before it, so a longer slot with a slot's
        /// key at that index continues it.</summary>
        public long PrefixKey(Slot s, int tokens)
        {
            int i = tokens / Q35Kernels.KvBlockTokens - 1;
            return i >= 0 && i < s.Blocks.Count && _blockKey.TryGetValue(s.Blocks[i], out long key) ? key : 0;
        }

        /// <inheritdoc cref="PrefixKey(Slot, int)"/>
        public static long PrefixKey(HostSlot h, int tokens)
        {
            int i = tokens / Q35Kernels.KvBlockTokens - 1;
            return i >= 0 && i < h.Blocks.Length ? h.Blocks[i].Key : 0;
        }

        /// <summary>Host bytes freeing <paramref name="h"/> would release now (its state, and the blocks
        /// no other host slot holds).</summary>
        public long HostReleasableBytes(HostSlot h)
        {
            if (h?.StateUnits == null)
                return 0;
            long bytes = h.StateUnits.Length * KvBlockBytes + h.DraftMetadataBytes;
            foreach (HostBlock hb in h.Blocks)
                if (hb.Refs == 1)
                    bytes += KvBlockBytes;
            return bytes;
        }

        /// <summary>Host bytes copying <paramref name="s"/> would add (blocks the host already holds
        /// by content are not counted).</summary>
        public long HostBytesToCopy(Slot s)
        {
            long bytes = HostStateBytes + HostDraftMetadataCharge(CaptureNinferDraftSlotState(s));
            int nb = (s.NPast + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens;
            for (int i = 0; i < nb; i++)
                if (!_blockKey.TryGetValue(s.Blocks[i], out long key) || !_hostBlocks.ContainsKey(key))
                    bytes += KvBlockBytes;
            return bytes;
        }

        private void FreeHostPool()
        {
            if (_copyStream != IntPtr.Zero)
            {
                CudaDriverApi.cuStreamSynchronize(_copyStream);
                CudaDriverApi.cuStreamDestroy(_copyStream);
                _copyStream = IntPtr.Zero;
            }
            while (_freeEvents.Count > 0)
                CudaDriverApi.cuEventDestroy(_freeEvents.Pop());
            Thread pinner;
            lock (_hostLock)
            {
                _pinnerStop = true;
                Monitor.PulseAll(_hostLock);
                pinner = _pinner;
            }
            pinner?.Join();
            lock (_hostLock)
            {
                foreach (IntPtr p in _hostChunks)
                    CudaDriverApi.cuMemFreeHost(p);
                _hostChunks.Clear();
                _freeUnits.Clear();
                _pinnedBytes = 0;
            }
            _hostBlocks.Clear();
            HostBytesInUse = 0;
            HostDraftMetadataBytesInUse = 0;
        }
    }
}
