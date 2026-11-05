// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Sequence slots of the Qwen3.5 / Qwen3.8 direct-CUDA engine.
//
// A slot's attention caches are a list of blocks of KvBlockTokens tokens; one block holds every
// attention layer's K and V rows for those tokens ([layer][K | V][tokens][nKv][272 B]). Blocks are
// reference-counted and shared: a copy of a slot (a prompt checkpoint, a clone of one for a new
// request) shares every block below its position and copies only the recurrent states and the
// drafter ring, since attention rows never change once written. A slot that is about to write
// into a block someone else holds copies it first, so sharing is invisible to the kernels, which
// find each sequence's blocks through its row of the block table.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private readonly Dictionary<IntPtr, int> _blockRefs = new Dictionary<IntPtr, int>();
        private readonly Stack<IntPtr> _freeBlocks = new Stack<IntPtr>();
        /// <summary>Freed blocks kept for reuse before they go back to the driver.</summary>
        private const int MaxFreeBlocks = 64;

        /// <summary>Device memory an allocation leaves free (the CUDA context, transient buffers).</summary>
        private const long GrowthReserveBytes = 512L << 20;

        /// <summary>Called when the device lacks memory for a block or a slot: frees one reclaimable
        /// slot (a retained prefix) and returns true, or false when none is left.</summary>
        public Func<bool> ReclaimMemory { get; set; }

        /// <summary>Bytes of one token's K (or V) rows in one attention layer.</summary>
        private long KvRowBytes => (long)_m.NKvHead * (_m.Ninfer == null ? Q35Kernels.Q8Row
            : _ninferKvFp8 ? NinferFp8KvRowBytes : Q35Kernels.AttnHeadDim * 2);
        /// <summary>NInfer FP8 KV row: 256 E4M3 codes and two F32 group-128 scales, padded to 16 bytes.</summary>
        internal const int NinferFp8KvRowBytes = 272;
        // TS_Q35_NINFER_KV=fp8 selects the FP8 KV rows for an NInfer target (the default stays BF16).
        private readonly bool _ninferKvFp8 = CudaNinferAttention.ResolveKvFp8(Environment.GetEnvironmentVariable(CudaNinferAttention.KvEnvironment));
        /// <summary>Bytes of one layer's K (or V) rows in a block.</summary>
        private long KvLayerBytes => (long)Q35Kernels.KvBlockTokens * KvRowBytes;
        private long KvBlockBytes => 2L * Math.Max(1, _nAttn) * KvLayerBytes;
        private long KvLayerOffset(Layer L) => 2L * L.Ordinal * KvLayerBytes;
        private long StateBytes => (long)_m.GdnVHeads * Q35Kernels.GdnHeadDim * Q35Kernels.GdnHeadDim * 4;
        private long ConvBytes => (long)(_m.DConv - 1) * ConvDim * 4;

        /// <summary>Tokens a new slot's attention caches are sized for; blocks are added as it grows.</summary>
        public const int InitialSlotTokens = 4096;

        /// <summary>Free device memory.</summary>
        public long FreeDeviceBytes()
        {
            MakeCurrent();
            CudaDriverApi.cuMemGetInfo(out UIntPtr free, out _).ThrowOnError();
            return (long)free;
        }

        /// <summary>Device bytes of a slot's recurrent states and drafter ring.</summary>
        public long SlotStateBytes => SlotBytes(0);

        /// <summary>Device bytes of a slot's attention blocks (shared ones included).</summary>
        public long SlotKvBytes(Slot s) => s.Blocks.Count * KvBlockBytes;

        /// <summary>Device bytes a slot of <paramref name="capacity"/> tokens takes (its blocks unshared).</summary>
        public long SlotBytes(int capacity) =>
            _nGdn * (ConvBytes + StateBytes)
            + (long)((capacity + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens) * KvBlockBytes
            + RingBytes;

        /// <summary>Makes room for <paramref name="bytes"/> of device memory: the free blocks first,
        /// then retained slots through <see cref="ReclaimMemory"/>.</summary>
        private void EnsureDeviceMemory(long bytes)
        {
            while (FreeDeviceBytes() < bytes + GrowthReserveBytes)
            {
                if (_freeBlocks.Count > 0 || _freeArenas.Count > 0)
                {
                    ReleaseFreeBlocks();
                    ReleaseFreeArenas();
                    continue;
                }
                if (ReclaimMemory == null || !ReclaimMemory())
                    throw new OutOfMemoryException($"[q35-cuda] no device memory for {bytes >> 20} MB more");
            }
        }

        private void ReleaseFreeBlocks()
        {
            while (_freeBlocks.Count > 0)
                CudaDriverApi.cuMemFree(_freeBlocks.Pop());
        }

        // A slot's recurrent states and drafter ring are one allocation, [conv | state per GDN layer]
        // then [K | V per draft layer | positions]: a prefix hit takes a slot per request, and ~110
        // separate allocations (and as many frees) cost several milliseconds each time. Freed arenas
        // are kept for the next slot.
        private readonly Stack<IntPtr> _freeArenas = new Stack<IntPtr>();
        private const int MaxFreeArenas = 2;

        private static long Align256(long bytes) => (bytes + 255) & ~255L;
        private long RecurrentBytes => _nGdn * (Align256(ConvBytes) + Align256(StateBytes));
        private int RingLayers => HasNinferDrafter ? 5 : _dl?.Length ?? 0;
        private int RingHeads => HasNinferDrafter ? 8 : _d?.NKvHead ?? 0;
        private int RingHeadDim => HasNinferDrafter ? 128 : _d?.HeadDim ?? 0;
        private int RingRows => HasNinferDrafter ? 2048 : _ringRows;
        private long RingLayerBytes => (long)RingHeads * RingRows * RingHeadDim * 2;
        private long RingBytes => !HasDrafter ? 0 : 2L * RingLayers * Align256(RingLayerBytes) + Align256(4L * RingRows);
        private long ArenaBytes => RecurrentBytes + RingBytes;

        private void ReleaseFreeArenas()
        {
            while (_freeArenas.Count > 0)
                CudaDriverApi.cuMemFree(_freeArenas.Pop());
        }

        private IntPtr AllocBlock()
        {
            IntPtr b;
            if (_freeBlocks.Count > 0)
                b = _freeBlocks.Pop();
            else
            {
                EnsureDeviceMemory(KvBlockBytes);
                CudaDriverApi.cuMemAlloc(out b, new UIntPtr((ulong)KvBlockBytes)).ThrowOnError();
            }
            _blockRefs[b] = 1;
            RekeyBlock(b);
            return b;
        }

        private void ReleaseBlock(IntPtr b)
        {
            int refs = _blockRefs[b] - 1;
            if (refs > 0)
            {
                _blockRefs[b] = refs;
                return;
            }
            _blockRefs.Remove(b);
            DropBlockKey(b);
            if (_freeBlocks.Count < MaxFreeBlocks)
                _freeBlocks.Push(b);
            else
                CudaDriverApi.cuMemFree(b);
        }

        private void ReleaseBlocks(Slot s)
        {
            foreach (IntPtr b in s.Blocks)
                ReleaseBlock(b);
            s.Blocks.Clear();
            s.Version++;
        }

        /// <summary>A sequence's caches for <paramref name="capacity"/> tokens, empty.</summary>
        public Slot CreateSlot(int capacity)
        {
            MakeCurrent();
            var s = new Slot { Conv = new IntPtr[_nGdn], State = new IntPtr[_nGdn] };
            try
            {
                if (_freeArenas.Count > 0)
                    s.Arena = _freeArenas.Pop();
                else
                {
                    EnsureDeviceMemory(SlotBytes(0));
                    CudaDriverApi.cuMemAlloc(out s.Arena, new UIntPtr((ulong)ArenaBytes)).ThrowOnError();
                }
                long off = 0;
                for (int i = 0; i < _nGdn; i++, off += Align256(ConvBytes))
                    s.Conv[i] = s.Arena + (nint)off;
                for (int i = 0; i < _nGdn; i++, off += Align256(StateBytes))
                    s.State[i] = s.Arena + (nint)off;
                if (HasDrafter)
                    PlaceRing(s, s.Arena + (nint)off);
                Grow(s, Math.Min(capacity, _m.NCtx));
            }
            catch
            {
                FreeSlot(s);
                throw;
            }
            ResetSlot(s);
            return s;
        }

        /// <summary>Blocks for at least <paramref name="tokens"/> tokens.</summary>
        private void Grow(Slot s, int tokens)
        {
            int need = (tokens + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens;
            if (need > _maxBlocks)
                throw new InvalidOperationException($"[q35-cuda] {tokens} tokens exceed the {_m.NCtx}-token context");
            while (s.Blocks.Count < need)
            {
                s.Blocks.Add(AllocBlock());
                s.Version++;
            }
        }

        /// <summary>Gives each slot room for <paramref name="rows"/> more tokens and sole ownership of
        /// the blocks those rows land in, whose content keys change (host copies of their old rows
        /// stay valid only for the host slots that hold them).</summary>
        private void EnsureRoom(Slot[] slots, int rows)
        {
            RequireNoPendingNinferVerification();
            if (HasNinferDrafter)
                foreach (var s in slots) RequireNinferDraftRing(s);
            foreach (var s in slots)
            {
                if (s.NPast + rows > _m.NCtx)
                    throw new InvalidOperationException($"[q35-cuda] {s.NPast + rows} tokens exceed the {_m.NCtx}-token context");
                Grow(s, s.NPast + rows);
                int last = (s.NPast + rows - 1) / Q35Kernels.KvBlockTokens;
                for (int i = s.NPast / Q35Kernels.KvBlockTokens; i <= last; i++)
                {
                    IntPtr old = s.Blocks[i];
                    if (_blockRefs[old] == 1)
                    {
                        RekeyBlock(old);
                        continue;
                    }
                    IntPtr own = AllocBlock();
                    CudaDriverApi.cuMemcpyDtoDAsync(own, old, new UIntPtr((ulong)KvBlockBytes), Stream).ThrowOnError();
                    ReleaseBlock(old);
                    s.Blocks[i] = own;
                    s.Version++;
                }
            }
        }

        /// <summary>Zeroes a sequence's recurrent states and moves it to position 0, keeping its blocks
        /// (copied before any reuse when shared).</summary>
        public void ResetSlot(Slot s)
        {
            ForgetNinferVerification(s);
            MakeCurrent();
            CudaDriverApi.cuMemsetD8Async(s.Arena, 0, new UIntPtr((ulong)RecurrentBytes), Stream).ThrowOnError();
            ResetRing(s);
            s.NPast = 0;
            s.RopeDelta = 0;
            ResetNinferDraftSlot(s);
            Sync();
        }

        /// <summary>
        /// Makes <paramref name="dst"/> a copy of <paramref name="src"/>'s first
        /// <paramref name="tokens"/> tokens (its position, when that is less): the attention blocks
        /// below that are shared, the recurrent states, the drafter ring and the M-RoPE delta copied.
        /// </summary>
        public void CopySlot(Slot src, Slot dst, int tokens = int.MaxValue)
        {
            if (ReferenceEquals(src, dst))
                throw new ArgumentException("A slot cannot be copied onto itself.", nameof(dst));
            NinferDraftSlotSnapshot draftState = null;
            if (HasNinferDrafter)
            {
                RequireNinferDraftRing(src);
                RequireNinferDraftRing(dst);
                if (Math.Min(tokens, src.NPast) != src.NPast)
                    throw new NotSupportedException("NInfer draft/GDN checkpoints cannot rewind a longer slot to a shorter prefix.");
                draftState = CaptureNinferDraftSlotState(src);
            }
            ForgetNinferVerification(dst);
            MakeCurrent();
            int rows = Math.Min(tokens, src.NPast);
            ReleaseBlocks(dst);
            int share = (rows + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens;
            for (int i = 0; i < share; i++)
            {
                IntPtr b = src.Blocks[i];
                _blockRefs[b]++;
                dst.Blocks.Add(b);
            }
            CudaDriverApi.cuMemcpyDtoDAsync(dst.Arena, src.Arena, new UIntPtr((ulong)RecurrentBytes), Stream).ThrowOnError();
            // The ring describes the source's whole history; a shorter copy keeps it too (the
            // draft mask only reads rows before the anchor, and the next commits overwrite the rest).
            CopyRing(src, dst);
            dst.NPast = rows;
            dst.RopeDelta = src.RopeDelta;
            if (HasNinferDrafter) RestoreNinferDraftSlotState(dst, draftState);
            Sync();
        }

        public void FreeSlot(Slot s)
        {
            if (s == null)
                return;
            ForgetNinferVerification(s);
            ForgetNinferDraftSlot(s);
            MakeCurrent();
            Sync();
            ReleaseBlocks(s);
            foreach (IntPtr p in s.Owned)
                CudaDriverApi.cuMemFree(p);
            s.Owned.Clear();
            // An arena of another layout (a slot made before the drafter, its ring apart) is not reused.
            if (s.Arena != IntPtr.Zero)
            {
                bool current = !HasDrafter ? s.RingBase == IntPtr.Zero : s.RingBase == s.Arena + (nint)RecurrentBytes;
                if (_freeArenas.Count < MaxFreeArenas && current)
                    _freeArenas.Push(s.Arena);
                else
                    CudaDriverApi.cuMemFree(s.Arena);
                s.Arena = IntPtr.Zero;
            }
            if (Array.IndexOf(_bound, s) >= 0)
                _bound = Array.Empty<Slot>();
        }

        /// <summary>Clears the active sequence (recurrent states zeroed, position 0).</summary>
        public void Reset() => ResetSlot(_active);

        /// <summary>Taps the residual entering each of <paramref name="layers"/> (0-based target
        /// layers, in order) for every verified row: <see cref="FeatureRows"/>.</summary>
        public void SetFeatureTaps(int[] layers)
        {
            InvalidateVerifyGraphs();
            _featureSlot = new int[_m.NLayer];
            Array.Fill(_featureSlot, -1);
            _featureCount = 0;
            if (layers == null)
                return;
            foreach (int l in layers)
            {
                if (l < 0 || l >= _m.NLayer)
                    throw new ArgumentOutOfRangeException(nameof(layers), $"[q35-cuda] feature tap at layer {l}");
                _featureSlot[l] = _featureCount++;
            }
        }

        /// <summary>Points the kernels' per-sequence tables at <paramref name="slots"/> (again when a
        /// bound slot's blocks changed).</summary>
        private void BindSlots(Slot[] slots)
        {
            if (slots.Length > MaxSeqs)
                throw new ArgumentException($"[q35-cuda] {slots.Length} sequences in one step (at most {MaxSeqs})");
            if (slots.Length == _bound.Length)
            {
                bool same = true;
                for (int i = 0; i < slots.Length && same; i++)
                    same = ReferenceEquals(slots[i], _bound[i]) && slots[i].Version == _boundVersions[i];
                if (same)
                    return;
            }
            var blocks = new IntPtr[MaxSeqs * _maxBlocks];
            int g = Math.Max(1, _nGdn);
            var states = new IntPtr[2 * g * MaxSeqs];
            for (int q = 0; q < slots.Length; q++)
            {
                for (int i = 0; i < slots[q].Blocks.Count; i++)
                    blocks[q * _maxBlocks + i] = slots[q].Blocks[i];
                for (int i = 0; i < _nGdn; i++)
                {
                    states[i * MaxSeqs + q] = slots[q].Conv[i];
                    states[(g + i) * MaxSeqs + q] = slots[q].State[i];
                }
            }
            fixed (IntPtr* b = blocks)
                CudaDriverApi.cuMemcpyHtoD(_tabBlocks, (IntPtr)b, new UIntPtr((ulong)blocks.Length * 8)).ThrowOnError();
            fixed (IntPtr* h = states)
            {
                CudaDriverApi.cuMemcpyHtoD(_tabConv, (IntPtr)h, new UIntPtr((ulong)g * MaxSeqs * 8)).ThrowOnError();
                CudaDriverApi.cuMemcpyHtoD(_tabState, (IntPtr)(h + g * MaxSeqs), new UIntPtr((ulong)g * MaxSeqs * 8)).ThrowOnError();
            }
            if (_d != null)
                BindRings(slots, new IntPtr[(2 * _dl.Length + 1) * MaxSeqs]);
            _bound = (Slot[])slots.Clone();
            _boundVersions = new int[slots.Length];
            for (int i = 0; i < slots.Length; i++)
                _boundVersions[i] = slots[i].Version;
        }

        private IntPtr TabConv(Layer L) => _tabConv + (nint)((long)L.Ordinal * MaxSeqs * 8);
        private IntPtr TabState(Layer L) => _tabState + (nint)((long)L.Ordinal * MaxSeqs * 8);
    }
}
