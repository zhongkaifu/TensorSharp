// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Qwen3.8-Flash-Next direct-CUDA forward: scratch, one sequence's caches, and the per-layer
// pipeline (see Q4eCudaEngine.cs for the model and its references).
using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private void AllocateScratch(Dev dev)
        {
            dev.MakeCurrent();
            var m = _m;
            int nt = m.NUbatch, e = m.NEmbd;
            long hcE = HcDim;
            bool anyGdn = false, anyAttn = false, anyQsa = false, ple = false;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != dev.Ordinal)
                    continue;
                anyGdn |= L.Recurrent;
                anyAttn |= !L.Recurrent;
                anyQsa |= L.Ratio > 0;
                ple |= L.Ple;
            }
            // The MTP block is an attention layer on the last device.
            anyAttn |= _mtpLayer != null && dev.Ordinal == _lastDev;

            dev.Xs = AllocF32(dev, nt, hcE);
            dev.Xn = AllocF32(dev, nt, hcE);
            dev.G = AllocF32(dev, nt, hcE);
            dev.Lo = AllocF32(dev, nt, m.HcLowRank);
            dev.Inject = AllocF32(dev, nt, m.Hc);
            dev.HcPartials = AllocF32(dev, PerSlotMaxRows, hcE / Q4eKernels.HcSlice * Q4eKernels.HcPartialFloats);
            dev.Cur = AllocF32(dev, nt, e);
            dev.BlockOut = AllocF32(dev, nt, e);
            // Every decode-size projection input: the widest is the hyper-connection streams.
            int sharedIn = Math.Max(Math.Max(Math.Max(e, (int)hcE), Math.Max(ValueDim, m.NHead * m.HeadDim)), Math.Max(m.NFfShexp, m.HcLowRank));
            dev.SharedQ81 = AllocT(dev, DType.UInt8, PerSlotMaxRows, (long)(sharedIn / 32) * 36);
            if (e % 32 == 0)
                dev.CurQ81 = AllocT(dev, DType.UInt8, PerSlotMaxRows, (long)(e / 32) * 36);
            dev.GemvScratch = AllocF32(dev, Dsv4Kernels.GemvScratchFloats);
            if (anyGdn)
            {
                dev.Qkv = AllocF32(dev, nt, ConvDim);
                dev.Z = AllocF32(dev, nt, ValueDim);
                dev.Alpha = AllocF32(dev, nt, m.GdnVHeads);
                dev.Beta = AllocF32(dev, nt, m.GdnVHeads);
                dev.Scr = AllocF32(dev, (long)nt * m.GdnVHeads * GlmKernels.KdaScratch);
                dev.Core = AllocF32(dev, nt, ValueDim);
                dev.GOut = AllocF32(dev, nt, ValueDim);
            }
            if (anyAttn)
            {
                long qd = (long)m.NHead * m.HeadDim;
                dev.Qg = AllocF32(dev, nt, 2 * qd);
                dev.K = AllocF32(dev, nt, (long)m.NKvHead * m.HeadDim);
                dev.V = AllocF32(dev, nt, (long)m.NKvHead * m.HeadDim);
                dev.Q = AllocF32(dev, nt, qd);
                dev.Gate = AllocF32(dev, nt, qd);
                dev.AttnO = AllocF32(dev, nt, qd);
                if (dev.QK.HasAttentionSplit && AttnSplitsEnabled)
                {
                    dev.AttnPart = AllocF32(dev, (long)PerSlotMaxRows * m.NHead * MaxAttnSplits, Q4eKernels.AttnSplitPartFloats);
                    dev.AttnArrive = AllocT(dev, DType.Int32, (long)PerSlotMaxRows * m.NHead);
                    Zero(dev, dev.AttnArrive, 4);
                }
            }
            if (anyQsa && _kvWindow > 0)
                AllocKvStaging(dev);
            if (anyQsa)
            {
                dev.IdxRaw = AllocF32(dev, nt, m.IdxDim);
                dev.IdxQ = AllocF32(dev, nt, (long)m.IdxHeads * m.IdxDim);
                dev.Scores = AllocF32(dev, nt, ScoreStride);
                dev.Cells = AllocT(dev, DType.Int32, nt, CellStride);
                dev.CellCnt = AllocT(dev, DType.Int32, nt);
            }
            if (ple)
            {
                dev.PleEmb = AllocF32(dev, nt, e);
                dev.PleKeyOut = AllocF32(dev, nt, hcE);
                dev.PleValueOut = AllocF32(dev, nt, e);
                dev.PleGated = AllocF32(dev, nt, hcE);
                dev.PleNorm = AllocF32(dev, nt, hcE);
                CudaDriverApi.cuMemHostAlloc(out dev.PlePinned, new UIntPtr((ulong)nt * (ulong)e * 4), 0x1).ThrowOnError();
            }
            dev.RouterLogits = AllocF32(dev, nt, m.NExpert);
            dev.Moe = new CudaMoeScratch((type, sizes) => AllocT(dev, type, sizes), nt, m.NExpert, m.NExpertUsed, e, m.NFfExp);
            dev.ShGate = AllocF32(dev, nt, m.NFfShexp);
            dev.ShUp = AllocF32(dev, nt, m.NFfShexp);
            dev.ShDown = AllocF32(dev, nt, e);
            // One position per row of a captured step: a solo decode reads element 0, a verify window row r's, a
            // batched step's (a batched decode step or a batched verify, BatchGraphs.cs) row r's at element r.
            dev.PosDev = AllocT(dev, DType.Int32, Math.Max(VerifyMaxRows, Math.Max(MaxBatchedDecodeRows, PerSlotMaxRows)));
            if (dev.Ordinal == 0)
            {
                dev.Tokens = AllocT(dev, DType.Int32, nt);
                // PORTABLE: every device may read them.
                CudaDriverApi.cuMemHostAlloc(out _pinnedTokens, new UIntPtr((ulong)nt * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _posPinned,
                    new UIntPtr((ulong)Math.Max(VerifyMaxRows, Math.Max(MaxBatchedDecodeRows, PerSlotMaxRows)) * 4), 0x1).ThrowOnError();
            }
            if (dev.Ordinal == _lastDev)
            {
                dev.Logits = AllocF32(dev, 1, m.NVocab);
                dev.BatchLogits = AllocF32(dev, MaxBatchedDecodeRows, m.NVocab);
                // A greedy verify's first-max tokens: a window, or a batched verify's windows (PerSlotMaxRows rows).
                dev.ArgIds = AllocT(dev, DType.Int32, Math.Max(VerifyMaxRows, PerSlotMaxRows));
                CudaDriverApi.cuMemHostAlloc(out _pinnedIds, new UIntPtr((ulong)Math.Max(VerifyMaxRows, PerSlotMaxRows) * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _pinnedLogits,
                    new UIntPtr((ulong)MaxBatchedDecodeRows * (ulong)m.NVocab * 4), 0x1).ThrowOnError();
                // Every row's streams of a ubatch (a speculative prompt chunk), a verify window or a batched step.
                CudaDriverApi.cuMemHostAlloc(out _pinnedHidden,
                    new UIntPtr((ulong)Math.Max(nt, Math.Max(MaxBatchedDecodeRows, VerifyMaxRows)) * (ulong)hcE * 4), 0x1).ThrowOnError();
                if (_mtpLayer != null)
                    AllocateMtpScratch(dev);
            }
            if (anyGdn || ple)
            {
                dev.BatchStatePtrs = AllocT(dev, DType.UInt8, BatchStateTableBytes);
                CudaDriverApi.cuMemHostAlloc(out dev.BatchStatePinned, new UIntPtr((ulong)BatchStateTableBytes), 0x1).ThrowOnError();
            }
            CudaDriverApi.cuMemHostAlloc(out dev.BoundaryPinned, new UIntPtr((ulong)((long)nt * hcE * 4)), 0x1).ThrowOnError();
            // A prompt read layer by layer (one GPU under expert residency, whose plan reserved the bytes).
            if (_residency != null && _devs.Length == 1 && _expertStage != null && PromptGroupBytes() > 0)
            {
                if (!dev.DK.HasStageKeep)
                    Console.Error.WriteLine("[q4e-cuda] prompts run in ubatch order: the loaded tensorsharp_dsv4_kernels module predates "
                        + "ts_dsv4_moe_stage_copy_from (a committed PTX baseline not yet regenerated), so each ubatch stages its missed "
                        + "experts again. Rebuild TensorSharp.Backends.Cuda with a CUDA 13 nvcc on PATH.");
                else if (_expertStage.Capacity < m.NExpert)
                    Console.Error.WriteLine($"[q4e-cuda] prompts run in ubatch order: a layer's staging holds {_expertStage.Capacity} "
                        + $"experts, fewer than its {m.NExpert} (ubatch {m.NUbatch}), so it cannot be kept across a layer's ubatches.");
                else
                {
                    _groupXs = AllocF32(dev, PromptGroupRows, hcE);
                    if (GroupMoeBytes() > 0 && dev.DK.HasMoeRowMap)
                        AllocateGroupMoe(dev);
                    long f16Bytes = GroupF16Bytes();
                    if (f16Bytes > 0)
                    {
                        try { _groupF16 = AllocT(dev, DType.UInt8, f16Bytes); }
                        catch (Exception ex) when (ex is OutOfMemoryException or CudaException or InvalidOperationException)
                        {
                            Console.Error.WriteLine($"[q4e-cuda] the prompt groups' F16 weight arena ({f16Bytes / 1048576.0:F0} MiB) did not fit "
                                + $"({ex.Message}); each ubatch dequantizes its weights again.");
                        }
                    }
                }
            }
            CudaDriverApi.cuEventCreate(out dev.XsReadyEv, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.CopyDoneEv, 0x02).ThrowOnError();
            if (_forkShexp)
            {
                CudaDriverApi.cuStreamCreate(out dev.ShStream, 0x1 /*NON_BLOCKING*/).ThrowOnError();
                CudaDriverApi.cuEventCreate(out dev.ShForkEv, 0x02).ThrowOnError();
                CudaDriverApi.cuEventCreate(out dev.ShJoinEv, 0x02).ThrowOnError();
            }
        }

        // TS_Q4E_FORK_SHEXP=0: the shared expert on the device's stream, between the router and the routed experts.
        private readonly bool _forkShexp = Environment.GetEnvironmentVariable("TS_Q4E_FORK_SHEXP") != "0";

        /// <summary>Row stride of the block scores: every block the context holds.</summary>
        private int ScoreStride => Math.Max(1, _m.NCtx / MinRatio);

        /// <summary>Cells a sparse row attends: top_k plus the rest of the query's own block.</summary>
        private int CellStride => _m.IdxTopK + MaxRatio - 1;

        private int MinRatio
        {
            get
            {
                int r = int.MaxValue;
                foreach (var L in _m.Layers)
                    if (!L.Recurrent && L.CompressRatio > 0) r = Math.Min(r, L.CompressRatio);
                return r == int.MaxValue ? 1 : r;
            }
        }

        private int MaxRatio
        {
            get
            {
                int r = 1;
                // The model's layers, not the placed ones: the resident KV window sizes itself before placement.
                foreach (var L in _m.Layers)
                    if (!L.Recurrent && L.CompressRatio > 0) r = Math.Max(r, L.CompressRatio);
                return r;
            }
        }

        // -------------------------------------------------------------------
        // Sequence caches
        // -------------------------------------------------------------------

        private Slot CreateSlot()
        {
            var m = _m;
            int capacity = InitialSlotTokens;
            var slot = new Slot
            {
                Id = _nextSlotId++,
                Capacity = capacity,
                Layers = new SlotLayer[m.NLayer],
                StatePtrs = new Tensor[_devs.Length],
                DeviceBytes = new long[_devs.Length],
            };
            var ptrs = new long[_devs.Length][];
            for (int d = 0; d < _devs.Length; d++)
                ptrs[d] = new long[2 * m.NLayer + 1];
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                var dev = _devs[L.Device];
                dev.MakeCurrent();
                var c = new SlotLayer();
                if (L.Recurrent)
                {
                    c.ConvState = SlotTensor(slot, dev, DType.Float32, (long)(m.DConv - 1) * ConvDim);
                    c.Ssm = SlotTensor(slot, dev, DType.Float32, (long)m.GdnVHeads * Q4eKernels.GdnHeadDim * Q4eKernels.GdnHeadDim);
                    ptrs[dev.Ordinal][2 * il] = (long)Ptr(c.ConvState);
                    ptrs[dev.Ordinal][2 * il + 1] = (long)Ptr(c.Ssm);
                    if (_rewindCheckpoint)
                    {
                        slot.ConvCp ??= new Tensor[m.NLayer];
                        slot.SsmCp ??= new Tensor[m.NLayer];
                        slot.ConvCp[il] = SlotTensor(slot, dev, DType.Float32, c.ConvState.ElementCount());
                        slot.SsmCp[il] = SlotTensor(slot, dev, DType.Float32, c.Ssm.ElementCount());
                    }
                }
                else
                {
                    if (KvStreamed(il))
                    {
                        // The resident KV window (TS_Q4E_KV_RESIDENT): a window of pages in VRAM over the whole context in
                        // pinned host memory, the raw indexer keys in a ring and the pooled ones for the whole context.
                        // None of it grows with the slot (GrowSlotCaches leaves the layer alone).
                        AllocKvStream(slot, dev, c);
                        c.IdxRing = _idxRing;
                        if (L.Ratio > 0)
                        {
                            c.IdxRaw = SlotTensor(slot, dev, DType.Float16, (long)c.IdxRing * m.IdxDim);
                            c.IdxPooled = SlotTensor(slot, dev, DType.Float32, (long)(m.NCtx / L.Ratio) * m.IdxDim);
                        }
                    }
                    else
                    {
                        // Caches for the slot's capacity, grown as its sequence grows (EnsureCapacity).
                        c = AllocAttentionCaches(dev, L, capacity, slot.DeviceBytes);
                        foreach (Tensor t in new[] { c.KCache, c.VCache, c.KScale, c.VScale, c.IdxRaw, c.IdxPooled })
                            if (t != null)
                                slot.Owned.Add(t);
                    }
                }
                if (L.Ple && PleHistRows > 0)
                {
                    slot.PleHist = SlotTensor(slot, dev, DType.Float32, (long)PleHistRows * HcDim);
                    ptrs[dev.Ordinal][2 * m.NLayer] = (long)Ptr(slot.PleHist);
                    if (_rewindCheckpoint)
                        slot.PleHistCp = SlotTensor(slot, dev, DType.Float32, (long)PleHistRows * HcDim);
                }
                slot.Layers[il] = c;
            }
            for (int d = 0; d < _devs.Length; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                long bytes = 8L * (2 * m.NLayer + 1);
                slot.StatePtrs[d] = SlotTensor(slot, dev, DType.UInt8, bytes);
                fixed (long* p = ptrs[d])
                    CudaDriverApi.cuMemcpyHtoD(Ptr(slot.StatePtrs[d]), (IntPtr)p, new UIntPtr((ulong)bytes)).ThrowOnError();
            }
            ClearSlot(slot);
            return slot;
        }

        /// <summary>An attention layer's caches for <paramref name="capacity"/> positions (K/V - F16, or Q8_0 codes and
        /// one F16 scale per 32 values - and the QSA raw and block keys), their bytes added to
        /// <paramref name="deviceBytes"/>.</summary>
        private SlotLayer AllocAttentionCaches(Dev dev, DevLayer L, int capacity, long[] deviceBytes)
        {
            var m = _m;
            dev.MakeCurrent();
            var c = new SlotLayer();
            Tensor Make(DType type, long elements)
            {
                var t = new Tensor(dev.Alloc, type, elements);
                deviceBytes[dev.Ordinal] += elements * type.Size();
                return t;
            }
            try
            {
                long kv = (long)capacity * m.NKvHead * m.HeadDim;
                if (m.KvQ8)
                {
                    c.KCache = Make(DType.UInt8, kv);
                    c.VCache = Make(DType.UInt8, kv);
                    c.KScale = Make(DType.Float16, kv / Q4eKernels.KvQ8Block);
                    c.VScale = Make(DType.Float16, kv / Q4eKernels.KvQ8Block);
                }
                else
                {
                    c.KCache = Make(DType.Float16, kv);
                    c.VCache = Make(DType.Float16, kv);
                }
                if (L.Ratio > 0)
                {
                    c.IdxRaw = Make(DType.Float16, (long)capacity * m.IdxDim);
                    c.IdxPooled = Make(DType.Float32, (long)(capacity / L.Ratio) * m.IdxDim);
                }
                return c;
            }
            catch
            {
                c.KCache?.Dispose(); c.VCache?.Dispose(); c.KScale?.Dispose(); c.VScale?.Dispose(); c.IdxRaw?.Dispose();
                throw;
            }
        }

        private static IntPtr PtrOrZero(Tensor t) => t == null ? IntPtr.Zero : Ptr(t);

        private static Tensor SlotTensor(Slot slot, Dev dev, DType type, long elements)
        {
            dev.MakeCurrent();
            var t = new Tensor(dev.Alloc, type, elements);
            slot.Owned.Add(t);
            slot.DeviceBytes[dev.Ordinal] += elements * type.Size();
            return t;
        }

        /// <summary>Zero the recurrent states and the PLE history (an attention cache needs no
        /// clearing: nothing past the head is ever read).</summary>
        private void ClearSlot(Slot slot)
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                var c = slot.Layers[il];
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                if (c.ConvState != null)
                {
                    Zero(dev, c.ConvState, 4);
                    Zero(dev, c.Ssm, 4);
                }
                if (c.Kv != null)
                {
                    ResetKvStream(dev, c.Kv);
                    if (ReferenceEquals(dev.KvStagedFor, c))
                        dev.KvStagedFor = null;
                }
                if (_layers[il].Ple && slot.PleHist != null)
                    Zero(dev, slot.PleHist, 4);
            }
            slot.NPast = 0;
            SlotPositionsDropped(slot, 0);
            slot.Failed = false;
            slot.PleTokens.Length = 0;
            slot.MtpPast = 0;
            slot.MtpStart = 0;
            slot.CpNPast = -1;
        }

        /// <summary>Copy <paramref name="slot"/>'s recurrent states into its checkpoint
        /// (<paramref name="restore"/> false) or back (true), on each layer's device stream.</summary>
        private void CopyCheckpoint(Slot slot, bool restore)
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                if (slot.ConvCp?[il] == null)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                var c = slot.Layers[il];
                CopyD(dev, restore ? c.ConvState : slot.ConvCp[il], restore ? slot.ConvCp[il] : c.ConvState);
                CopyD(dev, restore ? c.Ssm : slot.SsmCp[il], restore ? slot.SsmCp[il] : c.Ssm);
            }
            if (slot.PleHistCp != null)
            {
                var dev = _devs[_layers[_pleLayer].Device];
                dev.MakeCurrent();
                CopyD(dev, restore ? slot.PleHist : slot.PleHistCp, restore ? slot.PleHistCp : slot.PleHist);
            }
        }

        private static void CopyD(Dev dev, Tensor dst, Tensor src)
            => CudaDriverApi.cuMemcpyDtoDAsync(Ptr(dst), Ptr(src), new UIntPtr((ulong)src.ElementCount() * 4), dev.Stream).ThrowOnError();

        /// <summary>The active slot's states, at its head, become its rewind checkpoint: called at the
        /// end of every multi-token forward, a prompt boundary.</summary>
        private void CheckpointActive()
        {
            Slot s = _active;
            if (!_rewindCheckpoint || s.Failed || s.ConvCp == null)
                return;
            CopyCheckpoint(s, restore: false);
            s.CpNPast = s.NPast;
        }

        private static void Zero(Dev dev, Tensor t, int elementBytes)
            => CudaDriverApi.cuMemsetD8Async(Ptr(t), 0, new UIntPtr((ulong)t.ElementCount() * (ulong)elementBytes), dev.Stream).ThrowOnError();

        private void FreeSlot(Slot slot)
        {
            if (slot == null)
                return;
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream);
            }
            DestroySlotGraphs(slot.Id);
            foreach (var dev in _devs)
                if (dev.KvStagedFor != null && Array.IndexOf(slot.Layers, dev.KvStagedFor) >= 0)
                    dev.KvStagedFor = null;
            CountSlot(slot, -1);
            foreach (var t in slot.Owned)
                t.Dispose();
            slot.Owned.Clear();
            ReturnKvHostSets(slot);
            SlotPositionsDropped(slot, -1);
        }

        /// <summary>Back to position 0 on the active sequence.</summary>
        public void Reset() => ClearSlot(_active);

        /// <summary>Address of this slot's pointer to its [conv, ssm] state of a GDN layer (kind 0, 1)
        /// on <paramref name="dev"/>, or to its PLE history (il = NLayer, kind 0): a one-sequence
        /// pointer array for the kernels.</summary>
        private static IntPtr StatePtr(Slot slot, Dev dev, int il, int kind)
            => (IntPtr)((long)Ptr(slot.StatePtrs[dev.Ordinal]) + (2L * il + kind) * 8);

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut) => Forward(tokens, logitsOut, null);

        /// <summary>
        /// A forward that also returns the hidden streams (<see cref="HiddenWidth"/> floats a row, before the output
        /// mixer: what the MTP block reads): every row's when <paramref name="hOut"/> holds them all, else the last
        /// row's.
        /// </summary>
        public void Forward(int[] tokens, float[] logitsOut, float[] hOut)
        {
            bool deferred = _deferPromptCall;
            _deferPromptCall = false;
            _fwdWaitTicks = _fwdPleTicks = 0;
            // A prompt chunk left in flight is followed in stream order by the same sequence's next chunk, as one call's
            // ubatches follow each other; any other forward waits for it first.
            if (_promptInFlight && ReferenceEquals(_promptInFlightSlot, _active) && tokens != null && tokens.Length > PerSlotMaxRows)
                DrainLanes();
            else
                DrainDecodeLanes();
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[q4e-cuda] the sequence's last forward failed; reset it before reuse");
            if (slot.NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[q4e-cuda] context overflow: n_past={slot.NPast} + {tokens.Length} > n_ctx={_m.NCtx}");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            EnsureCapacity(slot, slot.NPast + tokens.Length);
            bool sampled = logitsOut != null && BeginSampledForward();
            _issueEnd = 0;
            slot.Failed = true;
            ArmHostExperts();
            try
            {
                bool allHidden = hOut != null && hOut.LongLength >= (long)tokens.Length * HcDim;
                if (tokens.Length == 1 && logitsOut != null && hOut == null && CanCaptureDecode && ForwardDecodeGraphed(slot, tokens[0], slot.NPast, logitsOut))
                {
                    slot.NPast++;
                }
                else if (UsePromptGroups(tokens.Length))
                {
                    ForwardGrouped(slot, tokens, logitsOut, hOut, allHidden);
                }
                else
                {
                    int done = 0;
                    while (done < tokens.Length)
                    {
                        int nt = Math.Min(_m.NUbatch, tokens.Length - done);
                        bool last = done + nt == tokens.Length;
                        ForwardUbatch(slot, tokens, done, nt, slot.NPast, last ? logitsOut : null,
                            hOut: allHidden || last ? hOut : null, hRowOffset: allHidden ? done : -1);
                        // A speculating request's rows: the MTP block takes them on its device, behind the ubatch.
                        if (MtpFollowsForward)
                            MtpFollowUbatch(slot, tokens, done, nt, slot.NPast);
                        slot.NPast += nt;
                        done += nt;
                    }
                }
            }
            finally
            {
                if (sampled)
                {
                    try { EndSampledForward(tokens.Length, tokens.Length == 1); }
                    catch (Exception) { _sampling = false; }
                }
            }
            CheckHostExperts(tokens.Length <= _hostExpertRows ? tokens.Length : 0);
            if (tokens.Length <= _hostExpertRows)
                AdaptResidency();
            else
                AdaptAfterPrompt(tokens.Length);
            CountForwardTime(start);
            slot.Failed = false;
            if (tokens.Length >= 2)
                CheckpointActive();
            if (deferred)
            {
                // Issued, not waited for: the last device still runs it (PromptPipeline.cs).
                _promptInFlight = true;
                _promptInFlightSlot = slot;
                PromptChunksDeferred++;
            }
            else
                DrainPrompt();
            if (_perf > 0)
            {
                double secs = sw.Elapsed.TotalSeconds;
            {
                // A prompt's host time: waiting for a device before its pinned inputs are rewritten, and the PLE rows.
                string host = tokens.Length > PerSlotMaxRows
                    ? $" (waited for a device {TicksMs(_fwdWaitTicks):F0}ms, PLE rows {TicksMs(_fwdPleTicks):F0}ms)" : "";
                Console.Error.WriteLine(deferred
                    ? $"[q4e-cuda] forward {tokens.Length} tokens issued in {secs:F3}s and left in flight (the next chunk follows it){host}"
                    : $"[q4e-cuda] forward {tokens.Length} tokens in {secs:F3}s ({tokens.Length / secs:F1} tok/s), "
                      + $"host issue {IssueMs(start):F1}ms{host}" + KvFetchedNote(slot, tokens.Length > PerSlotMaxRows ? 1 : 32));
            }
            }
        }

        /// <param name="rows">A batched decode step: one row per sequence (<paramref name="slot"/> is
        /// then unused), each at <paramref name="positions"/>[row]; null for one sequence's ubatch at
        /// <paramref name="p0"/>.</param>
        /// <param name="hRowOffset">The row of <paramref name="hOut"/> this ubatch's first row goes to; -1 writes only
        /// the last row, to the start.</param>
        private void ForwardUbatch(Slot slot, int[] tokens, int tokOff, int nt, int p0, float[] logitsOut,
            Slot[] rows = null, int[] positions = null, bool allRows = false, float[] hOut = null, int hRowOffset = 0)
        {
            var m = _m;
            int e = m.NEmbd;
            var dev0 = _devs[0];
            dev0.MakeCurrent();
            // A step that stages its missed experts needs the staging an overlapped swap round may hold.
            if (nt > _hostExpertRows)
                FlushAsyncSwaps();
            // The pinned tokens are rewritten per ubatch: the previous upload must have run.
            long waitStart = System.Diagnostics.Stopwatch.GetTimestamp();
            CudaDriverApi.cuStreamSynchronize(dev0.Stream).ThrowOnError();
            _fwdWaitTicks += System.Diagnostics.Stopwatch.GetTimestamp() - waitStart;
            Marshal.Copy(tokens, tokOff, _pinnedTokens, nt);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev0.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev0.Stream).ThrowOnError();
            StageBegin(dev0);
            dev0.DK.Embed(_tokEmbd.Ptr, dev0.Tokens, dev0.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, e, dev0.Stream);
            StageEnd(dev0, StEmbed);

            // The PLE rows are hashed and gathered on the host before the step's GPU work, as the
            // native span does: the sequences' token histories take this step's tokens first.
            if (_pleLayer >= 0)
                GatherPleRows(slot, tokens, tokOff, nt, p0, rows, positions);

            int curDev = 0;
            bool ready = false;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != curDev)
                {
                    ready = false;
                    Handoff(_devs[curDev], _devs[L.Device], nt);
                    curDev = L.Device;
                    StageEnd(_devs[curDev], StBoundary);
                }
                var dev = _devs[curDev];
                dev.MakeCurrent();
                ready = RunLayer(dev, il, slot, nt, p0, rows, positions, IntPtr.Zero, ready);
            }

            if (logitsOut != null || hOut != null)
                Head(nt, logitsOut, rows != null || allRows, hOut, hRowOffset);
        }

        // -------------------------------------------------------------------
        // Prompts read layer by layer
        // -------------------------------------------------------------------

        /// <summary>
        /// Rows of a prompt read layer by layer (TS_Q4E_PROMPT_GROUP, default 4096; 0 keeps ubatch order). Under expert
        /// residency a prompt ubatch stages the experts it selects that have no VRAM slot over the bus, and the next ubatch
        /// of the same layer stages most of them again: a 4K-token prompt in 512-row ubatches crossed each layer's cold
        /// experts eight times. A group runs every layer over all its ubatches before the next layer, the layer's
        /// staging kept between them (each expert crosses once per layer and group), at the price of the group's streams
        /// [rows, hc * E] held in VRAM. Each ubatch computes exactly what it computes in ubatch order: the same kernels
        /// over the same rows, the attention and recurrent caches filled in position order layer by layer.
        /// Strata reads a prompt in 8192-token chunks for the same reason (src/prefill/prefill.cpp).
        /// </summary>
        private readonly int _promptGroupEnv = Math.Max(0, EnvInt("TS_Q4E_PROMPT_GROUP", 4096));
        private Tensor _groupXs;
        // _groupXs's writes: each group counts one before it runs. The last group to finish leaves its rows' output streams
        // there (positions _groupHiddenP0 .. + _groupHiddenRows of _groupHiddenSlot) until the next group runs; the MTP
        // catch-up after a prompt reads them there (MtpUploadRows) instead of the copy the host hands back.
        private long _groupXsGen, _groupHiddenGen = -1;
        private Slot _groupHiddenSlot;
        private int _groupHiddenP0, _groupHiddenRows;

        /// <summary>
        /// <paramref name="slot"/>'s position fell to <paramref name="nPast"/> (a reset, a rewind, a verify's rollback, the slot
        /// freed with -1): the positions from there on will be written again, so the last prompt group's streams no longer
        /// stand for those positions if they covered any. Without this a new conversation on the slot - a short prompt over
        /// positions the last group had held - had its MTP catch-up read the old prompt's streams (the drafts' acceptance fell
        /// to 4-13% on long generation after a grouped multi-turn prompt; the answers stayed right, the verify decides them).
        /// </summary>
        private void SlotPositionsDropped(Slot slot, int nPast)
        {
            if (ReferenceEquals(slot, _groupHiddenSlot) && nPast < _groupHiddenP0 + _groupHiddenRows)
            {
                _groupHiddenGen = -1;
                _groupHiddenSlot = null;
            }
        }
        // Inside a grouped prompt: the routed experts stage (never the host's doorbell) and keep their staging per layer.
        private bool _groupedPrompt;

        /// <summary>Rows per group: whole ubatches, so the ubatches are the ones ubatch order runs.</summary>
        private int PromptGroupRows => _promptGroupEnv / _m.NUbatch * _m.NUbatch;

        /// <summary>The group's stream bytes residency plans around (one GPU only).</summary>
        private long PromptGroupBytes() => PromptGroupRows > _m.NUbatch ? (long)PromptGroupRows * HcDim * 4 : 0;

        // TS_Q4E_GROUP_F16=0: every ubatch of a prompt group dequantizes its layer's dense weights to F16 again.
        private readonly bool _groupF16Env = Environment.GetEnvironmentVariable("TS_Q4E_GROUP_F16") != "0";
        private Tensor _groupF16;

        /// <summary>F16 weight arena bytes of a prompt group: the largest layer's quantized dense projections as F16, each on
        /// a 256-byte boundary (the F16 GEMM route's copies; F32, F16 and BF16 weights are used as they are). Zero without
        /// prompt groups. Residency plans around it, as around the group's streams.</summary>
        private long GroupF16Bytes()
        {
            if (!_groupF16Env || PromptGroupBytes() == 0 || _m.Layers == null)
                return 0;
            static long F16(in CudaWeightDesc w)
                => w.GgmlType is 0 or 1 or 30 || w.Ne0 <= 0 || w.Ne1 <= 0 ? 0 : ((long)w.Ne0 * w.Ne1 * 2 + 255) & ~255L;
            long most = 0;
            foreach (var l in _m.Layers)
            {
                long sum = F16(l.HcAttnDown) + F16(l.HcAttnUp) + F16(l.HcFfnDown) + F16(l.HcFfnUp)
                    + F16(l.Qkv) + F16(l.Gate) + F16(l.SsmOut) + F16(l.Wq) + F16(l.Wk) + F16(l.Wv) + F16(l.Wo)
                    + F16(l.PleKey) + F16(l.PleValue) + F16(l.GateShexp) + F16(l.UpShexp) + F16(l.DownShexp);
                most = Math.Max(most, sum);
            }
            return most;
        }

        /// <summary>Dense weights the prompt groups dequantized into the arena, and GEMM calls that reused such a copy
        /// (a test proves the path engaged).</summary>
        internal long GroupF16Dequantized { get; private set; }
        internal long GroupF16Reused { get; private set; }

        // TS_Q4E_GROUP_EXPERTS=0: each ubatch of a prompt group runs its routed experts' gate/up itself.
        private readonly bool _groupExpertsEnv = Environment.GetEnvironmentVariable("TS_Q4E_GROUP_EXPERTS") != "0";

        /// <summary>A prompt group's routed-expert inputs while its gate/up waits for the layer's last ubatch: the
        /// selections, the tokens' int8 activations, the shared expert's outputs and the scatter logits per group row; the
        /// group's grouping plan and H; and the act rows a ubatch's down reads H by.</summary>
        private sealed class GroupMoeBuffers
        {
            public Tensor Sel, SelW, Xa, H, ShDown, Inject, Counts, Offsets, Cursors, RowOfSlot, SlotToken, ActRow;
        }

        private GroupMoeBuffers _groupMoe;

        /// <summary>Layers whose prompt groups ran the routed experts' gate/up once over the group (a test proves the path
        /// engaged).</summary>
        internal long GroupExpertsRun { get; private set; }

        /// <summary>Whether the prompt groups plan for group gate/up, from the descriptors: groups on, the int8 Q2_0 expert
        /// route switched on and taking the widths, every layer's experts Q2_0, and ubatches past the per-slot rows.</summary>
        private bool GroupExpertsPlanned()
        {
            var m = _m;
            if (!_groupExpertsEnv || PromptGroupBytes() == 0 || !CudaMoe.Q2_0ExpertsInt8 || m.Layers == null
                || m.NUbatch <= PerSlotMaxRows || !Dsv4Kernels.MoeQ2I8Supports(m.NEmbd, m.NFfExp))
                return false;
            foreach (var l in m.Layers)
                if (l.GateExps.GgmlType != TQ2_0 || l.UpExps.GgmlType != TQ2_0 || l.DownExps.GgmlType != TQ2_0)
                    return false;
            return true;
        }

        private const int TQ2_0 = 42;

        /// <summary>Bytes of <see cref="GroupMoeBuffers"/> (residency plans around them, as around the group's streams).</summary>
        private long GroupMoeBytes()
        {
            if (!GroupExpertsPlanned())
                return 0;
            var m = _m;
            long g = PromptGroupRows, slots = g * m.NExpertUsed;
            return slots * 8 + Dsv4Kernels.MoeQ2I8ActRowsBytes(g, m.NEmbd) + Dsv4Kernels.MoeQ2I8ActRowsBytes(slots, m.NFfExp)
                + g * m.NEmbd * 4L + g * m.Hc * 4L + 3L * m.NExpert * 4 + slots * 8 + (long)m.NUbatch * m.NExpertUsed * 4;
        }

        private void AllocateGroupMoe(Dev dev)
        {
            var m = _m;
            long g = PromptGroupRows, slots = g * m.NExpertUsed;
            var b = new GroupMoeBuffers();
            try
            {
                b.Sel = AllocT(dev, DType.Int32, g, m.NExpertUsed);
                b.SelW = AllocF32(dev, g, m.NExpertUsed);
                b.Xa = AllocT(dev, DType.UInt8, Dsv4Kernels.MoeQ2I8ActRowsBytes(g, m.NEmbd));
                b.H = AllocT(dev, DType.UInt8, Dsv4Kernels.MoeQ2I8ActRowsBytes(slots, m.NFfExp));
                b.ShDown = AllocF32(dev, g, m.NEmbd);
                b.Inject = AllocF32(dev, g, m.Hc);
                b.Counts = AllocT(dev, DType.Int32, m.NExpert);
                b.Offsets = AllocT(dev, DType.Int32, m.NExpert);
                b.Cursors = AllocT(dev, DType.Int32, m.NExpert);
                b.RowOfSlot = AllocT(dev, DType.Int32, slots);
                b.SlotToken = AllocT(dev, DType.Int32, slots);
                b.ActRow = AllocT(dev, DType.Int32, (long)m.NUbatch * m.NExpertUsed);
                _groupMoe = b;
            }
            catch (Exception ex) when (ex is OutOfMemoryException or CudaException or InvalidOperationException)
            {
                Console.Error.WriteLine($"[q4e-cuda] the prompt groups' routed-expert buffers ({GroupMoeBytes() / 1048576.0:F0} MiB) did not "
                    + $"fit ({ex.Message}); each ubatch runs its experts' gate/up itself.");
            }
        }

        /// <summary>Whether layer <paramref name="L"/> of a <paramref name="g"/>-row group runs its routed experts' gate/up
        /// once over the group: buffers and kernels there, the int8 route taking this layer's experts, staging that keeps
        /// across the layer's ubatches, and every ubatch past the per-slot rows (a short last ubatch runs the per-slot
        /// kernels in ubatch order, so the group keeps that order there).</summary>
        private bool GroupExpertsFor(Dev dev, DevLayer L, int g)
        {
            var m = _m;
            int ub = m.NUbatch, last = g - (g - 1) / ub * ub;
            return _groupMoe != null && g > ub && last > PerSlotMaxRows && dev.DK.HasMoeRowMap
                && CudaMoe.UsesQ2I8(CudaMoe.Q2_0ExpertsInt8, dev.DK.HasMoeQ2I8, L.GateExps, L.UpExps, L.DownExps, m.NEmbd, m.NFfExp, m.NExpertUsed)
                && CudaMoe.SelectRoute(last, PerSlotMaxRows, dev.DK.HasMoeMma, dev.DK.DecodesLowBitIqExperts, L.GateExps.Type,
                    L.DownExps.Type, m.NEmbd, m.NFfExp, dev.DK.DecodesLegacyQuants) == CudaMoe.ExpertRoute.TensorCore
                && CudaMoe.CanStageKept(dev.DK, _promptStage ?? _expertStage, L.Stage);
        }

        /// <summary>Phase one of a deferred ubatch (rows <paramref name="row0"/> of the group): its missed experts staged (kept
        /// for the layer), its selections, int8 activations, shared-expert outputs and scatter logits into the group's
        /// buffers. The router and the shared expert ran as for any ubatch.</summary>
        private void DeferExperts(Dev dev, DevLayer L, int nt, int row0)
        {
            var m = _m;
            var b = _groupMoe;
            int nUsed = m.NExpertUsed, e = m.NEmbd;
            IntPtr stream = dev.Stream;
            CudaMoe.StageKept(dev.DK, dev.Moe.Sel, nt * nUsed, _promptStage ?? _expertStage, L.Stage, stream);
            CudaDriverApi.cuMemcpyDtoDAsync(At(b.Sel, (long)row0 * nUsed), Ptr(dev.Moe.Sel), new UIntPtr((ulong)nt * (ulong)nUsed * 4), stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync(At(b.SelW, (long)row0 * nUsed), Ptr(dev.Moe.SelW), new UIntPtr((ulong)nt * (ulong)nUsed * 4), stream).ThrowOnError();
            dev.DK.MoeQ2I8QuantAct(Ptr(dev.Cur), (IntPtr)((long)Ptr(b.Xa) + Dsv4Kernels.MoeQ2I8ActRowsBytes(row0, e)), nt, e, stream);
            CudaDriverApi.cuMemcpyDtoDAsync(At(b.ShDown, (long)row0 * e), Ptr(dev.ShDown), new UIntPtr((ulong)nt * (ulong)e * 4), stream).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync(At(b.Inject, (long)row0 * m.Hc), Ptr(dev.Inject), new UIntPtr((ulong)nt * (ulong)m.Hc * 4), stream).ThrowOnError();
        }

        /// <summary>Phase two of a layer's deferred group: the group's grouping plan and gate/up once over all its rows
        /// (each expert's gate and up weights read once instead of once per ubatch), then per ubatch its own plan, its
        /// slots' H rows mapped from the group's, down, the combine with the shared expert, and the scatter into the
        /// streams. Each slot's products are those of ubatch order bit for bit.</summary>
        private void GroupExperts(Dev dev, DevLayer L, int g)
        {
            var m = _m;
            var b = _groupMoe;
            int ub = m.NUbatch, nUsed = m.NExpertUsed, e = m.NEmbd, ff = m.NFfExp, hc = HcDim;
            IntPtr stream = dev.Stream;
            using (Tensor sel = Block(b.Sel, 0, g, nUsed))
                CudaMoe.Plan(dev.DK, dev.Alloc.Kernels, sel, b.Counts, b.Offsets, b.Cursors, b.RowOfSlot, b.SlotToken, g * nUsed,
                    nUsed, m.NExpert, stream);
            dev.DK.MoeQ2I8GateUp(L.GateExps.Ptr, L.UpExps.Ptr, Ptr(b.Xa), b.Counts, b.Offsets, b.SlotToken, Ptr(b.H), ff, e,
                L.GateExps.RowBytes, 0f, m.NExpert, stream, L.GateExps.Table, L.UpExps.Table);
            for (int r0 = 0; r0 < g; r0 += ub)
            {
                int nt = Math.Min(ub, g - r0), slots = nt * nUsed;
                using Tensor sel = Block(b.Sel, r0, nt, nUsed);
                using Tensor selW = Block(b.SelW, r0, nt, nUsed);
                using Tensor shDown = Block(b.ShDown, r0, nt, e);
                CudaMoe.Plan(dev.DK, dev.Alloc.Kernels, sel, dev.Moe.Counts, dev.Moe.Offsets, dev.Moe.Cursors, dev.Moe.RowOfSlot,
                    dev.Moe.SlotToken, slots, nUsed, m.NExpert, stream);
                dev.DK.MoeRowMap(dev.Moe.RowOfSlot, b.RowOfSlot, r0 * nUsed, b.ActRow, slots, stream);
                dev.DK.MoeQ2I8Down(L.DownExps.Ptr, Ptr(b.H), dev.Moe.Counts, dev.Moe.Offsets, dev.Moe.ExpDown, e, ff,
                    L.DownExps.RowBytes, m.NExpert, stream, L.DownExps.Table, Ptr(b.ActRow));
                dev.DK.MoeScatterAdd(dev.Moe.ExpDown, dev.Moe.RowOfSlot, selW, shDown, dev.BlockOut, nt, nUsed, e, stream);
                using Tensor xs = Block(_groupXs, r0, nt, hc);
                dev.QK.HcPost(Ptr(xs), Ptr(dev.BlockOut), At(b.Inject, (long)r0 * m.Hc), nt, e, m.Hc, stream);
            }
            StageEnd(dev, StExperts);
            GroupExpertsRun++;
        }

        /// <summary>Prompt groups read layer by layer so far (a test proves the path engaged).</summary>
        internal long PromptGroupsRun { get; private set; }

        private bool UsePromptGroups(int rows)
            => _groupXs != null && rows > _m.NUbatch && _devs.Length == 1;

        private void ForwardGrouped(Slot slot, int[] tokens, float[] logitsOut, float[] hOut, bool allHidden)
        {
            int done = 0, n = tokens.Length, cap = PromptGroupRows;
            while (done < n)
            {
                int g = Math.Min(cap, n - done);
                bool last = done + g == n;
                ForwardGroup(slot, tokens, done, g, slot.NPast, last ? logitsOut : null,
                    allHidden || last ? hOut : null, allHidden ? done : -1);
                slot.NPast += g;
                done += g;
            }
        }

        /// <param name="hRowOffset">The row of <paramref name="hOut"/> the group's first row goes to; -1 writes only the
        /// last row, to the start.</param>
        private void ForwardGroup(Slot slot, int[] tokens, int tokOff, int g, int p0, float[] logitsOut, float[] hOut, int hRowOffset)
        {
            var m = _m;
            var dev = _devs[0];
            int ub = m.NUbatch, hc = HcDim;
            dev.MakeCurrent();
            FlushAsyncSwaps();
            Tensor ubatchXs = dev.Xs;
            _groupedPrompt = true;
            _groupXsGen++;
            bool prefetch = PromptPrefetchUsable();
            try
            {
                if (prefetch)
                {
                    // Both halves are free from here (the previous group unstaged in this stream's order).
                    for (int h = 0; h < 2; h++)
                        CudaDriverApi.cuEventRecord(_halfFree[h], dev.Stream).ThrowOnError();
                    PromptPrefetchGroups++;
                }
                bool staged = false;
                // Every row's streams start as its token's embedding.
                for (int r0 = 0; r0 < g; r0 += ub)
                {
                    int nt = Math.Min(ub, g - r0);
                    // The pinned tokens are rewritten per ubatch: the previous upload must have run.
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                    Marshal.Copy(tokens, tokOff + r0, _pinnedTokens, nt);
                    CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev.Stream).ThrowOnError();
                    StageBegin(dev);
                    using (Tensor xs = Block(_groupXs, r0, nt, hc))
                        dev.DK.Embed(_tokEmbd.Ptr, dev.Tokens, xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, m.NEmbd, dev.Stream);
                    StageEnd(dev, StEmbed);
                }
                for (int il = 0; il < m.NLayer; il++)
                {
                    var L = _layers[il];
                    // A layer whose host experts fit a half runs on it, staged ahead while the layer before ran when that
                    // one did too; any other layer stages per ubatch over the whole staging.
                    bool half = prefetch && _halfFits[il];
                    var stage = half ? _halfStages[il & 1] : _expertStage;
                    if (half)
                    {
                        if (staged)
                            CudaDriverApi.cuStreamWaitEvent(dev.Stream, _prefetchDone[il & 1], 0).ThrowOnError();
                        // The next layer's host experts cross the bus while this layer runs.
                        staged = il + 1 < m.NLayer && _halfFits[il + 1];
                        if (staged)
                            PrefetchLayer(il + 1, (il + 1) & 1, _prefetchStream, waitFree: true);
                    }
                    else
                        staged = false;
                    _promptStage = stage;
                    // The layer's ubatches share its dense weights' F16 copies: dequantized by the first, reused by the rest.
                    CudaQuantizedOps.F16WeightScope f16 = _groupF16 != null && g > ub
                        ? CudaQuantizedOps.BeginF16WeightScope(dev.Alloc, Ptr(_groupF16), _groupF16.ElementCount())
                        : null;
                    // The routed experts' gate/up once over the group, after its last ubatch: each expert's gate and up
                    // weights are read once per layer and group instead of once per ubatch.
                    bool deferExperts = GroupExpertsFor(dev, L, g);
                    try
                    {
                        for (int r0 = 0; r0 < g; r0 += ub)
                        {
                            int nt = Math.Min(ub, g - r0);
                            using Tensor xs = Block(_groupXs, r0, nt, hc);
                            dev.Xs = xs;
                            // The PLE rows of a ubatch are gathered when its PLE block runs (the histories take its tokens
                            // then, in position order as before).
                            if (L.Ple)
                                GatherPleRows(slot, tokens, tokOff + r0, nt, p0 + r0, null, null);
                            // No fused scatter into the next layer's mixer: the next call is the next ubatch of this layer.
                            RunLayer(dev, il, slot, nt, p0 + r0, null, null, IntPtr.Zero, deferRow: deferExperts ? r0 : -1);
                        }
                        if (deferExperts)
                            GroupExperts(dev, L, g);
                    }
                    finally
                    {
                        if (f16 != null)
                        {
                            GroupF16Dequantized += f16.Dequantized;
                            GroupF16Reused += f16.Reused;
                            f16.Dispose();
                        }
                    }
                    // The layer's ubatches shared one staging.
                    CudaMoe.Unstage(dev.DK, stage, L.Stage, dev.Stream);
                    // A half is free again once its layer unstaged; the whole staging frees both.
                    if (half)
                        CudaDriverApi.cuEventRecord(_halfFree[il & 1], dev.Stream).ThrowOnError();
                    else if (prefetch)
                        for (int h = 0; h < 2; h++)
                            CudaDriverApi.cuEventRecord(_halfFree[h], dev.Stream).ThrowOnError();
                }
                // Every layer has run: the group's output streams stay in _groupXs until the next group.
                _groupHiddenSlot = slot;
                _groupHiddenP0 = p0;
                _groupHiddenRows = g;
                _groupHiddenGen = _groupXsGen;
                if (hOut != null && hRowOffset >= 0 && hOut.LongLength >= ((long)hRowOffset + g) * hc)
                {
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                    fixed (float* dst = hOut)
                        CudaDriverApi.cuMemcpyDtoH((IntPtr)(dst + (long)hRowOffset * hc), Ptr(_groupXs),
                            new UIntPtr((ulong)g * (ulong)hc * 4)).ThrowOnError();
                    hOut = null;
                }
                PromptGroupsRun++;
                int lastNt = g - (g - 1) / ub * ub;
                if (logitsOut != null || hOut != null)
                    using (Tensor xs = Block(_groupXs, g - lastNt, lastNt, hc))
                    {
                        dev.Xs = xs;
                        Head(lastNt, logitsOut, false, hOut, -1);
                    }
            }
            finally
            {
                dev.Xs = ubatchXs;
                _groupedPrompt = false;
                _promptStage = null;
            }
        }

        /// <summary>One layer: the PLE block where it sits, then both mixed halves. <paramref name="pos"/>
        /// is the device copy of the row's position a captured decode step reads (zero: p0 is used); in
        /// a captured batched step (<paramref name="rows"/> set) row r reads element r.</summary>
        /// <param name="partialsReady">The previous layer's last scatter already ran this layer's first mixer pass.</param>
        /// <returns>Whether this layer's last scatter ran the next layer's first mixer pass (the next layer is on this
        /// device and has no PLE block ahead of it).</returns>
        /// <param name="deferRow">At or past zero: a prompt group's ubatch whose routed experts and last scatter run later,
        /// over the whole group (<see cref="GroupExperts"/>).</param>
        private bool RunLayer(Dev dev, int il, Slot slot, int nt, int p0, Slot[] rows, int[] positions, IntPtr pos,
            bool partialsReady = false, int deferRow = -1)
        {
            var m = _m;
            var L = _layers[il];
            int e = m.NEmbd;
            if (L.Ple)
                Ple(dev, L, slot, nt, rows);

            IntPtr curQ = HcPre(dev, L.HcAttnNorm, L.HcAttnDown, L.HcAttnUp, L.HcAttnInject, nt, partialsReady: partialsReady && !L.Ple);
            StageEnd(dev, StHc);
            if (L.Recurrent)
                Gdn(dev, L, il, slot, nt, rows, curQ);
            else if (rows != null && BatchedAttentionFor(dev, L, il, rows, nt, pos))
                AttentionRows(dev, L, il, rows, nt, positions, pos, curQ);
            else if (rows != null)
                // A batched step's rows, a sequence's at a time: one row each in a batched decode step, a sequence's whole
                // window (consecutive positions) in a batched verify, issued as that sequence's own verify issues it. A
                // captured step's sequences read their rows' device positions.
                for (int r = 0; r < nt; r += _batchRowsPerSeq)
                    Attention(dev, L, il, rows[r].Layers[il], r, _batchRowsPerSeq, positions[r],
                        pos == IntPtr.Zero ? IntPtr.Zero : (IntPtr)((long)pos + 4L * r), curQ, nt);
            else
                Attention(dev, L, il, slot.Layers[il], 0, nt, p0, pos, curQ, nt);
            // Decode-size halves meet in one launch: the scatter and the next mixer's first pass.
            bool fuse = nt <= PerSlotMaxRows && dev.QK.HasHcPostPartials && _fuseHc;
            if (fuse)
                dev.QK.HcPostPartials(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), Ptr(L.HcFfnNorm), Ptr(L.HcFfnInject),
                    Ptr(dev.HcPartials), nt, e, m.Hc, dev.Stream);
            else
                dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, m.Hc, dev.Stream);

            curQ = HcPre(dev, L.HcFfnNorm, L.HcFfnDown, L.HcFfnUp, L.HcFfnInject, nt, partialsReady: fuse);
            StageEnd(dev, StHc);
            Moe(dev, L, il, nt, pos == IntPtr.Zero && rows == null ? p0 : -1, curQ, deferRow);
            if (deferRow >= 0)
                return false;
            var next = il + 1 < m.NLayer ? _layers[il + 1] : null;
            bool fuseNext = fuse && next != null && next.Device == dev.Ordinal && !next.Ple;
            if (fuseNext)
                dev.QK.HcPostPartials(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), Ptr(next.HcAttnNorm), Ptr(next.HcAttnInject),
                    Ptr(dev.HcPartials), nt, e, m.Hc, dev.Stream);
            else
                dev.QK.HcPost(Ptr(dev.Xs), Ptr(dev.BlockOut), Ptr(dev.Inject), nt, e, m.Hc, dev.Stream);
            StageEnd(dev, StHc);
            return fuseNext;
        }

        // TS_Q4E_FUSE_HC=0: the scatter and the next mixer's first pass as two launches (the fused kernel computes the
        // same values; this is for measuring it).
        private readonly bool _fuseHc = Environment.GetEnvironmentVariable("TS_Q4E_FUSE_HC") != "0";

        /// <summary>Hand the streams to the next device through pinned host memory. Events are
        /// recorded on their own context's stream; only the waits cross contexts.</summary>
        private void Handoff(Dev src, Dev dst, int nt)
        {
            var bytes = new UIntPtr((ulong)((long)nt * HcDim * 4));
            src.MakeCurrent();
            CudaDriverApi.cuMemcpyDtoHAsync(src.BoundaryPinned, Ptr(src.Xs), bytes, src.Stream).ThrowOnError();
            StageEnd(src, StBoundary);
            CudaDriverApi.cuEventRecord(src.XsReadyEv, src.Stream).ThrowOnError();
            dst.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(dst.Stream, src.XsReadyEv, 0).ThrowOnError();
            StageBegin(dst);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dst.Xs), src.BoundaryPinned, bytes, dst.Stream).ThrowOnError();
            CudaDriverApi.cuEventRecord(dst.CopyDoneEv, dst.Stream).ThrowOnError();
            // Nothing may overwrite the source streams or the pinned slice before the copy ran.
            src.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(src.Stream, dst.CopyDoneEv, 0).ThrowOnError();
        }

        /// <summary>The low-rank mixer: the streams grouped-RMS-normed into Xn (and the scatter
        /// logits into Inject), the sigmoid gate from the rank-R bottleneck, and the gated stream
        /// mean into Cur.</summary>
        /// <param name="partialsReady">The first pass already ran, fused into the previous half's scatter
        /// (<see cref="HcPost"/>).</param>
        /// <returns>Cur quantized to q8_1 (CurQ81) when the decode-size collapse quantized it, for the half's projections
        /// in place of <see cref="QuantizeShared"/>; else zero.</returns>
        private IntPtr HcPre(Dev dev, Tensor normW, in DeviceWeight down, in DeviceWeight up, Tensor injW, int nt,
            Tensor xs = null, bool partialsReady = false)
        {
            var m = _m;
            IntPtr xsPtr = Ptr(xs ?? dev.Xs), inj = injW == null ? IntPtr.Zero : Ptr(injW);
            if (nt <= PerSlotMaxRows)
            {
                // Decode size: the norm in two passes over every slice of the streams (one block per
                // row would leave all but one SM idle), each pass quantizing what the next projection
                // reads.
                if (!partialsReady)
                    dev.QK.HcPartials(xsPtr, Ptr(normW), inj, Ptr(dev.HcPartials), nt, m.NEmbd, m.Hc, dev.Stream);
                dev.QK.HcApply(xsPtr, Ptr(normW), Ptr(dev.HcPartials), Ptr(dev.Xn), Ptr(dev.SharedQ81),
                    injW == null ? IntPtr.Zero : Ptr(dev.Inject), nt, m.NEmbd, m.Hc, m.RmsEps, dev.Stream);
                MatMul(dev, down, dev.Xn, Ptr(dev.SharedQ81), dev.Lo, nt);
                dev.QK.SiluQ81(Ptr(dev.Lo), Ptr(dev.SharedQ81), (long)nt * m.HcLowRank, 1.0f / m.Hc, dev.Stream);
                MatMul(dev, up, dev.Lo, Ptr(dev.SharedQ81), dev.G, nt);
                if (_collapseQ81 && dev.CurQ81 != null && dev.QK.HasHcCollapseQ81)
                {
                    dev.QK.HcCollapseQ81(Ptr(dev.Xn), Ptr(dev.G), Ptr(dev.Cur), Ptr(dev.CurQ81), nt, m.NEmbd, m.Hc, dev.Stream);
                    return Ptr(dev.CurQ81);
                }
            }
            else
            {
                dev.QK.HcNorm(xsPtr, Ptr(normW), inj, Ptr(dev.Xn), Ptr(dev.Inject), nt, m.NEmbd, m.Hc, m.RmsEps, dev.Stream);
                MatMulQ(dev, down, dev.Xn, dev.Lo, nt);
                dev.QK.SiluScale(Ptr(dev.Lo), (long)nt * m.HcLowRank, 1.0f / m.Hc, dev.Stream);
                MatMulQ(dev, up, dev.Lo, dev.G, nt);
            }
            dev.QK.HcCollapse(Ptr(dev.Xn), Ptr(dev.G), Ptr(dev.Cur), nt, m.NEmbd, m.Hc, dev.Stream);
            return IntPtr.Zero;
        }

        /// <summary>
        /// Whether a single sequence's <paramref name="nt"/> prompt rows take the chunked recurrence (Q4eKernels.GdnChunked)
        /// instead of KdaScan's token-by-token walk: 128 rows or more (below that its two launches cost more than they save,
        /// as Strata measured), three value heads a key head, and a GPU whose SMs hold every scan block in one wave
        /// (16 key heads x 8 column tiles: 128 SMs; with fewer it runs in waves and loses). TS_Q4E_GDN_CHUNKED=0 keeps KdaScan.
        /// </summary>
        private bool GdnChunkedFor(Dev dev, int nt)
            => _gdnChunked && nt >= GdnChunkedMinRows && dev.QK.HasGdnChunked && _m.GdnVHeads == 3 * _m.GdnKHeads
               && dev.QK.GdnChunkedSms >= _m.GdnKHeads * 8;

        internal const int GdnChunkedMinRows = 128;
        private readonly bool _gdnChunked = Environment.GetEnvironmentVariable("TS_Q4E_GDN_CHUNKED") != "0";

        /// <summary>The chunked recurrence's scratch on <paramref name="dev"/>, allocated on first use (~25 MB for Flash Next);
        /// null once an allocation failed, so the prompt keeps KdaScan.</summary>
        private Tensor GdnChunkScratch(Dev dev)
        {
            if (dev.GdnChunkScr != null || dev.GdnChunkUnavailable)
                return dev.GdnChunkScr;
            try { dev.GdnChunkScr = AllocF32(dev, Q4eKernels.GdnChunkScratchFloats(_m.GdnVHeads)); }
            catch (Exception ex) when (ex is OutOfMemoryException or CudaException or InvalidOperationException)
            {
                dev.GdnChunkUnavailable = true;
                Console.Error.WriteLine($"[q4e-cuda] the chunked prompt recurrence's scratch did not fit ({ex.Message}); prompts keep the token-by-token scan.");
            }
            return dev.GdnChunkScr;
        }

        // TS_Q4E_COLLAPSE_Q81=0: the decode-size collapse leaves Cur's q8_1 to each half's projections (a launch of its own,
        // the same values; this is for measuring it).
        private readonly bool _collapseQ81 = Environment.GetEnvironmentVariable("TS_Q4E_COLLAPSE_Q81") != "0";

        /// <param name="curQ">Cur already quantized by the mixer's collapse (<see cref="HcPre"/>), or zero.</param>
        private void Gdn(Dev dev, DevLayer L, int il, Slot slot, int nt, Slot[] rows, IntPtr curQ = default)
        {
            var m = _m;
            IntPtr cur = curQ != IntPtr.Zero ? curQ : QuantizeShared(dev, dev.Cur, m.NEmbd, nt);
            MatMul(dev, L.Qkv, dev.Cur, cur, dev.Qkv, nt);
            MatMul(dev, L.Gate, dev.Cur, cur, dev.Z, nt);
            MatMulF32(dev, L.SsmAlpha, dev.Cur, dev.Alpha, m.NEmbd, m.GdnVHeads, nt);
            MatMulF32(dev, L.SsmBeta, dev.Cur, dev.Beta, m.NEmbd, m.GdnVHeads, nt);
            StageEnd(dev, StGdnProj);

            // One sequence's nt rows in its own state, or the rows of each batched sequence in theirs (one a sequence in a
            // batched decode step, a window a sequence in a batched verify).
            IntPtr conv, ssm;
            int rowsPerSeq = nt, nSeq = 1;
            if (rows == null)
            {
                conv = StatePtr(slot, dev, il, 0);
                ssm = StatePtr(slot, dev, il, 1);
            }
            else
            {
                conv = BatchStatePtr(dev, il, 0);
                ssm = BatchStatePtr(dev, il, 1);
                rowsPerSeq = _batchRowsPerSeq;
                nSeq = nt / _batchRowsPerSeq;
            }
            bool capture = _verifyCapture && rows == null;
            if (capture)
                VerifySaveGdnState(dev, il, slot);
            dev.QK.GdnPrep(Ptr(dev.Qkv), conv, Ptr(L.ConvW), Ptr(dev.Alpha), Ptr(dev.Beta), Ptr(L.DtBias), Ptr(L.SsmA),
                Ptr(dev.Scr), rowsPerSeq, nSeq, m.GdnKHeads, m.GdnVHeads, m.DConv, m.RmsEps, dev.Stream);
            if (capture)
                VerifySaveGdnInputs(dev, il, nt);
            else if (_batchVerifyCapture && rows != null)
                BatchVerifySaveGdnInputs(dev, il, nt);
            // After the prep read the previous inputs.
            dev.QK.ConvUpdate(Ptr(dev.Qkv), conv, rowsPerSeq, nSeq, ConvDim, m.DConv - 1, dev.Stream);
            if (rows == null && !capture && GdnChunkedFor(dev, nt) && GdnChunkScratch(dev) is Tensor gs)
            {
                long tp = Q4eKernels.GdnChunkTPFloats(m.GdnVHeads) * 4;
                IntPtr tm = Ptr(gs);
                dev.QK.GdnChunked(Ptr(dev.Scr), ssm, Ptr(dev.Core), tm, (IntPtr)((long)tm + tp), (IntPtr)((long)tm + 2 * tp),
                    nt, m.GdnKHeads, m.GdnVHeads, dev.Stream);
            }
            else
                dev.GK.KdaScan(Ptr(dev.Scr), ssm, Ptr(dev.Core), rowsPerSeq, nSeq, m.GdnVHeads, dev.Stream);
            dev.GK.KdaOut(Ptr(dev.Core), Ptr(dev.Z), Ptr(L.SsmNorm), Ptr(dev.GOut), nt, m.GdnVHeads, m.RmsEps, dev.Stream);
            StageEnd(dev, StGdnCore);
            MatMulQ(dev, L.SsmOut, dev.GOut, dev.BlockOut, nt);
            StageEnd(dev, StAttnOut);
        }

        /// <summary>Gated attention for rows [row0, row0 + nt) of the ubatch, one sequence's rows at
        /// p0, p0 + 1, ... against its caches <paramref name="c"/>. A batched decode step calls it
        /// per row; the projections then run for every row on the first call.</summary>
        /// <param name="curQ">Cur's <paramref name="curQRows"/> rows already quantized by the mixer's collapse
        /// (<see cref="HcPre"/>), or zero.</param>
        private void Attention(Dev dev, DevLayer L, int il, SlotLayer c, int row0, int nt, int p0, IntPtr pos = default,
            IntPtr curQ = default, int curQRows = 0)
        {
            var m = _m;
            int qd = m.NHead * m.HeadDim, kvd = m.NKvHead * m.HeadDim;
            int allRows = row0 == 0 ? RowsInFlight(nt) : 0;
            if (allRows > 0)
            {
                IntPtr cur = curQ != IntPtr.Zero && curQRows == allRows ? curQ : QuantizeShared(dev, dev.Cur, m.NEmbd, allRows);
                MatMul(dev, L.Wq, dev.Cur, cur, dev.Qg, allRows);
                MatMul(dev, L.Wk, dev.Cur, cur, dev.K, allRows);
                MatMul(dev, L.Wv, dev.Cur, cur, dev.V, allRows);
                if (L.Ratio > 0)
                    MatMulF32(dev, L.IdxK, dev.Cur, dev.IdxRaw, m.NEmbd, m.IdxDim, allRows);
                StageEnd(dev, StAttnProj);
            }

            // A streamed layer (TS_Q4E_KV_RESIDENT): decode-size calls read and write the window through its page table, a
            // prompt runs on the staging pool (the context's cells copied in from the host copy).
            var kv = c.Kv;
            bool paged = kv != null && nt <= PerSlotMaxRows;
            bool staged = kv != null && !paged;
            IntPtr kCache = Ptr(c.KCache), vCache = Ptr(c.VCache), kScale = PtrOrZero(c.KScale), vScale = PtrOrZero(c.VScale);
            if (staged)
            {
                KvStageForPrompt(dev, c, p0);
                kCache = Ptr(dev.KvStageK);
                vCache = Ptr(dev.KvStageV);
                kScale = Ptr(dev.KvStageKs);
                vScale = Ptr(dev.KvStageVs);
            }
            if (paged)
            {
                if (ReferenceEquals(dev.KvStagedFor, c))
                    dev.KvStagedFor = null;
                dev.QK.AttnPrepPaged(At(dev.Qg, (long)row0 * 2 * qd), At(dev.K, (long)row0 * kvd), At(dev.V, (long)row0 * kvd),
                    Ptr(L.QNorm), Ptr(L.KNorm), At(dev.Q, (long)row0 * qd), At(dev.Gate, (long)row0 * qd),
                    kCache, vCache, kScale, vScale, pos, p0, m.NHead, m.NKvHead, m.NRot, m.RopeBase, m.RopeFreqScale,
                    m.RmsEps, nt, Ptr(kv.Table), kv.DevK, kv.DevV, kv.DevKs, kv.DevVs, dev.Stream);
            }
            else
            {
                dev.QK.AttnPrep(At(dev.Qg, (long)row0 * 2 * qd), At(dev.K, (long)row0 * kvd), At(dev.V, (long)row0 * kvd),
                    Ptr(L.QNorm), Ptr(L.KNorm), At(dev.Q, (long)row0 * qd), At(dev.Gate, (long)row0 * qd),
                    kCache, vCache, pos, p0, m.NHead, m.NKvHead, m.NRot, m.RopeBase, m.RopeFreqScale,
                    m.RmsEps, nt, dev.Stream, kScale, vScale);
                if (staged)
                    KvUnstagePrompt(dev, c, p0, nt);
            }

            IntPtr cells = IntPtr.Zero, cellCnt = IntPtr.Zero;
            if (L.Ratio > 0)
            {
                int R = L.Ratio, d = m.IdxDim;
                if (c.IdxRing > 0)
                    dev.QK.QsaStoreRing(At(dev.IdxRaw, (long)row0 * d), Ptr(c.IdxRaw), pos, p0, d, nt, c.IdxRing, dev.Stream);
                else
                    dev.QK.QsaStore(At(dev.IdxRaw, (long)row0 * d), Ptr(c.IdxRaw), pos, p0, d, nt, dev.Stream);
                // The blocks this ubatch completes (their last member now cached).
                int firstBlock = p0 / R, endBlock = (p0 + nt) / R;
                if (pos == IntPtr.Zero || nt == 1)
                    QsaPool(dev, L, c, pos, firstBlock, endBlock - firstBlock);
                else
                    // A captured window: the pooling kernel reads one row's position, so each row pools the block it
                    // completes, if it completes one.
                    for (int t = 0; t < nt; t++)
                        QsaPool(dev, L, c, (IntPtr)((long)pos + 4L * t), 0, 0);
                // Past the width the selection bites: score the complete blocks, keep the best.
                if (p0 + nt > m.IdxTopK + R - 1)
                {
                    IntPtr qIdx = At(dev.IdxQ, (long)row0 * m.IdxHeads * d);
                    using (Tensor rowsIn = Block(dev.Cur, row0, nt, m.NEmbd))
                    using (Tensor rowsOut = Block(dev.IdxQ, row0, nt, (long)m.IdxHeads * d))
                        MatMulF32(dev, L.IdxQ, rowsIn, rowsOut, m.NEmbd, m.IdxHeads * d, nt);
                    dev.QK.QsaQuery(qIdx, Ptr(L.IdxQNorm), pos, p0, m.IdxHeads, d, m.NRot, m.RopeBase, m.RopeFreqScale,
                        m.RmsEps, nt, dev.Stream);
                    IntPtr scores = At(dev.Scores, (long)row0 * ScoreStride);
                    cells = (IntPtr)((long)Ptr(dev.Cells) + (long)row0 * CellStride * 4);
                    cellCnt = (IntPtr)((long)Ptr(dev.CellCnt) + (long)row0 * 4);
                    // A captured step's grid covers every block the context can hold (its blocks past
                    // the row's position exit at once).
                    if (pos == IntPtr.Zero && QsaScoresMmaFor(dev, nt))
                        dev.QK.QsaScoresMma(qIdx, Ptr(c.IdxPooled), scores, p0, d, R, ScoreStride, (p0 + nt) / R, nt, dev.Stream);
                    else
                        dev.QK.QsaScores(qIdx, Ptr(c.IdxPooled), scores, pos, p0, m.IdxHeads, d, R, ScoreStride,
                            pos != IntPtr.Zero ? ScoreStride : (p0 + nt) / R, nt, dev.Stream);
                    dev.QK.QsaSelect(scores, cells, cellCnt, pos, p0, R, m.IdxTopK, ScoreStride, CellStride, nt, dev.Stream);
                    if (pos == IntPtr.Zero)
                        SparseProbe?.Invoke(CaptureSparse(dev, L, il, c, row0, nt, p0));
                }
            }
            if (paged)
            {
                // Every page the rows read made resident, the misses copied over the bus.
                dev.QK.KvResolve(cells, cellCnt, pos, p0, nt, CellStride, Ptr(kv.Table), Ptr(kv.SlotPage), Ptr(kv.SlotStamp),
                    Ptr(kv.Ctl), Ptr(kv.MissPage), Ptr(kv.MissSlot), kv.Slots, dev.Stream);
                dev.QK.KvPageCopy(Ptr(kv.Ctl), Ptr(kv.MissPage), Ptr(kv.MissSlot), kv.DevK, kv.DevV, kv.DevKs, kv.DevVs,
                    kCache, vCache, kScale, vScale, m.NKvHead, dev.Stream);
            }
            StageEnd(dev, StAttnPrep);

            int splits = AttnSplits(dev, il, nt, p0);
            if (paged)
            {
                if (splits > 0)
                    dev.QK.AttentionSplitPaged(At(dev.Q, (long)row0 * qd), kCache, vCache, kScale, vScale, At(dev.Gate, (long)row0 * qd),
                        cells, cellCnt, At(dev.AttnO, (long)row0 * qd), pos, p0, m.NHead, m.NKvHead, m.AttnScale,
                        CellStride, nt, splits, Ptr(dev.AttnPart), Ptr(dev.AttnArrive), Ptr(kv.Table), dev.Stream);
                else
                    dev.QK.AttentionPaged(At(dev.Q, (long)row0 * qd), kCache, vCache, kScale, vScale, At(dev.Gate, (long)row0 * qd),
                        cells, cellCnt, At(dev.AttnO, (long)row0 * qd), pos, p0, m.NHead, m.NKvHead, m.AttnScale,
                        CellStride, nt, Ptr(kv.Table), dev.Stream);
            }
            else if (splits == 0 && AttnMmaFor(dev, nt))
            {
                dev.QK.AttentionMma(At(dev.Q, (long)row0 * qd), kCache, vCache, At(dev.Gate, (long)row0 * qd),
                    cells, cellCnt, At(dev.AttnO, (long)row0 * qd), pos, p0, m.NHead, m.NKvHead, m.AttnScale,
                    CellStride, nt, dev.Stream, kScale, vScale);
                PromptAttentionMmaCalls++;
            }
            else if (splits > 0)
                dev.QK.AttentionSplit(At(dev.Q, (long)row0 * qd), kCache, vCache, At(dev.Gate, (long)row0 * qd),
                    cells, cellCnt, At(dev.AttnO, (long)row0 * qd), pos, p0, m.NHead, m.NKvHead, m.AttnScale,
                    CellStride, nt, splits, Ptr(dev.AttnPart), Ptr(dev.AttnArrive), dev.Stream, kScale, vScale);
            else
                dev.QK.Attention(At(dev.Q, (long)row0 * qd), kCache, vCache, At(dev.Gate, (long)row0 * qd),
                    cells, cellCnt, At(dev.AttnO, (long)row0 * qd), pos, p0, m.NHead, m.NKvHead, m.AttnScale,
                    CellStride, nt, dev.Stream, kScale, vScale);
            StageEnd(dev, StAttnCore);

            if (row0 + nt == RowsInFlight(nt))
            {
                MatMulQ(dev, L.Wo, dev.AttnO, dev.BlockOut, row0 + nt);
                StageEnd(dev, StAttnOut);
            }
        }

        /// <summary>The QSA keys of blocks [first, first + count) from the layer's raw keys (a ring on a streamed layer); with
        /// <paramref name="pos"/>, the block the row at pos[0] completes.</summary>
        private void QsaPool(Dev dev, DevLayer L, SlotLayer c, IntPtr pos, int first, int count)
        {
            var m = _m;
            if (c.IdxRing > 0)
                dev.QK.QsaPoolRing(Ptr(c.IdxRaw), Ptr(L.IdxKNorm), Ptr(c.IdxPooled), pos, first, count, L.Ratio, m.IdxDim,
                    m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, c.IdxRing, dev.Stream);
            else
                dev.QK.QsaPool(Ptr(c.IdxRaw), Ptr(L.IdxKNorm), Ptr(c.IdxPooled), pos, first, count, L.Ratio, m.IdxDim,
                    m.NRot, m.RopeBase, m.RopeFreqScale, m.RmsEps, dev.Stream);
        }

        // TS_Q4E_ATTN_SPLITS: blocks a decode-size row's attention cells split over (flash decoding; 1 or 0 keeps one
        // block per head and row). A trunk layer's split count is fixed (a captured step's grid cannot change with the
        // position); the MTP block, never captured, sizes it from its context.
        private readonly int _attnSplits = Math.Clamp(EnvInt("TS_Q4E_ATTN_SPLITS", 16), 0, MaxAttnSplits);
        private const int MaxAttnSplits = 32;
        private bool AttnSplitsEnabled => _attnSplits > 1;

        /// <summary>Blocks a decode-size attention call splits its rows' cells over, or 0 for the one-block kernel.</summary>
        private int AttnSplits(Dev dev, int il, int nt, int p0)
        {
            if (dev.AttnPart == null || nt > PerSlotMaxRows)
                return 0;
            if (il >= 0)
                return _attnSplits;
            // The MTP block's dense rows: about 128 cells a block.
            int cellsMax = p0 + nt;
            return Math.Clamp((cellsMax + 127) / 128, 1, MaxAttnSplits);
        }

        // TS_Q4E_ATTN_MMA=0: prompt-size attention keeps the one-block-per-(head, row) kernel.
        private readonly bool _attnMma = Environment.GetEnvironmentVariable("TS_Q4E_ATTN_MMA") != "0";

        // TS_Q4E_QSA_SCORES_MMA=0: a prompt's QSA block scores keep the warp-per-block kernel.
        private readonly bool _qsaScoresMma = Environment.GetEnvironmentVariable("TS_Q4E_QSA_SCORES_MMA") != "0";

        /// <summary>Whether a QSA layer's block scores for <paramref name="nt"/> prompt rows run on tensor cores
        /// (Q4eKernels.QsaScoresMma): a prompt-size call (decode steps, verify windows and batched steps keep the
        /// warp-per-block kernel, so a row scores as its own decode step does), the kernel built for this GPU, and four
        /// indexer heads of a width in 64s.</summary>
        private bool QsaScoresMmaFor(Dev dev, int nt)
            => _qsaScoresMma && nt > PerSlotMaxRows && dev.QK.HasQsaScoresMma && Q4eKernels.QsaScoresMmaSupports(_m.IdxHeads, _m.IdxDim);

        /// <summary>Attention calls that ran on tensor cores so far (a test proves the path engaged).</summary>
        internal long PromptAttentionMmaCalls { get; private set; }

        /// <summary>Whether an attention call of <paramref name="nt"/> rows runs on tensor cores: a prompt-size call (decode
        /// steps and verify windows keep their kernels, so a row's answer does not depend on the window), the kernel built
        /// for this GPU, and a query-head group of at most 16.</summary>
        private bool AttnMmaFor(Dev dev, int nt)
        {
            if (!_attnMma || nt <= PerSlotMaxRows)
                return false;
            if (dev.QK.HasAttentionMma && _m.NHead % _m.NKvHead == 0 && _m.NHead / _m.NKvHead <= Q4eKernels.AttnMmaMaxGroup)
                return true;
            if (!_attnMmaDeclineWarned)
            {
                _attnMmaDeclineWarned = true;
                Console.Error.WriteLine(dev.QK.HasAttentionMma
                    ? $"[q4e-cuda] prompt attention runs on the CUDA cores: {_m.NHead} query heads over {_m.NKvHead} KV heads exceed the "
                      + $"tensor-core kernel's {Q4eKernels.AttnMmaMaxGroup}-row group (prompts are slower; answers unchanged)."
                    : "[q4e-cuda] prompt attention runs on the CUDA cores: the loaded tensorsharp_q4e_kernels module has no tensor-core "
                      + "attention for this GPU (needs sm_80+ and a module built for it, not the compute_75 baseline); prompts are slower.");
            }
            return false;
        }

        private bool _attnMmaDeclineWarned;

        // The rows of the step in flight: a batched decode step sets it before its layers run.
        private int _rowsInFlight;

        private int RowsInFlight(int nt) => _rowsInFlight > 0 ? _rowsInFlight : nt;

        /// <summary>This step's n-gram rows into the PLE device's pinned staging: the tokens join their
        /// sequences' histories, then the host hashes and dequantizes.</summary>
        private void GatherPleRows(Slot slot, int[] tokens, int tokOff, int nt, int p0, Slot[] rows, int[] positions)
        {
            var histories = new Qwen4ExpPleHistory[nt];
            var pos = new int[nt];
            if (rows == null)
                slot.PleTokens.Put(new ReadOnlySpan<int>(tokens, tokOff, nt), p0);
            for (int t = 0; t < nt; t++)
            {
                if (rows != null)
                    rows[t].PleTokens.Put(new ReadOnlySpan<int>(tokens, tokOff + t, 1), positions[t]);
                histories[t] = rows == null ? slot.PleTokens : rows[t].PleTokens;
                pos[t] = rows == null ? p0 + t : positions[t];
            }
            var dev = _devs[_layers[_pleLayer].Device];
            dev.MakeCurrent();
            // The staging is reused per ubatch: the previous ubatch's upload must have run (a decode lane's own staging
            // was last read by its previous step, finished already).
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!_laneIssue)
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
            _m.Ple.GatherPleRows(histories, new ReadOnlySpan<int>(tokens, tokOff, nt), pos, (float*)dev.PlePinned);
            _fwdWaitTicks += t1 - t0;
            _fwdPleTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t1;
            _devs[0].MakeCurrent();
        }

        private void Ple(Dev dev, DevLayer L, Slot slot, int nt, Slot[] rows)
        {
            var m = _m;
            int e = m.NEmbd, hcE = HcDim;
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PleEmb), dev.PlePinned, new UIntPtr((ulong)nt * (ulong)e * 4), dev.Stream).ThrowOnError();

            IntPtr emb = QuantizeShared(dev, dev.PleEmb, e, nt);
            MatMul(dev, L.PleKey, dev.PleEmb, emb, dev.PleKeyOut, nt);
            MatMul(dev, L.PleValue, dev.PleEmb, emb, dev.PleValueOut, nt);
            dev.QK.PleGate(Ptr(dev.PleKeyOut), Ptr(dev.Xs), Ptr(L.PleNormKey), Ptr(L.PleNormQuery), Ptr(dev.PleValueOut),
                Ptr(dev.PleGated), nt, e, m.Hc, m.RmsEps, dev.Stream);
            dev.QK.GroupNorm(Ptr(dev.PleGated), Ptr(L.PleNormConv), Ptr(dev.PleNorm), nt, e, m.Hc, m.RmsEps, dev.Stream);
            IntPtr hists;
            int rowsPerSeq = nt, nSeq = 1;
            if (rows == null)
            {
                hists = StatePtr(slot, dev, m.NLayer, 0);
            }
            else
            {
                hists = BatchStatePtr(dev, m.NLayer, 0);
                rowsPerSeq = _batchRowsPerSeq;
                nSeq = nt / _batchRowsPerSeq;
            }
            dev.QK.PleConv(Ptr(dev.PleGated), Ptr(dev.PleNorm), hists, Ptr(L.PleConvT), Ptr(dev.Xs), rowsPerSeq, nSeq, hcE,
                m.PleConvKernel, m.PleDilation, dev.Stream);
            if (PleHistRows > 0)
            {
                if (_verifyCapture && rows == null)
                    VerifySavePle(dev, slot, nt);
                else if (_batchVerifyCapture && rows != null)
                    BatchVerifySavePle(dev, nt);
                dev.QK.ConvUpdate(Ptr(dev.PleNorm), hists, rowsPerSeq, nSeq, hcE, PleHistRows, dev.Stream);
            }
            StageEnd(dev, StPle);
        }

        /// <param name="probeP0">The position of the ubatch's first row on the single-sequence uncaptured
        /// path, where <see cref="RouterProbe"/> may read the router back; -1 elsewhere.</param>
        /// <param name="curQ">Cur already quantized by the mixer's collapse (<see cref="HcPre"/>), or zero: the shared
        /// expert's and the routed experts' input.</param>
        /// <param name="groupRow">At or past zero: a prompt group's ubatch (rows <paramref name="groupRow"/> of the group) whose
        /// routed experts run later over the whole group (<see cref="DeferExperts"/>, <see cref="GroupExperts"/>).</param>
        private void Moe(Dev dev, DevLayer L, int il, int nt, int probeP0, IntPtr curQ = default, int groupRow = -1)
        {
            var m = _m;
            int e = m.NEmbd;
            // A decode-size step runs the shared expert beside the router and the routed experts (Strata's forked
            // shared-expert stream): it reads only the mixed input, and its output is first read by the routed experts'
            // combine, which waits for it. A captured step holds both branches.
            bool fork = dev.ShStream != IntPtr.Zero && nt <= PerSlotMaxRows && _perf < 2
                && DecodeKernelFor(dev, L.GateShexp, nt) && DecodeKernelFor(dev, L.UpShexp, nt) && DecodeKernelFor(dev, L.DownShexp, nt);
            if (fork)
            {
                CudaDriverApi.cuEventRecord(dev.ShForkEv, dev.Stream).ThrowOnError();
                CudaDriverApi.cuStreamWaitEvent(dev.ShStream, dev.ShForkEv, 0).ThrowOnError();
                SharedExpert(dev, L, nt, dev.ShStream, curQ);
                CudaDriverApi.cuEventRecord(dev.ShJoinEv, dev.ShStream).ThrowOnError();
            }
            if (L.RouterBf16 != null && nt <= Dsv4Kernels.GemvMaxRows)
                dev.DK.GemvBf16(Ptr(dev.Cur), Ptr(L.RouterBf16), Ptr(dev.RouterLogits), e, m.NExpert, nt, dev.Stream);
            else
                MatMulF32(dev, L.Router ?? FloatRouter(dev, L), dev.Cur, dev.RouterLogits, e, m.NExpert, nt);
            dev.DK.MoeSelect(dev.RouterLogits, IntPtr.Zero, IntPtr.Zero, null, dev.Moe.Sel, dev.Moe.SelW,
                m.NExpert, m.NExpertUsed, 1, 1.0f, nt, dev.Stream, Dsv4Kernels.RouterSoftmax);
            if (probeP0 >= 0)
                RouterProbe?.Invoke(CaptureRouter(dev, il, probeP0, nt));
            StageEnd(dev, StRouter);
            bool host = HostExpertsFor(L, nt);
            // A prompt's selections teach adaptive residency what its answer will route (AdaptAfterPrompt).
            if (!host && il >= 0 && nt > _hostExpertRows && _promptCountsDev != IntPtr.Zero && L.Stage.IsValid)
                dev.QK.CountSelected(Ptr(dev.Moe.Sel), nt * m.NExpertUsed, (IntPtr)((long)_promptCountsDev + 4L * il * m.NExpert), dev.Stream);
            // The GPU's share of the misses: planned by the doorbell (which leaves it out of the host's), copied over the bus
            // after it, so the host starts at once.
            bool hybrid = host && _hybridStage != null && _doorbellMiss != IntPtr.Zero && L.Stage.IsValid && L.Stage.Staged;
            if (hybrid)
            {
                dev.DK.MoeDoorbellPublishHybrid(dev.Cur, dev.Moe.Sel, dev.Moe.SelW, L.Stage, nt * e, nt * m.NExpertUsed, il,
                    _hostBox, dev.Stream, _doorbellMiss, _adaptCountsDev, m.NExpert, m.NExpertUsed, _tailRank, _hybridStage, _hybridOdd);
                dev.DK.MoeStageCopy(L.Stage, _hybridStage, _hybridStage.Capacity, dev.Stream);
            }
            else if (host)
                dev.DK.MoeDoorbellPublish(dev.Cur, dev.Moe.Sel, dev.Moe.SelW, L.Stage.IsHost, nt * e, nt * m.NExpertUsed, il,
                    _hostBox, dev.Stream, _doorbellMiss, _adaptCountsDev, m.NExpert, m.NExpertUsed, _tailRank);

            if (!fork)
            {
                SharedExpert(dev, L, nt, dev.Stream, curQ);
                StageEnd(dev, StShexp);
            }

            if (groupRow >= 0)
            {
                DeferExperts(dev, L, nt, groupRow);
                StageEnd(dev, StExperts);
                return;
            }

            CudaMoe.Experts(dev.DK, dev.Alloc.Kernels, dev.Moe, L.GateExps, L.UpExps, L.DownExps,
                dev.Cur, dev.ShDown, dev.BlockOut, nt, m.NExpertUsed, m.NExpert, e, m.NFfExp, 0f,
                PerSlotMaxRows, dev.Stream, stage: host ? null : _promptStage ?? _expertStage, stageLayer: L.Stage,
                shDownReady: fork ? dev.ShJoinEv : IntPtr.Zero, keepStaged: _groupedPrompt, curQ81: curQ,
                q80VecDecode: ExpertVecDecode);
            if (host)
            {
                // The GPU's own experts end here; what follows is the wait for the CPU's share.
                StageEnd(dev, StExperts);
                dev.DK.MoeDoorbellWaitAdd(_hostBox, dev.BlockOut, nt * e, dev.Stream, _doorbellMiss);
                if (hybrid)
                    dev.DK.MoeUnstage(L.Stage, _hybridStage, dev.Stream);
                StageEnd(dev, StHostWait);
            }
            else
                StageEnd(dev, StExperts);
        }

        /// <summary>A prompt's router weights as F32 (cuBLAS reads them so) when the layer keeps them as BF16: widened
        /// into the device's scratch, which every layer of the ubatch reuses in turn.</summary>
        private Tensor FloatRouter(Dev dev, DevLayer L)
        {
            long n = (long)_m.NExpert * _m.NEmbd;
            dev.RouterF32 ??= AllocF32(dev, n);
            dev.QK.Bf16ToF32(Ptr(L.RouterBf16), Ptr(dev.RouterF32), n, dev.Stream);
            return dev.RouterF32;
        }

        /// <summary>The shared expert on <paramref name="stream"/>: gate and up from one quantized input, SwiGLU, down, and
        /// the sigmoid gate, into ShDown.</summary>
        private void SharedExpert(Dev dev, DevLayer L, int nt, IntPtr stream, IntPtr curQ = default)
        {
            var m = _m;
            int e = m.NEmbd;
            IntPtr cur = curQ != IntPtr.Zero ? curQ : QuantizeShared(dev, dev.Cur, e, nt, stream);
            MatMul(dev, L.GateShexp, dev.Cur, cur, dev.ShGate, nt, stream);
            MatMul(dev, L.UpShexp, dev.Cur, cur, dev.ShUp, nt, stream);
            dev.Alloc.Kernels.LaunchSiluMulClampF32(Ptr(dev.ShGate), Ptr(dev.ShGate), Ptr(dev.ShUp), (long)nt * m.NFfShexp, 0f, stream);
            MatMul(dev, L.DownShexp, dev.ShGate, QuantizeShared(dev, dev.ShGate, m.NFfShexp, nt, stream), dev.ShDown, nt, stream);
            dev.QK.SharedGate(Ptr(dev.Cur), Ptr(L.ShexpGate), Ptr(dev.ShDown), nt, e, stream);
        }

        /// <summary>The output mixer, then the LM head: the last row's logits, or every row's (a
        /// batched decode step). <paramref name="hOut"/> takes the rows' hidden streams (before the mixer)
        /// from row <paramref name="hRowOffset"/> on, or the last row's alone when that is -1 or they do not fit.</summary>
        private void Head(int nt, float[] logitsOut, bool allRows, float[] hOut = null, int hRowOffset = 0)
        {
            IssueHead(nt, logitsOut != null, allRows, hOut != null);
            // When the host had queued the step: telemetry's issue/wait split, and TS_Q4E_PERF.
            _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(nt, logitsOut, allRows, hOut, hRowOffset);
        }

        /// <summary>Floats of a row's hidden streams (<see cref="Forward(int[], float[], float[])"/>).</summary>
        public int HiddenWidth => HcDim;

        /// <summary>The head's kernels and the copies of its results into pinned memory.</summary>
        /// <param name="greedy">Each row's first-max token goes to pinned memory instead of its logits (a greedy verify).</param>
        private void IssueHead(int nt, bool logits, bool allRows, bool hidden, bool greedy = false)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            if (hidden)
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedHidden, Ptr(dev.Xs), new UIntPtr((ulong)nt * (ulong)HcDim * 4), dev.Stream).ThrowOnError();
            if (!logits)
                return;
            int normRows = allRows ? nt : 1;
            using (Tensor xs = normRows == nt ? dev.Xs.CopyRef() : Block(dev.Xs, nt - 1, 1, HcDim))
                HcPre(dev, _outHcNorm, _outHcDown, _outHcUp, null, normRows, xs);
            int headRows = allRows ? nt : 1;
            long count = (long)headRows * m.NVocab;
            {
                Tensor output = allRows ? dev.BatchLogits : dev.Logits;
                using (Tensor headIn = headRows == normRows ? dev.Cur.CopyRef() : Block(dev.Cur, nt - 1, 1, m.NEmbd))
                    MatMulQ(dev, _output, headIn, output, headRows);
                StageEnd(dev, StHead);
                if (greedy)
                {
                    dev.QK.ArgmaxRows(Ptr(output), Ptr(dev.ArgIds), headRows, m.NVocab, dev.Stream);
                    CudaDriverApi.cuMemcpyDtoHAsync(_pinnedIds, Ptr(dev.ArgIds), new UIntPtr((ulong)headRows * 4), dev.Stream).ThrowOnError();
                }
                else
                    CudaDriverApi.cuMemcpyDtoHAsync(_pinnedLogits, Ptr(output), new UIntPtr((ulong)count * 4), dev.Stream).ThrowOnError();
            }
        }

        /// <summary>Wait for the head and copy its results out of pinned memory.</summary>
        /// <param name="idsOut">A greedy verify's rows' tokens (the head ran with greedy), else null.</param>
        private void FinishHead(int nt, float[] logitsOut, bool allRows, float[] hOut, int hRowOffset = 0, int[] idsOut = null)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            long count = (long)(allRows ? nt : 1) * m.NVocab;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            if (idsOut != null)
                Marshal.Copy(_pinnedIds, idsOut, 0, nt);
            if (logitsOut != null)
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy((void*)_pinnedLogits, dst, logitsOut.LongLength * 4L, count * 4L);
            if (hOut != null)
            {
                long row = HcDim, all = nt * row;
                fixed (float* dst = hOut)
                {
                    if (hRowOffset >= 0 && hOut.LongLength >= (hRowOffset + nt) * row)
                        Buffer.MemoryCopy((void*)_pinnedHidden, dst + hRowOffset * row, (hOut.LongLength - hRowOffset * row) * 4L, all * 4L);
                    else
                        Buffer.MemoryCopy((float*)_pinnedHidden + all - row, dst, hOut.LongLength * 4L, row * 4L);
                }
            }
            _stages.Report();
        }

        // ---------------------------------------------------------------- test probe

        /// <summary>Test hook: each QSA layer's sparse selection on the single-sequence path, read
        /// back once its selection ran.</summary>
        internal Action<SparseLayerProbe> SparseProbe { get; set; }

        /// <summary>One QSA layer's indexer inputs and selection for a ubatch.</summary>
        internal sealed class SparseLayerProbe
        {
            public int Layer, P0, Nt, Ratio, TopK, IdxHeads, IdxDim, ScoreStride, CellStride, NRot;
            public float RopeBase, RopeFreqScale, Eps;
            /// <summary>The cached raw keys [P0 + Nt, IdxDim], as stored (F16).</summary>
            public System.Half[] RawKeys;
            /// <summary>The block keys [(P0 + Nt) / Ratio, IdxDim] and the normed, rotated queries [Nt, IdxHeads, IdxDim].</summary>
            public float[] Pooled, Queries;
            /// <summary>The block scores [Nt, ScoreStride], the cell lists [Nt, CellStride] and counts [Nt].</summary>
            public float[] Scores;
            public int[] Cells, CellCnt;
        }

        private SparseLayerProbe CaptureSparse(Dev dev, DevLayer L, int il, SlotLayer c, int row0, int nt, int p0)
        {
            var m = _m;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            int cached = p0 + nt, d = m.IdxDim;
            return new SparseLayerProbe
            {
                Layer = il, P0 = p0, Nt = nt, Ratio = L.Ratio, TopK = m.IdxTopK, IdxHeads = m.IdxHeads, IdxDim = d,
                ScoreStride = ScoreStride, CellStride = CellStride, NRot = m.NRot, RopeBase = m.RopeBase,
                RopeFreqScale = m.RopeFreqScale, Eps = m.RmsEps,
                // A streamed layer keeps only a ring of its raw keys (TS_Q4E_KV_RESIDENT).
                RawKeys = c.IdxRing > 0 ? null : Download<System.Half>(Ptr(c.IdxRaw), (long)cached * d),
                Pooled = Download<float>(Ptr(c.IdxPooled), (long)(cached / L.Ratio) * d),
                Queries = Download<float>(At(dev.IdxQ, (long)row0 * m.IdxHeads * d), (long)nt * m.IdxHeads * d),
                Scores = Download<float>(At(dev.Scores, (long)row0 * ScoreStride), (long)nt * ScoreStride),
                Cells = Download<int>((IntPtr)((long)Ptr(dev.Cells) + (long)row0 * CellStride * 4), (long)nt * CellStride),
                CellCnt = Download<int>((IntPtr)((long)Ptr(dev.CellCnt) + (long)row0 * 4), nt),
            };
        }

        /// <summary>Test hook: each layer's router logits and chosen experts on the single-sequence path, read
        /// back once the experts were chosen. Decode steps are not captured while it is set.</summary>
        internal Action<RouterLayerProbe> RouterProbe { get; set; }

        /// <summary>One layer's router for a ubatch: the logits [Nt, Experts] and the experts chosen [Nt, Used].</summary>
        internal sealed class RouterLayerProbe
        {
            public int Layer, P0, Nt, Experts, Used;
            public float[] Logits;
            public int[] Selected;
        }

        private RouterLayerProbe CaptureRouter(Dev dev, int il, int p0, int nt)
        {
            var m = _m;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            return new RouterLayerProbe
            {
                Layer = il, P0 = p0, Nt = nt, Experts = m.NExpert, Used = m.NExpertUsed,
                Logits = Download<float>(Ptr(dev.RouterLogits), (long)nt * m.NExpert),
                Selected = Download<int>(Ptr(dev.Moe.Sel), (long)nt * m.NExpertUsed),
            };
        }

        private static T[] Download<T>(IntPtr src, long count) where T : unmanaged
        {
            var a = new T[count];
            fixed (T* p = a)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, src, new UIntPtr((ulong)(count * sizeof(T)))).ThrowOnError();
            return a;
        }

        /// <summary>Tokens up to which the experts and projections run per row: a batched decode
        /// step's rows then compute exactly as each sequence's own step does.</summary>
        public const int PerSlotMaxRows = 16;

        // TS_Q4E_PERF=2 / 3: per-stage time, synchronized host time or GPU time (see CudaStageTimer).
        private const int StEmbed = 0, StHc = 1, StPle = 2, StGdnProj = 3, StGdnCore = 4, StAttnProj = 5, StAttnPrep = 6,
            StAttnCore = 7, StAttnOut = 8, StRouter = 9, StExperts = 10, StShexp = 11, StHead = 12, StBoundary = 13, StHostWait = 14;
        private static readonly string[] StageNames =
        {
            "embed", "hc", "ple", "gdnproj", "gdncore", "attnproj", "attnprep", "attncore", "attnout", "router",
            "experts", "shexp", "head", "boundary", "hostwait",
        };
        // TS_Q4E_PERF>=1: when the host had queued the whole step (the last launch before the
        // logits sync). Close to the step's wall time = the step is bound by launch overhead.
        private long _issueEnd;

        private double IssueMs(long start)
            => _issueEnd > start ? (_issueEnd - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency : 0;

        private void StageBegin(Dev dev) => (_sampling ? _sampleStages : _stages).SpanStart(dev.Alloc.Context, dev.Stream);

        private void StageEnd(Dev dev, int stage) => (_sampling ? _sampleStages : _stages).End(dev.Alloc.Context, dev.Stream, stage);

        /// <summary>Whether the next forward is a sampled one (TS_Q4E_PERF_SAMPLE); sets
        /// <see cref="_sampling"/> for it.</summary>
        private bool BeginSampledForward()
        {
            if (_perfSample <= 0 || StageObserver == null || ++_forwardsSinceSample < _perfSample)
                return false;
            // A sampled forward runs uncaptured with a GPU event a stage, several times a captured step's time, and on a
            // decode lane the other lane's step finishes first: at most one every TS_Q4E_PERF_SAMPLE_MIN_MS, so the
            // profile stays under a percent of a busy engine's time however fast its steps are (every 200th forward of
            // two lanes at 16 streams was one every two seconds, and 5-7% of the tokens).
            long now = System.Diagnostics.Stopwatch.GetTimestamp();
            if (_lastSampleAt != 0 && _perfSampleMinMs > 0
                && (now - _lastSampleAt) * 1000.0 / System.Diagnostics.Stopwatch.Frequency < _perfSampleMinMs)
                return false;
            _lastSampleAt = now;
            _forwardsSinceSample = 0;
            _sampling = true;
            return true;
        }

        // TS_Q4E_PERF_SAMPLE_MIN_MS: the least time between two sampled forwards (0: every TS_Q4E_PERF_SAMPLE-th forward).
        private readonly int _perfSampleMinMs = Math.Max(0, EnvInt("TS_Q4E_PERF_SAMPLE_MIN_MS", 10000));
        private long _lastSampleAt;

        /// <summary>A sampled forward's stage times to <see cref="StageObserver"/>, after every device
        /// finished it.</summary>
        private void EndSampledForward(int rows, bool decode)
        {
            if (!_sampling)
                return;
            _sampling = false;
            double[] ms = _sampleStages.CollectAndReset(synchronize: true);
            var observer = StageObserver;
            if (observer == null)
                return;
            for (int i = 0; i < ms.Length && i < StageNames.Length; i++)
            {
                if (ms[i] <= 0)
                    continue;
                try { observer(StageNames[i], ms[i] / 1000.0, rows, decode); }
                catch (Exception) { }
            }
        }
    }
}
