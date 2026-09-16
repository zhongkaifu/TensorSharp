// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : IDraftHead
    {
        private readonly record struct MtpWeight(IntPtr Data, int Type, long Bytes);
        private GgufFile _mtpFile;
        private Dictionary<string, MtpWeight> _mtpWeights;
        private long _mtpResidentBytes;
        private bool _mtpReady;
        private string _mtpPath;
        private int _mtpLayer;
        private Qwen4ExpMtpConfig _mtpConfig;
        private Qwen4ExpAttnArgs _mtpAttn;
        private Qwen4ExpFfnArgs _mtpFfn;
        private Qwen4ExpHeadArgs _mtpHead;
        private readonly Dictionary<object, MtpState> _mtpStates = new(ReferenceEqualityComparer.Instance);

        private sealed class MtpState
        {
            public Tensor K, V, RetiringK, RetiringV;
            public IntPtr Executor;
            public int Capacity, Position;
            public bool Failed;
        }

        public bool HasDraftHead => _mtpReady;
        public DraftHeadKind DraftHeadKind => HasDraftHead ? DraftHeadKind.PerToken : DraftHeadKind.None;
        public bool SupportsFusedCatchUpStep => HasDraftHead;
        public bool DraftHeadResumesAfterGap => false;

        public void LoadMtpDraftWeights(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);
            path = Path.GetFullPath(path);
            if (_mtpFile != null)
            {
                if (string.Equals(_mtpPath, path, StringComparison.OrdinalIgnoreCase)) return;
                throw new InvalidOperationException("qwen4exp already has a different MTP head attached.");
            }
            if (_layerDevice != null && (_cacheSeqLen != 0 || LayerSplitDegree > 1))
                throw new InvalidOperationException("Attach the qwen4exp MTP head at model construction, before multi-device placement or conversation state exists.");
            if (!IsGgmlBackend || !GgmlBasicOps.Qwen4ExpSpecApiAvailable())
                throw new NotSupportedException("qwen4exp MTP requires a GGML native library with the owned speculative APIs. Rebuild TensorSharp's native library.");

            var file = new GgufFile(path);
            try
            {
                int layer = ValidateMtpHead(_gguf, file);
                if (file.IsSplit)
                    throw new NotSupportedException("The shared qwen4exp MTP head must be a single GGUF file.");
                long fileBytes = new FileInfo(path).Length;
                var weights = new Dictionary<string, MtpWeight>(file.Tensors.Count, StringComparer.Ordinal);
                long residentBytes = 0;
                foreach (var tensor in file.Tensors.Values)
                {
                    long bytes = file.GetTensorByteCount(tensor);
                    long begin = checked(file.DataOffset + (long)tensor.Offset);
                    if (bytes <= 0 || begin < file.DataOffset || begin > fileBytes - bytes)
                        throw new InvalidDataException($"qwen4exp MTP tensor payload is truncated: {tensor.Name}.");
                    if (!file.TryGetTensorDataPointer(tensor, out IntPtr pointer))
                        throw new IOException($"Cannot map qwen4exp MTP tensor: {tensor.Name}.");
                    weights.Add(tensor.Name, new MtpWeight(pointer, (int)tensor.Type, bytes));
                    // These four tensors are present but inactive: the published
                    // head declares compression[48] == 0, validated above.
                    if (!tensor.Name.Contains(".indexer.", StringComparison.Ordinal))
                        residentBytes = checked(residentBytes + bytes);
                }
                _mtpFile = file;
                _mtpWeights = weights;
                _mtpResidentBytes = residentBytes;
                _mtpPath = path;
                _mtpLayer = layer;
                if (_layerDevice != null) FinalizeMtpHead();
            }
            catch
            {
                if (ReferenceEquals(_mtpFile, file)) DisposeMtpHead();
                else file.Dispose();
                throw;
            }
        }

        private void FinalizeMtpHead()
        {
            if (_mtpFile == null || _mtpReady) return;
            if (!TryResolveQuant("output.weight", out var output, out int outputType, out long outputBytes))
                throw new InvalidOperationException("qwen4exp MTP requires the target's separate output weight.");
            MtpWeight W(string suffix) => _mtpWeights[$"blk.{_mtpLayer}.{suffix}.weight"];
            var en = W("nextn.enorm"); var hn = W("nextn.hnorm"); var eh = W("nextn.eh_proj");
            _mtpConfig = new Qwen4ExpMtpConfig
            {
                Enorm = en.Data, Hnorm = hn.Data, EhProj = eh.Data, EhBytes = eh.Bytes, EhType = eh.Type,
                Hidden = Config.HiddenSize, Hc = _hc, HcLowRank = _hcLowRank,
                HeadDim = Config.HeadDim, Heads = Config.NumHeads, KvHeads = Config.NumKVHeads,
                RotaryDim = _ropeDimCount, Experts = _numExperts, UsedExperts = _numExpertsUsed,
                ExpertFf = _expertFf, SharedFf = _sharedFf,
                Device = DeviceForLayer(Config.NumLayers - 1),
                Eps = Config.Eps, RopeBase = Config.RopeBase, RopeScale = 1.0f / Config.RopeScale,
                AttnScale = _attnScale,
                RopeSection0 = _ropeSections[0], RopeSection1 = _ropeSections[1],
                RopeSection2 = _ropeSections[2], RopeSection3 = _ropeSections[3],
            };
            var ad = W("hc_attn_down"); var au = W("hc_attn_up"); var ai = W("hc_attn_inject");
            var q = W("attn_q"); var k = W("attn_k"); var v = W("attn_v"); var o = W("attn_output");
            _mtpAttn = new Qwen4ExpAttnArgs
            {
                HcNorm = W("hc_attn_norm").Data,
                HcDown = ad.Data, HcDownType = ad.Type, HcDownBytes = ad.Bytes,
                HcUp = au.Data, HcUpType = au.Type, HcUpBytes = au.Bytes,
                HcInject = ai.Data, HcInjectType = ai.Type, HcInjectBytes = ai.Bytes,
                Wq = q.Data, WqType = q.Type, WqBytes = q.Bytes,
                Wk = k.Data, WkType = k.Type, WkBytes = k.Bytes,
                Wv = v.Data, WvType = v.Type, WvBytes = v.Bytes,
                Wo = o.Data, WoType = o.Type, WoBytes = o.Bytes,
                QNorm = W("attn_q_norm").Data, KNorm = W("attn_k_norm").Data,
                KvType = (int)GgmlTensorType.F16,
            };
            var fd = W("hc_ffn_down"); var fu = W("hc_ffn_up"); var fi = W("hc_ffn_inject");
            var r = W("ffn_gate_inp"); var eg = W("ffn_gate_exps"); var eu = W("ffn_up_exps"); var ed = W("ffn_down_exps");
            var sg = W("ffn_gate_shexp"); var su = W("ffn_up_shexp"); var sd = W("ffn_down_shexp");
            _mtpFfn = new Qwen4ExpFfnArgs
            {
                HcNorm = W("hc_ffn_norm").Data,
                HcDown = fd.Data, HcDownType = fd.Type, HcDownBytes = fd.Bytes,
                HcUp = fu.Data, HcUpType = fu.Type, HcUpBytes = fu.Bytes,
                HcInject = fi.Data, HcInjectType = fi.Type, HcInjectBytes = fi.Bytes,
                Router = r.Data, RouterType = r.Type, RouterBytes = r.Bytes,
                GateExps = eg.Data, GateExpsType = eg.Type, GateExpsBytes = eg.Bytes,
                UpExps = eu.Data, UpExpsType = eu.Type, UpExpsBytes = eu.Bytes,
                DownExps = ed.Data, DownExpsType = ed.Type, DownExpsBytes = ed.Bytes,
                ShGateInp = W("ffn_gate_inp_shexp").Data,
                ShGate = sg.Data, ShGateType = sg.Type, ShGateBytes = sg.Bytes,
                ShUp = su.Data, ShUpType = su.Type, ShUpBytes = su.Bytes,
                ShDown = sd.Data, ShDownType = sd.Type, ShDownBytes = sd.Bytes,
            };
            var hd = W("nextn.hc_head_down"); var hu = W("nextn.hc_head_up");
            _mtpHead = new Qwen4ExpHeadArgs
            {
                HcNorm = W("nextn.hc_head_norm").Data,
                HcDown = hd.Data, HcDownType = hd.Type, HcDownBytes = hd.Bytes,
                HcUp = hu.Data, HcUpType = hu.Type, HcUpBytes = hu.Bytes,
                // Borrowed target CacheKey/storage; never disposed by the head.
                Head = output, HeadType = outputType, HeadBytes = outputBytes, Vocab = Config.VocabSize,
            };
            if (_backend == BackendType.GgmlCuda)
            {
                int previousRank = GgmlBasicOps.GetActiveRank();
                try
                {
                    GgmlBasicOps.SetActiveRank(_mtpConfig.Device);
                    CheckMtpDeviceBudget(_mtpResidentBytes);
                    foreach (var pair in _mtpWeights)
                    {
                        if (pair.Key.Contains(".indexer.", StringComparison.Ordinal) || pair.Value.Bytes < 4096) continue;
                        var info = _mtpFile.Tensors[pair.Key];
                        if (!GgmlBasicOps.PreloadQuantizedWeight(pair.Value.Data, pair.Value.Data,
                            pair.Value.Type, checked((long)info.Shape[0]), info.NumElements / (long)info.Shape[0], pair.Value.Bytes))
                            throw new NotSupportedException($"qwen4exp MTP weight exceeds the selected device's buffer limit: {pair.Key}.");
                    }
                }
                finally { GgmlBasicOps.SetActiveRank(previousRank); }
            }
            _mtpReady = true;
            Console.WriteLine($"  Qwen4Exp shared MTP ready: block={_mtpLayer}, features={_hcDim}, "
                + $"draftCompression=0, privateKV=F16, device={_mtpConfig.Device}, mappedBytes={_mtpResidentBytes}.");
        }

        private static void CheckMtpDeviceBudget(long additionalBytes)
        {
            const long scratchReserve = 64L * 1024 * 1024;
            if (GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out _)
                && free < checked(additionalBytes + scratchReserve))
                throw new OutOfMemoryException("qwen4exp MTP cannot admit its weights/cache plus a 64 MiB graph reserve on the selected device.");
        }

        private unsafe MtpState EnsureMtpState(int required)
        {
            if (!HasDraftHead || required < 0 || required > MaxContextLength || !EnsureGdnArgs())
                throw new InvalidOperationException("qwen4exp MTP has no usable head/state or the requested position exceeds context capacity.");
            object owner = ActiveSpecOwner;
            if (!_mtpStates.TryGetValue(owner, out var state))
            {
                // Reserve publication before allocating native/cache resources.
                _mtpStates.EnsureCapacity(_mtpStates.Count + 1);
                state = new MtpState();
                _mtpStates.Add(owner, state);
            }
            if (state.Failed) throw new InvalidOperationException("qwen4exp MTP conversation failed; reset before reuse.");
            if (required <= state.Capacity) return state;
            int cap = Math.Min(MaxContextLength, checked((required + 255) / 256 * 256));
            cap = Math.Max(cap, Math.Min(256, MaxContextLength));
            Tensor nextK = null, nextV = null;
            int previousRank = GgmlBasicOps.GetActiveRank();
            try
            {
                GgmlBasicOps.SetActiveRank(_mtpConfig.Device);
                CheckMtpDeviceBudget(checked(2L * Config.NumKVHeads * cap * Config.HeadDim * sizeof(ushort)));
                nextK = new Tensor(_allocator, DType.Float16, Config.NumKVHeads, cap, Config.HeadDim);
                nextV = new Tensor(_allocator, DType.Float16, Config.NumKVHeads, cap, Config.HeadDim);
                InitializeMtpCache(nextK);
                InitializeMtpCache(nextV);
                if (state.Position > 0)
                {
                    if (!GgmlBasicOps.Qwen4ExpMtpCopyKv(state.Executor,
                        TensorComputePrimitives.GetStoragePointer(state.K),
                        TensorComputePrimitives.GetStoragePointer(state.V), state.K.Storage.ByteLength))
                        throw new InvalidOperationException("qwen4exp MTP failed to preserve KV before growth.");
                    CopyCacheRows(state.K, nextK, state.Position);
                    CopyCacheRows(state.V, nextV, state.Position);
                }
                var config = _mtpConfig;
                config.Capacity = cap;
                var attn = _mtpAttn;
                attn.KCache = TensorComputePrimitives.GetStoragePointer(nextK);
                attn.VCache = TensorComputePrimitives.GetStoragePointer(nextV);
                attn.KvBytes = nextK.Storage.ByteLength;
                IntPtr executor = GgmlBasicOps.Qwen4ExpMtpCreate(ref config, ref attn, ref _mtpFfn, ref _mtpHead);
                if (executor == IntPtr.Zero) throw new InvalidOperationException("qwen4exp MTP executor creation failed.");
                // Graphs release their borrowed cache bindings before old storage.
                if (state.Executor != IntPtr.Zero) GgmlBasicOps.Qwen4ExpMtpFree(state.Executor);
                state.Executor = executor;
                state.RetiringK = state.K; state.RetiringV = state.V;
                state.K = nextK; state.V = nextV; state.Capacity = cap;
                nextK = nextV = null;
            }
            catch { state.Failed = true; throw; }
            finally
            {
                // Keep ownership if a storage release throws, so reset can retry.
                // A failed allocation has not replaced the live K/V pair.
                if (nextK != null) state.RetiringK = nextK;
                if (nextV != null) state.RetiringV = nextV;
                try
                {
                    try { ReleaseMtpCache(ref state.RetiringK); }
                    finally { ReleaseMtpCache(ref state.RetiringV); }
                }
                catch { state.Failed = true; throw; }
                finally { GgmlBasicOps.SetActiveRank(previousRank); }
            }
            return state;
        }

        private unsafe void RunMtp(int[] tokens, float[] previous, int position, float[] logits, float[] hidden)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(previous);
            if (tokens.Length == 0 || position < 0 || previous.Length < checked(tokens.Length * _hcDim)
                || (logits != null && logits.Length < Config.VocabSize) || (hidden != null && hidden.Length < _hcDim))
                throw new ArgumentException("qwen4exp MTP inputs/outputs do not cover the requested rows.");
            var state = EnsureMtpState(checked(position + tokens.Length));
            if (position > state.Position)
                throw new InvalidOperationException("qwen4exp MTP cannot draft across a gap in its own KV history.");
            using var embedding = Embedding(tokens); // the target owns the borrowed embedding weights
            _mtpPositions.TryGetValue(ActiveSpecOwner, out var positions);
            var (ropePosition, mrope) = ResolveMtpPositions(positions, position, tokens.Length, _mropeCacheGap);
            try
            {
                fixed (float* hp = previous)
                fixed (float* ho = hidden)
                fixed (float* lp = logits)
                fixed (int* mp = mrope)
                    if (!GgmlBasicOps.Qwen4ExpMtpForward(state.Executor, (IntPtr)GetFloatPtr(embedding),
                        (IntPtr)hp, tokens.Length, position, ropePosition,
                        (IntPtr)mp, (IntPtr)ho, (IntPtr)lp))
                        throw new InvalidOperationException("qwen4exp MTP forward failed; reset the conversation before reuse.");
                state.Position = position + tokens.Length;
                PruneMtpPositions(ActiveSpecOwner, state.Position);
            }
            catch { state.Failed = true; throw; }
        }

        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => RunMtp(new[] { token }, hPrev, pos, logitsOut, hOut);
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
            => RunMtp(tokens, hRows, startPos, null, null);
        public void DraftCatchUpAndStep(int[] tokens, float[] hRows, int startPos, float[] logitsOut, float[] hOut)
            => RunMtp(tokens, hRows, startPos, logitsOut, hOut);

        private unsafe void InitializeMtpCache(Tensor tensor)
        {
            // This storage seeds a separately owned native cache. Drain any work
            // on a recycled host address before clearing its F16 bytes on the host.
            tensor.Storage.EnsureHostReadable();
            InvalidateTensorDeviceCache(tensor);
            NativeMemory.Clear(TensorComputePrimitives.GetHalfPointer(tensor),
                checked((nuint)tensor.Storage.ByteLength));
        }

        private void DisposeMtpCache(Tensor tensor)
        {
            if (tensor == null) return;
            InvalidateTensorDeviceCache(tensor);
            tensor.Dispose();
        }

        private void ReleaseMtpCache(ref Tensor tensor)
        {
            DisposeMtpCache(tensor);
            tensor = null;
        }

        private void ReleaseMtpState(object owner)
        {
            if (owner != null) _mtpPositions?.Remove(owner);
            if (owner == null || _mtpStates == null || !_mtpStates.TryGetValue(owner, out var state)) return;
            state.Failed = true;
            if (state.Executor != IntPtr.Zero) GgmlBasicOps.Qwen4ExpMtpFree(state.Executor);
            state.Executor = IntPtr.Zero;
            ReleaseMtpCache(ref state.K); ReleaseMtpCache(ref state.V);
            ReleaseMtpCache(ref state.RetiringK); ReleaseMtpCache(ref state.RetiringV);
            _mtpStates.Remove(owner);
        }

        private void DisposeMtpHead()
        {
            _mtpReady = false;
            if (_mtpStates != null) foreach (var state in _mtpStates.Values)
            {
                state.Failed = true;
                if (state.Executor != IntPtr.Zero) GgmlBasicOps.Qwen4ExpMtpFree(state.Executor);
                state.Executor = IntPtr.Zero;
                ReleaseMtpCache(ref state.K); ReleaseMtpCache(ref state.V);
                ReleaseMtpCache(ref state.RetiringK); ReleaseMtpCache(ref state.RetiringV);
            }
            _mtpStates?.Clear();
            if (_mtpWeights != null)
                foreach (var weight in _mtpWeights.Values) GgmlBasicOps.InvalidateHostBuffer(weight.Data);
            _mtpWeights = null;
            _mtpFile?.Dispose();
            _mtpFile = null;
            _mtpResidentBytes = 0;
            _mtpPositions?.Clear();
        }
    }
}
