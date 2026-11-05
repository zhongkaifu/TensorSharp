// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class CudaNinferAttention : IDisposable
    {
        private readonly CudaModule _module;
        private readonly IntPtr _prep, _attention, _combine;
        // Decode and verification attention: GQA-packed split kernel with cp.async
        // double-buffered tiles. One CTA per (KV head, key split) reads each key once for all
        // six query heads and every row (eight verify rows fill 48 of its 64 MMA rows), where a
        // per-query-head core reads every key six times. It replaced both the unpipelined split
        // kernel and the Ignis Verify8 core: RTX 5090, 512-token answers after 7K / 33K / 62K,
        // 192.8 / 159.5 / 132.3 -> 217.2 / 189.8 / 183.2 tok/s.
        private readonly IntPtr _splitPipe;
        private readonly uint _splitPipeSmem;
        // FP8 KV (TS_Q35_NINFER_KV=fp8): 272-byte rows of E4M3 codes and two F32 group-128 scales.
        // Decode/verify stage FP8 tiles and widen them in shared memory; the BF16 prefill core reads
        // a per-layer BF16 copy of the visible history (_kvScratch, one 1 MiB K|V block per 256 keys).
        private readonly bool _kvFp8;
        private readonly IntPtr _fp8ToBf16;
        private IntPtr _kvScratch, _kvScratchTable;
        private int _kvScratchBlocks;
        internal const string KvEnvironment = "TS_Q35_NINFER_KV";
        internal bool KvFp8 => _kvFp8;
        internal static bool ResolveKvFp8(string raw) => raw switch
        {
            null or "bf16" => false,
            "fp8" => true,
            _ => throw new ArgumentException(KvEnvironment + " accepts only bf16 or fp8.")
        };
        /// <summary>A fixed split count that fills the SMs in one wave (each CTA holds 67.6 KB of
        /// tiles, so one per SM), its key range derived on the device (chunk 0): one launch
        /// shape, so one captured graph, for every history.</summary>
        internal Q35Kernels.SplitAttentionPlan SplitPlan(int sequences, int nKv)
            => new(0, PipeSplits(_sms, sequences, nKv));
        internal static int PipeSplits(int sms, int sequences, int nKv)
            => Math.Clamp(Math.Max(1, sms) / Math.Max(1, sequences * nKv), 1, Q35Kernels.AttnMaxSplitCount);
        private IntPtr _frequencies;
        private CudaNinferPrefillAttentionModule _prefill;
        private IntPtr _prefillWorkspace;
        private long _prefillWorkspaceBytes;
        private readonly int _maximumPrefillRows, _sms;
        internal const string PrefillThreadsEnvironment = "TS_Q35_NINFER_PREFILL_ATTN_THREADS";
        private readonly int _prefillThreads;
        private long _prefill128Submissions;
        internal bool PrefillEnabled => _prefill != null;
        internal string PrefillImageSha256 => _prefill?.ImageSha256;
        internal long PrefillCoreSubmissions => _prefill?.PagedCoreLaunches ?? 0;
        internal long PrefillWorkspaceBytes => _prefillWorkspaceBytes;
        internal long Prefill128CoreSubmissions => _prefill128Submissions;
        internal string PrefillNamespaceSuffix { get; private set; } = string.Empty;
        public string ImageSha256 => _module.ImageSha256;
        public CudaNinferAttention(float ropeBase = 10000000f, int maximumPrefillRows = 0,
            int maximumKeys = 0, int sms = 0,
            int heads = 24, int kvHeads = 4, int headDim = 256, bool automaticPrefill = false, bool kvFp8 = false)
        {
            _kvFp8 = kvFp8;
            string prefillRaw = Environment.GetEnvironmentVariable(CudaNinferPrefillAttentionModule.EnvironmentVariable);
            bool autoPrefill = automaticPrefill && prefillRaw == null && heads == CudaNinferPrefillAttentionModule.QueryHeads
                && kvHeads == CudaNinferPrefillAttentionModule.KvHeads && headDim == CudaNinferPrefillAttentionModule.HeadDim;
            _prefillThreads = ResolvePrefillThreads(Environment.GetEnvironmentVariable(PrefillThreadsEnvironment), prefillRaw != null);
            if (_prefillThreads == 128 && (maximumPrefillRows < 64 || maximumKeys < maximumPrefillRows
                || maximumKeys > CudaNinferPrefillAttentionModule.MaximumContext || sms < 1))
                throw new NotSupportedException("The explicit prefill schedule requires its bounded native target geometry.");
            string prefillPath = ResolvePrefillPath(prefillRaw,
                maximumPrefillRows, maximumKeys, sms);
            if (autoPrefill)
            {
                // The packaged Ignis-original image; absent -> legacy kernel.
                prefillPath = Q35CudaEngine.LocateNinferAutomaticAsset(Q35CudaEngine.NinferQualifiedPrefillAttentionAsset, CudaKernels.LocatePtxPath);
                if (prefillPath == null)
                    Console.Error.WriteLine("[q35-cuda] automatic NInfer prefill attention asset missing; using the legacy prefill kernel");
            }
            _maximumPrefillRows = maximumPrefillRows;
            _sms = sms;
            string path = CudaKernels.LocatePtxPath("tensorsharp_q35_ninfer_attention.ptx")
                ?? throw new NotSupportedException("Build the NInfer BF16 attention CUDA kernels before loading this model.");
            _module = CudaModule.LoadFromFile(path, true);
            try
            {
                _prep = _module.GetFunction("ts_ninfer_attn_prep_f32");
                _attention = _module.GetFunction(kvFp8 ? "ts_ninfer_attn_fp8_f32" : "ts_ninfer_attn_bf16_f32");
                _combine = _module.GetFunction("ts_ninfer_attn_combine_f32");
                _splitPipe = _module.GetFunction(kvFp8 ? "ts_ninfer_attn_fp8_split_pipe_f32" : "ts_ninfer_attn_bf16_split_pipe_f32");
                if (kvFp8) _fp8ToBf16 = _module.GetFunction("ts_ninfer_kv_fp8_to_bf16");
                CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(8u)).ThrowOnError();
                try
                {
                    void** pargs = stackalloc void*[] { &probe };
                    Launch(_module.GetFunction("ts_ninfer_attn_pipe_smem_bytes"), 1, 1, 1, 1, IntPtr.Zero, pargs);
                    int* bytes = stackalloc int[2];
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)bytes, probe, new UIntPtr(8u)).ThrowOnError();
                    _splitPipeSmem = (uint)bytes[kvFp8 ? 1 : 0];
                }
                finally { CudaDriverApi.cuMemFree(probe); }
                CudaDriverApi.cuFuncSetAttribute(_splitPipe, CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)_splitPipeSmem).ThrowOnError();
                var frequencies = new float[32];
                for (int i = 0; i < frequencies.Length; i++) frequencies[i] = (float)(1.0 / Math.Pow((double)ropeBase, 2.0*i/64.0));
                CudaDriverApi.cuMemAlloc(out _frequencies, new UIntPtr(128u)).ThrowOnError();
                fixed (float* values = frequencies)
                    CudaDriverApi.cuMemcpyHtoD(_frequencies, (IntPtr)values, new UIntPtr(128u)).ThrowOnError();
                if (prefillPath != null)
                {
                    try { _prefill = new CudaNinferPrefillAttentionModule(prefillPath); }
                    catch (Exception error) when (autoPrefill && Q35CudaEngine.NinferAutomaticModuleUnavailable(error))
                    {
                        Console.Error.WriteLine($"[q35-cuda] automatic NInfer prefill attention unavailable before use: {error.Message}; using the legacy prefill kernel");
                    }
                }
                if (_prefill != null)
                {
                    _prefillWorkspaceBytes = CudaNinferPrefillAttentionModule.WorkspaceCapacity(maximumPrefillRows, maximumKeys);
                    CudaDriverApi.cuMemAlloc(out _prefillWorkspace, new UIntPtr(checked((ulong)_prefillWorkspaceBytes))).ThrowOnError();
                    PrefillNamespaceSuffix = "|ninfer-prefill-attn=" + CudaNinferPrefillAttentionModule.Policy + ":"
                        + _prefill.ImageSha256.ToLowerInvariant() + ":ub=" + maximumPrefillRows + ":sms=" + sms;
                    Console.Error.WriteLine($"[q35-cuda] {(autoPrefill ? "automatic" : "explicit")} original BF16 prefill attention: policy={CudaNinferPrefillAttentionModule.Policy}, module={_prefill.ImageSha256}, workspaceBytes={_prefillWorkspaceBytes}, maxRows={maximumPrefillRows}; ordinary single-sequence prefill only");
                }
                else if (_prefillThreads == 128)
                {
                    PrefillNamespaceSuffix = "|ninfer-prefill-schedule=legacy-f32-br64-kt32-t128-v1:"
                        + ImageSha256.ToLowerInvariant() + ":ub=" + maximumPrefillRows;
                    Console.Error.WriteLine($"[q35-cuda] explicit legacy BF16 prefill schedule: threads=128, module={ImageSha256}; original arithmetic, ordinary single-sequence prefill only");
                }
            }
            catch
            {
                // Cleanup failures cannot replace the original module/allocation
                // admission error. No kernels or graphs have been published yet.
                if (_prefillWorkspace != IntPtr.Zero) CudaDriverApi.cuMemFree(_prefillWorkspace);
                _prefillWorkspace = IntPtr.Zero;
                try { _prefill?.Dispose(); } catch { }
                _prefill = null;
                if (_frequencies != IntPtr.Zero) CudaDriverApi.cuMemFree(_frequencies);
                _frequencies = IntPtr.Zero;
                try { _module.Dispose(); } catch { }
                throw;
            }
        }
        internal static int ResolvePrefillThreads(string raw, bool hasOriginalModule)
        {
            if (raw == null) return 256;
            if (hasOriginalModule) throw new NotSupportedException("Select one explicit prefill attention policy.");
            return raw switch
            {
                "128" => 128,
                "256" => 256,
                _ => throw new ArgumentException(PrefillThreadsEnvironment + " accepts only128 or256.")
            };
        }
        internal static bool Use128Prefill(int mode, int rows, int sequences, bool isVerify,
            bool allRowsValid, int heads, int kvHeads, int capacity)
            => mode == 128 && rows >= 64 && rows <= capacity && sequences == 1 && !isVerify
                && allRowsValid && heads == 24 && kvHeads == 4;
        internal static string ResolvePrefillPath(string raw, int maximumRows, int maximumKeys, int sms)
        {
            if (raw == null) return null;
            if (string.IsNullOrWhiteSpace(raw) || !Path.IsPathFullyQualified(raw) || raw != Path.GetFullPath(raw))
                throw new ArgumentException(CudaNinferPrefillAttentionModule.EnvironmentVariable + " requires an explicit normalized absolute module path.");
            if (maximumRows < 64 || maximumRows > CudaNinferPrefillAttentionModule.MaximumRows
                || maximumKeys < maximumRows || maximumKeys > CudaNinferPrefillAttentionModule.MaximumContext || sms < 1)
                throw new NotSupportedException("The original BF16 prefill attention policy requires its bounded target rows/context/device geometry.");
            if (!File.Exists(raw)) throw new FileNotFoundException("Original BF16 prefill attention module is unavailable.", raw);
            return raw;
        }
        private static void Launch(IntPtr f, uint x, uint y, uint z, uint threads, IntPtr stream, void** args)
            => CudaDriverApi.cuLaunchKernel(f, x, y, z, threads, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        public void Prepare(IntPtr qkv, int rows, int rowsPerSeq, int nHead, int nKv, IntPtr qNorm, IntPtr kNorm, float eps,
            IntPtr ropePos, int ropeStride, int sectT, int sectH, int sectW, float ropeBase, IntPtr cacheRow, IntPtr table, int maxBlocks,
            long kOff, long vOff, IntPtr q, IntPtr gate, IntPtr stream)
        {
            IntPtr frequencies = _frequencies;
            int fp8 = _kvFp8 ? 1 : 0;
            void** args = stackalloc void*[] { &qkv, &rows, &rowsPerSeq, &nHead, &nKv, &qNorm, &kNorm, &eps,
                &ropePos, &ropeStride, &sectT, &sectH, &sectW, &ropeBase, &cacheRow, &table, &maxBlocks, &kOff, &vOff, &q, &gate, &frequencies, &fp8 };
            Launch(_prep, checked((uint)(((long)rows * (nHead + 2*nKv)*32+255)/256)), 1, 1, 256, stream, args);
        }
        public void Attention(IntPtr q, IntPtr gate, IntPtr keyEnd, IntPtr table, int maxBlocks, long kOff, long vOff,
            int rowsPerSeq, int sequences, int nHead, int nKv, float scale, int maxKeys, IntPtr partial, IntPtr output, IntPtr stream,
            bool isVerify = false, bool allRowsValid = true)
        {
            if (_prefill != null && rowsPerSeq <= _maximumPrefillRows
                && nHead == CudaNinferPrefillAttentionModule.QueryHeads && nKv == CudaNinferPrefillAttentionModule.KvHeads
                && CudaNinferPrefillAttentionModule.Supports(rowsPerSeq, sequences, isVerify, allRowsValid))
            {
                if (_kvFp8)
                {
                    WidenFp8History(table, kOff, vOff, maxKeys, nKv, stream);
                    _prefill.RunF32(q, gate, _kvScratchTable, _kvScratchBlocks, 0, CudaNinferPrefillAttentionModule.KvPlaneBlockBytes,
                        rowsPerSeq, checked(maxKeys - rowsPerSeq), scale, output, _prefillWorkspace, _prefillWorkspaceBytes, _sms, stream);
                    return;
                }
                _prefill.RunF32(q, gate, table, maxBlocks, kOff, vOff, rowsPerSeq, checked(maxKeys - rowsPerSeq),
                    scale, output, _prefillWorkspace, _prefillWorkspaceBytes, _sms, stream);
                return;
            }
            if (rowsPerSeq * (nHead/nKv) <= Q35Kernels.AttnBlockRows && partial != IntPtr.Zero)
            {
                var plan = SplitPlan(sequences, nKv);
                int chunk = plan.Chunk, splits = plan.Splits;
                void** args = stackalloc void*[] { &q, &keyEnd, &table, &maxBlocks, &kOff, &vOff, &rowsPerSeq,
                    &nHead, &nKv, &scale, &chunk, &partial };
                CudaDriverApi.cuLaunchKernel(_splitPipe, (uint)splits, (uint)nKv, (uint)sequences, 128, 1, 1, _splitPipeSmem,
                    stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                void** merge = stackalloc void*[] { &partial, &gate, &splits, &output };
                Launch(_combine, checked((uint)(sequences*rowsPerSeq*nHead)), 1, 1, 256, stream, merge);
            }
            else
            {
                int threads = Use128Prefill(_prefillThreads, rowsPerSeq, sequences, isVerify,
                    allRowsValid, nHead, nKv, _maximumPrefillRows) ? 128 : 256;
                void** args = stackalloc void*[] { &q, &gate, &keyEnd, &table, &maxBlocks, &kOff, &vOff,
                    &rowsPerSeq, &nHead, &nKv, &scale, &output };
                int blockRows = threads / 2;
                Launch(_attention, checked((uint)((rowsPerSeq*(nHead/nKv)+blockRows-1)/blockRows)), (uint)nKv, (uint)sequences, (uint)threads, stream, args);
                if (threads == 128) _prefill128Submissions++;
            }
        }
        /// <summary>BF16 copy of keys [0, keys) of one sequence's layer for the prefill core.</summary>
        private void WidenFp8History(IntPtr table, long kOff, long vOff, int keys, int nKv, IntPtr stream)
        {
            int blocks = checked((keys + 255) / 256);
            if (blocks > _kvScratchBlocks)
            {
                if (_kvScratch != IntPtr.Zero) { CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError(); CudaDriverApi.cuMemFree(_kvScratch).ThrowOnError(); CudaDriverApi.cuMemFree(_kvScratchTable).ThrowOnError(); }
                // Grow by the visible history (plus 8 blocks), not by doubling: at 128K the scratch is
                // most of a GB that KV blocks need.
                int capacity = Math.Max(blocks + 8, 16);
                long blockBytes = 2 * CudaNinferPrefillAttentionModule.KvPlaneBlockBytes;
                CudaDriverApi.cuMemAlloc(out _kvScratch, new UIntPtr(checked((ulong)(capacity * blockBytes)))).ThrowOnError();
                CudaDriverApi.cuMemAlloc(out _kvScratchTable, new UIntPtr((ulong)capacity * 8)).ThrowOnError();
                var pointers = new long[capacity];
                for (int i = 0; i < capacity; i++) pointers[i] = _kvScratch.ToInt64() + i * blockBytes;
                fixed (long* p = pointers) CudaDriverApi.cuMemcpyHtoD(_kvScratchTable, (IntPtr)p, new UIntPtr((ulong)capacity * 8)).ThrowOnError();
                _kvScratchBlocks = capacity;
            }
            IntPtr scratchTable = _kvScratchTable;
            long plane = CudaNinferPrefillAttentionModule.KvPlaneBlockBytes;
            void** args = stackalloc void*[] { &table, &kOff, &vOff, &keys, &scratchTable, &plane };
            Launch(_fp8ToBf16, checked((uint)((keys + 7) / 8)), (uint)nKv, 2, 256, stream, args);
        }

        /// <summary>Frees the BF16 history scratch (it grows again with the next widened prompt chunk). It
        /// is sized to the longest history widened so far (about 1 GB at 128K) and kept for that whole
        /// time; on a full 32 GB card it left no room for the slot of a second chat on a cached 127K-token
        /// document. True when it held memory.</summary>
        internal bool ReleaseTransientScratch(IntPtr stream)
        {
            if (_kvScratch == IntPtr.Zero) return false;
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            CudaDriverApi.cuMemFree(_kvScratch).ThrowOnError();
            CudaDriverApi.cuMemFree(_kvScratchTable).ThrowOnError();
            _kvScratch = _kvScratchTable = IntPtr.Zero;
            _kvScratchBlocks = 0;
            return true;
        }

        public void Dispose()
        {
            if (_kvScratch != IntPtr.Zero) { CudaDriverApi.cuMemFree(_kvScratch); CudaDriverApi.cuMemFree(_kvScratchTable); _kvScratch = _kvScratchTable = IntPtr.Zero; }
            // The engine synchronizes and destroys its layer graphs before
            // disposing this module or any captured workspace pointer.
            if (_prefillThreads == 128)
                Console.Error.WriteLine($"[q35-cuda] legacy BF16 prefill schedule128 core submissions={_prefill128Submissions}; host submissions, not a timing measurement");
            if (_prefill != null)
            {
                Console.Error.WriteLine($"[q35-cuda] original BF16 prefill attention core submissions={_prefill.PagedCoreLaunches}; host submissions, not a timing measurement");
                _prefill.Dispose(); _prefill = null;
            }
            if (_prefillWorkspace != IntPtr.Zero) CudaDriverApi.cuMemFree(_prefillWorkspace).ThrowOnError();
            _prefillWorkspace = IntPtr.Zero;
            if (_frequencies != IntPtr.Zero) { CudaDriverApi.cuMemFree(_frequencies).ThrowOnError(); _frequencies = IntPtr.Zero; }
            _module.Dispose();
        }
    }
}
