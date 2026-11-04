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
        private readonly IntPtr _prep, _attention, _split, _combine;
        private IntPtr _frequencies;
        private CudaNinferPrefillAttentionModule _prefill;
        private IntPtr _prefillWorkspace;
        private long _prefillWorkspaceBytes;
        private readonly int _maximumPrefillRows, _sms;
        internal const string PrefillThreadsEnvironment = "TS_Q35_NINFER_PREFILL_ATTN_THREADS";
        private readonly int _prefillThreads;
        private long _prefill128Submissions;
        internal const string Verify8Environment = "TS_Q35_NINFER_VERIFY8_ATTN_PTX";
        private CudaNinferPrefillAttentionModule _verify8;
        private IntPtr _verify8Workspace;
        private long _verify8WorkspaceBytes;
        internal bool Verify8Enabled => _verify8 != null;
        internal string Verify8Selection { get; private set; } = "disabled-or-ineligible";
        internal string Verify8ImageSha256 { get; private set; }
        internal long Verify8CoreSubmissions => _verify8?.PagedCoreLaunches ?? 0;
        internal long Verify8WorkspaceBytes => _verify8WorkspaceBytes;
        internal bool PrefillEnabled => _prefill != null;
        internal string PrefillImageSha256 => _prefill?.ImageSha256;
        internal long PrefillCoreSubmissions => _prefill?.PagedCoreLaunches ?? 0;
        internal long PrefillWorkspaceBytes => _prefillWorkspaceBytes;
        internal long Prefill128CoreSubmissions => _prefill128Submissions;
        internal string PrefillNamespaceSuffix { get; private set; } = string.Empty;
        public string ImageSha256 => _module.ImageSha256;
        public CudaNinferAttention(float ropeBase = 10000000f, int maximumPrefillRows = 0,
            int maximumKeys = 0, int sms = 0, bool verificationEnabled = false,
            int heads = 24, int kvHeads = 4, int headDim = 256, bool automaticVerify8 = false)
        {
            string prefillRaw = Environment.GetEnvironmentVariable(CudaNinferPrefillAttentionModule.EnvironmentVariable);
            // Snapshot the explicit path once; captured graphs never consult a
            // changed environment, host cache position or changing page count.
            string explicitVerify8 = Environment.GetEnvironmentVariable(Verify8Environment);
            string verify8Path = ResolveVerify8Path(explicitVerify8,
                verificationEnabled, maximumPrefillRows, maximumKeys, sms, heads, kvHeads, headDim);
            bool autoVerify8 = UseAutomaticVerify8(automaticVerify8, explicitVerify8);
            if (autoVerify8)
            {
                verify8Path = Q35CudaEngine.LocateNinferAutomaticAsset(Q35CudaEngine.NinferQualifiedVerify8Asset, CudaKernels.LocatePtxPath);
                if (verify8Path == null)
                {
                    Verify8Selection = "asset-missing";
                    Console.Error.WriteLine("[q35-cuda] automatic NInfer Verify8 attention asset missing; using existing attention");
                }
            }
            _prefillThreads = ResolvePrefillThreads(Environment.GetEnvironmentVariable(PrefillThreadsEnvironment), prefillRaw != null);
            if (_prefillThreads == 128 && (maximumPrefillRows < 64 || maximumKeys < maximumPrefillRows
                || maximumKeys > CudaNinferPrefillAttentionModule.MaximumContext || sms < 1))
                throw new NotSupportedException("The explicit prefill schedule requires its bounded native target geometry.");
            string prefillPath = ResolvePrefillPath(prefillRaw,
                maximumPrefillRows, maximumKeys, sms);
            _maximumPrefillRows = maximumPrefillRows;
            _sms = sms;
            string path = CudaKernels.LocatePtxPath("tensorsharp_q35_ninfer_attention.ptx")
                ?? throw new NotSupportedException("Build the NInfer BF16 attention CUDA kernels before loading this model.");
            _module = CudaModule.LoadFromFile(path, true);
            try
            {
                _prep = _module.GetFunction("ts_ninfer_attn_prep_f32");
                _attention = _module.GetFunction("ts_ninfer_attn_bf16_f32");
                _split = _module.GetFunction("ts_ninfer_attn_bf16_split_f32");
                _combine = _module.GetFunction("ts_ninfer_attn_combine_f32");
                var frequencies = new float[32];
                for (int i = 0; i < frequencies.Length; i++) frequencies[i] = (float)(1.0 / Math.Pow((double)ropeBase, 2.0*i/64.0));
                CudaDriverApi.cuMemAlloc(out _frequencies, new UIntPtr(128u)).ThrowOnError();
                fixed (float* values = frequencies)
                    CudaDriverApi.cuMemcpyHtoD(_frequencies, (IntPtr)values, new UIntPtr(128u)).ThrowOnError();
                if (prefillPath != null)
                {
                    _prefill = new CudaNinferPrefillAttentionModule(prefillPath);
                    _prefillWorkspaceBytes = CudaNinferPrefillAttentionModule.WorkspaceCapacity(maximumPrefillRows, maximumKeys);
                    CudaDriverApi.cuMemAlloc(out _prefillWorkspace, new UIntPtr(checked((ulong)_prefillWorkspaceBytes))).ThrowOnError();
                    PrefillNamespaceSuffix = "|ninfer-prefill-attn=" + CudaNinferPrefillAttentionModule.Policy + ":"
                        + _prefill.ImageSha256.ToLowerInvariant() + ":ub=" + maximumPrefillRows + ":sms=" + sms;
                    Console.Error.WriteLine($"[q35-cuda] explicit original BF16 prefill attention: policy={CudaNinferPrefillAttentionModule.Policy}, module={_prefill.ImageSha256}, workspaceBytes={_prefillWorkspaceBytes}, maxRows={maximumPrefillRows}; ordinary single-sequence prefill only");
                }
                else if (_prefillThreads == 128)
                {
                    PrefillNamespaceSuffix = "|ninfer-prefill-schedule=legacy-f32-br64-kt32-t128-v1:"
                        + ImageSha256.ToLowerInvariant() + ":ub=" + maximumPrefillRows;
                    Console.Error.WriteLine($"[q35-cuda] explicit legacy BF16 prefill schedule: threads=128, module={ImageSha256}; original arithmetic, ordinary single-sequence prefill only");
                }
                if (verify8Path != null)
                {
                    // Only module admission can fall back. Allocation and every
                    // later launch remain outside the catch; partial execution
                    // must never be retried on another arithmetic policy.
                    try { _verify8 = new CudaNinferPrefillAttentionModule(verify8Path); }
                    catch (Exception error) when (autoVerify8 && Q35CudaEngine.NinferAutomaticModuleUnavailable(error))
                    {
                        Verify8Selection = "asset-unavailable";
                        Console.Error.WriteLine($"[q35-cuda] automatic NInfer Verify8 attention unavailable before use: {error.Message}; using existing attention");
                    }
                    if (_verify8 != null)
                    {
                        _verify8WorkspaceBytes = CudaNinferPrefillAttentionModule.WorkspaceVerify8();
                        CudaDriverApi.cuMemAlloc(out _verify8Workspace, new UIntPtr(checked((ulong)_verify8WorkspaceBytes))).ThrowOnError();
                        Verify8ImageSha256 = _verify8.ImageSha256;
                        Verify8Selection = autoVerify8 ? "auto-sm120-170sm" : "explicit";
                        PrefillNamespaceSuffix += "|ninfer-verify8-attn=" + CudaNinferPrefillAttentionModule.Verify8Policy + ":"
                            + Verify8ImageSha256.ToLowerInvariant() + ":context-cap=40960:rows=8:splits=4:sms=170";
                        Console.Error.WriteLine($"[q35-cuda] {(autoVerify8 ? "automatic" : "explicit")} NInfer Verify8 BF16 attention: selection={Verify8Selection}, option={(autoVerify8 ? Q35CudaEngine.NinferAutoVerify8Environment : Verify8Environment)}, policy={CudaNinferPrefillAttentionModule.Verify8Policy}, module={Verify8ImageSha256}, workspaceBytes={_verify8WorkspaceBytes}, path={verify8Path}; full-valid Verify T8/B1 only, visibleKeys<=40960, device-base graph input");
                    }
                }
            }
            catch
            {
                // Cleanup failures cannot replace the original module/allocation
                // admission error. No kernels or graphs have been published yet.
                if (_verify8Workspace != IntPtr.Zero) CudaDriverApi.cuMemFree(_verify8Workspace);
                _verify8Workspace = IntPtr.Zero;
                try { _verify8?.Dispose(); } catch { }
                _verify8 = null;
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
        internal static string ResolveVerify8Path(string raw, bool verificationEnabled, int maximumRows,
            int maximumKeys, int sms, int heads, int kvHeads, int headDim)
        {
            if (raw == null) return null;
            if (string.IsNullOrWhiteSpace(raw) || !Path.IsPathFullyQualified(raw) || raw != Path.GetFullPath(raw))
                throw new ArgumentException(Verify8Environment + " requires an explicit normalized absolute module path.");
            if (!verificationEnabled || maximumRows < 8 || maximumKeys < 8
                || maximumKeys > CudaNinferPrefillAttentionModule.MaximumContext
                || sms != 170 || heads != 24 || kvHeads != 4 || headDim != 256)
                throw new NotSupportedException("The explicit NInfer Verify8 attention policy requires verification enabled and validated170SM Q24/KV4/HD256 target geometry.");
            if (!File.Exists(raw)) throw new FileNotFoundException("Original BF16 Verify8 attention module is unavailable.", raw);
            return raw;
        }
        internal static bool UseAutomaticVerify8(bool qualifiedAutomatic, string explicitPath)
            => qualifiedAutomatic && explicitPath == null;
        internal static bool UseVerify8(bool enabled, int rows, int sequences, bool isVerify,
            bool allRowsValid, int heads, int kvHeads, int maxVisibleKeys)
            => enabled && heads == 24 && kvHeads == 4
                && CudaNinferPrefillAttentionModule.SupportsVerify8(rows, sequences, isVerify, allRowsValid, maxVisibleKeys);
        internal bool UsesVerify8(int rows, int sequences, bool isVerify, bool allRowsValid,
            int heads, int kvHeads, int maxVisibleKeys)
            => UseVerify8(Verify8Enabled, rows, sequences, isVerify, allRowsValid, heads, kvHeads, maxVisibleKeys);
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
            void** args = stackalloc void*[] { &qkv, &rows, &rowsPerSeq, &nHead, &nKv, &qNorm, &kNorm, &eps,
                &ropePos, &ropeStride, &sectT, &sectH, &sectW, &ropeBase, &cacheRow, &table, &maxBlocks, &kOff, &vOff, &q, &gate, &frequencies };
            Launch(_prep, checked((uint)(((long)rows * (nHead + 2*nKv)*32+255)/256)), 1, 1, 256, stream, args);
        }
        public void Attention(IntPtr q, IntPtr gate, IntPtr keyEnd, IntPtr table, int maxBlocks, long kOff, long vOff,
            int rowsPerSeq, int sequences, int nHead, int nKv, float scale, int maxKeys, IntPtr partial, IntPtr output, IntPtr stream,
            bool isVerify = false, bool allRowsValid = true, IntPtr positionsBaseDev = default)
        {
            if (UsesVerify8(rowsPerSeq, sequences, isVerify, allRowsValid, nHead, nKv, maxKeys))
            {
                // This is the physical cache-row base, not keyEnd's base+1 or
                // an mRoPE coordinate. The core dereferences it at graph replay.
                _verify8.RunF32Verify8(q, gate, table, maxBlocks, kOff, vOff, positionsBaseDev, maxKeys,
                    scale, output, _verify8Workspace, _verify8WorkspaceBytes, stream);
                return;
            }
            if (_prefill != null && rowsPerSeq <= _maximumPrefillRows
                && nHead == CudaNinferPrefillAttentionModule.QueryHeads && nKv == CudaNinferPrefillAttentionModule.KvHeads
                && CudaNinferPrefillAttentionModule.Supports(rowsPerSeq, sequences, isVerify, allRowsValid))
            {
                _prefill.RunF32(q, gate, table, maxBlocks, kOff, vOff, rowsPerSeq, checked(maxKeys - rowsPerSeq),
                    scale, output, _prefillWorkspace, _prefillWorkspaceBytes, _sms, stream);
                return;
            }
            if (rowsPerSeq * (nHead/nKv) <= Q35Kernels.AttnBlockRows && partial != IntPtr.Zero)
            {
                var plan = Q35Kernels.GetSplitAttentionPlan(maxKeys);
                int chunk = plan.Chunk, splits = plan.Splits;
                void** args = stackalloc void*[] { &q, &keyEnd, &table, &maxBlocks, &kOff, &vOff, &rowsPerSeq,
                    &nHead, &nKv, &scale, &chunk, &partial };
                Launch(_split, (uint)splits, (uint)nKv, (uint)sequences, 128, stream, args);
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
        public void Dispose()
        {
            // The engine synchronizes and destroys its layer graphs before
            // disposing this module or any captured workspace pointer.
            if (_verify8 != null)
            {
                Console.Error.WriteLine($"[q35-cuda] NInfer Verify8 BF16 attention core host submissions={_verify8.PagedCoreLaunches}; includes graph capture, excludes GPU graph replays; module={Verify8ImageSha256}");
                _verify8.Dispose(); _verify8 = null;
            }
            if (_verify8Workspace != IntPtr.Zero) CudaDriverApi.cuMemFree(_verify8Workspace).ThrowOnError();
            _verify8Workspace = IntPtr.Zero;
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
