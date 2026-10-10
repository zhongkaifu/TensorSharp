// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Native whole-model execution path for glm-dsa.
//
// GLM-5.2 is 744B parameters — 226 GiB even at IQ2_XXS — so on the GGML
// backends the model runs through TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp
// instead of the per-op managed forward: the native executor loads the split
// GGUF itself, spreads the layers across every visible GPU, keeps the MLA and
// indexer caches device-resident, and submits ONE graph per ubatch. The
// managed per-op path stays the reference implementation (and the only path
// for the pure-C# `cpu` and direct-CUDA `cuda` backends), and
// TS_GLM_NATIVE=0 selects it on a GGML backend for an A/B.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>The whole-model executor (native ggml, or the direct-CUDA engine); null on the
        /// managed per-op path.</summary>
        private IGlmExecutor _exec;
        private readonly object _nativeSync = new object();

        /// <summary>True when a whole-model executor drives this instance.</summary>
        private bool UsesNativeExecutor => _exec != null;

        private static bool IsGgmlBackendType(BackendType backend) =>
            backend == BackendType.GgmlCuda || backend == BackendType.GgmlVulkan ||
            backend == BackendType.GgmlMetal || backend == BackendType.GgmlCpu;

        /// <summary>
        /// Validate before the base constructor initializes its single-rank allocator.
        /// Keep the requested backend: the process-wide GGML backend may already be
        /// pinned by host startup or test discovery, and initializing CPU after CUDA
        /// is an error. The native executor owns multi-GPU placement; the managed
        /// base never creates a second TP group or loads the native executor's weights.
        /// </summary>
        private static BackendType ValidateParallelism(BackendType backend, int tpDegree,
            ITensorParallelGroup tpGroup, int layerSplitDegree)
        {
            if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree));
            if (layerSplitDegree < 1) throw new ArgumentOutOfRangeException(nameof(layerSplitDegree));
            if (layerSplitDegree > 1 && (tpDegree > 1 || tpGroup != null))
                throw new ArgumentException("--layer-split and --tp cannot be combined.");
            if (layerSplitDegree > 1 && (!NativeRequested(backend) ||
                backend is not (BackendType.GgmlCuda or BackendType.GgmlVulkan)))
                throw new NotSupportedException("GLM --layer-split requires the native ggml_cuda or ggml_vulkan executor; remove TS_GLM_NATIVE=0.");
            if (NativeRequested(backend) && tpGroup != null)
                throw new NotSupportedException("GLM native tensor parallelism is local/single-process only; remove --tp-node-id/--tp-peers.");
            string shardOverride = Environment.GetEnvironmentVariable("TS_GLM_TP_SHARD");
            if (NativeRequested(backend) && tpDegree > 1 && shardOverride != null &&
                (!int.TryParse(shardOverride, out int shards) || (shards & 3) == 0))
                throw new ArgumentException("--tp requires weight sharding; TS_GLM_TP_SHARD must enable at least one sharded projection. Unset it or choose a nonzero shard mask.");
            // Native tuning must not constrain the independent managed executors.
            // Validate native counts before initializing a backend or opening the GGUF.
            if (NativeRequested(backend))
                ResolveNativeGpuCount(tpDegree, layerSplitDegree);
            return backend;
        }

        internal static int ResolveNativeGpuCount(int tpDegree, int layerSplitDegree)
        {
            int requested = Math.Max(tpDegree, layerSplitDegree);
            // Matches MAX_GPUS in TensorSharp's native GLM executor. Its device
            // scan treats the request as a cap, so refuse an unrepresentable count.
            if (requested > 8)
                throw new NotSupportedException("The native GLM executor supports at most 8 GPUs; reduce --tp or --layer-split.");
            return requested;
        }

        private static bool NativeRequested(BackendType backend)
        {
            if (!IsGgmlBackendType(backend))
                return false;
            return !string.Equals(Environment.GetEnvironmentVariable("TS_GLM_NATIVE"), "0", StringComparison.Ordinal);
        }

        /// <summary>ggml backend registry name whose devices the native executor should use.</summary>
        private static string BackendRegistryName(BackendType backend) => backend switch
        {
            // GGML_CUDA_NAME is "CUDA" / "ROCm" / "MUSA" depending on how ggml-cuda
            // was built; the native side treats them as one family.
            BackendType.GgmlCuda => "CUDA",
            BackendType.GgmlVulkan => "Vulkan",
            BackendType.GgmlMetal => "Metal",
            // An explicit host-only run: without this the native loader's device
            // scan would pick up whatever GPU backend is linked in.
            BackendType.GgmlCpu => "CPU",
            _ => null,
        };

        /// <summary>
        /// Translate the process-wide <see cref="MoeCpuOffloadConfig"/> into the
        /// native loader's routed-expert offload policy. GLM-5.3-Flash on GGML
        /// CUDA defaults to the capacity plan; it keeps all experts on the GPU
        /// when they fit and otherwise offloads only the layers required by the
        /// per-device weight, context and scratch estimate. Explicit zero still
        /// requires a fully resident load. Other executors remain opt-in.
        /// </summary>
        private int ResolveCpuMoeLayers(BackendType backend)
            => MoeCpuOffloadConfig.ResolveNativeLayers(backend == BackendType.GgmlCuda &&
                GlmDsaArchitecture.IsGlm5Next(Config.Architecture));

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        private void InitNativeExecutor(string ggufPath, BackendType backend, int tpDegree, int layerSplitDegree, int maxContext)
        {
            // --tp shards weights; --layer-split only changes whole-layer placement.
            int tp = tpDegree > 1 ? tpDegree : 1;
            _nativeTp = tp;
            int nGpu = ResolveNativeGpuCount(tpDegree, layerSplitDegree);
            int requested = Math.Max(tpDegree, layerSplitDegree);
            if (requested > 1 && backend is BackendType.GgmlCuda or BackendType.GgmlVulkan)
            {
                var kind = backend == BackendType.GgmlCuda ? GgmlBackendType.Cuda : GgmlBackendType.Vulkan;
                int available = GgmlBasicOps.GetGpuDeviceCount(kind);
                if (available < requested)
                    throw new InvalidOperationException($"Requested {requested} GPU(s) but the GGML {kind} backend sees only {available}.");
            }
            // 1024, not llama.cpp's 512: the MoE expert GEMMs pad their tiles to
            // the number of rows routed to each expert, and with 256 experts at
            // top-8 a 512-token chunk leaves only ~16 rows per expert, so half the
            // tile is padding. Doubling the chunk measured pp2048 at 984 vs 696
            // tok/s on 3x RTX PRO 6000 with no change to decode. Raise it further
            // with TS_GLM_UBATCH when the prompts are long and VRAM allows.
            int nUbatch = ParseEnvInt("TS_GLM_UBATCH", 1024);
            _nativeUbatch = nUbatch;
            int nThreads = ParseEnvInt("TS_GLM_THREADS", Math.Min(Environment.ProcessorCount, 32));

            // GLM-5.2 advertises a 1M-token context, which is ~93 GiB of KV cache
            // across 78 layers — a whole card's worth on top of the weights.
            // Unless MAX_CONTEXT named
            // the number, the loader treats it as a ceiling and caps it to what
            // the devices hold, rather than loading 200 GiB and only then failing
            // to allocate the first sequence slot.
            bool ctxIsHardLimit = !string.IsNullOrWhiteSpace(
                Environment.GetEnvironmentVariable("MAX_CONTEXT"));

            IntPtr handle = GgmlGlmNative.LoadModel(ggufPath, nGpu, maxContext, nUbatch, nThreads,
                ResolveCpuMoeLayers(backend), BackendRegistryName(backend), tp, ctxIsHardLimit,
                NativeMtpRequested());
            if (handle == IntPtr.Zero)
                throw NativeLoadRefused("glm", ggufPath,
                    "TS_GLM_NATIVE=0 selects the per-op path instead of the native executor.");
            _exec = new NativeGlmExecutor(handle);

            _maxContextLength = _exec.ContextSize;
            // The native loader is the authority on whether the draft block
            // actually made it in: a trunk-only checkpoint, or a device that had
            // no room for the extra layer, both come back without one.
            HasDraftHead = _exec.HasDraftHead;
            if (HasDraftHead)
                _mtpLayer = _numTrunkLayers;
            int vocab = _exec.VocabSize;
            if (vocab > 0)
                Config.VocabSize = vocab;
            _logitsBuffer = new float[Config.VocabSize];
        }

        private float[] ForwardNative(int[] tokens)
        {
            lock (_nativeSync)
            {
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                BeforeGraphCall();
                bool ok;
                try { ok = _exec.Forward(tokens, _logitsBuffer); }
                finally { AfterGraphCall(); }
                if (!ok)
                    throw new InvalidOperationException("glm-dsa native forward failed (see stderr).");
                _cacheSeqLen = _exec.NPast;
                return _logitsBuffer;
            }
        }

        private void ResetNative()
        {
            lock (_nativeSync)
            {
                if (!_exec.ResetChecked())
                    throw new InvalidOperationException("glm-dsa native reset failed; slot remains unusable.");
                _cacheSeqLen = 0;
            }
        }

        /// <summary>
        /// The native executor rewinds by position only — the MLA and indexer
        /// caches are plain per-position rows, so dropping the tail is exact — but
        /// it can still REFUSE: a target past the slot's head, any rewind on
        /// glm5next other than to 0 or to the head (its KDA recurrence cannot go
        /// back), or a slot whose KDA restore failed. The refusal is reported, never
        /// swallowed. This used to be a void override, so the base
        /// <see cref="ModelBase.TryTruncateKVCache"/> answered true with the head
        /// where it was, and the caller decoded the rest of the turn against
        /// positions it believed it had dropped.
        /// </summary>
        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            if (!UsesNativeExecutor)
                return base.TryTruncateKVCacheCore(tokenCount);
            lock (_nativeSync)
            {
                if (!_exec.Rewind(tokenCount))
                    return false;
                _cacheSeqLen = tokenCount;
                return true;
            }
        }

        /// <summary>
        /// The non-refusable form, for callers with no re-prefill fallback. A
        /// native refusal throws rather than returning with the head where it was
        /// (the contract DeepSeek V4.1 and Gemma 4 keep).
        /// </summary>
        protected override void TruncateKVCacheCore(int tokenCount)
        {
            if (!UsesNativeExecutor)
            {
                base.TruncateKVCacheCore(tokenCount);
                return;
            }
            if (!TryTruncateKVCacheCore(tokenCount))
            {
                throw new InvalidOperationException(
                    $"{Config.Architecture} cannot truncate its KV cache to {tokenCount} tokens " +
                    $"(head at {_cacheSeqLen}). Use TryTruncateKVCache and re-prefill when it declines.");
            }
        }

        public override void WarmUpKernels()
        {
            if (!UsesNativeExecutor)
            {
                base.WarmUpKernels();
                return;
            }
            // One tiny forward allocates the scheduler's compute buffers so the
            // first real request does not pay for them mid-prompt.
            try
            {
                int bos = Tokenizer?.BosTokenId ?? 0;
                ForwardNative(new[] { bos });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[glm] warmup forward failed: {ex.Message}");
            }
            finally
            {
                ResetNative();
            }
        }

        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            lock (_nativeSync)
            {
                _exec?.Dispose();
                _exec = null;
            }

            // The per-op path's caches are this model's own tensors; the base
            // class only knows about weights, so they have to be released here
            // or they outlive the allocator that backs them.
            DisposeCaches();

            base.Dispose();
        }
    }
}
