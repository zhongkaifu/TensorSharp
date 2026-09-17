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
        private IntPtr _native = IntPtr.Zero;
        private readonly object _nativeSync = new object();

        /// <summary>True when this instance is driven by the native executor.</summary>
        private bool UsesNativeExecutor => _native != IntPtr.Zero;

        private static bool IsGgmlBackendType(BackendType backend) =>
            backend == BackendType.GgmlCuda || backend == BackendType.GgmlVulkan ||
            backend == BackendType.GgmlMetal || backend == BackendType.GgmlCpu;

        /// <summary>
        /// The base constructor must not build a second GPU context for a backend
        /// the native executor is going to own, and it must not try to page a
        /// 226 GiB checkpoint through the managed weight loader. Coerce those
        /// backends to the lightweight GGML CPU allocator; the native side picks
        /// its own devices from the backend the operator actually asked for.
        /// </summary>
        private static BackendType NormalizeBackend(BackendType backend)
            => NativeRequested(backend) ? BackendType.GgmlCpu : backend;

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
        /// native loader's routed-expert offload policy. Offload stays OFF unless
        /// the operator asks for it: on a host that does have the VRAM it costs
        /// most of the decode throughput for no reason, and a host that genuinely
        /// cannot fit the model gets a load error naming the fewest layers that
        /// would work.
        /// </summary>
        private static int ResolveCpuMoeLayers()
        {
            if (!MoeCpuOffloadConfig.IsExplicitlySet)
                return GgmlGlmNative.CpuMoeNone;
            return MoeCpuOffloadConfig.AllLayers ? int.MaxValue : MoeCpuOffloadConfig.CpuMoeLayers;
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        private void InitNativeExecutor(string ggufPath, BackendType backend, int tpDegree, int maxContext)
        {
            // --tp N means tensor parallelism across N ranks; without it every
            // visible GPU takes a contiguous run of layers.
            int tp = tpDegree > 1 ? tpDegree : 1;
            int nGpu = ParseEnvInt("TS_GLM_NGPU", tp > 1 ? tp : 0);   // 0 = every visible GPU
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

            _native = GgmlGlmNative.LoadModel(ggufPath, nGpu, maxContext, nUbatch, nThreads,
                ResolveCpuMoeLayers(), BackendRegistryName(backend), tp, ctxIsHardLimit,
                NativeMtpRequested());
            if (_native == IntPtr.Zero)
                throw NativeLoadRefused("glm", ggufPath,
                    "TS_GLM_NATIVE=0 selects the per-op path instead of the native executor.");

            _maxContextLength = GgmlGlmNative.CtxSize(_native);
            // The native loader is the authority on whether the draft block
            // actually made it in: a trunk-only checkpoint, or a device that had
            // no room for the extra layer, both come back without one.
            HasDraftHead = GgmlGlmNative.HasDraftHead(_native);
            if (HasDraftHead)
                _mtpLayer = _numTrunkLayers;
            int vocab = GgmlGlmNative.VocabSize(_native);
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
                if (!GgmlGlmNative.Forward(_native, tokens, _logitsBuffer))
                    throw new InvalidOperationException("glm-dsa native forward failed (see stderr).");
                _cacheSeqLen = GgmlGlmNative.NPast(_native);
                return _logitsBuffer;
            }
        }

        private void ResetNative()
        {
            lock (_nativeSync)
            {
                if (!GgmlGlmNative.ResetChecked(_native))
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
                if (!GgmlGlmNative.Rewind(_native, tokenCount))
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
                if (_native != IntPtr.Zero)
                {
                    GgmlGlmNative.Free(_native);
                    _native = IntPtr.Zero;
                }
            }

            // The per-op path's caches are this model's own tensors; the base
            // class only knows about weights, so they have to be released here
            // or they outlive the allocator that backs them.
            DisposeCaches();

            base.Dispose();
        }
    }
}
