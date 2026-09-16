// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DeepSeek V4 (Flash) — driver for the native whole-model executor.
//
// The architecture (4-stream hyper-connections, shared single-KV-head
// attention with sinks and inverse-RoPE outputs, per-layer raw/CSA/HCA
// compressed attention with a lightning indexer, sqrt-softplus MoE with hash
// routing) is implemented natively in
// TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp as one ggml graph per
// ubatch, scheduled across every visible GPU (layer split) so models larger
// than a single GPU's VRAM are hosted across all of them.
//
// This class parses metadata/tokenizer from the (first) GGUF shard, feeds
// token batches to the native executor, and returns last-token logits. KV
// truncation is unsupported (compressed caches ratchet forward), so
// multi-turn prompts re-prefill; the engine handles that automatically.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : ModelBase, IBatchedPagedModel
    {
        private IntPtr _handle;
        private DeepSeek4CpuExecutor _cpuExec;
        private DeepSeek4CudaExecutor _cudaExec;
        private readonly object _sync = new object();
        // The multiple a native KV truncation target must be, or 0 when this load cannot
        // truncate at all (plain V4, or a V4.1 served by the direct-CUDA or pure-C# executor
        // rather than the native one). Resolved once: it is a property of the checkpoint's
        // compression ratios, and SupportsKVCacheTruncation is read on hot scheduler paths.
        private readonly int _truncateAlign;
        protected IntPtr NativeHandle => _handle;
        protected object NativeSync => _sync;

        public DeepSeek4Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null)
            : base(ggufPath, NormalizeBackend(backend), 1, null)
        {
            string arch = _gguf.GetString("general.architecture") ?? "deepseek4";
            bool isV41 = string.Equals(arch, "deepseek41", StringComparison.Ordinal);
            if (isV41)
            {
                DeepSeek41Architecture.ValidateLoad(ggufPath, backend, ResolveDsparkPath(draftModelPath), tpDegree, tpGroup);
                // Once per load, and only here: ValidateLoad also runs as the
                // descriptor's ApplyNativeTunables, so warning from inside it
                // would print the same line twice.
                if (DeepSeek41Architecture.DescribeCpuBackendChoice(backend) is string cpuNote)
                    Console.Error.WriteLine(cpuNote);
            }
            else if (DescribeCpuBackendChoice(backend) is string v4CpuNote)
            {
                Console.Error.WriteLine(v4CpuNote);
            }
            Config = new ModelConfig { Architecture = arch };
            // V4.1 was refused in ValidateLoad above (and once more before the
            // factory ran); plain V4 has no pre-load hook, so refuse here.
            if (!isV41)
                DeepSeek4Architecture.RefuseBlockQuantizedKvCache("DeepSeek V4 (Flash)", v41: false);
            // Every executor of this family keeps F16 caches and ignores the
            // process-wide dtype, so report what is actually allocated rather
            // than whatever KV_CACHE_DTYPE happened to say.
            _kvCacheDtype = DeepSeek4Architecture.ExecutorKvCacheDtype;
            if (KvCacheDtypeConfig.IsExplicitlySet && KvCacheDtypeConfig.Current == KvCacheDtype.F32)
                Console.Error.WriteLine(DeepSeek4Architecture.DescribeF32KvCacheRequest(
                    isV41 ? "DeepSeek V4.1 Flash" : "DeepSeek V4 (Flash)"));
            ParseBaseConfig();
            if (isV41)
            {
                Config.NumExperts = (int)_gguf.GetUint32($"{arch}.expert_count");
                Config.NumExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count");
                Config.IntermediateSize = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length");
                Config.SlidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window");
                Config.OriginalContextLength = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length");
            }
            ParseTokenizer();

            int maxContext = ResolveConfiguredContextLength();
            // The GGUF advertises 1M context; cache rows scale with n_ctx, so keep a
            // practical default unless the operator asks for more via MAX_CONTEXT.
            string ctxEnv = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            if (string.IsNullOrWhiteSpace(ctxEnv))
                maxContext = Math.Min(maxContext, 65536);
            _maxContextLength = maxContext;

            // GPU prefill chunks amortize the MoE expert-GEMM tile padding (256
            // experts x top-6 leaves ~nt/42 rows per expert, so per-chunk MoE
            // cost is nearly flat in nt): 1024 measured ~11-21% faster overall
            // prefill than 512 and also halves what a non-multiple tail chunk
            // costs relative to the whole prompt. The CPU executor stays at 512
            // (activation memory bound, no tile padding to amortize).
            int nUbatch = ParseEnvInt("TS_DSV4_UBATCH", isV41 ? 256 : backend == BackendType.Cpu ? 512 : 1024);

            if (backend == BackendType.Cuda)
            {
                // Direct-CUDA whole-model executor: quantized weights resident in
                // device memory, layer-split across the visible GPUs, driver-API
                // kernels only (no ggml).
                int nGpu = ParseEnvInt("TS_DSV4_NGPU", tpDegree > 1 ? tpDegree : 0); // 0 = all visible GPUs
                string dspark = ResolveDsparkPath(draftModelPath);
                Console.WriteLine($"Model: {arch} (direct-CUDA whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}" +
                    (dspark != null ? ", DSpark drafter" : string.Empty));
                _cudaExec = new DeepSeek4CudaExecutor(ggufPath, maxContext, nUbatch, nGpu, dspark, ResolveCpuMoeLayers());
            }
            else if (_backend == BackendType.Cpu)
            {
                // Pure C# whole-model executor: quantized weights served straight
                // from the memory-mapped GGUF shards, managed SIMD kernels only.
                WarnDsparkUnavailable(draftModelPath, backend);
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Environment.ProcessorCount);
                Console.WriteLine($"Model: {arch} (pure C# CPU executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}");
                _cpuExec = new DeepSeek4CpuExecutor(ggufPath, maxContext, nUbatch, nThreads, _allocator);
            }
            else
            {
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Math.Min(Environment.ProcessorCount, 32));
                int nGpu = ParseEnvInt("TS_DSV4_NGPU", tpDegree > 1 ? tpDegree : 0); // 0 = all visible GPUs
                string dspark = backend == BackendType.GgmlCuda || (isV41 && backend == BackendType.GgmlCpu)
                    ? ResolveDsparkPath(draftModelPath) : null;
                if (dspark == null)
                    WarnDsparkUnavailable(draftModelPath, backend);

                Console.WriteLine($"Model: {arch} (native whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}" +
                    (dspark != null ? ", DSpark drafter" : string.Empty));

                int nCpuMoe = ResolveCpuMoeLayers();
                // `backend`, not `_backend`: the base ctor coerces everything to
                // GgmlCpu so no second GPU context is created, but the native
                // executor still has to pick its devices from the backend the
                // operator actually asked for.
                string backendName = BackendRegistryName(backend);
                _handle = dspark != null
                    ? GgmlDeepSeek4Native.LoadModelWithDspark(ggufPath, nGpu, maxContext, nUbatch, nThreads, dspark, nCpuMoe, backendName)
                    : GgmlDeepSeek4Native.LoadModel(ggufPath, nGpu, maxContext, nUbatch, nThreads, nCpuMoe, backendName);
                _nativeUBatch = nUbatch;
                _nativeDsparkBlock = _handle != IntPtr.Zero && dspark != null
                    ? GgmlDeepSeek4Native.DsparkBlockSize(_handle) : 0;
                if (_handle == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to load {arch} model from {ggufPath} (see stderr for details).");
                // Zero for plain V4: its compressor overlaps blocks, so a rewind reads state
                // rows an aligned target does not protect, and the native side declines.
                _truncateAlign = GgmlDeepSeek4Native.TruncateAlign(_handle);
            }
        }

        /// <summary>
        /// The one line a DeepSeek V4 load on the ggml CPU backend prints, null
        /// for every other backend.
        ///
        /// <para>Until <see cref="BackendRegistryName"/> learned "CPU",
        /// <c>ggml_cpu</c> reached the native loader as "any GPU" and a
        /// CUDA-capable host ran the GPUs. It now runs where it says, which is
        /// orders of magnitude slower for the same launch line — and this is the
        /// backend the server picks when <c>--backend</c> is omitted off macOS,
        /// so a V4 script that never named one changes behavior. The device list
        /// that follows says CPU but not that it used to say GPU, so say it
        /// here. V4.1 prints its own, longer note instead (see
        /// DeepSeek41Architecture.DescribeCpuBackendChoice).</para>
        /// </summary>
        internal static string DescribeCpuBackendChoice(BackendType backend)
            => backend != BackendType.GgmlCpu ? null
                : "[dsv4] --backend ggml_cpu: DeepSeek V4 will run on ONE CPU device, not on any GPU this host " +
                  "has. This is also the backend chosen when --backend is omitted off macOS. Pass --backend " +
                  "ggml_cuda or --backend cuda for a GPU executor.";

        /// <summary>
        /// Translate the process-wide <see cref="MoeCpuOffloadConfig"/> into the
        /// native loader's routed-expert offload policy.
        ///
        /// <para>Offload is OFF unless the operator asks for it, exactly as it is
        /// for every other architecture. DeepSeek V4 used to default to an
        /// automatic spill because a Q8_K_XL checkpoint (151 GiB) outweighs most
        /// hosts, but choosing that silently is the wrong trade on a host that
        /// *does* have the VRAM: it moves ~29 GiB of experts to the CPU and costs
        /// most of the decode throughput for no reason. A host that genuinely
        /// cannot fit the model now gets a load error naming the fewest layers
        /// that would (see the "[dsv4] does not fit" message), which is a better
        /// answer than a silent slowdown or an out-of-memory abort.</para>
        /// </summary>
        private static int ResolveCpuMoeLayers()
        {
            if (!MoeCpuOffloadConfig.IsExplicitlySet)
                return 0;
            return MoeCpuOffloadConfig.AllLayers ? int.MaxValue : MoeCpuOffloadConfig.CpuMoeLayers;
        }

        /// <summary>
        /// ggml backend registry name whose devices the native executor should
        /// run on. GgmlOps links every backend it was built with and the loader
        /// enumerates devices in registration order, so without this a
        /// CUDA-capable box runs <c>--backend ggml_vulkan</c> on its CUDA
        /// devices and the Vulkan path is never exercised. Null = any GPU.
        /// </summary>
        private protected static string BackendRegistryName(BackendType backend) => backend switch
        {
            // GGML_CUDA_NAME is "CUDA" / "ROCm" / "MUSA" depending on how
            // ggml-cuda was built; the native side treats them as one family.
            BackendType.GgmlCuda => "CUDA",
            BackendType.GgmlVulkan => "Vulkan",
            BackendType.GgmlMetal => "Metal",
            // "CPU" is the one name that reaches the loader's cpu_only branch,
            // where it builds a single CPU compute device instead of
            // enumerating accelerators. Without it --backend ggml_cpu fell into
            // the null case, which means "any GPU": on a CUDA box the operator
            // asked for the CPU and silently got the GPUs, and on a host with
            // no GPU at all the load failed with "refusing CPU-only run".
            BackendType.GgmlCpu => "CPU",
            _ => null,
        };

        /// <summary>Draft block size of the native (ggml) executor's drafter,
        /// 0 when none is loaded.</summary>
        private int _nativeDsparkBlock;

        /// <summary>Prefill micro-batch of the native executor.</summary>
        private int _nativeUBatch = 512;

        /// <summary>
        /// The DSpark drafter is implemented inside the direct-CUDA engine
        /// (Dsv4CudaEngine.Dspark.cs), on top of that engine's own kernels and
        /// cache rings. Every other executor serves plain decode, so say so
        /// instead of silently ignoring the drafter the operator asked for.
        /// </summary>
        private static void WarnDsparkUnavailable(string draftModelPath, BackendType backend)
        {
            if (ResolveDsparkPath(draftModelPath) == null)
                return;
            Console.Error.WriteLine(
                $"[dsv4] a DSpark drafter was configured but the {backend} backend has no speculative " +
                "path for DeepSeek V4 (it is implemented in the direct-CUDA and ggml_cuda engines); " +
                "decoding without it.");
        }

        private static BackendType NormalizeBackend(BackendType backend)
        {
            // BackendType.Cpu runs the pure C# executor. BackendType.Cuda runs
            // the direct-CUDA executor, which owns its own per-device contexts —
            // the base class only needs a lightweight CPU allocator, so it is
            // coerced to Cpu (the ctor dispatches on the ORIGINAL backend). The
            // other backends drive the native ggml executor; the managed side
            // only needs the GGML library loaded, so everything else is coerced
            // to GgmlCpu so no second GPU context is created.
            return backend switch
            {
                BackendType.Cpu => BackendType.Cpu,
                BackendType.Cuda => BackendType.Cpu,
                BackendType.GgmlCuda => BackendType.GgmlCuda,
                BackendType.GgmlVulkan => BackendType.GgmlVulkan,
                _ => BackendType.GgmlCpu,
            };
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        /// <summary>
        /// Partial KV reuse, on the V4.1 native executor only.
        ///
        /// <para>Why it matters here: V4.1's chat protocol re-renders a past assistant
        /// turn with its reasoning removed (ordinary chat drops it, per DeepSeek's
        /// reference encoder), so from the second turn on the rendered prompt diverges
        /// from the cache exactly one token after the previous turn's
        /// <c>&lt;|Assistant|&gt;</c> - and everything before that point is the WHOLE of
        /// the previous prompt. Without truncation the planner throws all of it away and
        /// re-prefills, which turns per-turn prefill into a function of the entire
        /// conversation instead of the newest answer.</para>
        ///
        /// <para>Why only V4.1, and only native: the native executor can rewind past its
        /// raw sliding-window ring because every slot carries a checkpoint of the modular
        /// rings taken at the last prompt boundary (TSGgml_Dsv4Truncate). Plain V4
        /// compresses OVERLAPPING blocks, so a boundary still reads the previous block's
        /// state rows and aligning the head to the ratio is not sufficient; the direct-CUDA
        /// and pure-C# V4.1 executors have no such checkpoint, and their position rewind
        /// (Dsv4CudaEngine.Rewind) is sized for a rejected speculative block, not a
        /// conversational one. Those three keep re-prefilling, which is correct, just not
        /// cheap.</para>
        /// </summary>
        public override bool SupportsKVCacheTruncation => _truncateAlign > 0;

        /// <summary>The compression-block alignment the native truncate requires (the lcm
        /// of the per-layer compress ratios: 2 for the released checkpoint).</summary>
        public override int KVCacheTruncationGranularity => Math.Max(1, _truncateAlign);

        /// <summary>
        /// Refusable truncation. A refusal is normal - it means the target is further back
        /// than the slot's checkpoint can reach - and the caller resets and re-prefills.
        /// </summary>
        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            lock (_sync)
            {
                if (!SupportsKVCacheTruncation) return false;
                return GgmlDeepSeek4Native.Truncate(_handle, tokenCount);
            }
        }

        /// <summary>
        /// The non-refusable form, for callers with no re-prefill fallback. It throws
        /// rather than returning with the head where it was: continuing from a stale head
        /// answers from positions the caller believes it dropped, and nothing downstream
        /// could tell.
        /// </summary>
        protected override void TruncateKVCacheCore(int tokenCount)
        {
            if (!TryTruncateKVCacheCore(tokenCount))
            {
                throw new InvalidOperationException(
                    $"DeepSeek {Config.Architecture} cannot truncate its KV cache to {tokenCount} tokens " +
                    $"(head at {CacheSeqLen}). Use TryTruncateKVCache and re-prefill when it declines.");
            }
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_sync)
            {
                var logits = new float[Config.VocabSize];
                if (_cudaExec != null)
                {
                    _cudaExec.Forward(tokens, logits);
                }
                else if (_cpuExec != null)
                {
                    _cpuExec.Forward(tokens, logits);
                }
                else if (!GgmlDeepSeek4Native.Forward(_handle, tokens, logits))
                {
                    throw new InvalidOperationException("DeepSeek V4 native forward failed (see stderr).");
                }
                return logits;
            }
        }

        protected override void ResetKVCacheCore()
        {
            lock (_sync)
            {
                if (_cudaExec != null)
                    _cudaExec.Reset();
                else if (_cpuExec != null)
                    _cpuExec.Reset();
                else if (_handle != IntPtr.Zero)
                {
                    if (!GgmlDeepSeek4Native.ResetChecked(_handle))
                        throw new InvalidOperationException("DSV4 native cache reset failed; cache remains unusable.");
                }
            }
        }

        public override void WarmUpKernels()
        {
            // One tiny forward allocates the scheduler's compute buffers so the
            // first user request does not pay the allocation cost.
            try
            {
                int bos = Tokenizer?.BosTokenId ?? 0;
                ForwardCore(new[] { bos });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[dsv4] warmup forward failed: {ex.Message}");
            }
            finally
            {
                ResetKVCacheCore();
            }
        }

        public override void Dispose()
        {
            lock (_sync)
            {
                if (_cudaExec != null)
                {
                    _cudaExec.Dispose();
                    _cudaExec = null;
                }
                if (_cpuExec != null)
                {
                    _cpuExec.Dispose();
                    _cpuExec = null;
                }
                if (_handle != IntPtr.Zero)
                {
                    GgmlDeepSeek4Native.Free(_handle);
                    _handle = IntPtr.Zero;
                }
            }
            base.Dispose();
        }
    }
}
