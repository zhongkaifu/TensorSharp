// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>V4.1 selects its own native graph; it cannot run the V4 graph.</summary>
    internal static class DeepSeek41Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "deepseek41",
            DisplayName = "DeepSeek V4.1 Flash",
            Aliases = new[] { "deepseek41" },
            MultiGpu = MultiGpuMode.TensorParallel,
            DistributedTensorParallelBackends = new[] { BackendType.GgmlCuda },
            SupportsLayerSplit = true,
            SupportsDistributedLayerSplit = true,
            // ggml_cuda's stage graph, and the direct-CUDA engine's stages
            // (Dsv4CudaEngine.Distributed.cs). Tensor parallelism stays ggml_cuda only.
            DistributedLayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.Cuda },
            MultiGpuLimitation = "DeepSeek V4.1 uses native routed-MoE tensor parallelism; attention and shared experts retain local layer placement.",
            SupportsNativeTensorParallel = (degree, backend) => backend == BackendType.GgmlCuda &&
                degree is >= 2 and <= 8,
            DescribeMultiGpuPlacement = DescribePlacement,
            LayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.Cuda },
            ApplyNativeTunables = c => ValidateLoad(c.GgufPath, c.Backend, c.DraftModelPath, c.TpGroup, c.TpDegree, c.Probe),
            ProjectorFileHints = new[] { "deepseek41.vision.gguf" },
            Factory = c => new DeepSeek41Model(c.GgufPath, c.Backend,
                c.TpDegree, c.TpGroup, c.DraftModelPath, c.LayerSplitDegree),
        };

        /// <param name="header">The checkpoint's already-open header, when the caller has one
        /// (the registry's probe, the model's own GGUF); otherwise the first shard's header is read
        /// from <paramref name="ggufPath"/> for the checks that need it, if that file exists.</param>
        internal static void ValidateLoad(string ggufPath, BackendType backend, string draftModelPath,
            ITensorParallelGroup tpGroup = null, int tpDegree = 1, GgufFile header = null)
        {
            // Every backend that has a V4.1 implementation is allowed here:
            // ggml_cuda (the ggml graph), `cuda` (the direct-CUDA engine, which
            // owns its own kernels and does not go through ggml at all),
            // ggml_cpu (the ggml cpu_only branch, where the architecture's fused
            // ops fall back to their scalar implementations) and `cpu` (the pure
            // C# executor). Only ggml_cuda is a serving path; the other three
            // are correctness and portability paths (the direct-CUDA engine is
            // not yet held to a numerical gate).
            // A non-CUDA ggml GPU backend can run V4.1 the same way ggml_cpu
            // does -- its architecture-specific ops fall to the CPU backend's
            // scalar implementations -- but at a host round trip per occurrence.
            // It is opt-in for the same reason the native loader makes it
            // opt-in: what was originally refused was a SILENT fallback onto
            // whichever GPU enumerated first, not an explicit request.
            bool nonCudaGpuAllowed =
                Environment.GetEnvironmentVariable("TS_DSV41_ALLOW_NON_CUDA_GPU") == "1" &&
                (backend == BackendType.GgmlVulkan || backend == BackendType.GgmlMetal);
            if (backend != BackendType.GgmlCuda && backend != BackendType.Cuda &&
                backend != BackendType.GgmlCpu && backend != BackendType.Cpu && !nonCudaGpuAllowed)
                throw new NotSupportedException(
                    "DeepSeek V4.1 Flash requires --backend ggml_cuda (the serving backend), or " +
                    "--backend cuda (the direct-CUDA engine, not yet held to a numerical gate), --backend ggml_cpu " +
                    "or --backend cpu (correctness and portability paths only). " +
                    "TS_DSV41_ALLOW_NON_CUDA_GPU=1 additionally permits ggml_vulkan/ggml_metal, whose "
                    + "architecture-specific ops run on the CPU backend.");
            // A q8_0/q4_0 cache cannot be honoured on any V4.1 executor; refuse it
            // here, before the checkpoint is opened, instead of loading with F16
            // caches while reporting the requested dtype (which is what happened).
            DeepSeek4Architecture.RefuseBlockQuantizedKvCache("DeepSeek V4.1 Flash", v41: true);
            ValidateTensorParallelGroup(backend, tpDegree, tpGroup);
            if (tpGroup?.NodeCount > 1 && !string.IsNullOrWhiteSpace(draftModelPath))
                throw new NotSupportedException(tpGroup is ILayerSplitGroup
                    ? "Distributed DeepSeek V4.1 layer split does not mirror DSpark operations; omit --draft-model."
                    : "Distributed DeepSeek V4.1 tensor parallelism does not mirror DSpark operations; omit --draft-model.");
            if (!string.IsNullOrWhiteSpace(draftModelPath) &&
                backend != BackendType.GgmlCuda && backend != BackendType.GgmlCpu)
                throw new NotSupportedException(
                    "DeepSeek V4.1 DSpark requires the TensorSharp ggml executor (--backend ggml_cuda or ggml_cpu) " +
                    "and a matching deepseek41-dspark drafter. Other executors do not implement the V4.1 draft graph.");
            string draft = DeepSeek4Model.ResolveDsparkPath(draftModelPath);
            if (draft != null)
            {
                using var file = GgufFile.OpenWithoutSiblingShards(draft);
                ValidateDsparkDraft(file.GetString("general.architecture"), file.Metadata.ContainsKey);
            }

            // Routed-MoE TP shards expert dimensions across GPUs, so the native
            // loader refuses it under cpu_only. Say so here instead, before a
            // 246 GiB checkpoint is opened.
            if (ResolveTensorParallelRanks(tpDegree) > 0 && backend != BackendType.GgmlCuda)
                throw new NotSupportedException(
                    "DeepSeek V4.1 routed-MoE tensor parallelism requires --backend ggml_cuda. " +
                    "Select ggml_cuda, or omit --tp.");

            // TS_DSV41_ENGRAM_DEVICE picks where the Engram tables live. The
            // native loader reads it only off cpu_only, so on this backend `=1`
            // — which means "require GPU residency and fail if it does not
            // fit" — would be accepted and then ignored, and a malformed
            // value that ggml_cuda rejects would pass. Refuse both here, with
            // the loader's own wording for the malformed case. `=0` asks for
            // the host mappings this path already uses, so it is allowed.
            if ((backend == BackendType.GgmlCpu || backend == BackendType.Cpu) &&
                Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_DEVICE") is string engramDevice &&
                engramDevice != "0")
            {
                string name = backend == BackendType.Cpu ? "cpu" : "ggml_cpu";
                throw new NotSupportedException(engramDevice == "1"
                    ? "TS_DSV41_ENGRAM_DEVICE=1 needs a GPU to place the Engram tables on and cannot be combined " +
                      $"with --backend {name}, where they are always host mappings. Unset it or run on ggml_cuda."
                    : "TS_DSV41_ENGRAM_DEVICE must be 0 or 1 (unset selects automatically).");
            }

            // YOCO bounded replay (TS_DSV41_YOCO_KEEP, on by default) lives in
            // the native ggml executor. The direct-CUDA engine and the pure C#
            // executor run every layer over every position, so an explicit
            // non-zero value would be accepted and then ignored there. `=0`
            // asks for what those paths already do, so it is allowed.
            if ((backend == BackendType.Cuda || backend == BackendType.Cpu) &&
                Environment.GetEnvironmentVariable(YocoKeepVariable) is string yocoKeep && yocoKeep != "0")
            {
                string name = backend == BackendType.Cuda ? "cuda" : "cpu";
                throw new NotSupportedException(
                    $"{YocoKeepVariable}={yocoKeep}: YOCO bounded replay is implemented by the ggml executor " +
                    $"(--backend ggml_cuda or ggml_cpu), not by --backend {name}, which runs every layer over every " +
                    $"position. Unset it, set it to 0, or run on ggml_cuda.");
            }

            // Last, because it is the one check that reads the header.
            if (backend == BackendType.Cuda || backend == BackendType.Cpu)
            {
                if (header != null)
                    RefuseUnreadableEngramLayout(backend, header);
                else if (File.Exists(ggufPath))
                {
                    using var file = GgufFile.OpenWithoutSiblingShards(ggufPath);
                    RefuseUnreadableEngramLayout(backend, file);
                }
            }
        }

        /// <summary>
        /// llama.cpp's Engram layout, which unsloth's DeepSeek-V4.1-Flash-1017 GGUFs use: one
        /// combined <c>per_layer_token_embd.weight</c> table described by <c>deepseek41.ple.*</c>
        /// metadata, with <c>blk.N.engram_kv</c> and the fused <c>blk.N.engram_gate</c> (q*k) in
        /// place of TensorSharp's <c>engram_wkv</c>/<c>engram_q</c>/<c>engram_k</c>. Decided as the
        /// native loader decides (has_combined_engram_metadata in dsv41_engram_gguf.h): the
        /// <c>ple</c> layer list is present and TensorSharp's <c>deepseek41.engram.layer_ids</c>,
        /// which wins when a header carries both key sets, is not.
        /// </summary>
        internal static bool UsesCombinedEngramLayout(GgufFile header) => Dsv41EngramData.HasCombinedMetadata(header);

        /// <summary>
        /// Every V4.1 executor reads both Engram layouts: the managed ones -- the direct-CUDA engine
        /// and the pure C# one -- through Dsv41EngramData, which exposes the combined table's
        /// per-layer slices as views the way the native loader does. What none of them can read is
        /// a header that commits to neither layout or mixes them: llama.cpp's combined table or
        /// <c>ple</c> keys without the <c>deepseek41.ple.layers</c> list that addresses the table,
        /// or a combined table beside a per-layer one. The managed executors used to fail inside
        /// the executor on those with a bare metadata key name; refuse them from the header
        /// instead, before a 300 GB checkpoint is mapped. Only the opened file's own tensors are
        /// visible here (shard 1 of a split checkpoint usually has none); Dsv41EngramData.Load
        /// repeats the table checks against every shard.
        /// </summary>
        internal static void RefuseUnreadableEngramLayout(BackendType backend, GgufFile header)
        {
            if (backend != BackendType.Cuda && backend != BackendType.Cpu)
                return;
            bool tensorSharp = header.Metadata.ContainsKey("deepseek41.engram.layer_ids");
            bool combined = UsesCombinedEngramLayout(header);
            bool table = header.Tensors.ContainsKey(Dsv41EngramData.CombinedTableName);
            bool perLayerTable = header.Tensors.Keys.Any(name =>
                name.StartsWith("blk.", StringComparison.Ordinal) &&
                name.EndsWith(".engram_embd.weight", StringComparison.Ordinal));
            bool pleKeys = header.Metadata.Keys.Any(key => key.StartsWith("deepseek41.ple.", StringComparison.Ordinal));
            string problem =
                !tensorSharp && !combined && (table || pleKeys)
                    ? "llama.cpp's combined Engram layout (deepseek41.ple.* metadata or a per_layer_token_embd.weight " +
                      "table) without the deepseek41.ple.layers list that addresses the table, and no TensorSharp " +
                      "deepseek41.engram.layer_ids either"
                : tensorSharp && table
                    ? "both Engram layouts: TensorSharp's per-layer metadata (deepseek41.engram.layer_ids) and " +
                      "llama.cpp's combined per_layer_token_embd.weight table"
                : combined && perLayerTable
                    ? "both Engram layouts: llama.cpp's combined metadata (deepseek41.ple.*) and a TensorSharp " +
                      "per-layer blk.N.engram_embd.weight table"
                : null;
            if (problem == null)
                return;
            string name = backend == BackendType.Cuda ? "cuda" : "cpu";
            string executor = backend == BackendType.Cuda ? "the direct-CUDA engine" : "the pure C# executor";
            throw new NotSupportedException(
                $"This DeepSeek V4.1 GGUF has {problem}. --backend {name} ({executor}) reads either TensorSharp's " +
                "per-layer Engram layout or llama.cpp's combined one (unsloth's DeepSeek-V4.1-Flash-1017 GGUFs), " +
                "never a mix or a part of one, so it cannot tell which rows belong to which Engram layer.");
        }

        /// <summary>
        /// Null unless the run is about to put V4.1 on the CPU backend, in which
        /// case the operator gets told. Emitted by the DeepSeek4Model
        /// constructor rather than from ValidateLoad, because ValidateLoad runs
        /// twice per load (once as the descriptor's ApplyNativeTunables, once in
        /// the constructor) and a warning printed twice reads like two problems.
        ///
        /// <para>This exists because the server's default backend is
        /// <c>ggml_cpu</c> on everything but macOS, so forgetting
        /// <c>--backend ggml_cuda</c> on a GPU box used to be a loud refusal and
        /// is now a load that succeeds. Succeeding is the point of the CPU path,
        /// but succeeding into an unusable 246 GiB scalar decode without a word
        /// would be worse than the refusal it replaced.</para>
        /// </summary>
        internal static string DescribeCpuBackendChoice(BackendType backend) => backend switch
        {
            BackendType.GgmlCpu =>
                "[dsv41] --backend ggml_cpu: DeepSeek V4.1 will run on ONE CPU device, with the scalar CPU " +
                "implementations of its architecture-specific ops. That is a correctness and portability path, " +
                "not a serving path: every decoded token reads six of 384 routed experts in each layer. " +
                "Pass --backend ggml_cuda for the GPU executor (this backend is also the default when --backend " +
                "is omitted off macOS). TS_DSV4_THREADS sets the compute thread count; it defaults to at most 32.",
            BackendType.Cpu =>
                "[dsv41] --backend cpu: DeepSeek V4.1 will run on the pure C# executor, with no ggml and no GPU. " +
                "That is a correctness and portability path, not a serving path: every decoded token reads six of " +
                "384 routed experts in each layer, straight out of the memory-mapped shards. Pass --backend " +
                "ggml_cuda for the GPU executor. TS_DSV4_THREADS sets the compute thread count.",
            _ => null,
        };

        /// <summary>The native executor's switch for YOCO bounded replay
        /// (dsv41_yoco.h): unset is on with DeepInfer's 140, 0 is off.</summary>
        internal const string YocoKeepVariable = "TS_DSV41_YOCO_KEEP";

        /// <summary>
        /// Null unless the run uses an executor without YOCO bounded replay while
        /// the setting is at its default, in which case the operator gets told
        /// that this backend's prefill is the exact one. Emitted once per load by
        /// the DeepSeek4Model constructor, like <see cref="DescribeCpuBackendChoice"/>.
        /// </summary>
        internal static string DescribeYocoAbsence(BackendType backend) =>
            (backend == BackendType.Cuda || backend == BackendType.Cpu) &&
            Environment.GetEnvironmentVariable(YocoKeepVariable) == null
                ? $"[dsv41] --backend {(backend == BackendType.Cuda ? "cuda" : "cpu")}: every layer prefills every " +
                  "position. YOCO bounded replay, on by default on ggml_cuda and ggml_cpu, is not implemented here, " +
                  "so this backend's prefill logits are the exact ones and can differ slightly from ggml_cuda's."
                : null;

        /// <summary>The routed-MoE tensor-parallel rank count the native loader is
        /// given: <c>--tp</c>'s degree, or 0 (layer placement) without one.</summary>
        internal static int ResolveTensorParallelRanks(int tpDegree)
        {
            if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree));
            if (tpDegree > 8)
                throw new NotSupportedException("DeepSeek V4.1 tensor parallelism supports 2 through 8 GPUs.");
            return tpDegree > 1 ? tpDegree : 0;
        }

        internal static void ValidateTensorParallelGroup(BackendType backend, int tpDegree, ITensorParallelGroup group)
        {
            ResolveTensorParallelRanks(tpDegree);
            if (group == null) return;
            if (group is ILayerSplitGroup layer)
            {
                // Whole-layer stages run on ggml_cuda's stage graph or on the direct-CUDA engine's.
                if (backend is not (BackendType.GgmlCuda or BackendType.Cuda) || tpDegree != 1 || layer.NodeCount < 2 ||
                    layer.Degree is < 1 or > 8 || layer.NodeId < 0 || layer.NodeId >= layer.NodeCount)
                    throw new NotSupportedException("DeepSeek distributed layer split requires --backend ggml_cuda or cuda, --tp 1 and 1..8 GPUs per node.");
                return;
            }
            if (backend != BackendType.GgmlCuda)
                throw new NotSupportedException("DeepSeek V4.1 distributed tensor parallelism requires --backend ggml_cuda.");
            if (group.NodeCount < 2 || group is not INestedTensorParallelGroup)
                throw new NotSupportedException("DeepSeek V4.1 distributed TP requires the existing nested group and its cross-node AllReduce transport.");
            if (group.Degree != tpDegree || group.Degree is < 1 or > 8 ||
                group.GlobalDegree <= group.Degree || group.GlobalDegree > 16 ||
                group.GlobalRankOffset < 0 || group.GlobalRankOffset > group.GlobalDegree - group.Degree ||
                group.GlobalDegree != (long)group.NodeCount * group.Degree || group.GlobalRankOffset % group.Degree != 0)
                throw new ArgumentException("DeepSeek V4.1 TP requires matching local --tp (1..8), 2..16 global ranks, and a complete contiguous global rank interval per node.");
        }

        // Agree before opening native weight buffers: differing ubatches,
        // offload layers or YOCO schedules otherwise enter incompatible
        // collectives. Bit votes are exact F32 integers even across 16 nodes;
        // summing a digest as ordinary float values could hide a mismatch.
        internal static void ValidateDistributedExecutionConfig(ITensorParallelGroup group, string configuration)
        {
            if (group == null || group.NodeCount <= 1) return;
            if (group is not INestedTensorParallelGroup nested)
                throw new NotSupportedException("Distributed DeepSeek V4.1 needs the existing cross-node AllReduce transport.");
            byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(configuration));
            var votes = new float[digest.Length * 8];
            for (int bit = 0; bit < votes.Length; ++bit)
                votes[bit] = (digest[bit / 8] >> (bit % 8)) & 1;
            nested.CrossNodeAllReduce(votes, votes.Length);
            for (int bit = 0; bit < votes.Length; ++bit)
                if (votes[bit] != (((digest[bit / 8] >> (bit % 8)) & 1) * group.NodeCount))
                    throw new InvalidOperationException("DeepSeek V4.1 distributed nodes have different model metadata or execution settings. Use the same checkpoint, context, prefill width, CPU-MoE count and YOCO policy on every node.");
        }

        /// <summary>
        /// The settings that change what the direct-CUDA engine computes, for a distributed run's startup
        /// digest: the prefill width (which also decides each projection's kernel), the dense matmul routing
        /// thresholds of the shared CUDA ops, the expert fusion switch, the decode attention partitioning and
        /// where the Engram tables live. <paramref name="read"/> reads one variable (the process environment
        /// unless a test stands in for one node's).
        /// </summary>
        internal static string DirectCudaNumericalOptions(Func<string, string> read)
        {
            var options = new StringBuilder();
            foreach (string name in new[]
            {
                "TS_DSV4_UBATCH", "TS_CUDA_QMM_F16GEMM_MAX_MB", "TS_CUDA_QMM_F16GEMM_MIN_ROWS", "TS_CUDA_Q80_VEC_MIN_OUT",
                "TS_CUDA_Q80_MMQ_MAX_ROWS", "TENSORSHARP_CUDA_MOE_FUSION", "TS_CUDA_DECODE_ATTN_PARTITION_THRESHOLD",
                "TS_DSV41_ENGRAM_DEVICE", "TS_DSV41_ENGRAM_OWNER", YocoKeepVariable,
            })
            {
                // The owner mode by meaning, not spelling: unset and 'local' are the same placement.
                string value = name == "TS_DSV41_ENGRAM_OWNER"
                    ? TensorSharp.Cuda.Dsv4StagePlacement.EngramOwnerModeName(read(name))
                    : read(name) ?? "<default>";
                options.Append('|').Append(name).Append('=').Append(value);
            }
            return options.ToString();
        }

        internal static string DistributedNumericalOptions()
        {
            var options = new StringBuilder();
            foreach (string name in new[]
            {
                "TS_DSV4_FUSED", "TS_DSV4_FA", "TS_DSV4_GATHER",
                "TS_DSV41_SPARSE_FA", "TS_DSV41_COMPACT_RAW_GATHER",
                "TS_DSV41_ENGRAM_DEVICE", "TS_DSV41_KV_MIRRORS", "TS_DSV41_REWIND_CHECKPOINT",
            })
                options.Append('|').Append(name).Append('=')
                    .Append(Environment.GetEnvironmentVariable(name) ?? "<default>");
            return options.ToString();
        }

        internal static void ValidateDsparkArchitecture(string architecture)
        {
            if (!string.Equals(architecture, "deepseek41-dspark", StringComparison.Ordinal))
                throw new NotSupportedException(
                    "DeepSeek V4.1 DSpark requires a deepseek41-dspark artifact or llama.cpp's DeepSeek V4.1 dflash " +
                    "export; V4 and other draft architectures are incompatible.");
        }

        /// <summary>
        /// llama.cpp's DSpark export (Unsloth's DeepSeek-V4.1-Flash-1017 drafters) is a
        /// <c>dflash</c> GGUF. Other families' block-diffusion drafters share that name,
        /// so accept one only when it carries V4.1's hyper-connections and grouped
        /// output projection; the native loader then checks its geometry and tensors
        /// against the target.
        /// </summary>
        internal static void ValidateDsparkDraft(string architecture, Func<string, bool> hasKey)
        {
            if (string.Equals(architecture, "dflash", StringComparison.Ordinal) &&
                hasKey("dflash.hyper_connection.count") && hasKey("dflash.attention.output_group_count"))
                return;
            ValidateDsparkArchitecture(architecture);
        }

        /// <summary>The <c>--layer-split N</c> placement line.</summary>
        private static string DescribePlacement(int layerSplitDegree)
            => $"  Multi-GPU: DeepSeek V4.1 uses {layerSplitDegree} GPUs by LAYER SPLIT (whole-layer placement). " +
               "Use --tp 2..8 for routed-MoE tensor parallelism.";

        /// <summary>The <c>--tp N</c> placement line.</summary>
        internal static string DescribeTensorParallelPlacement(int ranks)
            => $"  Multi-GPU: DeepSeek V4.1 routed-MoE tensor parallelism across {ranks} GPUs: " +
               "gate/up/down expert dimensions are sharded, with exact F32 activation/output gathers. " +
               "Attention and shared experts retain layer placement; CPU-offloaded layers retain whole CPU experts. " +
               "Distributed groups use the same shards and the existing cross-node F32 collective transport.";
    }
}
