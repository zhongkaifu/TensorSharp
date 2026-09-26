// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    // First-call warm-up: force lazy kernel compilation (Metal pipelines, CUDA JIT,
    // allocator pools) before the first real request, plus the integrated-GPU and
    // host-backed-weight warnings that only make sense at that moment.
    public abstract partial class ModelBase
    {
        /// <summary>
        /// Run a tiny forward pass to force lazy kernel compilation (Metal pipelines,
        /// CUDA JIT, memory pool warm-up, etc.) so the first real inference request
        /// doesn't pay the compilation cost.  Resets KV cache and timing counters
        /// afterwards so the warmup is invisible to callers.
        /// </summary>
        public virtual void WarmUpKernels()
        {
            if (_backend == BackendType.Mlx && !IsMlxKernelWarmupEnabled())
            {
                long nativeBytes = MlxNativePreloadableQuantizedBytes();
                Console.WriteLine(
                    $"  Skipping MLX kernel warmup by default ({nativeBytes / 1024 / 1024} MB of resident quantized weights). Set TS_MLX_KERNEL_WARMUP=1 to force it.");
                ResetForwardTiming();
                return;
            }

            if (HasMlxHostFallbackQuantizedWeights())
            {
                long fallbackBytes = MlxHostFallbackQuantizedBytes();
                Console.WriteLine(
                    $"  Skipping MLX kernel warmup: {fallbackBytes / 1024 / 1024} MB of quantized weights use GGUF row-dequant fallback.");
                ResetForwardTiming();
                return;
            }

            int safeToken = (Config?.VocabSize ?? 0) > 1 ? 1 : 0;

            if (_tpGroup != null && _tpGroup.NodeCount > 1)
            {
                Console.WriteLine("  Waiting for all nodes to finish loading...");
                _tpGroup.Barrier();
            }

            Console.WriteLine("  Warming up kernels (decode + prefill)...");
            _inWarmup = true;
            try
            {

            long decodeStart = Stopwatch.GetTimestamp();
            Forward(new[] { safeToken });
            ResetKVCache();
            double decodeMs = (Stopwatch.GetTimestamp() - decodeStart) * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"    Decode warmup: {decodeMs:F1} ms");

            // Prime the MULTI-TOKEN prefill path. The 1-token Forward above only
            // warms the decode-shaped graph; on CUDA/GGML a real prompt takes the
            // fused whole-model prefill ("verify") graph + flash-attention kernels,
            // whose first build/capture and gallocr reservation otherwise lands on
            // the first real request and inflates its TTFT by ~50-110 ms (measured
            // on Gemma 4 E4B: first long prompt 906 ms cold -> 796 ms warm). Short
            // prompts ("Hello", the typical discarded warmup) skip the fused path,
            // so prime it with a longer dummy prompt here. Previously this ran for
            // MLX only. Guarded so a model that dislikes a dummy refill can never
            // block startup; disable entirely via TS_PREFILL_WARMUP=0.
            if (!string.Equals(Environment.GetEnvironmentVariable("TS_PREFILL_WARMUP"), "0", StringComparison.Ordinal))
            {
                // MLX, Metal, and the managed CPU backend stay conservative (short
                // prompt); discrete CUDA/Vulkan GGML paths use a longer one to reach
                // the fused-verify prefill path that short prompts bypass. The long
                // warmup only pays off where a first real prompt builds/captures a
                // fused whole-model prefill graph
                // and reserves a gallocr (CUDA/GGML) -- the managed CPU backend has
                // none of that, so a 1024-token prefill there is pure cost: tens of
                // seconds of GEMM plus large O(N^2) activation/KV first-touch paging
                // that makes `--backend cpu` look hung at startup for no benefit.
                // TS_PREFILL_WARMUP_LEN overrides (e.g. to pre-size the prefill
                // gallocr on roomy GPUs); a larger value does NOT reliably help a
                // near-full GPU because the legacy ForwardRefill warmup graph sizes
                // the reused gallocr differently than the engine prefill graph.
                // Integrated GPUs (unified-memory iGPUs: Intel UHD / AMD APU via
                // ggml-vulkan, Tegra via ggml-cuda) are memory-bandwidth bound and
                // run the fused multi-token prefill an order of magnitude slower than
                // a discrete GPU. A 2048-token verify-prefill warmup there takes
                // MINUTES (measured ~6+ min on Intel UHD with Qwen3.6-27B), during
                // which the server prints "Startup model loaded" then appears hung
                // before it ever starts listening. They also get no CUDA-graph
                // capture benefit from the long warmup. Treat them like MLX/CPU: a
                // short warmup that still primes the fused-verify graph once, cheaply.
                bool integratedGpu = IsIntegratedGgmlGpu();
                if (integratedGpu)
                {
                    Console.WriteLine(BuildIntegratedGpuWarning());
                }
                // Native CUDA models whose weights are mostly a CUDA-unsupported quant
                // type never become GPU-resident: their matmuls dequantize on the CPU,
                // and a 2048-token prefill warmup then pegs every core for MINUTES while
                // the server prints "Startup model loaded" and looks hung (observed on a
                // Qwen3.5-9B IQ4_XS build before IQ4_XS residency was added). Detect that
                // up front and use the short warmup instead so startup stays responsive;
                // the inference itself is still slow, so point the operator at a
                // supported quant / backend.
                double hostBackedFrac = HostBackedQuantWeightFraction();
                bool mostlyHostBacked = hostBackedFrac >= 0.5;
                if (mostlyHostBacked)
                {
                    Console.WriteLine(
                        $"  {hostBackedFrac * 100:F0}% of quantized weights use a CUDA-unsupported quant type and stay host-backed (CPU matmul); using a lightweight startup warmup. Inference will be slow — prefer a quant the direct CUDA backend supports (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/IQ2/IQ3/IQ4_XS) or run with --backend ggml_cuda.");
                }
                // GGML CUDA/Vulkan build/capture a fused whole-model prefill graph and
                // pre-grow its gallocr. Native CUDA has no captured verify-prefill
                // graph, so its architecture default remains a lightweight 32-token
                // shape. A model may opt into a larger direct-CUDA warmup when it has
                // persistent activation/QMM scratch that is expensive to grow during
                // the first real request. Hybrid GatedDeltaNet models intentionally
                // retain the lightweight base default because a 2K warmup can take minutes.
                // Multi-token prefill of an MoE model under tensor parallelism is
                // slow enough that the full 2048-token warmup dominates startup —
                // measured at ~61 s on Gemma 4 26B A4B with --tp 2, during which
                // the process looks hung. Same treatment as the other known-slow
                // prefill configurations: warm the path cheaply and say why.
                bool moeUnderTp = MoEUnderTpIsSlow;
                if (moeUnderTp)
                {
                    Console.WriteLine(
                        "  MoE under tensor parallelism: using a lightweight startup warmup (the full one costs ~a minute here). " +
                        "Long-prompt prefill stays slower than a single GPU; TP is for models that do not fit on one.");
                }

                bool conservativeWarmup = UsesLightweightPrefillWarmupByDefault(_backend)
                    || integratedGpu || mostlyHostBacked || moeUnderTp;
                // 2048 matches ComputePrefillChunkSize, so the warmup runs ONE
                // fused verify chunk at the largest legacy-chunk shape: the shared
                // reuse-gallocr is pre-grown (and its device memory first-touched)
                // for every prompt up to 2048 tokens, which covers typical chat
                // prompts. Measured on gemma4-12B/Vulkan: first ~2k-token request
                // was ~300-500 ms slower than warm (gallocr growth + residency)
                // with the old 1024 warmup; with 2048 it starts warm.
                int? explicitWarmupLengthValue = null;
                {
                    string wl = Environment.GetEnvironmentVariable("TS_PREFILL_WARMUP_LEN");
                    if (!string.IsNullOrEmpty(wl) && int.TryParse(wl, out int wlv) && wlv >= 2)
                        explicitWarmupLengthValue = wlv;
                }
                int warmupLength = ResolvePrefillWarmupTargetLength(
                    _backend,
                    integratedGpu,
                    mostlyHostBacked,
                    moeUnderTp,
                    NativeCudaPrefillWarmupLength,
                    explicitWarmupLengthValue);
                bool explicitWarmupLength = explicitWarmupLengthValue.HasValue;
                // The point of the multi-token warmup is to pre-grow the shared
                // reuse-gallocr to the shape real requests will use, so a model
                // whose chunk width is narrower than the 2048 default must warm at
                // ITS width — warming wider grows that allocator to a size nothing
                // afterwards needs and hands the VRAM back to nobody.
                if (!explicitWarmupLength && !conservativeWarmup)
                {
                    int chunkWidth = GgmlPrefillChunkWarmupLength;
                    if (chunkWidth >= 2 && chunkWidth < warmupLength)
                        warmupLength = chunkWidth;
                }
                int warmupTokenOverhead =
                    _backend == BackendType.Cuda && !conservativeWarmup
                        ? NativeCudaPrefillWarmupTokenOverhead
                        : 0;
                warmupLength = ResolvePrefillWarmupInputLength(
                    warmupLength,
                    MaxContextLength,
                    warmupTokenOverhead,
                    explicitWarmupLength);

                int[] warmupPrompt = new int[warmupLength];
                Array.Fill(warmupPrompt, safeToken);

                try
                {
                    Console.WriteLine($"    Prefill warmup ({warmupLength} tokens): starting...");
                    long prefillStart = Stopwatch.GetTimestamp();
                    ForwardRefill(warmupPrompt);
                    Forward(new[] { safeToken });
                    double prefillMs = (Stopwatch.GetTimestamp() - prefillStart) * 1000.0 / Stopwatch.Frequency;
                    Console.WriteLine($"    Prefill warmup ({warmupLength} tokens): completed in {prefillMs:F1} ms");
                }
                // A dead GPU backend is not a skipped warm-up. Swallowing it left the
                // host serving on a backend that refuses every later graph, and the
                // first request then died naming its own embedding lookup rather than
                // the out-of-memory that killed the backend here (issue #226).
                catch (Exception ex) when (!BackendHasFailed())
                {
                    Console.WriteLine($"  Prefill warmup skipped: {ex.GetType().Name}: {ex.Message}");
                }
                finally
                {
                    // Resetting runs device ops too; on a dead backend it would fail
                    // and replace the exception that says why.
                    if (!BackendHasFailed())
                        ResetKVCache();
                }

                if (_backend == BackendType.Cuda &&
                    NativeCudaPrimeShortDecodeGraphAfterPrefill &&
                    CudaPrefillGraphCache.Enabled &&
                    CudaPrefillGraphCache.DecodeEnabled)
                {
                    try
                    {
                        // The one-token pass at the beginning of WarmUpKernels
                        // registered this graph key once. If the long prefill used
                        // a different attention tier, this second sighting captures
                        // the short route instead of charging the first request.
                        Forward(new[] { safeToken });
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  Short decode-graph warmup skipped: {ex.GetType().Name}: {ex.Message}");
                    }
                    finally
                    {
                        ResetKVCache();
                    }
                }
            }

            WarmUpMultimodalKernels();

            }
            finally { _inWarmup = false; }

            ResetForwardTiming();
        }

        public virtual void WarmUpMultimodalKernels()
        {
        }

        /// <summary>
        /// True when the active GGML device (ggml_cuda / ggml_vulkan) is an
        /// integrated, unified-memory GPU (Intel UHD / AMD APU / Tegra). Such
        /// devices are memory-bandwidth bound and cannot afford the heavy
        /// multi-token prefill warmup; startup warmup falls back to the short
        /// conservative path for them. Only ggml backends are queried; every other
        /// backend (and any query failure) reports false.
        /// </summary>
        private bool IsIntegratedGgmlGpu()
        {
            if (_backend != BackendType.GgmlCuda && _backend != BackendType.GgmlVulkan)
                return false;
            try { return GgmlBasicOps.IsActiveDeviceIntegrated(); }
            catch { return false; }
        }

        /// <summary>
        /// Builds the prominent startup banner shown when inference is running on an
        /// integrated, unified-memory GPU. A 27B model on an Intel UHD iGPU decodes at
        /// ~0.7 tok/s (measured, and identical to llama.cpp on the same device) versus
        /// ~15 tok/s on a discrete RTX 3080 — a ~20x gap that is purely hardware, not a
        /// software regression. The single-line notice this replaces was easy to miss in
        /// the startup log, so operators kept running large models on the iGPU by mistake.
        /// On ggml_vulkan (multiple adapters are enumerable) the banner names the selected
        /// adapter and lists every other device with the exact <c>--gpu-device N</c> flag to
        /// switch to it. Enumeration failures / single-device hosts (e.g. Tegra via
        /// ggml_cuda) fall back to the generic guidance.
        /// </summary>
        private string BuildIntegratedGpuWarning()
        {
            const string sep = "  ==============================================================================";
            var sb = new System.Text.StringBuilder();
            sb.AppendLine(sep);
            sb.AppendLine("  PERFORMANCE WARNING: inference is running on an INTEGRATED GPU.");
            sb.AppendLine("  Integrated (unified-memory) GPUs are memory-bandwidth bound: large models run");
            sb.AppendLine("  roughly an order of magnitude slower here than on a discrete GPU, and often");
            sb.AppendLine("  slower than the CPU backend. This is a hardware limit, not a TensorSharp issue.");

            bool namedAlternative = false;
            if (_backend == BackendType.GgmlVulkan)
            {
                try
                {
                    int selected = 0;
                    string sel = Environment.GetEnvironmentVariable(GgmlBasicOps.VulkanDeviceEnvVar);
                    if (!string.IsNullOrEmpty(sel) && int.TryParse(sel, out int s) && s >= 0)
                        selected = s;

                    int count = GgmlBasicOps.GetVulkanDeviceCount();
                    if (count > 0 && selected < count)
                        sb.AppendLine($"  Selected:  --gpu-device {selected}   {GgmlBasicOps.GetVulkanDeviceDescription(selected)}");

                    var others = new System.Collections.Generic.List<string>();
                    for (int i = 0; i < count; i++)
                    {
                        if (i == selected) continue;
                        others.Add($"    --gpu-device {i}   {GgmlBasicOps.GetVulkanDeviceDescription(i)}");
                    }
                    if (others.Count > 0)
                    {
                        sb.AppendLine("  For full performance re-run against a discrete GPU (see --list-gpus):");
                        foreach (var o in others) sb.AppendLine(o);
                        namedAlternative = true;
                    }
                }
                catch { /* enumeration unavailable — fall through to generic guidance */ }
            }
            if (!namedAlternative)
                sb.AppendLine("  For full performance use a discrete GPU (--gpu-device <index>, see --list-gpus), or --backend ggml_cpu.");
            sb.Append(sep);
            return sb.ToString();
        }

        /// <summary>
        /// Fraction (0..1) of quantized weight bytes whose quant type the direct
        /// CUDA backend cannot make GPU-resident (<see cref="CudaQuantizedOps.SupportsQuantizedType"/>).
        /// Such weights stay host-backed and their matmuls dequantize on the CPU, so a
        /// large fraction means a multi-token prefill (including the startup warmup)
        /// runs CPU-bound. Only the native <see cref="BackendType.Cuda"/> backend is
        /// evaluated (GGML backends upload weights to the device themselves); every
        /// other backend reports 0. Computed from the quant TYPE, not the live
        /// host-data flag, so it is unaffected by TS_GGML_RETAIN_HOST_WEIGHTS.
        /// </summary>
        private double HostBackedQuantWeightFraction()
        {
            if (_backend != BackendType.Cuda || _quantWeights.Count == 0)
                return 0.0;
            long total = 0, hostBacked = 0;
            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                total += qw.RawBytes;
                if (!CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType))
                    hostBacked += qw.RawBytes;
            }
            return total > 0 ? (double)hostBacked / total : 0.0;
        }

        private static bool IsMlxKernelWarmupEnabled()
        {
            return string.Equals(Environment.GetEnvironmentVariable("TS_MLX_KERNEL_WARMUP"), "1", StringComparison.Ordinal);
        }
    }
}
