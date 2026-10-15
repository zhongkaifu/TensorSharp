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
using System.Globalization;

namespace TensorSharp.Models
{
    /// <summary>
    /// Process-wide configuration for Mixture-of-Experts CPU offload — the
    /// TensorSharp equivalent of llama.cpp's <c>--n-cpu-moe N</c> / <c>--cpu-moe</c>.
    ///
    /// <para><b>What it does.</b> For every layer selected by
    /// <see cref="IsLayerOnCpu"/>, the routed-expert weights
    /// (<c>ffn_gate_exps</c> / <c>ffn_up_exps</c> / <c>ffn_down_exps</c>) are
    /// never uploaded to the accelerator, and that layer's MoE FFN runs on the
    /// ggml CPU backend reading the weights zero-copy straight out of the GGUF
    /// mmap. Everything else in the layer — attention, norms, the router, and
    /// the (small, always-active) shared expert — stays on the GPU.</para>
    ///
    /// <para><b>Why it is the right trade for low-VRAM MoE.</b> A 35B-A3B model
    /// spends ~90% of its bytes on routed experts but only ~3B parameters are
    /// active per token. Keeping those bytes in system RAM removes almost all of
    /// the VRAM requirement, while the per-layer cost paid on the host is a
    /// single <c>mul_mat_id</c> over the <c>n_expert_used</c> selected experts —
    /// exactly the work llama.cpp's CPU buffer-type override performs. The
    /// activation crossing the bus each layer is only
    /// <c>seq_len * hidden_size</c> floats (8 KB for a decode step), so the
    /// hybrid split is dominated by host matmul throughput, not by transfers.
    /// </para>
    ///
    /// <para><b>Layer selection.</b> Following llama.cpp, <c>--n-cpu-moe N</c>
    /// offloads the <i>first</i> N layers. The first layers are the right ones
    /// to give up: they are reached earliest in the forward pass, so the host
    /// matmul overlaps with nothing regardless of which layers are chosen, and
    /// keeping the <i>last</i> layers resident lets the GPU finish a token
    /// without a final host round trip.</para>
    ///
    /// <para>Configured once at start-up from the CLI (<c>--n-cpu-moe</c> /
    /// <c>--cpu-moe</c>) or the <c>TS_N_CPU_MOE</c> / <c>TS_CPU_MOE</c>
    /// environment variables, then read by <c>ModelBase</c> (weight residency)
    /// and by each MoE model's FFN dispatch (execution placement).</para>
    /// </summary>
    public static class MoeCpuOffloadConfig
    {
        private const string EnvVarLayers = "TS_N_CPU_MOE";
        private const string EnvVarAll = "TS_CPU_MOE";
        private const string EnvVarThreads = "TS_CPU_MOE_THREADS";

        private static int _cpuMoeLayers;
        private static bool _allLayers;
        private static bool _explicitlySet;

        /// <summary>
        /// Number of leading layers whose routed experts live on the CPU. Only
        /// meaningful when <see cref="AllLayers"/> is false. Zero means the
        /// feature is off.
        /// </summary>
        public static int CpuMoeLayers => _cpuMoeLayers;

        /// <summary>True when every MoE layer is offloaded (<c>--cpu-moe</c>).</summary>
        public static bool AllLayers => _allLayers;

        /// <summary>True when any MoE layer is offloaded to the host.</summary>
        public static bool IsEnabled => _allLayers || _cpuMoeLayers > 0;

        /// <summary>True when a CLI flag or environment variable set the policy.</summary>
        public static bool IsExplicitlySet => _explicitlySet;

        /// <summary>
        /// Native whole-model loaders with a capacity planner accept -1 to keep
        /// the fewest possible leading expert layers on the host. This is a
        /// per-loader choice, not a change to the global per-layer weight policy.
        /// In particular, explicitly setting zero must disable automatic offload.
        /// </summary>
        internal static int ResolveNativeLayers(bool planWhenUnspecified)
            => _explicitlySet ? (_allLayers ? int.MaxValue : _cpuMoeLayers)
                : planWhenUnspecified ? -1 : 0;

        /// <summary>
        /// Worker threads for the host-side MoE matmul. Zero (the default) lets
        /// the native layer pick <c>hardware_concurrency</c>. Exposed because
        /// the GPU-side graph submission thread is otherwise idle while the host
        /// matmul runs, so oversubscribing slightly can be a win on machines
        /// with SMT.
        /// </summary>
        public static int CpuThreads { get; private set; }

        /// <summary>Offload the routed experts of the first <paramref name="layers"/> layers.</summary>
        public static void SetLayers(int layers)
        {
            if (layers < 0)
                throw new ArgumentOutOfRangeException(nameof(layers), "--n-cpu-moe must be >= 0.");
            _cpuMoeLayers = layers;
            _allLayers = false;
            _explicitlySet = true;
        }

        /// <summary>Offload the routed experts of every MoE layer.</summary>
        public static void SetAllLayers()
        {
            _allLayers = true;
            _cpuMoeLayers = 0;
            _explicitlySet = true;
        }

        public static void SetCpuThreads(int threads)
        {
            CpuThreads = threads > 0 ? threads : 0;
            // The host matmul runs inside GgmlOps, and it must be TOLD the count:
            // .NET's SetEnvironmentVariable writes only the managed environment on
            // Linux, so the native std::getenv(TS_CPU_MOE_THREADS) never saw the
            // flag and --cpu-moe-threads silently did nothing there. The env var
            // is still published so a child process (and TS_CPU_MOE_THREADS set
            // from outside) keeps working.
            Environment.SetEnvironmentVariable(EnvVarThreads,
                CpuThreads > 0 ? CpuThreads.ToString(CultureInfo.InvariantCulture) : null);
            try
            {
                TensorSharp.GGML.GgmlBasicOps.SetHostMoeThreads(CpuThreads);
            }
            catch (Exception ex)
            {
                // No GGML native library on this platform/build (MLX-only, say).
                // The thread count is advisory; never fail start-up over it — but
                // when a count was actually requested, say it was not delivered.
                if (CpuThreads > 0 &&
                    System.Threading.Interlocked.Exchange(ref _threadsUndeliveredWarned, 1) == 0)
                {
                    Console.Error.WriteLine(
                        $"[moe-offload] WARNING: could not hand --cpu-moe-threads {CpuThreads} to the native " +
                        $"GGML layer ({ex.GetType().Name}: {ex.Message}); the host MoE matmul keeps its " +
                        "default thread count. Reported once.");
                }
            }
        }

        private static int _threadsUndeliveredWarned;

        /// <summary>
        /// Reset to the "no offload" default. Used by tests and by model reload
        /// so a previous run's policy cannot leak into a fresh load.
        /// </summary>
        public static void Reset()
        {
            _cpuMoeLayers = 0;
            _allLayers = false;
            _explicitlySet = false;
            CpuThreads = 0;
            ResetWarningLatch();
        }

        /// <summary>
        /// True when <paramref name="layer"/>'s routed experts should be kept in
        /// system RAM and multiplied on the host. Layers are zero-based.
        /// </summary>
        public static bool IsLayerOnCpu(int layer)
        {
            if (_allLayers)
                return true;
            return layer >= 0 && layer < _cpuMoeLayers;
        }

        /// <summary>
        /// Per the GGUF / llama.cpp naming convention every routed-expert tensor
        /// carries an <c>_exps.</c> infix (<c>blk.7.ffn_gate_exps.weight</c>).
        /// The always-active shared expert uses <c>_shexp</c> instead and is
        /// deliberately NOT matched: it is small, runs for every token, and
        /// keeping it on the GPU avoids a second host round trip per layer.
        /// </summary>
        public static bool IsRoutedExpertWeightName(string name)
            => !string.IsNullOrEmpty(name) && name.IndexOf("_exps.", StringComparison.Ordinal) >= 0;

        /// <summary>
        /// True when <paramref name="name"/> is a routed-expert tensor belonging
        /// to a layer that this policy places on the CPU. Weight names follow
        /// the GGUF <c>blk.&lt;layer&gt;.</c> prefix; a name without a parseable
        /// layer index is never offloaded (falling back to GPU residency is
        /// always safe, just less memory-efficient).
        /// </summary>
        public static bool IsOffloadedExpertWeightName(string name)
        {
            if (!IsEnabled || !IsRoutedExpertWeightName(name))
                return false;
            if (_allLayers)
                return true;
            return TryParseLayerIndex(name, out int layer) && layer < _cpuMoeLayers;
        }

        /// <summary>
        /// Parse the layer index out of a GGUF tensor name of the form
        /// <c>blk.&lt;n&gt;.&lt;rest&gt;</c>. Returns false for names that do
        /// not carry one (token_embd, output_norm, vision-tower tensors, ...).
        /// </summary>
        public static bool TryParseLayerIndex(string name, out int layer)
        {
            layer = -1;
            if (string.IsNullOrEmpty(name) || !name.StartsWith("blk.", StringComparison.Ordinal))
                return false;

            int start = 4;
            int end = name.IndexOf('.', start);
            if (end <= start)
                return false;

#if NET
            return int.TryParse(name.AsSpan(start, end - start), NumberStyles.None, CultureInfo.InvariantCulture, out layer);
#else
            return int.TryParse(name.Substring(start, end - start), NumberStyles.None, CultureInfo.InvariantCulture, out layer);
#endif
        }

        /// <summary>
        /// Parse a <c>--n-cpu-moe</c> value. Accepts a non-negative integer, or
        /// <c>all</c> / <c>max</c> as aliases for <c>--cpu-moe</c>.
        /// </summary>
        public static bool TryParse(string value, out int layers, out bool allLayers)
        {
            layers = 0;
            allLayers = false;
            if (string.IsNullOrWhiteSpace(value))
                return false;

            string trimmed = value.Trim();
            if (string.Equals(trimmed, "all", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(trimmed, "max", StringComparison.OrdinalIgnoreCase))
            {
                allLayers = true;
                return true;
            }

            return int.TryParse(trimmed, NumberStyles.Integer, CultureInfo.InvariantCulture, out layers) && layers >= 0;
        }

        /// <summary>
        /// Apply <c>TS_N_CPU_MOE</c> / <c>TS_CPU_MOE</c> / <c>TS_CPU_MOE_THREADS</c>
        /// as the process-wide default. An explicit CLI flag wins: it calls
        /// <see cref="SetLayers"/> / <see cref="SetAllLayers"/> directly and this
        /// method leaves an already-set policy alone.
        /// </summary>
        public static void ConfigureFromEnvironment()
        {
            if (int.TryParse(Environment.GetEnvironmentVariable(EnvVarThreads),
                    NumberStyles.Integer, CultureInfo.InvariantCulture, out int threads))
                SetCpuThreads(threads);

            if (_explicitlySet)
                return;

            string all = Environment.GetEnvironmentVariable(EnvVarAll);
            if (!string.IsNullOrWhiteSpace(all) && all.Trim() != "0")
            {
                SetAllLayers();
                return;
            }

            if (TryParse(Environment.GetEnvironmentVariable(EnvVarLayers), out int layers, out bool allLayers))
            {
                if (allLayers) SetAllLayers();
                else SetLayers(layers); // zero is an explicit refusal to offload
            }
            else if (!string.IsNullOrWhiteSpace(all) && all.Trim() == "0")
                SetLayers(0);
        }

        private static int _unsupportedWarned;

        /// <summary>
        /// Say, once, that the selected backend has no host-MoE path so this
        /// policy will not actually move anything off the accelerator.
        ///
        /// <para>The offload seam is implemented for the GGML backends (every
        /// MoE architecture) and, on the pure-C# <c>cuda</c> backend, only for
        /// DeepSeek V4. Everywhere else the routed experts are still uploaded
        /// and still multiplied on the GPU. Staying silent is the real hazard:
        /// the operator asked for a VRAM reduction, measured none, and had
        /// nothing to tell them why — on gemma-4-26B the flag moved peak VRAM
        /// from 14261 MiB to 14253 MiB, which reads exactly like a broken
        /// build.</para>
        /// </summary>
        public static void WarnUnsupportedBackend(string architecture, string backend)
        {
            if (!IsEnabled)
                return;
            if (System.Threading.Interlocked.Exchange(ref _unsupportedWarned, 1) != 0)
                return;
            Console.Error.WriteLine(
                $"[moe-offload] WARNING: --n-cpu-moe / --cpu-moe is not implemented for {architecture} on the " +
                $"{backend} backend; the routed experts stay on the accelerator and no VRAM is saved. Use " +
                "--backend ggml_cuda (or ggml_vulkan) for this model if you need the offload.");
        }

        /// <summary>Reset the once-only warning latches. Tests only.</summary>
        internal static void ResetWarningLatch()
        {
            System.Threading.Interlocked.Exchange(ref _unsupportedWarned, 0);
            System.Threading.Interlocked.Exchange(ref _threadsUndeliveredWarned, 0);
        }

        /// <summary>
        /// Human-readable summary for the model-load banner, e.g.
        /// <c>"first 32 layers"</c>. Returns null when offload is off.
        /// </summary>
        public static string Describe()
        {
            if (_allLayers) return "all layers";
            if (_cpuMoeLayers > 0) return $"first {_cpuMoeLayers} layer(s)";
            return null;
        }
    }
}
