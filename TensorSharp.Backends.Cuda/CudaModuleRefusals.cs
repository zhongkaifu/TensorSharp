// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The kernel modules an allocator loads on first use (tensorsharp_fp4_kernels, tensorsharp_flash_prefill,
// tensorsharp_ssm_prefill) and the warning a refused one earns. A module is refused where its image is missing, stale (a
// committed PTX not refreshed after a kernel change) or carries no code this GPU runs (the FP4 tensor-core products on any
// GPU but sm_120a / sm_121a, sm_103 included); every route that wanted it then takes its fallback. The loader used to
// drop the refusal's reason, so such a GPU ran its fallbacks without a word: the warning names the module, why it was
// refused and what runs instead, once per module and reason in the process.
using System;
using System.Collections.Concurrent;
using System.Threading;

namespace TensorSharp.Cuda
{
    /// <summary>A module factory: true with the module, or false with the reason it cannot run here.</summary>
    internal delegate bool CudaModuleFactory<T>(out T module, out string reason);

    internal sealed class CudaModuleRefusals
    {
        /// <summary>What runs where tensorsharp_fp4_kernels is refused, said where an NVFP4 weight is made resident (the
        /// module is also asked for by every prompt step's add-norm, so a model without NVFP4 weights loses nothing).</summary>
        internal const string Fp4Consequence =
            "NVFP4 weights stay in their GGUF blocks, so a product of more than 16 rows (a prompt step) dequantizes each " +
            "weight to F16 for cuBLAS instead of running on the FP4 tensor cores, and prompt processing is much slower";

        /// <summary>What runs where tensorsharp_flash_prefill is refused.</summary>
        internal const string FlashPrefillConsequence =
            "attention on the CUDA graph executor takes the per-op path instead of the flash kernels (prompt chunks, decode " +
            "rows, verify windows, sparse QSA regions), which is slower at long contexts, and decode steps record again as " +
            "the context grows";

        /// <summary>What runs where tensorsharp_ssm_prefill is refused.</summary>
        internal const string SsmPrefillConsequence =
            "a prompt chunk's gated delta rule runs the four-row kernel instead of the 16-row tensor-core chunks, so " +
            "Gated DeltaNet layers prefill more slowly";

        /// <summary>The process's warnings, on standard error.</summary>
        internal static CudaModuleRefusals Process { get; } = new(line => Console.Error.WriteLine(line));

        private readonly Action<string> _write;
        private readonly ConcurrentDictionary<string, bool> _warned = new(StringComparer.Ordinal);

        /// <param name="write">Where a warning line goes (tests capture it).</param>
        internal CudaModuleRefusals(Action<string> write) => _write = write ?? throw new ArgumentNullException(nameof(write));

        /// <summary>
        /// Warn that <paramref name="module"/> is unavailable on GPU <paramref name="deviceId"/> for
        /// <paramref name="reason"/>, naming <paramref name="consequence"/>: once per module and reason (several GPUs
        /// refusing a module for the same reason say it once). True when this call wrote the warning.
        /// </summary>
        internal bool Warn(string module, int deviceId, string reason, string consequence)
        {
            reason = string.IsNullOrWhiteSpace(reason) ? "no reason given" : reason.Trim().TrimEnd('.');
            if (!_warned.TryAdd(module + "\n" + reason, true))
                return false;
            try
            {
                _write($"WARNING: [cuda] {module} is unavailable on GPU {deviceId} ({reason}); {consequence}. Reported once.");
            }
            catch (Exception)
            {
                // Diagnostics must never break inference.
            }
            return true;
        }

        /// <summary>The module name of a PTX file name (tensorsharp_flash_prefill.ptx: tensorsharp_flash_prefill).</summary>
        internal static string ModuleName(string ptxFileName)
            => ptxFileName.EndsWith(".ptx", StringComparison.Ordinal) ? ptxFileName[..^4] : ptxFileName;

        /// <summary>
        /// <paramref name="module"/>, created on the first call: <paramref name="create"/> runs once, under
        /// <paramref name="sync"/> after <paramref name="prepare"/> (the context made current); a refusal keeps its reason
        /// in <paramref name="refusal"/> and hands it to <paramref name="refused"/>, once, outside the lock. Null while the
        /// module is refused. <paramref name="state"/>: 0 not tried yet, 1 loaded, 2 refused.
        /// </summary>
        internal static T LoadOnce<T>(object sync, ref T module, ref int state, ref string refusal, Action prepare,
            CudaModuleFactory<T> create, Action<string> refused) where T : class
        {
            if (Volatile.Read(ref state) != 0) return module;
            lock (sync)
            {
                if (state != 0) return module;
                prepare?.Invoke();
                bool loaded = create(out T created, out string reason);
                module = loaded ? created : null;
                refusal = loaded ? null : (string.IsNullOrWhiteSpace(reason) ? "no reason given" : reason);
                Volatile.Write(ref state, loaded ? 1 : 2);
                if (loaded) return module;
            }
            refused?.Invoke(refusal);
            return null;
        }
    }
}
