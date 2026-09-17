// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Runtime.Speculative;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Static knobs for the continuous batching scheduler. Mirrors a subset of
    /// vLLM's <c>SchedulerConfig</c>: per-step token budget, max in-flight
    /// sequence count, and chunk size for prefill.
    /// </summary>
    public sealed class SchedulerConfig
    {
        /// <summary>Maximum tokens forwarded across all sequences per step.
        /// Equivalent to vLLM's <c>max_num_batched_tokens</c>. Default 4096.</summary>
        public int MaxNumBatchedTokens { get; init; } = 4096;

        /// <summary>Maximum in-flight sequences. Once full, additional waiting
        /// sequences stay in the waiting queue until one finishes. Equivalent
        /// to vLLM's <c>max_num_seqs</c>. Default 16.</summary>
        public int MaxNumRunningSequences { get; init; } = 16;

        /// <summary>Maximum number of new prefill tokens to schedule per
        /// sequence in a mixed prefill+decode step. This bounds the time an
        /// already-streaming request waits behind one long-prompt forward.
        /// When every active request is still prefilling, the scheduler instead
        /// divides the complete <see cref="MaxNumBatchedTokens"/> budget evenly
        /// so the accelerator is not left half idle. Default 256. Overridable
        /// via <c>TS_SCHED_PREFILL_CHUNK</c> or
        /// <c>--prefill-chunk-size</c>.</summary>
        public int MaxPrefillChunkSize { get; init; } = 256;

        /// <summary>Per-step prefill token cap used ONLY when there is no GPU
        /// contention ÔÇö i.e. at most one sequence is in the system (running +
        /// waiting &lt;= 1). The small <see cref="MaxPrefillChunkSize"/> exists
        /// purely to let concurrent decode requests interleave at the GPU; for a
        /// lone request it is counter-productive. On GPU backends a chunk that
        /// crosses the model's sliding-window boundary drops off the fused
        /// single-graph prefill path onto the per-op path (which syncs the GPU
        /// after every op on CUDA, where async deferral is Metal-only), so
        /// splitting a solo prompt into small chunks is several times SLOWER than
        /// feeding it whole. Matching the CLI (which never sub-divides a solo
        /// prompt below ~5120 tokens) recovers full prefill throughput. Bounded
        /// by <see cref="MaxNumBatchedTokens"/> for activation-memory safety.
        /// Default 8192. Env: <c>TS_SCHED_SOLO_PREFILL_CHUNK</c>.</summary>
        public int SoloPrefillChunkSize { get; init; } = 8192;

        /// <summary>Number of physical KV blocks in the pool. The total KV-cache
        /// budget is <c>NumBlocks * BlockSize</c> tokens. When the model exposes
        /// its preferred block size that value is used here.</summary>
        public int NumBlocks { get; init; } = 256;

        /// <summary>Block size in tokens. Should match the model's preferred
        /// block size (we use the existing
        /// <see cref="PagedKvCacheConfig.BlockSize"/> as the default).</summary>
        public int BlockSize { get; init; } = 256;

        /// <summary>Enable LRU-based block eviction of cached prefix blocks
        /// when the free queue is empty. Default true.</summary>
        public bool EnablePrefixCaching { get; init; } = true;

        /// <summary>Radix owns prefix reuse by default. Legacy remains available for
        /// compatibility diagnostics through <c>TS_PREFIX_CACHE_MODE=legacy</c>.
        /// <see cref="EnablePrefixCaching"/> disables reuse in either mode.</summary>
        public PrefixCacheMode PrefixCacheMode { get; init; } = PrefixCacheMode.Tree;

        /// <summary>
        /// End a sequence whose output has locked into a loop (see
        /// <see cref="RepetitionGuard"/>) with the finish reason <c>repetition</c>,
        /// instead of running it to its token limit. On by default; a harness that
        /// deliberately generates the same token thousands of times turns it off.
        /// Env: <c>TS_SCHED_STOP_REPETITION</c>.
        /// </summary>
        public bool StopRepetition { get; init; } = true;

        /// <summary>How many decode steps a running sequence is allowed to run
        /// consecutively before the scheduler may swap to another sequence.
        /// In the current C# executor each session-switch pays a KV-state
        /// extract+inject round-trip, so we amortize that by running multiple
        /// decode tokens for the same session before switching. Set to 1 for
        /// strict per-token fairness; default is the block size so we naturally
        /// swap at block boundaries.</summary>
        public int DecodeQuantumTokens { get; init; } = 256;

        /// <summary>
        /// Speculative decoding policy for this engine: whether to speculate,
        /// with which algorithm, how wide a window and what confidence gate.
        /// Default: off. One value object rather than loose fields so a new
        /// knob reaches the executor without threading a parameter through the
        /// scheduler. See <see cref="SpeculationOptions"/>.
        /// </summary>
        public SpeculationOptions Speculation { get; init; } = SpeculationOptions.Disabled;

        public static SchedulerConfig Default => new();

        /// <summary>This configuration with a different speculation policy: the
        /// executor swaps it at run time when the host toggles speculation, so the
        /// planner (a pure function of the config) sees the change on the next step.</summary>
        public SchedulerConfig WithSpeculation(SpeculationOptions speculation) => new()
        {
            MaxNumBatchedTokens = MaxNumBatchedTokens,
            MaxNumRunningSequences = MaxNumRunningSequences,
            MaxPrefillChunkSize = MaxPrefillChunkSize,
            SoloPrefillChunkSize = SoloPrefillChunkSize,
            NumBlocks = NumBlocks,
            BlockSize = BlockSize,
            EnablePrefixCaching = EnablePrefixCaching,
            PrefixCacheMode = PrefixCacheMode,
            StopRepetition = StopRepetition,
            DecodeQuantumTokens = DecodeQuantumTokens,
            Speculation = speculation ?? SpeculationOptions.Disabled,
        };

        public static SchedulerConfig FromEnvironment()
        {
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = ReadInt("TS_SCHED_MAX_BATCHED_TOKENS", 4096),
                MaxNumRunningSequences = ReadInt("TS_SCHED_MAX_RUNNING_SEQS", 16),
                MaxPrefillChunkSize = ReadInt("TS_SCHED_PREFILL_CHUNK", 256),
                SoloPrefillChunkSize = ReadInt("TS_SCHED_SOLO_PREFILL_CHUNK", 8192),
                NumBlocks = ReadInt("TS_SCHED_NUM_BLOCKS", 256),
                BlockSize = ReadInt("TS_SCHED_BLOCK_SIZE", 256),
                EnablePrefixCaching = ReadBool("TS_SCHED_PREFIX_CACHE", true),
                PrefixCacheMode = ReadPrefixCacheMode(),
                StopRepetition = ReadBool("TS_SCHED_STOP_REPETITION", true),
                DecodeQuantumTokens = ReadInt("TS_SCHED_DECODE_QUANTUM", 256),
                Speculation = SpeculationOptions.FromEnvironment(),
            };
            return cfg;
        }

        private static int ReadInt(string name, int fallback)
        {
            string raw = System.Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v > 0)
                return v;
            return fallback;
        }

        private static PrefixCacheMode ReadPrefixCacheMode()
        {
            string raw = System.Environment.GetEnvironmentVariable("TS_PREFIX_CACHE_MODE");
            if (string.IsNullOrWhiteSpace(raw) || string.Equals(raw.Trim(), "tree", System.StringComparison.OrdinalIgnoreCase))
                return PrefixCacheMode.Tree;
            if (string.Equals(raw.Trim(), "legacy", System.StringComparison.OrdinalIgnoreCase))
                return PrefixCacheMode.Legacy;
            throw new System.ArgumentException("TS_PREFIX_CACHE_MODE must be 'tree' or 'legacy'.");
        }

        // Boolean flag reader that accepts "0"/"1" (and "true"/"false"). Unlike
        // ReadInt, this honours an explicit "0" so a flag can actually be disabled.
        private static bool ReadBool(string name, bool fallback)
        {
            string raw = System.Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrEmpty(raw)) return fallback;
            raw = raw.Trim();
            if (raw == "1") return true;
            if (raw == "0") return false;
            if (bool.TryParse(raw, out bool b)) return b;
            return fallback;
        }
    }
}
