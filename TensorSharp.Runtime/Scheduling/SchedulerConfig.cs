// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
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

        /// <summary>Maximum number of new tokens to schedule for a single
        /// sequence's prefill in one step. Chunked prefill caps the per-step
        /// work on long prompts and lets decode sequences interleave with
        /// long prompts. Default 1024.
        ///
        /// Historical note: an earlier benchmark with the (since-disabled-by-
        /// default) N=1 fast path in <see cref="BatchExecutor.ExecuteStep"/>
        /// showed smaller chunks hurt wall-clock for mixed prefill+decode
        /// workloads — that path got the first request through fused decode
        /// before the second one arrived, and small chunks pushed more of
        /// it onto the slow batched path. With the fast path off by default
        /// the chunk-size sensitivity is much milder; 1024 is still a sane
        /// default. Overridable via <c>TS_SCHED_PREFILL_CHUNK</c> env var
        /// or the <c>--prefill-chunk-size</c> CLI flag.</summary>
        public int MaxPrefillChunkSize { get; init; } = 1024;

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

        /// <summary>How many decode steps a running sequence is allowed to run
        /// consecutively before the scheduler may swap to another sequence.
        /// In the current C# executor each session-switch pays a KV-state
        /// extract+inject round-trip, so we amortize that by running multiple
        /// decode tokens for the same session before switching. Set to 1 for
        /// strict per-token fairness; default is the block size so we naturally
        /// swap at block boundaries.</summary>
        public int DecodeQuantumTokens { get; init; } = 256;

        public static SchedulerConfig Default => new();

        public static SchedulerConfig FromEnvironment()
        {
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = ReadInt("TS_SCHED_MAX_BATCHED_TOKENS", 4096),
                MaxNumRunningSequences = ReadInt("TS_SCHED_MAX_RUNNING_SEQS", 16),
                MaxPrefillChunkSize = ReadInt("TS_SCHED_PREFILL_CHUNK", 1024),
                NumBlocks = ReadInt("TS_SCHED_NUM_BLOCKS", 256),
                BlockSize = ReadInt("TS_SCHED_BLOCK_SIZE", 256),
                EnablePrefixCaching = ReadBool("TS_SCHED_PREFIX_CACHE", true),
                DecodeQuantumTokens = ReadInt("TS_SCHED_DECODE_QUANTUM", 256),
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
