// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Operator-level execution-path overrides for the engine's step routing,
    /// read from <c>TS_*</c> environment variables in ONE place instead of
    /// scattered <c>Environment.GetEnvironmentVariable</c> checks at each
    /// decision point. <see cref="BatchExecutor"/> materialises a snapshot per
    /// step (env reads are cheap and tests toggle these variables at runtime,
    /// so the values are deliberately NOT cached for the process lifetime) and
    /// hands it to <see cref="ExecutionPlanner"/> together with the model's
    /// <see cref="ExecutionCapabilities"/>.
    ///
    /// Every flag keeps the exact parse semantics of the check it replaced so
    /// existing deployments and A/B scripts behave identically.
    /// </summary>
    public sealed record ExecutionOptions
    {
        /// <summary>Force the per-sequence KV-swap fallback even when the model
        /// implements <see cref="IBatchedPagedModel"/>. Used to A/B the batched
        /// and per-sequence paths on the same workload.
        /// Env: <c>TS_SCHED_DISABLE_BATCHED</c> (default off).</summary>
        public bool BatchedPathDisabled { get; init; }

        /// <summary>Serve a solo scheduled sequence through the model's fused
        /// single-graph <c>Forward</c> (linear KV cache) instead of the op-by-op
        /// batched path — dramatically faster on models with a fused decode
        /// kernel. Env: <c>TS_BATCHED_N1_FAST_PATH</c> (default on; set 0 to A/B
        /// the fully-batched path).</summary>
        public bool BatchedN1FastPathEnabled { get; init; } = true;

        /// <summary>Serve concurrent (N&gt;=2) sequences on fused-capable models
        /// by running each through its own fused Forward with a per-request KV
        /// cache. Env: <c>TS_PER_SEQ_FUSED</c> (default on; set 0 to force the
        /// op-by-op batched paged path for A/B or debugging).</summary>
        public bool PerSeqFusedEnabled { get; init; } = true;

        /// <summary>TRUE token-batched fused decode inside the per-sequence
        /// fused path: decode one token for each of N sequences in a single
        /// fused graph (slot-stable arena KV + captured CUDA graph + on-device
        /// greedy sampling — see ggml_ops_gptoss_batched.cpp). This is what
        /// lifts concurrent decode from the round-robin ~1x ceiling to the
        /// vLLM-class batched rate, so it is ON by default; models that cannot
        /// batch a step decline per call and fall back per-sequence. Env:
        /// <c>TS_BATCHED_FUSED_DECODE</c> (set 0 to disable for A/B).</summary>
        public bool BatchedFusedDecodeEnabled { get; init; } = true;

        /// <summary>Retain finished request-owned fused holders for exact-prefix
        /// continuation across requests. A holder may be attention K/V alone
        /// (for example Gemma 4) or complete hybrid state (Qwen 3.5/3.6 keeps
        /// both attention K/V and GatedDeltaNet recurrent state). Applies only
        /// when the model advertises retained-holder support. Env:
        /// <c>TS_RETAINED_FUSED_CACHE</c> (default on; kill-switch for A/B or
        /// to cap VRAM use).</summary>
        public bool RetainedFusedCacheEnabled { get; init; } = true;

        /// <summary>How many finished fused holders to keep alive for
        /// cross-request prefix reuse; each pins the model's complete
        /// per-request continuation state, including recurrent state where
        /// applicable. Env: <c>TS_RETAINED_FUSED_CACHE_MAX</c> (default 4).</summary>
        public int RetainedFusedCacheBudget { get; init; } = 4;

        /// <summary>Take a checkpoint of the model's state at the end of the prompt
        /// prefix every conversation shares (see
        /// <c>IBatchedPagedModel.SupportsPrefixCheckpoints</c>) and start each new
        /// chat from a clone of it. Env: <c>TS_PREFIX_CHECKPOINTS</c> (default on).</summary>
        public bool PrefixCheckpointsEnabled { get; init; } = true;

        /// <summary>How many distinct shared prefixes to keep checkpointed at once —
        /// two covers a thinking toggle that changes the prefix (Gemma 4 marks it at
        /// the top of the system turn). Env: <c>TS_PREFIX_CHECKPOINTS_MAX</c>
        /// (default 2).</summary>
        public int PrefixCheckpointBudget { get; init; } = 2;

        /// <summary>Tokens of K/V a cache is given when it is CREATED — the model's
        /// primary cache at load, and every per-request holder — before any request
        /// has declared a budget of its own. 0 keeps the engine's policy: the whole
        /// window when <c>MAX_CONTEXT</c> is explicit, otherwise a backend-tuned
        /// default. The cache still grows on demand and a request still reserves what
        /// it declares (see <see cref="KvGenerationReserveMax"/>); this only decides
        /// what is committed before then. It matters where memory is the limit: every
        /// holder the engine keeps — the idle primary cache once the per-request path
        /// takes over, each retained conversation, each parked holder — is paid at
        /// this size, host copy and device mirror both, whether or not a token was
        /// ever written to it. Env: <c>TS_KV_INITIAL_TOKENS</c> (default 0).</summary>
        public int KvInitialTokens { get; init; }

        /// <summary>Cap on the GENERATION share of the K/V a request reserves before
        /// its first prefill chunk (<c>BatchExecutor.BuildPrefillChunk</c> reserves
        /// prompt + max_new_tokens in one allocation). A reply limit larger than the
        /// window otherwise reserves the whole window for every request, however short
        /// the conversation is. Past the cap the cache grows on demand while the reply
        /// runs. 0 = no cap. Env: <c>TS_KV_GENERATION_RESERVE_MAX</c> (default 0).</summary>
        public int KvGenerationReserveMax { get; init; }

        /// <summary>How many released per-request holders a model may PARK for reuse
        /// instead of freeing. Parking keeps stable host pointers (and, on discrete
        /// GPUs, the device mirrors) so a later request skips allocation and graph
        /// capture; each parked holder costs its whole K/V allocation for as long as it
        /// waits. Env: <c>TS_KV_HOLDER_POOL_MAX</c> (default 64).</summary>
        public int KvHolderPoolMax { get; init; } = 64;

        /// <summary>All defaults — the configuration used when no TS_* override
        /// is set. Handy for tests.</summary>
        public static ExecutionOptions Default { get; } = new();

        /// <summary>Read the current override state from the environment.</summary>
        public static ExecutionOptions FromEnvironment() => new()
        {
            BatchedPathDisabled = ReadFlag("TS_SCHED_DISABLE_BATCHED", false),
            BatchedN1FastPathEnabled = ReadFlag("TS_BATCHED_N1_FAST_PATH", true),
            PerSeqFusedEnabled = ReadFlag("TS_PER_SEQ_FUSED", true),
            BatchedFusedDecodeEnabled = ReadFlag("TS_BATCHED_FUSED_DECODE", true),
            RetainedFusedCacheEnabled = ReadFlag("TS_RETAINED_FUSED_CACHE", true),
            RetainedFusedCacheBudget = ReadNonNegativeInt("TS_RETAINED_FUSED_CACHE_MAX", 4),
            PrefixCheckpointsEnabled = ReadFlag("TS_PREFIX_CHECKPOINTS", true),
            PrefixCheckpointBudget = ReadNonNegativeInt("TS_PREFIX_CHECKPOINTS_MAX", 2),
            KvInitialTokens = ReadNonNegativeInt("TS_KV_INITIAL_TOKENS", 0),
            KvGenerationReserveMax = ReadNonNegativeInt("TS_KV_GENERATION_RESERVE_MAX", 0),
            KvHolderPoolMax = ReadNonNegativeInt("TS_KV_HOLDER_POOL_MAX", 64),
        };

        /// <summary>One-line summary of the non-default overrides in effect
        /// (empty string when everything is at its default).</summary>
        public string DescribeOverrides()
        {
            var parts = new System.Collections.Generic.List<string>();
            if (BatchedPathDisabled) parts.Add("TS_SCHED_DISABLE_BATCHED");
            if (!BatchedN1FastPathEnabled) parts.Add("TS_BATCHED_N1_FAST_PATH=0");
            if (!PerSeqFusedEnabled) parts.Add("TS_PER_SEQ_FUSED=0");
            if (!BatchedFusedDecodeEnabled) parts.Add("TS_BATCHED_FUSED_DECODE=0");
            if (!RetainedFusedCacheEnabled) parts.Add("TS_RETAINED_FUSED_CACHE=0");
            if (RetainedFusedCacheBudget != 4) parts.Add($"TS_RETAINED_FUSED_CACHE_MAX={RetainedFusedCacheBudget}");
            if (!PrefixCheckpointsEnabled) parts.Add("TS_PREFIX_CHECKPOINTS=0");
            if (PrefixCheckpointBudget != 2) parts.Add($"TS_PREFIX_CHECKPOINTS_MAX={PrefixCheckpointBudget}");
            if (KvInitialTokens != 0) parts.Add($"TS_KV_INITIAL_TOKENS={KvInitialTokens}");
            if (KvGenerationReserveMax != 0) parts.Add($"TS_KV_GENERATION_RESERVE_MAX={KvGenerationReserveMax}");
            if (KvHolderPoolMax != 64) parts.Add($"TS_KV_HOLDER_POOL_MAX={KvHolderPoolMax}");
            return string.Join(", ", parts);
        }

        // Loose boolean: unset -> default; "0"/"false" -> false; anything else -> true.
        // (The historical parse rule of the flags this type replaced.)
        private static bool ReadFlag(string name, bool fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrEmpty(raw)) return fallback;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        private static int ReadNonNegativeInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v >= 0)
                return v;
            return fallback;
        }
    }
}
