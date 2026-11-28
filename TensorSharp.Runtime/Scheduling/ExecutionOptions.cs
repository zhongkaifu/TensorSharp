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
    /// Operator-level settings for the engine's step routing and cache budgets,
    /// read from <c>TS_*</c> environment variables in ONE place instead of
    /// scattered <c>Environment.GetEnvironmentVariable</c> checks at each
    /// decision point. <see cref="BatchExecutor"/> materialises a snapshot per
    /// step (env reads are cheap and tests toggle these variables at runtime,
    /// so the values are deliberately NOT cached for the process lifetime) and
    /// hands it to <see cref="ExecutionPlanner"/> together with the model's
    /// <see cref="ExecutionCapabilities"/>.
    /// </summary>
    public sealed record ExecutionOptions
    {
        /// <summary>Force the per-sequence KV-swap path even when the model
        /// implements <see cref="IBatchedPagedModel"/>. Env:
        /// <c>TS_SCHED_DISABLE_BATCHED</c> (default off), which
        /// <c>--no-continuous-batching</c> sets.</summary>
        public bool BatchedPathDisabled { get; init; }

        /// <summary>Serve concurrent (N&gt;=2) sequences on fused-capable models
        /// by running each through its own fused Forward with a per-request KV
        /// cache (a holder). Env: <c>TS_PER_SEQ_FUSED</c> (default on). 0 serves
        /// them on the op-by-op batched paged path instead, whose K/V lives in the
        /// engine's shared host block pool: no per-request device K/V, at the
        /// batched path's lower decode rate and without retained holders.</summary>
        public bool PerSeqFusedEnabled { get; init; } = true;

        /// <summary>TRUE token-batched fused decode inside the per-sequence
        /// fused path: decode one token for each of N sequences in a single
        /// fused graph (the bespoke families' own batched graphs; the shared
        /// runtime's unified step always batches its slots and does not read
        /// it). This is what
        /// lifts concurrent decode from the round-robin ~1x ceiling to the
        /// vLLM-class batched rate, so it is ON by default; models that cannot
        /// batch a step decline per call and fall back per-sequence. Env:
        /// <c>TS_BATCHED_FUSED_DECODE</c>: 0 decodes each sequence in its own
        /// fused forward, whose output does not depend on which requests share
        /// a step, at the round-robin rate.</summary>
        public bool BatchedFusedDecodeEnabled { get; init; } = true;

        /// <summary>How many finished conversations' end states (retained fused
        /// holders, retained native slots) to keep alive for cross-request prefix
        /// reuse; each pins the model's complete per-request continuation state
        /// (attention K/V alone for Gemma 4, K/V plus GatedDeltaNet recurrent state
        /// for the Qwen 3.5 family). Null when unset: the count then follows the
        /// engine's concurrency (<see cref="RetainedFusedCacheBudgetFor"/>); 0 turns
        /// retention off. Env: <c>TS_RETAINED_FUSED_CACHE_MAX</c>.</summary>
        public int? RetainedFusedCacheBudget { get; init; }

        /// <summary>The fewest end states an unset <see cref="RetainedFusedCacheBudget"/> keeps.</summary>
        public const int MinDefaultRetainedFusedCacheBudget = 4;

        /// <summary>The end-state count in force for an engine that runs up to
        /// <paramref name="maxRunningSequences"/> sequences at once. Unset, every
        /// sequence that can run in parallel keeps its conversation's state for the
        /// next turn: at a fixed 4, eight parallel conversations evicted each other's
        /// and the four oldest re-prefilled every turn. The count is not what bounds
        /// memory; the prefix tree evicts past the device and host headroom it
        /// measures. An explicit value wins (TensorAgent pins 1 on the phone, and 0
        /// turns retention off).</summary>
        public int RetainedFusedCacheBudgetFor(int maxRunningSequences)
            => RetainedFusedCacheBudget ?? Math.Max(MinDefaultRetainedFusedCacheBudget, maxRunningSequences);

        /// <summary>How many public checkpoints of the model's state at the end of the
        /// prompt prefix every conversation shares (see
        /// <c>IBatchedPagedModel.SupportsPrefixCheckpoints</c>) to keep; each new chat
        /// starts from a clone of one, and 0 turns them off. A prompt publishes
        /// one per declared boundary (the end of the system instructions as well as the
        /// end of the shared prefix; see SequenceState.PublicCheckpointBoundaries), and a
        /// host that warms both thinking modes, whose prefixes differ from the first
        /// tokens, needs both of each: at 2 the second warm-up evicted the first mode's
        /// pair and every new chat in that mode prefilled the whole prompt again.
        /// Env: <c>TS_PREFIX_CHECKPOINTS_MAX</c> (default 4).</summary>
        public int PrefixCheckpointBudget { get; init; } = 4;

        /// <summary>Cache isolation for shared, stateless chat API sessions. Lineage
        /// (the default) requires a recorded conversation continuation; Strict gives
        /// every request a fresh scope. Shared is an operator opt-in for a trusted
        /// single-tenant server whose requests may reuse matching private prefixes
        /// even when a client edits earlier assistant turns or replaces screenshots.
        /// Named chat sessions retain their own isolation in every mode. Env:
        /// <c>TS_PREFIX_CACHE_ISOLATION</c> = lineage | strict | shared.</summary>
        public PrefixCache.IsolationMode PrefixCacheIsolation { get; init; } = PrefixCache.IsolationMode.Lineage;

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
            PerSeqFusedEnabled = ReadFlag("TS_PER_SEQ_FUSED", true),
            BatchedFusedDecodeEnabled = ReadFlag("TS_BATCHED_FUSED_DECODE", true),
            RetainedFusedCacheBudget = ReadOptionalNonNegativeInt("TS_RETAINED_FUSED_CACHE_MAX"),
            PrefixCheckpointBudget = ReadNonNegativeInt("TS_PREFIX_CHECKPOINTS_MAX", 4),
            PrefixCacheIsolation = ReadPrefixCacheIsolation(),
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
            if (!PerSeqFusedEnabled) parts.Add("TS_PER_SEQ_FUSED=0");
            if (!BatchedFusedDecodeEnabled) parts.Add("TS_BATCHED_FUSED_DECODE=0");
            if (RetainedFusedCacheBudget is int retained) parts.Add($"TS_RETAINED_FUSED_CACHE_MAX={retained}");
            if (PrefixCheckpointBudget != 4) parts.Add($"TS_PREFIX_CHECKPOINTS_MAX={PrefixCheckpointBudget}");
            if (PrefixCacheIsolation != PrefixCache.IsolationMode.Lineage)
                parts.Add($"TS_PREFIX_CACHE_ISOLATION={PrefixCacheIsolation.ToString().ToLowerInvariant()}");
            if (KvInitialTokens != 0) parts.Add($"TS_KV_INITIAL_TOKENS={KvInitialTokens}");
            if (KvGenerationReserveMax != 0) parts.Add($"TS_KV_GENERATION_RESERVE_MAX={KvGenerationReserveMax}");
            if (KvHolderPoolMax != 64) parts.Add($"TS_KV_HOLDER_POOL_MAX={KvHolderPoolMax}");
            return string.Join(", ", parts);
        }

        // Loose boolean: unset -> default; "0"/"false" -> false; anything else -> true.
        private static PrefixCache.IsolationMode ReadPrefixCacheIsolation()
        {
            string? raw = Environment.GetEnvironmentVariable("TS_PREFIX_CACHE_ISOLATION");
            return raw?.Trim().ToLowerInvariant() switch
            {
                null or "" or "lineage" => PrefixCache.IsolationMode.Lineage,
                "strict" => PrefixCache.IsolationMode.Strict,
                "shared" => PrefixCache.IsolationMode.Shared,
                _ => throw new ArgumentException("TS_PREFIX_CACHE_ISOLATION must be lineage, strict, or shared (trusted single-tenant only)."),
            };
        }

        private static bool ReadFlag(string name, bool fallback)
        {
            string? raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrEmpty(raw)) return fallback;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        private static int ReadNonNegativeInt(string name, int fallback)
        {
            string? raw = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v >= 0)
                return v;
            return fallback;
        }

        private static int? ReadOptionalNonNegativeInt(string name)
        {
            string? raw = Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v >= 0)
                return v;
            return null;
        }
    }
}
