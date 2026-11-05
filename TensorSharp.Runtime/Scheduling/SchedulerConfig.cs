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
        /// hands out the complete <see cref="MaxNumBatchedTokens"/> budget (see
        /// <see cref="PrefillOnlyFirstComeFirstServed"/>) so the accelerator is not
        /// left half idle. Default 256. Overridable via <c>TS_SCHED_PREFILL_CHUNK</c>
        /// or <c>--prefill-chunk-size</c>.</summary>
        public int MaxPrefillChunkSize { get; init; } = 256;

        /// <summary>
        /// How a step with no decode in flight shares its budget among prompts.
        /// False (default): an even share per prompt, so a burst of prompts finishes
        /// prefill together and decodes in shared steps. True: first come, first
        /// served - the oldest prompt takes as much of the budget as it can use and
        /// later prompts the rest. Measured on Qwen3.8-27B with four 18K-token image
        /// prompts, first-come-first-served lost (27.8 s against 17.4 s): the first
        /// prompt to finish then decoded one token per step beside the next prompt's
        /// prefill chunk. Env: <c>TS_SCHED_PREFILL_FCFS=1</c>.
        /// </summary>
        public bool PrefillOnlyFirstComeFirstServed { get; init; }

        /// <summary>
        /// Prompt tokens a step that also decodes may carry across ALL its prefilling
        /// sequences; 0 (default) bounds only each sequence, by
        /// <see cref="MaxPrefillChunkSize"/>. Every decoding sequence waits for the whole
        /// step, so two prompts' chunks beside the decodes doubled their per-round latency
        /// (Qwen3.8-27B agent lanes: ~650 ms rounds against ~350 ms with one chunk).
        /// Env: <c>TS_SCHED_MIXED_PREFILL_TOKENS</c> (0 turns the bound off).
        /// <para>A model that implements
        /// <see cref="Speculative.IMixedStepPrefillTarget.PreferredMixedStepPrefillTokens"/> sets
        /// the bound itself unless this value was set (in code or through the environment variable).</para>
        /// </summary>
        public int MixedStepPrefillBudget
        {
            get => _mixedStepPrefillBudget;
            init { _mixedStepPrefillBudget = value; MixedStepPrefillBudgetChosen = true; }
        }
        private readonly int _mixedStepPrefillBudget;

        /// <summary><see cref="MixedStepPrefillBudget"/> was set rather than left at its default, so a
        /// model's own bound is not used.</summary>
        public bool MixedStepPrefillBudgetChosen { get; private init; }

        /// <summary>
        /// With <see cref="MixedStepPrefillBudget"/>, give a mixed step's prompt budget to
        /// the oldest prefilling sequence until its prompt is done (what Velox does)
        /// instead of rotating it between prompts, so the first prompt starts decoding
        /// sooner. Env: <c>TS_SCHED_MIXED_PREFILL_FCFS=1</c>.
        /// </summary>
        public bool MixedStepPrefillFirstComeFirstServed { get; init; }

        /// <summary>Per-step prefill token cap used ONLY when there is no GPU
        /// contention — i.e. at most one sequence is in the system (running +
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
        /// Default 8192. Env: <c>TS_SCHED_SOLO_PREFILL_CHUNK</c>.
        /// <para>A model that implements <see cref="ISoloPrefillBudgetModel"/> (Muse Spark with
        /// pipelined prefill) chooses a lone prompt's steps itself, possibly above
        /// <see cref="MaxNumBatchedTokens"/>, unless this value was set (in code or through the
        /// environment variable): a chosen value always applies as documented above.</para></summary>
        public int SoloPrefillChunkSize
        {
            get => _soloPrefillChunkSize;
            init { _soloPrefillChunkSize = value; SoloPrefillChunkSizeChosen = true; }
        }
        private readonly int _soloPrefillChunkSize = 8192;

        /// <summary><see cref="SoloPrefillChunkSize"/> was set rather than left at its default, so a
        /// model's own lone-prompt steps (<see cref="ISoloPrefillBudgetModel"/>) are not used.</summary>
        public bool SoloPrefillChunkSizeChosen { get; private init; }

        /// <summary>Number of physical KV blocks in the pool. The total KV-cache
        /// budget is <c>NumBlocks * BlockSize</c> tokens. When the model exposes
        /// its preferred block size that value is used here.</summary>
        public int NumBlocks { get; init; } = 256;

        /// <summary>Block size in tokens; 0 leaves it to the model
        /// (<see cref="InferenceEngine"/> picks <see cref="PageFamilyBlockSize"/> for a family whose
        /// only cross-request reuse is its pages, <see cref="DefaultBlockSize"/> otherwise), which is
        /// what <see cref="FromEnvironment"/> reports when <c>TS_SCHED_BLOCK_SIZE</c> is unset.</summary>
        public int BlockSize { get; init; } = DefaultBlockSize;

        /// <summary>The block size of a family that keeps each conversation's exact state (holders, native
        /// slots): its pool only accounts for reservations.</summary>
        public const int DefaultBlockSize = 256;

        /// <summary>The block size of a family whose only cross-request reuse is whole pages.</summary>
        public const int PageFamilyBlockSize = 16;

        /// <summary>Reuse prompt prefixes across requests through the radix prefix
        /// cache. Default true; <c>TS_SCHED_PREFIX_CACHE=0</c> / <c>--no-prefix-cache</c>
        /// turn it off.</summary>
        public bool EnablePrefixCaching { get; init; } = true;

        /// <summary>The fewest otherwise-unprotected prompt tokens an automatic prompt-tail
        /// checkpoint of a recurrent model must protect, for a request whose reply is rendered
        /// back verbatim (<see cref="SequenceState.PromptTailCheckpointOptional"/>; every other
        /// request always takes it). The checkpoint stops prefill one
        /// token short of the prompt end, so it costs one extra single-row forward (18.7 ms
        /// on Qwen3.8-27B / RTX 5090) on every request it is taken for; a shorter unprotected
        /// tail is cheaper to prefill again on the rare exact retry. Default 1024; 0 takes
        /// one on every prompt. Env: <c>TS_SCHED_PROMPT_TAIL_CHECKPOINT_MIN</c>.</summary>
        public int PromptTailCheckpointMinTokens { get; init; } = 1024;

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
        public SchedulerConfig WithSpeculation(SpeculationOptions speculation)
            => Copy(BlockSize, speculation ?? SpeculationOptions.Disabled);

        /// <summary>This configuration with the block size the engine resolved.</summary>
        internal SchedulerConfig WithBlockSize(int blockSize) => Copy(blockSize, Speculation);

        /// <summary>This configuration with a chosen <see cref="SoloPrefillChunkSize"/>.</summary>
        public SchedulerConfig WithSoloPrefillChunkSize(int tokens) => Copy(BlockSize, Speculation, tokens);

        /// <summary>This configuration with a chosen <see cref="MixedStepPrefillBudget"/>.</summary>
        public SchedulerConfig WithMixedStepPrefillBudget(int tokens) => Copy(BlockSize, Speculation, mixedStepPrefillBudget: tokens);

        private SchedulerConfig Copy(int blockSize, SpeculationOptions speculation, int? soloPrefillChunk = null,
            int? mixedStepPrefillBudget = null) => new()
        {
            MaxNumBatchedTokens = MaxNumBatchedTokens,
            MaxNumRunningSequences = MaxNumRunningSequences,
            MaxPrefillChunkSize = MaxPrefillChunkSize,
            PrefillOnlyFirstComeFirstServed = PrefillOnlyFirstComeFirstServed,
            // Assigned in this order: the copy keeps whether the budget was chosen.
            MixedStepPrefillBudget = mixedStepPrefillBudget ?? MixedStepPrefillBudget,
            MixedStepPrefillBudgetChosen = mixedStepPrefillBudget.HasValue || MixedStepPrefillBudgetChosen,
            MixedStepPrefillFirstComeFirstServed = MixedStepPrefillFirstComeFirstServed,
            // Assigned in this order: the copy keeps whether the size was chosen.
            SoloPrefillChunkSize = soloPrefillChunk ?? SoloPrefillChunkSize,
            SoloPrefillChunkSizeChosen = soloPrefillChunk.HasValue || SoloPrefillChunkSizeChosen,
            NumBlocks = NumBlocks,
            BlockSize = blockSize,
            EnablePrefixCaching = EnablePrefixCaching,
            PromptTailCheckpointMinTokens = PromptTailCheckpointMinTokens,
            StopRepetition = StopRepetition,
            DecodeQuantumTokens = DecodeQuantumTokens,
            Speculation = speculation,
        };

        public static SchedulerConfig FromEnvironment()
        {
            if (!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("TS_PREFIX_CACHE_MODE")))
                throw new System.ArgumentException(
                    "TS_PREFIX_CACHE_MODE was removed: the radix prefix cache is the only prefix cache. " +
                    "Unset it; TS_SCHED_PREFIX_CACHE=0 (or --no-prefix-cache) turns prefix caching off.");
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = ReadInt("TS_SCHED_MAX_BATCHED_TOKENS", 4096),
                MaxNumRunningSequences = ReadInt("TS_SCHED_MAX_RUNNING_SEQS", 16),
                MaxPrefillChunkSize = ReadInt("TS_SCHED_PREFILL_CHUNK", 256),
                PrefillOnlyFirstComeFirstServed = ReadBool("TS_SCHED_PREFILL_FCFS", false),
                MixedStepPrefillFirstComeFirstServed = ReadBool("TS_SCHED_MIXED_PREFILL_FCFS", false),
                NumBlocks = ReadInt("TS_SCHED_NUM_BLOCKS", 256),
                BlockSize = ReadInt("TS_SCHED_BLOCK_SIZE", 0),
                EnablePrefixCaching = ReadBool("TS_SCHED_PREFIX_CACHE", true),
                PromptTailCheckpointMinTokens = ReadNonNegativeInt("TS_SCHED_PROMPT_TAIL_CHECKPOINT_MIN", 1024),
                StopRepetition = ReadBool("TS_SCHED_STOP_REPETITION", true),
                DecodeQuantumTokens = ReadInt("TS_SCHED_DECODE_QUANTUM", 256),
                Speculation = SpeculationOptions.FromEnvironment(),
            };
            // Only a value the user gave counts as chosen (see SoloPrefillChunkSize and
            // MixedStepPrefillBudget); an explicit 0 chooses no mixed-step bound.
            if (!string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable("TS_SCHED_MIXED_PREFILL_TOKENS")))
                cfg = cfg.WithMixedStepPrefillBudget(ReadInt("TS_SCHED_MIXED_PREFILL_TOKENS", 0));
            int solo = ReadInt("TS_SCHED_SOLO_PREFILL_CHUNK", 0);
            return solo > 0 ? cfg.WithSoloPrefillChunkSize(solo) : cfg;
        }

        private static int ReadNonNegativeInt(string name, int fallback)
        {
            string? raw = System.Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrEmpty(raw)) return fallback;
            if (!int.TryParse(raw, out int v) || v < 0)
                throw new System.ArgumentException($"{name} must be a non-negative integer.");
            return v;
        }

        private static int ReadInt(string name, int fallback)
        {
            string? raw = System.Environment.GetEnvironmentVariable(name);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out int v) && v > 0)
                return v;
            return fallback;
        }

        // Boolean flag reader that accepts "0"/"1" (and "true"/"false"). Unlike
        // ReadInt, this honours an explicit "0" so a flag can actually be disabled.
        private static bool ReadBool(string name, bool fallback)
        {
            string? raw = System.Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrEmpty(raw)) return fallback;
            raw = raw.Trim();
            if (raw == "1") return true;
            if (raw == "0") return false;
            if (bool.TryParse(raw, out bool b)) return b;
            return fallback;
        }
    }
}
