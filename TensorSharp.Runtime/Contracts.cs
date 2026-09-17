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
using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    public interface IModelArchitecture : IDisposable
    {
        ModelConfig Config { get; }
        ITokenizer Tokenizer { get; }
        IMultimodalInjector MultimodalInjector { get; }
        IBackendExecutionPlan ExecutionPlan { get; }
        float[] Forward(int[] tokens);
        void ResetKVCache();

        /// <summary>
        /// Process-wide GPU-compute serialisation lock. Every caller that
        /// drives the underlying GGML/Metal/CUDA backend through this model
        /// must take this lock for the duration of the GPU work. See
        /// <c>ModelBase.GpuComputeLock</c> for the rationale (backend
        /// command queues are not thread-safe; a parallel image-bearing
        /// request crashed the process by racing the engine's batch step).
        /// Default implementation returns a per-instance object; concrete
        /// models inherit a single shared lock from ModelBase.
        /// </summary>
        object GpuComputeLock => this; // overridden by ModelBase to return a real lock

        /// <summary>
        /// Whether this architecture can rewind its KV state to an earlier prefix length.
        /// Models with recurrent / SSM state (e.g. Qwen3.5 GatedDeltaNet, Nemotron Mamba2)
        /// cannot truncate because their running state cannot be reversed; for those the
        /// only valid reuse pattern is "cached prefix is a prefix of the new input".
        /// </summary>
        bool SupportsKVCacheTruncation { get; }
        void TruncateKVCache(int tokenCount);

        /// <summary>
        /// Truncate to <paramref name="tokenCount"/>, or report that this model cannot
        /// reach that far back and leave its state untouched so the caller can reset and
        /// re-prefill instead.
        ///
        /// <para>Why a model that says <see cref="SupportsKVCacheTruncation"/> may still
        /// refuse: an architecture whose caches are addressed modularly (a sliding-window
        /// ring, a compressor state ring) keeps only a bounded span of positions, so how
        /// far a rewind can go depends on where the sequence currently is - it is not a
        /// property of the architecture. DeepSeek V4.1 is the case in hand. A model with
        /// no such bound never refuses, which is why the default is
        /// "truncate and return true".</para>
        ///
        /// <para>Every caller with a reset-and-re-prefill fallback should prefer this over
        /// <see cref="TruncateKVCache"/>; the void form throws on refusal rather than
        /// continuing from a head that did not move.</para>
        /// </summary>
        bool TryTruncateKVCache(int tokenCount)
        {
            TruncateKVCache(tokenCount);
            return true;
        }

        /// <summary>
        /// The multiple a <see cref="TryTruncateKVCache"/> target must be. Callers align
        /// a reusable prefix length DOWN to this before asking, so a model that can only
        /// rewind to a block boundary keeps the reuse instead of refusing over one token.
        /// 1 (the default) means any length.
        /// </summary>
        int KVCacheTruncationGranularity => 1;

        /// <summary>
        /// Whether a cache holding <paramref name="cachedTokenCount"/> tokens can
        /// retain the requested prefix without losing history needed by later
        /// attention. This side-effect-free check also applies to inactive retained
        /// holders, so the scheduler can prefer an exact checkpoint before binding
        /// one. A true result still requires TryTruncateKVCache at execution time.
        /// </summary>
        bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount)
            => targetTokenCount >= 0 && targetTokenCount <= cachedTokenCount
                && (targetTokenCount == cachedTokenCount
                    || (SupportsKVCacheTruncation
                        && targetTokenCount % Math.Max(1, KVCacheTruncationGranularity) == 0));

        /// <summary>
        /// Whether this architecture exposes block-level snapshot / restore of its KV
        /// state through <see cref="TryExtractKVBlock"/> and <see cref="TryInjectKVBlock"/>.
        /// Required for the paged KV cache. Models with recurrent state should return
        /// false. Defaults to false in <c>ModelBase</c>; pure-attention models opt in.
        /// </summary>
        bool SupportsKVStateSnapshot => false;

        /// <summary>
        /// Whether K/V state captured by one sequence can be safely re-injected into a
        /// DIFFERENT sequence's freshly-reset cache. This drives two reuse paths:
        /// cross-request prefix-cache adoption and the executor's ownership swap.
        /// It is distinct from <see cref="SupportsKVStateSnapshot"/> (which only gates
        /// whether the paged engine can run at all): a model may snapshot fine for its
        /// own continuous decode yet be unable to faithfully restore a snapshot into a
        /// fresh cache. Gemma 4 caps this byte-snapshot path because its local K/V is
        /// circular. Qwen 3.5/3.6 opts out because attention K/V alone is not a
        /// complete cross-request continuation without the matching GatedDeltaNet
        /// recurrent state; it may instead retain a complete request-owned fused
        /// holder through <see cref="IBatchedPagedModel.SupportsRetainedFusedCache"/>.
        /// Such models return false to force a correct re-prefill when no complete
        /// holder applies. Defaults to <see cref="SupportsKVStateSnapshot"/>.
        /// </summary>
        bool SupportsCrossSequenceKvReuse => SupportsKVStateSnapshot;

        /// <summary>
        /// Maximum number of leading prompt tokens whose K/V snapshot can be faithfully
        /// re-injected into a different (or re-admitted) sequence. Full-attention models
        /// can reuse an unbounded prefix. Sliding-window / circular-cache models (Gemma 4)
        /// can only reliably restore the last window's worth of positions, so they cap
        /// this at the window size; the engine reuses up to the cap and re-prefills the
        /// rest. This describes byte snapshots, not the separate complete-holder
        /// retention contract. Defaults to unbounded.
        /// </summary>
        int MaxReusablePrefixTokens => int.MaxValue;

        /// <summary>
        /// Whether a cache that already holds a media span (image, video pair, audio
        /// clip) can be continued past it with the same result a fresh prefill gives.
        /// True for models whose positions after a span are the plain token index
        /// (Gemma 4 and every other absolute-position family). Qwen 3.5 returns false:
        /// its M-RoPE prompt positions compress after an image, but decode positions
        /// are the absolute index and the cache records no rope delta, so the state
        /// after an image turn is not the state a re-prefill of the same history
        /// builds. For such a model every prompt-reuse path stops at the first media
        /// span, and reuse of the text BEFORE the span is unaffected.
        /// </summary>
        bool SupportsReuseAcrossMediaSpan => true;

        /// <summary>
        /// Whether a prompt of <paramref name="promptTokens"/> tokens whose media span
        /// lies in the part still to prefill may continue a reused prefix, i.e. prefill
        /// that media at a non-zero start position and still match a fresh prefill. True
        /// by default, and true for every shipped model. A model that returns false has
        /// such a turn reuse nothing past its public prefix
        /// (SequenceState.SharedPrefixTokens): the prefill is cut there for the
        /// shared-prefix checkpoint anyway, so cloning the checkpoint changes nothing, and
        /// without a public prefix it prefills from zero. (Gemma 4 returned false past its
        /// sliding window until its image chunks after a reused prefix were made exact.)
        /// </summary>
        bool CanPrefillMediaAfterReusedPrefix(int promptTokens) => true;

        /// <summary>
        /// Maximum context length (in tokens) this model can serve — its KV cache
        /// grows on demand up to this bound. The paged engine uses it to size the
        /// KV block pool so a long in-context prompt cannot exhaust the block table
        /// mid-prefill (which would otherwise deadlock the sole running sequence:
        /// no free blocks to allocate and, solo, nothing to preempt). Returns 0
        /// when the model does not advertise a context length; the engine then
        /// keeps the configured default pool size.
        /// </summary>
        int MaxContextLength => 0;

        /// <summary>
        /// Hint, issued once at the first chunk of a fresh prefill (start_pos == 0),
        /// that the request may need <paramref name="requiredContextTokens"/> slots
        /// for its prompt plus declared generation budget.
        /// A model with a grow-on-demand KV cache can allocate it to the final size
        /// up front — at start_pos == 0 there is no committed K/V to copy, so the
        /// grow is essentially free, whereas incremental doubling during the prefill
        /// re-copies (and device↔host round-trips) the whole cache several times.
        /// Default no-op.
        /// </summary>
        void PrepareForPrefill(int requiredContextTokens) { }

        /// <summary>Release memory that only serves the NEXT request's speed — parked
        /// per-request holders, pooled host buffers — because the host has been told
        /// the process is about to run out. Called on the engine thread between steps,
        /// so no forward is in flight; a live cache is never touched. Default: nothing
        /// to release.</summary>
        void TrimIdleMemory() { }

        /// <summary>
        /// Stable identifier tying snapshots to a specific (model, layer count, head
        /// counts, head dim, KV dtype) tuple. Snapshots are only safe to restore into
        /// a model whose fingerprint matches the one in effect when they were captured.
        /// </summary>
        string KVStateFingerprint => string.Empty;

        /// <summary>
        /// Bytes occupied by a block of <paramref name="tokenCount"/> tokens worth of
        /// K/V state across all layers. Returns 0 if snapshotting is not supported.
        /// </summary>
        long ComputeKVBlockByteSize(int tokenCount) => 0;

        /// <summary>
        /// Element type of the bytes returned by <see cref="TryExtractKVBlock"/> /
        /// consumed by <see cref="TryInjectKVBlock"/>. Used by the paged tier's
        /// optional TurboQuant codec to decide how to interpret the raw payload
        /// before re-quantizing it. Defaults to <see cref="KvCodecElementType.Float32"/>;
        /// models with F16 or Q8_0 caches should override.
        /// </summary>
        KvCodecElementType KVStateElementType => KvCodecElementType.Float32;

        /// <summary>
        /// Whether this architecture must be snapshotted at every block boundary
        /// DURING prefill, rather than once at the end. Recurrent / SSM layers
        /// (Qwen 3.5 GatedDeltaNet, Nemotron Mamba2) need this because the running
        /// state at position N is a function of tokens 0..N-1; capturing post-
        /// prefill would record the same final state for every block.
        /// </summary>
        bool RequiresPerBlockCapture => false;

        /// <summary>
        /// Copy the bytes for token positions <c>[startToken, startToken+tokenCount)</c>
        /// of the model's per-layer K/V cache into <paramref name="destination"/>. The
        /// destination must be exactly <see cref="ComputeKVBlockByteSize"/> bytes wide.
        /// Returns false if the requested range is not valid (e.g. extends past the
        /// model's currently-cached tokens) or the model does not support snapshots.
        /// </summary>
        bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination) => false;

        /// <summary>
        /// Write a block of K/V bytes at token position <paramref name="destToken"/>
        /// of the model's per-layer K/V cache. After a successful call the model
        /// behaves as if <paramref name="tokenCount"/> tokens were forwarded into the
        /// cache at that position. <paramref name="destToken"/> must equal the
        /// current cached token count - in other words the manager always appends
        /// in order from position 0. Returns false on size mismatch or unsupported.
        /// </summary>
        bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source) => false;
    }

    public interface IPromptRenderer
    {
        string Render(
            string template,
            List<ChatMessage> messages,
            bool addGenerationPrompt = true,
            string? architecture = null,
            List<ToolFunction>? tools = null,
            bool enableThinking = false);

        /// <summary>
        /// Render with the request's <c>reasoning_effort</c> level. The default
        /// implementation ignores the level, which is right for every renderer whose
        /// family has no such prompt line; <see cref="GgufPromptRenderer"/> forwards it.
        /// </summary>
        string Render(
            string template,
            List<ChatMessage> messages,
            bool addGenerationPrompt,
            string? architecture,
            List<ToolFunction>? tools,
            bool enableThinking,
            string? reasoningEffort)
            => Render(template, messages, addGenerationPrompt, architecture, tools, enableThinking);
    }

    public interface IOutputProtocolParser
    {
        void Init(bool enableThinking, List<ToolFunction>? tools);
        ParsedOutput Add(string text, bool done);
        bool HasThinkingSupport { get; }
        bool HasToolSupport { get; }
        bool AlwaysRequired { get; }
    }

    public interface IMultimodalInjector
    {
        void LoadProjectors(string mmProjPath);

        /// <summary>
        /// Prepare media embeddings for a request and expand any media-placeholder tokens.
        /// When <paramref name="requestId"/> is provided the prepared embeddings are stored
        /// in a per-request bucket so concurrent requests don't clobber each other; when null
        /// (legacy single-threaded path) they go into a shared default bucket.
        /// </summary>
        List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens, string requestId = null);

        /// <summary>
        /// Queue any media embeddings whose insertion span lies AFTER <paramref name="reusablePrefixTokenCount"/>.
        /// Returns true if any embedding span overlaps the suffix that will be re-forwarded.
        /// </summary>
        bool QueuePromptEmbeddings(int reusablePrefixTokenCount, string requestId = null);

        /// <summary>
        /// Queue the portions of any prepared media embeddings that overlap the prompt-token
        /// slice <c>[promptStartToken, promptStartToken + tokenCount)</c>. Insert positions are
        /// adjusted so the embeddings line up with the sliced token batch passed to Forward.
        /// </summary>
        bool QueuePromptEmbeddingsForSlice(int promptStartToken, int tokenCount, string requestId = null);

        /// <summary>
        /// Find the largest prefix length &lt;= <paramref name="reusablePrefixTokenCount"/> that does
        /// not split a multimodal embedding span. The model's KV cache for any such span
        /// is only valid when the entire span has been forwarded.
        /// </summary>
        int ClampReusablePrefix(int reusablePrefixTokenCount, string requestId = null);

        /// <summary>
        /// Find the smallest trim-start position &gt;= <paramref name="trimStartTokenCount"/> that does
        /// not split a multimodal embedding span (used when truncating prompts that are too long).
        /// </summary>
        int ClampTrimStart(int trimStartTokenCount, string requestId = null);

        /// <summary>
        /// Drop / shift queued embedding spans after the prompt has been trimmed at the front.
        /// </summary>
        void TrimPreparedPrompt(int trimStartTokenCount, string requestId = null);

        /// <summary>True if the request has any prepared (not-yet-fully-consumed) embeddings.
        /// The engine uses this to force the per-seq forward path for multimodal sequences,
        /// because the batched paged path doesn't currently know how to inject embeddings.</summary>
        bool HasPendingEmbeddings(string requestId);

        /// <summary>The media spans prepared for <paramref name="requestId"/>, in prompt
        /// order, with the content identity of each. The engine compares them
        /// positionally when it continues a cached prefix (see
        /// <see cref="Scheduling.PromptMediaSpans"/>). Call after any front trim.</summary>
        IReadOnlyList<Scheduling.PromptMediaSpan> GetPreparedMediaSpans(string requestId)
            => Array.Empty<Scheduling.PromptMediaSpan>();

        /// <summary>Discard the per-request bucket. Called when a request finishes (success,
        /// error, or abort).</summary>
        void ClearPreparedPromptState(string requestId);
    }

    public interface IBackendExecutionPlan
    {
        BackendType BackendType { get; }
        bool UsesGgmlBackend { get; }
        bool ShouldStoreWeightQuantized(GgufTensorInfo info);
    }
}
