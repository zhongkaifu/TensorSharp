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
        /// fresh cache. Gemma 4's sliding-window / circular cache is exactly that case —
        /// the byte-level restore does not reproduce a fresh prefill, so reusing it
        /// produces corrupted output. Such models return false to force a correct
        /// re-prefill. Defaults to <see cref="SupportsKVStateSnapshot"/>.
        /// </summary>
        bool SupportsCrossSequenceKvReuse => SupportsKVStateSnapshot;

        /// <summary>
        /// Maximum number of leading prompt tokens whose K/V snapshot can be faithfully
        /// re-injected into a different (or re-admitted) sequence. Full-attention models
        /// can reuse an unbounded prefix. Sliding-window / circular-cache models (Gemma 4)
        /// can only reliably restore the last window's worth of positions, so they cap
        /// this at the window size; the engine reuses up to the cap and re-prefills the
        /// rest. Defaults to unbounded.
        /// </summary>
        int MaxReusablePrefixTokens => int.MaxValue;

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
    }

    public interface IOutputProtocolParser
    {
        void Init(bool enableThinking, List<ToolFunction> tools);
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
