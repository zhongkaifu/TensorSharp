// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Client-facing handle to an in-flight request. Streams sampled tokens
    /// through <see cref="Tokens"/> and exposes finalization metadata via
    /// <see cref="Completion"/>.
    ///
    /// Producers (the engine worker thread) publish via the internal
    /// PublishToken / CompleteFinished / CompleteWithError methods. Consumers
    /// read via the standard async channel pattern.
    /// </summary>
    public sealed class InferenceRequestHandle
    {
        private readonly Channel<int> _tokens = Channel.CreateUnbounded<int>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        private readonly TaskCompletionSource<InferenceCompletion> _completionTcs =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly CancellationTokenRegistration _ctReg;
        private int _publishedTokens;
        private long _prefillElapsedTicks;
        private long _decodeElapsedTicks;

        public string RequestId => Sequence.RequestId;
        public SequenceState Sequence { get; }
        public ChannelReader<int> Tokens => _tokens.Reader;
        public Task<InferenceCompletion> Completion => _completionTcs.Task;
        public DateTime SubmittedAt => Sequence.SubmittedAt;

        internal InferenceRequestHandle(SequenceState seq, InferenceEngine engine, CancellationToken ct)
        {
            Sequence = seq;
            _ctReg = ct.Register(() => engine.Abort(seq));
        }

        internal void PublishToken(int tokenId)
        {
            // Channel is unbounded; should never fail in practice.
            _tokens.Writer.TryWrite(tokenId);
            Interlocked.Increment(ref _publishedTokens);
        }

        // Only the engine worker writes these counters. Count each executor step
        // once, even when a speculative verification emits several tokens or EOS
        // ends a request without publishing any text.
        internal void RecordForwardTime(SequenceStepResult result)
        {
            if (result.IsPrefill)
                _prefillElapsedTicks += result.ForwardElapsedTicks;
            else
                _decodeElapsedTicks += result.ForwardElapsedTicks;
        }

        internal void CompleteFinished()
        {
            _tokens.Writer.TryComplete();
            _ctReg.Dispose();
            var completion = new InferenceCompletion
            {
                Status = Sequence.Status,
                FinishReason = Sequence.FinishReason,
                OutputTokenCount = Sequence.OutputTokens.Count,
                PromptTokenCount = Sequence.PromptTokens.Count,
                PrefixCacheReusedTokens = Sequence.PrefixCacheReusedTokens,
                FirstTokenAt = Sequence.FirstTokenAt,
                SubmittedAt = Sequence.SubmittedAt,
                PrefillElapsedTicks = _prefillElapsedTicks + Sequence.ReplayPrefillElapsedTicks,
                DecodeElapsedTicks = _decodeElapsedTicks + Sequence.ReplayDecodeElapsedTicks,
            };
            _completionTcs.TrySetResult(completion);
        }

        internal void CompleteWithError(Exception ex)
        {
            _tokens.Writer.TryComplete(ex);
            _ctReg.Dispose();
            _completionTcs.TrySetException(ex);
        }

        internal void CompleteAborted()
        {
            _tokens.Writer.TryComplete();
            _ctReg.Dispose();
            var completion = new InferenceCompletion
            {
                Status = SequenceStatus.FinishedAborted,
                FinishReason = "aborted",
                OutputTokenCount = Sequence.OutputTokens.Count,
                PromptTokenCount = Sequence.PromptTokens.Count,
                PrefixCacheReusedTokens = Sequence.PrefixCacheReusedTokens,
                FirstTokenAt = Sequence.FirstTokenAt,
                SubmittedAt = Sequence.SubmittedAt,
                PrefillElapsedTicks = _prefillElapsedTicks + Sequence.ReplayPrefillElapsedTicks,
                DecodeElapsedTicks = _decodeElapsedTicks + Sequence.ReplayDecodeElapsedTicks,
            };
            _completionTcs.TrySetResult(completion);
        }
    }

    public sealed class InferenceCompletion
    {
        public SequenceStatus Status { get; init; }
        public string? FinishReason { get; init; }
        public int PromptTokenCount { get; init; }
        public int OutputTokenCount { get; init; }
        public int PrefixCacheReusedTokens { get; init; }
        public DateTime? FirstTokenAt { get; init; }
        public DateTime SubmittedAt { get; init; }

        /// <summary>Prompt forward time in <see cref="System.Diagnostics.Stopwatch"/>
        /// ticks, including prompt replay after a cache restoration shortfall.
        /// Excludes prompt rendering, media encoding, scheduler waiting and
        /// cache bookkeeping. Shared batched forwards are divided equally among
        /// their sequences; these are compute durations, not request latency.</summary>
        public long PrefillElapsedTicks { get; init; }

        /// <summary>Decode forward time in <see cref="System.Diagnostics.Stopwatch"/>
        /// ticks, including speculative drafting/verification/replay and replay
        /// of generated tokens after a cache restoration shortfall. Each
        /// step is counted once regardless of how many tokens it emits. Shared
        /// batched forwards are divided equally among their sequences.</summary>
        public long DecodeElapsedTicks { get; init; }
    }
}
