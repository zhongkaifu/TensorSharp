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

        public string RequestId => Sequence.RequestId;
        public SequenceState Sequence { get; }
        public ChannelReader<int> Tokens => _tokens.Reader;
        public Task<InferenceCompletion> Completion => _completionTcs.Task;
        public DateTime SubmittedAt => Sequence.SubmittedAt;

        internal InferenceRequestHandle(SequenceState seq, InferenceEngine engine, CancellationToken ct)
        {
            Sequence = seq;
            _ctReg = ct.Register(() => engine.Abort(seq.RequestId));
        }

        internal void PublishToken(int tokenId)
        {
            // Channel is unbounded; should never fail in practice.
            _tokens.Writer.TryWrite(tokenId);
            Interlocked.Increment(ref _publishedTokens);
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
            };
            _completionTcs.TrySetResult(completion);
        }
    }

    public sealed class InferenceCompletion
    {
        public SequenceStatus Status { get; init; }
        public string FinishReason { get; init; }
        public int PromptTokenCount { get; init; }
        public int OutputTokenCount { get; init; }
        public int PrefixCacheReusedTokens { get; init; }
        public DateTime? FirstTokenAt { get; init; }
        public DateTime SubmittedAt { get; init; }
    }
}
