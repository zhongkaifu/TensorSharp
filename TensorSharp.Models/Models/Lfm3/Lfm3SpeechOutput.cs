// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// LFM3-Omni's speech output, loaded from its mmproj: the entry point a model wires into its decode loop.
    /// <para>
    /// Per audio slot of a response (an LM step whose output is a frame, not a token):
    /// <list type="number">
    /// <item><see cref="Lfm3FrameSampler.Sample"/> turns the LM's hidden row (after its final norm, from the previous
    /// position) into a frame of 12 codes through 3 depthformer passes;</item>
    /// <item>unless it is the end-of-audio frame of a plain TTS response, <see cref="FeedbackRow"/> makes the LM's next input
    /// row (the plain sum of the frame's 12 codebook embeddings), decoded in place of a token;</item>
    /// <item>a non-end frame goes to the request's <see cref="Lfm3CodecStream"/>, which returns its 80 ms of 24 kHz PCM
    /// (<see cref="Lfm3CodecStream.Flush()"/> at the end-of-audio frame returns the last 20 ms and resets it).</item>
    /// </list>
    /// <see cref="Lfm3SpeechTurn"/> is the slot schedule (text, audio, interleaved) and <see cref="Lfm3SpeechPrompt"/> the TTS
    /// prompt. The networks run as graph programs on the executor of the model's allocator (GGML metal/cuda/cpu/vulkan, the
    /// managed CPU, MLX, direct CUDA); a host ISTFT makes the waveform.
    /// </para>
    /// </summary>
    public sealed class Lfm3SpeechOutput : IDisposable
    {
        private readonly NetworkWeights _weights;
        private readonly Lfm3SpeechHostRows _feedback, _codeRows;
        private bool _disposed;

        internal Lfm3SpeechOutput(Lfm3SpeechConfig config, NetworkWeights weights, Lfm3SpeechHostRows feedback, Lfm3SpeechHostRows codeRows,
            IGraphExecutor executor = null)
        {
            Config = config;
            _weights = weights;
            _feedback = feedback;
            _codeRows = codeRows;
            if (feedback.RowWidth != config.HiddenSize || feedback.Rows != (long)Lfm3SpeechConfig.Codebooks * Lfm3SpeechConfig.CodebookSize)
                throw new ArgumentException($"the feedback table is [{feedback.RowWidth}, {feedback.Rows}], expected [{config.HiddenSize}, {Lfm3SpeechConfig.Codebooks * Lfm3SpeechConfig.CodebookSize}]");
            Depthformer = new Lfm3Depthformer(config, weights, codeRows, executor);
            Codec = new Lfm3CodecDecoder(config, weights, executor);
        }

        /// <summary>
        /// Load the speech output of <paramref name="mmproj"/> (an LFM3-Omni / LFM3-Audio mmproj) onto
        /// <paramref name="allocator"/> (the model's; it picks the graph executor) for <paramref name="backend"/>: the
        /// depthformer and codec weights (a.gen.*) as network weights, the LM feedback table (a.gen.code.out_embd) as host
        /// rows. Throws <see cref="System.IO.InvalidDataException"/> when the mmproj has none or its shapes differ.
        /// </summary>
        public static Lfm3SpeechOutput Load(GgufFile mmproj, IAllocator allocator, BackendType backend)
        {
            Lfm3SpeechConfig config = Lfm3SpeechConfig.FromGguf(mmproj);
            var weights = new NetworkWeights(allocator, backend);
            Lfm3SpeechHostRows feedback = null, codeRows = null;
            try
            {
                const string FeedbackTable = "a.gen.code.out_embd.weight";
                weights.LoadGguf(mmproj, n => n.StartsWith("a.gen.", StringComparison.Ordinal) && n != FeedbackTable);
                feedback = Lfm3SpeechHostRows.Load(mmproj, FeedbackTable);
                NetworkWeight table = weights.Get(Lfm3Depthformer.Table);
                codeRows = table.Quantized is { HasHostData: true } q
                    ? Lfm3SpeechHostRows.Borrow(q)
                    : Lfm3SpeechHostRows.Load(mmproj, Lfm3Depthformer.Table);
                return new Lfm3SpeechOutput(config, weights, feedback, codeRows);
            }
            catch
            {
                codeRows?.Dispose();
                feedback?.Dispose();
                weights.Dispose();
                throw;
            }
        }

        public Lfm3SpeechConfig Config { get; }

        public Lfm3Depthformer Depthformer { get; }

        public Lfm3CodecDecoder Codec { get; }

        /// <summary>Host bytes held for the feedback table and the code embeddings (the network weights' own aside).</summary>
        public long HostBytes => _feedback.OwnedBytes + _codeRows.OwnedBytes;

        /// <summary>The network weights (bytes held: <see cref="NetworkWeights.QuantizedBytes"/>, <see cref="NetworkWeights.DenseBytes"/>).</summary>
        public NetworkWeights Weights => _weights;

        /// <summary>A request's frame sampler (its own random stream from <paramref name="seed"/> and repetition history).</summary>
        public Lfm3FrameSampler CreateSampler(uint seed, Lfm3SpeechSampling sampling = null)
            => new(Depthformer, sampling ?? Lfm3SpeechSampling.Defaults(Config), seed);

        /// <summary>A request's codec stream (cold).</summary>
        public Lfm3CodecStream CreateCodecStream() => Codec.CreateStream();

        /// <summary>
        /// The LM input row of a frame: the plain sum of the 12 codebooks' embeddings (row code + codebook * 2017 of
        /// a.gen.code.out_embd), written to <paramref name="row"/> (<see cref="Lfm3SpeechConfig.HiddenSize"/> values). The
        /// end-of-audio frame's row (all 2016) is the input that continues a response in text after its speech.
        /// </summary>
        public void FeedbackRow(ReadOnlySpan<int> codes, Span<float> row)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            const int Cb = Lfm3SpeechConfig.Codebooks, V = Lfm3SpeechConfig.CodebookSize;
            if (codes.Length != Cb) throw new ArgumentException($"{Cb} codes", nameof(codes));
            if (row.Length < Config.HiddenSize) throw new ArgumentException($"{Config.HiddenSize} values", nameof(row));
            row = row[..Config.HiddenSize];
            row.Clear();
            for (int c = 0; c < Cb; c++)
            {
                if ((uint)codes[c] >= V) throw new ArgumentOutOfRangeException(nameof(codes), $"code {codes[c]} of codebook {c}");
                _feedback.AddTo((long)c * V + codes[c], row);
            }
        }

        /// <summary>The LM input row of a frame as a new array.</summary>
        public float[] FeedbackRow(ReadOnlySpan<int> codes)
        {
            var row = new float[Config.HiddenSize];
            FeedbackRow(codes, row);
            return row;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Depthformer.Dispose();
            _codeRows.Dispose();
            _feedback.Dispose();
            _weights.Dispose();
        }
    }
}
