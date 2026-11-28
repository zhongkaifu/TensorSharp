// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's conditioner on the diffusion runtime: the MiniMax-H3 text encoder (Qwen3-VL-32B truncated to 50
// layers, no final norm; the native trunk MiniMax-H3 video uses too) behind IConditioner. The raw prompt is tokenized with
// no chat template and the layer-50 hidden state of every token is the conditioning, exactly as the legacy MiniMaxH3ImageModel (deleted in stage S10)
// encodes it. An edit's pictures are presented as diffusers' MiniMaxH3ImageEditTextEncoderStep presents them: for each,
// the label "Picture i: " (no angle brackets), <|vision_start|>, one <|image_pad|> a merged patch, <|vision_end|>; then
// the prompt verbatim; each label and the prompt tokenized on their own and cut at 4,096 tokens. The vision tower's merged
// patches replace the pads and its DeepStack taps join after layers 0-2 (MiniMaxH3TextEncoder, as H3 video's Ref2VA uses
// it). Label and prompt rows are tagged text (1), every row of a vision block image (0).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>
    /// The MiniMax-H3 text encoder as an <see cref="IConditioner"/>. Making one maps the encoder GGUF, reads its tokenizer
    /// files and its small tensors on the host; its first encode makes the trunk resident. <see cref="ReleaseDevice"/>
    /// hands the trunk's device copies back and keeps the mapping, so the next encode uploads from memory.
    /// <para>On direct CUDA (<see cref="BackendType.Cuda"/>) the trunk and the vision tower are MiniMax-H3's direct
    /// implementations on the device's allocator (MiniMaxH3DirectTextEncoder and MiniMaxH3DirectVisionEncoder, the ones
    /// MiniMax-H3 video's conditioner runs there): the encoder opens at the first encode and <see cref="ReleaseDevice"/>
    /// disposes it (its weights' device copies live in the direct linears, which the GGML residency calls do not reach), as
    /// the video conditioner releases its encoder; the next encode opens it again.</para>
    /// </summary>
    public sealed class MiniMaxH3ImageConditioner : IConditioner
    {
        private readonly string _path;
        private readonly IAllocator _allocator;   // direct CUDA's device; null on GGML
        private MiniMaxH3TextEncoder _encoder;     // on direct CUDA opened on first use since the last release
        private MiniMaxH3TextEncoderConfig _config;
        private readonly object _gate = new();
        private bool _disposed;

        /// <param name="path">The encoder GGUF (unsloth/MiniMax-H3-GGUF qwen3vl_32b_minimax_h3-*.gguf) with vocab.json,
        /// merges.txt and tokenizer_config.json beside it.</param>
        /// <param name="allocator">The device on <see cref="BackendType.Cuda"/> (a <see cref="TensorSharp.Cuda.CudaAllocator"/>).</param>
        public MiniMaxH3ImageConditioner(string path, BackendType backend, IAllocator allocator = null)
        {
            ArgumentNullException.ThrowIfNull(path);
            if (backend is not (BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu or BackendType.Cuda))
                throw new NotSupportedException($"MiniMax-H3-Image's text encoder runs on ggml_cuda, ggml_metal, ggml_cpu or cuda, not {backend}.");
            if (backend == BackendType.Cuda && allocator is not TensorSharp.Cuda.CudaAllocator)
                throw new ArgumentException("MiniMax-H3-Image's text encoder on --backend cuda runs on a CUDA device's allocator.", nameof(allocator));
            Backend = backend;
            _path = path;
            _allocator = backend == BackendType.Cuda ? allocator : null;
            if (_allocator == null)
            {
                _encoder = new MiniMaxH3TextEncoder(path, null, backend, null);
                _config = _encoder.Config;
            }
            else
            {
                using var gguf = new GgufFile(path);
                _config = MiniMaxH3TextEncoder.DetectConfig(gguf);
            }
        }

        public BackendType Backend { get; }

        /// <summary>Whether the trunk is the direct implementation on a tensor backend's device (direct CUDA).</summary>
        private bool Direct => _allocator != null;

        /// <summary>The encoder, opened on direct CUDA at its first use since the last release.</summary>
        private MiniMaxH3TextEncoder Encoder => _encoder ??= new MiniMaxH3TextEncoder(_path, null, Backend, _allocator);

        /// <summary>The hidden width of a conditioning row.</summary>
        public int Width => _config.Hidden;

        /// <summary>Tokens the last encode read (the timing line's "text tokens").</summary>
        public int LastTokens { get; private set; }

        /// <summary>Tokens of a prompt or a label at most (max_sequence_length).</summary>
        public const int MaxSequenceLength = 4096;

        /// <summary>The trunk's device bytes, and the vision tower's once an edit has used it. On direct CUDA before the encoder
        /// opens, the file's bytes (its weights are uploaded as stored: MiniMax-H3 video's conditioner's count).</summary>
        public long WeightBytes
        {
            get
            {
                lock (_gate)
                {
                    if (_encoder != null) return _encoder.DeviceWeightBytes + (_visionUsed ? _encoder.Vision.DeviceWeightBytes : 0);
                    try { return new System.IO.FileInfo(_path).Length; }
                    catch (Exception e) when (e is System.IO.IOException or UnauthorizedAccessException) { return 0; }
                }
            }
        }

        private bool _visionUsed;

        /// <summary>The trunk's compute memory for <paramref name="workload"/>'s prompt (<see cref="WorkingBytesFor"/>), with an
        /// edit's vision rows and the vision tower's own for its largest picture.</summary>
        public long WorkingBytes(DiffusionWorkload workload)
        {
            if (workload.Pictures is not { Count: > 0 } pictures) return WorkingBytesFor(_config, workload.Prompt);
            long rows = 0, largest = 0;
            foreach (PresentedImage picture in pictures)
            {
                long tokens = VisionTokens(picture.Image.Width, picture.Image.Height);
                rows += tokens + 2 + 6;
                largest = Math.Max(largest, tokens);
            }
            long seq = Math.Max(16, (workload.Prompt?.Length ?? 0) / 3 + 16) + rows;
            return WorkingBytesFor(_config, workload.Prompt, rows) + VisionWorkingBytesFor(largest) +
                   ScoreBytesFor(seq, _config.Heads);
        }

        /// <summary>The F32 scores one query block of an edit's attention keeps (at most this, whole sequences below it): the
        /// trunk materializes them a block at a time (<see cref="QueryChunkFor"/>).</summary>
        internal const long ScoreBudgetBytes = 1L << 30;

        /// <summary>The query rows of one block of an edit's materialized attention: 0 (the whole [seq, seq, heads] scores)
        /// when those fit <see cref="ScoreBudgetBytes"/>, else as many rows as a [seq, rows, heads] block fits, at least 32.</summary>
        internal static int QueryChunkFor(long seq, int heads)
        {
            long rowBytes = seq * heads * 4L;
            if (seq * rowBytes <= ScoreBudgetBytes) return 0;
            return (int)Math.Clamp(ScoreBudgetBytes / rowBytes, 32, seq);
        }

        /// <summary>The scores and their softmax for one query block of a <paramref name="seq"/>-row edit (both F32, alive
        /// together).</summary>
        internal static long ScoreBytesFor(long seq, int heads)
        {
            int chunk = QueryChunkFor(seq, heads);
            return 2L * seq * (chunk > 0 ? chunk : seq) * heads * 4L;
        }

        /// <summary>The vision tower's merged tokens for a vision copy of <paramref name="width"/> x <paramref name="height"/>.</summary>
        internal static long VisionTokens(int width, int height) =>
            (long)(height / MiniMaxH3VisionEncoder.Factor) * (width / MiniMaxH3VisionEncoder.Factor);

        /// <summary>The vision tower's activations for a picture of <paramref name="mergedTokens"/> merged tokens (four patches
        /// each): about 40 F32 rows of its width a patch (the qkv, the MLP and the residual stream), flash attention.</summary>
        internal static long VisionWorkingBytesFor(long mergedTokens) => 128L * 1024 * 1024 + mergedTokens * 4 * 4L * 40 * 1152;

        /// <summary>
        /// The trunk's compute memory for <paramref name="prompt"/>: a fixed part (the graph's buffers and the backend's
        /// handles beside the weights: 615 MiB measured on ggml_cuda for a 59-token prompt, much less on ggml_metal) and per
        /// token the hidden state's copies, the q/k/v rows and the MLP's gate and up rows in F32 (about three characters a
        /// token; prompts are a few hundred tokens at most).
        /// </summary>
        internal static long WorkingBytesFor(MiniMaxH3TextEncoderConfig c, string prompt, long pictureRows = 0)
        {
            ArgumentNullException.ThrowIfNull(c);
            long tokens = Math.Max(16, (prompt?.Length ?? 0) / 3 + 16) + pictureRows;
            long perToken = 4L * (6L * c.Hidden + 2L * c.Heads * c.HeadDim + 2L * c.KvHeads * c.HeadDim + 3L * c.Intermediate);
            return 640L * 1024 * 1024 + tokens * perToken;
        }

        public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken)
        {
            if (pictures is { Count: > 0 }) return EncodeWithPictures(prompt, pictures, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                MiniMaxH3TextEncoder encoder = Encoder;
                // Raw prompt tokens (no chat template); an empty prompt is the encoder's BOS alone.
                List<int> ids = encoder.Tokenize(prompt ?? string.Empty);
                if (ids.Count == 0) ids.Add(encoder.Tokenizer.BosTokenId);
                float[] hidden = encoder.Encode(ids);
                LastTokens = ids.Count;
                return new ConditioningSet(hidden, ids.Count, encoder.Config.Hidden);
            }
        }

        /// <summary>The presentation of an edit (see the file's header) through the trunk with the pictures' vision outputs.</summary>
        private ConditioningSet EncodeWithPictures(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(prompt))
                throw new ArgumentException("A MiniMax-H3-Image edit needs a prompt that says what to make of its pictures.");
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                MiniMaxH3TextEncoder encoder = Encoder;
                if (!encoder.HasVision)
                    throw new InvalidOperationException(
                        "Editing needs the Qwen3-VL vision tower, and this text-encoder checkpoint does not carry one. Point " +
                        "--video-text-encoder at the full qwen3vl_32b_minimax_h3 GGUF.");
                BpeTokenizer tokenizer = encoder.Tokenizer;
                MiniMaxH3Requests.RequireVisionTokens(tokenizer);
                var markers = (Single(tokenizer, "<|vision_start|>"), Single(tokenizer, "<|image_pad|>"), Single(tokenizer, "<|vision_end|>"));
                var outputs = new MiniMaxH3VisionOutput[pictures.Count];
                var vision = Stopwatch.StartNew();
                for (int i = 0; i < pictures.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    outputs[i] = encoder.Vision.Encode(pictures[i].Image);
                    _visionUsed = true;
                }
                vision.Stop();
                var counts = new int[outputs.Length];
                for (int i = 0; i < counts.Length; i++) counts[i] = outputs[i].TokenCount;
                var (ids, tags, starts) = Present(encoder.Tokenize, markers, counts, prompt);
                var images = new List<MiniMaxH3PromptImage>(outputs.Length);
                for (int i = 0; i < outputs.Length; i++)
                    images.Add(new MiniMaxH3PromptImage { TokenIndex = starts[i], Vision = outputs[i] });
                cancellationToken.ThrowIfCancellationRequested();
                // The vision rows make the sequence thousands of tokens long, where the trunk's whole scores
                // ([seq, seq, heads] F32) would not fit: it materializes them a block of query rows at a time. (The direct
                // trunk's fused causal attention streams its keys, as MiniMax-H3 video's Ref2VA prompts run it.)
                int queryChunk = QueryChunkForPictures ?? QueryChunkFor(ids.Count, encoder.Config.Heads);
                float[] hidden = encoder.Encode(ids, images, flashAttention: FlashAttentionForPictures, queryChunk: queryChunk);
                LastTokens = ids.Count;
                LastPresentation = (ids.ToArray(), tags.ToArray());
                return new ConditioningSet(hidden, ids.Count, encoder.Config.Hidden, tags.ToArray())
                {
                    VisionSeconds = vision.Elapsed.TotalSeconds,
                };
            }

        }

        /// <summary>
        /// diffusers' MiniMaxH3ImageEditTextEncoderStep presentation as token ids: for picture i, <c>tokenize("Picture i: ")</c>
        /// (cut at <see cref="MaxSequenceLength"/>), the vision start marker, <paramref name="visionTokens"/>[i] pads and the
        /// end marker; then <c>tokenize(prompt)</c> (cut alike). The tags (1 text, 0 image: every row of a vision block,
        /// markers included) and where each picture's pads start.
        /// </summary>
        internal static (List<int> Ids, List<int> Tags, int[] PadStarts) Present(Func<string, List<int>> tokenize,
            (int Start, int Pad, int End) markers, IReadOnlyList<int> visionTokens, string prompt)
        {
            ArgumentNullException.ThrowIfNull(tokenize);
            ArgumentNullException.ThrowIfNull(visionTokens);
            var ids = new List<int>();
            var tags = new List<int>();
            var starts = new int[visionTokens.Count];
            for (int i = 0; i < visionTokens.Count; i++)
            {
                List<int> label = Truncate(tokenize($"Picture {i + 1}: "));
                ids.AddRange(label);
                for (int t = 0; t < label.Count; t++) tags.Add(MiniMaxH3Modality.Text);
                ids.Add(markers.Start);
                starts[i] = ids.Count;
                for (int t = 0; t < visionTokens[i]; t++) ids.Add(markers.Pad);
                ids.Add(markers.End);
                for (int t = 0; t < visionTokens[i] + 2; t++) tags.Add(MiniMaxH3Modality.Visual);
            }
            List<int> text = Truncate(tokenize(prompt ?? string.Empty));
            ids.AddRange(text);
            for (int t = 0; t < text.Count; t++) tags.Add(MiniMaxH3Modality.Text);
            return (ids, tags, starts);

            static List<int> Truncate(List<int> ids) => ids.Count > MaxSequenceLength ? ids.GetRange(0, MaxSequenceLength) : ids;
        }

        private static int Single(BpeTokenizer tokenizer, string marker)
        {
            List<int> ids = tokenizer.Encode(marker, addSpecial: false);
            if (ids.Count != 1) throw new InvalidOperationException($"The MiniMax-H3 tokenizer has no {marker} token.");
            return ids[0];
        }

        /// <summary>Whether an edit's sequence runs the trunk's flash attention instead (a measurement seam; ggml-cuda's
        /// flash kernel sums V in half precision: rel. L2 0.09 from the whole F32 scores at 1,076 rows on an A40).</summary>
        internal bool FlashAttentionForPictures { get; set; }

        /// <summary>The query block an edit's attention uses instead of <see cref="QueryChunkFor"/> (a measurement seam; 0 is
        /// the whole scores).</summary>
        internal int? QueryChunkForPictures { get; set; }

        /// <summary>The token ids and tags of the last edit's presentation (a parity seam).</summary>
        internal (int[] Ids, int[] Tags) LastPresentation { get; private set; }

        /// <summary>On direct CUDA the encoder is released after every encode, as MiniMax-H3 video's conditioner releases it
        /// there: that backend has no device budget for the residency plan to weigh (GpuMemoryBudget covers the GGML GPU
        /// backends), and the 17 GB trunk must not sit beside the denoiser. On GGML the plan decides.</summary>
        public bool ReleaseAfterPhase => Direct;

        /// <summary>The native trunk's compute buffers (MiniMax-H3's shared reuse allocator, which only this component uses
        /// on the runtime).</summary>
        public void ReleaseWorkspace()
        {
            lock (_gate)
                if (!_disposed && !Direct) ReleaseSharedScratch();
        }

        /// <summary>On GGML the trunk's (and the vision tower's) device copies go and the mapping stays; on direct CUDA the
        /// encoder is disposed (its direct linears own their copies) and the next encode opens it again.</summary>
        public void ReleaseDevice()
        {
            lock (_gate)
            {
                if (_disposed) return;
                if (Direct)
                {
                    _encoder?.Dispose();
                    _encoder = null;
                    _visionUsed = false;
                    return;
                }
                _encoder.ReleaseDeviceResidency();
                if (_visionUsed) _encoder.Vision.ReleaseDeviceResidency();
                ReleaseSharedScratch();
            }
        }

        // The reuse allocator is every GGML model's: it goes while no live model of the process runs a step.
        private static void ReleaseSharedScratch() => ModelBase.RunWhileGgmlModelsAreIdle(null, GgmlBasicOps.ReleaseReuseComputeBuffers);

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _encoder?.Dispose();
                _encoder = null;
            }
        }
    }
}
