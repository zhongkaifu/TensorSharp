// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;

using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// DiffusionGemma (architecture key <c>diffusion-gemma</c>): a block text-diffusion model on the Gemma 4 26B-A4B backbone,
    /// on the shared decoder runtime (<see cref="Gemma4DecoderModel"/>'s recipe, its caches and graph executors).
    /// <list type="bullet">
    /// <item>The prompt is prefilled into the runtime's cache as a causal prompt (sliding-window layers clip it, an image's soft
    /// tokens attend each other on those layers only) with each layer's output times its <c>enc_layer_output_scale</c>; a
    /// block's next prompt reuses the cached rows it extends.</item>
    /// <item>Each denoising step runs the canvas (<see cref="CanvasLength"/> rows after the prompt) as block rows
    /// (<see cref="StandardDecoderModel.RunBlockRows"/>): every canvas row reads every other on every layer and the prompt as a
    /// chunk at the canvas's first position does (a sliding-window layer from that row's window), each layer's output times
    /// its <c>layer_output_scale</c>; nothing is committed.</item>
    /// <item>The canvas input rows are <c>rms_norm(embed * sqrt(n_embd) [+ self-conditioning])</c> without a weight; the
    /// self-conditioning signal is a gated-GELU MLP (<c>self_cond_*</c>) over the previous step's top-32 soft embedding
    /// (<c>sum_k p_k embed[t_k] * sqrt(n_embd)</c>).</item>
    /// <item>The head (tied <c>token_embd</c>, softcapped) returns the canvas logits, or, where the executor samples on the device
    /// (<see cref="SupportsDeviceSampling"/>), each position's argmax, entropy, multinomial draw and top-32 from one graph, so
    /// only those cross to the host.</item>
    /// </list>
    /// Generation (the EntropyBound denoising loop) lives in <see cref="DiffusionGemmaSampler"/>.
    /// </summary>
    public sealed partial class DiffusionGemmaModel : Gemma4DecoderModel
    {
        /// <summary>Tokens of the previous step's distribution a position's self-conditioning reads: a tempInv &gt;= 1.25 softmax
        /// keeps more than 0.9999 of its mass in its top 32.</summary>
        private const int ScTopK = 32;

        private readonly int _canvasLength;
        private readonly int _maskTokenId;
        private readonly bool _hasSelfConditioning;
        private float[] _encScale;
        private bool _scEnabled;
        private bool _promptKv = true;
        private bool _prefilling;
        private float[] _iota;
        private float[] _logitsCopy;
        private readonly PromptCache _ownCache = new();
        private int _nextSequence;

        private readonly Stopwatch _swForward = new();
        private long _tPrefill, _tCanvas, _tSc, _tRead, _canvasSteps;

        /// <summary>What a cache holds: the prompt tokens it was prefilled with, the image spans' version then, and whether any
        /// forward wrote it (a fresh cache needs no reset).</summary>
        internal sealed class PromptCache
        {
            public int[] Tokens;
            public int VisionVersion;
            public bool Written;
        }

        public DiffusionGemmaModel(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            _canvasLength = (int)_gguf.GetUint32("diffusion.canvas_length", 256);
            _maskTokenId = (int)_gguf.GetUint32("tokenizer.ggml.mask_token_id", 0);
            _hasSelfConditioning = HasWeight("self_cond_pre_norm.weight") && HasWeight("self_cond_gate.weight")
                && HasWeight("self_cond_up.weight") && HasWeight("self_cond_down.weight");
            _scEnabled = _hasSelfConditioning;
            Console.WriteLine($"DiffusionGemma ready: canvas_length={_canvasLength}, mask_token={_maskTokenId}, " +
                $"self_conditioning={_scEnabled}, device_sampling={SupportsDeviceSampling}");
        }

        public int CanvasLength => _canvasLength;
        public int MaskTokenId => _maskTokenId;
        public int VocabSize => Config.VocabSize;

        /// <summary>Whether each step after the first adds the previous step's self-conditioning signal (on when the checkpoint
        /// has the <c>self_cond_*</c> weights).</summary>
        public bool SelfConditioningEnabled
        {
            get => _scEnabled;
            set => _scEnabled = value && _hasSelfConditioning;
        }

        /// <summary>Whether the sampler prefills a block's prompt once (<see cref="PrefillPrompt"/>) and then runs only the canvas
        /// each step (<see cref="DecodeCanvas"/>). The unified <see cref="ForwardCanvas"/> computes the same rows (the prompt is
        /// causal and never reads the canvas); tests turn this off to run it.</summary>
        public bool SupportsPromptKvCache
        {
            get => _promptKv;
            set => _promptKv = value;
        }

        /// <summary>Whether a step can sample on the device (<see cref="DecodeCanvasSampled"/>): the GGML CUDA executor, whose
        /// graph keeps the [vocab, canvas] logits on the device and returns only each position's picks. Elsewhere the logits
        /// come back and the sampler draws on the host, as it always has there.</summary>
        public bool SupportsDeviceSampling => _backend == BackendType.GgmlCuda || DeviceSamplingForTests;

        /// <summary>Tests: sample in the graph on any executor (the managed one holds the sampler's ops to the host's).</summary>
        internal bool DeviceSamplingForTests { get; set; }

        /// <summary>Canvas steps that ran several sequences in one graph (tests).</summary>
        internal int BatchedCanvasStepsForTests { get; private set; }

        /// <summary>The number of top tokens a device-sampled step returns for the next step's self-conditioning (0 without it).</summary>
        public int SelfCondTopK => _scEnabled ? Math.Min(ScTopK, Config.VocabSize) : 0;

        // ---------------------------------------------------------------- the runtime's declarations

        private protected override string FamilyNote => ", block diffusion";

        /// <summary>An image's soft tokens attend each other on the sliding-window layers only, each image apart (vLLM's
        /// gemma4_mm); the global layers stay causal over them.</summary>
        protected override bool MediaOpensFullAttentionLayers => false;
        protected override bool MediaSpansAttendApart => true;

        /// <summary>A prompt's rows take the encoder scalar, the canvas's (block rows) the decoder one.</summary>
        private protected override float LayerOutputScale(GraphStep step, int layer)
            => step.BlockRows ? _layerScale[layer] : _encScale?[layer] ?? 1f;

        protected override string FingerprintExtras => base.FingerprintExtras + "|diffusion";

        protected override void PrepareLoadedWeights()
        {
            base.PrepareLoadedWeights();
            _encScale = new float[Config.NumLayers];
            for (int l = 0; l < _encScale.Length; l++)
                _encScale[l] = _weights.TryGetValue(LayerName(l, "enc_layer_output_scale.weight"), out Tensor s) ? s.GetElementAsFloat(0) : 1f;
        }

        /// <summary>Autoregressive decoding is refused: the model denoises canvases (<see cref="DiffusionGemmaSampler"/>). Its
        /// prompt prefills run through the runtime's forward.</summary>
        protected override float[] ForwardCore(int[] tokens)
        {
            if (!_prefilling)
                throw new NotSupportedException("DiffusionGemma is a diffusion model and does not support autoregressive Forward(). " +
                    "Use DiffusionGemmaSampler.Generate() / the diffusion run mode.");
            return base.ForwardCore(tokens);
        }

        protected override void ResetKVCacheCore()
        {
            base.ResetKVCacheCore();
            // The active cache is the model's own unless a sequence's is bound (its PromptCache is reset by its caller).
            if (!SequenceBound)
            {
                _ownCache.Tokens = null;
                _ownCache.Written = false;
            }
        }

        // ---------------------------------------------------------------- prompts

        /// <summary>Prefill <paramref name="promptTokens"/> into the model's own cache (the single-request paths), reusing the
        /// rows a previous prompt it extends left there.</summary>
        public void PrefillPrompt(int[] promptTokens) => PrefillActive(promptTokens, ActiveVision, _ownCache, CancellationToken.None);

        /// <summary>Prefill the active cache with <paramref name="prompt"/>: the rows past what <paramref name="cache"/> already holds
        /// of it (all of them when it holds another prompt or other images), with the image spans of <paramref name="vision"/> in
        /// those rows written over their placeholders.</summary>
        private void PrefillActive(int[] prompt, VisionSet vision, PromptCache cache, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(prompt);
            if (prompt.Length == 0) throw new ArgumentException("A prompt needs at least one token.", nameof(prompt));
            // The canvas after it must fit too: its step's cache growth refuses past the context window.
            if (MaxContextLength > 0 && prompt.Length >= MaxContextLength)
                throw new ArgumentException($"The {prompt.Length}-token prompt leaves no room for a canvas in the {MaxContextLength}-token context window.", nameof(prompt));
            cancellationToken.ThrowIfCancellationRequested();
            long t0 = Stopwatch.GetTimestamp();
            _swForward.Start();
            try
            {
                int version = vision?.Version ?? 0;
                int keep = 0;
                if (cache.Tokens != null && cache.VisionVersion == version && cache.Tokens.Length <= prompt.Length
                    && prompt.AsSpan(0, cache.Tokens.Length).SequenceEqual(cache.Tokens))
                    keep = cache.Tokens.Length;
                if (keep == 0 && cache.Written) ResetKVCache();
                cache.Tokens = null;
                cache.Written = true;
                if (keep < prompt.Length)
                {
                    if (vision != null)
                        foreach ((Tensor rows, int position) in vision.Spans)
                        {
                            int count = (int)rows.Sizes[0];
                            if (position + count > prompt.Length)
                                throw new InvalidOperationException($"An image span [{position}, {position + count}) runs past the {prompt.Length}-token prompt.");
                            if (position >= keep) QueueMedia(Ops.NewContiguous(rows), position - keep);
                            else if (position + count > keep)
                                throw new InvalidOperationException($"An image span [{position}, {position + count}) straddles the cached rows' end {keep}.");
                        }
                    _prefilling = true;
                    try
                    {
                        ForwardRefill(prompt.AsSpan(keep).ToArray());
                    }
                    finally
                    {
                        _prefilling = false;
                    }
                }
                cache.Tokens = (int[])prompt.Clone();
                cache.VisionVersion = version;
            }
            finally
            {
                _swForward.Stop();
                _tPrefill += Stopwatch.GetTimestamp() - t0;
            }
        }

        // ---------------------------------------------------------------- canvas steps

        /// <summary>The canvas logits [C, vocab] (softcapped) of one step after the prefilled prompt. Self-conditioning reads
        /// <paramref name="scPrevLogits"/> (the previous step's logits) when <paramref name="scUse"/> is not 0. The returned
        /// buffer is reused by the next step, after that step has read its self-conditioning from it.</summary>
        public float[] DecodeCanvas(int[] canvasTokens, float[] scPrevLogits, float scUse, float prevTempInv)
            => CanvasLogits(null, canvasTokens, SoftEmbeddingFromLogits(scPrevLogits, scUse, prevTempInv, canvasTokens.Length));

        /// <summary>Prompt and canvas in one call (<paramref name="tokens"/> = [prompt | canvas], split at
        /// <paramref name="promptLen"/>): the prompt prefills the model's cache (reusing what it holds), then the canvas runs.</summary>
        public float[] ForwardCanvas(int[] tokens, int promptLen, float[] scPrevLogits = null, float scUse = 0f, float prevTempInv = 1f)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (promptLen <= 0 || promptLen >= tokens.Length) throw new ArgumentOutOfRangeException(nameof(promptLen));
            PrefillPrompt(tokens.AsSpan(0, promptLen).ToArray());
            return DecodeCanvas(tokens.AsSpan(promptLen).ToArray(), scPrevLogits, scUse, prevTempInv);
        }

        /// <summary>One step sampled on the device (<see cref="SupportsDeviceSampling"/>): per canvas position the argmax, the
        /// entropy of softmax(logits * <paramref name="tempInv"/>), the multinomial draw at <paramref name="u"/> (the first
        /// token whose cumulative probability reaches it) and, with <paramref name="K"/> &gt; 0, the top-K tokens with their
        /// softmax over the K at <paramref name="tempInv"/> (the next step's self-conditioning, read from
        /// <paramref name="scPrevTopTokens"/> / <paramref name="scPrevTopProbs"/>). False when the device cannot sample.</summary>
        public bool DecodeCanvasSampled(int[] canvasTokens, int[] scPrevTopTokens, float[] scPrevTopProbs, float scUse, float tempInv,
            float[] u, int K, int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut, float[] topProbsOut)
            => CanvasSampled(null, canvasTokens, scPrevTopTokens, scPrevTopProbs, scUse, tempInv, u, K, argmaxOut, entropyOut, sampledOut,
                topTokensOut, topProbsOut);

        /// <summary>The canvas's logits, on <paramref name="sequence"/>'s cache (null: the model's own).</summary>
        private float[] CanvasLogits(string sequence, int[] canvas, Tensor soft)
        {
            CachedGraph graph = null;
            bool owned = false;
            Tensor embedded = null;
            long t0 = Stopwatch.GetTimestamp();
            _swForward.Start();
            bool bound = Bind(sequence);
            try
            {
                int rows = canvas.Length, vocab = Config.VocabSize;
                embedded = Embedding(canvas);
                graph = RunBlockRows(null, new[] { rows }, step => EmitCanvas(step, embedded, soft, CanvasHead.Logits, null),
                    out StepOutputs outputs, out owned, "diffusion-canvas");
                long tr = Stopwatch.GetTimestamp();
                float[] logits = graph.Outputs[outputs.Logits];
                if (owned)
                {
                    // The graph goes with this call: its logits are copied into the buffer the next step reuses.
                    long total = (long)rows * vocab;
                    if (_logitsCopy == null || _logitsCopy.LongLength != total) _logitsCopy = new float[total];
                    Array.Copy(logits, _logitsCopy, total);
                    logits = _logitsCopy;
                }
                _tRead += Stopwatch.GetTimestamp() - tr;
                _canvasSteps++;
                return logits;
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded?.Dispose();
                soft?.Dispose();
                if (bound) RestorePrimaryCache();
                _swForward.Stop();
                _tCanvas += Stopwatch.GetTimestamp() - t0;
            }
        }

        private bool CanvasSampled(string sequence, int[] canvas, int[] scPrevTopTokens, float[] scPrevTopProbs, float scUse, float tempInv,
            float[] u, int K, int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut, float[] topProbsOut)
        {
            if (!SupportsDeviceSampling) return false;
            int rows = canvas.Length;
            bool wantTopK = K > 0 && topTokensOut != null && topProbsOut != null;
            int k = wantTopK ? Math.Min(K, Config.VocabSize) : 0;
            Tensor soft = _scEnabled && scUse != 0f && scPrevTopTokens != null && scPrevTopProbs != null
                ? SoftEmbedding(scPrevTopTokens, scPrevTopProbs, rows, Math.Min(K, Config.VocabSize)) : null;
            var sampling = new SamplingInputs(rows, k, tempInv, u);
            CachedGraph graph = null;
            bool owned = false;
            Tensor embedded = null;
            long t0 = Stopwatch.GetTimestamp();
            _swForward.Start();
            bool bound = Bind(sequence);
            try
            {
                embedded = Embedding(canvas);
                graph = RunBlockRows(null, new[] { rows }, step => EmitCanvas(step, embedded, soft, CanvasHead.Sample, sampling),
                    out StepOutputs outputs, out owned, "diffusion-canvas-sample");
                long tr = Stopwatch.GetTimestamp();
                ReadSampling(graph, outputs, rows, k, argmaxOut, entropyOut, sampledOut, topTokensOut, topProbsOut);
                _tRead += Stopwatch.GetTimestamp() - tr;
                _canvasSteps++;
                return true;
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded?.Dispose();
                soft?.Dispose();
                if (bound) RestorePrimaryCache();
                _swForward.Stop();
                _tCanvas += Stopwatch.GetTimestamp() - t0;
            }
        }

        /// <summary>Bind <paramref name="sequence"/>'s cache (true: the caller restores the primary after), or keep the active one.</summary>
        private bool Bind(string sequence)
        {
            if (sequence == null) return false;
            BindSequenceCache(sequence);
            return true;
        }

        private void ReadSampling(CachedGraph graph, in StepOutputs outputs, int rows, int k, int[] argmaxOut, float[] entropyOut,
            int[] sampledOut, int[] topTokensOut, float[] topProbsOut, int offset = 0)
        {
            int vocab = Config.VocabSize;
            float[] argmax = graph.Outputs[outputs.Draft[0]], entropy = graph.Outputs[outputs.Draft[1]], below = graph.Outputs[outputs.Draft[2]];
            for (int i = 0; i < rows; i++)
            {
                argmaxOut[i] = (int)argmax[offset + i];
                entropyOut[i] = entropy[offset + i];
                // The draw is the first token whose cumulative probability reaches u: the count of tokens below it.
                sampledOut[i] = Math.Min((int)below[offset + i], vocab - 1);
            }
            if (k == 0) return;
            float[] ids = graph.Outputs[outputs.Draft[3]], probs = graph.Outputs[outputs.Draft[4]];
            for (int i = 0; i < rows * k; i++)
            {
                topTokensOut[i] = (int)ids[offset * k + i];
                topProbsOut[i] = probs[offset * k + i];
            }
        }

        private enum CanvasHead { Logits, Sample, Rows }

        /// <summary>A device-sampled step's per-run inputs: the temperature (a column, so a replay reads the step's own) and the
        /// uniform draws.</summary>
        private sealed class SamplingInputs
        {
            public SamplingInputs(int rows, int k, float tempInv, float[] u)
            {
                K = k;
                TempInv = new float[rows];
                Array.Fill(TempInv, tempInv);
                U = new float[rows];
                Array.Copy(u, U, rows);
            }

            public int K { get; }
            public float[] TempInv { get; }
            public float[] U { get; }
        }

        /// <summary>
        /// A canvas step's graph: the canvas rows <c>rms_norm(embed * sqrt(n_embd) + selfcond(soft))</c> (no weight), every layer
        /// as block rows, <c>output_norm</c> and the softcapped head; then the logits, the device sampler's picks, or (structured
        /// reads) the logits of the rows at <paramref name="keepRows"/> only.
        /// </summary>
        private StepOutputs EmitCanvas(GraphStep step, Tensor embedded, Tensor soft, CanvasHead head, SamplingInputs sampling,
            int[] keepRows = null)
        {
            int rows = step.Rows, hidden = Config.HiddenSize, vocab = Config.VocabSize;
            var outputs = new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
            int x = step.Op(GraphOp.Scale, stackalloc[] { step.HiddenInput(embedded, hidden, rows) }, default, stackalloc[] { MathF.Sqrt(hidden) });
            if (soft != null) x = step.Op(GraphOp.Add, x, EmitSelfConditioning(step, step.Input(soft, hidden, rows)));
            x = step.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { Config.Eps });
            int h = EmitDecoderLayers(step, x);
            int normed = EmitStreamOutput(step, h);
            if (head == CanvasHead.Rows)
            {
                // The kept rows through the same head (their arithmetic is the full canvas's, row by row).
                int picked = keepRows.Length == rows ? normed : step.Op(GraphOp.GetRows, normed, step.ConstantInputI32(keepRows, keepRows.Length));
                outputs.Logits = step.Output(EmitHeadLogits(step, picked, keepRows.Length), (long)vocab * keepRows.Length);
                return outputs;
            }
            int logits = EmitHeadLogits(step, normed, rows);                                                    // [vocab, rows]
            if (head == CanvasHead.Logits)
            {
                outputs.Logits = step.Output(logits, (long)vocab * rows);
                return outputs;
            }
            outputs.Draft = EmitSampling(step, logits, rows, sampling);
            return outputs;
        }

        /// <summary>The self-conditioning signal [hidden, rows] from the soft embedding rows: down(gelu(gate(n)) * up(n)) with
        /// n = rms_norm(soft) by <c>self_cond_pre_norm</c>.</summary>
        private static int EmitSelfConditioning(GraphStep step, int soft)
        {
            int n = step.RmsNorm(soft, "self_cond_pre_norm.weight");
            int act = step.Op(GraphOp.GegluSplit, step.MatMul("self_cond_gate.weight", n), step.MatMul("self_cond_up.weight", n));
            return step.MatMul("self_cond_down.weight", act);
        }

        /// <summary>
        /// The device sampler over the softcapped logits [vocab, rows] (DiffusionGemmaSampler.DenoiseStep's host pass, as graph
        /// ops): s = logits * tempInv; p = softmax(s); argmax a; entropy H = lse(s) - sum p s with lse = s_a - log p_a; the draw
        /// counts the tokens whose cumulative probability stays below u; the top K tokens of s and their softmax over the K. The
        /// ids come back as values of an iota table (outputs are F32). Returns the outputs: ids, entropies, counts[, top ids, top
        /// probabilities].
        /// </summary>
        private int[] EmitSampling(GraphStep step, int logits, int rows, SamplingInputs sampling)
        {
            int vocab = Config.VocabSize, k = sampling.K;
            if (_iota == null || _iota.Length != vocab)
            {
                _iota = new float[vocab];
                for (int i = 0; i < vocab; i++) _iota[i] = i;
            }
            int iota = step.ConstantInputF32(_iota, 1, vocab);                                                 // [1, vocab]
            int s = step.Op(GraphOp.Mul, logits, step.InputF32(sampling.TempInv, 1, rows));
            int p = step.Op(GraphOp.SoftMaxExt, s);
            // [max(1, K), rows], largest first (a view of the row's sort: copied before it is sliced).
            int top = step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { s }, stackalloc long[] { Math.Max(1, k) }));
            int first = step.Op(GraphOp.Cont, step.View2(top, 1, rows, (long)Math.Max(1, k) * sizeof(int), 0));   // [1, rows]
            int argmax = step.Reshape(step.Op(GraphOp.GetRows, iota, step.Reshape(first, rows)), rows);
            int s3 = step.Reshape(s, 1, vocab, rows), p3 = step.Reshape(p, 1, vocab, rows);
            int sa = step.Reshape(step.Op(GraphOp.GetRows, s3, first), 1, rows);
            int pa = step.Reshape(step.Op(GraphOp.GetRows, p3, first), 1, rows);
            int sumPs = step.Op(GraphOp.SumRows, step.Op(GraphOp.Mul, p, s));                                    // [1, rows]
            int entropy = step.Op(GraphOp.Sub, step.Op(GraphOp.Sub, sa, step.Op(GraphOp.Log, pa)), sumPs);
            // u - cdf > 0 marks a token below the draw; times 1e30 and clamped to [0, 1] it is that indicator.
            int cdf = step.Op(GraphOp.CumSum, p);
            int over = step.Op(GraphOp.Sub, cdf, step.InputF32(sampling.U, 1, rows));                          // cdf - u
            int below = step.Op(GraphOp.Clamp, stackalloc[] { step.Op(GraphOp.Scale, stackalloc[] { over }, default, stackalloc[] { -1e30f }) },
                default, stackalloc[] { 0f, 1f });
            int count = step.Op(GraphOp.SumRows, below);
            var outputs = new List<int>
            {
                step.Output(step.Op(GraphOp.Cont, argmax), rows),
                step.Output(step.Op(GraphOp.Cont, entropy), rows),
                step.Output(step.Op(GraphOp.Cont, count), rows),
            };
            if (k > 0)
            {
                int topIds = step.Op(GraphOp.GetRows, iota, step.Reshape(top, (long)k * rows));                    // [1, K rows]
                int topS = step.Reshape(step.Op(GraphOp.GetRows, s3, top), k, rows);                               // [K, rows]
                outputs.Add(step.Output(step.Op(GraphOp.Cont, topIds), (long)k * rows));
                outputs.Add(step.Output(step.Op(GraphOp.SoftMaxExt, topS), (long)k * rows));
            }
            return outputs.ToArray();
        }

        // ---------------------------------------------------------------- self-conditioning inputs

        /// <summary>The soft embedding rows [rows, hidden] (or null) from the previous step's logits: each position's top-K tokens
        /// and their softmax at <paramref name="prevTempInv"/> over the K.</summary>
        private Tensor SoftEmbeddingFromLogits(float[] prevLogits, float scUse, float prevTempInv, int rows)
        {
            if (!_scEnabled || prevLogits == null || scUse == 0f) return null;
            long t0 = Stopwatch.GetTimestamp();
            int vocab = Config.VocabSize, k = Math.Min(ScTopK, vocab);
            if (prevLogits.LongLength < (long)rows * vocab)
                throw new ArgumentException($"Self-conditioning reads {rows} rows of {vocab} logits.", nameof(prevLogits));
            var tokens = new int[rows * k];
            var probs = new float[rows * k];
            Parallel.For(0, rows, c =>
            {
                TopK(prevLogits, (long)c * vocab, vocab, k, tokens, probs, c * k);
                float max = float.NegativeInfinity;
                for (int j = 0; j < k; j++) max = Math.Max(max, probs[c * k + j] *= prevTempInv);
                float sum = 0f;
                for (int j = 0; j < k; j++) sum += probs[c * k + j] = MathF.Exp(probs[c * k + j] - max);
                float inv = sum > 0f ? 1f / sum : 0f;
                for (int j = 0; j < k; j++) probs[c * k + j] *= inv;
            });
            _tSc += Stopwatch.GetTimestamp() - t0;
            return SoftEmbedding(tokens, probs, rows, k);
        }

        /// <summary>sum_k p_k embed[t_k] * sqrt(n_embd) per position [rows, hidden], from each position's <paramref name="k"/>
        /// tokens and weights, on the host (the rows are a handful per position of the dequantized table).</summary>
        private unsafe Tensor SoftEmbedding(int[] tokens, float[] probs, int rows, int k)
        {
            long t0 = Stopwatch.GetTimestamp();
            int hidden = Config.HiddenSize;
            float scale = MathF.Sqrt(hidden);
            var soft = new float[(long)rows * hidden];
            if (_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight table) && table.HasHostData)
            {
                long rowBytes = NativeDequant.RowSize(table.GgmlType, table.Ne0);
                byte* basePtr = (byte*)table.Data.ToPointer();
                float tableScale = table.Scale;
                Parallel.For(0, rows, () => new float[hidden], (c, _, row) =>
                {
                    fixed (float* r = row)
                    fixed (float* dst = &soft[(long)c * hidden])
                        for (int j = 0; j < k; j++)
                        {
                            int t = tokens[c * k + j];
                            if ((uint)t >= (uint)Config.VocabSize) continue;
                            ManagedQuantizedOps.DequantizeRowToFloat32(table.GgmlType, (IntPtr)(basePtr + t * rowBytes), r, hidden);
                            float w = probs[c * k + j] * scale * tableScale;
                            for (int i = 0; i < hidden; i++) dst[i] += w * r[i];
                        }
                    return row;
                }, _ => { });
            }
            else
            {
                using Tensor embedded = Embedding(tokens);
                float[] all = embedded.GetElementsAsFloat(rows * k * hidden);
                Parallel.For(0, rows, c =>
                {
                    for (int j = 0; j < k; j++)
                    {
                        float w = probs[c * k + j] * scale;
                        long src = ((long)c * k + j) * hidden, dst = (long)c * hidden;
                        for (int i = 0; i < hidden; i++) soft[dst + i] += w * all[src + i];
                    }
                });
            }
            _tSc += Stopwatch.GetTimestamp() - t0;
            return CreateFloatTensor(soft, rows, hidden);
        }

        /// <summary>The top <paramref name="k"/> values of logits[baseOff, baseOff + n) through a min-heap: their tokens into
        /// <paramref name="outTokens"/> and their raw values into <paramref name="outVals"/> from <paramref name="outBase"/>
        /// (unordered).</summary>
        private static void TopK(float[] logits, long baseOff, int n, int k, int[] outTokens, float[] outVals, int outBase)
        {
            int filled = 0;
            for (int v = 0; v < n; v++)
            {
                float val = logits[baseOff + v];
                if (filled < k)
                {
                    outVals[outBase + filled] = val;
                    outTokens[outBase + filled] = v;
                    if (++filled == k)
                        for (int i = k / 2 - 1; i >= 0; i--) SiftDown(outVals, outTokens, outBase, k, i);
                }
                else if (val > outVals[outBase])
                {
                    outVals[outBase] = val;
                    outTokens[outBase] = v;
                    SiftDown(outVals, outTokens, outBase, k, 0);
                }
            }
        }

        private static void SiftDown(float[] vals, int[] toks, int b, int k, int i)
        {
            while (true)
            {
                int l = 2 * i + 1, r = l + 1, smallest = i;
                if (l < k && vals[b + l] < vals[b + smallest]) smallest = l;
                if (r < k && vals[b + r] < vals[b + smallest]) smallest = r;
                if (smallest == i) return;
                (vals[b + i], vals[b + smallest]) = (vals[b + smallest], vals[b + i]);
                (toks[b + i], toks[b + smallest]) = (toks[b + smallest], toks[b + i]);
                i = smallest;
            }
        }

        // ---------------------------------------------------------------- per-sequence states (the batched scheduler)

        /// <summary>A fresh per-sequence state: its own cache in the runtime (bound by its id). Free it with
        /// <see cref="DisposeSeqState"/>.</summary>
        public DiffusionSeqState CreateSeqState()
            => new DiffusionSeqState("diffusion-seq-" + Interlocked.Increment(ref _nextSequence).ToString(CultureInfo.InvariantCulture));

        /// <summary>Release a sequence's cache and its image spans.</summary>
        public void DisposeSeqState(DiffusionSeqState seq)
        {
            if (seq == null) return;
            if (seq.Prefilled) OnSequenceReleased(seq.Id);
            seq.Prefilled = false;
            seq.Cache.Tokens = null;
            seq.Cache.Written = false;
            seq.DisposeVision();
        }

        /// <summary>Prefill one sequence's prompt into its own cache, with its own image spans (reusing the rows of the block before).</summary>
        public void PrefillSeq(DiffusionSeqState seq, int[] promptTokens) => PrefillSeq(seq, promptTokens, CancellationToken.None);

        private void PrefillSeq(DiffusionSeqState seq, int[] promptTokens, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(seq);
            BindSequenceCache(seq.Id);
            seq.Prefilled = true;
            try
            {
                PrefillActive(promptTokens, seq.Vision, seq.Cache, cancellationToken);
            }
            finally
            {
                RestorePrimaryCache();
            }
        }

        /// <summary>One canvas step on a sequence's cache: its logits [C, vocab].</summary>
        public float[] DecodeCanvasSeq(DiffusionSeqState seq, int[] canvasTokens, float[] scPrevLogits, float scUse, float prevTempInv)
        {
            RequirePrefilled(seq);
            return CanvasLogits(seq.Id, canvasTokens, SoftEmbeddingFromLogits(scPrevLogits, scUse, prevTempInv, canvasTokens.Length));
        }

        /// <summary>One device-sampled canvas step on a sequence's cache (<see cref="DecodeCanvasSampled"/>).</summary>
        public bool DecodeCanvasSampledSeq(DiffusionSeqState seq, int[] canvasTokens, int[] scPrevTopTokens, float[] scPrevTopProbs,
            float scUse, float tempInv, float[] u, int K, int[] argmaxOut, float[] entropyOut, int[] sampledOut, int[] topTokensOut,
            float[] topProbsOut)
        {
            RequirePrefilled(seq);
            return CanvasSampled(seq.Id, canvasTokens, scPrevTopTokens, scPrevTopProbs, scUse, tempInv, u, K, argmaxOut, entropyOut,
                sampledOut, topTokensOut, topProbsOut);
        }

        /// <summary>Several sequences' canvases in one step (one graph: the weights read once for every canvas, each canvas over
        /// its own cache): each sequence's logits [C, vocab].</summary>
        public float[][] DecodeCanvasBatched(DiffusionSeqState[] seqs, int[][] canvases, float[][] scPrev, float[] scUse, float[] prevTempInv)
        {
            ArgumentNullException.ThrowIfNull(seqs);
            int n = seqs.Length, vocab = Config.VocabSize, hidden = Config.HiddenSize;
            if (canvases.Length != n) throw new ArgumentException("One canvas a sequence.", nameof(canvases));
            var ids = new string[n];
            var counts = new int[n];
            int total = 0;
            for (int b = 0; b < n; b++)
            {
                RequirePrefilled(seqs[b]);
                ids[b] = seqs[b].Id;
                counts[b] = canvases[b].Length;
                total += counts[b];
            }
            var tokens = new int[total];
            for (int b = 0, at = 0; b < n; at += counts[b], b++) canvases[b].CopyTo(tokens, at);
            // Self-conditioning for the sequences that read it; the others' rows are zero, which the MLP keeps zero (no biases,
            // an RMSNorm of zero is zero).
            Tensor soft = null;
            var signals = new Tensor[n];
            try
            {
                bool any = false;
                for (int b = 0; b < n; b++)
                {
                    signals[b] = SoftEmbeddingFromLogits(scPrev?[b], scUse?[b] ?? 0f, prevTempInv?[b] ?? 1f, counts[b]);
                    any |= signals[b] != null;
                }
                if (any)
                {
                    var all = new float[(long)total * hidden];
                    for (int b = 0, at = 0; b < n; at += counts[b], b++)
                        if (signals[b] != null) Array.Copy(signals[b].GetElementsAsFloat(counts[b] * hidden), 0, all, (long)at * hidden, (long)counts[b] * hidden);
                    soft = CreateFloatTensor(all, total, hidden);
                }
            }
            finally
            {
                foreach (Tensor t in signals) t?.Dispose();
            }

            CachedGraph graph = null;
            bool owned = false;
            Tensor embedded = null;
            long t0 = Stopwatch.GetTimestamp();
            _swForward.Start();
            try
            {
                embedded = Embedding(tokens);
                graph = RunBlockRows(ids, counts, step => EmitCanvas(step, embedded, soft, CanvasHead.Logits, null),
                    out StepOutputs outputs, out owned, "diffusion-canvas-batch");
                float[] logits = graph.Outputs[outputs.Logits];
                var result = new float[n][];
                for (int b = 0, at = 0; b < n; at += counts[b], b++)
                {
                    result[b] = new float[(long)counts[b] * vocab];
                    Array.Copy(logits, (long)at * vocab, result[b], 0, result[b].LongLength);
                }
                _canvasSteps += n;
                BatchedCanvasStepsForTests++;
                return result;
            }
            finally
            {
                if (owned) graph?.Dispose();
                embedded?.Dispose();
                soft?.Dispose();
                _swForward.Stop();
                _tCanvas += Stopwatch.GetTimestamp() - t0;
            }
        }

        private static void RequirePrefilled(DiffusionSeqState seq)
        {
            ArgumentNullException.ThrowIfNull(seq);
            if (!seq.Prefilled || seq.Cache.Tokens == null)
                throw new InvalidOperationException("Prefill the sequence (PrefillSeq) before its canvas steps.");
        }

        // ---------------------------------------------------------------- lifecycle

        public override void WarmUpKernels()
        {
            // A one-token prompt and a four-row canvas: the products, norms, experts and attention kernels of both kinds of step.
            try
            {
                int bos = Tokenizer != null && Tokenizer.BosTokenId >= 0 ? Tokenizer.BosTokenId : 0;
                PrefillPrompt(new[] { bos });
                DecodeCanvas(new[] { _maskTokenId, _maskTokenId, _maskTokenId, _maskTokenId }, null, 0f, 1f);
                ResetKVCache();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  DiffusionGemma warmup skipped: {ex.Message}");
            }
            ResetForwardTiming();
            ResetDiffusionStageTiming();
        }

        /// <summary>The prefill / canvas split of the forwards so far.</summary>
        public void PrintForwardTiming()
        {
            double f = 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"  DiffusionGemma forwards: {_swForward.ElapsedMilliseconds} ms (prefill {_tPrefill * f:F0} ms, " +
                $"{_canvasSteps} canvas steps {_tCanvas * f:F0} ms of which self-conditioning {_tSc * f:F0} ms and readback {_tRead * f:F0} ms)");
        }

        /// <summary>Zero the prefill / canvas split <see cref="PrintForwardTiming"/> reports.</summary>
        public void ResetDiffusionStageTiming()
        {
            _swForward.Reset();
            _tPrefill = _tCanvas = _tSc = _tRead = _canvasSteps = 0;
        }

        public override void Dispose()
        {
            ClearStructuredCache();
            DisposeVisionState();
            base.Dispose();
        }
    }

    /// <summary>One in-flight diffusion request's state for the batched scheduler: its cache in the runtime (by
    /// <see cref="Id"/>), what that cache holds, and the request's own image spans. Created by
    /// <see cref="DiffusionGemmaModel.CreateSeqState"/>, freed by <see cref="DiffusionGemmaModel.DisposeSeqState"/>.</summary>
    public sealed class DiffusionSeqState
    {
        internal DiffusionSeqState(string id) => Id = id;

        internal string Id { get; }

        internal bool Prefilled;

        internal DiffusionGemmaModel.PromptCache Cache { get; } = new();

        /// <summary>This sequence's image spans (null until one is set): they live with the request, never on the model, so two
        /// concurrent image requests cannot overwrite each other's (vLLM's and SGLang's choice too).</summary>
        internal VisionSet Vision;

        internal bool HasVision => Vision != null && Vision.Spans.Count > 0;

        /// <summary>Releases this sequence's image embeddings with it.</summary>
        internal void DisposeVision()
        {
            Vision?.Dispose();
            Vision = null;
        }
    }
}
