// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3.8-Flash-Next (<c>qwen4exp</c>) on the shared decoder runtime: Qwen 3.5's Gated DeltaNet and gated
    /// attention (<see cref="Qwen35RuntimeModel"/>) under hyper-connections, with a PLE layer and the MoE's
    /// shared expert behind a scalar sigmoid gate.
    /// <list type="bullet">
    /// <item><b>Hyper-connections:</b> the residual stream is <c>hc</c> copies of the hidden row. A block reads
    /// it through a grouped RMS norm (one gamma per stream element), a low-rank gate
    /// <c>sigmoid(W_up · silu(W_down · xn / hc))</c> and the mean over the streams of <c>xn * gate</c>, and writes
    /// back <c>stream_c += out * 2 * sigmoid(W_inject · xn / hc)_c</c>. The final mixer is the output norm.</item>
    /// <item><b>Gated DeltaNet:</b> the mixer reads the hyper-connection mixer's row (no input norm) and gates
    /// its output with <c>sigmoid(z)</c>.</item>
    /// <item><b>PLE</b> (the layers in <c>ple.layers</c>): rows of an n-gram hash table are gathered on the host per
    /// token, projected to a key [hc·n] and a value [n]; per stream <c>s = key_c · query_c / sqrt(n)</c> (both grouped-
    /// normed, the query from the stream), <c>gated_c = value * sigmoid(sign(s) sqrt(max(|s|, 1e-6)))</c>, and
    /// <c>stream += gated + silu(conv(norm(gated)))</c> with a causal depthwise conv dilated by the n-gram size. The
    /// n-gram history and the conv history are the sequence's state.</item>
    /// <item><b>QSA</b> layers keep every cell's indexer key and attend the cells of their top blocks past
    /// <c>top_k / ratio * ratio + ratio - 1</c> cached tokens, densely within (Qwen4ExpRuntimeModel.Qsa.cs).</item>
    /// </list>
    /// Speculation (the MTP head, DFlash2) and expert tiers are not ported yet: <see cref="Qwen4ExpModel"/> serves
    /// them.
    /// </summary>
    public sealed partial class Qwen4ExpRuntimeModel : Qwen35RuntimeModel
    {
        private const string PleTable = "per_layer_token_embd.weight";
        private const float PleScoreFloor = 1e-6f;

        private PleSpec _ple;
        private readonly Dictionary<int, Tensor> _pleTaps = new();

        public Qwen4ExpRuntimeModel(string ggufPath, BackendType backend, string draftModelPath = null)
            : base(ggufPath, backend, 1, null, 1, draftModelPath)
        {
            PleSpec ple = Ple;
            if (ple.Hc < 2)
            {
                Dispose();
                throw new NotSupportedException($"{Config.Architecture}: {ple.Hc} hyper-connection streams; the runtime's qwen4exp needs 2 or more.");
            }
            if (GraphExecutor == null)
            {
                Dispose();
                throw new NotSupportedException($"{Config.Architecture}: backend {backend} registers no graph executor; qwen4exp runs only as graphs.");
            }
            for (int l = 0; l < Config.NumLayers; l++)
                if (ple.IsPle[l] && !IsRecurrentLayer(l))
                {
                    Dispose();
                    throw new NotSupportedException($"{Config.Architecture}: PLE on attention layer {l} (the runtime keeps PLE state with a recurrent layer's).");
                }
            if (ple.Layers > 0 && !_quantWeights.ContainsKey(PleTable) && !_weights.ContainsKey(PleTable))
            {
                Dispose();
                throw new NotSupportedException($"{Config.Architecture}: PLE layers without {PleTable}.");
            }
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"  Hyper-connections: {ple.Hc} streams, low rank {ple.HcLowRank}; PLE layers: {ple.Layers} " +
                $"(n-gram {ple.Ngram}, {ple.Heads} heads of {ple.HeadDim}, conv {ple.ConvKernel}); QSA dense up to {ple.QsaBudget} tokens, sparse past it."));
            // The table (~29 GB on Flash Next) is read a few scattered rows a token from the mapping: a fault should
            // read the row's page, not a read-ahead cluster around it (Qwen4ExpModel.PreparePleTableAccess).
            if (_quantWeights.TryGetValue(PleTable, out QuantizedWeight table) && table.IsFileBacked && table.AdviseRandomAccess())
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"  PLE n-gram table: {table.RawBytes / 1e9:F1} GB read on demand from the GGUF mapping (random-access advice)."));
        }

        // ---------------------------------------------------------------- configuration

        /// <summary>The family's constants, read from the GGUF on first use: the base constructor already calls
        /// the hooks below (states, stream width) before this class's fields are set.</summary>
        private PleSpec Ple => _ple ??= new PleSpec(_gguf, Config);

        private sealed class PleSpec
        {
            public readonly int Hc, HcLowRank, Hidden;
            public readonly bool[] IsPle;
            public readonly int Layers, Ngram, HeadsPerNgram, Heads, HeadDim, ConvKernel, Eos;
            public readonly ulong[] Multipliers, Offsets, VocabSizes;
            public readonly int[] CompressRatios;
            public readonly long QsaBudget;

            public PleSpec(GgufFile gguf, ModelConfig config)
            {
                string a = gguf.GetString("general.architecture") ?? "qwen4exp";
                Hidden = config.HiddenSize;
                Hc = (int)gguf.GetUint32($"{a}.hyper_connection.count", 1);
                HcLowRank = (int)gguf.GetUint32($"{a}.hyper_connection.low_rank", 0);
                int layers = config.NumLayers;
                IsPle = new bool[layers];
                if (gguf.GetInt32Array($"{a}.ple.layers") is { Length: > 0 } pleLayers)
                {
                    foreach (int l in pleLayers)
                    {
                        if (l < 0 || l >= layers) throw new NotSupportedException($"{a}.ple.layers names layer {l}, outside 0..{layers - 1}.");
                        IsPle[l] = true;
                        Layers++;
                    }
                    Ngram = (int)gguf.GetUint32($"{a}.ple.ngram_size");
                    HeadsPerNgram = (int)gguf.GetUint32($"{a}.ple.heads_per_ngram");
                    ConvKernel = (int)gguf.GetUint32($"{a}.ple.conv_kernel");
                    Eos = (int)gguf.GetUint32($"{a}.ple.eos_token_id");
                    HeadDim = (int)gguf.GetUint32($"{a}.embedding_length_per_layer_input");
                    Heads = (Ngram - 1) * HeadsPerNgram;
                    Multipliers = gguf.GetUint64Array($"{a}.ple.layer_multipliers");
                    Offsets = gguf.GetUint64Array($"{a}.ple.head_offsets");
                    VocabSizes = gguf.GetUint64Array($"{a}.ple.head_vocab_sizes");
                    if (Ngram < 2 || Heads * HeadDim != Hidden || ConvKernel < 1
                        || Multipliers == null || Multipliers.Length < Ngram || Offsets == null || Offsets.Length < Heads
                        || VocabSizes == null || VocabSizes.Length < Heads)
                        throw new NotSupportedException($"{a}: inconsistent PLE metadata (n-gram {Ngram}, {Heads} heads of {HeadDim} for hidden {Hidden}).");
                }
                CompressRatios = new int[layers];
                if (gguf.GetInt32Array($"{a}.attention.compress_ratios") is { } ratios)
                    for (int l = 0; l < layers && l < ratios.Length; l++) CompressRatios[l] = ratios[l];
                // indexer.top_k counts cells: a QSA layer attends the top top_k / ratio complete blocks of ratio cells
                // and the incomplete tail (llama.cpp src/models/qwen4exp.cpp: n_sel = ratio * (top_k / ratio) + ratio - 1),
                // so it attends everything, densely, up to that many cells (Flash Next: 2,051).
                long topK = gguf.GetUint32($"{a}.attention.indexer.top_k", 0);
                QsaBudget = long.MaxValue;
                for (int l = 0; l < layers; l++)
                    if (CompressRatios[l] > 0) QsaBudget = Math.Min(QsaBudget, topK / CompressRatios[l] * CompressRatios[l] + CompressRatios[l] - 1);
            }

            /// <summary>Rows of the PLE layers' conv history: (kernel - 1) taps, each n-gram positions apart.</summary>
            public int ConvHistory => (ConvKernel - 1) * Ngram;

            public long StreamFloats => (long)Hc * Hidden;
        }

        // ---------------------------------------------------------------- family hooks

        protected override bool MixerSigmoidGate => true;

        protected override bool RequiresGraphs => true;

        protected override int StreamWidth => checked((int)Ple.StreamFloats);

        // ---------------------------------------------------------------- the MTP head

        /// <summary>The shared-target MTP head (<c>--draft-model mtp-*.gguf</c>): one NextN block after the trunk.</summary>
        protected override GgufFile OpenSeparateDraftHead(string draftModelPath)
        {
            if (string.IsNullOrWhiteSpace(draftModelPath) || Qwen4ExpModel.IsDFlashDraftFile(draftModelPath)) return null;
            var file = new GgufFile(draftModelPath);
            try
            {
                if (file.IsSplit) throw new NotSupportedException("The shared qwen4exp MTP head must be a single GGUF file.");
                int layer = Qwen4ExpModel.ValidateMtpHead(_gguf, file);
                if (layer != Config.NumLayers)
                    throw new InvalidOperationException($"qwen4exp MTP block {layer} does not follow the {Config.NumLayers}-layer trunk.");
                return file;
            }
            catch
            {
                file.Dispose();
                throw;
            }
        }

        /// <summary>The head reads the trunk's whole stream and chains its own (Qwen4ExpModel's NextN block).</summary>
        protected override bool DraftChainsStream => true;

        /// <summary>Each verify row routes its own ten experts, so a draft the MTP head is unsure of costs more than it
        /// saves (Qwen4ExpModel's gate). UD-Q2_K_XL on the 48 GB M5 Pro, 200 greedy tokens: at 0.5 five runs gave 29.8 and
        /// 33.9-34.1 tok/s; at the per-token default 0.15 four gave 33.7-34.0 twice and 19.6-20.1 twice.</summary>
        public override float SpecPreferredDraftGate => 0.5f;

        protected override string[] DraftLayerWeights =>
            new[] { "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "attn_output.weight", "nextn.hc_head_norm.weight" };

        /// <summary>Each stream sees [rms(embedding) * enorm, grouped rms(stream) * hnorm] through eh_proj (a single
        /// norm over the whole stream, or one projection of all streams, is a different trained model).</summary>
        protected override int EmitDraftInput(GraphStep step, int layer, int embedded, int hidden)
        {
            PleSpec p = Ple;
            int rows = step.Rows, hc = p.Hc;
            long n = p.Hidden;
            int e = step.RmsNorm(embedded, LayerName(layer, "nextn.enorm.weight"));
            int h = GroupedNorm(step, hidden, rows, LayerName(layer, "nextn.hnorm.weight"));
            int joined = step.Op(GraphOp.Concat, stackalloc[] { Streams(step, step.Reshape(e, n, 1, rows), hc), step.Reshape(h, n, hc, rows) },
                stackalloc long[] { 0 });                                                                   // [2n, hc, rows]
            int x = step.MatMul(LayerName(layer, "nextn.eh_proj.weight"), step.Reshape(joined, 2 * n, (long)hc * rows));
            return step.Reshape(x, p.StreamFloats, rows);
        }

        /// <summary>The head's mixer on the last row (nextn.hc_head_*), then the trunk's output.weight.</summary>
        protected override int EmitDraftHead(GraphStep step, int layer, int last)
            => HcMix(step, last, 1, LayerName(layer, "nextn.hc_head_norm.weight"), LayerName(layer, "nextn.hc_head_down.weight"),
                LayerName(layer, "nextn.hc_head_up.weight"), null, out _);

        // ---------------------------------------------------------------- verify windows

        protected override void CommitHostVerify(int kept, int rows)
        {
            foreach (RecurrentState state in ActiveRecurrentStates())
                if (state is PleGdnState ple) ple.CommitVerify(kept);
        }

        /// <summary>The PLE conv window after the kept rows, from the verify's conv history.</summary>
        protected override void EmitCommitExtras(GraphProgramBuilder g, int layer, DeviceGdnState state, int kept, int rows)
        {
            if (state is not PleGdnState ple || ple.PleConvHistory == null) return;
            const long f = sizeof(float);
            PleSpec p = Ple;
            long width = p.StreamFloats;
            int history = p.ConvHistory;
            int all = GraphLeaf(g, ple.PleConvHistory, width, history + ple.SnapshotCapacity);
            int window = g.Op(GraphOp.View, stackalloc[] { all }, stackalloc long[] { 2, width, history, 1, 1, width * f, 0, 0, kept * width * f });
            g.Effect(g.Op(GraphOp.Cpy, window, GraphLeaf(g, ple.PleConv, width, history)));
        }

        protected override DeviceGdnState CreateDeviceGdnState(RankScope scope, int layer, int convRows, int channels, int vHeads, int dv, int dk)
            => Ple.IsPle[layer]
                ? new PleGdnState(scope.Allocator, GraphExecutor, HostPointer, convRows, channels, vHeads, dv, dk, Ple)
                : base.CreateDeviceGdnState(scope, layer, convRows, channels, vHeads, dv, dk);

        /// <summary>The embedding's hc copies, stream c at elements [c·n, (c+1)·n) of a row.</summary>
        protected override int EmitStreamInput(GraphStep step, int embedded)
        {
            int stream = embedded;
            for (int c = 1; c < Ple.Hc; c++)
                stream = step.Op(GraphOp.Concat, stackalloc[] { stream, embedded }, stackalloc long[] { 0 });
            return stream;
        }

        protected override int EmitBlockInput(GraphStep step, int layer, int stream, BlockSite site, out int context)
        {
            string part = site == BlockSite.Mixer ? "attn" : "ffn";
            return HcMix(step, stream, step.Rows, LayerName(layer, $"hc_{part}_norm.weight"), LayerName(layer, $"hc_{part}_down.weight"),
                LayerName(layer, $"hc_{part}_up.weight"), LayerName(layer, $"hc_{part}_inject.weight"), out context);
        }

        /// <summary>stream_c += output * 2 * sigmoid(inject_c / hc).</summary>
        protected override int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
        {
            PleSpec p = Ple;
            int rows = step.Rows, hc = p.Hc;
            int w = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Sigmoid, step.Op(GraphOp.Scale, stackalloc[] { context }, default, stackalloc[] { 1f / hc })) },
                default, stackalloc[] { 2f });
            int scaled = Outer(step, step.Cont(output), p.Hidden, step.Reshape(w, 1, hc, rows), hc, rows);
            return step.Op(GraphOp.Add, stream, step.Reshape(scaled, p.StreamFloats, rows));
        }

        /// <summary>The final mixer is the output norm (qwen4exp has no output_norm).</summary>
        protected override int EmitStreamOutput(GraphStep step, int stream)
            => HcMix(step, stream, step.Rows, "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", null, out _);

        /// <summary>The PLE table's rows are gathered on the host; the accelerator never binds it.</summary>
        protected override bool IsHostOnlyWeight(string name) => name == PleTable;

        /// <summary>DFlash2's encoder was trained on each tapped residual averaged over the hc streams
        /// (Qwen4ExpModel.DFlash.cs).</summary>
        protected override int EmitCaptureRow(GraphStep step, int stream)
        {
            PleSpec p = Ple;
            int rows = step.Rows;
            long n = p.Hidden, width = p.StreamFloats;
            const long f = sizeof(float);
            int sum = step.View2(stream, n, rows, width * f, 0);
            for (int c = 1; c < p.Hc; c++)
                sum = step.Op(GraphOp.Add, sum, step.View2(stream, n, rows, width * f, c * n * f));
            return step.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / p.Hc });
        }

        protected override int EmitLayerPrologue(GraphStep step, int layer, int stream)
        {
            PleSpec p = Ple;
            return p.IsPle[layer] ? EmitPle(step, layer, stream) : stream;
        }

        // ---------------------------------------------------------------- hyper-connections

        /// <summary>The grouped norm, low-rank gate and stream mean: [n, rows], with the inject [hc, rows] when
        /// <paramref name="inject"/> is named.</summary>
        private int HcMix(GraphStep step, int stream, int rows, string norm, string down, string up, string inject, out int injected)
        {
            PleSpec p = Ple;
            int hc = p.Hc;
            long n = p.Hidden, width = p.StreamFloats;
            int xn = GroupedNorm(step, stream, rows, norm);
            int lo = step.Op(GraphOp.Silu, step.Op(GraphOp.Scale, stackalloc[] { step.MatMul(down, xn) }, default, stackalloc[] { 1f / hc }));
            int gate = step.Op(GraphOp.Sigmoid, step.MatMul(up, lo));
            int gated = step.Op(GraphOp.Mul, xn, gate);
            const long f = sizeof(float);
            int mixed = step.View2(gated, n, rows, width * f, 0);
            for (int c = 1; c < hc; c++)
                mixed = step.Op(GraphOp.Add, mixed, step.View2(gated, n, rows, width * f, c * n * f));
            mixed = step.Op(GraphOp.Scale, stackalloc[] { mixed }, default, stackalloc[] { 1f / hc });
            injected = inject != null ? step.MatMul(inject, xn) : -1;
            return mixed;
        }

        /// <summary>RMS norm of each stream of x [hc·n, rows], times the [hc·n] gammas of <paramref name="weight"/>.</summary>
        private int GroupedNorm(GraphStep step, int x, int rows, string weight)
        {
            PleSpec p = Ple;
            int perStream = step.Op(GraphOp.RmsNorm, stackalloc[] { step.Reshape(step.Cont(x), p.Hidden, p.Hc, rows) }, default, stackalloc[] { Config.Eps });
            return step.Op(GraphOp.Mul, step.Reshape(perStream, p.StreamFloats, rows), step.Weight(weight));
        }

        /// <summary>x [n, rows] times each row's [hc] weights: [n, hc, rows], each element the one product x[m] * w[c]. As a
        /// one-term product (a [1, n, rows] by [1, hc, rows] mul_mat) instead of hc copies of x and a multiply: ggml-metal
        /// runs it as one kernel (Qwen4ExpModel's scatter).</summary>
        private static int Outer(GraphStep step, int x, long n, int weights, int hc, int rows)
            => step.Op(GraphOp.MulMat, step.Reshape(x, 1, n, rows), step.Reshape(weights, 1, hc, rows));

        /// <summary>x [n, 1, rows] as [n, hc, rows] (hc copies along the stream axis).</summary>
        private static int Streams(GraphStep step, int x, int hc)
        {
            int all = x;
            for (int c = 1; c < hc; c++)
                all = step.Op(GraphOp.Concat, stackalloc[] { all, x }, stackalloc long[] { 1 });
            return all;
        }

        // ---------------------------------------------------------------- PLE

        private int EmitPle(GraphStep step, int layer, int stream)
        {
            PleSpec p = Ple;
            int rows = step.Rows, hc = p.Hc;
            long n = p.Hidden, width = p.StreamFloats;
            const long f = sizeof(float);

            int embedded = step.InputF32(GatherPleRows(step, layer), n, rows);
            int key = GroupedNorm(step, step.MatMul(LayerName(layer, "ple_key.weight"), embedded), rows, LayerName(layer, "ple_norm_key.weight"));
            int value = step.MatMul(LayerName(layer, "ple_value.weight"), embedded);                  // [n, rows]
            int query = GroupedNorm(step, stream, rows, LayerName(layer, "ple_norm_query.weight"));
            // s = key_c · query_c / sqrt(n) per stream; gate = sigmoid(sign(s) sqrt(max(|s|, floor))).
            int dot = step.Op(GraphOp.SumRows, step.Reshape(step.Op(GraphOp.Mul, key, query), n, hc, rows));   // [1, hc, rows]
            int score = step.Op(GraphOp.Scale, stackalloc[] { dot }, default, stackalloc[] { 1f / MathF.Sqrt(n) });
            int magnitude = step.Op(GraphOp.Sqrt, step.Op(GraphOp.Sqr, score));
            int root = step.Op(GraphOp.Sqrt, stackalloc[] { step.Op(GraphOp.Clamp, stackalloc[] { magnitude }, default, stackalloc[] { PleScoreFloor, float.MaxValue }) });
            int sign = step.Op(GraphOp.Div, score, step.Op(GraphOp.Clamp, stackalloc[] { magnitude }, default, stackalloc[] { float.Epsilon * 1e8f, float.MaxValue }));
            int gate = step.Op(GraphOp.Sigmoid, step.Op(GraphOp.Mul, sign, root));                     // [1, hc, rows]
            int gated = step.Reshape(Outer(step, value, n, gate, hc, rows), width, rows);

            // The causal depthwise conv over [history | rows], taps n-gram positions apart, then silu.
            int normed = GroupedNorm(step, gated, rows, LayerName(layer, "ple_norm_conv.weight"));
            int taps = step.TensorWeight(PleTaps(layer), width, p.ConvKernel);
            int history = p.ConvHistory;
            int conv = -1;
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s];
                var state = (PleGdnState)step.SequenceState(s, layer);
                int rowsOf = step.Sequences == 1 ? normed : step.View2(normed, width, count, width * f, step.First[s] * width * f);
                int padded = rowsOf, past = 0;
                if (history > 0)
                {
                    past = step.State(state.PleConv, GraphFormat.F32, width, history);
                    padded = step.Op(GraphOp.Concat, stackalloc[] { past, rowsOf }, stackalloc long[] { 1 });   // [width, history + count]
                }
                int sum = -1;
                for (int k = 0; k < p.ConvKernel; k++)
                {
                    long back = (long)(p.ConvKernel - 1 - k) * p.Ngram;
                    int shifted = step.View2(padded, width, count, width * f, (history - back) * width * f);
                    int term = step.Op(GraphOp.Mul, shifted, step.View2(taps, width, 1, width * f, k * width * f));
                    sum = sum < 0 ? term : step.Op(GraphOp.Add, sum, term);
                }
                int part = step.Op(GraphOp.Silu, sum);
                // A verify window leaves the live history alone: the commit sets it from the window's conv history.
                bool keep = step.RecurrentSnapshots && count > 1;
                if (history > 0 && !keep)
                    step.Effect(step.Op(GraphOp.Cpy, step.View2(padded, width, history, width * f, (long)count * width * f), past));
                else if (history > 0)
                {
                    if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                    int all = step.State(state.PleConvHistory, GraphFormat.F32, width, history + state.SnapshotCapacity);
                    step.Effect(step.Op(GraphOp.Cpy, padded, step.View2(all, width, history + count, width * f, 0)));
                }
                conv = conv < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { conv, part }, stackalloc long[] { 1 });
            }
            return step.Op(GraphOp.Add, stream, step.Op(GraphOp.Add, gated, conv));
        }

        /// <summary>ple_conv1d [kernel, hc·n] (one weight per channel and tap) as tap rows on the device.</summary>
        private unsafe Tensor PleTaps(int layer)
        {
            if (_pleTaps.TryGetValue(layer, out Tensor taps)) return taps;
            PleSpec p = Ple;
            string name = LayerName(layer, "ple_conv1d.weight");
            long width = p.StreamFloats;
            int kernel = p.ConvKernel;
            var raw = new float[width * kernel];
            if (_weights.TryGetValue(name, out Tensor w))
                Array.Copy(w.GetElementsAsFloat((int)w.ElementCount()), raw, raw.Length);
            else if (_quantWeights.TryGetValue(name, out QuantizedWeight q))
                NativeDequant.DequantizeToFloat32(q.GgmlType, q.Data, raw, 0, raw.Length);
            else
                throw new NotSupportedException($"{Config.Architecture}: {name} is missing.");
            var rowsByTap = new float[raw.Length];
            for (long c = 0; c < width; c++)
                for (int k = 0; k < kernel; k++)
                    rowsByTap[k * width + c] = raw[c * kernel + k];
            taps = CreateFloatTensor(rowsByTap, kernel, width);
            _pleTaps[layer] = taps;
            return taps;
        }

        /// <summary>This step's PLE rows, [rows, n]: each sequence's n-gram hashes over its history and its rows (the
        /// history then advances), gathered from the host table.</summary>
        private const int PleExplicitReadMinRows = 1024;
        private byte[] _plePackedRows;

        /// <summary>The rows read explicitly into a packed buffer (row i at i * row bytes); 0 to read the mapping.</summary>
        private unsafe long ReadPleRowsPacked(QuantizedWeight table, int[] ids, long rowBytes)
        {
            long need = ids.Length * rowBytes;
            if (need > int.MaxValue) return 0;
            if (_plePackedRows == null || _plePackedRows.Length < need)
                _plePackedRows = GC.AllocateUninitializedArray<byte>((int)need, pinned: true);
            var packed = (IntPtr)System.Runtime.CompilerServices.Unsafe.AsPointer(ref _plePackedRows[0]);
            return GgmlBasicOps.HostReadRows(table.Data, rowBytes, ids, packed) ? (long)packed : 0;
        }

        private unsafe float[] GatherPleRows(GraphStep step, int layer)
        {
            PleSpec p = Ple;
            int[] tokens = StepTokens;
            if (tokens == null || tokens.Length != step.Rows)
                throw new InvalidOperationException($"{Config.Architecture}: the PLE layer needs the step's {step.Rows} tokens.");
            var ids = new int[(long)step.Rows * p.Heads];
            for (int s = 0; s < step.Sequences; s++)
            {
                var state = (PleGdnState)step.SequenceState(s, layer);
                state.HashRows(tokens, step.First[s], step.Counts[s], step.Starts[s], ids);
            }
            var embedded = new float[(long)step.Rows * p.Hidden];
            if (_quantWeights.TryGetValue(PleTable, out QuantizedWeight table))
            {
                int type = table.GgmlType, dim = p.HeadDim;
                long rowBytes = NativeDequant.RowSize(type, table.Ne0);
                // A row the page cache lacks is a read from disk: a prompt's rows are read explicitly and packed (one
                // pread each on the native I/O team; as faults a 4096-token chunk's 65K rows took 1.1-3.1 s on an M5
                // Pro), and the rows are dequantized concurrently, so a decode's 16 faults overlap (Qwen4ExpModel.Ple).
                long packed = table.IsFileBacked && ids.Length >= PleExplicitReadMinRows ? ReadPleRowsPacked(table, ids, rowBytes) : 0;
                long source = packed != 0 ? packed : (long)table.Data;
                fixed (float* dst = embedded)
                {
                    long dstA = (long)dst;
                    void Row(int i) => NativeDequant.DequantizeToFloat32Native(type, (IntPtr)(source + (packed != 0 ? i : ids[i]) * rowBytes),
                        (IntPtr)(dstA + (long)i * dim * sizeof(float)), dim);
                    if (ids.Length > 1 && (table.IsFileBacked || ids.Length >= 16 * p.Heads))
                        Parallel.For(0, ids.Length, Row);
                    else
                        for (int i = 0; i < ids.Length; i++) Row(i);
                }
            }
            else
            {
                Tensor dense = _weights[PleTable];
                for (long i = 0; i < ids.Length; i++)
                {
                    using Tensor row = dense.Narrow(0, ids[i], 1);
                    Array.Copy(row.GetElementsAsFloat(p.HeadDim), 0, embedded, i * p.HeadDim, p.HeadDim);
                }
            }
            return embedded;
        }

        /// <summary>A PLE layer's sequence state: the recurrence's, the PLE conv history [history, hc·n] and the
        /// n-gram history (the last n-1 tokens and the position after them, as floats in a tensor so copies,
        /// snapshots and prefix checkpoints carry it; a host mirror serves the hashing).</summary>
        private sealed class PleGdnState : DeviceGdnState
        {
            private readonly PleSpec _p;
            private readonly int[] _tokens, _undoTokens;
            private int _count, _next;
            private bool _stale;
            // The history before the last hashed step: a step's emission may run twice (a replayable step's
            // signature pass, then its recording), and both must hash with the history as of its start.
            private int _undoCount, _undoNext, _lastStart = -1, _lastCount = -1;

            public PleGdnState(IAllocator allocator, IGraphExecutor executor, Func<Tensor, IntPtr> hostPointer,
                int convRows, int channels, int vHeads, int dv, int dk, PleSpec p)
                : base(allocator, executor, hostPointer, convRows, channels, vHeads, dv, dk)
            {
                _p = p;
                _tokens = new int[p.Ngram - 1];
                _undoTokens = new int[p.Ngram - 1];
                PleConv = new Tensor(allocator, DType.Float32, Math.Max(1, p.ConvHistory), p.StreamFloats);
                PleTokens = new Tensor(allocator, DType.Float32, 2 + p.Ngram - 1);
                ResetPle();
            }

            public Tensor PleConv { get; }
            public Tensor PleTokens { get; }
            /// <summary>A verify window's [history | rows] conv input (the commit's source), with room for the
            /// verify capacity.</summary>
            public Tensor PleConvHistory { get; private set; }
            private int[] _verifyTokens = Array.Empty<int>();

            public override bool EnsureSpecBuffers(int rows)
            {
                bool grown = base.EnsureSpecBuffers(rows);
                if (!grown && PleConvHistory != null) return false;
                PleConvHistory?.Dispose();
                PleConvHistory = new Tensor(PleConv.Allocator, DType.Float32, Math.Max(1, _p.ConvHistory) + SnapshotCapacity, _p.StreamFloats);
                return true;
            }

            /// <summary>A verify kept <paramref name="kept"/> rows: the n-gram history as of the window's start, advanced
            /// by those rows (0: the history before the window).</summary>
            public void CommitVerify(int kept)
            {
                ReadTokens();
                int start = _lastStart;
                _count = _undoCount;
                _next = _undoNext;
                Array.Copy(_undoTokens, _tokens, _tokens.Length);
                if (kept > 0) Advance(_verifyTokens, 0, Math.Min(kept, _verifyTokens.Length), start);
                _lastStart = -1;
                WriteTokens();
            }

            public override void Reset()
            {
                base.Reset();
                ResetPle();
            }

            /// <summary>The token history is host-only (graphs never bind it), so rewriting it drops nothing.</summary>
            public override bool ResetInPlace()
            {
                if (!_host)
                {
                    Reset();
                    return true;
                }
                if (!base.ResetInPlace()) return false;
                ClearInPlace(PleConv);
                _count = _next = 0;
                _stale = false;
                _lastStart = -1;
                WriteTokens();
                return true;
            }

            private void ResetPle()
            {
                if (PleConv == null) return;   // the base constructor resets before this one ran
                Clear(PleConv);
                _count = _next = 0;
                _stale = false;
                _lastStart = -1;
                WriteTokens();
            }

            protected internal override IEnumerable<Tensor> HostTensors => Concat(base.HostTensors);

            protected internal override IEnumerable<Tensor> CheckpointTensors => Concat(base.CheckpointTensors);

            private IEnumerable<Tensor> Concat(IEnumerable<Tensor> first)
            {
                foreach (Tensor t in first) yield return t;
                yield return PleConv;
                yield return PleTokens;
            }

            protected internal override unsafe void CopyFrom(RecurrentState source)
            {
                base.CopyFrom(source);
                var other = (PleGdnState)source;
                foreach (var (from, to) in new[] { (other.PleConv, PleConv), (other.PleTokens, PleTokens) })
                {
                    if (!_host)
                    {
                        Ops.Copy(to, from);
                        continue;
                    }
                    long bytes = to.ElementCount() * sizeof(float);
                    Buffer.MemoryCopy((void*)_hostPointer(from), (void*)_hostPointer(to), bytes, bytes);
                    Invalidate(to);
                }
                _stale = true;
                _lastStart = -1;
            }

            protected internal override bool CopyFromInPlace(RecurrentState source)
            {
                if (!_host)
                {
                    CopyFrom(source);
                    return true;
                }
                if (!base.CopyFromInPlace(source)) return false;
                var other = (PleGdnState)source;
                CopyInPlace(other.PleConv, PleConv);
                CopyInPlace(other.PleTokens, PleTokens);
                _stale = true;
                _lastStart = -1;
                return true;
            }

            protected internal override Tensor[] GraphTensors
                => base.GraphTensors is { } graph ? [.. graph, PleConv] : null;

            /// <summary>The token history lives on the host only.</summary>
            protected internal override unsafe void AfterDeviceCopy(RecurrentState source)
            {
                base.AfterDeviceCopy(source);
                var other = (PleGdnState)source;
                long bytes = PleTokens.ElementCount() * sizeof(float);
                Buffer.MemoryCopy((void*)_hostPointer(other.PleTokens), (void*)_hostPointer(PleTokens), bytes, bytes);
                Invalidate(PleTokens);
                _stale = true;
                _lastStart = -1;
            }

            protected internal override void OnHostWritten()
            {
                base.OnHostWritten();
                if (_host) Invalidate(PleConv);
                _stale = true;
                _lastStart = -1;
            }

            /// <summary>Hash rows [first, first + count) of <paramref name="tokens"/> (positions from
            /// <paramref name="start"/>) into <paramref name="ids"/> (heads fastest), then advance the history:
            /// Qwen4ExpModel.ComputePleRows's rule, with this sequence's history.</summary>
            public void HashRows(int[] tokens, int first, int count, int start, int[] ids)
            {
                ReadTokens();
                if (start == _lastStart && count == _lastCount)
                {
                    _count = _undoCount;
                    _next = _undoNext;
                    Array.Copy(_undoTokens, _tokens, _tokens.Length);
                }
                else
                {
                    _undoCount = _count;
                    _undoNext = _next;
                    Array.Copy(_tokens, _undoTokens, _tokens.Length);
                }
                _lastStart = start;
                _lastCount = count;
                int ngram = _p.Ngram, eos = _p.Eos;
                var hist = new List<int>(ngram);
                if (_next == start)
                    for (int i = 0; i < _count; i++) hist.Add(_tokens[i]);
                while (hist.Count < ngram - 1) hist.Insert(0, eos);
                var ctx = new long[ngram];
                for (int i = 0; i < count; i++)
                {
                    int pos = start + i;
                    ctx[0] = tokens[first + i];
                    bool cut = false;
                    for (int s = 1; s < ngram; s++)
                    {
                        long tok;
                        if (cut)
                        {
                            tok = eos;
                        }
                        else if (i - s >= 0)
                        {
                            tok = tokens[first + i - s];
                        }
                        else
                        {
                            int back = s - i, idx = hist.Count - back;
                            tok = back > 0 && idx >= 0 && idx < hist.Count && pos - s >= 0 ? hist[idx] : eos;
                        }
                        ctx[s] = tok;
                        if (tok == eos) cut = true;
                    }
                    for (int g = 2; g <= ngram; g++)
                    {
                        ulong mixed = (ulong)ctx[0] * _p.Multipliers[0];
                        for (int j = 1; j < g; j++) mixed ^= (ulong)ctx[j] * _p.Multipliers[j];
                        int baseHead = (g - 2) * _p.HeadsPerNgram;
                        for (int hh = 0; hh < _p.HeadsPerNgram; hh++)
                        {
                            int h = baseHead + hh;
                            ids[(long)(first + i) * _p.Heads + h] = (int)(mixed % _p.VocabSizes[h] + _p.Offsets[h]);
                        }
                    }
                }
                _verifyTokens = tokens.AsSpan(first, count).ToArray();
                Advance(tokens, first, count, start);
                WriteTokens();
            }

            /// <summary>The last n-1 tokens after rows [first, first + count) at <paramref name="start"/> (after the
            /// history's when the rows continue it).</summary>
            private void Advance(int[] tokens, int first, int count, int start)
            {
                var all = new List<int>(_p.Ngram - 1 + count);
                if (_next == start) for (int i = 0; i < _count; i++) all.Add(_tokens[i]);
                for (int i = 0; i < count; i++) all.Add(tokens[first + i]);
                _count = Math.Min(_p.Ngram - 1, all.Count);
                for (int i = 0; i < _count; i++) _tokens[i] = all[all.Count - _count + i];
                _next = start + count;
            }

            private void WriteTokens()
            {
                var v = new float[PleTokens.ElementCount()];
                v[0] = _count;
                v[1] = _next;
                for (int i = 0; i < _count; i++) v[2 + i] = _tokens[i];
                PleTokens.SetElementsAsFloat(v);
                if (_host) Invalidate(PleTokens);
            }

            private void ReadTokens()
            {
                if (!_stale) return;
                float[] v = PleTokens.GetElementsAsFloat((int)PleTokens.ElementCount());
                _count = (int)v[0];
                _next = (int)v[1];
                for (int i = 0; i < _count; i++) _tokens[i] = (int)v[2 + i];
                _stale = false;
            }

            public override void Dispose()
            {
                base.Dispose();
                PleConv.Dispose();
                PleTokens.Dispose();
                PleConvHistory?.Dispose();
            }
        }

        public override void Dispose()
        {
            foreach (Tensor t in _pleTaps.Values) t.Dispose();
            _pleTaps.Clear();
            base.Dispose();
        }
    }
}
