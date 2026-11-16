// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Kokoro-82M (StyleTTS 2 with an iSTFTNet vocoder), as kokoro.KModel.forward_with_tokens runs it:
//
//   ids -> PL-BERT (ALBERT) -> bert_encoder -> DurationEncoder (style-conditioned LSTMs)
//       -> LSTM -> duration_proj -> durations (sigmoid sum / speed, rounded, at least 1)
//   alignment: every phoneme repeated for its duration
//   prosody: shared LSTM -> F0 and N (energy) curves at twice the frame rate (AdainResBlk1d)
//   text encoder: embedding -> 3 x (conv, LayerNorm, LeakyReLU) -> LSTM, aligned like the prosody
//   decoder: AdainResBlk1d stack -> iSTFTNet generator (harmonic-plus-noise source from F0,
//            two transposed-convolution upsamplers with Snake resblocks, magnitude and phase of a
//            20-point inverse STFT) -> 24 kHz audio
//
// The style vector (256) comes from the voice pack row for the phoneme count: its first half
// conditions the decoder, its second half the prosody predictor.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;
using O = TensorSharp.Models.Kokoro.KokoroOps;

namespace TensorSharp.Models.Kokoro
{
    internal sealed class KokoroNetwork
    {
        public const int SampleRate = 24000;

        private readonly Dictionary<string, (float[] Data, int[] Shape)> _w = new();
        private readonly int _hidden, _style, _bertHidden, _bertHeads, _bertLayers, _nFft, _hop;
        private readonly int[] _upRates, _upKernels, _rbKernels;
        private readonly int[][] _rbDilations;
        private readonly int _upInitial, _maxDur;

        /// <summary>When true the harmonic source draws no random phase or noise (validation against a
        /// reference with the same sources zeroed).</summary>
        public bool Deterministic { get; set; }

        public KokoroNetwork(GgufFile gguf, System.Text.Json.JsonElement cfg)
        {
            foreach (var (name, info) in gguf.Tensors)
            {
                if (name.StartsWith("voice.", StringComparison.Ordinal)) continue;
                int[] shape = info.Shape.Select(s => (int)s).Reverse().ToArray();
                _w[name] = (DirectOps.DequantTensor(gguf, name), shape);
            }
            _hidden = cfg.GetProperty("hidden_dim").GetInt32();
            _style = cfg.GetProperty("style_dim").GetInt32();
            _maxDur = cfg.GetProperty("max_dur").GetInt32();
            var bert = cfg.GetProperty("plbert");
            _bertHidden = bert.GetProperty("hidden_size").GetInt32();
            _bertHeads = bert.GetProperty("num_attention_heads").GetInt32();
            _bertLayers = bert.GetProperty("num_hidden_layers").GetInt32();
            var ist = cfg.GetProperty("istftnet");
            _nFft = ist.GetProperty("gen_istft_n_fft").GetInt32();
            _hop = ist.GetProperty("gen_istft_hop_size").GetInt32();
            _upRates = ist.GetProperty("upsample_rates").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            _upKernels = ist.GetProperty("upsample_kernel_sizes").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            _rbKernels = ist.GetProperty("resblock_kernel_sizes").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            _rbDilations = ist.GetProperty("resblock_dilation_sizes").EnumerateArray()
                .Select(a => a.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();
            _upInitial = ist.GetProperty("upsample_initial_channel").GetInt32();
        }

        private float[] W(string name) =>
            _w.TryGetValue(name, out var t) ? t.Data : throw new KeyNotFoundException($"Kokoro weight '{name}' is missing.");
        private int[] S(string name) => _w[name].Shape;
        private bool Has(string name) => _w.ContainsKey(name);

        /// <summary>The waveform and the duration (frames) predicted for each input id.</summary>
        public (float[] Audio, int[] Durations) Forward(int[] ids, float[] refS, float speed, Random rng)
        {
            int n = ids.Length;
            float[] sDec = refS.AsSpan(0, _style).ToArray();
            float[] sPros = refS.AsSpan(_style, _style).ToArray();

            // ---- PL-BERT -> bert_encoder: [n, hidden] ----
            float[] bertOut = Albert(ids);
            float[] dEn = O.Linear(bertOut, n, _bertHidden, W("bert_encoder.weight"), _hidden, W("bert_encoder.bias"));

            // ---- duration encoder: rows [n, hidden + style] ----
            float[] d = DurationEncoder(dEn, n, sPros);
            int dw = _hidden + _style;
            float[] lstm = O.BiLstm(d, n, dw, Lstm("predictor.lstm"));
            float[] logits = O.Linear(lstm, n, _hidden, W("predictor.duration_proj.linear_layer.weight"), _maxDur,
                W("predictor.duration_proj.linear_layer.bias"));
            var durations = new int[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int j = 0; j < _maxDur; j++) sum += O.Sigmoid(logits[(long)i * _maxDur + j]);
                double v = Math.Round((float)(sum / speed), MidpointRounding.ToEven);
                durations[i] = (int)Math.Max(1, v);
            }
            int frames = durations.Sum();

            // ---- alignment: en [dw, frames] (d^T @ aln), asr later from the text encoder ----
            float[] en = Align(d, n, dw, durations, frames, rowsAreTime: true);

            // ---- F0 / N ----
            var (f0, nCurve) = F0N(en, dw, frames, sPros);

            // ---- text encoder -> asr [hidden, frames] ----
            float[] tEn = TextEncoder(ids);                      // [hidden, n]
            float[] asr = Align(tEn, _hidden, n, durations, frames, rowsAreTime: false);

            float[] audio = Decoder(asr, frames, f0, nCurve, sDec, rng);
            return (audio, durations);
        }

        // ---------------------------------------------------------------- PL-BERT (ALBERT)
        private float[] Albert(int[] ids)
        {
            int n = ids.Length;
            var embW = W("bert.embeddings.word_embeddings.weight");
            var posW = W("bert.embeddings.position_embeddings.weight");
            var typeW = W("bert.embeddings.token_type_embeddings.weight");
            int e = S("bert.embeddings.word_embeddings.weight")[1];
            var x = new float[(long)n * e];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < e; j++)
                    x[(long)i * e + j] = embW[(long)ids[i] * e + j] + posW[(long)i * e + j] + typeW[j];
            O.LayerNormRows(x, n, e, W("bert.embeddings.LayerNorm.weight"), W("bert.embeddings.LayerNorm.bias"), 1e-12f);
            int h = _bertHidden;
            float[] hs = O.Linear(x, n, e, W("bert.encoder.embedding_hidden_mapping_in.weight"), h,
                W("bert.encoder.embedding_hidden_mapping_in.bias"));
            const string p = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
            int heads = _bertHeads, hd = h / heads;
            int ffn = S(p + "ffn.weight")[0];
            for (int layer = 0; layer < _bertLayers; layer++)
            {
                float[] q = O.Linear(hs, n, h, W(p + "attention.query.weight"), h, W(p + "attention.query.bias"));
                float[] k = O.Linear(hs, n, h, W(p + "attention.key.weight"), h, W(p + "attention.key.bias"));
                float[] v = O.Linear(hs, n, h, W(p + "attention.value.weight"), h, W(p + "attention.value.bias"));
                var ctx = new float[(long)n * h];
                float scale = 1f / MathF.Sqrt(hd);
                Parallel.For(0, heads, head =>
                {
                    var scores = new float[n];
                    for (int i = 0; i < n; i++)
                    {
                        var qi = new ReadOnlySpan<float>(q, i * h + head * hd, hd);
                        float max = float.NegativeInfinity;
                        for (int j = 0; j < n; j++)
                        {
                            scores[j] = scale * TensorPrimitives.Dot(qi, new ReadOnlySpan<float>(k, j * h + head * hd, hd));
                            if (scores[j] > max) max = scores[j];
                        }
                        double sum = 0;
                        for (int j = 0; j < n; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                        for (int j = 0; j < n; j++)
                        {
                            float pj = (float)(scores[j] / sum);
                            for (int c = 0; c < hd; c++) ctx[(long)i * h + head * hd + c] += pj * v[(long)j * h + head * hd + c];
                        }
                    }
                });
                float[] attn = O.Linear(ctx, n, h, W(p + "attention.dense.weight"), h, W(p + "attention.dense.bias"));
                O.AddInPlace(attn, hs);
                O.LayerNormRows(attn, n, h, W(p + "attention.LayerNorm.weight"), W(p + "attention.LayerNorm.bias"), 1e-12f);
                float[] f = O.Linear(attn, n, h, W(p + "ffn.weight"), ffn, W(p + "ffn.bias"));
                for (long i = 0; i < f.LongLength; i++) f[i] = GeluTanh(f[i]);   // ALBERT's gelu_new
                float[] fo = O.Linear(f, n, ffn, W(p + "ffn_output.weight"), h, W(p + "ffn_output.bias"));
                O.AddInPlace(fo, attn);
                O.LayerNormRows(fo, n, h, W(p + "full_layer_layer_norm.weight"), W(p + "full_layer_layer_norm.bias"), 1e-12f);
                hs = fo;
            }
            return hs;
        }

        private static float GeluTanh(float x) =>
            0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));

        // ---------------------------------------------------------------- prosody predictor
        private KokoroLstm Lstm(string p) => new()
        {
            Hidden = S(p + ".weight_hh_l0")[1],
            WihF = W(p + ".weight_ih_l0"), WhhF = W(p + ".weight_hh_l0"),
            BihF = W(p + ".bias_ih_l0"), BhhF = W(p + ".bias_hh_l0"),
            WihR = W(p + ".weight_ih_l0_reverse"), WhhR = W(p + ".weight_hh_l0_reverse"),
            BihR = W(p + ".bias_ih_l0_reverse"), BhhR = W(p + ".bias_hh_l0_reverse"),
        };

        /// <summary>DurationEncoder: rows [n, hidden + style] (the last AdaLayerNorm's output with the style appended).</summary>
        private float[] DurationEncoder(float[] dEn, int n, float[] s)
        {
            int hid = _hidden, sd = _style, w = hid + sd;
            float[] x = AppendStyle(dEn, n, hid, s);
            for (int block = 0; ; block += 2)
            {
                string lstm = $"predictor.text_encoder.lstms.{block}";
                if (!Has(lstm + ".weight_ih_l0")) break;
                float[] h = O.BiLstm(x, n, w, Lstm(lstm));          // [n, hid]
                string ln = $"predictor.text_encoder.lstms.{block + 1}.fc";
                float[] gb = O.Linear(s, 1, sd, W(ln + ".weight"), 2 * hid, W(ln + ".bias"));
                O.LayerNormRows(h, n, hid, null, null, 1e-5f);
                for (int i = 0; i < n; i++)
                    for (int c = 0; c < hid; c++)
                        h[(long)i * hid + c] = (1f + gb[c]) * h[(long)i * hid + c] + gb[hid + c];
                x = AppendStyle(h, n, hid, s);
            }
            return x;
        }

        private static float[] AppendStyle(float[] rows, int n, int d, float[] s)
        {
            int w = d + s.Length;
            var y = new float[(long)n * w];
            for (int i = 0; i < n; i++)
            {
                Array.Copy(rows, (long)i * d, y, (long)i * w, d);
                Array.Copy(s, 0, y, (long)i * w + d, s.Length);
            }
            return y;
        }

        /// <summary>Repeat each token for its duration: [channels, frames]. The source is rows [n, c] when
        /// <paramref name="rowsAreTime"/>, else channel-major [c, n].</summary>
        private static float[] Align(float[] src, int a, int b, int[] durations, int frames, bool rowsAreTime)
        {
            int n = rowsAreTime ? a : b, c = rowsAreTime ? b : a;
            var y = new float[(long)c * frames];
            int f = 0;
            for (int i = 0; i < n; i++)
                for (int r = 0; r < durations[i]; r++, f++)
                    for (int ch = 0; ch < c; ch++)
                        y[(long)ch * frames + f] = rowsAreTime ? src[(long)i * c + ch] : src[(long)ch * n + i];
            return y;
        }

        private (float[] F0, float[] N) F0N(float[] en, int c, int frames, float[] s)
        {
            float[] rows = O.Transpose(en, c, frames);                      // [frames, c]
            float[] shared = O.BiLstm(rows, frames, c, Lstm("predictor.shared"));
            float[] x = O.Transpose(shared, frames, _hidden);               // [hidden, frames]
            float[] Branch(string name)
            {
                float[] y = x;
                int ch = _hidden, t = frames;
                for (int i = 0; i < 3; i++)
                    (y, ch, t) = AdainResBlk($"predictor.{name}.{i}", y, ch, t, s);
                float[] proj = O.Conv1d(y, ch, t, W($"predictor.{name}_proj.weight"), 1, 1, W($"predictor.{name}_proj.bias"));
                return proj;
            }
            float[] f0 = null, nn = null;
            Parallel.Invoke(() => f0 = Branch("F0"), () => nn = Branch("N"));
            return (f0, nn);
        }

        /// <summary>StyleTTS2 AdainResBlk1d: (residual + shortcut) / sqrt 2, optionally upsampling time x2.</summary>
        private (float[], int, int) AdainResBlk(string p, float[] x, int cin, int t, float[] s)
        {
            int cout = S(p + ".conv1.weight")[0];
            bool upsample = Has(p + ".pool.weight");
            float[] r = O.AdaIn(x, cin, t, W(p + ".norm1.fc.weight"), W(p + ".norm1.fc.bias"), s);
            O.LeakyRelu(r, 0.2f);
            int t2 = t;
            if (upsample)
            {
                r = O.DepthwiseConvTranspose1d(r, cin, t, W(p + ".pool.weight"), 3, W(p + ".pool.bias"), 2, 1, 1);
                t2 = 2 * t;
            }
            r = O.Conv1d(r, cin, t2, W(p + ".conv1.weight"), cout, 3, W(p + ".conv1.bias"), 1, 1);
            r = O.AdaIn(r, cout, t2, W(p + ".norm2.fc.weight"), W(p + ".norm2.fc.bias"), s);
            O.LeakyRelu(r, 0.2f);
            r = O.Conv1d(r, cout, t2, W(p + ".conv2.weight"), cout, 3, W(p + ".conv2.bias"), 1, 1);
            float[] sc = upsample ? O.UpsampleNearest(x, cin, t, 2) : x;
            if (Has(p + ".conv1x1.weight"))
                sc = O.Conv1d(sc, cin, t2, W(p + ".conv1x1.weight"), cout, 1, null);
            var y = new float[r.Length];
            float inv = 1f / MathF.Sqrt(2f);
            for (long i = 0; i < y.LongLength; i++) y[i] = (r[i] + sc[i]) * inv;
            return (y, cout, t2);
        }

        // ---------------------------------------------------------------- text encoder
        private float[] TextEncoder(int[] ids)
        {
            int n = ids.Length, c = _hidden;
            float[] emb = W("text_encoder.embedding.weight");
            var x = new float[(long)c * n];
            for (int i = 0; i < n; i++)
                for (int ch = 0; ch < c; ch++) x[(long)ch * n + i] = emb[(long)ids[i] * c + ch];
            for (int i = 0; ; i++)
            {
                string p = $"text_encoder.cnn.{i}";
                if (!Has(p + ".0.weight")) break;
                int k = S(p + ".0.weight")[2];
                x = O.Conv1d(x, c, n, W(p + ".0.weight"), c, k, W(p + ".0.bias"), 1, (k - 1) / 2);
                O.LayerNormChannels(x, c, n, W(p + ".1.gamma"), W(p + ".1.beta"));
                O.LeakyRelu(x, 0.2f);
            }
            float[] rows = O.Transpose(x, c, n);
            float[] l = O.BiLstm(rows, n, c, Lstm("text_encoder.lstm"));
            return O.Transpose(l, n, c);
        }

        // ---------------------------------------------------------------- decoder
        private float[] Decoder(float[] asr, int frames, float[] f0Curve, float[] nCurve, float[] s, Random rng)
        {
            int t2 = 2 * frames;
            float[] f0 = O.Conv1d(f0Curve, 1, t2, W("decoder.F0_conv.weight"), 1, 3, W("decoder.F0_conv.bias"), 2, 1);
            float[] nn = O.Conv1d(nCurve, 1, t2, W("decoder.N_conv.weight"), 1, 3, W("decoder.N_conv.bias"), 2, 1);
            int t = frames;
            float[] x = O.ConcatChannels(t, (asr, _hidden), (f0, 1), (nn, 1));
            int c;
            (x, c, t) = AdainResBlk("decoder.encode", x, _hidden + 2, t, s);
            int resCh = S("decoder.asr_res.0.weight")[0];
            float[] asrRes = O.Conv1d(asr, _hidden, frames, W("decoder.asr_res.0.weight"), resCh, 1, W("decoder.asr_res.0.bias"));
            bool res = true;
            for (int i = 0; Has($"decoder.decode.{i}.conv1.weight"); i++)
            {
                if (res)
                {
                    x = O.ConcatChannels(t, (x, c), (asrRes, resCh), (f0, 1), (nn, 1));
                    c += resCh + 2;
                }
                bool up = Has($"decoder.decode.{i}.pool.weight");
                (x, c, t) = AdainResBlk($"decoder.decode.{i}", x, c, t, s);
                if (up) res = false;
            }
            return Generator(x, c, t, s, f0Curve, rng);
        }

        private float[] Generator(float[] x, int c, int t, float[] s, float[] f0Curve, Random rng)
        {
            int upTotal = 1;
            foreach (int r in _upRates) upTotal *= r;
            int scale = upTotal * _hop;               // 300 samples per generator frame
            int tAudio = t * scale;

            // ---- harmonic source (SourceModuleHnNSF with SineGen, 8 harmonics) ----
            const int dim = 9;
            var f0Up = new float[tAudio];
            for (int i = 0; i < tAudio; i++) f0Up[i] = f0Curve[i / scale];
            // torch's SineGen arithmetic, float32 step for step (the reference's phase is float32, and at tens
            // of thousands of radians its rounding is audible in a sample comparison): rad = (f0 h / sr) % 1,
            // linear down to the frame rate, cumsum (double accumulation, float result), * 2 * pi * 300, linear
            // back up, sin.
            var rad = new float[(long)dim * tAudio];      // [dim, tAudio]
            for (int h = 0; h < dim; h++)
                for (int i = 0; i < tAudio; i++)
                {
                    float v = f0Up[i] * (h + 1) / SampleRate;
                    rad[(long)h * tAudio + i] = v - MathF.Floor(v);
                }
            if (!Deterministic)
                for (int h = 1; h < dim; h++) rad[(long)h * tAudio] += (float)rng.NextDouble();
            float[] radDown = InterpolateLinearF32(rad, dim, tAudio, scale, t);
            var phaseDown = new float[radDown.Length];
            for (int h = 0; h < dim; h++)
            {
                double acc = 0;
                for (int i = 0; i < t; i++)
                {
                    acc += radDown[(long)h * t + i];
                    phaseDown[(long)h * t + i] = (float)acc * 2f * (float)Math.PI * scale;
                }
            }
            float[] phase = InterpolateLinearF32(phaseDown, dim, t, 1f / scale, tAudio);
            float[] mergeW = W("decoder.generator.m_source.l_linear.weight");
            float mergeB = W("decoder.generator.m_source.l_linear.bias")[0];
            var har = new float[tAudio];
            int noiseSeed = rng.Next();
            Parallel.For(0, (tAudio + 8191) / 8192, blk =>
            {
                var local = Deterministic ? null : new Random(unchecked(noiseSeed * 31 + blk));
                int end = Math.Min(tAudio, (blk + 1) * 8192);
                for (int i = blk * 8192; i < end; i++)
                {
                    bool voiced = f0Up[i] > 10f;
                    float noiseAmp = voiced ? 0.003f : 0.1f / 3f;
                    float sum = 0f;
                    for (int h = 0; h < dim; h++)
                    {
                        float sine = MathF.Sin(phase[(long)h * tAudio + i]) * 0.1f;
                        float noise = local == null ? 0f : noiseAmp * Gaussian(local);
                        sum += mergeW[h] * ((voiced ? sine : 0f) + noise);
                    }
                    har[i] = MathF.Tanh(sum + mergeB);
                }
            });
            var (hMag, hPhase, hFrames) = O.Stft(har, _nFft, _hop);
            int bins = _nFft / 2 + 1;
            float[] harSpec = O.ConcatChannels(hFrames, (hMag, bins), (hPhase, bins));
            int harCh = 2 * bins;

            // ---- upsampling stages ----
            int nk = _rbKernels.Length;
            for (int i = 0; i < _upRates.Length; i++)
            {
                O.LeakyRelu(x, 0.1f);
                string nc = $"decoder.generator.noise_convs.{i}";
                int[] ncs = S(nc + ".weight");
                int ncOut = ncs[0], ncK = ncs[2];
                float[] xs;
                int tSrc;
                if (i + 1 < _upRates.Length)
                {
                    int strideF0 = 1;
                    for (int j = i + 1; j < _upRates.Length; j++) strideF0 *= _upRates[j];
                    xs = O.Conv1d(harSpec, harCh, hFrames, W(nc + ".weight"), ncOut, ncK, W(nc + ".bias"),
                        strideF0, (strideF0 + 1) / 2);
                    tSrc = (hFrames + 2 * ((strideF0 + 1) / 2) - ncK) / strideF0 + 1;
                }
                else
                {
                    xs = O.Conv1d(harSpec, harCh, hFrames, W(nc + ".weight"), ncOut, 1, W(nc + ".bias"));
                    tSrc = hFrames;
                }
                xs = AdaInResBlock1($"decoder.generator.noise_res.{i}", xs, ncOut, tSrc, s);

                string up = $"decoder.generator.ups.{i}";
                int cout = S(up + ".weight")[1];
                int k = _upKernels[i], u = _upRates[i];
                x = O.ConvTranspose1d(x, c, t, W(up + ".weight"), cout, k, W(up + ".bias"), u, (k - u) / 2);
                c = cout;
                t = (t - 1) * u - 2 * ((k - u) / 2) + k;
                if (i == _upRates.Length - 1)
                {
                    // ReflectionPad1d((1, 0))
                    var padded = new float[(long)c * (t + 1)];
                    for (int ch = 0; ch < c; ch++)
                    {
                        padded[(long)ch * (t + 1)] = x[(long)ch * t + 1];
                        Array.Copy(x, (long)ch * t, padded, (long)ch * (t + 1) + 1, t);
                    }
                    x = padded;
                    t += 1;
                }
                if (t != tSrc)
                    throw new InvalidOperationException($"Kokoro generator stage {i}: {t} frames against a {tSrc}-frame source.");
                O.AddInPlace(x, xs);
                float[] acc = null;
                for (int j = 0; j < nk; j++)
                {
                    float[] r = AdaInResBlock1($"decoder.generator.resblocks.{i * nk + j}", x, c, t, s);
                    if (acc == null) acc = r; else O.AddInPlace(acc, r);
                }
                O.Scale(acc, 1f / nk);
                x = acc;
            }
            O.LeakyRelu(x, 0.01f);
            int postOut = S("decoder.generator.conv_post.weight")[0];
            float[] post = O.Conv1d(x, c, t, W("decoder.generator.conv_post.weight"), postOut, 7,
                W("decoder.generator.conv_post.bias"), 1, 3);
            var mag = new float[(long)bins * t];
            var ph = new float[(long)bins * t];
            for (long i = 0; i < mag.LongLength; i++)
            {
                mag[i] = MathF.Exp(post[i]);
                ph[i] = MathF.Sin(post[(long)bins * t + i]);
            }
            return O.Istft(mag, ph, t, _nFft, _hop);
        }

        /// <summary>AdaINResBlock1: three (AdaIN, Snake, dilated conv, AdaIN, Snake, conv) residual units.</summary>
        private float[] AdaInResBlock1(string p, float[] x, int c, int t, float[] s)
        {
            int k = S(p + ".convs1.0.weight")[2];
            int[] dil = { 1, 3, 5 };
            for (int i = 0; i < 3; i++)
            {
                float[] xt = O.AdaIn(x, c, t, W($"{p}.adain1.{i}.fc.weight"), W($"{p}.adain1.{i}.fc.bias"), s);
                O.Snake(xt, c, t, W($"{p}.alpha1.{i}"));
                xt = O.Conv1d(xt, c, t, W($"{p}.convs1.{i}.weight"), c, k, W($"{p}.convs1.{i}.bias"), 1,
                    (k * dil[i] - dil[i]) / 2, dil[i]);
                xt = O.AdaIn(xt, c, t, W($"{p}.adain2.{i}.fc.weight"), W($"{p}.adain2.{i}.fc.bias"), s);
                O.Snake(xt, c, t, W($"{p}.alpha2.{i}"));
                xt = O.Conv1d(xt, c, t, W($"{p}.convs2.{i}.weight"), c, k, W($"{p}.convs2.{i}.bias"), 1, (k - 1) / 2);
                O.AddInPlace(xt, x);
                x = xt;
            }
            return x;
        }


        /// <summary>torch's upsample_linear1d (align_corners=False) in float32: src = scale * (dst + 0.5) - 0.5,
        /// clamped at 0, where scale is the reciprocal of the scale_factor.</summary>
        private static float[] InterpolateLinearF32(float[] x, int c, int t, float scale, int tOut)
        {
            var y = new float[(long)c * tOut];
            Parallel.For(0, c, ch =>
            {
                for (int i = 0; i < tOut; i++)
                {
                    float src = scale * (i + 0.5f) - 0.5f;
                    if (src < 0) src = 0;
                    int i0 = (int)src;
                    if (i0 > t - 1) i0 = t - 1;
                    int i1 = i0 < t - 1 ? i0 + 1 : i0;
                    float l1 = src - i0, l0 = 1f - l1;
                    y[(long)ch * tOut + i] = l0 * x[(long)ch * t + i0] + l1 * x[(long)ch * t + i1];
                }
            });
            return y;
        }

        private static float Gaussian(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
    }
}
