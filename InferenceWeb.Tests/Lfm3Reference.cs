// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace InferenceWeb.Tests;

/// <summary>
/// An independent LFM3 forward in double precision over a <see cref="Lfm3SyntheticModelBuilder"/>'s F32 weights, written from
/// llama.cpp's graph (src/models/lfm3.cpp, lfm-base.cpp) and nothing of TensorSharp's: unpermuted heads rotated by ggml's NeoX
/// rope with the rope_freqs factors on the full layers, llama.cpp's sliding-window mask (a key is masked when
/// p_query - p_key &gt;= window), the short convolution over its carried B*x products, the head-wise sigmoid gate, the
/// sandwich norms and the tied head. One sequence, token by token.
/// </summary>
internal sealed class Lfm3Reference
{
    private readonly Lfm3SyntheticModelBuilder _b;
    private readonly Dictionary<string, float[]> _w;
    private readonly List<double[]>[] _convColumns;   // [layer] the last taps - 1 products b*x, oldest first
    private readonly List<double[]>[] _keys, _values; // [layer] per position: [kvHeads * headDim]
    private int _position;

    public Lfm3Reference(Lfm3SyntheticModelBuilder builder)
    {
        _b = builder;
        _w = builder.Tensors().ToDictionary(t => t.Name, t => t.Data);
        _convColumns = new List<double[]>[builder.Layers];
        _keys = new List<double[]>[builder.Layers];
        _values = new List<double[]>[builder.Layers];
        Reset();
    }

    public void Reset()
    {
        _position = 0;
        for (int l = 0; l < _b.Layers; l++)
        {
            _convColumns[l] = Enumerable.Range(0, Lfm3SyntheticModelBuilder.ConvTaps - 1)
                .Select(_ => new double[Lfm3SyntheticModelBuilder.Hidden]).ToList();
            _keys[l] = new List<double[]>();
            _values[l] = new List<double[]>();
        }
    }

    /// <summary>The prompt's last logits.</summary>
    public float[] Prompt(int[] tokens)
    {
        float[] logits = null;
        foreach (int t in tokens) logits = Step(t);
        return logits;
    }

    public float[] Step(int token)
    {
        const int H = Lfm3SyntheticModelBuilder.Hidden;
        float[] embd = _w["token_embd.weight"];
        var h = new double[H];
        for (int i = 0; i < H; i++) h[i] = embd[token * H + i];
        for (int l = 0; l < _b.Layers; l++)
        {
            string p = $"blk.{l}.";
            double[] x = Norm(h, _w[p + "attn_norm.weight"]);
            double[] mixed = _b.IsConv(l) ? ShortConv(l, x) : Attention(l, x);
            mixed = Norm(mixed, _w[p + "post_attention_norm.weight"]);
            for (int i = 0; i < H; i++) h[i] += mixed[i];
            double[] f = Norm(h, _w[p + "ffn_norm.weight"]);
            double[] gate = MatVec(_w[p + "ffn_gate.weight"], f, Lfm3SyntheticModelBuilder.Ff);
            double[] up = MatVec(_w[p + "ffn_up.weight"], f, Lfm3SyntheticModelBuilder.Ff);
            for (int i = 0; i < gate.Length; i++) gate[i] = gate[i] / (1 + Math.Exp(-gate[i])) * up[i];
            double[] ffn = Norm(MatVec(_w[p + "ffn_down.weight"], gate, H), _w[p + "post_ffw_norm.weight"]);
            for (int i = 0; i < H; i++) h[i] += ffn[i];
        }
        double[] final = Norm(h, _w["output_norm.weight"]);
        float[] head = _w.TryGetValue("output.weight", out float[] untied) ? untied : embd;
        var logits = new float[Lfm3SyntheticModelBuilder.Vocab];
        for (int v = 0; v < logits.Length; v++)
        {
            double sum = 0;
            for (int i = 0; i < H; i++) sum += head[v * H + i] * final[i];
            logits[v] = (float)sum;
        }
        _position++;
        return logits;
    }

    /// <summary>[b | c | x] = in_proj · h; y = c * conv(b * x) over the carried products; out_proj · y.</summary>
    private double[] ShortConv(int l, double[] x)
    {
        const int H = Lfm3SyntheticModelBuilder.Hidden, taps = Lfm3SyntheticModelBuilder.ConvTaps;
        string p = $"blk.{l}.";
        double[] bcx = MatVec(_w[p + "shortconv.in_proj.weight"], x, 3 * H);
        var bx = new double[H];
        for (int c = 0; c < H; c++) bx[c] = bcx[c] * bcx[2 * H + c];
        List<double[]> columns = _convColumns[l];
        columns.Add(bx);   // [taps - 1 previous | this one]
        float[] kernel = _w[p + "shortconv.conv.weight"];   // ne [taps, H]: channel c's taps at c * taps
        var y = new double[H];
        for (int c = 0; c < H; c++)
        {
            double sum = 0;
            for (int k = 0; k < taps; k++) sum += columns[k][c] * kernel[c * taps + k];
            y[c] = bcx[H + c] * sum;
        }
        columns.RemoveAt(0);
        return MatVec(_w[p + "shortconv.out_proj.weight"], y, H);
    }

    private double[] Attention(int l, double[] x)
    {
        const int H = Lfm3SyntheticModelBuilder.Hidden, heads = Lfm3SyntheticModelBuilder.Heads, kvHeads = Lfm3SyntheticModelBuilder.KvHeads;
        const int d = Lfm3SyntheticModelBuilder.HeadDim;
        string p = $"blk.{l}.";
        double[] q = MatVec(_w[p + "attn_q.weight"], x, heads * d);
        double[] k = MatVec(_w[p + "attn_k.weight"], x, kvHeads * d);
        double[] v = MatVec(_w[p + "attn_v.weight"], x, kvHeads * d);
        bool sliding = _b.IsSliding(l);
        double freqBase = sliding ? Lfm3SyntheticModelBuilder.RopeBaseSliding : Lfm3SyntheticModelBuilder.RopeBase;
        float[] factors = sliding ? null : _w["rope_freqs.weight"];
        for (int hd = 0; hd < heads; hd++) RmsRope(q, hd * d, _w[p + "attn_q_norm.weight"], freqBase, factors);
        for (int hd = 0; hd < kvHeads; hd++) RmsRope(k, hd * d, _w[p + "attn_k_norm.weight"], freqBase, factors);
        _keys[l].Add(k);
        _values[l].Add(v);

        double[] gate = MatVec(_w[p + "attn_gate.weight"], x, heads);
        var o = new double[heads * d];
        double scale = 1.0 / Math.Sqrt(d);
        int group = heads / kvHeads;
        for (int hd = 0; hd < heads; hd++)
        {
            int kv = hd / group;
            var scores = new double[_keys[l].Count];
            double max = double.NegativeInfinity;
            for (int j = 0; j < scores.Length; j++)
            {
                if (sliding && _position - j >= Lfm3SyntheticModelBuilder.Window) { scores[j] = double.NegativeInfinity; continue; }
                double s = 0;
                for (int i = 0; i < d; i++) s += q[hd * d + i] * _keys[l][j][kv * d + i];
                scores[j] = s * scale;
                max = Math.Max(max, scores[j]);
            }
            double sum = 0;
            for (int j = 0; j < scores.Length; j++) sum += scores[j] = Math.Exp(scores[j] - max);
            double g = 1.0 / (1.0 + Math.Exp(-gate[hd]));
            for (int j = 0; j < scores.Length; j++)
                for (int i = 0; i < d; i++) o[hd * d + i] += scores[j] / sum * _values[l][j][kv * d + i] * g;
        }
        return MatVec(_w[p + "attn_output.weight"], o, H);
    }

    /// <summary>One head's RMSNorm (times its weight), then ggml's NeoX rope over the whole head at this position, each pair i
    /// turning by position * base^(-2i/d) / factor[i].</summary>
    private void RmsRope(double[] x, int offset, float[] norm, double freqBase, float[] factors)
    {
        const int d = Lfm3SyntheticModelBuilder.HeadDim;
        double ss = 0;
        for (int i = 0; i < d; i++) ss += x[offset + i] * x[offset + i];
        double inv = 1.0 / Math.Sqrt(ss / d + Lfm3SyntheticModelBuilder.Eps);
        for (int i = 0; i < d; i++) x[offset + i] *= inv * norm[i];
        for (int i = 0; i < d / 2; i++)
        {
            double theta = _position * Math.Pow(freqBase, -2.0 * i / d) / (factors?[i] ?? 1.0);
            double c = Math.Cos(theta), s = Math.Sin(theta);
            double x0 = x[offset + i], x1 = x[offset + i + d / 2];
            x[offset + i] = x0 * c - x1 * s;
            x[offset + i + d / 2] = x0 * s + x1 * c;
        }
    }

    private static double[] Norm(double[] x, float[] weight)
    {
        double ss = 0;
        foreach (double v in x) ss += v * v;
        double inv = 1.0 / Math.Sqrt(ss / x.Length + Lfm3SyntheticModelBuilder.Eps);
        var y = new double[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * inv * weight[i];
        return y;
    }

    /// <summary>W · x for a GGUF matrix of ne [x.Length, rows] (row r at r * x.Length).</summary>
    private static double[] MatVec(float[] w, double[] x, int rows)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int i = 0; i < x.Length; i++) sum += w[r * x.Length + i] * x[i];
            y[r] = sum;
        }
        return y;
    }
}
