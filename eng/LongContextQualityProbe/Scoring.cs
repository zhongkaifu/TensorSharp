// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// One row's next-token distribution, scored: the target's negative log-likelihood, the argmax, the top-2 logit gap, the K most
// likely tokens' log-probabilities (a reference for another arm, written into the record) and, against a reference's top K,
// the KL divergence of the partition {each of the reference's top K tokens, the rest} (a lower bound on the full KL that is
// tight when the top K hold most of the mass).
using System.Text.Json.Nodes;

namespace LongContextQualityProbe;

internal sealed class RowScores
{
    public readonly int K;
    public readonly List<double> Nll = new(), Kl = new(), Gap = new();
    public readonly List<int> Top1 = new();
    public readonly List<int> TopIds = new();
    public readonly List<float> TopLogProbs = new();

    public RowScores(int k) => K = k;

    public int Count => Nll.Count;

    /// <summary>Score <paramref name="logits"/> (one row, the vocabulary wide) for <paramref name="target"/> (-1: none), against
    /// <paramref name="reference"/>'s row <see cref="Count"/> when given.</summary>
    public void Add(ReadOnlySpan<float> logits, int target, TopK reference) => Add(Score(logits, target, K, reference, Count));

    public void Add(Row row)
    {
        Nll.Add(row.Nll);
        Top1.Add(row.Top1);
        Gap.Add(row.Gap);
        if (!double.IsNaN(row.Kl)) Kl.Add(row.Kl);
        TopIds.AddRange(row.Ids);
        TopLogProbs.AddRange(row.LogProbs);
    }

    public readonly struct Row
    {
        public readonly double Nll, Kl, Gap;
        public readonly int Top1;
        public readonly int[] Ids;
        public readonly float[] LogProbs;

        public Row(double nll, int top1, double gap, double kl, int[] ids, float[] logProbs)
        {
            Nll = nll; Top1 = top1; Gap = gap; Kl = kl; Ids = ids; LogProbs = logProbs;
        }
    }

    public static Row Score(ReadOnlySpan<float> logits, int target, int k, TopK reference, int position)
    {
        int vocab = logits.Length;
        float max = float.NegativeInfinity, second = float.NegativeInfinity;
        int top1 = 0;
        for (int i = 0; i < vocab; i++)
        {
            float v = logits[i];
            if (v > max) { second = max; max = v; top1 = i; }
            else if (v > second) second = v;
        }
        double sum = 0;
        for (int i = 0; i < vocab; i++) sum += Math.Exp(logits[i] - max);
        double lse = max + Math.Log(sum);
        double nll = target >= 0 ? lse - logits[target] : double.NaN;

        // The k largest logits (a small sorted buffer; replacements are rare past the first rows of the vocabulary).
        var ids = new int[k];
        var vals = new float[k];
        int filled = 0;
        for (int i = 0; i < vocab && k > 0; i++)
        {
            float v = logits[i];
            if (filled == k && v <= vals[k - 1]) continue;
            int at = filled < k ? filled++ : k - 1;
            while (at > 0 && vals[at - 1] < v) { vals[at] = vals[at - 1]; ids[at] = ids[at - 1]; at--; }
            vals[at] = v; ids[at] = i;
        }
        var lps = new float[k];
        for (int j = 0; j < k; j++) lps[j] = (float)(vals[j] - lse);

        double kl = double.NaN;
        if (reference != null && position < reference.Rows)
        {
            double pMass = 0, qMass = 0, sumKl = 0;
            for (int j = 0; j < reference.K; j++)
            {
                int id = reference.Ids[position * reference.K + j];
                double lp = reference.LogProbs[position * reference.K + j], lq = logits[id] - lse;
                double p = Math.Exp(lp);
                pMass += p;
                qMass += Math.Exp(lq);
                sumKl += p * (lp - lq);
            }
            double pTail = 1 - pMass, qTail = Math.Max(1 - qMass, 1e-30);
            if (pTail > 1e-12) sumKl += pTail * (Math.Log(pTail) - Math.Log(qTail));
            kl = Math.Max(0, sumKl);
        }
        return new Row(nll, top1, max - second, kl, ids, lps);
    }

    public JsonObject ToJson(bool withTopK)
    {
        var o = new JsonObject
        {
            ["nll"] = Floats(Nll),
            ["top1"] = new JsonArray(Top1.Select(t => (JsonNode)t).ToArray()),
        };
        if (Kl.Count > 0) o["kl"] = Floats(Kl);
        if (withTopK && K > 0)
        {
            o["topk_k"] = K;
            o["topk"] = TopK.Encode(TopIds, TopLogProbs);
        }
        return o;
    }

    public static JsonArray Floats(IEnumerable<double> values)
        => new(values.Select(v => double.IsFinite(v) ? (JsonNode)Math.Round(v, 6) : null).ToArray());
}

/// <summary>A record's top-K tokens a row (ids, then log-probabilities, as little-endian int32 / float32 in base64), with the
/// record's input (the sha256 of its token ids) and the sha256 of its topk string, which names the distribution a KL was
/// computed against.</summary>
internal sealed class TopK
{
    public int K, Rows;
    public int[] Ids;
    public float[] LogProbs;
    public string InputSha256, Digest;

    public static string Encode(List<int> ids, List<float> logProbs)
    {
        var bytes = new byte[(ids.Count + logProbs.Count) * 4];
        Buffer.BlockCopy(ids.ToArray(), 0, bytes, 0, ids.Count * 4);
        Buffer.BlockCopy(logProbs.ToArray(), 0, bytes, ids.Count * 4, logProbs.Count * 4);
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("big-endian host");
        return Convert.ToBase64String(bytes);
    }

    public static TopK Decode(JsonNode record)
    {
        int k = record["topk_k"]?.GetValue<int>() ?? 0;
        string data = record["topk"]?.GetValue<string>();
        if (k <= 0 || data == null) return null;
        byte[] bytes = Convert.FromBase64String(data);
        int rows = bytes.Length / (8 * k);
        var t = new TopK
        {
            K = k, Rows = rows, Ids = new int[rows * k], LogProbs = new float[rows * k],
            InputSha256 = record["input_sha256"]?.GetValue<string>(),
            Digest = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(data))).ToLowerInvariant(),
        };
        Buffer.BlockCopy(bytes, 0, t.Ids, 0, rows * k * 4);
        Buffer.BlockCopy(bytes, rows * k * 4, t.LogProbs, 0, rows * k * 4);
        return t;
    }
}
