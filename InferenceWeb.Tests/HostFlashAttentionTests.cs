using TensorSharp.Models.Graph;

namespace InferenceWeb.Tests;

/// <summary>
/// The managed executor's tiled attention for several query rows (HostFlashAttention, ARM64) against a double-precision
/// softmax reference: causal masks aligned to the last key with and without a floor, no mask, and an additive band mask
/// (a sliding window), over F16 and F32 caches, GQA groups of 1, 2 and 4, head dims with and without a padded tail, and
/// row and key counts off the kernel's 16-row and 64-key blocks.
/// </summary>
public class HostFlashAttentionTests
{
    [Theory]
    [InlineData(16, 4, 128, 37, 37, 0, "causal", true)]
    [InlineData(16, 4, 128, 512, 1100, 0, "causal", true)]
    [InlineData(16, 4, 128, 33, 300, 100, "causal", true)]
    [InlineData(8, 4, 64, 20, 129, 0, "none", true)]
    [InlineData(6, 3, 80, 17, 70, 5, "causal", false)]
    [InlineData(4, 4, 128, 9, 65, 0, "band", true)]
    [InlineData(16, 4, 128, 70, 768, 0, "band", false)]
    [InlineData(16, 4, 128, 1, 300, 44, "causal", true)]
    public unsafe void MatchesTheSoftmaxReference(int heads, int kvHeads, int d, int nq, int valid, int floor, string mode, bool half)
    {
        if (!HostFlashAttention.Supported) return;
        var rng = new Random(heads + nq * 7 + valid * 13 + d);
        int capacity = valid + 11;
        float[] q = new float[heads * nq * d];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 11 == 0 ? 4f : 1f);
        float[] kf = new float[kvHeads * capacity * d], vf = new float[kf.Length];
        ushort[] kh = new ushort[kf.Length], vh = new ushort[kf.Length];
        for (int i = 0; i < kf.Length; i++)
        {
            kh[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2 - 1));
            vh[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2 - 1));
            kf[i] = (float)BitConverter.UInt16BitsToHalf(kh[i]);
            vf[i] = (float)BitConverter.UInt16BitsToHalf(vh[i]);
        }
        const int window = 24;
        float[] mask = new float[nq * valid];
        for (int r = 0; r < nq; r++)
            for (int j = 0; j < valid; j++)
            {
                int pos = valid - nq + r;   // the row's position: rows end at the last key
                mask[r * valid + j] = j <= pos && j > pos - window ? 0f : float.NegativeInfinity;
            }
        bool Visible(int r, int j) => mode switch
        {
            "causal" => j >= floor && j < Math.Clamp(valid - nq + r + 1, floor, valid),
            "none" => j >= floor && j < valid,
            _ => !float.IsNegativeInfinity(mask[r * valid + j]),
        };
        float scale = 1f / MathF.Sqrt(d);
        int group = heads / kvHeads;
        float[] expected = new float[nq * heads * d];
        for (int h = 0; h < heads; h++)
            for (int r = 0; r < nq; r++)
            {
                int kvh = h / group;
                double[] s = new double[valid];
                double max = double.NegativeInfinity;
                for (int j = 0; j < valid; j++)
                {
                    if (!Visible(r, j)) { s[j] = double.NegativeInfinity; continue; }
                    double dot = 0;
                    for (int c = 0; c < d; c++) dot += (double)q[(h * nq + r) * d + c] * kf[((long)kvh * capacity + j) * d + c];
                    s[j] = dot * scale;
                    max = Math.Max(max, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < valid; j++) { s[j] = double.IsNegativeInfinity(s[j]) ? 0 : Math.Exp(s[j] - max); sum += s[j]; }
                for (int c = 0; c < d; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < valid; j++) acc += s[j] * vf[((long)kvh * capacity + j) * d + c];
                    expected[(r * heads + h) * d + c] = (float)(acc / sum);
                }
            }

        float[] actual = new float[expected.Length];
        var kind = mode switch
        {
            "causal" => HostFlashAttention.MaskKind.Causal,
            "none" => HostFlashAttention.MaskKind.None,
            _ => HostFlashAttention.MaskKind.Additive,
        };
        fixed (float* qp = q)
        fixed (float* kfp = kf)
        fixed (float* vfp = vf)
        fixed (ushort* khp = kh)
        fixed (ushort* vhp = vh)
        fixed (float* mp = mask)
        fixed (float* o = actual)
        {
            HostFlashAttention.Run(qp, (long)nq * d, half ? khp : kfp, half ? vhp : vfp, (long)capacity * d, half,
                heads, kvHeads, nq, d, scale, kind, kind == HostFlashAttention.MaskKind.Additive ? 0 : floor, valid,
                mp, valid, false, o);
        }
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        Assert.True(worst <= 2e-5f, $"{mode} heads={heads}/{kvHeads} d={d} nq={nq} keys={valid} floor={floor}: max |diff| {worst:E2}");
    }
}
