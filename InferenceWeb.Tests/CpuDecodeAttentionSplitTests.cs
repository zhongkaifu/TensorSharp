namespace InferenceWeb.Tests;

/// <summary>
/// The split F16 decode attention (ModelBase.CpuAttentionSplit.cs: keys and head-dim slices spread over the workers) against
/// the one-pass kernel it replaces at long contexts, BIT FOR BIT.
/// </summary>
public class CpuDecodeAttentionSplitTests
{
    /// <summary>The split F16 decode attention (keys and head-dim slices spread over the workers) against the one-pass
    /// kernel, bit for bit: GQA groups of 4 and others, key counts off the chunk and 4-value boundaries, head dims with
    /// and without a partial slice, sharply peaked rows (flushed exp tails).</summary>
    [Theory]
    [InlineData(16, 4, 128, 130, 200)]
    [InlineData(16, 4, 128, 257, 768)]
    [InlineData(16, 4, 128, 1024, 1500)]
    [InlineData(16, 4, 128, 16383, 16400)]
    [InlineData(16, 4, 128, 3001, 3100)]
    [InlineData(8, 4, 64, 1027, 1030)]
    [InlineData(6, 3, 80, 2050, 2050)]
    [InlineData(4, 4, 128, 1025, 1100)]
    [InlineData(12, 2, 96, 4097, 4200)]
    public unsafe void SplitF16DecodeAttention_IsBitIdenticalToOnePass(int heads, int kvHeads, int headDim, int keys, int capacity)
    {
        if (!ModelBase.SplitDecodeAttentionSupported)
            return;   // the split kernel runs only where VecExpRow takes its 128-bit path
        var rng = new Random(heads * 31 + keys);
        float[] q = new float[heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 7 == 0 ? 6f : 1f);
        ushort[] k = new ushort[(long)kvHeads * capacity * headDim], v = new ushort[k.Length];
        for (int i = 0; i < k.Length; i++)
        {
            k[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextDouble() * 2 - 1) * (i % 1013 == 0 ? 8 : 1)));
            v[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2 - 1));
        }
        float[] expected = new float[heads * headDim], actual = new float[heads * headDim];
        float scale = 1f / MathF.Sqrt(headDim);
        fixed (float* qp = q)
        fixed (ushort* kp = k)
        fixed (ushort* vp = v)
        fixed (float* e = expected)
        fixed (float* a = actual)
        {
            ModelBase.AttentionDecodeGroupedF16OnePass(qp, kp, vp, e, heads, kvHeads, headDim, capacity, keys, scale);
            ModelBase.AttentionDecodeGroupedF16Split(qp, kp, vp, a, heads, kvHeads, headDim, capacity, keys, scale);
        }
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"heads={heads}/{kvHeads} d={headDim} keys={keys} [{i}]: one-pass {expected[i]:R} vs split {actual[i]:R}");
    }
}
