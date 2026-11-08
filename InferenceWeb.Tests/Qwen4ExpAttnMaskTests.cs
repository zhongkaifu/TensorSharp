// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// qwen4exp builds a prompt chunk's causal mask on the host. Filled cell by cell it took 85 / 168 / 253 / 341 ms
/// for the four 4096-token chunks of a 16K-token prompt; text (identity QSA positions) now fills each row as two
/// runs, on several threads for a prompt-sized mask. The runs must equal the cell rule exactly.
/// </summary>
public sealed class Qwen4ExpAttnMaskTests
{
    private static ushort[] Reference(int seqLen, int stride, int startPos, int totalLen)
    {
        var m = new ushort[seqLen * stride];
        for (int t = 0; t < seqLen; t++)
            for (int j = 0; j < stride; j++)
                m[t * stride + j] = j < totalLen && j <= startPos + t ? (ushort)0 : (ushort)0xFC00;
        return m;
    }

    [Theory]
    [InlineData(1, 256, 37, 38)]          // decode-like: one row
    [InlineData(32, 256, 0, 32)]          // short prompt from the start
    [InlineData(100, 512, 300, 400)]      // continued prompt, padded past the context
    [InlineData(4096, 8192, 4000, 8096)]  // prompt-sized: the multithreaded fill
    public void TheRunFill_EqualsTheCellRule(int seqLen, int stride, int startPos, int totalLen)
    {
        var fast = new ushort[seqLen * stride];
        Array.Fill(fast, (ushort)0x1234);   // stale contents from an earlier, larger mask
        Qwen4ExpModel.FillCausalMask(fast, seqLen, stride, startPos, totalLen);
        Assert.True(fast.AsSpan().SequenceEqual(Reference(seqLen, stride, startPos, totalLen)));
    }

    [Fact]
    public void OnlyTextPositions_AreTheIdentity()
    {
        int[] text = new int[3 * 10];
        Qwen4ExpModel.WriteQsaPositions(text, 0, 10, 0, null);
        Assert.True(Qwen4ExpModel.QsaPositionsAreIdentity(text, 10));

        int[] gapped = new int[3 * 10];
        Qwen4ExpModel.WriteQsaPositions(gapped, 0, 4, 0, null);
        Qwen4ExpModel.WriteQsaPositions(gapped, 4, 6, 9, null);   // a rotary gap after media
        Assert.False(Qwen4ExpModel.QsaPositionsAreIdentity(gapped, 10));
        Assert.True(Qwen4ExpModel.QsaPositionsAreIdentity(gapped, 4));

        int[] media = new int[3 * 4];
        Qwen4ExpModel.WriteQsaPositions(media, 0, 4, 0, new[] { 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1 });
        Assert.False(Qwen4ExpModel.QsaPositionsAreIdentity(media, 4));
        Assert.False(Qwen4ExpModel.QsaPositionsAreIdentity(null, 4));
    }
}
