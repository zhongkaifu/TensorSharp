using TensorSharp.Models;
using TensorSharp.GGML;
using TensorSharp;
using System.Runtime.InteropServices;

namespace InferenceWeb.Tests;

public class NemotronConvTests
{
    [Fact]
    public void Mamba2Conv1dStepVectorized_MatchesScalarReference()
    {
        const int xBCSize = 19;
        const int convDim = 3;
        int dConv = convDim + 1;

        float[] x = Enumerable.Range(0, xBCSize)
            .Select(i => (i - 7) * 0.13f)
            .ToArray();
        float[] state = Enumerable.Range(0, convDim * xBCSize)
            .Select(i => ((i * 17) % 23 - 11) * 0.07f)
            .ToArray();
        float[] rowMajorWeight = Enumerable.Range(0, xBCSize * dConv)
            .Select(i => ((i * 5) % 29 - 14) * 0.031f)
            .ToArray();
        float[] bias = Enumerable.Range(0, xBCSize)
            .Select(i => (i % 5 - 2) * 0.11f)
            .ToArray();

        float[] transposed = new float[rowMajorWeight.Length];
        for (int ch = 0; ch < xBCSize; ch++)
            for (int k = 0; k < dConv; k++)
                transposed[k * xBCSize + ch] = rowMajorWeight[ch * dConv + k];

        float[] expected = new float[xBCSize];
        for (int ch = 0; ch < xBCSize; ch++)
        {
            float sum = bias[ch];
            for (int k = 0; k < convDim; k++)
                sum += state[k * xBCSize + ch] * rowMajorWeight[ch * dConv + k];
            sum += x[ch] * rowMajorWeight[ch * dConv + convDim];
            expected[ch] = sum;
        }

        float[] actual = new float[xBCSize];
        NemotronModel.Mamba2Conv1dStepVectorizedForTest(x, state, convDim, transposed, bias, actual);

        for (int i = 0; i < xBCSize; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-6f, $"index {i}: expected {expected[i]}, actual {actual[i]}");
    }

    [Fact]
    public void NativeMamba2Prefill_MatchesManagedReferenceOnCpu()
    {
        const int seqLen = 5;
        const int dInner = 4;
        const int dState = 3;
        const int nHead = 2;
        const int headDim = 2;
        const int nGroup = 1;
        const int dConv = 3;
        const float eps = 1e-5f;

        int convDim = dConv - 1;
        int xBCSize = dInner + 2 * nGroup * dState;
        int dInProjTotal = 2 * dInner + 2 * nGroup * dState + nHead;

        float[] projectedData = Enumerable.Range(0, seqLen * dInProjTotal)
            .Select(i => ((i * 17) % 37 - 18) * 0.023f)
            .ToArray();
        float[] convStateInitial = Enumerable.Range(0, convDim * xBCSize)
            .Select(i => ((i * 11) % 29 - 14) * 0.017f)
            .ToArray();
        float[] ssmStateInitial = Enumerable.Range(0, dState * headDim * nHead)
            .Select(i => ((i * 7) % 31 - 15) * 0.011f)
            .ToArray();
        float[] convWeight = Enumerable.Range(0, xBCSize * dConv)
            .Select(i => ((i * 5) % 23 - 11) * 0.019f)
            .ToArray();
        float[] convBias = Enumerable.Range(0, xBCSize)
            .Select(i => (i % 5 - 2) * 0.013f)
            .ToArray();
        float[] dtBias = { -0.2f, 0.11f };
        float[] a = { -0.35f, -0.27f };
        float[] d = { 0.08f, -0.05f };
        float[] norm = Enumerable.Range(0, dInner)
            .Select(i => 0.8f + i * 0.07f)
            .ToArray();

        float[] expected = ManagedMamba2Reference(
            projectedData, convStateInitial.ToArray(), ssmStateInitial.ToArray(),
            convWeight, convBias, dtBias, a, d, norm,
            seqLen, dInner, dState, nHead, headDim, nGroup, dConv, eps,
            out float[] expectedConvState, out float[] expectedSsmState);

        float[] actualConvState = convStateInitial.ToArray();
        float[] actualSsmState = ssmStateInitial.ToArray();

        IntPtr convWeightPtr = AllocAndCopy(convWeight);
        IntPtr convBiasPtr = AllocAndCopy(convBias);
        IntPtr dtBiasPtr = AllocAndCopy(dtBias);
        IntPtr aPtr = AllocAndCopy(a);
        IntPtr dPtr = AllocAndCopy(d);
        IntPtr normPtr = AllocAndCopy(norm);

        try
        {
            var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
            var allocator = new GgmlAllocator(context, 0);
            using var projected = new Tensor(allocator, DType.Float32, seqLen, dInProjTotal);
            using var output = new Tensor(allocator, DType.Float32, seqLen, dInner);
            projected.SetElementsAsFloat(projectedData);

            GgmlBasicOps.NemotronMamba2Prefill(
                projected, output,
                actualConvState, actualSsmState,
                convWeightPtr, convBiasPtr, dtBiasPtr, aPtr, dPtr, normPtr,
                dInner, dState, nHead, headDim, nGroup, dConv, eps);

            AssertClose(expected, output.GetElementsAsFloat(expected.Length), 1e-4f);
            AssertClose(expectedConvState, actualConvState, 1e-6f);
            AssertClose(expectedSsmState, actualSsmState, 1e-4f);
        }
        finally
        {
            GgmlBasicOps.AlignedFree(convWeightPtr);
            GgmlBasicOps.AlignedFree(convBiasPtr);
            GgmlBasicOps.AlignedFree(dtBiasPtr);
            GgmlBasicOps.AlignedFree(aPtr);
            GgmlBasicOps.AlignedFree(dPtr);
            GgmlBasicOps.AlignedFree(normPtr);
        }
    }

    [Fact]
    public void NativeMamba2Decode_PersistentStateMatchesManagedReferenceOnCpu()
    {
        const int seqLen = 2;
        const int dInner = 4;
        const int dState = 3;
        const int nHead = 2;
        const int headDim = 2;
        const int nGroup = 1;
        const int dConv = 3;
        const float eps = 1e-5f;
        const ulong modelKey = 0xC0DE0001UL;
        ulong stateKey = (modelKey << 32) | 11UL;

        int convDim = dConv - 1;
        int xBCSize = dInner + 2 * nGroup * dState;
        int dInProjTotal = 2 * dInner + 2 * nGroup * dState + nHead;

        float[] projectedData = Enumerable.Range(0, seqLen * dInProjTotal)
            .Select(i => ((i * 19) % 41 - 20) * 0.021f)
            .ToArray();
        float[] convStateInitial = Enumerable.Range(0, convDim * xBCSize)
            .Select(i => ((i * 13) % 31 - 15) * 0.015f)
            .ToArray();
        float[] ssmStateInitial = Enumerable.Range(0, dState * headDim * nHead)
            .Select(i => ((i * 9) % 37 - 18) * 0.009f)
            .ToArray();
        float[] convWeight = Enumerable.Range(0, xBCSize * dConv)
            .Select(i => ((i * 7) % 29 - 14) * 0.017f)
            .ToArray();
        float[] convBias = Enumerable.Range(0, xBCSize)
            .Select(i => (i % 7 - 3) * 0.011f)
            .ToArray();
        float[] dtBias = { -0.18f, 0.09f };
        float[] a = { -0.31f, -0.23f };
        float[] d = { 0.06f, -0.04f };
        float[] norm = Enumerable.Range(0, dInner)
            .Select(i => 0.75f + i * 0.05f)
            .ToArray();

        float[] expected = ManagedMamba2Reference(
            projectedData, convStateInitial.ToArray(), ssmStateInitial.ToArray(),
            convWeight, convBias, dtBias, a, d, norm,
            seqLen, dInner, dState, nHead, headDim, nGroup, dConv, eps,
            out float[] expectedConvState, out float[] expectedSsmState);

        float[] actual = new float[expected.Length];
        float[] actualConvState = convStateInitial.ToArray();
        float[] actualSsmState = ssmStateInitial.ToArray();

        IntPtr convWeightPtr = AllocAndCopy(convWeight);
        IntPtr convBiasPtr = AllocAndCopy(convBias);
        IntPtr dtBiasPtr = AllocAndCopy(dtBias);
        IntPtr aPtr = AllocAndCopy(a);
        IntPtr dPtr = AllocAndCopy(d);
        IntPtr normPtr = AllocAndCopy(norm);

        try
        {
            GgmlBasicOps.NemotronMamba2DecodeClear(modelKey);
            var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
            var allocator = new GgmlAllocator(context, 0);
            using var projected = new Tensor(allocator, DType.Float32, 1, dInProjTotal);
            using var output = new Tensor(allocator, DType.Float32, 1, dInner);

            for (int s = 0; s < seqLen; s++)
            {
                projected.SetElementsAsFloat(projectedData.Skip(s * dInProjTotal).Take(dInProjTotal).ToArray());

                GgmlBasicOps.NemotronMamba2Decode(
                    stateKey, projected, output,
                    actualConvState, actualSsmState,
                    initializeState: s == 0,
                    downloadState: s == seqLen - 1,
                    convWeightPtr, convBiasPtr, dtBiasPtr, aPtr, dPtr, normPtr,
                    dInner, dState, nHead, headDim, nGroup, dConv, eps);

                Array.Copy(output.GetElementsAsFloat(dInner), 0, actual, s * dInner, dInner);
            }

            AssertClose(expected, actual, 1e-4f);
            AssertClose(expectedConvState, actualConvState, 1e-6f);
            AssertClose(expectedSsmState, actualSsmState, 1e-4f);
        }
        finally
        {
            GgmlBasicOps.NemotronMamba2DecodeClear(modelKey);
            GgmlBasicOps.AlignedFree(convWeightPtr);
            GgmlBasicOps.AlignedFree(convBiasPtr);
            GgmlBasicOps.AlignedFree(dtBiasPtr);
            GgmlBasicOps.AlignedFree(aPtr);
            GgmlBasicOps.AlignedFree(dPtr);
            GgmlBasicOps.AlignedFree(normPtr);
        }
    }

    private static IntPtr AllocAndCopy(float[] values)
    {
        IntPtr ptr = GgmlBasicOps.AlignedAlloc(values.Length * sizeof(float));
        Marshal.Copy(values, 0, ptr, values.Length);
        return ptr;
    }

    private static float[] ManagedMamba2Reference(
        float[] projected,
        float[] convState,
        float[] ssmState,
        float[] convWeight,
        float[] convBias,
        float[] dtBias,
        float[] a,
        float[] d,
        float[] norm,
        int seqLen,
        int dInner,
        int dState,
        int nHead,
        int headDim,
        int nGroup,
        int dConv,
        float eps,
        out float[] finalConvState,
        out float[] finalSsmState)
    {
        int convDim = dConv - 1;
        int xBCSize = dInner + 2 * nGroup * dState;
        int dInProjTotal = 2 * dInner + 2 * nGroup * dState + nHead;
        int headsPerGroup = nHead / nGroup;
        int statePerHead = dState * headDim;
        float[] output = new float[seqLen * dInner];
        float[] convOut = new float[xBCSize];
        float[] y = new float[dInner];

        for (int s = 0; s < seqLen; s++)
        {
            int row = s * dInProjTotal;
            int xbcOffset = row + dInner;
            int dtOffset = row + 2 * dInner + 2 * nGroup * dState;

            for (int ch = 0; ch < xBCSize; ch++)
            {
                float sum = convBias[ch];
                for (int k = 0; k < convDim; k++)
                    sum += convState[k * xBCSize + ch] * convWeight[ch * dConv + k];
                sum += projected[xbcOffset + ch] * convWeight[ch * dConv + convDim];
                convOut[ch] = SiLU(sum);
            }

            if (convDim > 1)
                Array.Copy(convState, xBCSize, convState, 0, (convDim - 1) * xBCSize);
            if (convDim > 0)
                Array.Copy(projected, xbcOffset, convState, (convDim - 1) * xBCSize, xBCSize);

            Array.Clear(y);
            for (int h = 0; h < nHead; h++)
            {
                float dt = Softplus(projected[dtOffset + h] + dtBias[h]);
                float da = MathF.Exp(dt * a[h]);
                int group = h / headsPerGroup;
                int stateBase = h * statePerHead;
                int xBase = h * headDim;
                int bBase = dInner + group * dState;
                int cBase = dInner + nGroup * dState + group * dState;

                for (int hd = 0; hd < headDim; hd++)
                {
                    float sum = 0;
                    int stateOffset = stateBase + hd * dState;
                    float xdt = convOut[xBase + hd] * dt;
                    for (int si = 0; si < dState; si++)
                    {
                        float state = ssmState[stateOffset + si] * da + convOut[bBase + si] * xdt;
                        ssmState[stateOffset + si] = state;
                        sum += state * convOut[cBase + si];
                    }

                    y[xBase + hd] = sum + d[h] * convOut[xBase + hd];
                }
            }

            for (int i = 0; i < dInner; i++)
                y[i] *= SiLU(projected[row + i]);

            int innerPerGroup = dInner / nGroup;
            for (int g = 0; g < nGroup; g++)
            {
                int offset = g * innerPerGroup;
                float sumSq = 0;
                for (int i = 0; i < innerPerGroup; i++)
                    sumSq += y[offset + i] * y[offset + i];
                float inv = 1.0f / MathF.Sqrt(sumSq / innerPerGroup + eps);
                for (int i = 0; i < innerPerGroup; i++)
                    output[s * dInner + offset + i] = y[offset + i] * inv * norm[offset + i];
            }
        }

        finalConvState = convState;
        finalSsmState = ssmState;
        return output;
    }

    private static float SiLU(float x) => x / (1.0f + MathF.Exp(-x));

    private static float Softplus(float x) =>
        x > 20.0f ? x : MathF.Log(1.0f + MathF.Exp(x));

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"index {i}: expected {expected[i]}, actual {actual[i]}");
    }
}
