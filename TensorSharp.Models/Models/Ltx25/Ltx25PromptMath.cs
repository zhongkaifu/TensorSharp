using System;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.Ltx25;

internal static class Ltx25PromptMath
{
    internal static float Bf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7f800000) != 0x7f800000) bits += 0x7fff + ((bits >> 16) & 1);
        return BitConverter.UInt32BitsToSingle(bits & 0xffff0000);
    }

    internal static void RoundBf16(Tensor tensor)
    {
        if (MlxLtx25PromptOps.TryRoundBf16(tensor)) return;
        if (CudaLtx25PromptOps.TryRoundBf16(tensor)) return;
        float[] data = tensor.GetElementsAsFloat(checked((int)tensor.ElementCount()));
        for (int i = 0; i < data.Length; ++i) data[i] = Bf16(data[i]);
        tensor.SetElementsAsFloat(data);
    }

    internal static Tensor Norm(DirectContext context, Tensor input, Tensor gain, float epsilon, bool bf16)
    {
        var output = DirectOps.RmsNorm(context, input, gain ?? context.Ones(input.Sizes[1]), epsilon);
        if (bf16) RoundBf16(output);
        return output;
    }

    internal static void GemmaRope(Tensor input, int heads, int dim, int firstPosition, float theta, float proportion)
    {
        if (input == null || input.DimensionCount != 2 || heads <= 0 || dim <= 0 || dim % 2 != 0 ||
            input.Sizes[1] != (long)heads * dim || input.Sizes[0] is < 1 or > int.MaxValue ||
            !float.IsFinite(theta) || theta <= 0 || !float.IsFinite(proportion) || proportion < 0 || proportion > 1)
            throw new ArgumentException("Invalid Gemma rotary geometry.");
        int half = dim / 2, rotated = (int)(proportion * half);
        if (rotated == 0) return;
        int tokens = checked((int)input.Sizes[0]);
        var inverse = new float[rotated];
        for (int i = 0; i < rotated; ++i) inverse[i] = 1f / MathF.Pow(theta, 2f * i / dim);
        // MathF and BF16 rounding are part of the established conditioning
        // contract. Compute each angle once on the host and share it across
        // heads; device kernels only apply the original rounded arithmetic.
        var cosine = new float[checked(tokens * rotated)];
        var sine = new float[cosine.Length];
        for (int token = 0; token < tokens; ++token)
        for (int i = 0; i < rotated; ++i)
        {
            float angle = (firstPosition + token) * inverse[i];
            cosine[token * rotated + i] = Bf16(MathF.Cos(angle));
            sine[token * rotated + i] = Bf16(MathF.Sin(angle));
        }
        if (input.Allocator is CudaAllocator or MlxAllocator)
        {
            using var cos = Tensor.FromArray(input.Allocator, cosine);
            using var sin = Tensor.FromArray(input.Allocator, sine);
            if (CudaLtx25PromptOps.TryGemmaRope(input, cos, sin, heads, dim, rotated) ||
                MlxLtx25PromptOps.TryGemmaRope(input, cos, sin, heads, dim, rotated)) return;
        }
        var data = input.GetElementsAsFloat(checked((int)input.ElementCount()));
        for (int token = 0; token < tokens; ++token)
        for (int head = 0; head < heads; ++head)
        for (int i = 0; i < rotated; ++i)
        {
            float c = cosine[token * rotated + i], s = sine[token * rotated + i];
            int a = (token * heads + head) * dim + i, b = a + half;
            float x = data[a], y = data[b];
            data[a] = Bf16(Bf16(x * c) - Bf16(y * s));
            data[b] = Bf16(Bf16(y * c) + Bf16(x * s));
        }
        input.SetElementsAsFloat(data);
    }

    internal static Tensor RepeatKv(DirectContext context, Tensor source, int queryHeads, int kvHeads, int head)
    {
        using var shaped = source.View(source.Sizes[0], kvHeads, 1, head);
        using var expanded = shaped.Expand(source.Sizes[0], kvHeads, queryHeads / kvHeads, head);
        using var contiguous = Ops.NewContiguous(expanded);
        return contiguous.View(source.Sizes[0], (long)queryHeads * head);
    }
}
