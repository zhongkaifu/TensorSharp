// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;

namespace TensorSharp.Models.MiniMaxH3;

/// <summary>Device operations for H3's partial NeoX rotary tables, causal GQA,
/// interleaved VAE projections, and exact-form vision GELU. Hidden states retain
/// F32 storage, including the text encoder's unnormalized output.</summary>
internal sealed class MiniMaxH3TensorOps : IDisposable
{
    private readonly DirectContext _context;
    private float[] _lastCos, _lastSin;
    private int _lastRotary, _lastSequence;
    private Tensor _cos, _sin;
    internal bool IsAccelerated => _context.Allocator is CudaAllocator or MlxAllocator;

    internal MiniMaxH3TensorOps(DirectContext context) => _context = context;

    internal void RopeHalf(Tensor values, float[] cos, float[] sin, int sequence, int heads, int head, int rotary)
    {
        if (rotary == 0) return;
        if (rotary < 0 || rotary > head || rotary % 2 != 0 ||
            cos.Length < checked(sequence * rotary) || sin.Length != cos.Length)
            throw new ArgumentException("Invalid H3 rotary geometry or tables.");
        int half = rotary / 2;
        if (!ReferenceEquals(cos, _lastCos) || !ReferenceEquals(sin, _lastSin) || rotary != _lastRotary || sequence != _lastSequence)
        {
            _cos?.Dispose(); _sin?.Dispose();
            _cos = _sin = null;
            var compactCos = new float[checked(sequence * half)];
            var compactSin = new float[compactCos.Length];
            for (int row = 0; row < sequence; ++row)
            {
                Array.Copy(cos, row * rotary, compactCos, row * half, half);
                Array.Copy(sin, row * rotary, compactSin, row * half, half);
                for (int channel = 0; channel < half; ++channel)
                    if (cos[row * rotary + channel] != cos[row * rotary + half + channel] ||
                        sin[row * rotary + channel] != sin[row * rotary + half + channel])
                        throw new ArgumentException("H3 NeoX rotary tables must duplicate their two halves.");
            }
            _cos = _context.FromFloats(compactCos, compactCos.Length);
            _sin = _context.FromFloats(compactSin, compactSin.Length);
            _lastCos = cos; _lastSin = sin; _lastRotary = rotary; _lastSequence = sequence;
        }
        bool success = _context.IsCuda
            ? CudaFusedOps.TryNeoxRopeFlat(values, _cos, _sin, heads, sequence, head, half)
            : MlxFusedOps.TryNeoXRoPEFlatInPlace(values, _cos, _sin, heads, sequence, head, half);
        if (!success) throw new NotSupportedException("The selected H3 backend cannot execute its F32 partial NeoX rotary operation.");
    }

    internal Tensor CausalGqaAttention(Tensor query, Tensor key, Tensor value, int sequence, int heads, int kvHeads, int head,
        int maximumQueryRows = 0)
    {
        using var qFlat = query.View(sequence, heads, head);
        using var kFlat = key.View(sequence, kvHeads, head);
        using var vFlat = value.View(sequence, kvHeads, head);
        using var qView = qFlat.Permute(1, 0, 2);
        using var kView = kFlat.Permute(1, 0, 2);
        using var vView = vFlat.Permute(1, 0, 2);
        using var q = Ops.NewContiguous(qView);
        using var k = Ops.NewContiguous(kView);
        using var v = Ops.NewContiguous(vView);
        var result = _context.NewF32(sequence, checked(heads * head));
        try
        {
            float scale = 1 / MathF.Sqrt(head);
            bool success = _context.IsCuda
                ? CudaFusedOps.TryGqaPrefillAttention(result, q, k, v, heads, kvHeads, head, sequence, sequence, 0, 0, scale)
                : MlxMiniMaxH3AttentionOps.TryCausalAttention(result, q, k, v, heads, kvHeads, head, sequence, scale, maximumQueryRows);
            if (!success) throw new NotSupportedException("The selected H3 backend cannot execute F32 causal GQA prefill.");
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    internal static Tensor SplitInterleaved(Tensor projection, int sequence, int heads, int head, int component)
    {
        if ((uint)component >= 3) throw new ArgumentOutOfRangeException(nameof(component));
        using var shaped = projection.View(sequence, heads, 3, head);
        using var part = shaped.Narrow(2, component, 1);
        using var contiguous = Ops.NewContiguous(part);
        return contiguous.View(sequence, checked(heads * head));
    }

    internal void GeluErf(Tensor values)
    {
        if (_context.IsMlx || _context.IsCuda)
        {
            bool success = _context.IsCuda
                ? CudaMiniMaxH3Ops.TryTransform(values, values, null, null, 5, 0)
                : MlxMiniMaxH3Ops.TryTransform(values, values, null, null, 5, 0);
            if (!success)
                throw new NotSupportedException("The selected H3 backend cannot execute its F32 vision GELU-erf operation.");
            return;
        }
        // The A&S polynomial is the same bounded approximation as the managed
        // vision merger. Every operation stays on the selected device.
        using var absolute = _context.NewF32(values.Sizes);
        using var t = _context.NewF32(values.Sizes);
        using var polynomial = _context.NewF32(values.Sizes);
        using var exponential = _context.NewF32(values.Sizes);
        using var sign = _context.NewF32(values.Sizes);
        Ops.Mul(absolute, values, 0.7071067811865475f);
        Ops.Sign(sign, absolute);
        Ops.Abs(absolute, absolute);
        Ops.Mul(t, absolute, 0.3275911f);
        Ops.Add(t, t, 1);
        Ops.Div(t, 1, t);
        Ops.Mul(polynomial, t, 1.061405429f);
        Ops.Sub(polynomial, polynomial, 1.453152027f);
        Ops.Mul(polynomial, polynomial, t);
        Ops.Add(polynomial, polynomial, 1.421413741f);
        Ops.Mul(polynomial, polynomial, t);
        Ops.Sub(polynomial, polynomial, 0.284496736f);
        Ops.Mul(polynomial, polynomial, t);
        Ops.Add(polynomial, polynomial, 0.254829592f);
        Ops.Mul(polynomial, polynomial, t);
        Ops.Mul(exponential, absolute, absolute);
        Ops.Neg(exponential, exponential);
        Ops.Exp(exponential, exponential);
        Ops.Mul(polynomial, polynomial, exponential);
        Ops.Sub(polynomial, 1, polynomial);
        Ops.Mul(polynomial, polynomial, sign);
        Ops.Add(polynomial, polynomial, 1);
        Ops.Mul(polynomial, polynomial, 0.5f);
        Ops.Mul(values, values, polynomial);
    }

    public void Dispose()
    {
        _cos?.Dispose(); _sin?.Dispose();
        _cos = _sin = null;
        _lastCos = _lastSin = null;
    }
}
