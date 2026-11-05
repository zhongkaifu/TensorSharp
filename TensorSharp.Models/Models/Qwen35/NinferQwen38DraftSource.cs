// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>Bounded original draft loading bridge. Owns converted plane readers, borrows the artifact reader.</summary>
internal sealed class NinferQwen38DraftSource : IDisposable
{
    private readonly NinferQwen38DraftPlan _plan;
    private readonly List<IDisposable> _sources = new();
    private bool _disposed;
    internal CudaNinferDraftDesc Descriptor { get; }

    internal NinferQwen38DraftSource(NinferQwen38DraftPlan plan, int draftTokens, CudaNinferDraftHeadPolicy headPolicy)
    {
        ArgumentNullException.ThrowIfNull(plan); _plan = plan;
        if (draftTokens < 1 || draftTokens > NinferQwen38DraftPlan.MaximumDraftTokens) throw new ArgumentOutOfRangeException(nameof(draftTokens));
        if (!Enum.IsDefined(headPolicy)) throw new ArgumentOutOfRangeException(nameof(headPolicy));
        if (headPolicy == CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist && !plan.HasShortlist)
            throw new NotSupportedException("The selected original shortlist is absent from the artifact.");
        try
        {
            var layers = new CudaNinferDraftLayerDesc[NinferQwen38DraftPlan.LayerCount];
            for (int i = 0; i < layers.Length; i++)
            {
                string p = $"dflash2/layers/{i}/";
                layers[i] = new(Read(p + "input_norm"), Read(p + "attention/query_norm"), Read(p + "attention/key_norm"),
                    Read(p + "post_attention_norm"), Read(p + "attention_conv/base_kernel"), Read(p + "mlp_conv/base_kernel"),
                    Projection(p + "attention/query_key_value"), Projection(p + "attention/output"), Projection(p + "attention_conv/kernel_projection"),
                    Projection(p + "mlp/gate_up", gateUp: true), Projection(p + "mlp/down"), Projection(p + "mlp_conv/kernel_projection"));
            }
            CudaNinferDraftShortlistDesc? shortlist = null;
            if (headPolicy == CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist)
            {
                var head = plan.GetTensor("proposal/head");
                ValidateShortlistScales(plan, head);
                shortlist = new(new NinferTensorWeightSource(plan.Reader, head.Descriptor), checked((long)head.Descriptor.ByteLength),
                    ReadShortlistTokenIds(plan, plan.GetTensor("proposal/token_ids")));
            }
            Descriptor = new(draftTokens, headPolicy, shortlist, Projection("dflash2/feature_projection"),
                Read("dflash2/context_norm"), Read("dflash2/final_norm"), layers,
                Projection("dflash2/candidate_selector/hidden_projection"), Projection("dflash2/candidate_selector/predecessor_codebook"),
                Projection("dflash2/candidate_selector/successor_codebook"),
                $"{NinferQwen38DraftPlan.BindingVersion}:{plan.DirectorySha256}:{CudaNinferDraftDesc.NumericalPolicy}:draft{draftTokens}:head{headPolicy}");
        }
        catch { Dispose(); throw; }
    }

    private float[] Read(string name) => ReadPlainParameter(_plan, _plan.GetTensor(name));
    private CudaNinferProjectionDesc Projection(string name, bool gateUp = false)
    {
        var d = _plan.GetTensor(name).Descriptor;
        int rows = checked((int)d.Shape[0]), columns = checked((int)d.Shape[1]);
        if (d.Format == NinferTensorFormat.BFloat16)
            return new(new NinferTensorWeightSource(_plan.Reader, d), 0, checked((long)d.ByteLength), rows, columns,
                CudaNinferProjectionFormat.BFloat16, allowA4: false);
        // q8_g32: gate|up interleaved by 8 rows so the SwiGLU projection holds a gate and its up value.
        var planar = new NinferRowPlanarWeightSource(_plan.Reader, d, gateUp ? NinferRowPlanarWeightSource.GateUpRows8(rows) : null);
        return new(planar, 0, planar.Bytes, rows, columns, CudaNinferProjectionFormat.W8G32, allowA4: false);
    }

    internal static float[] ReadPlainParameter(NinferQwen38DraftPlan plan, NinferQwen38DraftBinding binding)
    {
        if (binding.Descriptor.Format != NinferTensorFormat.BFloat16 || binding.Descriptor.ByteLength > 1 << 20)
            throw new InvalidDataException("Original draft plain parameters must be bounded BF16 vectors or convolution bases.");
        byte[] raw = new byte[checked((int)binding.Descriptor.ByteLength)]; plan.ReadTensorRange(binding, 0, raw);
        // All draft norms are plain; dynamic convolution stays in original [side][tap][channel] order.
        return NinferQwen38ModelSource.DecodePlainFloats(raw, NinferTensorFormat.BFloat16);
    }

    internal static int[] ReadShortlistTokenIds(NinferQwen38DraftPlan plan, NinferQwen38DraftBinding binding)
    {
        if (binding.Role != NinferQwen38DraftRole.ShortlistTokenIds || binding.Descriptor.ByteLength != 4 * NinferQwen38DraftPlan.ShortlistRows)
            throw new InvalidDataException("Expected the original I32 shortlist map.");
        byte[] raw = new byte[checked((int)binding.Descriptor.ByteLength)]; plan.ReadTensorRange(binding, 0, raw);
        int[] ids = new int[NinferQwen38DraftPlan.ShortlistRows];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = BinaryPrimitives.ReadInt32LittleEndian(raw.AsSpan(i * 4, 4));
            if ((uint)ids[i] >= NinferQwen38DraftPlan.VocabularySize) throw new InvalidDataException("Original shortlist token ID is outside the physical target vocabulary.");
        }
        return ids;
    }

    internal static void ValidateShortlistScales(NinferQwen38DraftPlan plan, NinferQwen38DraftBinding binding)
    {
        if (binding.Role != NinferQwen38DraftRole.ShortlistHead) throw new InvalidDataException("Expected original Q4G64 shortlist.");
        byte[] scratch = new byte[64 << 10];
        ulong offset = binding.Descriptor.Geometry.ScalePlaneOffset!.Value, left = binding.Descriptor.Geometry.ScalePlaneBytes!.Value;
        while (left != 0)
        {
            int take = checked((int)Math.Min((ulong)scratch.Length, left));
            plan.ReadTensorRange(binding, offset, scratch.AsSpan(0, take));
            for (int at = 0; at < take; at += 2)
                if ((BinaryPrimitives.ReadUInt16LittleEndian(scratch.AsSpan(at, 2)) & 0x7c00) == 0x7c00)
                    throw new InvalidDataException("Nonfinite original shortlist F16 scale.");
            offset += (ulong)take; left -= (ulong)take;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var source in _sources) source.Dispose();
        _sources.Clear(); // The caller still owns the original reader.
    }
}
