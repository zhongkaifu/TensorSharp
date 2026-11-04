// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public sealed partial class Qwen35CudaExecutor
{
    internal const string NinferBodyModuleEnvironment = "TS_Q35_NINFER_PTX";
    internal const string NinferDraftModuleEnvironment = "TS_Q35_NINFER_DRAFT_PTX";
    internal const string NinferDraftShortlistModuleEnvironment = "TS_Q35_NINFER_DRAFT_SHORTLIST_PTX";

    internal Qwen35CudaExecutor(NinferQwen38ModelSource source, int maxContext, int nUbatch)
        => InitializeNinfer(source, maxContext, nUbatch);

    private void InitializeNinfer(NinferQwen38ModelSource source, int maxContext, int nUbatch)
    {
        if (maxContext <= 0 || maxContext > source.Frontend.MaximumContextTokens)
            throw new ArgumentOutOfRangeException(nameof(maxContext));
        if (nUbatch <= 0) throw new ArgumentOutOfRangeException(nameof(nUbatch));
        try
        {
            var model = BuildNinferModelDesc(source, maxContext, nUbatch,
                ResolveNinferModulePath(NinferBodyModuleEnvironment, "tensorsharp_q35_ninfer.ptx"),
                ResolveNinferModulePath(W8HeadModuleEnvironment, "tensorsharp_w8g32_head.ptx"));
            _engine = new Q35CudaEngine(model);
            if (Environment.GetEnvironmentVariable(NinferDraftModuleEnvironment) is string draftModule)
            {
                string? shortlistModule = Environment.GetEnvironmentVariable(NinferDraftShortlistModuleEnvironment);
                using var draft = new NinferQwen38DraftSource(NinferQwen38DraftPlan.Create(source.Reader),
                    NinferQwen38DraftPlan.MaximumDraftTokens, shortlistModule == null ? CudaNinferDraftHeadPolicy.FullTargetVocabulary : CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist);
                _engine.AttachNinferDrafter(draft.Descriptor,
                    ResolveNinferModulePath(NinferDraftModuleEnvironment, "tensorsharp_q35_ninfer_draft.ptx", draftModule, AppContext.BaseDirectory),
                    shortlistModule == null ? null : ResolveNinferModulePath(NinferDraftShortlistModuleEnvironment, "tensorsharp_ninfer_shortlist.fatbin", shortlistModule, AppContext.BaseDirectory));
            }
            else if (Environment.GetEnvironmentVariable(NinferDraftShortlistModuleEnvironment) != null)
                throw new ArgumentException("An original shortlist requires its original NInfer draft module.");
            _slots.Add(_engine.Active);
            ActiveSlot = 0;
        }
        catch { Dispose(); throw; }
    }

    internal static string ResolveNinferModulePath(string environmentName, string filename)
        => ResolveNinferModulePath(environmentName, filename,
            Environment.GetEnvironmentVariable(environmentName), AppContext.BaseDirectory);

    internal static string ResolveNinferModulePath(string environmentName, string filename, string? configured, string baseDirectory)
    {
        if (configured != null && string.IsNullOrWhiteSpace(configured))
            throw new ArgumentException($"{environmentName} must name a native module when set.");
        string path = Path.GetFullPath(configured ?? Path.Combine(baseDirectory, filename));
        if (configured == null && !File.Exists(path))
        {
            string cudaAsset = Path.GetFullPath(Path.Combine(baseDirectory, "cuda_kernels", filename));
            if (File.Exists(cudaAsset)) return cudaAsset;
        }
        if (!File.Exists(path))
            throw new FileNotFoundException($"NInfer native module unavailable. Build the TensorSharp-owned NInfer/W8 kernels and set {environmentName} to their PTX/fatbin path.", path);
        return path;
    }

    internal static Q35CudaEngine.ModelDesc BuildNinferModelDesc(NinferQwen38ModelSource source,
        int nCtx, int nUbatch, string bodyModule, string headModule)
    {
        var p = source.Plan;
        float[] Norm(NinferQwen38TensorBinding tensor) => NinferQwen38ModelSource.EffectiveNorm(source.ReadFloats(tensor), tensor.NormConvention);
        CudaNinferProjectionDesc Projection(NinferQwen38TensorBinding tensor)
        {
            var descriptor = tensor.Descriptor;
            int rows = checked((int)descriptor.Shape[0]), columns = checked((int)descriptor.Shape[1]);
            if (descriptor.Format == NinferTensorFormat.NvFp4)
            {
                if (tensor.InputScaleDivisor is null) throw new InvalidDataException("NVFP4 projection lacks activation calibration: " + descriptor.Name);
                Span<byte> raw = stackalloc byte[4]; source.Reader.ReadObjectRange(tensor.InputScaleDivisor, 0, raw);
                float inputDivisor = PositiveDivisor(BinaryPrimitives.ReadUInt32LittleEndian(raw));
                var planes = source.ConvertedProjection(tensor);
                return new(planes, 0, planes.Bytes, rows, columns, CudaNinferProjectionFormat.NvFp4Planes,
                    inputDivisor, descriptor.WeightDivisor!.Value);
            }
            if (descriptor.Format != NinferTensorFormat.BFloat16) throw new NotSupportedException("Unexpected target projection format.");
            bool attention = tensor.RowOrder == NinferQwen38RowOrder.AttentionQueryKeyGateValue;
            return new(new NinferTensorWeightSource(source.Reader, descriptor, attention), 0, checked((long)descriptor.ByteLength),
                rows, columns, CudaNinferProjectionFormat.BFloat16, allowA4: false);
        }
        string provenance = JsonSerializer.Serialize(new
        {
            recipe = "ninfer-qwen38-target-nvfp4full-v2",
            source.Reader.FullPath, source.Reader.FileBytes, source.Reader.Identity,
            source.Reader.DirectorySha256, source.Reader.PrefixSha256, source.Reader.FileLastWriteUtcTicks,
            cacheIdentity = source.CacheIdentity,
            fullArtifactSha256Recomputed = false,
            frontendResources = source.Frontend.Resources.Values.Select(x => new { x.Name, x.Sha256, x.ByteLength }),
            embedding = "original-W8G32-F16S", output = "original-W8G32-F16S",
            projectionCalibration = "original-FP32-input-divisor-and-weight-divisor",
            draft = "not-bound", vision = "not-bound"
        });
        var embedding = p.TokenEmbedding.Descriptor;
        var output = p.OutputHead.Descriptor;
        var native = new CudaNinferModelDesc
        {
            Embedding = new(new NinferTensorWeightSource(source.Reader, embedding), 0, checked((long)embedding.ByteLength),
                NinferQwen38TargetPlan.VocabularySize, NinferQwen38TargetPlan.HiddenSize, CudaNinferProjectionFormat.W8G32, allowA4: false),
            Output = new(new NinferTensorWeightSource(source.Reader, output), 0, checked((long)output.ByteLength),
                NinferQwen38TargetPlan.VocabularySize, NinferQwen38TargetPlan.HiddenSize, headModule, provenance,
                profile: "ninfer-qwen38-target-w8g32-f16s-bf16-v1"),
            Layers = new CudaNinferLayerDesc[NinferQwen38TargetPlan.LayerCount],
            ModulePath = bodyModule,
            Provenance = provenance
        };
        var model = new Q35CudaEngine.ModelDesc
        {
            NLayer = NinferQwen38TargetPlan.LayerCount, NEmbd = NinferQwen38TargetPlan.HiddenSize,
            NFf = NinferQwen38TargetPlan.IntermediateSize, NVocab = NinferQwen38TargetPlan.VocabularySize,
            NHead = NinferQwen38TargetPlan.AttentionQueryHeads, NKvHead = NinferQwen38TargetPlan.AttentionKeyValueHeads,
            HeadDim = NinferQwen38TargetPlan.AttentionHeadDimension,
            GdnKHeads = NinferQwen38TargetPlan.GdnQueryKeyHeads, GdnVHeads = NinferQwen38TargetPlan.GdnValueHeads,
            DConv = NinferQwen38TargetPlan.ConvolutionWidth, SectT = 11, SectH = 11, SectW = 10,
            RopeBase = NinferQwen38TargetPlan.RopeTheta, Eps = NinferQwen38TargetPlan.RmsEpsilon,
            NCtx = nCtx, NUbatch = nUbatch, OutputNorm = Norm(p.FinalNorm), Ninfer = native,
            Layers = new Q35CudaEngine.LayerDesc[NinferQwen38TargetPlan.LayerCount]
        };
        foreach (var layer in p.Layers)
        {
            var d = new Q35CudaEngine.LayerDesc
            {
                Recurrent = layer.Kind == NinferQwen38LayerKind.GatedDeltaNet,
                AttnNorm = Norm(layer.InputNorm), PostNorm = Norm(layer.PostAttentionNorm)
            };
            var n = new CudaNinferLayerDesc
            {
                Input = Projection(layer.InputProjection), Output = Projection(layer.OutputProjection),
                GateUp = Projection(layer.GateUp), Down = Projection(layer.Down)
            };
            if (d.Recurrent)
            {
                n.AlphaBeta = Projection(layer.AlphaBeta!);
                d.ConvW = NinferQwen38ModelSource.ChannelMajorConvolution(source.ReadFloats(layer.Convolution!), 10240,
                    NinferQwen38TargetPlan.ConvolutionWidth);
                d.DtBias = source.ReadFloats(layer.DtBias!);
                d.SsmA = source.ReadFloats(layer.ALog!); // The NInfer CUDA prep owns the original -expf convention.
                d.SsmNorm = Norm(layer.GdnNorm!);
            }
            else { d.QNorm = Norm(layer.QueryNorm!); d.KNorm = Norm(layer.KeyNorm!); }
            model.Layers[layer.Index] = d;
            native.Layers[layer.Index] = n;
        }
        return model;
    }

    internal static float PositiveDivisor(uint bits)
    {
        float value = BitConverter.Int32BitsToSingle(unchecked((int)bits));
        if (!float.IsFinite(value) || value <= 0) throw new InvalidDataException("NInfer activation divisor must be positive and finite.");
        return value;
    }
}
