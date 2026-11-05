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
    internal bool HasEmbeddedNinferDrafter { get; private set; }

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
                ResolveNinferModulePath(NinferBodyModuleEnvironment, "tensorsharp_q35_ninfer.ptx"));
            _engine = new Q35CudaEngine(model);
            string? draftModule = Environment.GetEnvironmentVariable(NinferDraftModuleEnvironment);
            if (draftModule == null && Environment.GetEnvironmentVariable(NinferDraftShortlistModuleEnvironment) != null)
                throw new ArgumentException("An original shortlist requires its original NInfer draft module.");
            // --no-spec (TS_SPEC=0) vetoes the embedded drafter for the whole process: not attaching it
            // keeps its weights out of VRAM and its per-token context append off the decode path.
            if (draftModule != null && !TensorSharp.Runtime.Speculative.SpeculationOptions.FromEnvironment().ExplicitlyDisabled)
            {
                string? shortlistModule = Environment.GetEnvironmentVariable(NinferDraftShortlistModuleEnvironment);
                using var draft = new NinferQwen38DraftSource(NinferQwen38DraftPlan.Create(source.Reader),
                    NinferQwen38DraftPlan.MaximumDraftTokens, shortlistModule == null ? CudaNinferDraftHeadPolicy.FullTargetVocabulary : CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist);
                _engine.AttachNinferDrafter(draft.Descriptor,
                    ResolveNinferModulePath(NinferDraftModuleEnvironment, "tensorsharp_q35_ninfer_draft.ptx", draftModule, AppContext.BaseDirectory),
                    shortlistModule == null ? null : ResolveNinferModulePath(NinferDraftShortlistModuleEnvironment, "tensorsharp_ninfer_shortlist.fatbin", shortlistModule, AppContext.BaseDirectory));
                // This attachment was read from the target's own source.Reader,
                // rather than an external drafter attached to a target wrapper.
                HasEmbeddedNinferDrafter = true;
            }
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
        int nCtx, int nUbatch, string bodyModule)
    {
        var p = source.Plan;
        float[] Norm(NinferQwen38TensorBinding tensor) => NinferQwen38ModelSource.EffectiveNorm(source.ReadFloats(tensor), tensor.NormConvention);
        CudaNinferProjectionDesc Projection(NinferQwen38TensorBinding tensor)
        {
            var descriptor = tensor.Descriptor;
            int rows = checked((int)descriptor.Shape[0]), columns = checked((int)descriptor.Shape[1]);
            switch (descriptor.Format)
            {
                case NinferTensorFormat.NvFp4:
                {
                    float inputDivisor = source.Reader.ReadScalar(tensor.InputScaleDivisor
                        ?? throw new InvalidDataException("NVFP4 projection lacks activation calibration: " + tensor.Name));
                    if (!float.IsFinite(inputDivisor) || inputDivisor <= 0) throw new InvalidDataException("NInfer activation divisor must be positive and finite.");
                    var planes = source.ConvertedProjection(tensor);
                    return new(planes, 0, planes.Bytes, rows, columns, CudaNinferProjectionFormat.NvFp4Planes,
                        inputDivisor, descriptor.WeightDivisor!.Value);
                }
                case NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale:
                {
                    Func<int, int>? order = tensor.RowOrder switch
                    {
                        NinferQwen38RowOrder.AttentionQueryKeyGateValue => NinferRowPlanarWeightSource.AttentionQueryGateKeyValue,
                        NinferQwen38RowOrder.MlpGateThenUp => NinferRowPlanarWeightSource.GateUpRows8(rows),
                        _ => null
                    };
                    var planar = new NinferRowPlanarWeightSource(source.Reader, descriptor, order);
                    return new(planar, 0, planar.Bytes, rows, columns, CudaNinferProjectionFormat.Fp8Row, allowA4: false);
                }
                case NinferTensorFormat.BFloat16:
                    if (tensor.RowOrder is not (NinferQwen38RowOrder.Unchanged or NinferQwen38RowOrder.GdnAlphaThenBeta))
                        throw new NotSupportedException("BF16 target projections keep their stored row order: " + tensor.Name);
                    return new(new NinferTensorWeightSource(source.Reader, descriptor), 0, checked((long)descriptor.ByteLength),
                        rows, columns, CudaNinferProjectionFormat.BFloat16, allowA4: false);
                default:
                    throw new NotSupportedException($"Unexpected target projection format {descriptor.Format}: {tensor.Name}");
            }
        }
        string provenance = JsonSerializer.Serialize(new
        {
            recipe = source.Reader.Identity.WeightsId, binding = NinferQwen38TargetPlan.BindingVersion,
            source.Reader.FullPath, source.Reader.FileBytes, source.Reader.Identity, source.Reader.ArtifactId,
            source.Reader.DirectorySha256, source.Reader.PrefixSha256, source.Reader.FileLastWriteUtcTicks,
            cacheIdentity = source.CacheIdentity,
            frontendResources = source.Frontend.Resources.Values.Select(x => new { x.Name, x.Sha256, x.ByteLength }),
            embedding = "FP8-E4M3-row-BF16", head = "FP8-E4M3-row-BF16",
            projections = "FP8-row (A16 small-T, A8 prefill) / NVFP4 (A4, FP32 activation divisor) / BF16"
        });
        var native = new CudaNinferModelDesc
        {
            Embedding = Projection(p.TokenEmbedding),
            Head = Projection(p.OutputHead),
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
