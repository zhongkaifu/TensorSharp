// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's audio-video denoiser weights as graph leaves: every projection a pointer into the GGUF's mapping in its stored
// format (the device caches key on it), every norm and bias an F32 copy on the host (BF16 and F16 widened exactly), the AdaLN
// curve table and the RoPE inverse frequencies F32 on the host (they are interpolated and tabulated there).
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>A weight leaf keyed by its host bytes: a projection [Ne0, Ne1] in the file's mapping, or (<see cref="Vector"/>)
    /// a pinned F32 vector [Ne0] the denoiser owns.</summary>
    internal sealed record MiniMaxH3DitWeight(int Format, long Ne0, long Ne1, IntPtr Data, long Bytes, bool Vector = false) : IGraphWeight
    {
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;

        /// <summary>ggml's quantized block formats: a product over one takes MiniMax-H3's q8_1 overflow guard (h3_mm).</summary>
        public bool Quantized => Format is not (GraphFormat.F32 or GraphFormat.F16 or GraphFormat.BF16);
    }

    /// <summary>A projection: its weight [in, out] and, where the checkpoint has one, its F32 bias [out].</summary>
    internal sealed record MiniMaxH3DitLinear(MiniMaxH3DitWeight Weight, MiniMaxH3DitWeight Bias);

    /// <summary>One transformer block (a token-refiner block has no modulation: <see cref="AdaLn"/> null).</summary>
    internal sealed class MiniMaxH3DitBlock
    {
        public MiniMaxH3DitWeight Norm1, Norm2, QNorm, KNorm;
        public MiniMaxH3DitLinear AdaLn, Qkv, Out, Fc1, Fc2;
    }

    /// <summary>The tensors of a MiniMax-H3 audio-video denoiser GGUF, bound for the denoiser's programs.</summary>
    internal sealed class MiniMaxH3DenoiserWeights : IDisposable
    {
        private readonly MiniMaxH3WeightFile _file;
        private readonly List<GCHandle> _pins = new();
        private readonly List<MiniMaxH3DitWeight> _all = new();
        private IntPtr[] _directRead;
        private bool _disposed;

        public MiniMaxH3DenoiserWeights(string path, BackendType backend)
        {
            _file = new MiniMaxH3WeightFile(path ?? throw new ArgumentNullException(nameof(path)));
            try
            {
                Config = MiniMaxH3Config.Detect(_file.Tensors);
                if (!Config.HasAudio || Config.PddSteps > 1)
                    throw new NotSupportedException(
                        $"'{System.IO.Path.GetFileName(path)}' is a MiniMax-H3-Image denoiser (no audio stream), which runs as the " +
                        "minimax-h3-image architecture, not as the MiniMax-H3 audio-video denoiser.");
                if (_file.Gguf == null)
                    throw new NotSupportedException($"MiniMax-H3's denoiser is a GGUF; '{System.IO.Path.GetFileName(path)}' is not one.");
                if (Config.TimeEmbedding != MiniMaxH3TimeEmbedding.CurveTable)
                    throw new NotSupportedException(
                        "this MiniMax-H3 checkpoint uses the sinusoidal time-embedder variant, which is not implemented; " +
                        "the released checkpoints ship the curve table.");
                string p = Config.Prefix;
                VideoPatch = Linear(p + "video_patch_proj.weight", p + "video_patch_proj.bias");
                AudioPatch = Linear(p + "audio_patch_proj.weight", p + "audio_patch_proj.bias");
                ConditionProj = Linear(p + "condition_proj.weight", p + "condition_proj.bias");
                FinalAdaLn = Linear(p + "final_layer.adaln_proj.linear.weight", p + "final_layer.adaln_proj.linear.bias");
                FinalVideoOut = Linear(p + "final_layer.video_out.weight", p + "final_layer.video_out.bias");
                FinalAudioOut = Linear(p + "final_layer.audio_out.weight", p + "final_layer.audio_out.bias");
                FinalNorm = Vector(p + "final_layer.norm.weight");
                RefinerFinalNorm = Vector(p + "token_refiner.final_norm.weight");
                Refiner = new MiniMaxH3DitBlock[Config.TokenRefinerLayers];
                for (int i = 0; i < Refiner.Length; i++) Refiner[i] = Block($"{p}token_refiner.blocks.{i}.", modulated: false);
                Blocks = new MiniMaxH3DitBlock[Config.NumLayers];
                for (int i = 0; i < Blocks.Length; i++) Blocks[i] = Block($"{p}blocks.{i}.", modulated: true);
                // Stored as [timeEmbedDim, grid] in ne order: [grid][dim] row-major here.
                CurveTable = ReadF32(p + "adaln_t_table");
                InvFreq = ReadF32(p + "rope.inv_freq");
                // The native uploads read these mappings from the file where the device allows it (MiniMaxH3DirectRead).
                _directRead = MiniMaxH3DirectRead.Register(_file.MappedRegions(), backend);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public MiniMaxH3Config Config { get; }
        public string Path => _file.Path;
        public MiniMaxH3DitLinear VideoPatch { get; }
        public MiniMaxH3DitLinear AudioPatch { get; }
        public MiniMaxH3DitLinear ConditionProj { get; }
        public MiniMaxH3DitLinear FinalAdaLn { get; }
        public MiniMaxH3DitLinear FinalVideoOut { get; }
        public MiniMaxH3DitLinear FinalAudioOut { get; }
        public MiniMaxH3DitWeight FinalNorm { get; }
        public MiniMaxH3DitWeight RefinerFinalNorm { get; }
        public MiniMaxH3DitBlock[] Refiner { get; }
        public MiniMaxH3DitBlock[] Blocks { get; }

        /// <summary>The AdaLN curve table, [grid][timeEmbedDim] row-major.</summary>
        public float[] CurveTable { get; }
        public float[] InvFreq { get; }

        /// <summary>Every weight leaf a step binds.</summary>
        public IReadOnlyList<MiniMaxH3DitWeight> All => _all;

        /// <summary>The device bytes the weight leaves take.</summary>
        public long Bytes
        {
            get
            {
                long total = 0;
                foreach (MiniMaxH3DitWeight w in _all) total += w.Bytes;
                return total;
            }
        }

        private MiniMaxH3DitBlock Block(string b, bool modulated) => new()
        {
            Norm1 = Vector(b + "norm1.weight"),
            Norm2 = Vector(b + "norm2.weight"),
            QNorm = Vector(b + "attn.q_norm.weight"),
            KNorm = Vector(b + "attn.k_norm.weight"),
            AdaLn = modulated ? Linear(b + "adaln_proj.linear.weight", b + "adaln_proj.linear.bias") : null,
            Qkv = Linear(b + "attn.qkv_proj.weight", null),
            Out = Linear(b + "attn.out_proj.weight", null),
            Fc1 = Linear(b + "mlp.fc1.weight", null),
            Fc2 = Linear(b + "mlp.fc2.weight", null),
        };

        private MiniMaxH3DitLinear Linear(string weight, string bias)
        {
            if (!_file.Tensors.TryGetValue(weight, out GgufTensorInfo info))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            if (info.Shape.Length != 2) throw new InvalidOperationException($"'{weight}' is not a matrix");
            if (!_file.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            long ne0 = (long)info.Shape[0], ne1 = (long)info.Shape[1];
            long bytes = ne0 / GgufFile.GetBlockSize(info.Type) * GgufFile.GetTypeSize(info.Type) * ne1;
            var w = new MiniMaxH3DitWeight((int)info.Type, ne0, ne1, ptr, bytes);
            _all.Add(w);
            return new MiniMaxH3DitLinear(w, bias != null ? Vector(bias) : null);
        }

        /// <summary>A small tensor as a pinned F32 vector leaf, null when the file has none.</summary>
        private MiniMaxH3DitWeight Vector(string name)
        {
            if (!_file.Tensors.ContainsKey(name)) return null;
            float[] values = ReadF32(name);
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            var w = new MiniMaxH3DitWeight(GraphFormat.F32, values.LongLength, 1, h.AddrOfPinnedObject(), values.LongLength * sizeof(float),
                Vector: true);
            _all.Add(w);
            return w;
        }

        /// <summary>A small tensor as F32, converted by dtype (BF16 and F16 widened exactly).</summary>
        internal unsafe float[] ReadF32(string name)
        {
            GgufTensorInfo info = _file.Tensors[name];
            long n = info.NumElements;
            var data = new float[n];
            if (!_file.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{name}'.");
            switch (info.Type)
            {
                case GgmlTensorType.F32:
                {
                    float* src = (float*)ptr;
                    for (long i = 0; i < n; i++) data[i] = src[i];
                    break;
                }
                case GgmlTensorType.F16:
                {
                    ushort* src = (ushort*)ptr;
                    for (long i = 0; i < n; i++) data[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                    break;
                }
                case GgmlTensorType.BF16:
                {
                    ushort* src = (ushort*)ptr;
                    for (long i = 0; i < n; i++) data[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
                    break;
                }
                default:
                    throw new NotSupportedException($"'{name}' has unexpected type {info.Type}.");
            }
            return data;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_directRead != null) MiniMaxH3DirectRead.Unregister(_directRead);
            _directRead = null;
            foreach (GCHandle h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _file.Dispose();
        }
    }
}
