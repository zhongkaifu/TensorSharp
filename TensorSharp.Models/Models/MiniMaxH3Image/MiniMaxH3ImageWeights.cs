// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's denoiser weights as graph leaves: every projection a pointer into the file's mapping in its stored
// format (the device caches key on it), every norm and bias an F32 copy on the host (BF16 and F16 widened exactly), as
// the retired MiniMaxH3DiT bound them for the native graph. The PDD head bank and the curve table stay F32 on the host: the head of
// a step is combined there and bound as an input, the curve table is interpolated there.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Graph;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>A weight leaf keyed by its host bytes: a projection [Ne0, Ne1] in the file's mapping, or (<see cref="Vector"/>)
    /// a pinned F32 vector [Ne0] the denoiser owns.</summary>
    internal sealed record MiniMaxH3ImageWeight(int Format, long Ne0, long Ne1, IntPtr Data, long Bytes, bool Vector = false) : IGraphWeight
    {
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;

        /// <summary>ggml's quantized block formats: a product over one takes MiniMax-H3's q8_1 overflow guard (h3_mm).</summary>
        public bool Quantized => Format is not (GraphFormat.F32 or GraphFormat.F16 or GraphFormat.BF16);
    }

    /// <summary>A projection: its weight [in, out] and, where the native graph adds one, its F32 bias [out].</summary>
    internal sealed record MiniMaxH3ImageLinear(MiniMaxH3ImageWeight Weight, MiniMaxH3ImageWeight Bias);

    /// <summary>One transformer block (a token refiner block has no modulation: <see cref="AdaLn"/> null).</summary>
    internal sealed class MiniMaxH3ImageBlock
    {
        public MiniMaxH3ImageWeight Norm1, Norm2, QNorm, KNorm;
        public MiniMaxH3ImageLinear AdaLn, Qkv, Out, Fc1, Fc2;
    }

    /// <summary>
    /// The tensors of a MiniMax-H3-Image denoiser file bound the way the retired MiniMaxH3DiT bound them for
    /// TSGgml_MiniMaxH3DitForward: the same host bytes for every projection (so both share one device copy), F32 norms and
    /// biases converted as MiniMaxH3DiT converted them, and no bias where its graph adds none (qkv, out, fc1, fc2).
    /// </summary>
    internal sealed class MiniMaxH3ImageWeights : IDisposable
    {
        private readonly MiniMaxH3WeightFile _file;
        private readonly bool _ownsFile;
        private readonly List<GCHandle> _pins = new();
        private readonly List<MiniMaxH3ImageWeight> _all = new();
        private IntPtr[] _directRead;
        private bool _disposed;

        public MiniMaxH3ImageWeights(MiniMaxH3WeightFile file, bool ownsFile, BackendType backend)
        {
            _file = file ?? throw new ArgumentNullException(nameof(file));
            _ownsFile = ownsFile;
            try
            {
                Config = MiniMaxH3Config.Detect(file.Tensors);
                if (Config.HasAudio)
                    throw new NotSupportedException($"{System.IO.Path.GetFileName(file.Path)} is a MiniMax-H3 audio-video denoiser, not MiniMax-H3-Image.");
                if (Config.PddSteps < 2)
                    throw new NotSupportedException($"{System.IO.Path.GetFileName(file.Path)} has one output head; MiniMax-H3-Image stacks one per step.");
                if (Config.TimeEmbedding != MiniMaxH3TimeEmbedding.CurveTable)
                    throw new NotSupportedException("this MiniMax-H3 checkpoint uses the sinusoidal time-embedder variant, which is not implemented.");
                string p = Config.Prefix;
                VideoPatch = Linear(p + "video_patch_proj.weight", p + "video_patch_proj.bias");
                ConditionProj = Linear(p + "condition_proj.weight", p + "condition_proj.bias");
                FinalAdaLn = Linear(p + "final_layer.adaln_proj.linear.weight", p + "final_layer.adaln_proj.linear.bias");
                FinalNorm = Vector(p + "final_layer.norm.weight");
                RefinerFinalNorm = Vector(p + "token_refiner.final_norm.weight");
                Refiner = new MiniMaxH3ImageBlock[Config.TokenRefinerLayers];
                for (int i = 0; i < Refiner.Length; i++) Refiner[i] = Block($"{p}token_refiner.blocks.{i}.", modulated: false);
                Blocks = new MiniMaxH3ImageBlock[Config.NumLayers];
                for (int i = 0; i < Blocks.Length; i++) Blocks[i] = Block($"{p}blocks.{i}.", modulated: true);
                PddWeight = ReadF32(p + "final_layer.video_out.weight");
                PddBias = ReadF32(p + "final_layer.video_out.bias");
                CurveTable = ReadF32(p + "adaln_t_table");
                InvFreq = ReadF32(p + "rope.inv_freq");
                // The native uploads read these mappings from the file where the device allows it (MiniMaxH3DirectRead).
                if (ownsFile) _directRead = MiniMaxH3DirectRead.Register(file.MappedRegions(), backend);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public MiniMaxH3Config Config { get; }
        public MiniMaxH3WeightFile File => _file;

        /// <summary>Whether the mapping is this set's (else another owner's, whose projections' device copies are its own).</summary>
        public bool OwnsFile => _ownsFile;
        public MiniMaxH3ImageLinear VideoPatch { get; }
        public MiniMaxH3ImageLinear ConditionProj { get; }
        public MiniMaxH3ImageLinear FinalAdaLn { get; }
        public MiniMaxH3ImageWeight FinalNorm { get; }
        public MiniMaxH3ImageWeight RefinerFinalNorm { get; }
        public MiniMaxH3ImageBlock[] Refiner { get; }
        public MiniMaxH3ImageBlock[] Blocks { get; }

        /// <summary>The stacked output heads [PddSteps * patch][hidden] and their biases, F32.</summary>
        public float[] PddWeight { get; }
        public float[] PddBias { get; }

        /// <summary>The AdaLN curve table, [grid][timeEmbedDim] row-major.</summary>
        public float[] CurveTable { get; }
        public float[] InvFreq { get; }

        /// <summary>Every weight leaf a step binds.</summary>
        public IReadOnlyList<MiniMaxH3ImageWeight> All => _all;

        /// <summary>The device bytes the weight leaves take.</summary>
        public long Bytes
        {
            get
            {
                long total = 0;
                foreach (MiniMaxH3ImageWeight w in _all) total += w.Bytes;
                return total;
            }
        }

        private MiniMaxH3ImageBlock Block(string b, bool modulated) => new()
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

        private MiniMaxH3ImageLinear Linear(string weight, string bias)
        {
            if (!_file.Tensors.TryGetValue(weight, out GgufTensorInfo info))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            if (info.Shape.Length != 2) throw new InvalidOperationException($"'{weight}' is not a matrix");
            if (!_file.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            long ne0 = (long)info.Shape[0], ne1 = (long)info.Shape[1];
            long bytes = ne0 / GgufFile.GetBlockSize(info.Type) * GgufFile.GetTypeSize(info.Type) * ne1;
            var w = new MiniMaxH3ImageWeight((int)info.Type, ne0, ne1, ptr, bytes);
            _all.Add(w);
            return new MiniMaxH3ImageLinear(w, bias != null ? Vector(bias) : null);
        }

        /// <summary>A small tensor as a pinned F32 vector leaf, null when the file has none (MiniMaxH3DiT.PinF32).</summary>
        private MiniMaxH3ImageWeight Vector(string name)
        {
            if (!_file.Tensors.ContainsKey(name)) return null;
            float[] values = ReadF32(name);
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            var w = new MiniMaxH3ImageWeight(GraphFormat.F32, values.LongLength, 1, h.AddrOfPinnedObject(), values.LongLength * sizeof(float),
                Vector: true);
            _all.Add(w);
            return w;
        }

        /// <summary>A small tensor as F32, converted by dtype as MiniMaxH3DiT.ReadVectorF32 converts it.</summary>
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
            if (_ownsFile) _file.Dispose();
        }
    }
}
