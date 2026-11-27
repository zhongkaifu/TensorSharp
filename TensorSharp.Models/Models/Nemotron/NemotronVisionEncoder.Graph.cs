// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Cpu;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// The RADIO vision tower and its projector as one graph program per image tile, on the executor of the model's
    /// allocator (GGML cpu / cuda / metal / vulkan, the managed CPU, MLX, direct CUDA): the managed path's arithmetic in
    /// ggml's layout. The patch rows are packed on the host (<see cref="NemotronImageProcessor.PackPatchesCHW"/>) and the
    /// position table resized there (cached per grid), as the managed path does; then the patch embedding, the class tokens,
    /// every pre-norm block (LayerNorm, the [Q | K | V] projection or the split one, attention over all tokens, the output
    /// projection, LayerNorm, the GELU MLP), the tower's final LayerNorm where the mmproj has one, the pixel shuffle and the
    /// RMSNorm / ReLU² projector. The managed path's weights are read once at load (its validation), then dropped but for the
    /// position and class tables.
    /// </summary>
    public sealed partial class NemotronVisionEncoder
    {
        private NetworkWeights _graphWeights;
        private NetworkGraph _graph;
        private IAllocator _outputAllocator;
        private bool _splitQkv;

        /// <summary>True when images encode as graph programs.</summary>
        public bool RunsAsGraphs => _graph != null;

        /// <summary>The graph programs' plan cache and executor (tests).</summary>
        internal NetworkGraph Graph => _graph;

        /// <summary>The mmproj at <paramref name="mmProjPath"/> as graph programs on <paramref name="allocator"/>'s executor (the
        /// model's), its embeddings handed back on <paramref name="hostAllocator"/>; the managed path over
        /// <paramref name="hostAllocator"/> where no executor is registered for the allocator.</summary>
        public static NemotronVisionEncoder Load(string mmProjPath, IAllocator allocator, BackendType backend, IAllocator hostAllocator)
            => GraphExecutors.For(allocator) == null
                ? new NemotronVisionEncoder(mmProjPath, hostAllocator ?? allocator)
                : new NemotronVisionEncoder(mmProjPath, allocator, backend, hostAllocator, null);

        /// <summary>The graph encoder: the managed load and validation on a host allocator, then the network's weights on
        /// <paramref name="allocator"/> and the programs on its executor (or <paramref name="executor"/>).</summary>
        internal NemotronVisionEncoder(string mmProjPath, IAllocator allocator, BackendType backend, IAllocator outputAllocator,
            IGraphExecutor executor)
            : this(mmProjPath, new CpuAllocator(BlasEnum.DotNet))
        {
            _outputAllocator = outputAllocator;
            try
            {
                _graphWeights = new NetworkWeights(allocator, backend);
                using (var gguf = new GgufFile(mmProjPath))
                {
                    _splitQkv = gguf.Tensors.ContainsKey("v.blk.0.attn_q.weight") && !gguf.Tensors.ContainsKey("v.blk.0.attn_qkv.weight");
                    _graphWeights.LoadGguf(gguf, n => (n.StartsWith("v.", StringComparison.Ordinal) || n.StartsWith("mm.", StringComparison.Ordinal))
                        && n != VideoPatchEmbeddingName && n != "v.position_embd.weight");
                }
                _graph = new NetworkGraph(_graphWeights, executor, "nemotron vision encoder");
            }
            catch
            {
                Dispose();
                throw;
            }
            // The managed weights did their part (validation); the position table stays for the per-grid resize, the class
            // table for nothing but symmetry with the load (both are small).
            foreach (string name in new List<string>(_weights.Keys))
            {
                if (name is "v.position_embd.weight" or "v.class_embd") continue;
                _weights[name].Dispose();
                _weights.Remove(name);
            }
        }

        private Tensor EncodeGraph(float[] pixelValues, int imageWidth, int imageHeight)
        {
            int patchesX = imageWidth / _patchSize, patchesY = imageHeight / _patchSize, numPatches = patchesX * patchesY;
            int patchDim = _numChannels * _patchSize * _patchSize;
            float[] packed = NemotronImageProcessor.PackPatchesCHW(pixelValues, imageWidth, imageHeight, _numChannels, _patchSize);
            float[] positions = _hasPositionEmbed ? PositionRows(patchesX, patchesY, numPatches) : null;
            int output = 0;
            float[][] outputs = _graph.Run(b => output = b.Output(Emit(b, packed, positions, patchDim, patchesX, patchesY)));
            float[] rows = outputs[output];
            int scale = Math.Max(1, _scaleFactor);
            int tokens = patchesX / scale * (patchesY / scale);
            var result = new Tensor(_outputAllocator ?? _graphWeights.Allocator, DType.Float32, tokens, _projectionDim);
            try
            {
                result.SetElementsAsFloat(rows);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>The position rows the patches add ([patches][hidden]): the table's first rows on its own grid, else the
        /// table bilinearly resized to the grid (cached), as <see cref="AddPositionEmbeddings"/> takes them.</summary>
        private float[] PositionRows(int patchesW, int patchesH, int numPatches)
        {
            if (patchesW == _positionSourceSide && patchesH == _positionSourceSide && _positionTokens >= numPatches)
            {
                Tensor table = _weights["v.position_embd.weight"];
                using Tensor first = table.Narrow(0, 0, numPatches);
                using Tensor packed = Ops.NewContiguous(first);
                return packed.GetElementsAsFloat(numPatches * _hiddenSize);
            }
            return _resizedPositionEmbeddings.GetOrAdd((patchesW, patchesH), key =>
            {
                Tensor table = _weights["v.position_embd.weight"];
                float[] values = table.GetElementsAsFloat((int)table.ElementCount());
                return ResizePositionEmbedding(values, _hiddenSize, _positionSourceSide, _positionSourceSide, key.W, key.H);
            });
        }

        private NetValue Emit(NetworkBuilder b, float[] packed, float[] positions, int patchDim, int patchesX, int patchesY)
        {
            int h = _hiddenSize, n = patchesX * patchesY;
            NetValue x = b.Linear(b.Input(packed, patchDim, n), b.Weight("v.patch_embd.weight", patchDim, h),
                Has("v.patch_embd.bias") ? b.Vector("v.patch_embd.bias", h) : default);
            if (positions != null) x = b.Add(x, b.Input(positions, h, n));
            int tokens = n;
            if (_numClassTokens > 0)
            {
                x = b.Concat(b.Weight("v.class_embd", h, _numClassTokens), x, 1);
                tokens += _numClassTokens;
            }
            for (int i = 0; i < _blockCount; i++)
            {
                string p = $"v.blk.{i}";
                x = b.Add(x, Attention(b, Norm(b, x, p + ".ln1"), p, tokens));
                NetValue up = b.Linear(Norm(b, x, p + ".ln2"), p + ".ffn_up.weight", Bias(p + ".ffn_up"));
                NetValue act = _useGelu ? b.Op(GraphOp.Gelu, stackalloc[] { b.Cont(up) }, stackalloc long[] { up.Ne0, up.Ne1 }) : b.Relu(up);
                x = b.Add(x, b.Linear(act, p + ".ffn_down.weight", Bias(p + ".ffn_down")));
            }
            if (Has("v.post_ln.weight")) x = Norm(b, x, "v.post_ln");
            if (_numClassTokens > 0) x = b.Slice(x, 1, _numClassTokens, n);

            // Pixel shuffle ([outY, sy, outX, sx, hidden] -> [outY, outX, sy, sx, hidden] row-major): hidden and sx are adjacent,
            // so the swap is of the sy and outX axes of [hidden * s, outX, sy, outY].
            int s = Math.Max(1, _scaleFactor), outX = patchesX / s, outY = patchesY / s;
            NetValue grid = b.Reshape(b.Cont(x), (long)h * s, outX, s, outY);
            NetValue merged = b.Reshape(b.Cont(b.Permute(grid, 0, 2, 1, 3)), (long)h * s * s, (long)outX * outY);
            NetValue y = b.RmsNorm(merged, 1e-5f, b.Vector("mm.model.mlp.0.weight", (long)h * s * s));
            y = b.Relu(b.Linear(y, "mm.model.mlp.1.weight"));
            return b.Linear(b.Mul(y, y), "mm.model.mlp.3.weight");
        }

        private NetValue Attention(NetworkBuilder b, NetValue x, string prefix, int tokens)
        {
            int h = _hiddenSize;
            NetValue q, k, v;
            if (_splitQkv)
            {
                q = b.Linear(x, prefix + ".attn_q.weight", Bias(prefix + ".attn_q"));
                k = b.Linear(x, prefix + ".attn_k.weight", Bias(prefix + ".attn_k"));
                v = b.Linear(x, prefix + ".attn_v.weight", Bias(prefix + ".attn_v"));
            }
            else
            {
                NetValue qkv = b.Linear(x, prefix + ".attn_qkv.weight", Bias(prefix + ".attn_qkv"));
                q = b.Columns(qkv, 0, h);
                k = b.Columns(qkv, h, h);
                v = b.Columns(qkv, 2L * h, h);
            }
            NetValue Heads(NetValue t) => b.Reshape(b.Cont(t), _headDim, _numHeads, tokens);
            NetValue attended = b.Attention(Heads(q), Heads(k), Heads(v), 1f / MathF.Sqrt(_headDim));
            return b.Linear(attended, prefix + ".attn_out.weight", Bias(prefix + ".attn_out"));
        }

        private NetValue Norm(NetworkBuilder b, NetValue x, string prefix)
            => b.LayerNorm(x, _eps, b.Vector(prefix + ".weight", _hiddenSize), Has(prefix + ".bias") ? b.Vector(prefix + ".bias", _hiddenSize) : default);

        private bool Has(string name) => _graphWeights.Contains(name);

        private string Bias(string prefix) => Has(prefix + ".bias") ? prefix + ".bias" : null;

        private void DisposeGraph()
        {
            _graph?.Dispose();
            _graph = null;
            _graphWeights?.Dispose();
            _graphWeights = null;
        }
    }
}
