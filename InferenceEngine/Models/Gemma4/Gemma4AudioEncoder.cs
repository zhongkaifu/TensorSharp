// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceEngine
{
    public class Gemma4AudioEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly IAllocator _allocator;

        private readonly int _hiddenSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _ffnSize;
        private readonly int _numLayers;
        private readonly int _melBins;
        private readonly float _eps;
        private readonly int _projectionDim;

        private readonly int _chunkSize = 12;
        private readonly int _maxPast = 12;
        private readonly int _maxFuture = 0;
        private readonly int _contextSize;
        private readonly float _logitCap = 50f;
        private readonly float _residualWeight = 0.5f;
        private readonly float _gradClip = 1e10f;

        private struct ClampParams
        {
            public float InMin, InMax, OutMin, OutMax;
            public bool HasClamp;
        }
        private readonly Dictionary<string, ClampParams> _clampParams = new();

        private bool _useOllamaNames;
        private Tensor _onesForNorm;

        public int ProjectionDim => _projectionDim;

        public Gemma4AudioEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            var gguf = new GgufFile(mmProjPath);

            _hiddenSize = (int)gguf.GetUint32("clip.audio.embedding_length",
                (uint)gguf.GetUint32("gemma4.audio.embedding_length", 1024));
            _numHeads = (int)gguf.GetUint32("clip.audio.attention.head_count",
                (uint)gguf.GetUint32("gemma4.audio.attention.head_count", 8));
            _headDim = _hiddenSize / _numHeads;
            _ffnSize = (int)gguf.GetUint32("clip.audio.feed_forward_length",
                (uint)gguf.GetUint32("gemma4.audio.feed_forward_length", 4096));
            _numLayers = (int)gguf.GetUint32("clip.audio.block_count",
                (uint)gguf.GetUint32("gemma4.audio.block_count", 12));
            _melBins = (int)gguf.GetUint32("clip.audio.num_mel_bins", 128);
            _eps = gguf.GetFloat32("clip.audio.attention.layer_norm_epsilon",
                gguf.GetFloat32("gemma4.audio.attention.layer_norm_epsilon", 1e-6f));
            _projectionDim = (int)gguf.GetUint32("clip.audio.projection_dim", 2560);
            _contextSize = _chunkSize + _maxPast + _maxFuture;

            Console.WriteLine($"Audio encoder: hidden={_hiddenSize}, heads={_numHeads}, headDim={_headDim}, " +
                $"ffn={_ffnSize}, layers={_numLayers}, melBins={_melBins}, eps={_eps}");
            Console.WriteLine($"  chunk={_chunkSize}, maxPast={_maxPast}, maxFuture={_maxFuture}, context={_contextSize}");
            Console.WriteLine($"  projDim={_projectionDim}");

            LoadWeights(gguf);
            gguf.Dispose();

            _useOllamaNames = _weights.ContainsKey("a.blk.0.ln1.weight");
            Console.WriteLine($"  GGUF naming: {(_useOllamaNames ? "Ollama" : "mmproj/Unsloth")}");
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading audio encoder weights...");
            int count = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                if (!info.Name.StartsWith("a.") && !info.Name.StartsWith("mm.a."))
                    continue;

                byte[] raw = gguf.ReadTensorData(info);
                long numElements = info.NumElements;
                float[] f32 = new float[numElements];

                if (info.Type == GgmlTensorType.F32)
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                else
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                count++;

                if (info.Name.Contains("input_min") || info.Name.Contains("input_max") ||
                    info.Name.Contains("output_min") || info.Name.Contains("output_max"))
                {
                    string linearKey = info.Name.Substring(0, info.Name.LastIndexOf('.'));
                    if (!_clampParams.ContainsKey(linearKey))
                        _clampParams[linearKey] = new ClampParams
                        {
                            InMin = float.MinValue, InMax = float.MaxValue,
                            OutMin = float.MinValue, OutMax = float.MaxValue,
                            HasClamp = false
                        };

                    var cp = _clampParams[linearKey];
                    cp.HasClamp = true;
                    if (info.Name.EndsWith("input_min")) cp.InMin = f32[0];
                    else if (info.Name.EndsWith("input_max")) cp.InMax = f32[0];
                    else if (info.Name.EndsWith("output_min")) cp.OutMin = f32[0];
                    else if (info.Name.EndsWith("output_max")) cp.OutMax = f32[0];
                    _clampParams[linearKey] = cp;
                }
            }
            Console.WriteLine($" done ({count} tensors, {_clampParams.Count} clampable linears)");

        }

        public unsafe Tensor Encode(float[] melData, int numFrames)
        {
            Console.Write("Audio encoder SSCP...");

            // melData is [numFrames, melBins] row-major. We need [numFrames, melBins] as TensorSharp tensor.
            // In GGML: mel features is [melBins, numFrames] (ne0=melBins, ne1=numFrames).
            // But we work in TensorSharp row-major: [numFrames, melBins].

            // SSCP Conv2D: process as 2D convolution over frequency and time.
            // We implement Conv2D manually since TensorSharp may not have it.
            // Input: [F=melBins, T=numFrames], Conv0: [3,3,1,128] stride 2, pad 1
            var conv0Out = Conv2DBlock(melData, _melBins, numFrames, 1, "a.conv1d.0");
            int f0Out = (_melBins + 2 - 3) / 2 + 1;
            int t0Out = (numFrames + 2 - 3) / 2 + 1;
            int c0Out = GetConvOutChannels("a.conv1d.0");

            // Conv1
            var conv1Out = Conv2DBlock(conv0Out, f0Out, t0Out, c0Out, "a.conv1d.1");
            int f1Out = (f0Out + 2 - 3) / 2 + 1;
            int t1Out = (t0Out + 2 - 3) / 2 + 1;
            int c1Out = GetConvOutChannels("a.conv1d.1");

            Console.Write($" conv=[{f1Out},{t1Out},{c1Out}]");

            // conv1Out layout: element(f, t, c) = conv1Out[f + t * f1Out + c * f1Out * t1Out]
            // GGML: Permute [F, T, C] → [C, F, T], then Reshape to [C*F, T]
            // After reshape: element(cf, t) where cf = c + f * C
            // In TensorSharp [T, sscpDim]: projected[t * sscpDim + cf]
            int sscpDim = c1Out * f1Out;
            float[] projected = new float[t1Out * sscpDim];

            for (int t = 0; t < t1Out; t++)
                for (int c = 0; c < c1Out; c++)
                    for (int f = 0; f < f1Out; f++)
                    {
                        int cf = c + f * c1Out;
                        projected[t * sscpDim + cf] = conv1Out[f + t * f1Out + c * f1Out * t1Out];
                    }

            Console.Write($" reshape=[{t1Out},{sscpDim}]");

            // SSCP linear projection to conformer hidden size (_hiddenSize).
            // The SSCP projection weight name depends on the converter:
            //   Ollama GGUF: a.pre_encode.out.weight [hiddenSize, hiddenSize]
            //   mmproj/Unsloth: a.input_projection.weight [hiddenSize, hiddenSize]
            //     (mmproj's a.pre_encode.out.weight is actually the FC layer, not SSCP)
            var projTensor = new Tensor(_allocator, DType.Float32, t1Out, sscpDim);
            projTensor.SetElementsAsFloat(projected);

            string sscpWeightName = _weights.ContainsKey("a.input_projection.weight")
                ? "a.input_projection.weight"
                : "a.pre_encode.out.weight";

            Tensor hiddenTensor;
            int hidDim;
            if (_weights.TryGetValue(sscpWeightName, out var sscpWeight))
            {
                int outDim = (int)sscpWeight.Sizes[0];
                hidDim = outDim;
                hiddenTensor = new Tensor(_allocator, DType.Float32, t1Out, hidDim);
                using (var wT = sscpWeight.Transpose())
                    Ops.Addmm(hiddenTensor, 0, hiddenTensor, 1f, projTensor, wT);
                projTensor.Dispose();

                string biasName = sscpWeightName.Replace(".weight", ".bias");
                if (_weights.TryGetValue(biasName, out var sscpBias))
                    AddBias(hiddenTensor, sscpBias, t1Out, hidDim);
            }
            else
            {
                hidDim = sscpDim;
                hiddenTensor = projTensor;
            }

            int seqLen = t1Out;
            Console.Write($" proj=[{seqLen},{hidDim}]");

            // Build causal-valid mask
            float[] causalMask = BuildCausalValidMask();

            // Conformer blocks
            for (int i = 0; i < _numLayers; i++)
            {
                Console.Write($"\r  Audio conformer block {i + 1}/{_numLayers}...                    ");
                hiddenTensor = ConformerBlock(hiddenTensor, i, seqLen, hidDim, causalMask);
            }
            Console.Write("\r  Audio conformer done.                                         \n");

            // Output projection: Ollama uses a.output_proj.weight, neither file has it currently.
            if (_weights.TryGetValue("a.output_proj.weight", out var outProjWeight))
            {
                int outDim = (int)outProjWeight.Sizes[0];
                var outProj = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                using (var wT = outProjWeight.Transpose())
                    Ops.Addmm(outProj, 0, outProj, 1f, hiddenTensor, wT);
                if (_weights.TryGetValue("a.output_proj.bias", out var outProjBias))
                    AddBias(outProj, outProjBias, seqLen, outDim);
                hiddenTensor.Dispose();
                hiddenTensor = outProj;
                hidDim = outDim;
            }

            // Audio multimodal projector FC layer.
            // Ollama GGUF: mm.a.fc.weight [hiddenSize, fcDim]
            // mmproj/Unsloth: a.pre_encode.out.weight [hiddenSize, fcDim] (repurposed)
            string fcWeightName = _weights.ContainsKey("mm.a.fc.weight")
                ? "mm.a.fc.weight"
                : (_weights.ContainsKey("a.pre_encode.out.weight") &&
                   (int)_weights["a.pre_encode.out.weight"].Sizes[0] != _hiddenSize
                    ? "a.pre_encode.out.weight"
                    : null);
            if (fcWeightName != null && _weights.TryGetValue(fcWeightName, out var fcWeight))
            {
                int fcOutDim = (int)fcWeight.Sizes[0];
                var fcOut = new Tensor(_allocator, DType.Float32, seqLen, fcOutDim);
                using (var wT = fcWeight.Transpose())
                    Ops.Addmm(fcOut, 0, fcOut, 1f, hiddenTensor, wT);

                string fcBiasName = fcWeightName.Replace(".weight", ".bias");
                if (_weights.TryGetValue(fcBiasName, out var fcBias))
                    AddBias(fcOut, fcBias, seqLen, fcOutDim);
                hiddenTensor.Dispose();
                hiddenTensor = fcOut;
                hidDim = fcOutDim;
            }

            // Unweighted RMSNorm
            ApplyUnweightedRMSNorm(hiddenTensor, seqLen, hidDim);

            // Embedding projection to text hidden size
            if (_weights.ContainsKey("mm.a.input_projection.weight"))
            {
                var old = hiddenTensor;
                hiddenTensor = AudioClippableLinearForward(hiddenTensor, "mm.a.input_projection", seqLen);
                old.Dispose();
                hidDim = (int)hiddenTensor.Sizes[1];
            }

            Console.WriteLine($"Audio encoder: [{seqLen}, {hidDim}] done");
            return hiddenTensor;
        }

        private int GetConvOutChannels(string prefix)
        {
            var w = _weights[$"{prefix}.weight"];
            return (int)w.Sizes[0]; // TensorSharp reversed: GGUF [kW, kH, C_in, C_out] -> TS [C_out, C_in, kH, kW]
        }

        private unsafe float[] Conv2DBlock(float[] input, int inW, int inH, int inC, string prefix)
        {
            // GGML layout: [ne0=F(freq/W), ne1=T(time/H), ne2=C]
            // In flat array: element(f, t, c) = input[f + t * inW + c * inW * inH]
            // melData is [numFrames, melBins] row-major = [T, F] = GGML [F, T] which gives
            //   melData[t * F + f] = input[f + t * F] ✓
            var kernelTensor = _weights[$"{prefix}.weight"];
            int cOut = (int)kernelTensor.Sizes[0];
            int cIn = (int)kernelTensor.Sizes[1];
            int kH = (int)kernelTensor.Sizes[2];
            int kW = (int)kernelTensor.Sizes[3];

            if (cIn != inC)
                throw new Exception($"Conv2D channel mismatch: kernel expects {cIn} but input has {inC}");

            int stride = 2, pad = 1;
            int outW = (inW + 2 * pad - kW) / stride + 1;
            int outH = (inH + 2 * pad - kH) / stride + 1;

            float* kPtr = GetFloatPtr(kernelTensor);
            int inSpatial = inW * inH;
            int outSpatial = outW * outH;
            float[] output = new float[outSpatial * cOut];

            for (int oc = 0; oc < cOut; oc++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float sum = 0;
                        for (int ic = 0; ic < cIn; ic++)
                        {
                            for (int ky = 0; ky < kH; ky++)
                            {
                                for (int kx = 0; kx < kW; kx++)
                                {
                                    int ih = oh * stride - pad + ky;
                                    int iw = ow * stride - pad + kx;
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
                                    {
                                        float pixel = input[iw + ih * inW + ic * inSpatial];
                                        int kIdx = ((oc * cIn + ic) * kH + ky) * kW + kx;
                                        sum += pixel * kPtr[kIdx];
                                    }
                                }
                            }
                        }
                        output[ow + oh * outW + oc * outSpatial] = sum;
                    }
                }
            }

            // LayerNorm over channels for each spatial position.
            // Output layout: element(f', t', c) = output[f' + t' * outW + c * outSpatial]
            if (_weights.TryGetValue($"{prefix}.norm.weight", out var normWeight))
            {
                float* nw = GetFloatPtr(normWeight);
                float* nb = _weights.TryGetValue($"{prefix}.norm.bias", out var normBias)
                    ? GetFloatPtr(normBias) : null;

                for (int s = 0; s < outSpatial; s++)
                {
                    float mean = 0;
                    for (int c = 0; c < cOut; c++)
                        mean += output[s + c * outSpatial];
                    mean /= cOut;

                    float var_ = 0;
                    for (int c = 0; c < cOut; c++)
                    {
                        float d = output[s + c * outSpatial] - mean;
                        var_ += d * d;
                    }
                    var_ /= cOut;
                    float invStd = 1f / MathF.Sqrt(var_ + _eps);

                    for (int c = 0; c < cOut; c++)
                    {
                        float val = (output[s + c * outSpatial] - mean) * invStd;
                        val *= nw[c];
                        if (nb != null) val += nb[c];
                        output[s + c * outSpatial] = val;
                    }
                }
            }

            // ReLU
            for (int i = 0; i < output.Length; i++)
                if (output[i] < 0) output[i] = 0;

            return output;
        }

        private Tensor ConformerBlock(Tensor x, int blockIdx, int seqLen, int hidDim, float[] causalMask)
        {
            string prefix = $"a.blk.{blockIdx}";

            x = ForwardFFW(x, prefix, "ffn_norm", "ffn_up", "ffn_down", "ffn_post_norm", seqLen, hidDim);
            x = ForwardAttention(x, prefix, seqLen, hidDim, causalMask);
            x = ForwardLightConv(x, prefix, seqLen, hidDim);
            x = ForwardFFW(x, prefix, "ffn_norm_1", "ffn_up_1", "ffn_down_1", "ffn_post_norm_1", seqLen, hidDim);

            Clamp(x, -_gradClip, _gradClip);
            string blockNormName = ResolveName(prefix, "block_norm") + ".weight";
            var old = x;
            x = ApplyRMSNorm(x, blockNormName, seqLen, hidDim);
            old.Dispose();

            return x;
        }

        private Tensor ForwardFFW(Tensor x, string prefix, string normName,
            string upName, string downName, string postNormName, int seqLen, int hidDim)
        {
            var residual = x;

            var clamped = CloneTensor(x, seqLen, hidDim);
            Clamp(clamped, -_gradClip, _gradClip);

            var normed = ApplyRMSNorm(clamped, $"{prefix}.{normName}.weight", seqLen, hidDim);
            clamped.Dispose();

            var upOut = AudioClippableLinearForward(normed, $"{prefix}.{upName}", seqLen);
            normed.Dispose();
            int upDim = (int)upOut.Sizes[1];

            // SiLU
            ApplySiLU(upOut);

            var downOut = AudioClippableLinearForward(upOut, $"{prefix}.{downName}", seqLen);
            upOut.Dispose();

            Clamp(downOut, -_gradClip, _gradClip);

            var postNormed = ApplyRMSNorm(downOut, $"{prefix}.{postNormName}.weight", seqLen, hidDim);
            downOut.Dispose();

            // result = residual + postNormed * residualWeight
            Ops.AddMulV(postNormed, residual, postNormed, _residualWeight);
            residual.Dispose();

            return postNormed;
        }

        private unsafe Tensor ForwardAttention(Tensor x, string prefix, int seqLen, int hidDim, float[] causalMask)
        {
            var residual = x;

            var clamped = CloneTensor(x, seqLen, hidDim);
            Clamp(clamped, -_gradClip, _gradClip);

            string preNormName = ResolveName(prefix, "attn_pre_norm") + ".weight";
            var normed = ApplyRMSNorm(clamped, preNormName, seqLen, hidDim);
            clamped.Dispose();

            // QKV projections
            var q = AudioClippableLinearForward(normed, $"{prefix}.attn_q", seqLen);
            var k = AudioClippableLinearForward(normed, $"{prefix}.attn_k", seqLen);
            var v = AudioClippableLinearForward(normed, $"{prefix}.attn_v", seqLen);
            normed.Dispose();

            // Per-dim scaling for Q
            float qScale = (float)(Math.Pow(_headDim, -0.5) / Math.Log(2));
            Ops.Mul(q, q, qScale);
            if (_weights.TryGetValue($"{prefix}.per_dim_scale.weight", out var perDimScale))
                ApplyPerDimScale(q, perDimScale, seqLen);

            // Key scaling
            float kScale = (float)(Math.Log(1 + Math.E) / Math.Log(2));
            Ops.Mul(k, k, kScale);

            // Build position embeddings
            int maxSpan = _maxPast + _maxFuture + 1;
            float[] posEmb = BuildPositionEmbeddings(prefix, maxSpan);

            // Block-local chunked attention
            int numChunks = (seqLen + _chunkSize - 1) / _chunkSize;
            int paddedLen = numChunks * _chunkSize;
            int padT = paddedLen - seqLen;

            // Reshape Q/K/V from [seqLen, hidDim] to arrays for processing
            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(k);
            float* vPtr = GetFloatPtr(v);

            // Pad arrays if needed
            float[] qArr = ExtractAndPad(qPtr, seqLen, hidDim, paddedLen);
            float[] kArr = ExtractAndPad(kPtr, seqLen, hidDim, paddedLen);
            float[] vArr = ExtractAndPad(vPtr, seqLen, hidDim, paddedLen);
            q.Dispose(); k.Dispose(); v.Dispose();

            // Pad K/V for context extraction
            int padLeft = _maxPast;
            int padRight = _maxFuture + _chunkSize - 1;
            int kPaddedLen = padLeft + paddedLen + padRight;
            float[] kPadded = new float[kPaddedLen * hidDim];
            float[] vPadded = new float[kPaddedLen * hidDim];
            Array.Copy(kArr, 0, kPadded, padLeft * hidDim, paddedLen * hidDim);
            Array.Copy(vArr, 0, vPadded, padLeft * hidDim, paddedLen * hidDim);

            float[] attnOutput = new float[paddedLen * hidDim];

            for (int u = 0; u < numChunks; u++)
            {
                ChunkedAttention(qArr, kPadded, vPadded, posEmb, causalMask, attnOutput,
                    u, seqLen, hidDim, maxSpan, padLeft);
            }

            // Trim to original seqLen
            var result = new Tensor(_allocator, DType.Float32, seqLen, hidDim);
            float* rPtr = GetFloatPtr(result);
            fixed (float* aPtr = attnOutput)
                Buffer.MemoryCopy(aPtr, rPtr, seqLen * hidDim * 4, seqLen * hidDim * 4);

            // Output projection
            var projected = AudioClippableLinearForward(result, $"{prefix}.attn_out", seqLen);
            result.Dispose();

            Clamp(projected, -_gradClip, _gradClip);
            string attnPostNormName = ResolveName(prefix, "attn_post_norm") + ".weight";
            var postNormed = ApplyRMSNorm(projected, attnPostNormName, seqLen, hidDim);
            projected.Dispose();

            Ops.Add(postNormed, postNormed, residual);
            residual.Dispose();

            return postNormed;
        }

        private unsafe void ChunkedAttention(float[] qArr, float[] kPadded, float[] vPadded,
            float[] posEmb, float[] causalMask, float[] attnOutput,
            int chunkIdx, int seqLen, int hidDim, int maxSpan, int padLeft)
        {
            int cs = _chunkSize;
            int ctx = _contextSize;

            for (int h = 0; h < _numHeads; h++)
            {
                for (int qi = 0; qi < cs; qi++)
                {
                    int globalQIdx = chunkIdx * cs + qi;
                    if (globalQIdx >= seqLen)
                    {
                        // Padded position - zero output
                        continue;
                    }

                    float[] logits = new float[ctx];

                    for (int ci = 0; ci < ctx; ci++)
                    {
                        // Content-content: q[qi] dot k[ci]
                        float dotCC = 0;
                        int qOffset = globalQIdx * hidDim + h * _headDim;
                        int kGlobalIdx = chunkIdx * cs + ci; // position in kPadded
                        int kOffset = kGlobalIdx * hidDim + h * _headDim;

                        for (int d = 0; d < _headDim; d++)
                            dotCC += qArr[qOffset + d] * kPadded[kOffset + d];

                        // Content-position: q[qi] dot posEmb[relPos]
                        float dotCP = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            int posIdx = RelativeShiftIndex(qi, ci, maxSpan);
                            if (posIdx >= 0 && posIdx < maxSpan)
                                dotCP += qArr[qOffset + d] * posEmb[(posIdx * _numHeads + h) * _headDim + d];
                        }

                        logits[ci] = dotCC + dotCP;

                        // Logit softcap
                        logits[ci] = MathF.Tanh(logits[ci] / _logitCap) * _logitCap;

                        // Apply mask
                        int actualTime = chunkIdx * cs + ci - padLeft;
                        bool causalOK = causalMask[qi * ctx + ci] > 0;
                        bool validOK = actualTime >= 0 && actualTime < seqLen;
                        if (!causalOK || !validOK)
                            logits[ci] = -1e9f;
                    }

                    // Softmax
                    float maxLogit = float.NegativeInfinity;
                    for (int ci = 0; ci < ctx; ci++)
                        if (logits[ci] > maxLogit) maxLogit = logits[ci];
                    float sumExp = 0;
                    for (int ci = 0; ci < ctx; ci++)
                    {
                        logits[ci] = MathF.Exp(logits[ci] - maxLogit);
                        sumExp += logits[ci];
                    }
                    float invSum = 1f / sumExp;
                    for (int ci = 0; ci < ctx; ci++)
                        logits[ci] *= invSum;

                    // Weighted sum of values
                    int outOffset = globalQIdx * hidDim + h * _headDim;
                    for (int d = 0; d < _headDim; d++)
                    {
                        float sum = 0;
                        for (int ci = 0; ci < ctx; ci++)
                        {
                            int vGlobalIdx = chunkIdx * cs + ci;
                            sum += logits[ci] * vPadded[vGlobalIdx * hidDim + h * _headDim + d];
                        }
                        attnOutput[outOffset + d] = sum;
                    }
                }
            }
        }

        private int RelativeShiftIndex(int queryInChunk, int contextIdx, int maxSpan)
        {
            // Maps (queryInChunk, contextIdx) to position embedding index.
            // Matches the GGML relative shift trick: posIdx = contextIdx - queryInChunk.
            // posEmb[posIdx] encodes relPos = maxPast - posIdx.
            int posIdx = contextIdx - queryInChunk;
            if (posIdx < 0 || posIdx >= maxSpan) return -1;
            return posIdx;
        }

        private float[] BuildPositionEmbeddings(string prefix, int maxSpan)
        {
            int halfDim = _hiddenSize / 2;
            double logInc = Math.Log(10000.0) / Math.Max(halfDim - 1, 1);

            float[] sinEmb = new float[maxSpan * _hiddenSize];
            for (int p = 0; p < maxSpan; p++)
            {
                double relPos = _maxPast - p;
                for (int d = 0; d < halfDim; d++)
                {
                    double angle = relPos * Math.Exp(-d * logInc);
                    sinEmb[p * _hiddenSize + d] = (float)Math.Sin(angle);
                    sinEmb[p * _hiddenSize + halfDim + d] = (float)Math.Cos(angle);
                }
            }

            string relKey = ResolveName(prefix, "attn_k_rel") + ".weight";
            if (!_weights.TryGetValue(relKey, out var relWeight))
                return sinEmb;

            int relOutDim = (int)relWeight.Sizes[0];
            int inDim = (int)relWeight.Sizes[1];

            using var sinTensor = new Tensor(_allocator, DType.Float32, maxSpan, _hiddenSize);
            sinTensor.SetElementsAsFloat(sinEmb);

            using var sinSlice = sinTensor.Narrow(1, 0, inDim);
            using var sinContig = Ops.NewContiguous(sinSlice);
            using var wT = relWeight.Transpose();
            var result = new Tensor(_allocator, DType.Float32, maxSpan, relOutDim);
            Ops.Addmm(result, 0, result, 1f, sinContig, wT);

            float[] projected = new float[maxSpan * relOutDim];
            result.CopyToArray(projected);
            result.Dispose();

            return projected;
        }

        private unsafe Tensor ForwardLightConv(Tensor x, string prefix, int seqLen, int hidDim)
        {
            var residual = x;

            var normed = ApplyRMSNorm(x, $"{prefix}.conv_norm.weight", seqLen, hidDim);

            // Pointwise conv1 (doubles channels)
            var pw1Out = AudioClippableLinearForward(normed, $"{prefix}.conv_pw1", seqLen);
            normed.Dispose();
            int pw1Dim = (int)pw1Out.Sizes[1];

            // GLU: split in half, sigmoid gate
            int halfDim = pw1Dim / 2;
            Tensor dataHalf, gateHalf;
            if (seqLen == 1)
            {
                dataHalf = pw1Out.Narrow(1, 0, halfDim);
                gateHalf = pw1Out.Narrow(1, halfDim, halfDim);
            }
            else
            {
                using (var dv = pw1Out.Narrow(1, 0, halfDim))
                    dataHalf = Ops.NewContiguous(dv);
                using (var gv = pw1Out.Narrow(1, halfDim, halfDim))
                    gateHalf = Ops.NewContiguous(gv);
            }
            var gluOut = Ops.SigmoidMul(null, dataHalf, gateHalf);
            dataHalf.Dispose();
            gateHalf.Dispose();
            pw1Out.Dispose();

            // Depthwise Conv1d (kernel size 5)
            if (_weights.TryGetValue($"{prefix}.conv_dw.weight", out var dwWeight))
            {
                int kSize = (int)dwWeight.Sizes[1];
                float* dwPtr = GetFloatPtr(dwWeight);
                float* gluPtr = GetFloatPtr(gluOut);

                var convOut = new Tensor(_allocator, DType.Float32, seqLen, halfDim);
                float* convPtr = GetFloatPtr(convOut);

                for (int t = 0; t < seqLen; t++)
                {
                    for (int d = 0; d < halfDim; d++)
                    {
                        float sum = 0;
                        for (int k = 0; k < kSize; k++)
                        {
                            int shift = kSize - 1 - k;
                            int srcT = t - shift;
                            float val = srcT >= 0 ? gluPtr[srcT * halfDim + d] : 0f;
                            sum += val * dwPtr[d * kSize + k];
                        }
                        convPtr[t * halfDim + d] = sum;
                    }
                }
                gluOut.Dispose();
                gluOut = convOut;
            }

            Clamp(gluOut, -_gradClip, _gradClip);
            var normConv = ApplyRMSNorm(gluOut, $"{prefix}.norm_conv.weight", seqLen, halfDim);
            gluOut.Dispose();

            // SiLU
            ApplySiLU(normConv);

            // Pointwise conv2
            var pw2Out = AudioClippableLinearForward(normConv, $"{prefix}.conv_pw2", seqLen);
            normConv.Dispose();

            // Residual
            Ops.Add(pw2Out, pw2Out, residual);
            residual.Dispose();

            return pw2Out;
        }

        #region Helpers

        private string ResolveName(string blockPrefix, string shortName)
        {
            if (_useOllamaNames)
            {
                return shortName switch
                {
                    "attn_pre_norm" => $"{blockPrefix}.ln1",
                    "block_norm" => $"{blockPrefix}.layer_pre_norm",
                    "attn_post_norm" => $"{blockPrefix}.ln2",
                    "attn_k_rel" => $"{blockPrefix}.linear_pos",
                    _ => $"{blockPrefix}.{shortName}"
                };
            }
            return shortName switch
            {
                "block_norm" => $"{blockPrefix}.ln2",
                _ => $"{blockPrefix}.{shortName}"
            };
        }

        private Tensor AudioClippableLinearForward(Tensor input, string prefix, int seqLen)
        {
            string weightName = $"{prefix}.weight";
            var weight = _weights[weightName];
            int outDim = (int)weight.Sizes[0];

            bool hasClamp = _clampParams.TryGetValue(prefix, out var cp) && cp.HasClamp;

            Tensor src = input;
            if (hasClamp)
            {
                src = CloneTensor(input, seqLen, (int)input.Sizes[1]);
                Clamp(src, cp.InMin, cp.InMax);
            }

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            using (var wT = weight.Transpose())
                Ops.Addmm(result, 0, result, 1f, src, wT);

            if (hasClamp && src != input) src.Dispose();

            // Add bias if present
            if (_weights.TryGetValue($"{prefix}.bias", out var bias))
                AddBias(result, bias, seqLen, outDim);

            if (hasClamp)
                Clamp(result, cp.OutMin, cp.OutMax);

            return result;
        }

        private void AddBias(Tensor t, Tensor bias, int seqLen, int dim)
        {
            Ops.Add(t, t, bias);
        }

        private Tensor ApplyRMSNorm(Tensor input, string weightName, int seqLen, int dim)
        {
            if (!_weights.TryGetValue(weightName, out var normWeight))
                return Ops.NewContiguous(input);

            return Ops.RMSNorm(null, input, normWeight, null, _eps);
        }

        private void ApplyUnweightedRMSNorm(Tensor data, int seqLen, int dim)
        {
            if (_onesForNorm == null || (int)_onesForNorm.Sizes[0] != dim)
            {
                _onesForNorm?.Dispose();
                _onesForNorm = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(_onesForNorm, 1f);
            }
            Ops.RMSNorm(data, data, _onesForNorm, null, _eps);
        }

        private void ApplyPerDimScale(Tensor q, Tensor perDimScale, int seqLen)
        {
            int dim = (int)q.Sizes[1];
            using var reshaped = q.View(seqLen * _numHeads, _headDim);
            Ops.Mul(reshaped, reshaped, perDimScale);
        }

        private void ApplySiLU(Tensor t)
        {
            Ops.SiLU(t, t);
        }

        private unsafe void Clamp(Tensor t, float min, float max)
        {
            float* ptr = GetFloatPtr(t);
            int count = (int)t.ElementCount();
            for (int i = 0; i < count; i++)
            {
                if (ptr[i] < min) ptr[i] = min;
                else if (ptr[i] > max) ptr[i] = max;
            }
        }

        private Tensor CloneTensor(Tensor src, int rows, int cols)
        {
            return Ops.NewContiguous(src);
        }

        private unsafe float[] ExtractAndPad(float* ptr, int seqLen, int dim, int paddedLen)
        {
            float[] result = new float[paddedLen * dim];
            fixed (float* dst = result)
                Buffer.MemoryCopy(ptr, dst, seqLen * dim * 4, seqLen * dim * 4);
            return result;
        }

        private float[] BuildCausalValidMask()
        {
            int upperDiag = _maxPast + _maxFuture;
            float[] result = new float[_chunkSize * _contextSize];
            for (int r = 0; r < _chunkSize; r++)
            {
                for (int c = 0; c < _contextSize; c++)
                {
                    bool lower = r <= c;
                    bool upper = c <= r + upperDiag;
                    result[r * _contextSize + c] = (lower && upper) ? 1f : 0f;
                }
            }
            return result;
        }

        private static unsafe float* GetFloatPtr(Tensor t)
        {
            if (t.Storage is TensorSharp.GGML.GgmlStorage gs)
                return (float*)gs.PtrAtElement(t.StorageOffset);
            if (t.Storage is CpuStorage cs)
                return (float*)cs.PtrAtElement(t.StorageOffset);
            throw new NotSupportedException("Requires GgmlStorage or CpuStorage");
        }

        #endregion

        public void Dispose()
        {
            _onesForNorm?.Dispose();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
        }
    }
}
