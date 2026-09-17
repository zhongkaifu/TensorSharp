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
using System.Linq;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Pixtral-style vision encoder for Mistral 3.
    /// Architecture:
    /// - Conv2D patch embedding
    /// - RMSNorm on patch embeddings (encoder_norm)
    /// - 2D RoPE positional embeddings (computed on-the-fly)
    /// - Transformer blocks with RMSNorm, gated MLP (GELU, or SiLU when the file declares clip.use_silu)
    /// - Patch merger with spatial merge
    /// - Multi-modal projector: RMSNorm → PatchMerger → Linear → GELU → Linear
    /// </summary>
    public class Mistral3VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, QuantizedWeight> _quantWeights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;
        // Cooperative GpuComputeLock yielding (see Gemma4VisionEncoder).
        private ModelBase _hostModel;
        public void SetHostModel(ModelBase model) => _hostModel = model;

        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly float _visionRopeBase;
        private readonly int _spatialMergeSize;

        // Multi-modal projector config
        private readonly float _textEps;

        /// <summary>Activation of the gated vision MLP. HF <c>PixtralVisionConfig.hidden_act</c>,
        /// recorded by llama.cpp as <c>clip.use_gelu</c> / <c>clip.use_silu</c>.</summary>
        internal VisionFfnActivation FfnActivation { get; }

        internal enum VisionFfnActivation { Gelu, Silu }

        public int PatchSize => _patchSize;
        public int SpatialMergeSize => _spatialMergeSize;
        public int ImageSize => _imageSize;

        public Mistral3VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            var gguf = new GgufFile(mmProjPath);

            _imageSize = (int)gguf.GetUint32("vision.image_size",
                          (uint)gguf.GetUint32("clip.vision.image_size", 1540));
            _patchSize = (int)gguf.GetUint32("vision.patch_size",
                          (uint)gguf.GetUint32("clip.vision.patch_size", 14));
            _hiddenSize = (int)gguf.GetUint32("vision.embedding_length",
                           (uint)gguf.GetUint32("clip.vision.embedding_length", 1024));
            _numHeads = (int)gguf.GetUint32("vision.attention.head_count",
                         (uint)gguf.GetUint32("clip.vision.attention.head_count", 16));
            _headDim = (int)gguf.GetUint32("vision.attention.key_length",
                        (uint)(_hiddenSize / _numHeads));
            _blockCount = (int)gguf.GetUint32("vision.block_count",
                           (uint)gguf.GetUint32("clip.vision.block_count", 24));
            _eps = gguf.GetFloat32("vision.attention.layer_norm_epsilon",
                   gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f));
            _visionRopeBase = gguf.GetFloat32("vision.rope.freq_base", 10000.0f);
            _spatialMergeSize = (int)gguf.GetUint32("spatial_merge_size",
                                 (uint)gguf.GetUint32("clip.vision.spatial_merge_size", 2));
            _textEps = gguf.GetFloat32("text_config.rms_norm_eps", 1e-5f);
            FfnActivation = ResolveFfnActivation(gguf);

            Console.WriteLine($"Mistral3 Vision: imageSize={_imageSize}, patchSize={_patchSize}, " +
                $"hidden={_hiddenSize}, heads={_numHeads}, headDim={_headDim}, " +
                $"blocks={_blockCount}, ropeBase={_visionRopeBase}, mergeSize={_spatialMergeSize}, ffn={FfnActivation}");

            LoadWeights(gguf);
            gguf.Dispose();
            try
            {
                ValidateLoadedWeights(mmProjPath);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        // llama.cpp's clip converter (the mmproj bartowski and most GGUF repos ship
        // beside Mistral Small 3.x) and Ollama's converter name the same Pixtral
        // tensors differently. The encoder reads the Ollama names, so a llama.cpp file
        // is renamed on load; before this every image request on such a file threw
        // KeyNotFoundException('v.patch_conv.weight') and returned HTTP 500.
        private static readonly (Regex From, string To)[] LlamaCppClipTensorNames =
        {
            (new Regex(@"^v\.patch_embd\.(weight|bias)$"), "v.patch_conv.$1"),
            (new Regex(@"^v\.pre_ln\.weight$"), "v.encoder_norm.weight"),
            (new Regex(@"^v\.blk\.(\d+)\.ln1\.weight$"), "v.blk.$1.attn_norm.weight"),
            (new Regex(@"^v\.blk\.(\d+)\.ln2\.weight$"), "v.blk.$1.ffn_norm.weight"),
            (new Regex(@"^v\.blk\.(\d+)\.attn_out\.weight$"), "v.blk.$1.attn_output.weight"),
            (new Regex(@"^mm\.input_norm\.weight$"), "mm.norm.weight"),
            (new Regex(@"^mm\.patch_merger\.weight$"), "mm.patch_merger.merging_layer.weight"),
            (new Regex(@"^mm\.1\.(weight|bias)$"), "mm.linear_1.$1"),
            (new Regex(@"^mm\.2\.(weight|bias)$"), "mm.linear_2.$1"),
        };

        /// <summary>
        /// Mistral Small 3.1's vision tower is GELU-gated (<c>hidden_act = "gelu"</c>, which
        /// is also <c>PixtralVisionConfig</c>'s default); the file says so through
        /// <c>clip.use_gelu</c>. A file that declares <c>clip.use_silu</c> gets SiLU. The
        /// encoder used SiLU unconditionally, which alone left the model unable to read the
        /// image. Ollama's converter records no activation, so its files take the HF default.
        /// </summary>
        internal static VisionFfnActivation ResolveFfnActivation(GgufFile gguf)
        {
            bool gelu = gguf.GetBool("clip.use_gelu", false);
            bool silu = gguf.GetBool("clip.use_silu", false);
            if (gelu && silu)
                throw new NotSupportedException("Pixtral projector declares both clip.use_gelu and clip.use_silu.");
            return silu ? VisionFfnActivation.Silu : VisionFfnActivation.Gelu;
        }

        /// <summary>
        /// llama.cpp's converter permutes the vision Q/K projections from HF's rotate-half
        /// layout into interleaved pairs (<c>LlamaModel.permute</c>) because its 2D RoPE
        /// rotates adjacent dims; this encoder rotates HF's (d, d + headDim/2) pairs, so a
        /// llama.cpp file's rows are put back: HF row <c>half * headDim/2 + j</c> of each
        /// head is file row <c>2j + half</c>.
        /// </summary>
        internal static float[] UnpermuteInterleavedRows(float[] weights, int numHeads, int headDim)
        {
            int rows = numHeads * headDim;
            if (weights.Length % rows != 0 || headDim % 2 != 0)
                throw new NotSupportedException($"Vision Q/K projection of {weights.Length} values does not split into {numHeads} heads of {headDim}.");
            int cols = weights.Length / rows;
            int halfDim = headDim / 2;
            var result = new float[weights.Length];
            for (int h = 0; h < numHeads; h++)
            {
                for (int j = 0; j < halfDim; j++)
                {
                    for (int half = 0; half < 2; half++)
                    {
                        int src = (h * headDim + 2 * j + half) * cols;
                        int dst = (h * headDim + half * halfDim + j) * cols;
                        Array.Copy(weights, src, result, dst, cols);
                    }
                }
            }
            return result;
        }

        private static readonly Regex VisionQkProjection = new(@"^v\.blk\.\d+\.attn_[qk]\.(weight|bias)$");

        /// <summary>Maps a llama.cpp clip tensor name to the name this encoder reads;
        /// every other name (Ollama layout, or tensors the encoder does not use such as
        /// <c>v.token_embd.img_break</c>) is returned unchanged.</summary>
        internal static string CanonicalTensorName(string name)
        {
            foreach (var (from, to) in LlamaCppClipTensorNames)
            {
                if (from.IsMatch(name))
                    return from.Replace(name, to);
            }
            return name;
        }

        /// <summary>
        /// The encoder's helpers skip a missing norm and return null for a missing
        /// linear, so a tensor-name mismatch would otherwise surface as a crash deep in
        /// the first image request (or, for a norm, as silently wrong embeddings).
        /// Checked once at load instead. Biases are refused because the linear and
        /// conv paths here apply none (Mistral Small 3.x has none).
        /// </summary>
        private void ValidateLoadedWeights(string mmProjPath)
        {
            var required = new List<string>
            {
                "v.patch_conv.weight", "v.encoder_norm.weight", "mm.norm.weight",
                "mm.patch_merger.merging_layer.weight", "mm.linear_1.weight", "mm.linear_2.weight",
            };
            for (int i = 0; i < _blockCount; i++)
            {
                foreach (string t in new[] { "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down" })
                    required.Add($"v.blk.{i}.{t}.weight");
            }
            var missing = required.Where(n => !_weights.ContainsKey(n)).ToList();
            if (missing.Count > 0)
            {
                throw new NotSupportedException(
                    $"'{System.IO.Path.GetFileName(mmProjPath)}' is not a Pixtral (Mistral 3) projector this encoder can run: " +
                    $"missing {string.Join(", ", missing.Take(6))}{(missing.Count > 6 ? $" and {missing.Count - 6} more" : "")}. " +
                    "Accepted layouts are llama.cpp's clip names (v.patch_embd, v.pre_ln, v.blk.N.ln1, mm.patch_merger, mm.1, mm.2) " +
                    "and Ollama's (v.patch_conv, v.encoder_norm, v.blk.N.attn_norm, mm.patch_merger.merging_layer, mm.linear_1, mm.linear_2).");
            }
            var biases = _weights.Keys.Where(n => n.EndsWith(".bias", StringComparison.Ordinal)
                && (n.StartsWith("v.blk.", StringComparison.Ordinal) || n.StartsWith("mm.", StringComparison.Ordinal))).ToList();
            if (biases.Count > 0)
            {
                throw new NotSupportedException(
                    $"'{System.IO.Path.GetFileName(mmProjPath)}' has linear biases ({string.Join(", ", biases.Take(4))}) that the " +
                    "Mistral 3 vision encoder does not apply; refusing it rather than producing wrong image embeddings.");
            }
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading Mistral3 vision encoder weights...");
            int count = 0;
            bool llamaCppLayout = gguf.Tensors.Keys.Any(n => CanonicalTensorName(n) != n);
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                string name = CanonicalTensorName(info.Name);
                long numElements = info.NumElements;

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                byte[] raw = gguf.ReadTensorData(info);
                float[] f32 = new float[numElements];
                if (info.Type == GgmlTensorType.F32)
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                else
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                if (llamaCppLayout && VisionQkProjection.IsMatch(name))
                    f32 = UnpermuteInterleavedRows(f32, _numHeads, _headDim);

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[name] = tensor;
                count++;
            }
            Console.WriteLine($" done ({count} tensors)");
        }

        /// <summary>
        /// Encode an image into vision embeddings ready for the text model.
        /// Input: normalized pixel data, image dimensions.
        /// Output: Tensor of shape [numOutputTokens, textHiddenSize].
        /// </summary>
        public unsafe Tensor Encode(float[] pixelValues, int imageWidth, int imageHeight)
        {
            int numPatchesW = imageWidth / _patchSize;
            int numPatchesH = imageHeight / _patchSize;
            int numPatches = numPatchesW * numPatchesH;

            // Patch embedding via Conv2D
            var hidden = PatchEmbed(pixelValues, imageWidth, imageHeight, numPatchesW, numPatchesH);

            // Encoder norm
            using var normed = RMSNormOp(hidden, "v.encoder_norm.weight");
            hidden.Dispose();
            hidden = Ops.NewContiguous(normed);

            // 2D RoPE positional embeddings
            var (cos, sin) = Compute2DRoPE(numPatchesW, numPatchesH);

            for (int i = 0; i < _blockCount; i++)
            {
                Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                hidden = EncoderBlock(hidden, i, numPatches, cos, sin);
                // Yield GpuComputeLock between encoder blocks (see
                // Gemma4VisionEncoder).
                _hostModel?.YieldGpuComputeLock();
            }
            Console.WriteLine(" done");

            cos.Dispose();
            sin.Dispose();

            // Multi-modal projector
            var projected = MultiModalProject(hidden, numPatchesW, numPatchesH);
            hidden.Dispose();

            return projected;
        }

        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgW, int imgH,
            int patchesW, int patchesH)
        {
            int numPatches = patchesW * patchesH;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* dst = GetFloatPtr(result);

            var convWeight = _weights["v.patch_conv.weight"];
            float* wPtr = GetFloatPtr(convWeight);
            float* biasPtr = _weights.ContainsKey("v.patch_conv.bias")
                ? GetFloatPtr(_weights["v.patch_conv.bias"]) : null;

            int C = 3;
            int P = _patchSize;

            for (int py = 0; py < patchesH; py++)
            {
                for (int px = 0; px < patchesW; px++)
                {
                    int patchIdx = py * patchesW + px;
                    float* outPatch = dst + patchIdx * _hiddenSize;

                    for (int f = 0; f < _hiddenSize; f++)
                    {
                        float sum = biasPtr != null ? biasPtr[f] : 0f;

                        for (int c = 0; c < C; c++)
                        {
                            for (int ky = 0; ky < P; ky++)
                            {
                                for (int kx = 0; kx < P; kx++)
                                {
                                    int imgY = py * P + ky;
                                    int imgX = px * P + kx;
                                    float pixel = pixelValues[c * imgH * imgW + imgY * imgW + imgX];
                                    int wIdx = f * C * P * P + c * P * P + ky * P + kx;
                                    sum += pixel * wPtr[wIdx];
                                }
                            }
                        }
                        outPatch[f] = sum;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Compute 2D RoPE embeddings for the vision transformer.
        /// Returns (cos, sin) tensors of shape [headDim, 1, numPatches].
        /// </summary>
        private (Tensor cos, Tensor sin) Compute2DRoPE(int patchesW, int patchesH)
        {
            int numPatches = patchesW * patchesH;
            float[] angles = BuildVisionRopeAngles(patchesW, patchesH, _headDim, _visionRopeBase);

            float[] cosVals = new float[angles.Length];
            float[] sinVals = new float[angles.Length];
            for (int i = 0; i < angles.Length; i++)
            {
                cosVals[i] = MathF.Cos(angles[i]);
                sinVals[i] = MathF.Sin(angles[i]);
            }

            var cosTensor = new Tensor(_allocator, DType.Float32, numPatches, 1, _headDim);
            cosTensor.SetElementsAsFloat(cosVals);
            var sinTensor = new Tensor(_allocator, DType.Float32, numPatches, 1, _headDim);
            sinTensor.SetElementsAsFloat(sinVals);

            return (cosTensor, sinTensor);
        }

        /// <summary>
        /// Rotary angles of HF <c>PixtralRotaryEmbedding</c>, laid out PATCH-major
        /// (<c>[patch * headDim + d]</c>) because that is how <see cref="ApplyVisionRoPE"/>
        /// reads them. With <c>freqs[i] = base^(-2i/headDim)</c>, the first headDim/4
        /// dims rotate by row (<c>h * freqs[0::2]</c>), the next headDim/4 by column
        /// (<c>w * freqs[1::2]</c>), and the second half repeats the first (rotate-half
        /// pairs). The table used to be filled frequency-major and read patch-major, which
        /// gave every patch another patch's (and another frequency's) angles.
        /// </summary>
        internal static float[] BuildVisionRopeAngles(int patchesW, int patchesH, int headDim, float ropeBase)
        {
            int quarter = headDim / 4;
            int half = headDim / 2;
            var freqs = new double[half];
            for (int i = 0; i < half; i++)
                freqs[i] = 1.0 / Math.Pow(ropeBase, 2.0 * i / headDim);

            var angles = new float[patchesW * patchesH * headDim];
            for (int h = 0; h < patchesH; h++)
            {
                for (int w = 0; w < patchesW; w++)
                {
                    int row = (h * patchesW + w) * headDim;
                    for (int f = 0; f < quarter; f++)
                    {
                        float byRow = (float)(h * freqs[2 * f]);
                        float byCol = (float)(w * freqs[2 * f + 1]);
                        angles[row + f] = byRow;
                        angles[row + quarter + f] = byCol;
                        angles[row + half + f] = byRow;
                        angles[row + half + quarter + f] = byCol;
                    }
                }
            }
            return angles;
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches,
            Tensor cos, Tensor sin)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");
            using var attnOut = VisionSelfAttention(normed, prefix, numPatches, cos, sin);

            Ops.Add(attnOut, attnOut, hidden);
            hidden.Dispose();

            using var normed2 = RMSNormOp(attnOut, $"{prefix}.ffn_norm.weight");
            using var mlpOut = VisionMLP(normed2, prefix);

            var result = new Tensor(_allocator, DType.Float32, attnOut.Sizes);
            Ops.Add(result, attnOut, mlpOut);

            return result;
        }

        private unsafe Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches,
            Tensor cos, Tensor sin)
        {
            using var q = LinearForward(input, $"{prefix}.attn_q.weight");
            using var k = LinearForward(input, $"{prefix}.attn_k.weight");
            using var v = LinearForward(input, $"{prefix}.attn_v.weight");

            // Reshape to [numPatches, numHeads, headDim]
            using var qR = q.View(numPatches, _numHeads, _headDim);
            using var kR = k.View(numPatches, _numHeads, _headDim);
            using var vR = v.View(numPatches, _numHeads, _headDim);

            // Apply 2D RoPE
            var qRoped = ApplyVisionRoPE(qR, cos, sin, numPatches);
            var kRoped = ApplyVisionRoPE(kR, cos, sin, numPatches);

            float scale = 1f / MathF.Sqrt(_headDim);

            if (_useNativeAttention)
            {
                using var q4 = qRoped.View(1, numPatches, _numHeads, _headDim);
                using var k4 = kRoped.View(1, numPatches, _numHeads, _headDim);
                using var v4 = vR.View(1, numPatches, _numHeads, _headDim);
                using var attn4 = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, scale);
                qRoped.Dispose();
                kRoped.Dispose();
                using var flat = attn4.View(numPatches, _hiddenSize);
                return LinearForward(flat, $"{prefix}.attn_output.weight");
            }

            // Manual attention path
            using var qT0 = qRoped.Transpose(0, 1);
            using var kT0 = kRoped.Transpose(0, 1);
            using var vT0 = vR.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qT0);
            using var kHeads = Ops.NewContiguous(kT0);
            using var vHeads = Ops.NewContiguous(vT0);
            qRoped.Dispose();
            kRoped.Dispose();

            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, numPatches);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, _headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1.0f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flatContig = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return LinearForward(flatContig, $"{prefix}.attn_output.weight");
        }

        /// <summary>
        /// Apply rotary position embeddings (2D RoPE) for vision.
        /// Uses rotate_half style: [-x1, x0] * sin + [x0, x1] * cos
        /// </summary>
        private unsafe Tensor ApplyVisionRoPE(Tensor input, Tensor cos, Tensor sin, int numPatches)
        {
            // input: [numPatches, numHeads, headDim]
            var result = new Tensor(_allocator, DType.Float32, input.Sizes);
            float* inPtr = GetFloatPtr(input);
            float* outPtr = GetFloatPtr(result);
            float* cosPtr = GetFloatPtr(cos);
            float* sinPtr = GetFloatPtr(sin);

            int halfDim = _headDim / 2;

            for (int p = 0; p < numPatches; p++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    float* inHead = inPtr + (long)p * _numHeads * _headDim + h * _headDim;
                    float* outHead = outPtr + (long)p * _numHeads * _headDim + h * _headDim;

                    for (int d = 0; d < halfDim; d++)
                    {
                        float x0 = inHead[d];
                        float x1 = inHead[d + halfDim];
                        float c = cosPtr[p * _headDim + d];
                        float s = sinPtr[p * _headDim + d];

                        // rotate_half: cos*x - sin*rotate_half(x)
                        outHead[d] = x0 * c - x1 * s;
                        outHead[d + halfDim] = x1 * c + x0 * s;
                    }
                }
            }

            return result;
        }

        private Tensor VisionMLP(Tensor input, string prefix)
        {
            using var gate = LinearForward(input, $"{prefix}.ffn_gate.weight");
            using var up = LinearForward(input, $"{prefix}.ffn_up.weight");
            if (FfnActivation == VisionFfnActivation.Silu)
                Ops.SiLUMul(gate, gate, up);
            else
                Ops.GELUMul(gate, gate, up);
            return LinearForward(gate, $"{prefix}.ffn_down.weight");
        }

        /// <summary>
        /// Multi-modal projector: vision → text space.
        /// Steps: RMSNorm → PatchMerger → Linear1 → GELU → Linear2
        /// </summary>
        private unsafe Tensor MultiModalProject(Tensor visionOutput, int patchesW, int patchesH)
        {
            int numPatches = patchesW * patchesH;

            // RMSNorm
            using var normed = RMSNormOp(visionOutput, "mm.norm.weight");

            // Patch merger: merge spatialMergeSize x spatialMergeSize patches
            int mergedW = patchesW / _spatialMergeSize;
            int mergedH = patchesH / _spatialMergeSize;
            int mergedPatches = mergedW * mergedH;
            int mergeInputDim = _hiddenSize * _spatialMergeSize * _spatialMergeSize;

            var mergeInput = new Tensor(_allocator, DType.Float32, mergedPatches, mergeInputDim);
            float* srcPtr = GetFloatPtr(normed);
            float* dstPtr = GetFloatPtr(mergeInput);
            MergePatches(srcPtr, dstPtr, patchesW, patchesH, _hiddenSize, _spatialMergeSize);

            // Patch merger linear
            using var merged = LinearForward(mergeInput, "mm.patch_merger.merging_layer.weight");
            mergeInput.Dispose();

            // Linear1 → GELU → Linear2
            using var proj1 = LinearForward(merged, "mm.linear_1.weight");
            Ops.GELU(proj1, proj1);
            var proj2 = LinearForward(proj1, "mm.linear_2.weight");

            Console.WriteLine($"Vision projector: {numPatches} patches → {mergedPatches} merged tokens " +
                $"({(int)proj2.Sizes[0]}x{(int)proj2.Sizes[1]})");

            return proj2;
        }

        /// <summary>
        /// Gathers each <paramref name="merge"/> x <paramref name="merge"/> window of the
        /// patch grid into one row in the order the merging layer was trained on: HF
        /// <c>Mistral3PatchMerger</c> runs <c>torch.nn.functional.unfold</c> over the
        /// [d, h, w] grid, whose output is CHANNEL-major - row feature
        /// <c>c * merge * merge + ky * merge + kx</c> holds channel c of window cell
        /// (ky, kx). llama.cpp and Ollama reproduce it with im2col. Rows and columns past
        /// the last full window are dropped, as unfold does.
        /// </summary>
        internal static unsafe void MergePatches(float* src, float* dst, int patchesW, int patchesH, int hidden, int merge)
        {
            int mergedW = patchesW / merge;
            int mergedH = patchesH / merge;
            int window = merge * merge;
            for (int my = 0; my < mergedH; my++)
            {
                for (int mx = 0; mx < mergedW; mx++)
                {
                    float* outRow = dst + (long)(my * mergedW + mx) * hidden * window;
                    for (int ky = 0; ky < merge; ky++)
                    {
                        for (int kx = 0; kx < merge; kx++)
                        {
                            long srcIdx = (long)(my * merge + ky) * patchesW + (mx * merge + kx);
                            float* srcRow = src + srcIdx * hidden;
                            int cell = ky * merge + kx;
                            for (int c = 0; c < hidden; c++)
                                outRow[(long)c * window + cell] = srcRow[c];
                        }
                    }
                }
            }
        }

        private Tensor RMSNormOp(Tensor input, string weightName)
        {
            if (!_weights.ContainsKey(weightName))
                return Ops.NewContiguous(input);
            return Ops.RMSNorm(null, input, _weights[weightName], null, _eps);
        }

        private Tensor LinearForward(Tensor input, string weightName)
        {
            if (!_weights.ContainsKey(weightName))
                return null;

            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;
            Ops.Addmm(result, 0, result, 1.0f, src, GetOrCreateTransposedWeight(weightName));

            contiguousInput?.Dispose();
            return result;
        }

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            using var weightViewT = _weights[weightName].Transpose();
            transposed = Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        public void Dispose()
        {
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
            foreach (var qw in _quantWeights.Values)
                qw.Dispose();
            _quantWeights.Clear();
        }
    }
}
