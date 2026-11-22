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
using System.Numerics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public class Gemma4AudioEncoder : IDisposable
    {
        // Optional ModelBase reference for cooperative GpuComputeLock
        // yielding between conformer blocks. See Gemma4VisionEncoder.
        private ModelBase _hostModel;
        public void SetHostModel(ModelBase model) => _hostModel = model;
        // Independent validation can inspect stage outputs without changing
        // the public model API. The absent observer adds no device readbacks.
        internal Action<string, float[], int, int> ValidationTrace { get; set; }

        private void TraceTensor(string name, Tensor tensor)
        {
            var observer = ValidationTrace;
            if (observer == null) return;
            int rows = checked((int)tensor.Sizes[0]), columns = checked((int)tensor.Sizes[1]);
            observer(name, tensor.GetElementsAsFloat(checked(rows * columns)), rows, columns);
        }

        private void TracePlanar(string name, float[] values, int frequency, int time, int channels)
        {
            var observer = ValidationTrace;
            if (observer == null) return;
            int spatial = checked(frequency * time);
            var flattened = new float[checked(spatial * channels)];
            for (int position = 0; position < spatial; ++position)
                for (int channel = 0; channel < channels; ++channel)
                    flattened[position * channels + channel] = values[channel * spatial + position];
            observer(name, flattened, time, checked(frequency * channels));
        }

        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private sealed class CompressedMatrix
        {
            private GCHandle _pinned;
            public readonly int Input, Output, Bytes;
            public IntPtr Pointer => _pinned.AddrOfPinnedObject();
            public CompressedMatrix(byte[] bytes, int input, int output)
            {
                Input = input; Output = output; Bytes = bytes.Length;
                _pinned = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            }
            public void Free() { if (_pinned.IsAllocated) _pinned.Free(); }
        }
        private readonly Dictionary<string, CompressedMatrix> _compressedMatrices = new();
        private readonly Dictionary<int, GgmlGemma4AudioBlock> _conformerBlocks = new();
        private readonly bool _useConformerBlockGraph;
        private readonly bool _useSscpGraph;
        private readonly bool _requireGgmlFloat32Precision;
        private readonly bool _usePreciseCudaSscp;
        private readonly bool _usePreciseCudaLinears;
        private GgmlGemma4AudioConvolution[] _sscpConvolutions;
        private GgmlGemma4AudioLinear _sscpProjection;
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
        private readonly Dictionary<string, float[]> _positionEmbeddingCache = new();
        private readonly Dictionary<string, Tensor> _devicePositionEmbeddingCache = new();

        private bool _useOllamaNames;
        private Tensor _onesForNorm;
        private readonly float[] _causalMask;

        private readonly string _projectorType;
        private readonly bool _isEncoderFree;

        public int ProjectionDim => _projectionDim;

        /// <summary>True when this mmproj uses the Gemma 4 "unified" multimodal
        /// embedder (projector_type "gemma4ua"). These models have no conformer
        /// audio encoder: the raw 16 kHz waveform is chunked into 640-sample
        /// frames, RMS-normalized, and projected directly into text-embedding
        /// space. Used by e.g. gemma-4-12b. See <see cref="EncodeRawWaveform"/>.</summary>
        public bool IsEncoderFree => _isEncoderFree;

        public Gemma4AudioEncoder(string mmProjPath, IAllocator allocator)
            : this(mmProjPath, allocator, requireValidatedFloat32Precision: false) { }

        /// <summary>Loads the audio tower. The precision requirement selects
        /// model-qualified learned projection and SSCP F32 paths without changing process-wide
        /// backend math modes. EmbeddingGemma 2 uses this requirement; existing
        /// Gemma chat callers retain their constructor behavior.</summary>
        public Gemma4AudioEncoder(string mmProjPath, IAllocator allocator, bool requireValidatedFloat32Precision)
        {
            ArgumentNullException.ThrowIfNull(allocator);
            _allocator = allocator;
            static bool Enabled(string name, bool defaultEnabled = false)
            {
                string value = Environment.GetEnvironmentVariable(name);
                return value == "1" || defaultEnabled && value != "0";
            }
            bool metal = allocator is GgmlAllocator ga && ga.Context.BackendType == GgmlBackendType.Metal;
            bool ggmlCuda = allocator is GgmlAllocator gc && gc.Context.BackendType == GgmlBackendType.Cuda;
            bool ggmlCpu = allocator is GgmlAllocator cp && cp.Context.BackendType == GgmlBackendType.Cpu;
            bool qualifiedGgml = metal || ggmlCuda || ggmlCpu;
            _requireGgmlFloat32Precision = metal && Enabled("TS_GEMMA4_AUDIO_METAL_MPS_MM", requireValidatedFloat32Precision) ||
                ggmlCuda && Enabled("TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC", requireValidatedFloat32Precision) ||
                ggmlCpu && requireValidatedFloat32Precision;
            _useConformerBlockGraph = allocator is GgmlAllocator &&
                Enabled("TS_GEMMA4_AUDIO_BLOCK_GRAPH", qualifiedGgml && requireValidatedFloat32Precision);
            _useSscpGraph = allocator is GgmlAllocator &&
                Enabled("TS_GEMMA4_AUDIO_SSCP_GRAPH", qualifiedGgml && requireValidatedFloat32Precision);
            _usePreciseCudaSscp = allocator is TensorSharp.Cuda.CudaAllocator &&
                Enabled("TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP", requireValidatedFloat32Precision);
            _usePreciseCudaLinears = allocator is TensorSharp.Cuda.CudaAllocator &&
                Enabled("TS_GEMMA4_AUDIO_CUDA_PRECISE_LINEAR", requireValidatedFloat32Precision);
            using var gguf = new GgufFile(mmProjPath);

            _projectorType = gguf.GetString("clip.audio.projector_type", "gemma4a") ?? "gemma4a";
            _isEncoderFree = string.Equals(_projectorType, "gemma4ua", StringComparison.Ordinal);

            _hiddenSize = (int)gguf.GetUint32("clip.audio.embedding_length",
                (uint)gguf.GetUint32("gemma4.audio.embedding_length", 1024));
            _numHeads = (int)gguf.GetUint32("clip.audio.attention.head_count",
                (uint)gguf.GetUint32("gemma4.audio.attention.head_count", 8));
            // Encoder-free "gemma4ua" mmproj files (e.g. gemma-4-12b) have no
            // conformer, so head_count / block_count / feed_forward_length are
            // written as 0. These models go through EncodeRawWaveform and never
            // touch the conformer attention path, so _headDim is unused — guard
            // the division to avoid a DivideByZeroException at load time.
            // Conformer mmproj files always report head_count > 0.
            _headDim = _numHeads > 0 ? _hiddenSize / _numHeads : 0;
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

            try { LoadWeights(gguf); }
            catch { Dispose(); throw; }

            _useOllamaNames = _weights.ContainsKey("a.blk.0.ln1.weight");
            _causalMask = BuildCausalValidMask();
            Console.WriteLine($"  GGUF naming: {(_useOllamaNames ? "Ollama" : "mmproj/Unsloth")}");
            Console.WriteLine($"  projector_type: {_projectorType}" +
                (_isEncoderFree ? " (encoder-free / raw-waveform)" : " (conformer)"));
        }

        private bool ShouldRetainCompressedMatrix(string name)
        {
            if (_allocator is TensorSharp.MLX.MlxAllocator) return true;
            if (_allocator is not GgmlAllocator ga || ga.Context.BackendType != GgmlBackendType.Metal) return false;
            // MPS consumes the already retained exact F32 weights. Avoid pinning
            // an additional copy of Q8 matrices that only these graphs consume.
            // Relative-position and final projector weights retain their Q8 path.
            if (_requireGgmlFloat32Precision && _useSscpGraph &&
                (name == "a.input_projection.weight" || name == "a.pre_encode.out.weight")) return false;
            if (_requireGgmlFloat32Precision && _useConformerBlockGraph && name.StartsWith("a.blk.", StringComparison.Ordinal))
            {
                string[] suffixes = [".ffn_up.weight", ".ffn_down.weight", ".attn_q.weight", ".attn_k.weight",
                    ".attn_v.weight", ".attn_out.weight", ".conv_pw1.weight", ".conv_pw2.weight",
                    ".ffn_up_1.weight", ".ffn_down_1.weight"];
                foreach (string suffix in suffixes) if (name.EndsWith(suffix, StringComparison.Ordinal)) return false;
            }
            return true;
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
                else if (_allocator is GgmlAllocator)
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);
                else
                    ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                // Keep the exact GGUF blocks alongside the scalar-oracle
                // weights. Weight-only kernels consume F32 activations and
                // produce F32 outputs; no conformer hidden state is narrowed.
                // CPU and unvalidated CUDA paths retain the dense F32 math.
                if (info.Type == GgmlTensorType.Q8_0 && info.Shape.Length == 2 &&
                    ShouldRetainCompressedMatrix(info.Name))
                    _compressedMatrices.Add(info.Name, new CompressedMatrix(raw,
                        checked((int)info.Shape[0]), checked((int)info.Shape[1])));
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
            ModelBase.LogCudaVram(_allocator, "after audio encoder weight load");
        }

        public unsafe Tensor Encode(float[] melData, int numFrames)
        {
            if (_allocator is TensorSharp.MLX.MlxAllocator && !TensorSharp.MLX.MlxWorker.Shared.IsOnWorkerThread)
                return TensorSharp.MLX.MlxWorker.Shared.Invoke(() => Encode(melData, numFrames));
            Console.Write("Audio encoder SSCP...");

            Tensor hiddenTensor;
            int hidDim, seqLen;
            if (_useSscpGraph)
            {
                hiddenTensor = EncodeSscpGraph(melData, numFrames);
                seqLen = checked((int)hiddenTensor.Sizes[0]);
                hidDim = checked((int)hiddenTensor.Sizes[1]);
            }
            else
            {

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
            TracePlanar("sscp0", conv0Out, f0Out, t0Out, c0Out);

            // Conv1
            var conv1Out = Conv2DBlock(conv0Out, f0Out, t0Out, c0Out, "a.conv1d.1");
            int f1Out = (f0Out + 2 - 3) / 2 + 1;
            int t1Out = (t0Out + 2 - 3) / 2 + 1;
            int c1Out = GetConvOutChannels("a.conv1d.1");
            TracePlanar("sscp1", conv1Out, f1Out, t1Out, c1Out);

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

            if (_weights.TryGetValue(sscpWeightName, out var sscpWeight))
            {
                int outDim = (int)sscpWeight.Sizes[0];
                hidDim = outDim;
                hiddenTensor = new Tensor(_allocator, DType.Float32, t1Out, hidDim);
                Project(hiddenTensor, projTensor, sscpWeightName);
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

            seqLen = t1Out;
            }
            Console.Write($" proj=[{seqLen},{hidDim}]");
            TraceTensor("input_projection", hiddenTensor);

            // Build causal-valid mask
            // Conformer blocks
            if (_useConformerBlockGraph && _hostModel == null && _numLayers > 0)
            {
                var stackResult = new Tensor(_allocator, DType.Float32, seqLen, hidDim);
                try
                {
                    var blocks = new GgmlGemma4AudioBlock[_numLayers];
                    for (int index = 0; index < blocks.Length; ++index) blocks[index] = GetConformerBlockDescriptor(index);
                    if (!GgmlGemma4AudioOps.TryConformerStack(stackResult, hiddenTensor, blocks, seqLen, _numHeads, _headDim))
                        throw new InvalidOperationException("The selected backend rejected the Gemma4 F32 conformer stack descriptors.");
                    hiddenTensor.Dispose(); hiddenTensor = stackResult;
                }
                catch { stackResult.Dispose(); hiddenTensor.Dispose(); throw; }
            }
            else for (int i = 0; i < _numLayers; i++)
            {
                Console.Write($"\r  Audio conformer block {i + 1}/{_numLayers}...                    ");
                hiddenTensor = ConformerBlock(hiddenTensor, i, seqLen, hidDim, _causalMask);
                // See Gemma4VisionEncoder for rationale: yield the GPU
                // compute lock between conformer blocks so the engine
                // worker can run an inference step. Keeps concurrent
                // chat/decode requests responsive during long audio encodes.
                _hostModel?.YieldGpuComputeLock();
            }
            Console.Write("\r  Audio conformer done.                                         \n");

            // Output projection: Ollama uses a.output_proj.weight, neither file has it currently.
            if (_weights.TryGetValue("a.output_proj.weight", out var outProjWeight))
            {
                int outDim = (int)outProjWeight.Sizes[0];
                var outProj = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                Project(outProj, hiddenTensor, "a.output_proj.weight");
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
                Project(fcOut, hiddenTensor, fcWeightName);

                string fcBiasName = fcWeightName.Replace(".weight", ".bias");
                if (_weights.TryGetValue(fcBiasName, out var fcBias))
                    AddBias(fcOut, fcBias, seqLen, fcOutDim);
                hiddenTensor.Dispose();
                hiddenTensor = fcOut;
                hidDim = fcOutDim;
            }

            // Unweighted RMSNorm
            TraceTensor("encoder_output_projection", hiddenTensor);
            ApplyUnweightedRMSNorm(hiddenTensor, seqLen, hidDim);
            TraceTensor("embedding_norm", hiddenTensor);

            // Embedding projection to text hidden size
            if (_weights.ContainsKey("mm.a.input_projection.weight"))
            {
                var old = hiddenTensor;
                hiddenTensor = AudioClippableLinearForward(hiddenTensor, "mm.a.input_projection", seqLen);
                old.Dispose();
                hidDim = (int)hiddenTensor.Sizes[1];
            }

            Console.WriteLine($"Audio encoder: [{seqLen}, {hidDim}] done");
            TraceTensor("soft_rows", hiddenTensor);
            return hiddenTensor;
        }

        /// <summary>
        /// Encoder-free audio embedding path for Gemma 4 "unified" models
        /// (projector_type "gemma4ua", e.g. gemma-4-12b). Mirrors llama.cpp's
        /// clip_graph_gemma4ua: the raw 16 kHz mono waveform is chunked into
        /// fixed 640-sample frames (40 ms/token, last frame zero-padded), each
        /// frame is RMS-normalized over its 640 samples (unweighted, eps=1e-6),
        /// and projected straight into text-embedding space via
        /// mm.a.input_projection.weight [640 -> textHidden]. There is no
        /// conformer, mel spectrogram, bias or output clamping.
        /// </summary>
        public unsafe Tensor EncodeRawWaveform(float[] samples)
        {
            const string projName = "mm.a.input_projection.weight";
            if (!_weights.TryGetValue(projName, out var projWeight))
                throw new InvalidOperationException(
                    $"Gemma4 unified audio encoder requires '{projName}' in the mmproj file.");

            int outDim = (int)projWeight.Sizes[0];    // text hidden size (e.g. 3840)
            int frameSize = (int)projWeight.Sizes[1]; // raw samples per token (640)

            int nTokens = (samples.Length + frameSize - 1) / frameSize;
            if (nTokens <= 0)
                throw new InvalidOperationException("Audio waveform produced zero frames.");

            Console.Write($"Audio encoder (gemma4ua): frames={nTokens} x {frameSize} -> [{nTokens},{outDim}]...");

            // Pack samples into [nTokens, frameSize] row-major, zero-padding the
            // trailing frame. Row t holds samples[t*frameSize .. t*frameSize+frameSize).
            float[] framed = new float[(long)nTokens * frameSize];
            int copyCount = Math.Min(samples.Length, framed.Length);
            Array.Copy(samples, framed, copyCount);

            var input = new Tensor(_allocator, DType.Float32, nTokens, frameSize);
            input.SetElementsAsFloat(framed);

            // embedding_pre_projection_norm: unweighted RMSNorm over frameSize.
            ApplyUnweightedRMSNorm(input, nTokens, frameSize);

            var output = new Tensor(_allocator, DType.Float32, nTokens, outDim);
            Project(output, input, projName);
            input.Dispose();

            Console.WriteLine(" done");
            return output;
        }

        private int GetConvOutChannels(string prefix)
        {
            var w = _weights[$"{prefix}.weight"];
            return (int)w.Sizes[0]; // TensorSharp reversed: GGUF [kW, kH, C_in, C_out] -> TS [C_out, C_in, kH, kW]
        }

        // SSCP's two small strided convolutions can use the same backend GEMM
        // as the conformer projections. Packing is linear in the input size;
        // convolution, channel normalization, and ReLU execute on the allocator.
        internal static Tensor SscpConvolution(IAllocator allocator, float[] planarInput,
            int inW, int inH, int inChannels, int kernelW, int kernelH,
            Tensor projection, Tensor normWeight, Tensor normBias, float epsilon, bool preferFloat32 = false)
        {
            ArgumentNullException.ThrowIfNull(allocator);
            ArgumentNullException.ThrowIfNull(planarInput);
            ArgumentNullException.ThrowIfNull(projection);
            if (inW <= 0 || inH <= 0 || inChannels <= 0 || kernelW <= 0 || kernelH <= 0 ||
                planarInput.Length != checked(inW * inH * inChannels) ||
                projection.DimensionCount != 2 || projection.Sizes[0] != checked(inChannels * kernelW * kernelH) ||
                projection.Sizes[1] <= 0 || projection.ElementType != DType.Float32 ||
                !float.IsFinite(epsilon) || epsilon <= 0)
                throw new ArgumentException("SSCP convolution dimensions or normalization epsilon are invalid.");
            int outW = (inW + 2 - kernelW) / 2 + 1, outH = (inH + 2 - kernelH) / 2 + 1;
            if (outW <= 0 || outH <= 0) throw new ArgumentException("SSCP convolution has no output positions.");
            int kernelColumns = checked(inChannels * kernelH * kernelW), outChannels = checked((int)projection.Sizes[1]);
            int positions = checked(outW * outH), inSpatial = checked(inW * inH);
            var columns = new float[checked(positions * kernelColumns)];
            for (int oh = 0; oh < outH; ++oh)
                for (int ow = 0; ow < outW; ++ow)
                    for (int channel = 0; channel < inChannels; ++channel)
                        for (int ky = 0; ky < kernelH; ++ky)
                            for (int kx = 0; kx < kernelW; ++kx)
                            {
                                int y = oh * 2 - 1 + ky, x = ow * 2 - 1 + kx;
                                if ((uint)y < (uint)inH && (uint)x < (uint)inW)
                                    columns[(oh * outW + ow) * kernelColumns + (channel * kernelH + ky) * kernelW + kx]
                                        = planarInput[channel * inSpatial + y * inW + x];
                            }
            using var input = new Tensor(allocator, DType.Float32, positions, kernelColumns);
            input.SetElementsAsFloat(columns);
            var output = new Tensor(allocator, DType.Float32, positions, outChannels);
            try
            {
                if (allocator is TensorSharp.MLX.MlxAllocator)
                {
                    if (!TensorSharp.MLX.MlxExactOps.TryMatmul(output, input, projection))
                        throw new NotSupportedException("MLX requires the exact F32 SSCP projection kernel.");
                }
                else if (allocator is TensorSharp.Cuda.CudaAllocator &&
                    (preferFloat32 || Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP") == "1"))
                {
                    if (!TensorSharp.Cuda.CudaGemma4AudioOps.TryProjectionFloat32(output, input, projection))
                        throw new NotSupportedException("CUDA precise SSCP requires contiguous F32 operands on one allocator.");
                }
                else Ops.Addmm(output, 0, output, 1, input, projection);
                if (normWeight != null || normBias != null) Ops.LayerNorm(output, output, normWeight, normBias, epsilon);
                Ops.Relu(output, output);
                return output;
            }
            catch { output.Dispose(); throw; }
        }

        private Tensor EncodeSscpGraph(float[] melData, int numFrames)
        {
            if (_sscpConvolutions == null)
            {
                _sscpConvolutions = new GgmlGemma4AudioConvolution[2];
                for (int layer = 0; layer < 2; ++layer)
                {
                    string prefix = $"a.conv1d.{layer}";
                    var weight = _weights[prefix + ".weight"];
                    _weights.TryGetValue(prefix + ".norm.weight", out var norm);
                    _weights.TryGetValue(prefix + ".norm.bias", out var bias);
                    _sscpConvolutions[layer] = new(weight, checked((int)weight.Sizes[1]), checked((int)weight.Sizes[0]),
                        checked((int)weight.Sizes[3]), checked((int)weight.Sizes[2]), norm, bias);
                }
                string name = _weights.ContainsKey("a.input_projection.weight") ? "a.input_projection.weight" : "a.pre_encode.out.weight";
                _weights.TryGetValue(name.Replace(".weight", ".bias"), out var projectionBias);
                _sscpProjection = new(_weights[name], projectionBias);
                if (_compressedMatrices.TryGetValue(name, out var compressed))
                    _sscpProjection = _sscpProjection with
                    {
                        CompressedWeight = compressed.Pointer, CompressedWeightType = (int)GgmlTensorType.Q8_0,
                        CompressedWeightBytes = compressed.Bytes,
                    };
            }
            int time = numFrames;
            foreach (var convolution in _sscpConvolutions) time = (time + 2 - convolution.KernelTime) / 2 + 1;
            using var input = new Tensor(_allocator, DType.Float32, numFrames, _melBins);
            input.SetElementsAsFloat(melData);
            var result = new Tensor(_allocator, DType.Float32, time, _sscpProjection.Weight.Sizes[0]);
            try
            {
                if (!GgmlGemma4AudioOps.TrySscp(result, input, _sscpConvolutions, _sscpProjection, _melBins, numFrames, 1, _eps, _requireGgmlFloat32Precision))
                    throw new InvalidOperationException("The selected backend rejected the Gemma4 SSCP descriptors.");
                return result;
            }
            catch { result.Dispose(); throw; }
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

            string projectionKey = prefix + ".sscp_projection";
            if (!_transposedWeights.TryGetValue(projectionKey, out var projection))
            {
                using var flattened = kernelTensor.View(cOut, checked(cIn * kH * kW));
                using var transposed = flattened.Transpose();
                projection = Ops.NewContiguous(transposed);
                _transposedWeights.Add(projectionKey, projection);
            }
            _weights.TryGetValue(prefix + ".norm.weight", out var channelNorm);
            _weights.TryGetValue(prefix + ".norm.bias", out var channelBias);
            using var result = SscpConvolution(_allocator, input, inW, inH, inC, kW, kH,
                projection, channelNorm, channelBias, _eps, _usePreciseCudaSscp);
            var rowMajor = result.GetElementsAsFloat(checked(outW * outH * cOut));
            int spatial = checked(outW * outH);
            var planar = new float[rowMajor.Length];
            for (int position = 0; position < spatial; ++position)
                for (int channel = 0; channel < cOut; ++channel)
                    planar[position + channel * spatial] = rowMajor[position * cOut + channel];
            return planar;
        }

        private Tensor ConformerBlock(Tensor x, int blockIdx, int seqLen, int hidDim, float[] causalMask)
        {
            string prefix = $"a.blk.{blockIdx}";
            if (_useConformerBlockGraph)
            {
                var result = new Tensor(_allocator, DType.Float32, seqLen, hidDim);
                try
                {
                    var block = GetConformerBlockDescriptor(blockIdx);
                    if (!GgmlGemma4AudioOps.TryConformerBlock(result, x, block.Relative, block.Norms, block.Linears,
                        seqLen, _numHeads, _headDim, block.MaximumPast, block.MaximumFuture, block.Kernel,
                        block.Epsilon, block.Cap, block.GradientClip, block.ResidualWeight, block.RequireFloat32Precision))
                        throw new InvalidOperationException("The selected backend rejected the Gemma4 F32 conformer block descriptors.");
                    x.Dispose();
                    return result;
                }
                catch { result.Dispose(); x.Dispose(); throw; }
            }

            x = ForwardFFW(x, prefix, "ffn_norm", "ffn_up", "ffn_down", "ffn_post_norm", seqLen, hidDim);
            TraceTensor($"layer{blockIdx}_ffn0", x);
            x = ForwardAttention(x, prefix, seqLen, hidDim, causalMask);
            TraceTensor($"layer{blockIdx}_attention", x);
            x = ForwardLightConv(x, prefix, seqLen, hidDim);
            TraceTensor($"layer{blockIdx}_convolution", x);
            x = ForwardFFW(x, prefix, "ffn_norm_1", "ffn_up_1", "ffn_down_1", "ffn_post_norm_1", seqLen, hidDim);

            Clamp(x, -_gradClip, _gradClip);
            string blockNormName = ResolveName(prefix, "block_norm") + ".weight";
            var old = x;
            x = ApplyRMSNorm(x, blockNormName, seqLen, hidDim);
            old.Dispose();
            TraceTensor($"layer{blockIdx}", x);

            return x;
        }

        private GgmlGemma4AudioBlock GetConformerBlockDescriptor(int blockIdx)
        {
            if (_conformerBlocks.TryGetValue(blockIdx, out var block)) return block;
            string prefix = $"a.blk.{blockIdx}";
            Tensor Weight(string name) => _weights.GetValueOrDefault(name + ".weight");
            Tensor[] norms = [Weight(prefix + ".ffn_norm"), Weight(prefix + ".ffn_post_norm"),
                Weight(ResolveName(prefix, "attn_pre_norm")), Weight(ResolveName(prefix, "attn_post_norm")),
                Weight(prefix + ".conv_norm"), Weight(prefix + ".norm_conv"),
                Weight(prefix + ".ffn_norm_1"), Weight(prefix + ".ffn_post_norm_1"),
                Weight(ResolveName(prefix, "block_norm")), Weight(prefix + ".per_dim_scale"), Weight(prefix + ".conv_dw")];
            string[] names = ["ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_out", "conv_pw1", "conv_pw2", "ffn_up_1", "ffn_down_1"];
            var linears = new GgmlGemma4AudioLinear[names.Length];
            for (int index = 0; index < names.Length; ++index)
            {
                string name = prefix + "." + names[index];
                bool clamp = _clampParams.TryGetValue(name, out var cp) && cp.HasClamp;
                linears[index] = new(Weight(name), _weights.GetValueOrDefault(name + ".bias"), clamp,
                    cp.InMin, cp.InMax, cp.OutMin, cp.OutMax);
                if (_compressedMatrices.TryGetValue(name + ".weight", out var compressed))
                    linears[index] = linears[index] with { CompressedWeight = compressed.Pointer,
                        CompressedWeightType = (int)GgmlTensorType.Q8_0, CompressedWeightBytes = compressed.Bytes };
            }
            int pastInclusive = _maxPast - 1;
            var relative = GetDevicePositionEmbeddings(prefix, pastInclusive + _maxFuture + 1);
            int kernel = norms[10] == null ? 1 : checked((int)norms[10].Sizes[1]);
            block = new(relative, norms, linears, pastInclusive, _maxFuture, kernel, _eps, _logitCap, _gradClip, _residualWeight, _requireGgmlFloat32Precision);
            _conformerBlocks.Add(blockIdx, block);
            return block;
        }

        private Tensor ForwardFFW(Tensor x, string prefix, string normName,
            string upName, string downName, string postNormName, int seqLen, int hidDim)
        {
            var residual = x;

            var clamped = ClampedCopy(x, seqLen, hidDim, -_gradClip, _gradClip);

            var normed = ApplyRMSNorm(clamped, $"{prefix}.{normName}.weight", seqLen, hidDim);
            clamped.Dispose();

            var upOut = AudioClippableLinearForward(normed, $"{prefix}.{upName}", seqLen);
            normed.Dispose();

            // SiLU
            ApplySiLU(upOut);

            var downOut = AudioClippableLinearForward(upOut, $"{prefix}.{downName}", seqLen);
            upOut.Dispose();

            Clamp(downOut, -_gradClip, _gradClip);

            var postNormed = ApplyRMSNorm(downOut, $"{prefix}.{postNormName}.weight", seqLen, hidDim);
            downOut.Dispose();

            // result = residual + postNormed * residualWeight
            if (_allocator is TensorSharp.MLX.MlxAllocator or GgmlAllocator)
            {
                // MLX/GGML's generic addmulv registrations execute on CPU. The
                // exact binary factor 0.5 permits the same residual on-device.
                Ops.Mul(postNormed, postNormed, _residualWeight);
                Ops.Add(postNormed, postNormed, residual);
            }
            else Ops.AddMulV(postNormed, residual, postNormed, _residualWeight);
            residual.Dispose();

            return postNormed;
        }

        private unsafe Tensor ForwardAttention(Tensor x, string prefix, int seqLen, int hidDim, float[] causalMask)
        {
            var residual = x;

            var clamped = ClampedCopy(x, seqLen, hidDim, -_gradClip, _gradClip);

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
            TraceTensor(prefix + "_attention_qkv", q);

            int maxSpan = _maxPast + _maxFuture + 1;
            Tensor result;
            if (Gemma4AudioTensorOps.UsesDevicePath(_allocator))
            {
                result = new Tensor(_allocator, DType.Float32, seqLen, hidDim);
                try
                {
                    Tensor relative = GetDevicePositionEmbeddings(prefix, _maxPast + _maxFuture);
                    if (!Gemma4AudioTensorOps.TryAttention(result, q, k, v, relative, seqLen, _numHeads, _headDim,
                        _maxPast - 1, _maxFuture, _logitCap))
                        throw new InvalidOperationException("The selected backend does not provide the required Gemma4 F32 relative audio attention.");
                }
                catch { result.Dispose(); throw; }
                finally { q.Dispose(); k.Dispose(); v.Dispose(); }
            }
            else
            {
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

            CpuWorkers.Shared.For(checked(numChunks * _numHeads), job =>
            {
                ChunkedAttention(qArr, kPadded, vPadded, posEmb, causalMask, attnOutput,
                    job / _numHeads, job % _numHeads, seqLen, hidDim, maxSpan, padLeft);
            });

            // Trim to original seqLen
            result = new Tensor(_allocator, DType.Float32, seqLen, hidDim);
            float* rPtr = GetFloatPtr(result);
            fixed (float* aPtr = attnOutput)
                Buffer.MemoryCopy(aPtr, rPtr, seqLen * hidDim * 4, seqLen * hidDim * 4);
            }

            TraceTensor(prefix + "_attention_core", result);
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
            int chunkIdx, int h, int seqLen, int hidDim, int maxSpan, int padLeft)
        {
            int cs = _chunkSize;
            int ctx = _contextSize;

            {
                float[] logitsBuffer = new float[ctx];
                for (int qi = 0; qi < cs; qi++)
                {
                    int globalQIdx = chunkIdx * cs + qi;
                    if (globalQIdx >= seqLen)
                    {
                        continue;
                    }

                    Span<float> logits = logitsBuffer;
                    int qOffset = globalQIdx * hidDim + h * _headDim;

                    for (int ci = 0; ci < ctx; ci++)
                    {
                        int actualTime = chunkIdx * cs + ci - padLeft;
                        bool causalOK = causalMask[qi * ctx + ci] > 0;
                        bool validOK = actualTime >= 0 && actualTime < seqLen;
                        if (!causalOK || !validOK)
                        {
                            logits[ci] = -1e9f;
                            continue;
                        }

                        float dotCC = 0;
                        int kGlobalIdx = chunkIdx * cs + ci;
                        int kOffset = kGlobalIdx * hidDim + h * _headDim;

                        dotCC = Dot(qArr, qOffset, kPadded, kOffset, _headDim);

                        float dotCP = 0;
                        int posIdx = RelativeShiftIndex(qi, ci, maxSpan);
                        if (posIdx >= 0 && posIdx < maxSpan)
                        {
                            int posOffset = (posIdx * _numHeads + h) * _headDim;
                            dotCP = Dot(qArr, qOffset, posEmb, posOffset, _headDim);
                        }

                        logits[ci] = dotCC + dotCP;
                        logits[ci] = MathF.Tanh(logits[ci] / _logitCap) * _logitCap;
                    }

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

                    int outOffset = globalQIdx * hidDim + h * _headDim;
                    int d = 0;
                    for (; d + Vector<float>.Count <= _headDim; d += Vector<float>.Count)
                    {
                        var sum = Vector<float>.Zero;
                        for (int ci = 0; ci < ctx; ci++)
                        {
                            int vOffset = (chunkIdx * cs + ci) * hidDim + h * _headDim + d;
                            sum += new Vector<float>(logits[ci]) * new Vector<float>(vPadded, vOffset);
                        }
                        sum.CopyTo(attnOutput, outOffset + d);
                    }
                    for (; d < _headDim; d++)
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

        private static float Dot(float[] left, int leftOffset, float[] right, int rightOffset, int count)
        {
            var sum = Vector<float>.Zero;
            int index = 0;
            for (; index + Vector<float>.Count <= count; index += Vector<float>.Count)
                sum += new Vector<float>(left, leftOffset + index) * new Vector<float>(right, rightOffset + index);
            float result = Vector.Sum(sum);
            for (; index < count; ++index) result += left[leftOffset + index] * right[rightOffset + index];
            return result;
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
            if (_positionEmbeddingCache.TryGetValue(prefix, out var cached))
                return cached;

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
            {
                _positionEmbeddingCache[prefix] = sinEmb;
                return sinEmb;
            }

            int relOutDim = (int)relWeight.Sizes[0];
            int inDim = (int)relWeight.Sizes[1];

            using var sinTensor = new Tensor(_allocator, DType.Float32, maxSpan, _hiddenSize);
            sinTensor.SetElementsAsFloat(sinEmb);

            using var sinSlice = sinTensor.Narrow(1, 0, inDim);
            using var sinContig = Ops.NewContiguous(sinSlice);
            var result = new Tensor(_allocator, DType.Float32, maxSpan, relOutDim);
            Project(result, sinContig, relKey);

            float[] projected = new float[maxSpan * relOutDim];
            result.CopyToArray(projected);
            result.Dispose();

            _positionEmbeddingCache[prefix] = projected;
            return projected;
        }

        private Tensor GetDevicePositionEmbeddings(string prefix, int maxSpan)
        {
            if (_devicePositionEmbeddingCache.TryGetValue(prefix, out var cached)) return cached;
            int halfDim = _hiddenSize / 2;
            double logInc = Math.Log(10000.0) / Math.Max(halfDim - 1, 1);
            var values = new float[checked(maxSpan * _hiddenSize)];
            for (int p = 0; p < maxSpan; ++p)
                for (int d = 0; d < halfDim; ++d)
                {
                    // HF passes left-window12 and tests distance<12. The
                    // compact device window therefore projects distances11..0.
                    double angle = (_maxPast - 1 - p) * Math.Exp(-d * logInc);
                    values[p * _hiddenSize + d] = (float)Math.Sin(angle);
                    values[p * _hiddenSize + halfDim + d] = (float)Math.Cos(angle);
                }
            var sinusoidal = new Tensor(_allocator, DType.Float32, maxSpan, _hiddenSize);
            try
            {
                sinusoidal.SetElementsAsFloat(values);
                string name = ResolveName(prefix, "attn_k_rel") + ".weight";
                if (!_weights.TryGetValue(name, out var weight))
                {
                    _devicePositionEmbeddingCache.Add(prefix, sinusoidal);
                    return sinusoidal;
                }
                int input = (int)weight.Sizes[1], output = (int)weight.Sizes[0];
                using var slice = sinusoidal.Narrow(1, 0, input);
                using var contiguous = Ops.NewContiguous(slice);
                var projected = new Tensor(_allocator, DType.Float32, maxSpan, output);
                try
                {
                    Project(projected, contiguous, name);
                    _devicePositionEmbeddingCache.Add(prefix, projected);
                    return projected;
                }
                catch { projected.Dispose(); throw; }
            }
            finally
            {
                if (!_devicePositionEmbeddingCache.TryGetValue(prefix, out var stored) || !ReferenceEquals(stored, sinusoidal))
                    sinusoidal.Dispose();
            }
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
                var convOut = new Tensor(_allocator, DType.Float32, seqLen, halfDim);
                if (Gemma4AudioTensorOps.UsesDevicePath(_allocator))
                {
                    try
                    {
                        if (!Gemma4AudioTensorOps.TryDepthwise(convOut, gluOut, dwWeight, seqLen, halfDim, kSize))
                            throw new InvalidOperationException("The selected backend does not provide the required Gemma4 F32 causal depthwise convolution.");
                    }
                    catch { convOut.Dispose(); throw; }
                }
                else
                {
                float* dwPtr = GetFloatPtr(dwWeight);
                float* gluPtr = GetFloatPtr(gluOut);
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
                src = ClampedCopy(input, seqLen, (int)input.Sizes[1], cp.InMin, cp.InMax);
            }

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            Project(result, src, weightName);

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
            using var reshaped = q.View(seqLen * _numHeads, _headDim);
            Ops.Mul(reshaped, reshaped, perDimScale);
        }

        private void ApplySiLU(Tensor t)
        {
            Ops.SiLU(t, t);
        }

        private void Clamp(Tensor t, float min, float max) => Ops.Clamp(t, t, min, max);

        private Tensor CloneTensor(Tensor src, int rows, int cols)
        {
            return Ops.NewContiguous(src);
        }

        private Tensor ClampedCopy(Tensor src, int rows, int cols, float minimum, float maximum)
        {
            // A GGML contiguous copy uses host memcpy. A distinct clamp output
            // performs the same copy/clipping on the selected device instead.
            if (Gemma4AudioTensorOps.UsesDevicePath(_allocator)) return Ops.Clamp(null, src, minimum, maximum);
            var copy = CloneTensor(src, rows, cols);
            Clamp(copy, minimum, maximum);
            return copy;
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
            float[] result = new float[_chunkSize * _contextSize];
            for (int r = 0; r < _chunkSize; r++)
            {
                for (int c = 0; c < _contextSize; c++)
                {
                    result[r * _contextSize + c] = IsWithinAudioAttentionHorizon(r, c - _maxPast, _maxPast, _maxFuture) ? 1f : 0f;
                }
            }
            return result;
        }

        // Pinned modeling_gemma4.py sliding_window_mask_function uses strict
        // inequalities; context-left13 is converted to left-window12 first.
        internal static bool IsWithinAudioAttentionHorizon(int query, int key, int leftWindow, int rightWindow)
        {
            long distance = (long)query - key;
            return (distance >= 0 && distance < leftWindow) || (distance < 0 && -distance < rightWindow);
        }

        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            using var weightViewT = _weights[weightName].Transpose();
            // GGML consumes the original [out,in] storage through a logical
            // [in,out] column-strided view. Materializing that transpose would
            // force the native GEMM bridge to repack the whole weight each call.
            transposed = _allocator is GgmlAllocator ? weightViewT.CopyRef() : Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        private void Project(Tensor result, Tensor input, string weightName)
        {
            if (_requireGgmlFloat32Precision && _allocator is GgmlAllocator ga &&
                ga.Context.BackendType == GgmlBackendType.Metal)
            {
                if (!GgmlBasicOps.TryGemma4ProjectionFloat32(result, input, _weights[weightName]))
                    throw new NotSupportedException("Metal precise audio projection requires compatible F32 operands: " + weightName);
                return;
            }
            if (_compressedMatrices.TryGetValue(weightName, out var compressed))
            {
                if (_allocator is GgmlAllocator)
                    GgmlBasicOps.AddmmQuant(result, input, compressed.Pointer,
                        (int)GgmlTensorType.Q8_0, compressed.Input, compressed.Output, compressed.Bytes);
                else if (!TensorSharp.MLX.MlxQuantizedOps.TryAddmmQuantizedToFloat32(result, input,
                    compressed.Pointer, compressed.Pointer, (int)GgmlTensorType.Q8_0,
                    compressed.Input, compressed.Output, compressed.Bytes, preserveFloat32: true))
                    throw new InvalidOperationException("MLX F32 audio compressed projection failed: " + weightName);
            }
            else if (_allocator is GgmlAllocator)
            {
                // The native descriptor cache accepts F32 as well as GGUF
                // quantization. Keep immutable [out,in] weights resident across
                // calls, including CUDA where Metal's host zero-copy is absent.
                var weight = _weights[weightName];
                GgmlBasicOps.AddmmQuant(result, input, weight.Storage.PtrAtElement(weight.StorageOffset),
                    (int)GgmlTensorType.F32, weight.Sizes[1], weight.Sizes[0], checked(weight.ElementCount() * sizeof(float)));
            }
            else if (_allocator is TensorSharp.Cuda.CudaAllocator &&
                _usePreciseCudaLinears)
            {
                if (!TensorSharp.Cuda.CudaGemma4AudioOps.TryProjectionFloat32(result, input, GetOrCreateTransposedWeight(weightName)))
                    throw new NotSupportedException("CUDA precise audio projection requires contiguous F32 operands on one allocator.");
            }
            else Ops.Addmm(result, 0, result, 1, input, GetOrCreateTransposedWeight(weightName));
        }

        #endregion

        public void Dispose()
        {
            foreach (var matrix in _compressedMatrices.Values)
            {
                try
                {
                    if (_allocator is TensorSharp.MLX.MlxAllocator ma)
                        TensorSharp.MLX.MlxQuantizedOps.ReleaseQuantizedWeight(ma, matrix.Pointer);
                    else GgmlBasicOps.InvalidateHostBuffer(matrix.Pointer);
                }
                finally { matrix.Free(); }
            }
            _compressedMatrices.Clear();
            _conformerBlocks.Clear();
            // Every tensor this encoder made (the SSCP projection's contiguous copy above all, which the native SSCP binds
            // as a weight) may have a GGML device copy keyed by its host bytes: drop it before the pool can hand the block
            // out again.
            void Release(Tensor t)
            {
                if (t == null) return;
                if (_allocator is GgmlAllocator) GgmlBasicOps.InvalidateTensorHostBuffer(t);
                t.Dispose();
            }
            Release(_onesForNorm);
            foreach (var w in _transposedWeights.Values)
                Release(w);
            _transposedWeights.Clear();
            _positionEmbeddingCache.Clear();
            foreach (var position in _devicePositionEmbeddingCache.Values) Release(position);
            _devicePositionEmbeddingCache.Clear();
            foreach (var w in _weights.Values)
            {
                if (_allocator is GgmlAllocator) GgmlBasicOps.InvalidateTensorHostBuffer(w);
                w.Dispose();
            }
            _weights.Clear();
        }
    }
}

