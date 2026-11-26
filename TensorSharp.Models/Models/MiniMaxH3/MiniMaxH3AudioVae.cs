// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 audio VAE: 32 latent channels at 40 Hz -> stereo 32 kHz PCM.
//
// The decoder is mono, so the two stereo planes are decoded independently — which
// is also how the model produced them, as two separate token streams anchored to
// the left and right of the video's width axis.
//
// The checkpoint stores its convolutions ALREADY weight-norm-fused (plain .weight,
// no weight_g/weight_v) and ships the kaiser resampling filters as buffers, so
// nothing here has to fuse or synthesize anything.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The MiniMax-H3 audio VAE (BigVGAN vocoder).</summary>
    public sealed class MiniMaxH3AudioVae : IDisposable
    {
        public const int SampleRate = 32000;
        public const int LatentChannels = 32;
        /// <summary>Product of the upsample rates; 32000 / 800 = the 40 Hz latent rate.</summary>
        public const int TotalUpsample = 800;
        public const int AmpsPerStage = 3;      // resblock_kernel_sizes = [3, 7, 11]
        public const int ActivationKernel = 12; // alias-free resample taps

        private static readonly int[] Rates = { 5, 5, 2, 2, 2, 2, 2 };
        /// <summary>Encoder downsample strides. NOT the reverse of <see cref="Rates"/>:
        /// the decoder upsamples in seven stages and the encoder downsamples in five.
        /// Only the product, 800, is shared.</summary>
        private static readonly int[] EncoderStrides = { 2, 4, 4, 5, 5 };
        /// <summary>Dilations of the three residual units inside each encoder stage.</summary>
        private static readonly int[] EncoderDilations = { 1, 3, 9 };
        public const int TrunkChannels = 2048;
        public const int AttnHeads = 8;
        private static readonly int[] ResblockKernels = { 3, 7, 11 };
        private static readonly int[] Dilations = { 1, 3, 5 };

        private readonly SafetensorsModel _model;
        private readonly List<GCHandle> _pins = new();
        private readonly List<IntPtr> _bound = new();
        private readonly H3Conv1d[] _convs;
        private readonly H3Act1d[] _acts;
        private readonly int[] _rates;
        // Non-null on direct CPU, CUDA and MLX backends.
        private readonly IAllocator _allocator;
        private readonly MiniMaxH3DirectAudioVae _direct;
        private MiniMaxH3DirectAudioVaeEncoder _directEnc;
        private readonly bool _useDirect;
        private GCHandle _convsPin, _actsPin, _ratesPin;
        private H3Conv1d _decInProj, _convPre, _convPost;
        private H3AudioEncBlock[] _encBlocks;
        private GCHandle _encBlocksPin;
        private H3Conv1d _encConvIn, _encFinalConv, _meanProj;
        private H3Snake1d _encFinalAct;
        private H3AudioAttnProj _pre;
        private bool _encoderBuilt;
        private bool _disposed;

        public float[] LatentsMean { get; }
        public float[] LatentsStd { get; }
        public int NumStages => Rates.Length;

        public MiniMaxH3AudioVae(string path) : this(path, BackendType.GgmlCuda, false) { }

        public MiniMaxH3AudioVae(string path, BackendType backend, bool useDirect, IAllocator allocator = null)
        {
            _useDirect = useDirect;
            _allocator = allocator;
            _model = SafetensorsModel.Open(path);
            if (!_model.HasTensor("dec_in_proj.weight"))
                throw new InvalidOperationException(
                    $"'{path}' is not a MiniMax-H3 audio VAE (no dec_in_proj).");

            _decInProj = Conv("dec_in_proj", padding: 0);
            _convPre = Conv("decoder.conv_pre", padding: 3);
            _convPost = Conv("decoder.conv_post", padding: 3);

            // convs: the seven transposed upsamples first, then per (stage, amp) the
            // three dilated convs1 followed by the three convs2.
            var convs = new List<H3Conv1d>();
            var acts = new List<H3Act1d>();
            for (int s = 0; s < Rates.Length; s++)
                convs.Add(Conv($"decoder.ups.{s}.0", padding: 0, transposed: true));
            for (int s = 0; s < Rates.Length; s++)
                for (int a = 0; a < AmpsPerStage; a++)
                {
                    int rb = s * AmpsPerStage + a;
                    int k = ResblockKernels[a];
                    for (int dl = 0; dl < 3; dl++)
                        convs.Add(Conv($"decoder.resblocks.{rb}.convs1.{dl}",
                            padding: (k * Dilations[dl] - Dilations[dl]) / 2,
                            dilation: Dilations[dl]));
                    for (int dl = 0; dl < 3; dl++)
                        convs.Add(Conv($"decoder.resblocks.{rb}.convs2.{dl}",
                            padding: (k - 1) / 2));
                    for (int ai = 0; ai < 6; ai++)
                        acts.Add(Act($"decoder.resblocks.{rb}.activations.{ai}"));
                }
            acts.Add(Act("decoder.activation_post"));

            _convs = convs.ToArray();
            _acts = acts.ToArray();
            _rates = (int[])Rates.Clone();
            _convsPin = GCHandle.Alloc(_convs, GCHandleType.Pinned);
            _actsPin = GCHandle.Alloc(_acts, GCHandleType.Pinned);
            _ratesPin = GCHandle.Alloc(_rates, GCHandleType.Pinned);

            LatentsMean = _model.HasTensor("latents_mean") ? _model.ReadFloat32("latents_mean") : null;
            LatentsStd = _model.HasTensor("latents_std") ? _model.ReadFloat32("latents_std") : null;

            // Backends with no ggml graph to call decode through the managed vocoder.
            if (_useDirect) _direct = BuildDirect();
        }

        // ---- direct (managed) decoder ---------------------------------------

        /// <summary>Rebuild the decoder weights as managed float arrays for the
        /// backends with no ggml graph to call. Same tensors, same order, same
        /// padding/dilation arithmetic as the pinned-pointer path above.</summary>
        /// <summary>One managed 1-D conv descriptor. Shared by the decoder and the
        /// encoder builders so the two cannot drift in how they read a kernel.</summary>
        private MiniMaxH3DirectAudioVae.Conv ReadConv(string prefix, int padding, int dilation = 1,
                                                      bool transposed = false, int stride = 1)
        {
            string weight = prefix + ".weight";
            var info = _model.TensorOwners[weight].GetInfo(weight);
            return new MiniMaxH3DirectAudioVae.Conv
            {
                W = _model.ReadFloat32(weight),
                B = _model.HasTensor(prefix + ".bias") ? _model.ReadFloat32(prefix + ".bias") : null,
                Oc = (int)info.Shape[0],
                Ic = (int)info.Shape[1],
                K = (int)info.Shape[2],
                Stride = stride,
                Pad = padding,
                Dilation = dilation,
                Transposed = transposed,
            };
        }

        /// <summary>
        /// Build the managed encoder on first use. Mirrors EnsureEncoder tensor for
        /// tensor: the encoder stores Snake alpha LINEARLY (the decoder stores it in
        /// log scale), the DAC downsample convention is kernel 2*stride with padding
        /// ceil(stride/2), and the QKV bias is assembled on the host as
        /// concat(q_bias, zeros, v_bias) because the checkpoint has no key bias.
        /// </summary>
        private MiniMaxH3DirectAudioVaeEncoder BuildDirectEncoder()
        {
            MiniMaxH3DirectAudioVaeEncoder.Snake ReadSnake(string prefix)
                => new(_model.ReadFloat32(prefix + ".alpha"));

            MiniMaxH3DirectAudioVaeEncoder.Lin ReadLin(string prefix)
            {
                string weight = prefix + ".weight";
                var info = _model.TensorOwners[weight].GetInfo(weight);
                return new MiniMaxH3DirectAudioVaeEncoder.Lin
                {
                    W = _model.ReadFloat32(weight),
                    B = _model.HasTensor(prefix + ".bias") ? _model.ReadFloat32(prefix + ".bias") : null,
                    Ne0 = (int)info.Shape[1],
                    Ne1 = (int)info.Shape[0],
                };
            }

            MiniMaxH3DirectAudioVaeEncoder.ResUnit ReadUnit(string prefix, int dilation) => new()
            {
                Act1 = ReadSnake($"{prefix}.0"),
                // k7 dilated, padded to keep the length: (7*d - d)/2 = 3d.
                Conv1 = ReadConv($"{prefix}.1", padding: 3 * dilation, dilation: dilation),
                Act2 = ReadSnake($"{prefix}.2"),
                Conv2 = ReadConv($"{prefix}.3", padding: 0),
            };

            var blocks = new MiniMaxH3DirectAudioVaeEncoder.Block[EncoderStrides.Length];
            for (int i = 0; i < EncoderStrides.Length; i++)
            {
                string bp = $"encoder.block.{i + 1}.block";
                int stride = EncoderStrides[i];
                blocks[i] = new MiniMaxH3DirectAudioVaeEncoder.Block
                {
                    Units = new[]
                    {
                        ReadUnit($"{bp}.0.block", EncoderDilations[0]),
                        ReadUnit($"{bp}.1.block", EncoderDilations[1]),
                        ReadUnit($"{bp}.2.block", EncoderDilations[2]),
                    },
                    Act = ReadSnake($"{bp}.3"),
                    // DAC convention: kernel = 2*stride, padding = ceil(stride/2).
                    Down = ReadConv($"{bp}.4", padding: (stride + 1) / 2, stride: stride),
                };
            }

            float[] q = _model.ReadFloat32("pre_block.attn.q_bias");
            float[] v = _model.ReadFloat32("pre_block.attn.v_bias");
            var qkvBias = new float[3 * TrunkChannels];
            Array.Copy(q, 0, qkvBias, 0, q.Length);
            // the middle third stays zero: there is no key bias in the checkpoint
            Array.Copy(v, 0, qkvBias, 2 * TrunkChannels, v.Length);

            var pre = new MiniMaxH3DirectAudioVaeEncoder.AttnBlock
            {
                Norm1W = _model.ReadFloat32("pre_block.norm1.weight"),
                Norm1B = _model.ReadFloat32("pre_block.norm1.bias"),
                Norm3W = _model.ReadFloat32("pre_block.norm3.weight"),
                Norm3B = _model.ReadFloat32("pre_block.norm3.bias"),
                Norm2W = _model.ReadFloat32("pre_block.norm2.weight"),
                Norm2B = _model.ReadFloat32("pre_block.norm2.bias"),
                MlpNormW = _model.ReadFloat32("pre_block.mlp.norm.weight"),
                MlpNormB = _model.ReadFloat32("pre_block.mlp.norm.bias"),
                Qkv = ReadLin("pre_block.attn.qkv"),
                QkvBias = qkvBias,
                AttnProj = ReadLin("pre_block.attn.proj"),
                Proj = ReadLin("pre_block.proj"),
                W0 = ReadLin("pre_block.mlp.w0"),
                W1 = ReadLin("pre_block.mlp.w1"),
                W2 = ReadLin("pre_block.mlp.w2"),
                Heads = AttnHeads,
            };

            return new MiniMaxH3DirectAudioVaeEncoder(
                ReadConv("encoder.block.0", padding: 3),
                blocks,
                ReadSnake("encoder.block.6"),
                ReadConv("encoder.block.7", padding: 1),
                pre,
                ReadConv("mean_proj", padding: 0),
                LatentChannels, TrunkChannels, 1e-5f, _allocator);
        }

        private MiniMaxH3DirectAudioVae BuildDirect()
        {
            MiniMaxH3DirectAudioVae.Act ReadAct(string prefix)
            {
                // alpha/beta are stored in log scale (snake_logscale); exponentiate
                // here and fold in the divide-by-zero guard, exactly as the GGML
                // path does on the host.
                float[] alpha = _model.ReadFloat32(prefix + ".act.alpha");
                float[] beta = _model.ReadFloat32(prefix + ".act.beta");
                for (int i = 0; i < alpha.Length; i++) alpha[i] = MathF.Exp(alpha[i]);
                for (int i = 0; i < beta.Length; i++) beta[i] = MathF.Exp(beta[i]) + 1e-9f;
                float[] up = _model.ReadFloat32(prefix + ".upsample.filter");
                Array.Reverse(up);   // zero-stuff + convolve needs the reversed filter
                return new MiniMaxH3DirectAudioVae.Act
                {
                    Alpha = alpha,
                    Beta = beta,
                    UpFilter = up,
                    DownFilter = _model.ReadFloat32(prefix + ".downsample.lowpass.filter"),
                    Channels = alpha.Length,
                    Kernel = ActivationKernel,
                };
            }

            var convs = new List<MiniMaxH3DirectAudioVae.Conv>();
            var acts = new List<MiniMaxH3DirectAudioVae.Act>();
            for (int s = 0; s < Rates.Length; s++)
                convs.Add(ReadConv($"decoder.ups.{s}.0", padding: 0, transposed: true));
            for (int s = 0; s < Rates.Length; s++)
                for (int a = 0; a < AmpsPerStage; a++)
                {
                    int rb = s * AmpsPerStage + a;
                    int k = ResblockKernels[a];
                    for (int dl = 0; dl < 3; dl++)
                        convs.Add(ReadConv($"decoder.resblocks.{rb}.convs1.{dl}",
                            padding: (k * Dilations[dl] - Dilations[dl]) / 2,
                            dilation: Dilations[dl]));
                    for (int dl = 0; dl < 3; dl++)
                        convs.Add(ReadConv($"decoder.resblocks.{rb}.convs2.{dl}",
                            padding: (k - 1) / 2));
                    for (int ai = 0; ai < 6; ai++)
                        acts.Add(ReadAct($"decoder.resblocks.{rb}.activations.{ai}"));
                }
            acts.Add(ReadAct("decoder.activation_post"));

            return new MiniMaxH3DirectAudioVae(
                ReadConv("dec_in_proj", padding: 0),
                ReadConv("decoder.conv_pre", padding: 3),
                ReadConv("decoder.conv_post", padding: 3),
                convs.ToArray(), acts.ToArray(), (int[])Rates.Clone(), AmpsPerStage, _allocator);
        }

        // ---- weights --------------------------------------------------------

        private H3Conv1d Conv(string prefix, int padding, int dilation = 1, bool transposed = false,
                              int stride = 1)
        {
            string weight = prefix + ".weight";
            if (!_model.TensorOwners.TryGetValue(weight, out var owner))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            var info = owner.GetInfo(weight);
            if (info.Shape.Length != 3)
                throw new InvalidOperationException(
                    $"'{weight}' has rank {info.Shape.Length}; expected a 1-D conv kernel.");
            if (!owner.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            _bound.Add(ptr);

            // torch stores [OC, IC, K] (and [Cin, Cout, K] when transposed); reversing
            // the dims for ggml gives [K, IC, OC] either way. Only the bias length
            // differs, which is why it is carried separately.
            long oc = info.Shape[0], ic = info.Shape[1], k = info.Shape[2];
            return new H3Conv1d
            {
                W = ptr,
                B = _model.HasTensor(prefix + ".bias") ? PinF32(prefix + ".bias") : IntPtr.Zero,
                K = k, Ic = ic, Oc = oc,
                BiasLen = transposed ? ic : oc,
                Type = (int)GgmlTensorType.F32,
                Stride = stride,
                Padding = padding,
                Dilation = dilation,
            };
        }

        private H3Act1d Act(string prefix)
        {
            // alpha/beta are stored in log scale (snake_logscale). Exponentiating here
            // — and folding the divide-by-zero guard into beta — keeps the graph free
            // of scalar constants, which a no_alloc ggml context cannot allocate.
            float[] alpha = _model.ReadFloat32(prefix + ".act.alpha");
            float[] beta = _model.ReadFloat32(prefix + ".act.beta");
            for (int i = 0; i < alpha.Length; i++) alpha[i] = MathF.Exp(alpha[i]);
            for (int i = 0; i < beta.Length; i++) beta[i] = MathF.Exp(beta[i]) + 1e-9f;
            return new H3Act1d
            {
                Alpha = Pin(alpha),
                Beta = Pin(beta),
                // The upsample is emulated as zero-stuff + convolution, which needs
                // the filter reversed; doing it here keeps ~20 nodes out of the graph.
                UpFilter = PinReversedF32(prefix + ".upsample.filter"),
                DownFilter = PinF32(prefix + ".downsample.lowpass.filter"),
                Channels = alpha.Length,
                Kernel = ActivationKernel,
            };
        }

        private IntPtr Pin(float[] data)
        {
            var h = GCHandle.Alloc(data, GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        private IntPtr PinReversedF32(string name)
        {
            if (!_model.HasTensor(name)) return IntPtr.Zero;
            float[] data = _model.ReadFloat32(name);
            Array.Reverse(data);
            var h = GCHandle.Alloc(data, GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        private IntPtr PinF32(string name)
        {
            if (!_model.HasTensor(name)) return IntPtr.Zero;
            var h = GCHandle.Alloc(_model.ReadFloat32(name), GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        // ---- encoder --------------------------------------------------------

        /// <summary>True when the checkpoint carries the DAC encoder, which reference
        /// audio needs and plain generation never touches.</summary>
        public bool HasEncoder => _model.HasTensor("encoder.block.0.weight");

        /// <summary>Latent frames a waveform of this length encodes to.</summary>
        public static int FramesFor(int samples) => samples / TotalUpsample;

        // Build the encoder descriptors on first use: they are ~110 tensors that a
        // plain text-to-video request never looks at.
        private void EnsureEncoder()
        {
            if (_encoderBuilt) return;
            if (!HasEncoder)
                throw new InvalidOperationException(
                    "this audio VAE checkpoint has no encoder, so reference audio is unavailable.");

            _encConvIn = Conv("encoder.block.0", padding: 3);
            _encBlocks = new H3AudioEncBlock[EncoderStrides.Length];
            for (int i = 0; i < EncoderStrides.Length; i++)
            {
                string bp = $"encoder.block.{i + 1}.block";
                int stride = EncoderStrides[i];
                _encBlocks[i] = new H3AudioEncBlock
                {
                    Unit0 = ResUnit($"{bp}.0.block", EncoderDilations[0]),
                    Unit1 = ResUnit($"{bp}.1.block", EncoderDilations[1]),
                    Unit2 = ResUnit($"{bp}.2.block", EncoderDilations[2]),
                    Act = Snake($"{bp}.3"),
                    // DAC convention: kernel = 2*stride, padding = ceil(stride/2).
                    Down = Conv($"{bp}.4", padding: (stride + 1) / 2, stride: stride),
                };
            }
            _encBlocksPin = GCHandle.Alloc(_encBlocks, GCHandleType.Pinned);
            _encFinalAct = Snake("encoder.block.6");
            _encFinalConv = Conv("encoder.block.7", padding: 1);
            _meanProj = Conv("mean_proj", padding: 0);
            _pre = BuildAttnProjection();
            _encoderBuilt = true;
        }

        private H3AudioEncResUnit ResUnit(string prefix, int dilation) => new()
        {
            Act1 = Snake($"{prefix}.0"),
            // k7 dilated, padded to keep the length: (7*d - d)/2 = 3d.
            Conv1 = Conv($"{prefix}.1", padding: 3 * dilation, dilation: dilation),
            Act2 = Snake($"{prefix}.2"),
            Conv2 = Conv($"{prefix}.3", padding: 0),
        };

        /// <summary>Read one Snake1d. The encoder stores alpha LINEARLY — exponentiating
        /// it, as the decoder's SnakeBeta requires, would be a plausible-looking silent
        /// error, so the two are read by different helpers on purpose.</summary>
        private H3Snake1d Snake(string prefix)
        {
            float[] alpha = _model.ReadFloat32(prefix + ".alpha");
            var eps = new float[alpha.Length];
            for (int i = 0; i < alpha.Length; i++) eps[i] = alpha[i] + 1e-9f;
            return new H3Snake1d
            {
                Alpha = Pin(alpha),
                AlphaEps = Pin(eps),
                Channels = alpha.Length,
            };
        }

        private H3AudioAttnProj BuildAttnProjection()
        {
            // The checkpoint holds q_bias and v_bias but no key bias, so the fused qkv
            // bias is built here with zeros in the middle — exactly what the reference
            // does with its `zero_k_bias` buffer.
            float[] q = _model.ReadFloat32("pre_block.attn.q_bias");
            float[] v = _model.ReadFloat32("pre_block.attn.v_bias");
            var qkvBias = new float[3 * TrunkChannels];
            Array.Copy(q, 0, qkvBias, 0, q.Length);
            Array.Copy(v, 0, qkvBias, 2 * TrunkChannels, v.Length);

            return new H3AudioAttnProj
            {
                Norm1W = PinF32("pre_block.norm1.weight"),
                Norm1B = PinF32("pre_block.norm1.bias"),
                Norm3W = PinF32("pre_block.norm3.weight"),
                Norm3B = PinF32("pre_block.norm3.bias"),
                Norm2W = PinF32("pre_block.norm2.weight"),
                Norm2B = PinF32("pre_block.norm2.bias"),
                MlpNormW = PinF32("pre_block.mlp.norm.weight"),
                MlpNormB = PinF32("pre_block.mlp.norm.bias"),
                Qkv = Lin("pre_block.attn.qkv"),
                QkvBias = Pin(qkvBias),
                AttnProj = Lin("pre_block.attn.proj"),
                Proj = Lin("pre_block.proj"),
                W0 = Lin("pre_block.mlp.w0"),
                W1 = Lin("pre_block.mlp.w1"),
                W2 = Lin("pre_block.mlp.w2"),
                Heads = AttnHeads,
            };
        }

        // A plain 2-D linear, stored [out, in] by torch and read [in, out] here.
        private H3Lin Lin(string prefix)
        {
            var owner = _model.TensorOwners[prefix + ".weight"];
            var info = owner.GetInfo(prefix + ".weight");
            if (!owner.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{prefix}.weight'.");
            _bound.Add(ptr);
            return new H3Lin
            {
                W = ptr,
                B = _model.HasTensor(prefix + ".bias") ? PinF32(prefix + ".bias") : IntPtr.Zero,
                Ne0 = info.Shape[1],
                Ne1 = info.Shape[0],
                Type = (int)GgmlTensorType.F32,
            };
        }

        /// <summary>Encode one mono plane of 32 kHz PCM into its latent, [channel][time].
        ///
        /// <para>The result is in the VAE's own scale; <see cref="NormalizePlane"/> puts
        /// it into the whitened space the denoiser conditions on.</para></summary>
        public float[] EncodeMono(float[] samples)
        {
            if (samples is null || samples.Length < TotalUpsample)
                throw new ArgumentException(
                    $"at least {TotalUpsample} samples (one latent frame) are required.", nameof(samples));
            EnsureEncoder();

            // The trunk's strided convolutions only land on a whole latent frame when
            // the input is a whole multiple of the hop.
            int frames = FramesFor(samples.Length);
            int usable = frames * TotalUpsample;
            var wave = new float[usable];
            Array.Copy(samples, wave, usable);

            if (_useDirect)
            {
                // Pure-managed path: the identical encoder, with no native call.
                _directEnc ??= BuildDirectEncoder();
                return _directEnc.Encode(wave, frames);
            }

            var outp = new float[(long)frames * LatentChannels];
            var wPin = GCHandle.Alloc(wave, GCHandleType.Pinned);
            var oPin = GCHandle.Alloc(outp, GCHandleType.Pinned);
            try
            {
                var args = new H3AudioVaeEncodeArgs
                {
                    StructBytes = Marshal.SizeOf<H3AudioVaeEncodeArgs>(),
                    NumBlocks = _encBlocks.Length,
                    Wave = wPin.AddrOfPinnedObject(),
                    Out = oPin.AddrOfPinnedObject(),
                    ConvIn = _encConvIn,
                    Blocks = _encBlocksPin.AddrOfPinnedObject(),
                    FinalAct = _encFinalAct,
                    FinalConv = _encFinalConv,
                    Pre = _pre,
                    MeanProj = _meanProj,
                    Samples = usable,
                    Frames = frames,
                    LatentChannels = LatentChannels,
                    TrunkChannels = TrunkChannels,
                    Eps = 1e-5f,
                };
                if (!GgmlBasicOps.TryMiniMaxH3AudioVaeEncode(in args))
                    throw new InvalidOperationException("MiniMax-H3 audio encode failed.");
            }
            finally { wPin.Free(); oPin.Free(); }
            return outp;
        }

        // ---- decode ---------------------------------------------------------

        /// <summary>Samples produced for a given latent length.</summary>
        public static int SamplesFor(int latentFrames) => latentFrames * TotalUpsample;

        /// <summary>Undo the latent normalization, in place, on one [channel][time]
        /// plane.
        ///
        /// <para>The denoiser works in a whitened latent space — that is what makes
        /// flow matching from N(0,1) noise well-conditioned — while the decoder wants
        /// the VAE's own scale. <see cref="LatentsStd"/> averages about 1.9, so
        /// skipping this does not produce silence or noise: it produces a track that
        /// is spectrally plausible and merely wrong, which is the hardest kind of
        /// error to notice.</para></summary>
        public void DenormalizePlane(float[] plane, int latentFrames)
        {
            if (LatentsMean == null || LatentsStd == null) return;
            for (int c = 0; c < LatentChannels; c++)
            {
                float std = LatentsStd[c], mean = LatentsMean[c];
                long b = (long)c * latentFrames;
                for (int t = 0; t < latentFrames; t++) plane[b + t] = plane[b + t] * std + mean;
            }
        }

        /// <summary>Apply the latent normalization, in place, on one [channel][time]
        /// plane — the inverse of <see cref="DenormalizePlane"/>, for latents produced
        /// by the encoder and handed to the denoiser.</summary>
        public void NormalizePlane(float[] plane, int latentFrames)
        {
            if (LatentsMean == null || LatentsStd == null) return;
            for (int c = 0; c < LatentChannels; c++)
            {
                float std = LatentsStd[c], mean = LatentsMean[c];
                long b = (long)c * latentFrames;
                for (int t = 0; t < latentFrames; t++) plane[b + t] = (plane[b + t] - mean) / std;
            }
        }

        /// <summary>Decode one mono latent plane. <paramref name="latent"/> is
        /// [channel][time] with time contiguous, i.e. ggml's [T, 32].</summary>
        public float[] DecodeMono(float[] latent, int latentFrames)
        {
            if (latent is null) throw new ArgumentNullException(nameof(latent));
            long expected = (long)latentFrames * LatentChannels;
            if (latent.Length != expected)
                throw new ArgumentException(
                    $"latent must hold {expected} floats for {latentFrames} frames, got {latent.Length}.",
                    nameof(latent));

            int samples = SamplesFor(latentFrames);
            if (_direct != null)
            {
                // Pure-managed path: the identical vocoder, with no native call.
                return _direct.Decode(latent, LatentChannels, latentFrames, samples);
            }

            var outp = new float[samples];
            var zPin = GCHandle.Alloc(latent, GCHandleType.Pinned);
            var oPin = GCHandle.Alloc(outp, GCHandleType.Pinned);
            try
            {
                var args = new H3AudioVaeDecodeArgs
                {
                    StructBytes = Marshal.SizeOf<H3AudioVaeDecodeArgs>(),
                    NumStages = Rates.Length,
                    NumConvs = _convs.Length,
                    NumActs = _acts.Length,
                    Latent = zPin.AddrOfPinnedObject(),
                    Out = oPin.AddrOfPinnedObject(),
                    DecInProj = _decInProj,
                    ConvPre = _convPre,
                    ConvPost = _convPost,
                    Convs = _convsPin.AddrOfPinnedObject(),
                    Acts = _actsPin.AddrOfPinnedObject(),
                    Rates = _ratesPin.AddrOfPinnedObject(),
                    LatentLen = latentFrames,
                    LatentChannels = LatentChannels,
                    AmpsPerStage = AmpsPerStage,
                    Samples = samples,
                    SnakeEps = 1e-9f,
                };
                if (!GgmlBasicOps.TryMiniMaxH3AudioVaeDecode(in args))
                    throw new InvalidOperationException("MiniMax-H3 audio VAE decode failed.");
            }
            finally { zPin.Free(); oPin.Free(); }
            return outp;
        }

        /// <summary>Decode the DiT's packed audio latent into a stereo track.
        /// <paramref name="tokens"/> is the transformer's token layout —
        /// [channel][time][32] with the 32 latent dims contiguous — which has to be
        /// transposed per plane into the decoder's [32][time].
        ///
        /// <para>The denoiser emits a WHITENED latent, so each plane is denormalized
        /// on the way in — exactly as the video path denormalizes before its own
        /// decode. <see cref="DecodeMono"/> is the raw entry point for a latent that
        /// is already in the VAE's native scale.</para></summary>
        public GeneratedVideoAudio DecodeStereo(float[] tokens, int latentFrames)
        {
            const int stereo = 2;
            long perPlane = (long)latentFrames * LatentChannels;
            if (tokens.LongLength != perPlane * stereo)
                throw new ArgumentException(
                    $"expected {perPlane * stereo} floats for {latentFrames} stereo latent frames, " +
                    $"got {tokens.LongLength}.", nameof(tokens));

            var channels = new float[stereo][];
            for (int ch = 0; ch < stereo; ch++)
            {
                var plane = new float[perPlane];
                for (int t = 0; t < latentFrames; t++)
                    for (int c = 0; c < LatentChannels; c++)
                    {
                        plane[(long)c * latentFrames + t] =
                            tokens[(ch * (long)latentFrames + t) * LatentChannels + c];
                    }
                DenormalizePlane(plane, latentFrames);
                channels[ch] = DecodeMono(plane, latentFrames);
            }
            return new GeneratedVideoAudio { Channels = channels, SampleRate = SampleRate };
        }

        /// <summary>Hand the device copies of the weights back without tearing the VAE down
        /// (as the denoiser releases its own); the next encode or decode
        /// binds them again from the same host memory.</summary>
        public void ReleaseDeviceResidency()
        {
            if (_disposed) return;
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _directEnc?.Dispose(); _direct?.Dispose();
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
            _bound.Clear();
            if (_encBlocksPin.IsAllocated) _encBlocksPin.Free();
            if (_convsPin.IsAllocated) _convsPin.Free();
            if (_actsPin.IsAllocated) _actsPin.Free();
            if (_ratesPin.IsAllocated) _ratesPin.Free();
            foreach (var h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _model?.Dispose();
        }
    }
}
