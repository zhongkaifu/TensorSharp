// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen-Image-2.1's conditioner and VAE as components of the diffusion runtime. Both release their device memory after every
// phase they run (IResidentComponent.ReleaseAfterPhase), as the pipeline before the runtime did: the text encoder is gone before
// the transformer runs, the transformer's device copies before the VAE decodes at full resolution.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>Qwen3-VL-8B's last hidden states of the chat-templated prompt (and of each reference's vision slots, with the
    /// vision tower's DeepStack taps) as an <see cref="IConditioner"/>. The encoder is made at a phase's first encode, shared by
    /// its CFG branches, and disposed when the phase ends.</summary>
    internal sealed class QwenImage21ConditionerComponent : IConditioner
    {
        private readonly string _textPath, _visionPath;
        private readonly BackendType _backend;
        private readonly Action _releaseCompute;
        private QwenImage21Conditioner _inner;

        /// <param name="releaseCompute">Drops the GGML scratch and the model's device memory (the retired pipeline's
        /// ReleaseComputeBuffers).</param>
        public QwenImage21ConditionerComponent(string textPath, string visionPath, BackendType backend, Action releaseCompute)
        {
            _textPath = textPath ?? throw new ArgumentNullException(nameof(textPath));
            _visionPath = visionPath;
            _backend = backend;
            _releaseCompute = releaseCompute;
        }

        public bool ReleaseAfterPhase => true;

        public long WeightBytes => Length(_textPath) + Length(_visionPath);

        private static long Length(string path)
        {
            try { return string.IsNullOrEmpty(path) ? 0 : new FileInfo(path).Length; }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
        }

        /// <summary>The prompt's rows and each reference's vision slots (a 32-pixel cell a slot) at Qwen3-VL-8B's width.</summary>
        public long WorkingBytes(DiffusionWorkload workload)
        {
            long rows = Math.Max(1, (workload.Prompt?.Length ?? 0) / 4) + 64;
            foreach (PresentedImage p in workload.Pictures) rows += (long)(p.Image.Width / 32) * (p.Image.Height / 32);
            return 256L * 1024 * 1024 + rows * QwenImage21Family.TextDim * 4L * 24;
        }

        public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var refs = new RgbImage[pictures?.Count ?? 0];
            for (int i = 0; i < refs.Length; i++) refs[i] = pictures[i].Image;
            _inner ??= new QwenImage21Conditioner(_textPath, _visionPath, _backend);
            var (hidden, length, slots) = _inner.EncodePrompt(prompt, refs);
            return new ConditioningSet(hidden, length, QwenImage21Family.TextDim, slots);
        }

        public void ReleaseWorkspace() { }

        public void ReleaseDevice()
        {
            _inner?.Dispose();
            _inner = null;
            _releaseCompute?.Invoke();
            // On the cpu backend the encoder's packed vision weights and working buffers are managed arrays (GBs for an
            // edit); nothing during the denoise allocates managed memory, so without a collection they stay committed.
            if (_backend == BackendType.Cpu)
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: false);
        }

        public void Dispose()
        {
            _inner?.Dispose();
            _inner = null;
        }
    }

    /// <summary>The single-frame Qwen-Image-2.1 RGBA VAE (64 normalized channels at 1/16) as an <see cref="IVaeCodec"/>: a
    /// deterministic encoder (the posterior's mean), the decoder, and a preview that average-pools the latent to at most 24 cells
    /// a side first.</summary>
    internal sealed class QwenImage21VaeCodec : IVaeCodec, IGgmlHostMemoryOwner
    {
        private readonly Func<IFloatTensorStore> _source;
        private readonly BackendType _backend;
        private readonly Action<IGgmlHostMemoryOwner> _release;
        private readonly long _bytes;
        private QwenImage21Vae _vae;

        /// <param name="release">Drops the GGML scratch and this codec's device memory.</param>
        public QwenImage21VaeCodec(Func<IFloatTensorStore> source, BackendType backend, long bytes, Action<IGgmlHostMemoryOwner> release)
        {
            _source = source ?? throw new ArgumentNullException(nameof(source));
            _backend = backend;
            _bytes = bytes;
            _release = release;
        }

        private QwenImage21Vae Vae => _vae ??= new QwenImage21Vae(_source(), _backend);

        public bool ReleaseAfterPhase => true;

        public long WeightBytes => _bytes * 2;   // a BF16 release, upcast to F32

        /// <summary>The decoder's full-resolution feature maps: about 128 channels of F32 at the output's size, three of them.</summary>
        public long WorkingBytes(DiffusionWorkload workload) =>
            256L * 1024 * 1024 + (long)workload.Target.Height * 16 * workload.Target.Width * 16 * 128 * 4 * 3;

        public LatentPosterior Encode(RgbImage image, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            VaeLatent latent = Vae.Encode(image);
            return new LatentPosterior(new LatentShape(latent.Channels, 1, latent.Height, latent.Width), latent.Data, null);
        }

        public RgbImage Decode(LatentTensor latent, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(latent);
            cancellationToken.ThrowIfCancellationRequested();
            LatentShape s = latent.Shape;
            return Vae.Decode(new VaeLatent(s.Channels, s.Height, s.Width, latent.Data));
        }

        /// <summary>The latent average-pooled to at most 24 cells a side (a 384-pixel preview), then decoded.</summary>
        public RgbImage Preview(LatentTensor latent, int maxSide)
        {
            ArgumentNullException.ThrowIfNull(latent);
            LatentShape s = latent.Shape;
            int height = s.Height, width = s.Width, channels = s.Channels, count = height * width;
            int factor = Math.Max(1, (Math.Max(height, width) + 23) / 24);
            if (factor == 1) return Vae.Decode(new VaeLatent(channels, height, width, latent.Data));
            int h = Math.Max(1, height / factor), w = Math.Max(1, width / factor);
            var pooled = new float[(long)channels * h * w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < channels; c++)
                    {
                        float sum = 0;
                        int n = 0;
                        for (int yy = y * factor; yy < Math.Min(height, (y + 1) * factor); yy++)
                            for (int xx = x * factor; xx < Math.Min(width, (x + 1) * factor); xx++)
                            { sum += latent.Data[c * count + yy * width + xx]; n++; }
                        pooled[(c * h + y) * w + x] = sum / n;
                    }
            return Vae.Decode(new VaeLatent(channels, h, w, pooled));
        }

        public void ReleaseWorkspace() { }

        /// <summary>The fused decoder graph's device copies and the GGML scratch go; the F32 weights stay on the host.</summary>
        public void ReleaseDevice() => _release?.Invoke(this);

        void IGgmlHostMemoryOwner.CollectHostMemory(List<(IntPtr Base, long Length)> ranges, List<IntPtr> keys) =>
            (_vae as IGgmlHostMemoryOwner)?.CollectHostMemory(ranges, keys);

        public void Dispose()
        {
            _vae?.Dispose();
            _vae = null;
        }
    }
}
