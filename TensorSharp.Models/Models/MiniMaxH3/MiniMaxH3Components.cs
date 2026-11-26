// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's conditioner, video codec and audio codec as the diffusion runtime's components: the Qwen3-VL-32B text encoder
// (with its vision tower), the video VAE (single-frame and causal 3-D encoders, the tiled ViT decoder) and the 32 kHz stereo
// audio VAE, each made on first use and handing its device memory back when the runtime releases it. The text encoder is the
// largest upload of a clip and runs once, so it is released after every encode (the 17 GB encoder never sits beside the
// denoiser), as the retired pipeline disposed it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The MiniMax-H3 text encoder (and vision tower) as an <see cref="IConditioner"/>. The H3 family presents its
    /// prompt and pictures itself (<see cref="Encoder"/>); releasing the component disposes the encoder, and the next encode
    /// opens it again.</summary>
    internal sealed class MiniMaxH3Conditioner : IConditioner
    {
        private readonly string _path;
        private readonly BackendType _backend;
        private readonly IAllocator _allocator;
        private readonly object _gate = new();
        private MiniMaxH3TextEncoder _encoder;
        private MiniMaxH3TextEncoderConfig _config;
        private bool _disposed;

        public MiniMaxH3Conditioner(string path, BackendType backend, IAllocator allocator)
        {
            _path = path ?? throw new ArgumentNullException(nameof(path));
            _backend = backend;
            _allocator = allocator;
        }

        public string Path => _path;

        /// <summary>Seconds the last <see cref="Encoder"/> open took (0 when it was open already).</summary>
        public double LastOpenSeconds { get; private set; }

        /// <summary>The encoder, opened on first use since the last release.</summary>
        public MiniMaxH3TextEncoder Encoder
        {
            get
            {
                lock (_gate)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    if (_encoder != null)
                    {
                        LastOpenSeconds = 0;
                        return _encoder;
                    }
                    var clock = System.Diagnostics.Stopwatch.StartNew();
                    _encoder = new MiniMaxH3TextEncoder(_path, null, _backend, _allocator);
                    LastOpenSeconds = clock.Elapsed.TotalSeconds;
                    return _encoder;
                }
            }
        }

        /// <summary>The trunk's file: its weights are uploaded as stored.</summary>
        public long WeightBytes
        {
            get
            {
                try { return new FileInfo(_path).Length; }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
            }
        }

        /// <summary>The trunk's activations for the prompt (and, with pictures, its vision rows): MiniMax-H3-Image's estimate of
        /// the same encoder.</summary>
        public long WorkingBytes(DiffusionWorkload workload) => WorkingBytesFor(workload.Prompt, 0);

        /// <summary>The trunk's compute memory for <paramref name="prompt"/> and <paramref name="visionRows"/> rows of vision tokens.</summary>
        public long WorkingBytesFor(string prompt, long visionRows)
        {
            MiniMaxH3TextEncoderConfig config = Config();
            return config == null ? 0 : MiniMaxH3Image.MiniMaxH3ImageConditioner.WorkingBytesFor(config, prompt, visionRows);
        }

        private MiniMaxH3TextEncoderConfig Config()
        {
            lock (_gate)
            {
                if (_config != null) return _config;
                if (_encoder != null) return _config = _encoder.Config;
                try
                {
                    using var gguf = new GgufFile(_path);
                    return _config = MiniMaxH3TextEncoder.DetectConfig(gguf);
                }
                catch (Exception e) when (e is IOException or InvalidOperationException or ArgumentException) { return null; }
            }
        }

        /// <summary>Released after every encode, as the retired pipeline disposed it: the denoiser needs the room.</summary>
        public bool ReleaseAfterPhase => true;

        public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken)
        {
            if (pictures is { Count: > 0 })
                throw new NotSupportedException("MiniMax-H3 presents its pictures through its family (keyframes and references).");
            cancellationToken.ThrowIfCancellationRequested();
            MiniMaxH3TextEncoder te = Encoder;
            List<int> ids = te.Tokenize(prompt ?? string.Empty);
            if (ids.Count == 0) ids.Add(te.Tokenizer.BosTokenId);
            float[] hidden = te.Encode(ids);
            return new ConditioningSet(hidden, ids.Count, te.Config.Hidden);
        }

        public void ReleaseWorkspace() { }

        public void ReleaseDevice()
        {
            lock (_gate)
            {
                _encoder?.Dispose();
                _encoder = null;
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _encoder?.Dispose();
                _encoder = null;
            }
        }
    }

    /// <summary>The MiniMax-H3 video VAE as a runtime component: made on first use, kept between clips; releasing it hands
    /// its device copies back and keeps it (the next use uploads from memory).</summary>
    internal sealed class MiniMaxH3VideoCodec : IResidentComponent
    {
        private readonly string _path;
        private readonly BackendType _backend;
        private readonly IAllocator _allocator;
        private readonly object _gate = new();
        private MiniMaxH3VideoVae _vae;
        private bool _disposed;

        public MiniMaxH3VideoCodec(string path, BackendType backend, IAllocator allocator)
        {
            _path = path ?? throw new ArgumentNullException(nameof(path));
            _backend = backend;
            _allocator = allocator;
        }

        /// <summary>Whether the VAE is made (a decode then needs no open).</summary>
        public bool Made { get { lock (_gate) return _vae != null; } }

        public MiniMaxH3VideoVae Vae
        {
            get
            {
                lock (_gate)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    return _vae ??= new MiniMaxH3VideoVae(_path, _backend, _allocator);
                }
            }
        }

        /// <summary>The file: the VAE is dense F16, uploaded as stored.</summary>
        public long WeightBytes
        {
            get
            {
                try { return new FileInfo(_path).Length; }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
            }
        }

        /// <summary>The decode's activations: bounded by its 256-pixel tiles and five-latent-frame chunks, an eighth of the
        /// weights (the retired pipeline's estimate of the decode beside its weights).</summary>
        public long WorkingBytes(DiffusionWorkload workload) => WeightBytes / 8;

        /// <summary>On Metal every weight the VAE binds is a zero-copy wrap of the mapped file held resident, which wires its
        /// pages: kept between clips, the VAEs and the denoiser sat beside the next clip's 17 GB text encoder (33.3 GB of
        /// wired memory against 19.1 GB released, TensorAgent's tier sizing). So on <c>ggml_metal</c> the codecs hand their
        /// device copies back after every phase, as the retired pipeline did before each text encoder; the pages stay in the
        /// file cache and the next bind wraps them again. Elsewhere the residency plan decides.</summary>
        public bool ReleaseAfterPhase => _backend == BackendType.GgmlMetal;

        public void ReleaseWorkspace() { }

        public void ReleaseDevice()
        {
            lock (_gate) _vae?.ReleaseDeviceResidency();
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _vae?.Dispose();
                _vae = null;
            }
        }
    }

    /// <summary>The MiniMax-H3 audio VAE as a runtime component (made on first use, kept between clips).</summary>
    internal sealed class MiniMaxH3AudioCodec : IResidentComponent
    {
        private readonly string _path;
        private readonly BackendType _backend;
        private readonly bool _direct;
        private readonly IAllocator _allocator;
        private readonly object _gate = new();
        private MiniMaxH3AudioVae _vae;
        private bool _disposed;

        public MiniMaxH3AudioCodec(string path, BackendType backend, bool direct, IAllocator allocator)
        {
            _path = path ?? throw new ArgumentNullException(nameof(path));
            _backend = backend;
            _direct = direct;
            _allocator = allocator;
        }

        public MiniMaxH3AudioVae Vae
        {
            get
            {
                lock (_gate)
                {
                    ObjectDisposedException.ThrowIf(_disposed, this);
                    return _vae ??= new MiniMaxH3AudioVae(_path, _backend, _direct, _allocator);
                }
            }
        }

        public long WeightBytes
        {
            get
            {
                try { return new FileInfo(_path).Length; }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return 0; }
            }
        }

        public long WorkingBytes(DiffusionWorkload workload) => WeightBytes / 8;

        /// <summary>As the video VAE's (<see cref="MiniMaxH3VideoCodec.ReleaseAfterPhase"/>).</summary>
        public bool ReleaseAfterPhase => _backend == BackendType.GgmlMetal;

        public void ReleaseWorkspace() { }

        public void ReleaseDevice()
        {
            lock (_gate) _vae?.ReleaseDeviceResidency();
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _vae?.Dispose();
                _vae = null;
            }
        }
    }
}
