// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>V4.1 text executor with its separately prepared vision companion.</summary>
    public sealed class DeepSeek41Model : DeepSeek4Model, IVisionCapableModel, IMultimodalPromptExpander
    {
        private IntPtr _vision;
        private readonly CpuAllocator _visionAllocator = new(BlasEnum.DotNet);
        private readonly DeepSeek41VisionQueue _visionQueue = new();
        internal DeepSeek41ImageProcessor ImageProcessor { get; private set; }
        internal int ImageTokenId { get; private set; }

        /// <summary>The backend the operator asked for, kept because the base
        /// class normalizes its own copy (see DeepSeek4Model.NormalizeBackend)
        /// and the vision companion has to load where the text executor did.</summary>
        private readonly BackendType _requestedBackend;

        public DeepSeek41Model(string ggufPath, BackendType backend, int tpDegree = 1,
            ITensorParallelGroup tpGroup = null, string draftModelPath = null, int layerSplitDegree = 1)
            : base(ggufPath, backend, tpDegree, tpGroup, draftModelPath, layerSplitDegree)
            => _requestedBackend = backend;

        /// <summary>V4.1's executor keeps each request's complete state in its own native slot: the engine sizes its
        /// zero-byte block pool for every running request's context and refuses a request longer than one slot
        /// (InferenceEngine.UsesNativeSequenceSlots), declared here rather than recognised by the architecture's name.</summary>
        public override bool UsesNativeSequenceSlots => true;

        /// <summary>
        /// ggml registry name the vision companion loads on. It must be the text
        /// executor's own backend: the encoder's output rows are handed straight
        /// to the text graph, and the router biases this class attaches are
        /// allocated on the text model's device buffers. A hardcoded "CUDA" here
        /// pulled a GPU into an explicitly CPU-only run — and on a host with no
        /// CUDA device it failed inside the native loader with the generic
        /// "requested backend device is unavailable", naming neither the option
        /// the operator set nor the one that would work.
        /// </summary>
        internal static string ResolveVisionBackendName(BackendType backend)
            => BackendRegistryName(backend) ?? throw new NotSupportedException(
                $"The DeepSeek V4.1 vision companion has no ggml backend for {backend}; " +
                "load the model with --backend ggml_cuda or --backend ggml_cpu.");

        public bool IsVisionEncoderLoaded { get { lock (NativeSync) return _vision != IntPtr.Zero; } }

        public void LoadVisionEncoder(string mmProjPath)
        {
            if (IsDistributedNativeTensorParallel || IsDistributedNativeLayerSplit)
                throw new NotSupportedException("DeepSeek V4.1 distributed tensor parallelism currently accepts text input; vision embeddings are not mirrored by the distributed control channel.");
            lock (NativeSync)
            {
                if (_vision != IntPtr.Zero)
                    throw new InvalidOperationException("The V4.1 vision companion is already loaded.");
                if (NativeHandle == IntPtr.Zero)
                    throw new ObjectDisposedException(nameof(DeepSeek41Model));
                IntPtr vision = GgmlDeepSeek41VisionNative.TSGgml_Dsv41VisionLoad(mmProjPath,
                    ResolveVisionBackendName(_requestedBackend), 0, Math.Min(Environment.ProcessorCount, 32));
                if (vision == IntPtr.Zero)
                    throw new InvalidDataException($"Cannot load DeepSeek V4.1 vision companion {mmProjPath} (see stderr).");
                try
                {
                    int[] info = GgmlDeepSeek41VisionNative.Info(vision);
                    if (info[2] != Config.HiddenSize || info[6] != 129264 ||
                        Tokenizer.LookupToken(ChatTemplate.DeepSeek41ImagePlaceholder) != info[6])
                        throw new InvalidDataException("The V4.1 vision companion does not match this text model/tokenizer.");
                    var processor = new DeepSeek41ImageProcessor(info[0], info[1], info[4], info[3], info[5]);
                    if (GgmlDeepSeek41VisionNative.TSGgml_Dsv41AttachVision(NativeHandle, vision) != 0)
                        throw new InvalidDataException("Cannot attach V4.1 vision companion to the text executor (see stderr).");
                    ImageTokenId = info[6];
                    ImageProcessor = processor;
                    _vision = vision;
                }
                catch
                {
                    GgmlDeepSeek41VisionNative.TSGgml_Dsv41VisionFree(vision);
                    throw;
                }
            }
        }

        internal Tensor EncodeImage(string path)
        {
            lock (NativeSync)
            {
                if (_vision == IntPtr.Zero)
                    throw new InvalidOperationException("Image input requires a V4.1 vision companion: --mmproj with the prepared deepseek41.vision.gguf or llama.cpp's deepseek41v mmproj.");
                var (patches, grid) = ImageProcessor.ProcessImage(path);
                var output = new float[checked(grid.TokenCount * Config.HiddenSize)];
                int rows = GgmlDeepSeek41VisionNative.Encode(_vision, patches, grid.PatchRows, grid.PatchColumns, output);
                if (rows != grid.TokenCount)
                    throw new InvalidOperationException("DeepSeek V4.1 vision encoding failed or returned an unexpected image span (see stderr).");
                var tensor = new Tensor(_visionAllocator, DType.Float32, rows, Config.HiddenSize);
                try { tensor.SetElementsAsFloat(output); return tensor; }
                catch { tensor.Dispose(); throw; }
            }
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            ArgumentNullException.ThrowIfNull(embeddings);
            try
            {
                lock (NativeSync)
                {
                    if (_vision == IntPtr.Zero)
                        throw new InvalidOperationException("Image input requires a V4.1 vision companion: --mmproj with the prepared deepseek41.vision.gguf or llama.cpp's deepseek41v mmproj.");
                    if (embeddings.DimensionCount != 2 || embeddings.Sizes[1] != Config.HiddenSize)
                        throw new ArgumentException("V4.1 image embeddings must have shape [rows, text hidden size].");
                    int rows = checked((int)embeddings.Sizes[0]);
                    Tensor contiguous = embeddings.IsContiguous() ? embeddings : Ops.NewContiguous(embeddings);
                    try
                    {
                        _visionQueue.Add(contiguous.GetElementsAsFloat(checked(rows * Config.HiddenSize)), rows, insertPosition);
                    }
                    finally { if (!ReferenceEquals(contiguous, embeddings)) contiguous.Dispose(); }
                }
            }
            catch
            {
                lock (NativeSync) _visionQueue.Clear();
                throw;
            }
            finally { embeddings.Dispose(); }
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (NativeSync)
            {
                if (_visionQueue.Count == 0)
                    return base.ForwardCore(tokens);
                try
                {
                    var (mask, rows, embeddings) = _visionQueue.Materialize(tokens, Config.HiddenSize, ImageTokenId);
                    var logits = new float[Config.VocabSize];
                    if (GgmlDeepSeek41VisionNative.Forward(NativeHandle, tokens, mask, embeddings, rows, logits) != 0)
                        throw new InvalidOperationException("DeepSeek V4.1 image/text forward failed (see stderr).");
                    return logits;
                }
                finally { _visionQueue.Clear(); }
            }
        }

        protected override void ResetKVCacheCore()
        {
            lock (NativeSync) { _visionQueue.Clear(); base.ResetKVCacheCore(); }
        }

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(ModelMultimodalInjector injector,
            List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessDeepSeek41History(this, history, inputTokens);

        public override void Dispose()
        {
            lock (NativeSync)
            {
                _visionQueue.Clear();
                if (_vision != IntPtr.Zero)
                {
                    GgmlDeepSeek41VisionNative.TSGgml_Dsv41VisionFree(_vision);
                    _vision = IntPtr.Zero;
                }
                base.Dispose();
            }
        }
    }

    // Compact rows avoid an allocation proportional to text prompt length * hidden size.
    // The queue is consumed by exactly the next Forward call, including partial-image
    // prefill chunks supplied by ModelMultimodalInjector.
    internal sealed class DeepSeek41VisionQueue
    {
        private readonly List<(float[] Values, int Rows, int Start)> _spans = new();
        public int Count => _spans.Count;
        public void Clear() => _spans.Clear();
        public void Add(float[] values, int rows, int start)
        {
            if (start < 0 || rows <= 0)
                throw new ArgumentOutOfRangeException(nameof(start), "Image spans need a nonnegative start and positive row count.");
            _spans.Add((values, rows, start));
        }

        public (byte[] Mask, int Rows, float[] Embeddings) Materialize(int[] tokens, int hiddenSize, int imageTokenId)
        {
            _spans.Sort((a, b) => a.Start.CompareTo(b.Start));
            int total = 0, end = 0;
            foreach (var span in _spans)
            {
                if (span.Start < end || (long)span.Start + span.Rows > tokens.Length ||
                    span.Values.Length != (long)span.Rows * hiddenSize)
                    throw new ArgumentException("Queued V4.1 image spans overlap or do not fit this prefill chunk.");
                end = span.Start + span.Rows;
                total = checked(total + span.Rows);
                for (int i = span.Start; i < end; i++)
                    if (tokens[i] != imageTokenId)
                        throw new ArgumentException("Queued image embeddings do not align with V4.1 image tokens.");
            }
            var mask = new byte[tokens.Length];
            var data = new float[checked(total * hiddenSize)];
            int offset = 0;
            foreach (var span in _spans)
            {
                Array.Fill(mask, (byte)1, span.Start, span.Rows);
                span.Values.CopyTo(data, offset);
                offset += span.Values.Length;
            }
            return (mask, total, data);
        }
    }
}
