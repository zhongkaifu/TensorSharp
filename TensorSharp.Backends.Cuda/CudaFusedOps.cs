using System;
namespace TensorSharp.Cuda
{
    public static class CudaFusedOps
    {
        /// <summary>Compute the byte count for a sub-weight that is a concatenated
        /// portion (by ne1 column count) of a larger quantized weight.</summary>
        public static long QuantizedWeightSliceBytes(long totalBytes, long totalNe1, long subNe1)
        {
            return totalBytes * subNe1 / totalNe1;
        }
        /// <summary>
        /// Load the CUDA driver library (cuda.dll) so that <see cref="CtxGetCurrent"/>
        /// and <see cref="CtxSetCurrent"/> can be used (they DllImport from cuda.dll).
        /// Safe to call multiple times. Must be called before <see cref="CtxGetCurrent"/>
        /// when the CUDA driver may not already be loaded (e.g. GgmlCuda backend mode).
        /// </summary>
        public static void EnsureCudaDriverLoaded()
        {
            Interop.CudaLibraryResolver.Register();
        }

        /// <summary>Sync the NULL (default) CUDA stream ÔÇö ensures all pending
        /// GDN kernels launched via GdnDirectBridge have completed.</summary>
        public static void SyncNullStream()
        {
            Interop.CudaDriverApi.cuStreamSynchronize(IntPtr.Zero);
        }

        /// <summary>
        /// Save the current CUDA context for the calling thread.
        /// Returns <see cref="IntPtr.Zero"/> if no context is current.
        /// Call <see cref="CtxSetCurrent"/> to restore it later.
        /// </summary>
        public static System.IntPtr CtxGetCurrent()
        {
            Interop.CudaDriverApi.cuCtxGetCurrent(out System.IntPtr ctx);
            return ctx;
        }

        /// <summary>
        /// Restore a previously saved CUDA context on the calling thread.
        /// Pass <see cref="IntPtr.Zero"/> to set no context.
        /// </summary>
        public static void CtxSetCurrent(System.IntPtr ctx)
        {
            Interop.CudaDriverApi.cuCtxSetCurrent(ctx);
        }
        // Go/no-go PoC for the CUDA-graph rearchitecture: measure how much of a
        // launch-heavy op sequence is per-op CPU/WDDM launch overhead (which a captured
        // graph replays in ONE launch) vs GPU compute. `issueOneLaunch` must issue
        // exactly one capturable launch on `onStream`'s stream (no host sync / alloc).
        // Returns wall ms for `iters` per-op launches vs one graph replay of the same.
        public static (double peropMs, double graphMs, bool captured) MeasureGraphReplay(
            Tensor onStream, int iters, System.Action issueOneLaunch)
        {
            var storage = onStream.Storage as CudaStorage;
            if (storage == null) return (0, 0, false);
            System.IntPtr stream = storage.AllocatorImpl.Stream.Handle;
            storage.AllocatorImpl.Context.MakeCurrent();

            for (int i = 0; i < iters; i++) issueOneLaunch();   // warm up
            Interop.CudaDriverApi.cuStreamSynchronize(stream);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) issueOneLaunch();
            Interop.CudaDriverApi.cuStreamSynchronize(stream);
            sw.Stop();
            double peropMs = sw.Elapsed.TotalMilliseconds;

            if (Interop.CudaDriverApi.cuStreamBeginCapture(stream,
                    Interop.CudaDriverApi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) != 0)
                return (peropMs, 0, false);
            for (int i = 0; i < iters; i++) issueOneLaunch();
            if (Interop.CudaDriverApi.cuStreamEndCapture(stream, out System.IntPtr graph) != 0)
                return (peropMs, 0, false);
            if (Interop.CudaDriverApi.cuGraphInstantiateWithFlags(out System.IntPtr exec, graph, 0) != 0)
            {
                Interop.CudaDriverApi.cuGraphDestroy(graph);
                return (peropMs, 0, false);
            }

            Interop.CudaDriverApi.cuGraphLaunch(exec, stream);   // warm up replay
            Interop.CudaDriverApi.cuStreamSynchronize(stream);

            var sw2 = System.Diagnostics.Stopwatch.StartNew();
            Interop.CudaDriverApi.cuGraphLaunch(exec, stream);
            Interop.CudaDriverApi.cuStreamSynchronize(stream);
            sw2.Stop();
            double graphMs = sw2.Elapsed.TotalMilliseconds;

            Interop.CudaDriverApi.cuGraphExecDestroy(exec);
            Interop.CudaDriverApi.cuGraphDestroy(graph);
            return (peropMs, graphMs, true);
        }

        public static bool TryGqaPrefillAttention(
            Tensor result,
            Tensor query,
            Tensor key,
            Tensor value,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale,
            int kvStride = -1)
        {
            return CudaKernelOps.TryGqaPrefillAttention(
                result,
                query,
                key,
                value,
                numQHeads,
                numKVHeads,
                headDim,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale,
                kvStride);
        }

        public static bool TryAttentionSoftmaxWithSinks(
            Tensor scores,
            Tensor sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryAttentionSoftmaxWithSinks(
                scores,
                sinks,
                numHeads,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale);
        }

        public static bool TrySliceColumns(Tensor result, Tensor src, int colOffset, int width)
        {
            return CudaKernelOps.TrySliceColumns(result, src, colOffset, width);
        }

        public static bool TryDeinterleaveQGate(Tensor q, Tensor gate, Tensor src, int rows, int numHeads, int headDim)
        {
            return CudaKernelOps.TryDeinterleaveQGate(q, gate, src, rows, numHeads, headDim);
        }

        /// <summary>Device buffer address of a CUDA tensor's storage (0 for other
        /// backends). Used as an identity component in CUDA-graph cache keys: a
        /// reallocated KV cache yields a new pointer, so stale graphs never hit.</summary>
        public static System.IntPtr GetDevicePointer(Tensor tensor)
        {
            return tensor?.Storage is CudaStorage storage ? storage.DeviceBuffer : System.IntPtr.Zero;
        }

        /// <summary>Force a host-dirty CUDA tensor's contents onto the device NOW.
        /// Used to pre-warm inputs before a CUDA-graph capture so no host-to-device
        /// copy (whose host pointer would be baked into the graph) lands inside it.</summary>
        public static bool TryEnsureDeviceResident(Tensor tensor)
        {
            if (tensor?.Storage is not CudaStorage storage)
                return false;
            storage.EnsureDeviceCurrent();
            return true;
        }

        public static bool TryGqaDecodeAttention(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            return CudaKernelOps.TryGqaDecodeAttention(
                result,
                query,
                keyCache,
                valueCache,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular,
                scale);
        }

        /// <summary>
        /// Returns the decode-attention launch tier used by a CUDA graph:
        /// 0 for the single-block route and 1 for partition-and-reduce.
        /// Graph keys must include this value because the two routes bake
        /// different grids and scratch allocations into the captured graph.
        /// </summary>
        public static int GetGqaDecodeAttentionRouteTier(
            bool keyIsHalf,
            bool circular,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendLen,
            int cacheSize,
            int hasSinks = 0)
        {
            return CudaKernelOps.GetGqaDecodeAttentionRouteTier(
                keyIsHalf,
                circular,
                numQHeads,
                numKVHeads,
                headDim,
                attendLen,
                cacheSize,
                hasSinks);
        }

        /// <summary>
        /// Low-level GQA decode attention that accepts raw device pointers +
        /// a <see cref="CudaAllocator"/>. Skips CudaStorage extraction for backends
        /// that use GgmlStorage with CUDA UVA (e.g. GgmlCuda).
        /// </summary>
        public static bool TryGqaDecodeAttentionRaw(
            System.IntPtr query, System.IntPtr key, System.IntPtr value,
            System.IntPtr sinks, System.IntPtr result,
            int numQHeads, int numKVHeads, int headDim,
            int attendStart, int attendLen, int cacheSize,
            bool circular, float scale, bool isHalf,
            CudaAllocator allocator)
        {
            if (allocator == null) return false;
            var kernels = allocator.Kernels;
            if (kernels == null) return false;

            // Save the prior CUDA context so GGML's own context (when the calling
            // backend is GgmlCuda) is not replaced on the thread TLS. On the native
            // Cuda backend the call stack already happens to be inside the allocator's
            // context, so cuCtxSetCurrent(prevCtx) is a no-op in that case.
            Interop.CudaDriverApi.cuCtxGetCurrent(out System.IntPtr prevCtx);
            allocator.Context.MakeCurrent();
            int hasSinks = sinks != System.IntPtr.Zero ? 1 : 0;

            if (CudaKernelOps.TryLaunchPartitionedGqaDecodeAttention(
                    kernels, allocator,
                    query, key, value, sinks, result,
                    numQHeads, numKVHeads, headDim,
                    attendStart, attendLen, cacheSize,
                    circular, scale, hasSinks, isHalf))
            {
                Interop.CudaDriverApi.cuStreamSynchronize(allocator.Stream.Handle);
                Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
                return true;
            }

            if (attendLen > CudaKernelOps.DecodeAttentionSingleBlockMaxTokens)
            {
                Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
                return false;
            }

            if (isHalf)
                kernels.LaunchGqaDecodeAttentionSinksF16(
                    query, key, value, sinks, result,
                    numQHeads, numKVHeads, headDim,
                    attendStart, attendLen, cacheSize,
                    circular ? 1 : 0, scale, hasSinks,
                    allocator.Stream.Handle);
            else
                kernels.LaunchGqaDecodeAttentionSinksF32(
                    query, key, value, sinks, result,
                    numQHeads, numKVHeads, headDim,
                    attendStart, attendLen, cacheSize,
                    circular ? 1 : 0, scale, hasSinks,
                    allocator.Stream.Handle);

            // Synchronise the stream so writes are visible to other contexts/streams
            // (e.g. ggml_cuda when the calling backend uses GgmlStorage tensors).
            Interop.CudaDriverApi.cuStreamSynchronize(allocator.Stream.Handle);
            Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
            return true;
        }

        public static bool TryGqaDecodeAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            return CudaKernelOps.TryGqaDecodeAttentionWithSinks(
                result,
                query,
                keyCache,
                valueCache,
                sinks,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular,
                scale);
        }

        public static bool TryGqaPrefillAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int cacheSize,
            int maskStart,
            int windowSize,
            float scale)
        {
            return CudaKernelOps.TryGqaPrefillAttentionWithSinks(
                result,
                query,
                keyCache,
                valueCache,
                sinks,
                numQHeads,
                numKVHeads,
                headDim,
                seqLen,
                kvLen,
                cacheSize,
                maskStart,
                windowSize,
                scale);
        }

        public static bool TryAddBiasRows(Tensor tensor, Tensor bias)
        {
            return CudaKernelOps.TryAddBiasRows(tensor, bias);
        }

        public static bool TryFlatToHeadFirst(Tensor result, Tensor src, int numHeads, int seqLen, int headDim)
        {
            return CudaKernelOps.TryFlatToHeadFirst(result, src, numHeads, seqLen, headDim);
        }

        public static bool TrySplitQkvToHeadFirst(Tensor result, Tensor qkv, int colOffset, int numHeads, int seqLen, int headDim)
        {
            return CudaKernelOps.TrySplitQkvToHeadFirst(result, qkv, colOffset, numHeads, seqLen, headDim);
        }

        public static bool TryCopyHeadFirstToCache(Tensor cache, Tensor src, int startPos, int seqLen, int cacheSize, bool circular)
        {
            return CudaKernelOps.TryCopyHeadFirstToCache(cache, src, startPos, seqLen, cacheSize, circular);
        }

        /// <summary>
        /// Enqueues the kernel that refreshes the two cached RoPE position
        /// tensors from <paramref name="dynParams"/>'s device block. Called
        /// inside a decode-graph capture (see CudaDecodeDynParams).
        /// </summary>
        public static bool TryFillRopePositions(Tensor posQ, Tensor posK, CudaDecodeDynParams dynParams)
        {
            if (dynParams == null || !dynParams.IsValid)
                return false;
            return CudaKernelOps.TryFillRopePositions(posQ, posK, dynParams.DevicePtr);
        }

        /// <summary>
        /// Flags a tensor's device copy as authoritative (host mirror stale).
        /// Needed after a CUDA-graph replay rewrote it without going through
        /// the C# launchers; a later host read would otherwise trust a clean
        /// host mirror and return stale data.
        /// </summary>
        public static bool TryMarkDeviceModified(Tensor tensor)
        {
            if (tensor?.Storage is not CudaStorage storage)
                return false;
            storage.MarkDeviceModified();
            return true;
        }

        public static bool TryGatherCircularHeadFirst(Tensor result, Tensor cache, int startPos, int seqLen, int cacheSize)
        {
            return CudaKernelOps.TryGatherCircularHeadFirst(result, cache, startPos, seqLen, cacheSize);
        }

        public static bool TryExpandKvHeads(Tensor result, Tensor cache, int groupSize, int seqLen)
        {
            return CudaKernelOps.TryExpandKvHeads(result, cache, groupSize, seqLen);
        }

        public static bool TryConcatHeadFirst(Tensor result, Tensor a, Tensor b)
        {
            return CudaKernelOps.TryConcatHeadFirst(result, a, b);
        }

        public static bool TryNeoXRoPEHeadFirst(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            return CudaKernelOps.TryNeoXRoPEHeadFirst(data, cosTable, sinTable, numHeads, seqLen, headDim, ropeHalf);
        }

        public static bool TryNeoXRoPEFlatInPlace(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            return CudaKernelOps.TryNeoXRoPEFlatInPlace(data, cosTable, sinTable, numHeads, seqLen, headDim, ropeHalf);
        }

        /// <summary>
        /// Builds stable local/global single-token NeoX RoPE tables using the
        /// live position from <see cref="CudaDecodeDynParams"/>.
        /// </summary>
        public static bool TryFillNeoXRopeTablesDynamic(
            Tensor localCos,
            Tensor localSin,
            Tensor localFrequencies,
            Tensor globalCos,
            Tensor globalSin,
            Tensor globalFrequencies)
        {
            return CudaKernelOps.TryFillNeoXRopeTablesDynamic(
                localCos, localSin, localFrequencies,
                globalCos, globalSin, globalFrequencies);
        }

        /// <summary>
        /// Fused QK-RMSNorm + NeoX RoPE: normalizes each head-row via RMSNorm and
        /// applies NeoX rotary position embeddings in a single kernel pass.
        /// Requires CudaStorage-backed contiguous F32 tensors and an int32 positions tensor.
        /// </summary>
        public static bool TryQKNormRopeNeox(
            Tensor data,
            Tensor alpha,
            Tensor positions,
            int rows,
            int cols,
            int ropeHalf,
            float eps,
            float ropeBase,
            float ropeFreqScale)
        {
            return CudaKernelOps.TryQKNormRopeNeox(data, alpha, positions, rows, cols, ropeHalf, eps, ropeBase, ropeFreqScale);
        }

        /// <summary>
        /// Fused GDN: reads directly from raw projection buffers (qkv, z, beta, alpha)
        /// instead of a pre-packed buffer.  Eliminates the pack kernel + intermediate buffer.
        /// </summary>
        public static bool TryQwen35GdnFused(
            Tensor result,
            Tensor qkv, Tensor z, Tensor beta, Tensor alpha,
            Tensor convState, Tensor ssmState, Tensor convWeight,
            Tensor dtBias, Tensor aLog, Tensor ssmNorm,
            int seqLen, int qkvDim, int zDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int convWriteIdx, float eps)
        {
            return CudaKernelOps.TryQwen35GdnFused(
                result, qkv, z, beta, alpha,
                convState, ssmState, convWeight,
                dtBias, aLog, ssmNorm,
                seqLen, qkvDim, zDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps);
        }

        // residual += rms_norm(input, alpha) (Gemma post-norm), fused into one kernel.
        public static bool TryRmsNormResidualAdd(Tensor residual, Tensor input, Tensor alpha, float eps)
        {
            return CudaKernelOps.TryRmsNormResidualAdd(residual, input, alpha, eps);
        }

        public static bool TryGELUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            return CudaKernelOps.TryGELUMulSplit(result, gateUp, halfDim);
        }

        private static IntPtr DeviceBufferOf(Tensor t)
        {
            return t?.Storage is CudaStorage cs
                ? cs.DevicePtrAtElement(t.StorageOffset)
                : IntPtr.Zero;
        }

        private static bool IsCudaVector(
            Tensor tensor, DType elementType, long length,
            CudaAllocator allocator, out CudaStorage storage)
        {
            storage = tensor?.Storage as CudaStorage;
            return storage != null
                && (allocator == null || ReferenceEquals(storage.AllocatorImpl, allocator))
                && tensor.ElementType == elementType
                && tensor.DimensionCount == 1
                && tensor.Sizes[0] == length
                && tensor.IsContiguous();
        }

        private static bool IsCudaMatrix(
            Tensor tensor, DType elementType, long rows, long columns,
            CudaAllocator allocator, out CudaStorage storage)
        {
            storage = tensor?.Storage as CudaStorage;
            return storage != null
                && (allocator == null || ReferenceEquals(storage.AllocatorImpl, allocator))
                && tensor.ElementType == elementType
                && tensor.DimensionCount == 2
                && tensor.Sizes[0] == rows
                && tensor.Sizes[1] == columns
                && tensor.IsContiguous();
        }

        /// <summary>Bytes per q8_1 activation block (ts_block_q8_1): a 32-value
        /// block = two fp16 (scale + d*sum) + 32 int8. Callers sizing q8_1 scratch
        /// for the dp4a paths use this.</summary>
        public const int Q81BlockBytes = CudaKernels.Q81BlockBytes;

        /// <summary>Maximum top-k width supported by the fixed-size local arrays
        /// in the native on-device MoE router kernels.</summary>
        public const int MaxMoEExpertsUsed = 32;

        internal static bool IsMoERouterConfigurationSupported(int numExperts, int nUsed)
            => numExperts > 0 && nUsed > 0
               && nUsed <= numExperts && nUsed <= MaxMoEExpertsUsed;

        internal static bool IsMoeDp4aDimensionSupported(int type, int inDim)
        {
            if (inDim <= 0)
                return false;
            return type switch
            {
                2 => (inDim & 31) == 0,                       // Q4_0 block
                12 or 16 or 22 => (inDim & 255) == 0,        // Q4_K / IQ2_XXS / IQ2_S
                _ => false,
            };
        }

        private static bool MoeStageDp4aOk(bool requested, int type, int inDim, Tensor scratch)
            => requested
               && IsMoeDp4aDimensionSupported(type, inDim)
               && scratch?.Storage is CudaStorage;

        /// <summary>Device pointer of a CUDA-resident tensor, flushing any pending
        /// host upload first. Returns <see cref="IntPtr.Zero"/> for a non-CUDA
        /// tensor. Lets callers in other assemblies (e.g. the models) hand a
        /// device-resident buffer (a per-expert scale vector) to a raw kernel.</summary>
        public static IntPtr GetDeviceResidentPtr(Tensor t)
        {
            if (t?.Storage is CudaStorage cs)
            {
                cs.EnsureDeviceCurrent();
                return cs.DevicePtrAtElement(t.StorageOffset);
            }
            return IntPtr.Zero;
        }

        /// <summary>
        /// Fully on-device Gemma 4 MoE expert FFN for decode (seqLen == 1): router
        /// top-k + softmax, per-expert gate/up matvec, GEGLU, per-expert down
        /// matvec and routing-weighted accumulation — all as device kernels with
        /// NO host readback, so a decode layer loop using it is CUDA-graph
        /// capturable. The per-expert quantized weights are addressed through the
        /// device pointer tables (<paramref name="gateUpPtrTable"/> /
        /// <paramref name="downPtrTable"/>) indexed by the on-device expert id, so
        /// one captured graph replays for every token regardless of routing.
        /// </summary>
        public static bool TryMoEExpertFFNDecode(
            Tensor logits,            // [1, numExperts] F32 router logits (device)
            Tensor moeInput,          // [1, hiddenDim] F32 RMSNorm'd MoE input (device)
            Tensor output,            // [1, hiddenDim] F32 result (written)
            Tensor selectedExperts,   // [nUsed] Int32 scratch
            Tensor routingWeights,    // [nUsed] F32 scratch
            Tensor gateUpOut,         // [nUsed, 2*nFf] F32 scratch
            Tensor hAll,              // [nUsed, nFf] F32 scratch
            IntPtr perExpertScalePtr, // device [numExperts] F32 or IntPtr.Zero
            IntPtr gateUpPtrTable,    // device [numExperts] u64
            IntPtr downPtrTable,      // device [numExperts] u64
            int quantType, int numExperts, int nUsed, int hiddenDim, int nFf,
            // Q4_K dp4a fast path (see the ts_moe_expert_*_q4k_dp4a kernels): q8_1
            // scratch for the MoE input (moeInputQ8, hiddenDim/32 blocks) and the
            // GEGLU output (hAllQ8, nUsed * nFf/32 blocks). Both null -> the generic
            // scalar-dequant kernels run instead (any quant type).
            Tensor moeInputQ8 = null,
            Tensor hAllQ8 = null,
            bool useDp4a = false)
        {
            // The native router uses fixed-size local arrays (MAX_K == 32).
            // Reject invalid configurations here, before any kernel is enqueued:
            // otherwise nUsed > 32 leaves downstream route slots uninitialized,
            // while nUsed == 0 produces an invalid zero-height expert launch.
            if (!IsMoERouterConfigurationSupported(numExperts, nUsed)
                || hiddenDim <= 0 || nFf <= 0 || nFf > int.MaxValue / 2
                || !CudaQuantizedOps.SupportsQuantizedType(quantType))
            {
                return false;
            }

            if (!IsCudaMatrix(output, DType.Float32, 1, hiddenDim, null, out CudaStorage outStorage))
            {
                return false;
            }
            CudaAllocator allocator = outStorage.AllocatorImpl;
            int twoNff = nFf * 2;
            if (!IsCudaMatrix(logits, DType.Float32, 1, numExperts, allocator, out CudaStorage logitsStorage)
                || !IsCudaMatrix(moeInput, DType.Float32, 1, hiddenDim, allocator, out CudaStorage moeInStorage)
                || !IsCudaVector(selectedExperts, DType.Int32, nUsed, allocator, out _)
                || !IsCudaVector(routingWeights, DType.Float32, nUsed, allocator, out _)
                || !IsCudaMatrix(gateUpOut, DType.Float32, nUsed, twoNff, allocator, out _)
                || !IsCudaMatrix(hAll, DType.Float32, nUsed, nFf, allocator, out _))
            {
                return false;
            }
            long moeInputQ8Bytes = (long)(hiddenDim / 32) * Q81BlockBytes;
            long hAllQ8Bytes = (long)nUsed * (nFf / 32) * Q81BlockBytes;
            if (moeInputQ8 != null
                && !IsCudaVector(moeInputQ8, DType.UInt8, moeInputQ8Bytes, allocator, out _))
                return false;
            if (hAllQ8 != null
                && !IsCudaVector(hAllQ8, DType.UInt8, hAllQ8Bytes, allocator, out _))
                return false;
            if (gateUpPtrTable == IntPtr.Zero || downPtrTable == IntPtr.Zero)
                return false;

            var kernels = allocator?.Kernels;
            if (kernels == null)
                return false;

            IntPtr logitsPtr = DeviceBufferOf(logits);
            IntPtr moeInPtr = DeviceBufferOf(moeInput);
            IntPtr outPtr = DeviceBufferOf(output);
            IntPtr selPtr = DeviceBufferOf(selectedExperts);
            IntPtr rwPtr = DeviceBufferOf(routingWeights);
            IntPtr guPtr = DeviceBufferOf(gateUpOut);
            IntPtr hPtr = DeviceBufferOf(hAll);

            // Router logits / MoE input are device-produced upstream; ensure any
            // pending upload is flushed (no-op when already device-current, so it
            // stays capture-safe).
            logitsStorage.EnsureDeviceCurrent();
            moeInStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            IntPtr stream = allocator.Stream.Handle;

            kernels.LaunchMoERouterF32(logitsPtr, perExpertScalePtr, selPtr, rwPtr, numExperts, nUsed, stream);

            // Q4_0 / Q4_K experts: dp4a over q8_1-quantized activations (~2x the
            // scalar qvalue_at path; the expert FFN is the bulk of MoE decode).
            // Both dims are a multiple of 32 (block size) for these quants; falls
            // back to the generic kernels for any other quant type or missing
            // scratch.
            bool dp4a = useDp4a && (quantType == 2 || quantType == 12)
                && moeInputQ8?.Storage is CudaStorage && hAllQ8?.Storage is CudaStorage
                && (hiddenDim & 31) == 0 && (nFf & 31) == 0;

            if (dp4a)
            {
                IntPtr moeInQ8 = DeviceBufferOf(moeInputQ8);
                IntPtr hQ8 = DeviceBufferOf(hAllQ8);
                kernels.LaunchQuantizeQ81Rows(moeInPtr, moeInQ8, hiddenDim, 1, stream, warpCooperative: true);
                kernels.LaunchMoEExpertGateUpDp4a(gateUpPtrTable, selPtr, moeInQ8, guPtr, quantType, hiddenDim, twoNff, nUsed, stream);
                kernels.LaunchGELUMulSplitF32(guPtr, hPtr, nUsed, nFf, stream);
                kernels.LaunchQuantizeQ81Rows(hPtr, hQ8, nFf, nUsed, stream, warpCooperative: true);
                kernels.LaunchMoEExpertDownDp4a(downPtrTable, selPtr, rwPtr, hQ8, outPtr, quantType, nFf, hiddenDim, nUsed, stream);
            }
            else
            {
                kernels.LaunchMoEExpertGateUpVecF32(gateUpPtrTable, selPtr, moeInPtr, guPtr, quantType, hiddenDim, twoNff, nUsed, stream);
                kernels.LaunchGELUMulSplitF32(guPtr, hPtr, nUsed, nFf, stream);
                kernels.LaunchMoEExpertDownAccumF32(downPtrTable, selPtr, rwPtr, hPtr, outPtr, quantType, nFf, hiddenDim, nUsed, stream);
            }

            outStorage.MarkDeviceModified();
            ((CudaStorage)selectedExperts.Storage).MarkDeviceModified();
            ((CudaStorage)routingWeights.Storage).MarkDeviceModified();
            ((CudaStorage)gateUpOut.Storage).MarkDeviceModified();
            ((CudaStorage)hAll.Storage).MarkDeviceModified();
            if (dp4a)
            {
                ((CudaStorage)moeInputQ8.Storage).MarkDeviceModified();
                ((CudaStorage)hAllQ8.Storage).MarkDeviceModified();
            }
            return true;
        }

        /// <summary>
        /// On-device SwiGLU MoE expert FFN for direct-CUDA decode (Qwen3.5/3.6 MoE).
        /// Router top-k + softmax-over-selected, separate gate/up expert projections,
        /// SwiGLU (silu(gate)*up), down + routing-weighted accumulate, and an optional
        /// gated shared expert — all device kernels with no host readback, so the
        /// enclosing decode layer loop stays CUDA-graph capturable. Numerically mirrors
        /// SelectTopKRouteWeights (normalized top-k softmax) + RunMoEExpertsReused
        /// (SiLUMul) + the shared-expert SigmoidScalar(VecDot) gate.
        ///
        /// Unlike <see cref="TryMoEExpertFFNDecode"/> (Gemma, fused gate_up + GEGLU),
        /// gate and up are separate weights/tables here. The gate/up and down
        /// projections independently use q8_1+dp4a (Q4_0=2 / Q4_K=12 /
        /// IQ2_XXS=16 / IQ2_S=22) when requested; otherwise they use the generic
        /// scalar-dequant kernels. <paramref name="sharedDown"/> is the
        /// already-computed shared-expert down output ([1, hiddenDim]) or null.
        /// </summary>
        public static bool TryMoEExpertFFNDecodeSwiGLU(
            Tensor logits, Tensor moeInput, Tensor output,
            Tensor selectedExperts, Tensor routingWeights, Tensor gateOut, Tensor upOut,
            IntPtr perExpertScalePtr,
            IntPtr gatePtrTable, IntPtr upPtrTable, IntPtr downPtrTable,
            int gateUpType, int downType, int numExperts, int nUsed, int hiddenDim, int nFf,
            Tensor sharedDown, IntPtr sharedGateVecPtr,
            Tensor moeInputQ8 = null, Tensor gateOutQ8 = null,
            bool useGateUpDp4a = false, bool useDownDp4a = false)
        {
            if (!IsMoERouterConfigurationSupported(numExperts, nUsed)
                || hiddenDim <= 0 || nFf <= 0
                || !CudaQuantizedOps.SupportsQuantizedType(gateUpType)
                || !CudaQuantizedOps.SupportsQuantizedType(downType))
            {
                return false;
            }

            if (!IsCudaMatrix(output, DType.Float32, 1, hiddenDim, null, out CudaStorage outStorage))
            {
                return false;
            }
            CudaAllocator allocator = outStorage.AllocatorImpl;
            if (!IsCudaMatrix(logits, DType.Float32, 1, numExperts, allocator, out CudaStorage logitsStorage)
                || !IsCudaMatrix(moeInput, DType.Float32, 1, hiddenDim, allocator, out CudaStorage moeInStorage)
                || !IsCudaVector(selectedExperts, DType.Int32, nUsed, allocator, out _)
                || !IsCudaVector(routingWeights, DType.Float32, nUsed, allocator, out _)
                || !IsCudaMatrix(gateOut, DType.Float32, nUsed, nFf, allocator, out _)
                || !IsCudaMatrix(upOut, DType.Float32, nUsed, nFf, allocator, out _))
            {
                return false;
            }
            if (sharedDown != null
                && !IsCudaMatrix(sharedDown, DType.Float32, 1, hiddenDim, allocator, out _))
                return false;
            long moeInputQ8Bytes = (long)(hiddenDim / 32) * Q81BlockBytes;
            long gateOutQ8Bytes = (long)nUsed * (nFf / 32) * Q81BlockBytes;
            if (moeInputQ8 != null
                && !IsCudaVector(moeInputQ8, DType.UInt8, moeInputQ8Bytes, allocator, out _))
                return false;
            if (gateOutQ8 != null
                && !IsCudaVector(gateOutQ8, DType.UInt8, gateOutQ8Bytes, allocator, out _))
                return false;
            if (gatePtrTable == IntPtr.Zero || upPtrTable == IntPtr.Zero || downPtrTable == IntPtr.Zero)
                return false;

            var kernels = allocator?.Kernels;
            if (kernels == null)
                return false;

            IntPtr logitsPtr = DeviceBufferOf(logits);
            IntPtr moeInPtr = DeviceBufferOf(moeInput);
            IntPtr outPtr = DeviceBufferOf(output);
            IntPtr selPtr = DeviceBufferOf(selectedExperts);
            IntPtr rwPtr = DeviceBufferOf(routingWeights);
            IntPtr gPtr = DeviceBufferOf(gateOut);
            IntPtr uPtr = DeviceBufferOf(upOut);

            logitsStorage.EnsureDeviceCurrent();
            moeInStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            IntPtr stream = allocator.Stream.Handle;

            kernels.LaunchMoERouterF32(logitsPtr, perExpertScalePtr, selPtr, rwPtr, numExperts, nUsed, stream);

            // Each projection dp4a's over q8_1-quantized activations when its quant
            // type has a MoE dp4a dot
            // (Q4_0=2 / Q4_K=12 / IQ2_XXS=16 / IQ2_S=22) and the q8_1
            // scratch is present; else the generic scalar-dequant kernel runs (any
            // qvalue_at type, e.g. IQ3_S down experts in UD dynamic quants).
            // gate/up in_dim is hidden, down in_dim is n_ff; both are multiples of 32
            // (256 additionally required for the K/IQ 256-value super-blocks).
            bool gateUpDp4a = MoeStageDp4aOk(useGateUpDp4a, gateUpType, hiddenDim, moeInputQ8);
            bool downDp4a = MoeStageDp4aOk(useDownDp4a, downType, nFf, gateOutQ8);

            if (gateUpDp4a)
            {
                IntPtr moeInQ8 = DeviceBufferOf(moeInputQ8);
                kernels.LaunchQuantizeQ81Rows(moeInPtr, moeInQ8, hiddenDim, 1, stream, warpCooperative: true);
                kernels.LaunchMoEExpertGateUpDp4a(gatePtrTable, selPtr, moeInQ8, gPtr, gateUpType, hiddenDim, nFf, nUsed, stream);
                kernels.LaunchMoEExpertGateUpDp4a(upPtrTable, selPtr, moeInQ8, uPtr, gateUpType, hiddenDim, nFf, nUsed, stream);
            }
            else
            {
                kernels.LaunchMoEExpertGateUpVecF32(gatePtrTable, selPtr, moeInPtr, gPtr, gateUpType, hiddenDim, nFf, nUsed, stream);
                kernels.LaunchMoEExpertGateUpVecF32(upPtrTable, selPtr, moeInPtr, uPtr, gateUpType, hiddenDim, nFf, nUsed, stream);
            }

            kernels.LaunchSiluMulF32(gPtr, gPtr, uPtr, (long)nUsed * nFf, stream);

            if (downDp4a)
            {
                IntPtr hQ8 = DeviceBufferOf(gateOutQ8);
                kernels.LaunchQuantizeQ81Rows(gPtr, hQ8, nFf, nUsed, stream, warpCooperative: true);
                kernels.LaunchMoEExpertDownDp4a(downPtrTable, selPtr, rwPtr, hQ8, outPtr, downType, nFf, hiddenDim, nUsed, stream);
            }
            else
            {
                kernels.LaunchMoEExpertDownAccumF32(downPtrTable, selPtr, rwPtr, gPtr, outPtr, downType, nFf, hiddenDim, nUsed, stream);
            }

            // Shared expert: output[j] += sigmoid(moeInput . gateVec) * sharedDown[j].
            if (sharedDown?.Storage is CudaStorage sharedStorage)
            {
                sharedStorage.EnsureDeviceCurrent();
                IntPtr sdPtr = DeviceBufferOf(sharedDown);
                IntPtr gateInput = sharedGateVecPtr != IntPtr.Zero ? moeInPtr : IntPtr.Zero;
                kernels.LaunchMoESharedGatedAdd(outPtr, sdPtr, gateInput, sharedGateVecPtr, hiddenDim, hiddenDim, stream);
            }

            outStorage.MarkDeviceModified();
            ((CudaStorage)selectedExperts.Storage).MarkDeviceModified();
            ((CudaStorage)routingWeights.Storage).MarkDeviceModified();
            ((CudaStorage)gateOut.Storage).MarkDeviceModified();
            ((CudaStorage)upOut.Storage).MarkDeviceModified();
            if (gateUpDp4a)
                ((CudaStorage)moeInputQ8.Storage).MarkDeviceModified();
            if (downDp4a)
                ((CudaStorage)gateOutQ8.Storage).MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// Batched (multi-token) on-device SwiGLU MoE for PREFILL (seqLen == numTokens
        /// > 1). Same math and correctness as <see cref="TryMoEExpertFFNDecodeSwiGLU"/>,
        /// with a token dimension added to every kernel so the whole prefill MoE runs on
        /// device with no host gather/scatter/routing round-trip (which dominated the
        /// old host prefill path and blocked the prefill CUDA-graph capture). Scratch
        /// tensors are [numTokens, ...]; <paramref name="sharedDown"/> is [numTokens,
        /// hiddenDim] or null. Weights are re-read per (token, expert) — no cross-token
        /// batched GEMM reuse yet.
        /// </summary>
        public static bool TryMoEExpertFFNPrefillSwiGLU(
            Tensor logits, Tensor moeInput, Tensor output,
            Tensor selectedExperts, Tensor routingWeights, Tensor gateOut, Tensor upOut,
            IntPtr perExpertScalePtr,
            IntPtr gatePtrTable, IntPtr upPtrTable, IntPtr downPtrTable,
            int gateUpType, int downType, int numExperts, int nUsed, int hiddenDim, int nFf, int numTokens,
            Tensor sharedDown, IntPtr sharedGateVecPtr,
            Tensor moeInputQ8 = null, Tensor gateOutQ8 = null,
            bool useGateUpDp4a = false, bool useDownDp4a = false)
        {
            if (!IsMoERouterConfigurationSupported(numExperts, nUsed)
                || hiddenDim <= 0 || nFf <= 0
                || numTokens <= 0 || numTokens > 65535
                || !CudaQuantizedOps.SupportsQuantizedType(gateUpType)
                || !CudaQuantizedOps.SupportsQuantizedType(downType))
            {
                return false;
            }

            if (!IsCudaMatrix(output, DType.Float32, numTokens, hiddenDim, null, out CudaStorage outStorage))
            {
                return false;
            }
            CudaAllocator allocator = outStorage.AllocatorImpl;
            long routedRows = (long)numTokens * nUsed;
            if (!IsCudaMatrix(logits, DType.Float32, numTokens, numExperts, allocator, out CudaStorage logitsStorage)
                || !IsCudaMatrix(moeInput, DType.Float32, numTokens, hiddenDim, allocator, out CudaStorage moeInStorage)
                || !IsCudaVector(selectedExperts, DType.Int32, routedRows, allocator, out _)
                || !IsCudaVector(routingWeights, DType.Float32, routedRows, allocator, out _)
                || !IsCudaMatrix(gateOut, DType.Float32, routedRows, nFf, allocator, out _)
                || !IsCudaMatrix(upOut, DType.Float32, routedRows, nFf, allocator, out _))
            {
                return false;
            }
            if (sharedDown != null
                && !IsCudaMatrix(sharedDown, DType.Float32, numTokens, hiddenDim, allocator, out _))
                return false;
            long moeInputQ8Bytes = (long)numTokens * (hiddenDim / 32) * Q81BlockBytes;
            long gateOutQ8Bytes = routedRows * (nFf / 32) * Q81BlockBytes;
            if (moeInputQ8 != null
                && !IsCudaVector(moeInputQ8, DType.UInt8, moeInputQ8Bytes, allocator, out _))
                return false;
            if (gateOutQ8 != null
                && !IsCudaVector(gateOutQ8, DType.UInt8, gateOutQ8Bytes, allocator, out _))
                return false;
            if (gatePtrTable == IntPtr.Zero || upPtrTable == IntPtr.Zero || downPtrTable == IntPtr.Zero)
                return false;

            var kernels = allocator?.Kernels;
            if (kernels == null)
                return false;

            IntPtr logitsPtr = DeviceBufferOf(logits);
            IntPtr moeInPtr = DeviceBufferOf(moeInput);
            IntPtr outPtr = DeviceBufferOf(output);
            IntPtr selPtr = DeviceBufferOf(selectedExperts);
            IntPtr rwPtr = DeviceBufferOf(routingWeights);
            IntPtr gPtr = DeviceBufferOf(gateOut);
            IntPtr uPtr = DeviceBufferOf(upOut);

            logitsStorage.EnsureDeviceCurrent();
            moeInStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            IntPtr stream = allocator.Stream.Handle;

            kernels.LaunchMoERouterBatched(logitsPtr, perExpertScalePtr, selPtr, rwPtr, numExperts, nUsed, numTokens, stream);

            bool gateUpDp4a = MoeStageDp4aOk(useGateUpDp4a, gateUpType, hiddenDim, moeInputQ8);
            bool downDp4a = MoeStageDp4aOk(useDownDp4a, downType, nFf, gateOutQ8);

            if (gateUpDp4a)
            {
                IntPtr moeInQ8 = DeviceBufferOf(moeInputQ8);
                kernels.LaunchQuantizeQ81Rows(moeInPtr, moeInQ8, hiddenDim, numTokens, stream, warpCooperative: true);
                kernels.LaunchMoEExpertGateUpBatchedDp4a(gatePtrTable, selPtr, moeInQ8, gPtr, gateUpType, hiddenDim, nFf, nUsed, numTokens, stream);
                kernels.LaunchMoEExpertGateUpBatchedDp4a(upPtrTable, selPtr, moeInQ8, uPtr, gateUpType, hiddenDim, nFf, nUsed, numTokens, stream);
            }
            else
            {
                kernels.LaunchMoEExpertGateUpBatchedVec(gatePtrTable, selPtr, moeInPtr, gPtr, gateUpType, hiddenDim, nFf, nUsed, numTokens, stream);
                kernels.LaunchMoEExpertGateUpBatchedVec(upPtrTable, selPtr, moeInPtr, uPtr, gateUpType, hiddenDim, nFf, nUsed, numTokens, stream);
            }

            kernels.LaunchSiluMulF32(gPtr, gPtr, uPtr, (long)numTokens * nUsed * nFf, stream);

            if (downDp4a)
            {
                IntPtr hQ8 = DeviceBufferOf(gateOutQ8);
                kernels.LaunchQuantizeQ81Rows(gPtr, hQ8, nFf, numTokens * nUsed, stream, warpCooperative: true);
                kernels.LaunchMoEExpertDownBatchedDp4a(downPtrTable, selPtr, rwPtr, hQ8, outPtr, downType, nFf, hiddenDim, nUsed, numTokens, stream);
            }
            else
            {
                kernels.LaunchMoEExpertDownBatchedAccum(downPtrTable, selPtr, rwPtr, gPtr, outPtr, downType, nFf, hiddenDim, nUsed, numTokens, stream);
            }

            if (sharedDown?.Storage is CudaStorage sharedStorage)
            {
                sharedStorage.EnsureDeviceCurrent();
                IntPtr sdPtr = DeviceBufferOf(sharedDown);
                IntPtr gateInput = sharedGateVecPtr != IntPtr.Zero ? moeInPtr : IntPtr.Zero;
                kernels.LaunchMoESharedGatedAddBatched(outPtr, sdPtr, gateInput, sharedGateVecPtr, hiddenDim, hiddenDim, numTokens, stream);
            }

            outStorage.MarkDeviceModified();
            ((CudaStorage)selectedExperts.Storage).MarkDeviceModified();
            ((CudaStorage)routingWeights.Storage).MarkDeviceModified();
            ((CudaStorage)gateOut.Storage).MarkDeviceModified();
            ((CudaStorage)upOut.Storage).MarkDeviceModified();
            if (gateUpDp4a)
                ((CudaStorage)moeInputQ8.Storage).MarkDeviceModified();
            if (downDp4a)
                ((CudaStorage)gateOutQ8.Storage).MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// Scatter one expert's grouped rows into a token-major MoE accumulator.
        /// Row indices must be unique within this call; expert groups are issued
        /// serially on the allocator stream.
        /// </summary>
        public static bool TryMoEScatterAddWeightedRows(
            Tensor output, Tensor expertOutput, Tensor rowIndices, Tensor routingWeights)
        {
            if (output?.Storage is not CudaStorage outStorage
                || output.ElementType != DType.Float32
                || output.DimensionCount != 2
                || !output.IsContiguous())
            {
                return false;
            }

            int numTokens = checked((int)output.Sizes[0]);
            int hidden = checked((int)output.Sizes[1]);
            if (numTokens <= 0 || hidden <= 0
                || expertOutput == null || expertOutput.DimensionCount != 2)
            {
                return false;
            }
            int batchSize = checked((int)expertOutput.Sizes[0]);

            CudaAllocator allocator = outStorage.AllocatorImpl;
            if (batchSize <= 0
                || !IsCudaMatrix(expertOutput, DType.Float32, batchSize, hidden, allocator, out CudaStorage expertStorage)
                || !IsCudaVector(rowIndices, DType.Int32, batchSize, allocator, out CudaStorage rowsStorage)
                || !IsCudaVector(routingWeights, DType.Float32, batchSize, allocator, out CudaStorage weightsStorage))
            {
                return false;
            }
            CudaKernels kernels = allocator?.Kernels;
            if (kernels == null)
                return false;

            outStorage.EnsureDeviceCurrent();
            expertStorage.EnsureDeviceCurrent();
            rowsStorage.EnsureDeviceCurrent();
            weightsStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchMoEScatterAddWeightedRows(
                DeviceBufferOf(output), DeviceBufferOf(expertOutput),
                DeviceBufferOf(rowIndices), DeviceBufferOf(routingWeights),
                batchSize, numTokens, hidden, allocator.Stream.Handle);
            outStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>Add a dense shared-expert result to every token on device,
        /// optionally gated by sigmoid(input dot gateVector).</summary>
        public static bool TryMoEAddSharedExpertBatched(
            Tensor output, Tensor sharedDown, Tensor input, Tensor gateVector = null)
        {
            if (output?.Storage is not CudaStorage outStorage
                || output.ElementType != DType.Float32
                || output.DimensionCount != 2
                || !output.IsContiguous())
            {
                return false;
            }

            int numTokens = checked((int)output.Sizes[0]);
            int hidden = checked((int)output.Sizes[1]);
            if (numTokens <= 0 || hidden <= 0)
                return false;

            CudaAllocator allocator = outStorage.AllocatorImpl;
            if (!IsCudaMatrix(sharedDown, DType.Float32, numTokens, hidden, allocator, out CudaStorage sharedStorage)
                || !IsCudaMatrix(input, DType.Float32, numTokens, hidden, allocator, out CudaStorage inputStorage))
            {
                return false;
            }
            CudaStorage gateStorage = null;
            if (gateVector != null)
            {
                if (!IsCudaVector(gateVector, DType.Float32, hidden, allocator, out CudaStorage gs))
                {
                    return false;
                }
                gateStorage = gs;
            }

            CudaKernels kernels = allocator?.Kernels;
            if (kernels == null)
                return false;

            outStorage.EnsureDeviceCurrent();
            sharedStorage.EnsureDeviceCurrent();
            inputStorage.EnsureDeviceCurrent();
            gateStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchMoESharedGatedAddBatched(
                DeviceBufferOf(output), DeviceBufferOf(sharedDown),
                gateStorage != null ? DeviceBufferOf(input) : IntPtr.Zero,
                gateStorage != null ? DeviceBufferOf(gateVector) : IntPtr.Zero,
                hidden, hidden, numTokens, allocator.Stream.Handle);
            outStorage.MarkDeviceModified();
            return true;
        }

        public static bool TrySwiGluOaiSplit(Tensor result, Tensor gateUp, int halfDim, float alpha, float limit)
        {
            return CudaKernelOps.TrySwiGluOaiSplit(result, gateUp, halfDim, alpha, limit);
        }

        public static bool TryQwen35GatedDeltaNetPacked(
            Tensor result,
            Tensor packed,
            Tensor convState,
            Tensor ssmState,
            Tensor convWeight,
            Tensor dtBias,
            Tensor aLog,
            Tensor ssmNorm,
            int seqLen,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int convWriteIdx,
            float eps)
        {
            return CudaKernelOps.TryQwen35GatedDeltaNetPacked(
                result,
                packed,
                convState,
                ssmState,
                convWeight,
                dtBias,
                aLog,
                ssmNorm,
                seqLen,
                packedDim,
                qkvDim,
                qkDim,
                vDim,
                numKHeads,
                numVHeads,
                headKDim,
                headVDim,
                convKernel,
                convWriteIdx,
                eps);
        }

        public static bool TryQwen35GatedDeltaNetPackInputs(
            Tensor packed,
            Tensor qkv,
            Tensor z,
            Tensor beta,
            Tensor alpha,
            int seqLen,
            int qkvDim,
            int zDim,
            int numVHeads,
            int packedDim)
        {
            return CudaKernelOps.TryQwen35GatedDeltaNetPackInputs(
                packed,
                qkv,
                z,
                beta,
                alpha,
                seqLen,
                qkvDim,
                zDim,
                numVHeads,
                packedDim);
        }

        /// <summary>
        /// Low-level version of <see cref="TryQwen35GatedDeltaNetPacked"/> that accepts
        /// raw device/host-pinned pointers for all inputs except
        /// <paramref name="convState"/> (which must be a <see cref="CudaStorage"/>-
        /// backed Tensor) and a <see cref="CudaAllocator"/> for the kernel launch.
        /// Intended for backends whose tensors use host-pinned/device memory
        /// reachable via CUDA UVA (e.g. GgmlCuda).
        /// The caller is responsible for synchronisation between the allocator's
        /// stream and the source backend.
        /// </summary>
        public static bool TryQwen35GatedDeltaNetPackedRaw(
            System.IntPtr result, System.IntPtr packed, System.IntPtr ssmState,
            System.IntPtr convWeight, System.IntPtr dtBias, System.IntPtr aLog, System.IntPtr ssmNorm,
            Tensor convState,
            int seqLen, int packedDim, int qkvDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int convWriteIdx, float eps,
            CudaAllocator allocator)
        {
            if (allocator == null) return false;
            var kernels = allocator.Kernels;
            if (kernels == null) return false;

            var cs = convState?.Storage as CudaStorage;
            if (cs == null) return false;
            System.IntPtr convStatePtr = cs.DeviceBuffer;

            Interop.CudaDriverApi.cuCtxGetCurrent(out System.IntPtr prevCtx);
            allocator.Context.MakeCurrent();
            kernels.LaunchQwen35GatedDeltaNetPackedF32(
                packed, convStatePtr, ssmState, convWeight, dtBias, aLog, ssmNorm, result,
                seqLen, packedDim, qkvDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps,
                allocator.Stream.Handle);
            Interop.CudaDriverApi.cuStreamSynchronize(allocator.Stream.Handle);
            Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
            return true;
        }

        /// <summary>
        /// GDN kernel wrapper for architectures that ship separate QKV / gate /
        /// beta / alpha projections (e.g. DeepSeek-V4-Flash Qwen3.5-9B GGUF, where
        /// `ssm_in_proj.weight` is absent and the four source weights have
        /// *different* ggml quantization types so host-side fusion is impossible).
        ///
        /// Pipeline (all on the sidecar CudaAllocator's stream, single sync):
        ///   1. cuMemAlloc device scratch for qkv / z / beta / alpha / packed.
        ///   2. cuMemcpyHtoDAsync the four F32 projection host buffers to device.
        ///   3. Launch the pack kernel -> writes packed F32 [seqLen, packedDim] on device.
        ///   4. Launch the GDN packed kernel with packed (device) + ssmState /
        ///      convWeight / dtBias / aLog / ssmNorm (CUDA UVA device pointers from
        ///      ggml_cuda) + convState (CudaStorage, device) + resultDevOrUva
        ///      (gated tensor's UVA device pointer ÔÇö kernel writes directly, no DtoH).
        ///   5. cuStreamSynchronize + cuMemFree all scratch.
        ///
        /// The caller is responsible for providing F32 contiguous host buffers for
        /// all projection inputs and for the result. This function never touches
        /// the sidecar's MakeCurrent outside of the cuCtxGetCurrent / SetCurrent
        /// pair, so it is safe from MTP-spec paths where the ggml primary context
        /// must keep ownership of pending ops.
        /// </summary>
        public static bool TryQwen35GatedDeltaNetFromSeparateRaw(
            System.IntPtr qkvHost,    System.IntPtr zHost,    System.IntPtr betaHost,    System.IntPtr alphaHost,
            System.IntPtr ssmState,    System.IntPtr convWeight, System.IntPtr dtBias,
            System.IntPtr aLog,        System.IntPtr ssmNorm,
            System.IntPtr resultDevOrUva,
            Tensor convState,         // must be CudaStorage-backed
            int seqLen,
            int qkvDim, int qkDim, int vDim, int zDim, int numVHeads,
            int numKHeads, int headKDim, int headVDim,
            int packedDim, int convKernel, int convWriteIdx, float eps,
            CudaAllocator allocator)
        {
            if (allocator == null) return false;
            var kernels = allocator.Kernels;
            if (kernels == null) return false;

            var cs = convState?.Storage as CudaStorage;
            if (cs == null) return false;
            System.IntPtr convStatePtr = cs.DeviceBuffer;

            Interop.CudaDriverApi.cuCtxGetCurrent(out System.IntPtr prevCtx);
            allocator.Context.MakeCurrent();
            System.IntPtr stream = allocator.Stream.Handle;

            // Compute byte sizes (all F32).
            long qkvBytes    = (long)qkvDim    * seqLen * 4L;
            long zBytes      = (long)zDim      * seqLen * 4L;
            long betaBytes   = (long)numVHeads * seqLen * 4L;
            long alphaBytes  = (long)numVHeads * seqLen * 4L;
            long packedBytes = (long)packedDim * seqLen * 4L;

            // 1) Allocate device scratch for projections and packed buffer.
            int ec;
            ec = Interop.CudaDriverApi.cuMemAlloc(out System.IntPtr qkvDev,   (System.UIntPtr)qkvBytes);    if (ec != 0) goto fail_ctx;
            ec = Interop.CudaDriverApi.cuMemAlloc(out System.IntPtr zDev,     (System.UIntPtr)zBytes);      if (ec != 0) goto fail_qkv;
            ec = Interop.CudaDriverApi.cuMemAlloc(out System.IntPtr betaDev,  (System.UIntPtr)betaBytes);   if (ec != 0) goto fail_z;
            ec = Interop.CudaDriverApi.cuMemAlloc(out System.IntPtr alphaDev, (System.UIntPtr)alphaBytes);  if (ec != 0) goto fail_beta;
            ec = Interop.CudaDriverApi.cuMemAlloc(out System.IntPtr packedDev,(System.UIntPtr)packedBytes); if (ec != 0) goto fail_alpha;

            // 2) HtoD copies for projections (qkv/z/beta/alpha are intermediate
            //    activations in host-pinned memory that need to be on device for
            //    the pack kernel).
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(qkvDev,   qkvHost,   (System.UIntPtr)qkvBytes,   stream);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(zDev,     zHost,     (System.UIntPtr)zBytes,     stream);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(betaDev,  betaHost,  (System.UIntPtr)betaBytes,  stream);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(alphaDev, alphaHost, (System.UIntPtr)alphaBytes, stream);

            // 3) Pack kernel: writes packed F32 [seqLen, packedDim] on device.
            kernels.LaunchQwen35GatedDeltaNetPackInputsF32(
                qkvDev, zDev, betaDev, alphaDev, packedDev,
                seqLen, qkvDim, zDim, numVHeads, packedDim,
                stream);

            // 4) GDN packed kernel.
            //    - packedDev is device scratch (cuMemAlloc).
            //    - ssmState/convWeight/dtBias/aLog/ssmNorm are CUDA UVA device
            //      pointers (ggml allocates model weights in CUDA device memory).
            //    - resultDevOrUva is the gated tensor's UVA device pointer ÔÇö the
            //      kernel writes directly to it, no DtoH copy needed.
            //    - convState is CudaStorage device memory.
            kernels.LaunchQwen35GatedDeltaNetPackedF32(
                packedDev, convStatePtr, ssmState, convWeight, dtBias, aLog, ssmNorm, resultDevOrUva,
                seqLen, packedDim, qkvDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps,
                stream);

            // 5) Synchronize so result is visible to the caller (ggml forward pass).
            Interop.CudaDriverApi.cuStreamSynchronize(stream);

            // 6) Free device scratch.
            Interop.CudaDriverApi.cuMemFree(packedDev);
            Interop.CudaDriverApi.cuMemFree(alphaDev);
            Interop.CudaDriverApi.cuMemFree(betaDev);
            Interop.CudaDriverApi.cuMemFree(zDev);
            Interop.CudaDriverApi.cuMemFree(qkvDev);

            Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
            return true;

        fail_packed:
            Interop.CudaDriverApi.cuMemFree(packedDev);
        fail_alpha:
            Interop.CudaDriverApi.cuMemFree(alphaDev);
        fail_beta:
            Interop.CudaDriverApi.cuMemFree(betaDev);
        fail_z:
            Interop.CudaDriverApi.cuMemFree(zDev);
        fail_qkv:
            Interop.CudaDriverApi.cuMemFree(qkvDev);
        fail_ctx:
            Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
            return false;
        }

        /// <summary>
        /// GQA decode attention bridge for GgmlStorage-backed tensors on GgmlCuda.
        /// Allocates CudaStorage scratch buffers on <paramref name="allocator"/>,
        /// copies GgmlStorage host data ÔåÆ device via cuMemcpyHtoDAsync, launches the
        /// kernel, and copies the result back device ÔåÆ host via cuMemcpyDtoHAsync.
        /// </summary>
        public static bool TryGqaDecodeAttentionGgmlCuda(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale,
            bool isHalf,
            CudaAllocator allocator)
        {
            if (allocator == null) return false;
            var kernels = allocator.Kernels;
            if (kernels == null) return false;

            // Compute byte sizes
            int qElems = numQHeads * headDim;
            int kvElems = numKVHeads * headDim * cacheSize;
            int qBytes = qElems * 4;
            int kvBytes = kvElems * (isHalf ? 2 : 4);
            int resultBytes = qElems * 4;

            // Get GgmlStorage host pointers
            System.IntPtr qHost = query.Storage.PtrAtElement(0);
            System.IntPtr kHost = keyCache.Storage.PtrAtElement(0);
            System.IntPtr vHost = valueCache.Storage.PtrAtElement(0);
            System.IntPtr rHost = result.Storage.PtrAtElement(0);

            Interop.CudaDriverApi.cuCtxGetCurrent(out System.IntPtr prevCtx);
            allocator.Context.MakeCurrent();
            System.IntPtr stream = allocator.Stream.Handle;

            // Allocate CudaStorage scratch buffers on the sidecar allocator.
            var qDev = new CudaStorage(allocator, DType.Float32, qElems);
            var kDev = new CudaStorage(allocator, isHalf ? DType.Float16 : DType.Float32, kvElems);
            var vDev = new CudaStorage(allocator, isHalf ? DType.Float16 : DType.Float32, kvElems);
            var rDev = new CudaStorage(allocator, DType.Float32, qElems);

            System.IntPtr qPtr = qDev.DeviceBuffer;
            System.IntPtr kPtr = kDev.DeviceBuffer;
            System.IntPtr vPtr = vDev.DeviceBuffer;
            System.IntPtr rPtr = rDev.DeviceBuffer;

            // Async copies host ÔåÆ device
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(qPtr, qHost, (System.UIntPtr)(ulong)qBytes, stream);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(kPtr, kHost, (System.UIntPtr)(ulong)kvBytes, stream);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(vPtr, vHost, (System.UIntPtr)(ulong)kvBytes, stream);

            // Launch kernel (writes to rDev)
            if (CudaKernelOps.TryLaunchPartitionedGqaDecodeAttention(
                    kernels, allocator,
                    qPtr, kPtr, vPtr, System.IntPtr.Zero, rPtr,
                    numQHeads, numKVHeads, headDim,
                    attendStart, attendLen, cacheSize,
                    circular, scale, 0, isHalf))
            {
                Interop.CudaDriverApi.cuMemcpyDtoHAsync(rHost, rPtr, (System.UIntPtr)(ulong)resultBytes, stream);
                Interop.CudaDriverApi.cuStreamSynchronize(stream);
                qDev.Release(); kDev.Release(); vDev.Release(); rDev.Release();
                Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
                return true;
            }

            // Single-block fallback
            if (attendLen > CudaKernelOps.DecodeAttentionSingleBlockMaxTokens)
            {
                qDev.Release(); kDev.Release(); vDev.Release(); rDev.Release();
                Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
                return false;
            }

            if (isHalf)
                kernels.LaunchGqaDecodeAttentionSinksF16(
                    qPtr, kPtr, vPtr, System.IntPtr.Zero, rPtr,
                    numQHeads, numKVHeads, headDim,
                    attendStart, attendLen, cacheSize,
                    circular ? 1 : 0, scale, 0,
                    stream);
            else
                kernels.LaunchGqaDecodeAttentionSinksF32(
                    qPtr, kPtr, vPtr, System.IntPtr.Zero, rPtr,
                    numQHeads, numKVHeads, headDim,
                    attendStart, attendLen, cacheSize,
                    circular ? 1 : 0, scale, 0,
                    stream);

            Interop.CudaDriverApi.cuMemcpyDtoHAsync(rHost, rPtr, (System.UIntPtr)(ulong)resultBytes, stream);
            Interop.CudaDriverApi.cuStreamSynchronize(stream);
            qDev.Release(); kDev.Release(); vDev.Release(); rDev.Release();
            Interop.CudaDriverApi.cuCtxSetCurrent(prevCtx);
            return true;
        }
    }

    /// <summary>
    /// Direct GDN bridge that uses ggml_cuda's existing CUDA context.
    /// Avoids creating a new CudaAllocator/CudaContext which corrupts
    /// ggml_cuda's Runtime API state (thread-local context stack mismatch
    /// between Driver API cuCtxSetCurrent and Runtime API cudaSetDevice).
    ///
    /// Usage:
    ///   1. Call EnsureInitialized() on the SAME thread that ggml_cuda uses
    ///      for its forward pass (the thread where cuCtxGetCurrent returns
    ///      ggml's context).
    ///   2. Call TryRun() from the same thread.
    ///   3. All cuMemAlloc calls happen on ggml's existing context ÔÇö no
    ///      new context is ever created.
    /// </summary>
    public static class GdnDirectBridge
    {
        private static CudaKernels _kernels;
        private static bool _initAttempted;

        public static bool EnsureInitialized()
        {
            if (_initAttempted) return _kernels != null;
            _initAttempted = true;

            CudaFusedOps.EnsureCudaDriverLoaded();

            int ec = Interop.CudaDriverApi.cuCtxGetCurrent(out IntPtr ctx);
            if (ec != 0 || ctx == IntPtr.Zero) return false;

            _kernels = CudaKernels.TryCreate();
            return _kernels != null;
        }

        public static bool IsReady => _kernels != null;

        /// <summary>
        /// Allocate per-layer convState device buffer via cuMemAlloc on ggml's context.
        /// Returns device pointer, or IntPtr.Zero on failure.
        /// </summary>
        public static IntPtr AllocConvState(int convDim, int qkvDim)
        {
            long bytes = (long)convDim * qkvDim * 4L;
            int ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr ptr, (System.UIntPtr)bytes);
            if (ec != 0) return IntPtr.Zero;
            // Zero-fill
            Interop.CudaDriverApi.cuMemsetD8(ptr, 0, (System.UIntPtr)bytes);
            return ptr;
        }

        public static void FreeConvState(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero) Interop.CudaDriverApi.cuMemFree(ptr);
        }

        /// <summary>
        /// Run the GDN packed kernel via ggml's existing CUDA context.
        /// All state/weight pointers (ssmState, convWeight, dtBias, aLog,
        /// ssmNorm) are CUDA UVA device pointers from ggml ÔÇö the kernel
        /// accesses them directly on the GPU. The result (resultDevOrUva)
        /// is also a UVA device pointer (gated tensor's storage).
        /// convStateDevPtr is a cuMemAlloc'd device buffer.
        /// </summary>
        public static bool TryRun(
            System.IntPtr qkvHost, System.IntPtr zHost,
            System.IntPtr betaHost, System.IntPtr alphaHost,
            System.IntPtr ssmState, System.IntPtr convWeight,
            System.IntPtr dtBias, System.IntPtr aLog, System.IntPtr ssmNorm,
            System.IntPtr resultDevOrUva,
            System.IntPtr convStateDevPtr,
            int seqLen,
            int qkvDim, int qkDim, int vDim, int zDim, int numVHeads,
            int numKHeads, int headKDim, int headVDim,
            int packedDim, int convKernel, int convWriteIdx, float eps)
        {
            if (_kernels == null) return false;

            long qkvBytes    = (long)qkvDim    * seqLen * 4L;
            long zBytes      = (long)zDim      * seqLen * 4L;
            long betaBytes   = (long)numVHeads * seqLen * 4L;
            long alphaBytes  = (long)numVHeads * seqLen * 4L;
            long packedBytes = (long)packedDim * seqLen * 4L;

            // Allocate device scratch for projections and packed buffer.
            int ec;
            ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr qkvDev,   (System.UIntPtr)qkvBytes);    if (ec != 0) return false;
            ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr zDev,     (System.UIntPtr)zBytes);      if (ec != 0) { Interop.CudaDriverApi.cuMemFree(qkvDev); return false; }
            ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr betaDev,  (System.UIntPtr)betaBytes);   if (ec != 0) { Interop.CudaDriverApi.cuMemFree(zDev); Interop.CudaDriverApi.cuMemFree(qkvDev); return false; }
            ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr alphaDev, (System.UIntPtr)alphaBytes);  if (ec != 0) { Interop.CudaDriverApi.cuMemFree(betaDev); Interop.CudaDriverApi.cuMemFree(zDev); Interop.CudaDriverApi.cuMemFree(qkvDev); return false; }
            ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr packedDev,(System.UIntPtr)packedBytes); if (ec != 0) { Interop.CudaDriverApi.cuMemFree(alphaDev); Interop.CudaDriverApi.cuMemFree(betaDev); Interop.CudaDriverApi.cuMemFree(zDev); Interop.CudaDriverApi.cuMemFree(qkvDev); return false; }

            // HtoD copies for projections (host-pinned intermediate activations).
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(qkvDev,   qkvHost,   (System.UIntPtr)qkvBytes,   IntPtr.Zero);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(zDev,     zHost,     (System.UIntPtr)zBytes,     IntPtr.Zero);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(betaDev,  betaHost,  (System.UIntPtr)betaBytes,  IntPtr.Zero);
            Interop.CudaDriverApi.cuMemcpyHtoDAsync(alphaDev, alphaHost, (System.UIntPtr)alphaBytes, IntPtr.Zero);

            // Pack kernel: writes packed F32 [seqLen, packedDim] on device.
            _kernels.LaunchQwen35GatedDeltaNetPackInputsF32(
                qkvDev, zDev, betaDev, alphaDev, packedDev,
                seqLen, qkvDim, zDim, numVHeads, packedDim,
                IntPtr.Zero);

            // GDN packed kernel.
            _kernels.LaunchQwen35GatedDeltaNetPackedF32(
                packedDev, convStateDevPtr, ssmState, convWeight, dtBias, aLog, ssmNorm, resultDevOrUva,
                seqLen, packedDim, qkvDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps,
                IntPtr.Zero);

            Interop.CudaDriverApi.cuStreamSynchronize(IntPtr.Zero);

            // Free device scratch.
            Interop.CudaDriverApi.cuMemFree(packedDev);
            Interop.CudaDriverApi.cuMemFree(alphaDev);
            Interop.CudaDriverApi.cuMemFree(betaDev);
            Interop.CudaDriverApi.cuMemFree(zDev);
            Interop.CudaDriverApi.cuMemFree(qkvDev);

            return true;
        }

        /// <summary>
        /// Device-pointer variant of <see cref="TryRun"/>. All projection pointers
        /// (qkv, z, beta, alpha) are CUDA UVA device pointers from the ggml backend
        /// and must already be on the GPU. This method skips the cuMemAlloc +
        /// cuMemcpyHtoDAsync for projections and passes them directly to the pack
        /// kernel, only allocating scratch for the packed output buffer.
        ///
        /// CRITICAL: cuMemcpyHtoDAsync treats its source as host memory. Passing a
        /// UVA device address to cuMemcpyHtoDAsync corrupts the CUDA heap because
        /// it reads from device address space as if it were host. This method avoids
        /// that by never calling cuMemcpyHtoDAsync on device pointers.
        ///
        /// Uses dedicated GDN stream with event-based ggml sync.
        /// </summary>
        public static bool TryRunDevice(
            System.IntPtr qkvDev, System.IntPtr zDev,
            System.IntPtr betaDev, System.IntPtr alphaDev,
            System.IntPtr ssmState, System.IntPtr convWeight,
            System.IntPtr dtBias, System.IntPtr aLog, System.IntPtr ssmNorm,
            System.IntPtr resultDevOrUva,
            System.IntPtr convStateDevPtr,
            int seqLen,
            int qkvDim, int qkDim, int vDim, int zDim, int numVHeads,
            int numKHeads, int headKDim, int headVDim,
            int packedDim, int convKernel, int convWriteIdx, float eps)
        {
            if (_kernels == null) return false;

            long packedBytes = (long)packedDim * seqLen * 4L;

            int ec;
            ec = Interop.CudaDriverApi.cuMemAlloc(out IntPtr packedDev, (System.UIntPtr)packedBytes);
            if (ec != 0) return false;

            _kernels.LaunchQwen35GatedDeltaNetPackInputsF32(
                qkvDev, zDev, betaDev, alphaDev, packedDev,
                seqLen, qkvDim, zDim, numVHeads, packedDim,
                IntPtr.Zero);

            _kernels.LaunchQwen35GatedDeltaNetPackedF32(
                packedDev, convStateDevPtr, ssmState, convWeight, dtBias, aLog, ssmNorm, resultDevOrUva,
                seqLen, packedDim, qkvDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps,
                IntPtr.Zero);

            Interop.CudaDriverApi.cuStreamSynchronize(IntPtr.Zero);
            Interop.CudaDriverApi.cuMemFree(packedDev);

            return true;
        }

        /// <summary>
        /// Device-pointer variant for pre-packed input. The caller already has
        /// a packed F32 buffer [seqLen, packedDim] on the device (UVA). This
        /// method passes it directly to the GDN kernel without an additional
        /// pack step or any cuMemAlloc/cuMemcpyHtoDAsync for projections.
        /// Kernel runs on the NULL stream (default, synchronizing). The NULL
        /// stream serializes all work ÔÇö including ggml's compute stream ÔÇö so
        /// no explicit per-layer sync is needed.
        /// </summary>
        public static bool TryRunPackedDevice(
            System.IntPtr packedDev,
            System.IntPtr convStateDevPtr,
            System.IntPtr ssmState, System.IntPtr convWeight,
            System.IntPtr dtBias, System.IntPtr aLog, System.IntPtr ssmNorm,
            System.IntPtr resultDevOrUva,
            int seqLen, int packedDim, int qkvDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int convWriteIdx, float eps)
        {
            if (_kernels == null) return false;

            _kernels.LaunchQwen35GatedDeltaNetPackedF32(
                packedDev, convStateDevPtr, ssmState, convWeight, dtBias, aLog, ssmNorm, resultDevOrUva,
                seqLen, packedDim, qkvDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps,
                IntPtr.Zero);

            return true;
        }
    }
}
