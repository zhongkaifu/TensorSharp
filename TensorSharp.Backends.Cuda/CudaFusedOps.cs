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

        public static bool TryGatherCircularHeadFirst(Tensor result, Tensor cache, int startPos, int seqLen, int cacheSize)
        {
            return CudaKernelOps.TryGatherCircularHeadFirst(result, cache, startPos, seqLen, cacheSize);
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
