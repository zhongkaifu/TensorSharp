using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public static class CudaQuantizedOps
    {
        /// <summary>
        /// The resident layout of an NVFP4 matrix on a GPU with the FP4 tensor-core kernels (<see cref="CudaFp4Kernels"/>):
        /// codes [rows][k/2] then UE4M3 scales [rows][k/16], the bytes of its ggml blocks repacked once as it becomes
        /// resident (the 16-byte-aligned runs the TMA and vector loads need). Not a ggml type: only this class's matmul and
        /// row lookups read it, and <see cref="TryGetResidentDevicePtr"/> does not hand such a weight out.
        /// </summary>
        internal const int Nvfp4PlanarType = 1040;

        private sealed class DeviceWeight : IDisposable
        {
            public IntPtr DevicePtr;
            public long RawBytes;
            /// <summary>The layout the device bytes are in: the ggml type uploaded, or <see cref="Nvfp4PlanarType"/>.</summary>
            public int GgmlType;
            /// <summary>The ggml type the weight was uploaded as.</summary>
            public int SourceType;
            public long Ne0;
            public long Ne1;
            public int DeviceId;
            /// <summary>When true the bytes live inside a <see cref="WeightArena"/>
            /// slab (bump-suballocated), so this weight must NOT be individually
            /// cuMemFree'd — the arena owns and frees the whole slab.</summary>
            public bool ArenaBacked;
            public WeightArena ArenaOwner;

            public void Dispose()
            {
                if (DevicePtr != IntPtr.Zero)
                {
                    if (ArenaBacked)
                        ArenaOwner?.Release(DevicePtr);
                    else
                        CudaDriverApi.cuMemFree(DevicePtr);
                    DevicePtr = IntPtr.Zero;
                }
            }
        }

        // Model weights are allocated in bulk at load and freed together at
        // unload. Handing each of the (7000+ for a big MoE) weight tensors its
        // own cuMemAlloc wastes ~1.6 GB on a 26B model: every driver allocation
        // is rounded up to the device page granularity, and those roundings add
        // up until the model no longer fits and inference spills into WDDM shared
        // memory. Packing weights into a few large slabs (like ggml's model
        // buffers) removes that rounding waste almost entirely.
        private sealed class WeightArena
        {
            // 256 MB slabs: large enough that per-tensor rounding disappears, small
            // enough that the final slab wastes little tail and never fails to
            // allocate late in the preload when free VRAM is low.
            private const long SlabBytes = 256L << 20;
            private readonly CudaAllocator allocator;
            private sealed class Slab
            {
                public IntPtr Pointer;
                public long Size;
                public int LiveAllocations;
            }
            private readonly List<Slab> slabs = new List<Slab>();
            private readonly Dictionary<IntPtr, Slab> allocations = new Dictionary<IntPtr, Slab>();
            private Slab currentSlab;
            private long currentOffset;

            public WeightArena(CudaAllocator allocator) => this.allocator = allocator;

            /// <summary>Bump-suballocate <paramref name="rawBytes"/> from a slab.
            /// Each block is 256-byte aligned and carries 16 trailing slack bytes
            /// (the cp.async MMQ kernel over-reads up to 8 bytes past the last
            /// block), matching the old per-tensor <c>cuMemAlloc(rawBytes + 16)</c>.</summary>
            public IntPtr Allocate(long rawBytes)
            {
                const long align = 256;
                long need = checked((checked(rawBytes + 16 + align - 1) / align) * align);
                if (currentSlab == null || need > currentSlab.Size - currentOffset)
                {
                    long slabSize = Math.Max(SlabBytes, need);
                    allocator.Context.MakeCurrent();
                    CudaDriverApi.cuMemAlloc(out IntPtr slab, new UIntPtr((ulong)slabSize)).ThrowOnError();
                    currentSlab = new Slab { Pointer = slab, Size = slabSize };
                    slabs.Add(currentSlab);
                    currentOffset = 0;
                }

                IntPtr ptr = new IntPtr(currentSlab.Pointer.ToInt64() + currentOffset);
                currentOffset += need;
                allocations.Add(ptr, currentSlab);
                currentSlab.LiveAllocations++;
                return ptr;
            }

            /// <summary>Reclaim a slab when its last weight is released. A slab
            /// shared with another live model remains resident. Individual freed
            /// slices are not reused, so pending kernels and captured graphs
            /// cannot observe another weight at the same address.</summary>
            public void Release(IntPtr pointer)
            {
                // Allocate, release and FreeAll are called under the cache Sync.
                if (!allocations.Remove(pointer, out Slab slab)) return;
                if (--slab.LiveAllocations != 0) return;
                allocator.Context.MakeCurrent();
                // Driver free waits for work using the allocation before returning.
                CudaDriverApi.cuMemFree(slab.Pointer).ThrowOnError();
                slabs.Remove(slab);
                if (ReferenceEquals(currentSlab, slab))
                {
                    currentSlab = null;
                    currentOffset = 0;
                }
            }

            public void FreeAll()
            {
                allocator.Context.MakeCurrent();
                foreach (Slab slab in slabs)
                {
                    if (slab.Pointer != IntPtr.Zero)
                        CudaDriverApi.cuMemFree(slab.Pointer);
                }
                slabs.Clear();
                allocations.Clear();
                currentSlab = null;
                currentOffset = 0;
            }
        }

        private static readonly object Sync = new object();
        private static readonly Dictionary<CacheKey, DeviceWeight> Cache = new Dictionary<CacheKey, DeviceWeight>();
        private static readonly Dictionary<CudaAllocator, WeightArena> Arenas = new Dictionary<CudaAllocator, WeightArena>();

        private sealed class ScratchAllocation
        {
            public IntPtr Ptr;
            public long Bytes;
            public readonly List<IntPtr> Retired = new List<IntPtr>();
        }

        private sealed class ScratchPool
        {
            public readonly Dictionary<CudaAllocator, ScratchAllocation> Allocations =
                new Dictionary<CudaAllocator, ScratchAllocation>();
        }

        // Scratch is allocator-owned rather than device-owned: different
        // allocators have different streams and may execute concurrently.
        // Growth retains the previous generation because CUDA graphs bake raw
        // scratch pointers into kernel nodes. Geometric growth bounds all
        // retained generations to less than the current allocation for gradual
        // growth; CudaAllocator.Dispose releases them after synchronizing.
        //
        // q8_1-quantized activation scratch for the dp4a Q8_0 matmul path.
        private static readonly ScratchPool Q81Scratch = new ScratchPool();
        private static IntPtr EnsureQ81Scratch(CudaAllocator allocator, long bytes)
            => EnsureScratch(Q81Scratch, allocator, bytes);

        // Large-row (prefill-sized) matmuls: dequantize the weight ONCE to f16 and
        // run a tensor-core cuBLAS GEMM (mirrors ggml_cuda's dequant+cuBLAS route).
        // The block-tile quant kernels re-read the whole weight every rows/tile-rows
        // output tile, so a 2048-token prefill costs hundreds of full weight sweeps;
        // this path costs ~3 sweeps (int8 read + f16 write + GEMM read) regardless
        // of row count. MIN_ROWS/MAX_MB tune the activation-row threshold and the
        // f16 scratch cap (the LM head exceeds it).
        internal static readonly int F16GemmMinRows = EnvInt("TS_CUDA_QMM_F16GEMM_MIN_ROWS", 32);
        internal static readonly long F16GemmMaxWeightBytes = EnvInt("TS_CUDA_QMM_F16GEMM_MAX_MB", 768) * 1024L * 1024L;

        // ggml-style warp-cooperative Q8_0 -> F16 whole-weight conversion for
        // RunF16Gemm. Settable so tests and the matmul benchmark can compare it
        // with the generic one-thread-per-scalar dequantizer.
        public static bool Q80F16DequantEnabled { get; set; } = true;

        private static int EnvInt(string name, int fallback)
        {
            string s = Environment.GetEnvironmentVariable(name);
            return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
        }

        // Per-allocator reusable f16 scratch for the dequant+GEMM path (weight
        // and activation panels).
        private static readonly ScratchPool WF16Scratch = new ScratchPool();
        private static readonly ScratchPool AF16Scratch = new ScratchPool();

        // Split q8_1 activation scratch for the cp.async MMQ path: dense qs byte
        // rows + separate float scales (see ts_quantize_q8_1_split_rows_f32).
        private static readonly ScratchPool Q81SplitQsScratch = new ScratchPool();
        private static readonly ScratchPool Q81SplitDScratch = new ScratchPool();

        // NVFP4 activation planes, row scales, split-K partials and the split-K tile counters (zeroed) of the FP4
        // tensor-core products (see RunNvfp4Planar).
        private static readonly ScratchPool Fp4CodesScratch = new ScratchPool();
        private static readonly ScratchPool Fp4ScalesScratch = new ScratchPool();
        private static readonly ScratchPool Fp4RowScaleScratch = new ScratchPool();
        private static readonly ScratchPool Fp4PartialScratch = new ScratchPool();
        private static readonly ScratchPool Fp4CounterScratch = new ScratchPool();

        private static IntPtr EnsureScratch(ScratchPool pool, CudaAllocator allocator, long bytes)
        {
            lock (Sync)
            {
                if (pool.Allocations.TryGetValue(allocator, out ScratchAllocation scratch) &&
                    scratch.Bytes >= bytes)
                {
                    return scratch.Ptr;
                }

                allocator.Context.MakeCurrent();
                long doubled = scratch == null || scratch.Bytes > long.MaxValue / 2
                    ? 0
                    : scratch.Bytes * 2;
                long alloc = Math.Max(bytes, Math.Max(64 * 1024L, doubled));
                CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)alloc)).ThrowOnError();

                if (scratch == null)
                {
                    scratch = new ScratchAllocation();
                    pool.Allocations.Add(allocator, scratch);
                }
                else if (scratch.Ptr != IntPtr.Zero)
                {
                    scratch.Retired.Add(scratch.Ptr);
                }

                scratch.Ptr = ptr;
                scratch.Bytes = alloc;
                return ptr;
            }
        }

        /// <summary><see cref="EnsureScratch"/> whose bytes are zero when first handed out (the FP4 GEMM's split-K
        /// counters, which the kernels return to zero after each product).</summary>
        private static IntPtr EnsureZeroedScratch(ScratchPool pool, CudaAllocator allocator, long bytes)
        {
            lock (Sync)
            {
                pool.Allocations.TryGetValue(allocator, out ScratchAllocation before);
                IntPtr previous = before?.Ptr ?? IntPtr.Zero;
                IntPtr ptr = EnsureScratch(pool, allocator, bytes);
                if (ptr != previous)
                {
                    long size = pool.Allocations[allocator].Bytes;
                    CudaDriverApi.cuMemsetD8Async(ptr, 0, new UIntPtr((ulong)size), allocator.Stream.Handle).ThrowOnError();
                }
                return ptr;
            }
        }

        internal static void ReleaseScratch(CudaAllocator allocator)
        {
            lock (Sync)
            {
                allocator.Context.MakeCurrent();
                ReleaseScratch(Q81Scratch, allocator);
                ReleaseScratch(WF16Scratch, allocator);
                ReleaseScratch(AF16Scratch, allocator);
                ReleaseScratch(Q81SplitQsScratch, allocator);
                ReleaseScratch(Q81SplitDScratch, allocator);
                ReleaseScratch(Fp4CodesScratch, allocator);
                ReleaseScratch(Fp4ScalesScratch, allocator);
                ReleaseScratch(Fp4RowScaleScratch, allocator);
                ReleaseScratch(Fp4PartialScratch, allocator);
                ReleaseScratch(Fp4CounterScratch, allocator);
            }
        }

        private static void ReleaseScratch(ScratchPool pool, CudaAllocator allocator)
        {
            if (!pool.Allocations.TryGetValue(allocator, out ScratchAllocation scratch))
                return;

            pool.Allocations.Remove(allocator);
            if (scratch.Ptr != IntPtr.Zero)
                CudaDriverApi.cuMemFree(scratch.Ptr);
            foreach (IntPtr ptr in scratch.Retired)
                CudaDriverApi.cuMemFree(ptr);
        }

        /// <summary>
        /// C[rows, outDim] (row-major) = A[rows, inDim] x W[outDim, inDim]^T via
        /// cuBLAS, for operands already in <paramref name="wType"/>/<paramref name="aType"/>
        /// device layout. F32 accumulate, F32 output.
        /// </summary>
        /// <remarks>
        /// F32 operands compute as F32 by name (<c>CUBLAS_COMPUTE_32F_PEDANTIC</c>), not by the handle's
        /// math mode. The F32 weights here are the small dense tensors whose outputs pick things:
        /// every engine's <c>MatMulF32</c> past its GEMV rows (MoE routers, DeepSeek V4's
        /// hyper-connection mixers, Qwen3.8's GDN gates and indexer projections, GLM's indexer head
        /// weights) and F32 <c>QuantizedWeight</c>s. Rounded operands moved DeepSeek V4's routing; the
        /// TensorSharp-owned ggml path pins the same compute type (ggml_ops_matmul_precision.cu). Its
        /// cost against <c>CUBLAS_COMPUTE_32F</c> under the default mode depends on the GPU: 0.90-1.00x on
        /// an RTX 3080 Laptop, 1.05-1.11x on an A40 at the engines' 512-row shapes (under 0.5 ms per
        /// 512-row prefill). 16-bit operands (F16/BF16 weights, and activations this
        /// file converts on purpose) keep <c>CUBLAS_COMPUTE_32F</c>: tensor cores, F32 accumulation.
        /// </remarks>
        private static void RunGemm(
            CudaAllocator allocator, IntPtr weightPtr, int wType, IntPtr aPtr, int aType,
            IntPtr resultPtr, int inDim, int outDim, int rows)
        {
            allocator.Blas.SetStream(allocator.Stream.Handle);
            float alpha = 1.0f, beta = 0.0f;
            int computeType = wType == CublasApi.CUDA_R_32F || aType == CublasApi.CUDA_R_32F
                ? CublasApi.CUBLAS_COMPUTE_32F_PEDANTIC
                : CublasApi.CUBLAS_COMPUTE_32F;
            CublasApi.cublasGemmEx(
                allocator.Blas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                outDim, rows, inDim,
                ref alpha,
                weightPtr, wType, inDim,
                aPtr, aType, inDim,
                ref beta,
                resultPtr, CublasApi.CUDA_R_32F, outDim,
                computeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        /// <summary>
        /// A prompt-size product through an F16 copy of the weight dequantized for the call and the tensor-core GEMM (the
        /// route <see cref="RunResidentMatmul"/> takes for large batches), for a layout the shared route does not list
        /// (Q2_0, when <see cref="CudaKernels.DequantsQ2_0"/>). F32 activations [rows, inDim] to F32 [rows, outDim].
        /// </summary>
        internal static void RunPrefillF16Gemm(CudaAllocator allocator, IntPtr weightPtr, int ggmlType, IntPtr inputPtr,
            IntPtr resultPtr, int inDim, int outDim, int rows)
            => RunF16Gemm(allocator, allocator.Kernels, weightPtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);

        /// <summary>
        /// While open on a thread, the F16 GEMM route keeps each weight it dequantizes in <paramref name="arena"/> (as long as
        /// it fits) and later calls with the same weight on the same allocator reuse that copy instead of dequantizing again:
        /// an engine that runs several ubatches of one layer back to back (Flash Next's prompt groups) dequantizes each of the
        /// layer's weights once. The copy is the same kernel's output, so every product is the same bit for bit. The caller
        /// owns the arena, keeps it alive while the scope is open, and must not let another weight take a cached address
        /// (a scope lives within one layer's calls).
        /// </summary>
        internal static F16WeightScope BeginF16WeightScope(CudaAllocator allocator, IntPtr arena, long capacity)
        {
            var scope = new F16WeightScope(allocator, arena, capacity);
            t_f16Scope = scope;
            return scope;
        }

        [ThreadStatic] private static F16WeightScope t_f16Scope;

        internal sealed class F16WeightScope : IDisposable
        {
            private readonly CudaAllocator _allocator;
            private readonly IntPtr _arena;
            private readonly long _capacity;
            private long _used;
            private readonly Dictionary<IntPtr, IntPtr> _copies = new();

            /// <summary>Weights dequantized into the arena, and calls that reused one.</summary>
            public int Dequantized { get; private set; }
            public int Reused { get; private set; }

            internal F16WeightScope(CudaAllocator allocator, IntPtr arena, long capacity)
            {
                _allocator = allocator;
                _arena = arena;
                _capacity = capacity;
            }

            /// <summary>The cached copy of <paramref name="weight"/> (<paramref name="fresh"/> false), or room for one
            /// (<paramref name="fresh"/> true, the caller dequantizes into it), or false when the arena has no room.</summary>
            internal bool TryTake(CudaAllocator allocator, IntPtr weight, long bytes, out IntPtr f16, out bool fresh)
            {
                fresh = false;
                if (!ReferenceEquals(allocator, _allocator) || _arena == IntPtr.Zero)
                {
                    f16 = IntPtr.Zero;
                    return false;
                }
                if (_copies.TryGetValue(weight, out f16))
                {
                    Reused++;
                    return true;
                }
                long size = (bytes + 255) & ~255L;     // cuBLAS sees the alignment the scratch gives
                if (_used + size > _capacity)
                    return false;
                f16 = (IntPtr)((long)_arena + _used);
                _used += size;
                _copies[weight] = f16;
                fresh = true;
                Dequantized++;
                return true;
            }

            public void Dispose()
            {
                if (ReferenceEquals(t_f16Scope, this))
                    t_f16Scope = null;
            }
        }

        private static void RunF16Gemm(
            CudaAllocator allocator, CudaKernels kernels,
            IntPtr weightPtr, int ggmlType, IntPtr inputPtr, IntPtr resultPtr,
            int inDim, int outDim, int rows)
        {
            long wElems = (long)inDim * outDim;
            long aElems = (long)rows * inDim;
            IntPtr aF16 = EnsureScratch(AF16Scratch, allocator, aElems * 2);
            allocator.Context.MakeCurrent();
            IntPtr stream = allocator.Stream.Handle;
            IntPtr wF16;
            if (ggmlType == 1)
            {
                // F16 weights are already the cuBLAS operand in the same
                // [outDim, inDim] row-major layout the dequant kernels emit --
                // use the resident weight directly, no scratch or dequant pass.
                wF16 = weightPtr;
            }
            else
            {
                F16WeightScope scope = t_f16Scope;
                bool fresh = true;
                if (scope == null || !scope.TryTake(allocator, weightPtr, wElems * 2, out wF16, out fresh))
                    wF16 = EnsureScratch(WF16Scratch, allocator, wElems * 2);
                if (fresh)
                {
                    if (ggmlType == 8 && Q80F16DequantEnabled)
                        kernels.LaunchDequantWeightQ80F16(weightPtr, wF16, wElems, stream);
                    else
                        kernels.LaunchDequantWeightF16(weightPtr, wF16, ggmlType, inDim, wElems, stream);
                }
            }
            kernels.LaunchConvertF32F16(inputPtr, aF16, aElems, stream);

            // C[rows, outDim] (row-major) == C_col[outDim, rows]:
            //   C_col = (W_col[inDim, outDim])^T x A_col[inDim, rows]
            // f16 inputs, f32 accumulate + f32 output (CUBLAS_COMPUTE_32F).
            RunGemm(allocator, wF16, CublasApi.CUDA_R_16F, aF16, CublasApi.CUDA_R_16F,
                resultPtr, inDim, outDim, rows);
        }

        // Q4_0 (the dominant dense quant) uses the int8 dp4a matmul for decode AND
        // verify (~memory-bound, matches ggml's mul_mat_vec_q). Settable so the
        // exact-reference tests can pin the FP32 dequant kernels while a separate
        // test checks the dp4a path against the (looser) int8 tolerance.
        public static bool Q40Dp4aEnabled { get; set; } = true;

        // Q8_0 single-row decode uses the warp-per-column dp4a matvec (quantizes the
        // activation row to q8_1, like ggml's mul_mat_vec_q; measured 26 -> 33 tok/s
        // on Qwen3.5-9B-Q8_0 decode vs the scalar per-byte kernel). Settable for the
        // same exact-vs-int8-tolerance test split as Q40Dp4aEnabled.
        public static bool Q80VecDp4aEnabled { get; set; } = true;

        // Q4_K single-token decode via q8_1-quantized activation + dp4a (ggml
        // mul_mat_vec_q4_K). The generic scalar vec kernel re-parses the Q4_K
        // super-block header per nibble; on a Q4_K-heavy model (every projection
        // in the 26B-A4B is Q4_K) that leaves decode ~2x behind ggml. Settable so
        // the exact-reference tests can pin the exact scalar kernel, while a
        // separate test checks the dp4a path against the looser int8 tolerance.
        public static bool Q4KDp4aEnabled { get; set; } = true;

        // Q5_K / Q6_K single-token decode via ggml-style q8_1 activation dots.
        // The scalar fallback calls qvalue_at for every weight, repeatedly parsing
        // K-quant headers and bitfields. These paths quantize the activation once
        // and use packed dp4a dots. Each flag is independently settable for parity
        // tests and A/B profiling.
        public static bool Q5KDp4aEnabled { get; set; } = true;

        public static bool Q6KDp4aEnabled { get; set; } = true;

        // IQ2_XXS on-device MoE experts via q8_1-quantized activation + the existing
        // dot_iq2_xxs_q8_1 dp4a dot (ggml vec_dot_iq2_xxs_q8_1), replacing the
        // per-element scalar grid-lookup path for the gate/up expert projections of
        // UD-IQ2_XXS MoE models (Qwen3.6-35B-A3B). Settable so tests can compare it
        // with the scalar MoE kernels.
        public static bool Iq2xxsDp4aEnabled { get; set; } = true;

        // IQ2_S on-device MoE experts via q8_1 + the ggml-compatible
        // vec_dot_iq2_s_q8_1 dp4a dot. The Qwen3.6-35B-A3B UD-IQ2_XXS file uses
        // IQ2_XXS for expert gate/up but IQ2_S for expert down, so both formats
        // need a device vec-dot to keep the full MoE decode on the fast path.
        // Settable so tests can compare it with the scalar expert kernel.
        public static bool Iq2sDp4aEnabled { get; set; } = true;

        // Single-row IQ2_XXS/IQ2_S matvec using one global q8_1 scratch row.
        // This quantizes the activation once per projection instead of once per
        // output CTA. Settable so tests can compare it with the per-CTA paths.
        public static bool Iq2VecDp4aEnabled { get; set; } = true;

        /// <summary>
        /// Returns whether the q8_1 + dp4a MoE projection is enabled for the
        /// specified quantized weight type. The gate/up and down stages may use
        /// different quant types, so callers must query each stage independently.
        /// </summary>
        public static bool IsMoeDp4aEnabledForType(int ggmlType)
        {
            return ggmlType switch
            {
                2 => Q40Dp4aEnabled,
                12 => Q4KDp4aEnabled,
                16 => Iq2xxsDp4aEnabled,
                22 => Iq2sDp4aEnabled,
                _ => false,
            };
        }

        // Quantize each 32-value activation block cooperatively with one warp.
        // This mirrors ggml_cuda's q8_1 quantizer; the one-thread-per-block
        // kernel, whose 32 serial loads/stores are poorly coalesced for
        // decode-sized rows, stays as the tests' reference.
        public static bool Q81WarpQuantizeEnabled { get; set; } = true;

        // Diagnostic/tuning gate: only use the q8_1 vec matvec when the output is at
        // least this wide (0 = always). Lets A/B runs isolate which decode
        // projections tolerate the activation's q8_1 round-trip.
        internal static readonly int Q80VecMinOutDim = EnvInt("TS_CUDA_Q80_VEC_MIN_OUT", 0);

        // Direct int8 tensor-core GEMM over raw Q8_0 blocks for prefill-sized rows
        // (mma.m16n8k32, ggml MMQ-style). MAX_ROWS is the crossover where the
        // dequant+cuBLAS F16 route wins (weight sweeps grow as ceil(rows/128) on
        // the MMQ side). Settable so tests and the benchmark can pin either route.
        public static bool Q80MmqEnabled { get; set; } = true;
        internal static readonly int Q80MmqMaxRows = EnvInt("TS_CUDA_Q80_MMQ_MAX_ROWS", 512);

        // cp.async staging variant of the MMQ kernel (split q8_1 scratch, raw
        // weight windows async-copied to shared). Requires inDim % 256 == 0
        // (true for every real model dim); bit-identical results to the base
        // register-prefetch MMQ kernel, which serves the other shapes.
        public static bool Q80Mmq2Enabled { get; set; } = true;

        public static bool AreKernelsAvailable(CudaAllocator allocator)
        {
            return allocator != null && allocator.Kernels != null;
        }

        public static bool SupportsQuantizedType(int ggmlType)
        {
            return ggmlType == 2 ||  // Q4_0
                ggmlType == 3 ||     // Q4_1
                ggmlType == 6 ||     // Q5_0
                ggmlType == 7 ||     // Q5_1
                ggmlType == 8 ||     // Q8_0
                ggmlType == 9 ||     // Q8_1
                ggmlType == 10 ||    // Q2_K
                ggmlType == 11 ||    // Q3_K
                ggmlType == 12 ||    // Q4_K
                ggmlType == 13 ||    // Q5_K
                ggmlType == 14 ||    // Q6_K
                ggmlType == 16 ||    // IQ2_XXS
                ggmlType == 17 ||    // IQ2_XS
                ggmlType == 18 ||    // IQ3_XXS
                ggmlType == 19 ||    // IQ1_S
                ggmlType == 20 ||    // IQ4_NL
                ggmlType == 21 ||    // IQ3_S
                ggmlType == 22 ||    // IQ2_S
                ggmlType == 23 ||    // IQ4_XS
                ggmlType == 29 ||    // IQ1_M
                ggmlType == 39 ||    // MXFP4 (gpt-oss experts)
                ggmlType == 40;      // NVFP4
        }

        /// <summary>
        /// Unquantized weight types whose resident bytes are directly a cuBLAS
        /// operand: F32 (0), F16 (1) and BF16 (30). These skip the dequant pass
        /// entirely (and BF16 must, to stay bit-exact).
        /// </summary>
        public static bool IsDenseGemmType(int ggmlType)
            => ggmlType == 0 || ggmlType == 1 || ggmlType == 30;

        /// <summary>Every weight type <see cref="RunResidentMatmul"/> can serve.</summary>
        public static bool SupportsMatmulType(int ggmlType)
            => IsDenseGemmType(ggmlType) || SupportsQuantizedType(ggmlType);

        public static void PreloadQuantizedWeight(
            CudaAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (allocator == null)
                throw new ArgumentNullException(nameof(allocator));
            EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
        }

        // The LTX file uploader publishes into the same cache only after its
        // synchronous callback has finished every DMA read of its source buffers.
        internal static void PreloadQuantizedWeightFromSource(CudaAllocator allocator,
            IntPtr cacheKey, int ggmlType, long ne0, long ne1, long rawBytes, Action<IntPtr> upload)
        {
            ArgumentNullException.ThrowIfNull(allocator);
            ArgumentNullException.ThrowIfNull(upload);
            EnsureWeight(allocator, cacheKey, IntPtr.Zero, ggmlType, ne0, ne1, rawBytes, upload);
        }

        public static void ReleaseQuantizedWeight(CudaAllocator allocator, IntPtr cacheKey)
        {
            if (allocator == null || cacheKey == IntPtr.Zero)
                return;

            var key = new CacheKey(allocator, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out DeviceWeight entry))
                {
                    allocator.Context.MakeCurrent();
                    entry.Dispose();
                    Cache.Remove(key);
                }
            }
        }

        public static void ClearDeviceCache(CudaAllocator allocator)
        {
            if (allocator == null)
                return;

            lock (Sync)
            {
                allocator.Context.MakeCurrent();
                var remove = new List<CacheKey>();
                foreach (var kv in Cache)
                {
                    if (ReferenceEquals(kv.Key.Allocator, allocator))
                    {
                        kv.Value.Dispose();
                        remove.Add(kv.Key);
                    }
                }

                foreach (CacheKey key in remove)
                    Cache.Remove(key);

                ReleaseArenaLocked(allocator);
            }
        }

        /// <summary>Frees the arena slabs backing this allocator's resident weights.
        /// Call once all weight pointers are done being used (model unload /
        /// allocator dispose). Cache entries that point into the freed slabs are
        /// purged so a later lookup re-uploads instead of returning a dangling
        /// pointer.</summary>
        public static void ReleaseArena(CudaAllocator allocator)
        {
            if (allocator == null)
                return;
            lock (Sync)
            {
                allocator.Context.MakeCurrent();
                // Drop only entries owned by this allocator's arena. Other
                // allocators on the same device have independent slabs/streams
                // and must remain resident when this model is unloaded.
                var remove = new List<CacheKey>();
                foreach (var kv in Cache)
                {
                    if (ReferenceEquals(kv.Key.Allocator, allocator) && kv.Value.ArenaBacked)
                        remove.Add(kv.Key);
                }
                foreach (CacheKey key in remove)
                    Cache.Remove(key);

                ReleaseArenaLocked(allocator);
            }
        }

        private static void ReleaseArenaLocked(CudaAllocator allocator)
        {
            if (Arenas.TryGetValue(allocator, out WeightArena arena))
            {
                arena.FreeAll();
                Arenas.Remove(allocator);
            }
        }

        /// <summary>Returns the device pointer of an already-preloaded quantized
        /// weight (by its host cache key), so callers can build a device-side
        /// table of per-expert weight pointers for the on-device MoE kernels.</summary>
        public static bool TryGetResidentDevicePtr(CudaAllocator allocator, IntPtr cacheKey, out IntPtr devicePtr, out int ggmlType)
        {
            devicePtr = IntPtr.Zero;
            ggmlType = 0;
            if (allocator == null || cacheKey == IntPtr.Zero)
                return false;

            var key = new CacheKey(allocator, cacheKey);
            lock (Sync)
            {
                // A planar NVFP4 matrix is not ggml blocks: its callers read the bytes by their ggml type.
                if (Cache.TryGetValue(key, out DeviceWeight w) && w.DevicePtr != IntPtr.Zero && w.GgmlType != Nvfp4PlanarType)
                {
                    devicePtr = w.DevicePtr;
                    ggmlType = w.GgmlType;
                    return true;
                }
            }
            return false;
        }

        /// <summary>Uploads a small array of device pointers to a device buffer
        /// (used by the on-device MoE kernels to index per-expert weights by the
        /// device-computed expert id). Free with <see cref="FreeDeviceBuffer"/>.</summary>
        public static unsafe IntPtr CreateDevicePointerTable(CudaAllocator allocator, IntPtr[] hostPtrs)
        {
            if (allocator == null)
                throw new ArgumentNullException(nameof(allocator));
            if (hostPtrs == null || hostPtrs.Length == 0)
                throw new ArgumentException("Pointer table must be non-empty.", nameof(hostPtrs));

            long bytes = (long)hostPtrs.Length * sizeof(ulong);
            var tmp = new ulong[hostPtrs.Length];
            for (int i = 0; i < hostPtrs.Length; i++)
                tmp[i] = (ulong)hostPtrs[i].ToInt64();

            allocator.Context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out IntPtr dev, new UIntPtr((ulong)bytes)).ThrowOnError();
            try
            {
                fixed (ulong* p = tmp)
                    CudaDriverApi.cuMemcpyHtoD(dev, (IntPtr)p, new UIntPtr((ulong)bytes)).ThrowOnError();
            }
            catch
            {
                CudaDriverApi.cuMemFree(dev);
                throw;
            }
            return dev;
        }

        public static void FreeDeviceBuffer(CudaAllocator allocator, IntPtr ptr)
        {
            if (allocator == null || ptr == IntPtr.Zero)
                return;
            allocator.Context.MakeCurrent();
            CudaDriverApi.cuMemFree(ptr);
        }

        // q8Kernel override for the Q8_0 path (lets a test drive each kernel in one
        // process): 0 = auto, 1 = dp4a, 3 = the exact scalar FP32 kernel.
        public static bool TryAddmmQuantizedToFloat32(
            Tensor result,
            Tensor input,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            int q8Kernel = 0)
        {
            // F16 (1), BF16 (30) and F32 (0) are not block-quantized types but are
            // accepted here: their resident bytes already ARE the cuBLAS operand
            // (see RunF16Gemm / RunGemm). Without this, a dense tensor inside a
            // quantized GGUF (Gemma E4B's 55 MB per_layer_model_proj in F16, or
            // every dense/attention tensor of a UD-Q8_K_XL DeepSeek V4 in BF16)
            // fell to the managed CPU matmul on EVERY prefill chunk and decode
            // token, with DtoH/HtoD mirror churn on top.
            if (!IsDenseGemmType(ggmlType) && !SupportsQuantizedType(ggmlType))
                return false;

            if (!CudaKernelOps.TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !CudaKernelOps.TryGetContiguousFloat(input, out CudaStorage inputStorage, out IntPtr inputPtr, out int inputCount) ||
                input.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                input.Sizes[1] != ne0 ||
                result.Sizes[0] != input.Sizes[0] ||
                result.Sizes[1] != ne1 ||
                inputCount != input.Sizes[0] * ne0 ||
                resultCount != result.Sizes[0] * ne1)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                return false;

            DeviceWeight weight = EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            inputStorage.EnsureDeviceCurrent();
            RunResidentMatmul(
                allocator, kernels, weight.DevicePtr, weight.GgmlType,
                inputPtr, resultPtr, checked((int)ne0), checked((int)ne1),
                checked((int)input.Sizes[0]), q8Kernel);
            resultStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// Tensor-level quantized/dense linear against a weight the CALLER already
        /// holds in device memory: result[rows, ne1] = input[rows, ne0] x W^T.
        ///
        /// Same kernel routing as <see cref="TryAddmmQuantizedToFloat32"/>, minus
        /// the <see cref="EnsureWeight"/> upload cache — for executors that own
        /// their own weight residency (DeepSeek V4 streams a whole model into
        /// per-device arenas and layer-splits it across GPUs, so it must not hand
        /// its weights to a per-allocator cache keyed by host pointer).
        ///
        /// <paramref name="result"/> and <paramref name="input"/> must be
        /// contiguous 2-D F32 tensors on the same CUDA allocator as the weight.
        /// </summary>
        public static void AddmmResidentToFloat32(
            Tensor result,
            Tensor input,
            IntPtr deviceWeight,
            int ggmlType,
            long ne0,
            long ne1,
            int q8Kernel = 0,
            bool rowInvariant = false)
        {
            if (deviceWeight == IntPtr.Zero)
                throw new ArgumentException("Resident weight pointer cannot be null.", nameof(deviceWeight));
            if (!SupportsMatmulType(ggmlType))
                throw new NotSupportedException($"CUDA matmul does not support ggml weight type {ggmlType}.");

            if (!CudaKernelOps.TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !CudaKernelOps.TryGetContiguousFloat(input, out CudaStorage inputStorage, out IntPtr inputPtr, out int inputCount) ||
                input.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                input.Sizes[1] != ne0 ||
                result.Sizes[0] != input.Sizes[0] ||
                result.Sizes[1] != ne1 ||
                inputCount != input.Sizes[0] * ne0 ||
                resultCount != result.Sizes[0] * ne1)
            {
                throw new ArgumentException(
                    $"Resident quantized matmul needs contiguous F32 [rows,{ne0}] x [rows,{ne1}] tensors.");
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                throw new InvalidOperationException("CUDA kernels are unavailable for this allocator.");

            inputStorage.EnsureDeviceCurrent();
            RunResidentMatmul(
                allocator, kernels, deviceWeight, ggmlType,
                inputPtr, resultPtr, checked((int)ne0), checked((int)ne1),
                checked((int)input.Sizes[0]), q8Kernel, rowInvariant);
            resultStorage.MarkDeviceModified();
        }

        /// <summary>Quantize <paramref name="rows"/> activation rows to q8_1 into the caller's buffer, as
        /// the decode kernels read them, for several projections of one input
        /// (<see cref="TryResidentMatmulQ81"/>).</summary>
        internal static void QuantizeRowsQ81(CudaAllocator allocator, IntPtr input, IntPtr xq, int inDim, int rows)
            => allocator.Kernels.LaunchQuantizeQ81Rows(input, xq, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);

        /// <summary>
        /// A decode-size resident matmul (1..16 rows) over activations already quantized by
        /// <see cref="QuantizeRowsQ81"/>: exactly the kernel <see cref="RunResidentMatmul"/> picks for
        /// these rows with <c>rowInvariant</c>, minus the per-call quantization, so an engine that
        /// projects one input several ways quantizes it once. False, with nothing launched, for a
        /// type or shape that route would not send to a q8_1 kernel; the caller then takes
        /// <see cref="AddmmResidentToFloat32"/>.
        /// </summary>
        /// <param name="stream">The stream to launch on; zero: the allocator's.</param>
        internal static bool TryResidentMatmulQ81(CudaAllocator allocator, IntPtr weightPtr, int ggmlType, IntPtr xq,
            IntPtr resultPtr, int inDim, int outDim, int rows, IntPtr stream = default)
        {
            if (!ResidentMatmulTakesQ81(ggmlType, inDim, outDim, rows))
                return false;
            CudaKernels kernels = allocator.Kernels;
            if (stream == IntPtr.Zero)
                stream = allocator.Stream.Handle;
            if (ggmlType == 10 || ggmlType == 11)
                kernels.LaunchQuantMatmulLowKDp4a(ggmlType, weightPtr, xq, resultPtr, inDim, outDim, rows, stream);
            else if (ggmlType == 14)
                kernels.LaunchQuantMatmulQ6KDp4aRows(weightPtr, xq, resultPtr, inDim, outDim, rows, stream);
            else
                kernels.LaunchQuantMatmulQ80VecRows(weightPtr, xq, resultPtr, inDim, outDim, rows, stream);
            return true;
        }

        /// <summary>Whether <see cref="TryResidentMatmulQ81"/> takes this weight type and shape: the
        /// q8_1 decode kernels <see cref="RunResidentMatmul"/> would pick for these rows.</summary>
        internal static bool ResidentMatmulTakesQ81(int ggmlType, int inDim, int outDim, int rows)
        {
            if (rows < 1 || rows > CudaKernels.Q80VecMaxRows)
                return false;
            if (ggmlType == 10 || ggmlType == 11)
                return rows <= CudaKernels.LowKMaxRows && (inDim & 255) == 0;
            if (ggmlType == 14)
                return Q6KDp4aEnabled && (inDim & 255) == 0;
            return ggmlType == 8 && Q80VecDp4aEnabled && (inDim & 31) == 0
                && (rows >= 2 || outDim >= Q80VecMinOutDim);
        }

        /// <summary>
        /// Kernel-selection core of the quantized/dense linear, factored out so it
        /// can also serve callers that already own their weights in device memory
        /// (the DeepSeek V4 engine keeps a whole model resident in per-device
        /// arenas and must not go through the <see cref="EnsureWeight"/> cache).
        /// Computes result[rows, outDim] = input[rows, inDim] x W^T for the
        /// resident weight at <paramref name="weightPtr"/>.
        ///
        /// This is the ONE place kernel routing lives: everything below dispatches
        /// on quant type and row count only, so any executor calling it picks up
        /// the same MMQ / dp4a / dequant+cuBLAS decisions.
        ///
        /// <paramref name="rowInvariant"/>: up to 16 rows, compute every row exactly as a
        /// one-row call would (an executor batching several sequences' decode steps
        /// wants a sequence's result not to depend on its batchmates). Q2_K/Q3_K
        /// always do; Q8_0 then takes its matvec rather than the int8 MMA GEMM.
        /// </summary>
        /// <summary>
        /// result[rows, outDim] = input[rows, inDim] W^T for an NVFP4 matrix in the planar layout: up to 8 rows on the q8
        /// matvec; more quantized to NVFP4 and multiplied on the FP4 tensor cores, TMA-fed from 65 rows, otherwise through
        /// the cp.async GEMM split over K (up to 64 rows) as far as one wave of CTAs.
        /// </summary>
        private static void RunNvfp4Planar(CudaAllocator allocator, IntPtr planes, IntPtr inputPtr, IntPtr resultPtr, int inDim,
            int outDim, int rows)
        {
            CudaFp4Kernels fp4 = allocator.Fp4Kernels
                ?? throw new InvalidOperationException("an NVFP4 weight is resident in the planar layout, but the FP4 kernels are not loaded");
            IntPtr stream = allocator.Stream.Handle;
            if (rows <= CudaFp4Kernels.GemvMaxRows)
            {
                IntPtr scratch = EnsureQ81Scratch(allocator, CudaFp4Kernels.GemvScratchBytes(rows, inDim));
                fp4.Gemv(planes, outDim, inDim, inputPtr, rows, scratch, resultPtr, stream);
                return;
            }
            var (codeBytes, scaleBytes, rowScaleBytes) = CudaFp4Kernels.ActivationBytes(rows, inDim);
            IntPtr codes = EnsureScratch(Fp4CodesScratch, allocator, codeBytes);
            IntPtr scales = EnsureScratch(Fp4ScalesScratch, allocator, scaleBytes);
            IntPtr rowScale = EnsureScratch(Fp4RowScaleScratch, allocator, rowScaleBytes);
            fp4.Quantize(inputPtr, codes, scales, rowScale, inDim, rows, stream);
            if (CudaFp4Kernels.TakesTma(rows, outDim, inDim, planes, codes, scales))
            {
                fp4.GemmTma(planes, outDim, inDim, codes, scales, rowScale, rows, 1.0f, resultPtr, stream);
                return;
            }
            int splits = fp4.Splits(outDim, inDim, rows, rows <= 64 ? 8 : 1);
            IntPtr partial = IntPtr.Zero, counters = IntPtr.Zero;
            if (splits > 1)
            {
                partial = EnsureScratch(Fp4PartialScratch, allocator, (long)splits * rows * outDim * sizeof(float));
                counters = EnsureZeroedScratch(Fp4CounterScratch, allocator, (long)CudaFp4Kernels.Tiles(outDim, rows) * sizeof(int));
            }
            fp4.Gemm(planes, outDim, inDim, codes, scales, rowScale, rows, 1.0f, resultPtr, partial, counters, splits, stream);
        }

        internal static void RunResidentMatmul(
            CudaAllocator allocator,
            CudaKernels kernels,
            IntPtr weightPtr,
            int ggmlType,
            IntPtr inputPtr,
            IntPtr resultPtr,
            int inDim,
            int outDim,
            int rows,
            int q8Kernel = 0,
            bool rowInvariant = false)
        {
            allocator.Context.MakeCurrent();

            if (ggmlType == Nvfp4PlanarType)
            {
                RunNvfp4Planar(allocator, weightPtr, inputPtr, resultPtr, inDim, outDim, rows);
                return;
            }

            // F16 weights: the resident bytes are already the GEMM operand, so
            // every row count (a rows==1 decode is just a GEMV) goes straight
            // to cuBLAS with only the activation's f32->f16 convert on top.
            if (ggmlType == 1)
            {
                RunF16Gemm(allocator, kernels, weightPtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);
                return;
            }

            // BF16 weights stay bit-exact in VRAM: converting them to F16 would
            // flush everything below 6.1e-5 to a subnormal. Decode (rows == 1) is
            // bandwidth-bound, so the dedicated matvec beats a cuBLAS GEMM with
            // n = 1; multi-row goes to the Ampere BF16 tensor cores.
            if (ggmlType == 30)
            {
                if (rows == 1 && (inDim & 7) == 0)
                {
                    kernels.LaunchMatvecBf16(weightPtr, inputPtr, resultPtr, inDim, outDim, allocator.Stream.Handle);
                }
                else
                {
                    IntPtr aBf16 = EnsureScratch(AF16Scratch, allocator, (long)rows * inDim * 2);
                    kernels.LaunchConvertF32Bf16(inputPtr, aBf16, (long)rows * inDim, allocator.Stream.Handle);
                    RunGemm(allocator, weightPtr, CublasApi.CUDA_R_16BF, aBf16, CublasApi.CUDA_R_16BF,
                        resultPtr, inDim, outDim, rows);
                }
                return;
            }

            // F32 weights (small dense tensors kept unquantized: MoE routers, the
            // hyper-connection mixing matrices) go straight to cuBLAS.
            if (ggmlType == 0)
            {
                RunGemm(allocator, weightPtr, CublasApi.CUDA_R_32F, inputPtr, CublasApi.CUDA_R_32F,
                    resultPtr, inDim, outDim, rows);
                return;
            }

            // Q2_K/Q3_K decode and batched decode (up to 16 rows): q8_1 activations and dp4a
            // vec-dots with each sub-block's weights decoded once for every row, one kernel for
            // every row count so a sequence's rows compute alike alone and batched. The scalar path
            // dequantized every element through a generic per-value decoder and read these at a
            // fraction of bandwidth (DeepSeek V4.1's Q2_K checkpoint keeps every dense projection at
            // Q2_K/Q3_K).
            if (rows <= CudaKernels.LowKMaxRows && (inDim & 255) == 0 && (ggmlType == 10 || ggmlType == 11))
            {
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulLowKDp4a(ggmlType, weightPtr, xqScratch, resultPtr,
                    inDim, outDim, rows, allocator.Stream.Handle);
                return;
            }

            // Q6_K decode, batched decode and verify windows for an executor that wants every row computed
            // as a one-row call computes it: the one-row dp4a kernel's arithmetic for up to 16 rows (the
            // generic multi-row kernel below reads F32 activations instead of q8_1).
            if (rowInvariant && ggmlType == 14 && Q6KDp4aEnabled
                && rows >= 2 && rows <= CudaKernels.Q80VecMaxRows && (inDim & 255) == 0)
            {
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulQ6KDp4aRows(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                return;
            }

            if (rowInvariant && ggmlType == 8 && q8Kernel == 0 && Q80VecDp4aEnabled
                && rows >= 2 && rows <= CudaKernels.Q80VecMaxRows && (inDim & 31) == 0)
            {
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulQ80VecRows(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                return;
            }

            // NVFP4 decode, batched decode and verify windows (1..16 rows): q8_1 activations and dp4a against the
            // resident blocks, the E2M1 codes decoded by byte permutes (the generic kernel below decoded each
            // value through a per-element decoder: Qwen3.8-27B NVFP4 decoded at 10.4 tok/s on an RTX PRO 4500).
            if (ggmlType == 40 && rows <= CudaKernels.Q80VecMaxRows && (inDim & 63) == 0)
            {
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulNvfp4VecRows(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                return;
            }

            // Q8_0 prefill-sized batches: direct int8 tensor-core GEMM over the raw
            // Q8_0 blocks (mma.m16n8k32, ggml MMQ-style). Weight DRAM traffic is
            // ceil(rows/128) sweeps with NO f16 dequant round trip, so it beats the
            // dequant+cuBLAS route up to ~TS_CUDA_Q80_MMQ_MAX_ROWS rows; above that
            // cuBLAS' larger tiles win and the F16 route below takes over.
            if (Q80MmqEnabled && kernels.HasQ80Mmq && ggmlType == 8 && q8Kernel == 0
                && rows >= F16GemmMinRows && rows <= Q80MmqMaxRows && (inDim & 31) == 0)
            {
                if (Q80Mmq2Enabled && (inDim & 255) == 0)
                {
                    // cp.async variant: split activation scratch (dense qs rows +
                    // float scales), raw weight windows async-staged in shared.
                    IntPtr qsScratch = EnsureScratch(Q81SplitQsScratch, allocator, (long)rows * inDim);
                    IntPtr dScratch = EnsureScratch(Q81SplitDScratch, allocator, (long)rows * (inDim / 32) * sizeof(float));
                    kernels.LaunchQuantizeQ81SplitRows(inputPtr, qsScratch, dScratch, inDim, rows, allocator.Stream.Handle);
                    kernels.LaunchQuantMatmulQ80Mmq2(
                        weightPtr, qsScratch, dScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                }
                else
                {
                    long mmqScratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                    IntPtr mmqXq = EnsureQ81Scratch(allocator, mmqScratchBytes);
                    kernels.LaunchQuantizeQ81Rows(
                        inputPtr, mmqXq, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                    kernels.LaunchQuantMatmulQ80Mmq(
                        weightPtr, mmqXq, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                }
                return;
            }

            // Prefill-sized batches: dequant-once + tensor-core cuBLAS GEMM (see
            // F16GemmMinRows). Applies to every supported quant type; q8Kernel != 0
            // means a test pinned a specific Q8_0 kernel, so honor that instead.
            if (q8Kernel == 0 && rows >= F16GemmMinRows
                && 2L * inDim * outDim <= F16GemmMaxWeightBytes)
            {
                RunF16Gemm(allocator, kernels, weightPtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);
                return;
            }
            // Small multi-row batches (speculative MTP verify windows, short
            // prefill chunks) run the per-row quant kernels once per output row --
            // the whole weight matrix is re-read (and re-dequantized) per row, so a
            // B-row matmul costs ~B x a single decode and speculative verification
            // can never amortize. The generic scalar batched kernel tiles
            // TS_QMM_ROW_TILE rows per block, streaming each weight ONCE and reusing
            // it across the tile (measured 2.7-2.9x at B>=4 for the k-quant
            // ffn_down / LM-head weights).
            //   * Q4_0 (the dominant dense quant -- the 12B QAT model is entirely
            //     Q4_0) has its OWN row-tiled kernel below that keeps the efficient
            //     per-row nibble unpacking AND amortizes the weight read + dequant
            //     across the tile; the generic scalar batched path is ~2x slower for
            //     it (warp-per-column under-fills vs the block-per-4-columns design,
            //     and qvalue_at re-branches per element).
            //   * IQ2_XXS / Q8_0 keep their own multi-row paths (the q8_1 dp4a/MMA
            //     GEMM and the IQ2_XXS q8_1 kernel) which already read the weight
            //     once per tile and beat the scalar batched kernel for those types.
            // The batched kernel tiles rows down grid.y (any row count), so it also
            // covers LARGE prefill batches: the generic quant types here (K-quants,
            // IQ2_S / IQ3_S / IQ3_XXS / IQ4_XS) have no specialized large-batch GEMM,
            // and the per-row scalar fallback re-dequantizes the whole weight for
            // EVERY output row -- a 2048-token prefill then costs ~2048x the (memory-
            // bound) weight traffic and dominates TTFT (and the startup warmup). Route
            // all batch sizes (not just <=QuantMatmulBatchMaxRows) through the tiled
            // kernel so weight traffic drops ~TS_QMM_ROW_TILE x; it is numerically
            // identical (verified by the K-quant / IQ4_XS matmul tests).
            //
            // rows == 1 (single-token decode) deliberately stays OUT of this path.
            // Mixed "unsloth dynamic" quants (e.g. UD-IQ2_XXS) keep a large share
            // of tensors at Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS,
            // and at rows==1 this kernel's one-warp-per-column design (32 threads
            // striding in_dim, TS_QMM_ROW_TILE amortization providing zero benefit
            // with only one row) has LESS per-column parallelism than the scalar
            // path's cols_per_block=4/blockDim=256 split -- measured on
            // Qwen3.6-27B-UD-IQ2_XXS: routing rows==1 here made decode SLOWER
            // (5.18 -> 4.54 tok/s) despite the scalar kernel's 76% GPU-time share,
            // because the wide lm_head/large-in_dim tensors dominate that share
            // and regressed further under 32-thread columns. See
            // ts_quant_matmul_vec_f32 below for the actual decode fix: one BLOCK
            // (not one warp) per output column, matching the scalar path's
            // per-column thread budget without its 4-columns-per-block
            // serialization.
            if (rows >= 2
                && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
            {
                kernels.LaunchQuantMatmulBatchedF32(
                    weightPtr, inputPtr, resultPtr,
                    ggmlType, inDim, outDim, rows, allocator.Stream.Handle);
                return;
            }
            // Single-token decode for the same generic quant-type set: one BLOCK
            // per output column (full blockDim.x threads via block_reduce_sum),
            // not the row-batched kernel's one-WARP-per-column (32 threads). At
            // rows==1 there is no cross-row weight reuse to amortize, so the
            // batched kernel's only advantage disappears and its 8x-narrower
            // per-column thread count directly costs latency on wide tensors
            // (e.g. a K-quant lm_head/large-in_dim projection).
            // Q4_K single-token decode: q8_1 activation + dp4a (see the kernel).
            // inDim % 256 == 0 holds for every real model dim (Q4_K is a 256-block
            // quant); the fallbacks below cover any exotic shape.
            if (Q4KDp4aEnabled && ggmlType == 12 && rows == 1 && (inDim & 255) == 0)
            {
                long scratchBytes = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulQ4KDp4a(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, allocator.Stream.Handle);
                return;
            }
            // Q5_K/Q6_K decode: use the same global q8_1 scratch as Q4_K, with
            // format-specific packed dp4a vec-dots. Keep this strictly rows==1;
            // batched prefill retains its weight-reusing row-tiled path.
            if (rows == 1 && (inDim & 255) == 0
                && ((ggmlType == 13 && Q5KDp4aEnabled)
                    || (ggmlType == 14 && Q6KDp4aEnabled)))
            {
                long scratchBytes = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, 1,
                    allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                if (ggmlType == 13)
                {
                    kernels.LaunchQuantMatmulQ5KDp4a(
                        weightPtr, xqScratch, resultPtr,
                        inDim, outDim, allocator.Stream.Handle);
                }
                else
                {
                    kernels.LaunchQuantMatmulQ6KDp4a(
                        weightPtr, xqScratch, resultPtr,
                        inDim, outDim, allocator.Stream.Handle);
                }
                return;
            }
            // Decode IQ2 matvec: quantize the activation row once into the global
            // allocator scratch, then four warp-owned output columns per CTA reuse
            // it. The older IQ2_XXS kernel rebuilt the same q8_1 row in shared
            // memory in every CTA; IQ2_S otherwise fell through to scalar qvalue_at.
            if (Iq2VecDp4aEnabled && rows == 1
                && (ggmlType == 16 || ggmlType == 22)
                && (inDim & 255) == 0)
            {
                long scratchBytes = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, 1,
                    allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulIq2VecQ81F32(
                    weightPtr, xqScratch, resultPtr,
                    ggmlType, inDim, outDim, allocator.Stream.Handle);
                return;
            }
            if (rows == 1
                && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
            {
                kernels.LaunchQuantMatmulVecF32(
                    weightPtr, inputPtr, resultPtr,
                    ggmlType, inDim, outDim, allocator.Stream.Handle);
                return;
            }
            if (ggmlType == 2)
            {
                // Q4_0 int8 dp4a: quantize the activation rows to q8_1 ONCE, then the
                // block-tile dp4a GEMM. This is the fast path for BOTH single-token
                // decode (rows==1 — the dominant cost, and where the scalar FP32
                // dequant kernel read the LM head's Q4_0 weights at ~26 GB/s) and the
                // verify window. dp4a does 4 int8 MACs/instruction so it is ~memory-
                // bound, matching ggml's mul_mat_vec_q. inDim is always a multiple of
                // 32 for these models; non-conforming shapes fall through to FP32.
                if (Q40Dp4aEnabled && (inDim & 31) == 0)
                {
                    long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                    IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                    kernels.LaunchQuantizeQ81Rows(
                        inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                    kernels.LaunchQuantMatmulQ40Dp4a(
                        weightPtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
                    return;
                }
                // FP32 fallback: row-tiled batched kernel for the verify window
                // (decode each weight nibble ONCE and reuse it across the tile's rows)
                // and the per-row kernel for single-row decode.
                if (rows >= 2 && rows <= CudaKernels.QuantMatmulBatchMaxRows)
                {
                    kernels.LaunchQuantMatmulQ40BatchedF32(
                        weightPtr, inputPtr, resultPtr,
                        inDim, outDim, rows, allocator.Stream.Handle);
                    return;
                }
                kernels.LaunchQuantMatmulQ40F32(
                    weightPtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else if (ggmlType == 8 && (inDim & 31) == 0
                     && (q8Kernel == 0 || q8Kernel == 1)
                     && (rows >= 2 || (Q80VecDp4aEnabled && outDim >= Q80VecMinOutDim)))
            {
                // Q8_0 int8 fast path (rows >= 1): quantize the activation rows to q8_1
                // ONCE into a reused scratch (single-stream per allocator makes reuse
                // safe), then the block-tile dp4a GEMM. dp4a covers single-token
                // decode too (4 int8 MACs/instruction, ~memory-bound like ggml's
                // mul_mat_vec_q) - the scalar per-byte dequant kernel it replaces left
                // ~25% of weight bandwidth unused, and decode reads every weight once
                // per token.
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(allocator, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, allocator.Stream.Handle, Q81WarpQuantizeEnabled);
                if (rows == 1)
                    kernels.LaunchQuantMatmulQ80Vec(
                        weightPtr, xqScratch, resultPtr, inDim, outDim, allocator.Stream.Handle);
                else
                    kernels.LaunchQuantMatmulQ80Dp4a(
                        weightPtr, xqScratch, resultPtr, inDim, outDim, rows, allocator.Stream.Handle);
            }
            else if (ggmlType == 8)
            {
                kernels.LaunchQuantMatmulQ80F32(
                    weightPtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else if (ggmlType == 16 && (inDim & 255) == 0 && ((inDim / 32) * 36) <= 48 * 1024)
            {
                kernels.LaunchQuantMatmulIq2XxsQ81F32(
                    weightPtr,
                    inputPtr,
                    resultPtr,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchQuantMatmulF32(
                    weightPtr,
                    inputPtr,
                    resultPtr,
                    ggmlType,
                    inDim,
                    outDim,
                    rows,
                    allocator.Stream.Handle);
            }

        }


        public static bool TryGetRowsQuantizedToFloat32(
            Tensor result,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            Tensor indices)
        {
            if (!SupportsQuantizedType(ggmlType))
                return false;

            if (!CudaKernelOps.TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !CudaKernelOps.TryGetContiguous(indices, out CudaStorage indicesStorage, out IntPtr indicesPtr, out long indexCount) ||
                cols != ne0 ||
                indexCount != rows ||
                result.ElementType != DType.Float32 ||
                (indices.ElementType != DType.Int32 && indices.ElementType != DType.Float32))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            CudaKernels kernels = allocator.Kernels;
            if (kernels == null)
                return false;

            DeviceWeight weight = EnsureWeight(allocator, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            indicesStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            if (weight.GgmlType == Nvfp4PlanarType)
            {
                allocator.Fp4Kernels.GetRows(weight.DevicePtr, checked((int)ne1), checked((int)ne0), indicesPtr,
                    indices.ElementType != DType.Int32, rows, resultPtr, allocator.Stream.Handle);
                resultStorage.MarkDeviceModified();
                return true;
            }
            kernels.LaunchQuantGetRowsF32(
                weight.DevicePtr,
                indicesPtr,
                resultPtr,
                ggmlType,
                checked((int)ne0),
                rows,
                indices.ElementType == DType.Int32 ? 1 : 0,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>The ggml blocks just uploaded to <paramref name="devicePtr"/> rewritten in place as the planar layout
        /// (through a temporary copy of the blocks).</summary>
        private static void RepackNvfp4(CudaAllocator allocator, CudaFp4Kernels fp4, IntPtr devicePtr, long rawBytes, int rows, int k)
        {
            CudaDriverApi.cuMemAlloc(out IntPtr blocks, new UIntPtr((ulong)rawBytes)).ThrowOnError();
            try
            {
                CudaDriverApi.cuMemcpyDtoDAsync(blocks, devicePtr, new UIntPtr((ulong)rawBytes), allocator.Stream.Handle).ThrowOnError();
                fp4.Repack(blocks, devicePtr, rows, k, allocator.Stream.Handle);
                allocator.Stream.Synchronize();
            }
            finally
            {
                CudaDriverApi.cuMemFree(blocks);
            }
        }

        private static DeviceWeight EnsureWeight(
            CudaAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            Action<IntPtr> upload = null)
        {
            if (cacheKey == IntPtr.Zero)
                cacheKey = hostData;
            if (cacheKey == IntPtr.Zero)
                throw new ArgumentException("CUDA quantized weight cache key cannot be zero.", nameof(cacheKey));

            var key = new CacheKey(allocator, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out DeviceWeight existing))
                {
                    if (upload != null && (existing.RawBytes != rawBytes || existing.SourceType != ggmlType ||
                        existing.Ne0 != ne0 || existing.Ne1 != ne1))
                        throw new InvalidOperationException("Source upload dimensions differ from the resident CUDA weight.");
                    return existing;
                }

                if (hostData == IntPtr.Zero && upload == null)
                    throw new InvalidOperationException("Quantized weight is not preloaded on this CUDA device and no host data was provided.");

                allocator.Context.MakeCurrent();
                // Suballocate from the per-allocator weight arena instead of one
                // cuMemAlloc per tensor: model weights live for the model's whole
                // lifetime and are freed together at unload, so packing them into
                // large slabs removes the per-tensor driver rounding that otherwise
                // wastes ~1.6 GB on a 26B MoE (the +16 trailing slack the cp.async
                // MMQ kernel needs is reserved inside Allocate).
                if (!Arenas.TryGetValue(allocator, out WeightArena arena))
                {
                    arena = new WeightArena(allocator);
                    Arenas.Add(allocator, arena);
                }

                IntPtr devicePtr = arena.Allocate(rawBytes);
                try
                {
                    if (upload == null)
                        CudaDriverApi.cuMemcpyHtoD(devicePtr, hostData, new UIntPtr((ulong)rawBytes)).ThrowOnError();
                    else
                        upload(devicePtr);
                }
                catch
                {
                    arena.Release(devicePtr);
                    throw;
                }
                // NVFP4 on a GPU with the FP4 tensor-core kernels: the planar layout they read (the same bytes).
                int layout = ggmlType;
                if (ggmlType == 40 && CudaFp4Kernels.TakesWidth(ne0) && allocator.Fp4Kernels is { } fp4)
                {
                    RepackNvfp4(allocator, fp4, devicePtr, rawBytes, checked((int)ne1), checked((int)ne0));
                    layout = Nvfp4PlanarType;
                }
                var entry = new DeviceWeight
                {
                    DevicePtr = devicePtr,
                    RawBytes = rawBytes,
                    GgmlType = layout,
                    SourceType = ggmlType,
                    Ne0 = ne0,
                    Ne1 = ne1,
                    DeviceId = allocator.DeviceId,
                    ArenaBacked = true,
                    ArenaOwner = arena,
                };
                Cache.Add(key, entry);
                return entry;
            }
        }

        private readonly struct CacheKey : IEquatable<CacheKey>
        {
            public CacheKey(CudaAllocator allocator, IntPtr key)
            {
                Allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
                Key = key;
            }

            public CudaAllocator Allocator { get; }
            public IntPtr Key { get; }

            public bool Equals(CacheKey other)
            {
                return ReferenceEquals(Allocator, other.Allocator) && Key == other.Key;
            }

            public override bool Equals(object obj)
            {
                return obj is CacheKey other && Equals(other);
            }

            public override int GetHashCode()
            {
                // CudaAllocator is sealed and uses object-identity equality/hash.
                return HashCode.Combine(Allocator, Key);
            }
        }
    }
}
