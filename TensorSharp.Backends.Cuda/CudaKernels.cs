using System;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class CudaKernels : IDisposable
    {
        private const int BlockSize = 256;

        private readonly CudaModule module;
        private readonly IntPtr fillF32;
        private readonly IntPtr fillF16;
        private readonly IntPtr unaryF32;
        private readonly IntPtr binaryF32;
        private readonly IntPtr scalarF32;
        private readonly IntPtr ternaryF32;
        private readonly IntPtr addMulScalarF32;
        private readonly IntPtr mulMulAddF32;
        private readonly IntPtr binaryActivationF32;
        private readonly IntPtr addBiasRowsF32;
        private readonly IntPtr siluMulSplitF32;
        private readonly IntPtr geluMulSplitF32;
        private readonly IntPtr swigluOaiSplitF32;
        private readonly IntPtr rmsNormF32;
        private readonly IntPtr softmaxF32;
        private readonly IntPtr attentionSoftmaxSinksF32;
        private readonly IntPtr scaledDotProductAttentionF32;
        private readonly IntPtr gqaPrefillAttentionF32;
        private readonly IntPtr gqaPrefillAttentionF16;
        private readonly IntPtr gqaPrefillAttentionSinksF32;
        private readonly IntPtr gqaPrefillAttentionSinksF16;
        private readonly IntPtr gqaDecodeAttentionF32;
        private readonly IntPtr gqaDecodeAttentionF16;
        private readonly IntPtr gqaDecodeAttentionSinksF32;
        private readonly IntPtr gqaDecodeAttentionSinksF16;
        private readonly IntPtr gqaDecodeAttentionPartitionF32;
        private readonly IntPtr gqaDecodeAttentionPartitionF16;
        private readonly IntPtr gqaDecodeAttentionPartitionReduceF32;
        private readonly IntPtr sliceColumnsF32;
        private readonly IntPtr flatToHeadFirstF32;
        private readonly IntPtr splitQkvHeadFirstF32;
        private readonly IntPtr copyHeadFirstToCacheF32;
        private readonly IntPtr copyHeadFirstToCacheF16;
        private readonly IntPtr gatherCircularHeadFirstF32;
        private readonly IntPtr gatherCircularHeadFirstF16;
        private readonly IntPtr concatHeadFirstF32;
        private readonly IntPtr neoxRopeHeadFirstF32;
        private readonly IntPtr indexSelectF32;
        private readonly IntPtr addCausalMaskF32;
        private readonly IntPtr ropeF32;
        private readonly IntPtr ropeExF32;
        private readonly IntPtr quantMatmulF32;
        private readonly IntPtr quantMatmulIq2XxsQ81F32;
        private readonly IntPtr quantMatmulQ40F32;
        private readonly IntPtr quantMatmulQ80SingleF32;
        private readonly IntPtr quantMatmulQ80F32;
        private readonly IntPtr quantGetRowsF32;

        private CudaKernels(CudaModule module)
        {
            this.module = module;
            fillF32 = module.GetFunction("ts_fill_f32");
            fillF16 = module.GetFunction("ts_fill_f16");
            unaryF32 = module.GetFunction("ts_unary_f32");
            binaryF32 = module.GetFunction("ts_binary_f32");
            scalarF32 = module.GetFunction("ts_scalar_f32");
            ternaryF32 = module.GetFunction("ts_ternary_f32");
            addMulScalarF32 = module.GetFunction("ts_addmul_scalar_f32");
            mulMulAddF32 = module.GetFunction("ts_mulmuladd_f32");
            binaryActivationF32 = module.GetFunction("ts_binary_activation_f32");
            addBiasRowsF32 = module.GetFunction("ts_add_bias_rows_f32");
            siluMulSplitF32 = module.GetFunction("ts_silu_mul_split_f32");
            geluMulSplitF32 = module.GetFunction("ts_gelu_mul_split_f32");
            swigluOaiSplitF32 = module.GetFunction("ts_swiglu_oai_split_f32");
            rmsNormF32 = module.GetFunction("ts_rmsnorm_f32");
            softmaxF32 = module.GetFunction("ts_softmax_f32");
            attentionSoftmaxSinksF32 = module.GetFunction("ts_attention_softmax_sinks_f32");
            scaledDotProductAttentionF32 = module.GetFunction("ts_scaled_dot_product_attention_f32");
            gqaPrefillAttentionF32 = module.GetFunction("ts_gqa_prefill_attention_f32");
            gqaPrefillAttentionF16 = module.GetFunction("ts_gqa_prefill_attention_f16");
            gqaPrefillAttentionSinksF32 = module.GetFunction("ts_gqa_prefill_attention_sinks_f32");
            gqaPrefillAttentionSinksF16 = module.GetFunction("ts_gqa_prefill_attention_sinks_f16");
            gqaDecodeAttentionF32 = module.GetFunction("ts_gqa_decode_attention_f32");
            gqaDecodeAttentionF16 = module.GetFunction("ts_gqa_decode_attention_f16");
            gqaDecodeAttentionSinksF32 = module.GetFunction("ts_gqa_decode_attention_sinks_f32");
            gqaDecodeAttentionSinksF16 = module.GetFunction("ts_gqa_decode_attention_sinks_f16");
            gqaDecodeAttentionPartitionF32 = module.GetFunction("ts_gqa_decode_attention_partition_f32");
            gqaDecodeAttentionPartitionF16 = module.GetFunction("ts_gqa_decode_attention_partition_f16");
            gqaDecodeAttentionPartitionReduceF32 = module.GetFunction("ts_gqa_decode_attention_partition_reduce_f32");
            sliceColumnsF32 = module.GetFunction("ts_slice_columns_f32");
            flatToHeadFirstF32 = module.GetFunction("ts_flat_to_head_first_f32");
            splitQkvHeadFirstF32 = module.GetFunction("ts_split_qkv_head_first_f32");
            copyHeadFirstToCacheF32 = module.GetFunction("ts_copy_head_first_to_cache_f32");
            copyHeadFirstToCacheF16 = module.GetFunction("ts_copy_head_first_to_cache_f16");
            gatherCircularHeadFirstF32 = module.GetFunction("ts_gather_circular_head_first_f32");
            gatherCircularHeadFirstF16 = module.GetFunction("ts_gather_circular_head_first_f16");
            concatHeadFirstF32 = module.GetFunction("ts_concat_head_first_f32");
            neoxRopeHeadFirstF32 = module.GetFunction("ts_neox_rope_head_first_f32");
            indexSelectF32 = module.GetFunction("ts_index_select_f32");
            addCausalMaskF32 = module.GetFunction("ts_add_causal_mask_f32");
            ropeF32 = module.GetFunction("ts_rope_f32");
            ropeExF32 = module.GetFunction("ts_rope_ex_f32");
            quantMatmulF32 = module.GetFunction("ts_quant_matmul_f32");
            quantMatmulIq2XxsQ81F32 = module.GetFunction("ts_quant_matmul_iq2_xxs_q8_1_f32");
            quantMatmulQ40F32 = module.GetFunction("ts_quant_matmul_q4_0_f32");
            quantMatmulQ80SingleF32 = module.GetFunction("ts_quant_matmul_q8_0_single_f32");
            quantMatmulQ80F32 = module.GetFunction("ts_quant_matmul_q8_0_f32");
            quantGetRowsF32 = module.GetFunction("ts_quant_get_rows_f32");
        }

        public static CudaKernels TryCreate()
        {
            string path = LocatePtxPath();
            if (path == null)
                return null;

            try
            {
                return new CudaKernels(CudaModule.LoadFromFile(path));
            }
            catch
            {
                return null;
            }
        }

        public void LaunchFillF32(IntPtr output, int count, float value, IntPtr stream)
        {
            IntPtr outArg = output;
            int countArg = count;
            float valueArg = value;
            void** args = stackalloc void*[] { &outArg, &countArg, &valueArg };
            Launch(fillF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchFillF16(IntPtr output, int count, float value, IntPtr stream)
        {
            IntPtr outArg = output;
            int countArg = count;
            float valueArg = value;
            void** args = stackalloc void*[] { &outArg, &countArg, &valueArg };
            Launch(fillF16, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchUnaryF32(IntPtr input, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &countArg, &opArg };
            Launch(unaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchBinaryF32(IntPtr lhs, IntPtr rhs, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr lhsArg = lhs;
            IntPtr rhsArg = rhs;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &lhsArg, &rhsArg, &outputArg, &countArg, &opArg };
            Launch(binaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchScalarF32(IntPtr input, IntPtr output, int count, float value, int op, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int countArg = count;
            float valueArg = value;
            int opArg = op;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &countArg, &valueArg, &opArg };
            Launch(scalarF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchTernaryF32(IntPtr x, IntPtr y, IntPtr z, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr xArg = x;
            IntPtr yArg = y;
            IntPtr zArg = z;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &xArg, &yArg, &zArg, &outputArg, &countArg, &opArg };
            Launch(ternaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAddMulScalarF32(IntPtr x, IntPtr y, IntPtr output, int count, float value, IntPtr stream)
        {
            IntPtr xArg = x;
            IntPtr yArg = y;
            IntPtr outputArg = output;
            int countArg = count;
            float valueArg = value;
            void** args = stackalloc void*[] { &xArg, &yArg, &outputArg, &countArg, &valueArg };
            Launch(addMulScalarF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchMulMulAddF32(IntPtr x, IntPtr y, IntPtr z, IntPtr w, IntPtr output, int count, IntPtr stream)
        {
            IntPtr xArg = x;
            IntPtr yArg = y;
            IntPtr zArg = z;
            IntPtr wArg = w;
            IntPtr outputArg = output;
            int countArg = count;
            void** args = stackalloc void*[] { &xArg, &yArg, &zArg, &wArg, &outputArg, &countArg };
            Launch(mulMulAddF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchBinaryActivationF32(IntPtr a, IntPtr b, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr aArg = a;
            IntPtr bArg = b;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &aArg, &bArg, &outputArg, &countArg, &opArg };
            Launch(binaryActivationF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAddBiasRowsF32(IntPtr tensor, IntPtr bias, int rows, int cols, int biasCols, IntPtr stream)
        {
            IntPtr tensorArg = tensor;
            IntPtr biasArg = bias;
            int rowsArg = rows;
            int colsArg = cols;
            int biasColsArg = biasCols;
            int count = checked(rows * cols);
            void** args = stackalloc void*[] { &tensorArg, &biasArg, &rowsArg, &colsArg, &biasColsArg };
            Launch(addBiasRowsF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSiLUMulSplitF32(IntPtr gateUp, IntPtr output, int rows, int halfDim, IntPtr stream)
        {
            IntPtr gateUpArg = gateUp;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int halfDimArg = halfDim;
            int count = checked(rows * halfDim);
            void** args = stackalloc void*[] { &gateUpArg, &outputArg, &rowsArg, &halfDimArg };
            Launch(siluMulSplitF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchGELUMulSplitF32(IntPtr gateUp, IntPtr output, int rows, int halfDim, IntPtr stream)
        {
            IntPtr gateUpArg = gateUp;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int halfDimArg = halfDim;
            int count = checked(rows * halfDim);
            void** args = stackalloc void*[] { &gateUpArg, &outputArg, &rowsArg, &halfDimArg };
            Launch(geluMulSplitF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSwiGluOaiSplitF32(IntPtr gateUp, IntPtr output, int rows, int halfDim, float alpha, float limit, IntPtr stream)
        {
            IntPtr gateUpArg = gateUp;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int halfDimArg = halfDim;
            float alphaArg = alpha;
            float limitArg = limit;
            int count = checked(rows * halfDim);
            void** args = stackalloc void*[] { &gateUpArg, &outputArg, &rowsArg, &halfDimArg, &alphaArg, &limitArg };
            Launch(swigluOaiSplitF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchRMSNormF32(IntPtr input, IntPtr alpha, IntPtr beta, IntPtr output, int rows, int cols, float eps, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr alphaArg = alpha;
            IntPtr betaArg = beta;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            float epsArg = eps;
            void** args = stackalloc void*[] { &inputArg, &alphaArg, &betaArg, &outputArg, &rowsArg, &colsArg, &epsArg };
            Launch(rmsNormF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSoftmaxF32(IntPtr input, IntPtr output, int rows, int cols, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &rowsArg, &colsArg };
            Launch(softmaxF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAttentionSoftmaxSinksF32(
            IntPtr scores,
            IntPtr sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr scoresArg = scores;
            IntPtr sinksArg = sinks;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &scoresArg, &sinksArg, &numHeadsArg, &seqLenArg, &kvLenArg,
                &maskStartArg, &windowSizeArg, &scaleArg, &hasSinksArg
            };
            Launch(attentionSoftmaxSinksF32, (uint)(numHeads * seqLen), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchScaledDotProductAttentionF32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr mask,
            IntPtr output,
            int batch,
            int seqQ,
            int seqK,
            int heads,
            int keyDim,
            int valueDim,
            float scale,
            int hasMask,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr maskArg = mask;
            IntPtr outputArg = output;
            int batchArg = batch;
            int seqQArg = seqQ;
            int seqKArg = seqK;
            int headsArg = heads;
            int keyDimArg = keyDim;
            int valueDimArg = valueDim;
            float scaleArg = scale;
            int hasMaskArg = hasMask;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &maskArg, &outputArg, &batchArg, &seqQArg,
                &seqKArg, &headsArg, &keyDimArg, &valueDimArg, &scaleArg, &hasMaskArg
            };
            Launch(scaledDotProductAttentionF32, (uint)(batch * heads), (uint)seqQ, 1, BlockSize, 1, 1, (uint)(seqK * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionF32(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg
            };
            Launch(gqaPrefillAttentionF32, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionF16(
            IntPtr query,
            IntPtr key,
            IntPtr value,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyArg = key;
            IntPtr valueArg = value;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyArg, &valueArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &seqLenArg, &kvLenArg, &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg
            };
            Launch(gqaPrefillAttentionF16, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionSinksF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int cacheSize,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int cacheSizeArg = cacheSize;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &seqLenArg, &kvLenArg, &cacheSizeArg,
                &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg, &hasSinksArg
            };
            Launch(gqaPrefillAttentionSinksF32, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaPrefillAttentionSinksF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int seqLen,
            int kvLen,
            int cacheSize,
            int headDim,
            int maskStart,
            int windowSize,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int seqLenArg = seqLen;
            int kvLenArg = kvLen;
            int cacheSizeArg = cacheSize;
            int headDimArg = headDim;
            int maskStartArg = maskStart;
            int windowSizeArg = windowSize;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &seqLenArg, &kvLenArg, &cacheSizeArg,
                &headDimArg, &maskStartArg, &windowSizeArg, &scaleArg, &hasSinksArg
            };
            Launch(gqaPrefillAttentionSinksF16, (uint)numQHeads, (uint)seqLen, 1, BlockSize, 1, 1, (uint)(kvLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &headDimArg, &attendStartArg, &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg
            };
            Launch(gqaDecodeAttentionF32, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(attendLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &outputArg, &numQHeadsArg, &numKVHeadsArg,
                &headDimArg, &attendStartArg, &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg
            };
            Launch(gqaDecodeAttentionF16, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(attendLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionSinksF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg
            };
            Launch(gqaDecodeAttentionSinksF32, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(attendLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionSinksF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr output,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &outputArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg
            };
            Launch(gqaDecodeAttentionSinksF16, (uint)numQHeads, 1, 1, BlockSize, 1, 1, (uint)(attendLen * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionPartitionF32(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr partial,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            int numPartitions,
            int partitionSize,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr partialArg = partial;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int numPartitionsArg = numPartitions;
            int partitionSizeArg = partitionSize;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &partialArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg,
                &numPartitionsArg, &partitionSizeArg
            };
            Launch(gqaDecodeAttentionPartitionF32, (uint)numQHeads, (uint)numPartitions, 1, BlockSize, 1, 1, (uint)(partitionSize * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionPartitionF16(
            IntPtr query,
            IntPtr keyCache,
            IntPtr valueCache,
            IntPtr sinks,
            IntPtr partial,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            int circular,
            float scale,
            int hasSinks,
            int numPartitions,
            int partitionSize,
            IntPtr stream)
        {
            IntPtr queryArg = query;
            IntPtr keyCacheArg = keyCache;
            IntPtr valueCacheArg = valueCache;
            IntPtr sinksArg = sinks;
            IntPtr partialArg = partial;
            int numQHeadsArg = numQHeads;
            int numKVHeadsArg = numKVHeads;
            int headDimArg = headDim;
            int attendStartArg = attendStart;
            int attendLenArg = attendLen;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            float scaleArg = scale;
            int hasSinksArg = hasSinks;
            int numPartitionsArg = numPartitions;
            int partitionSizeArg = partitionSize;
            void** args = stackalloc void*[]
            {
                &queryArg, &keyCacheArg, &valueCacheArg, &sinksArg, &partialArg,
                &numQHeadsArg, &numKVHeadsArg, &headDimArg, &attendStartArg,
                &attendLenArg, &cacheSizeArg, &circularArg, &scaleArg, &hasSinksArg,
                &numPartitionsArg, &partitionSizeArg
            };
            Launch(gqaDecodeAttentionPartitionF16, (uint)numQHeads, (uint)numPartitions, 1, BlockSize, 1, 1, (uint)(partitionSize * sizeof(float)), stream, args);
        }

        public void LaunchGqaDecodeAttentionPartitionReduceF32(
            IntPtr partial,
            IntPtr output,
            int numQHeads,
            int headDim,
            int numPartitions,
            IntPtr stream)
        {
            IntPtr partialArg = partial;
            IntPtr outputArg = output;
            int numQHeadsArg = numQHeads;
            int headDimArg = headDim;
            int numPartitionsArg = numPartitions;
            void** args = stackalloc void*[] { &partialArg, &outputArg, &numQHeadsArg, &headDimArg, &numPartitionsArg };
            Launch(gqaDecodeAttentionPartitionReduceF32, (uint)numQHeads, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSliceColumnsF32(
            IntPtr source,
            IntPtr output,
            int rows,
            int sourceCols,
            int colOffset,
            int width,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int sourceColsArg = sourceCols;
            int colOffsetArg = colOffset;
            int widthArg = width;
            int count = checked(rows * width);
            void** args = stackalloc void*[] { &sourceArg, &outputArg, &rowsArg, &sourceColsArg, &colOffsetArg, &widthArg };
            Launch(sliceColumnsF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchFlatToHeadFirstF32(
            IntPtr source,
            IntPtr output,
            int seqLen,
            int numHeads,
            int headDim,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int seqLenArg = seqLen;
            int numHeadsArg = numHeads;
            int headDimArg = headDim;
            int count = checked(seqLen * numHeads * headDim);
            void** args = stackalloc void*[] { &sourceArg, &outputArg, &seqLenArg, &numHeadsArg, &headDimArg };
            Launch(flatToHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchSplitQkvHeadFirstF32(
            IntPtr source,
            IntPtr output,
            int seqLen,
            int sourceCols,
            int colOffset,
            int numHeads,
            int headDim,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr outputArg = output;
            int seqLenArg = seqLen;
            int sourceColsArg = sourceCols;
            int colOffsetArg = colOffset;
            int numHeadsArg = numHeads;
            int headDimArg = headDim;
            int count = checked(seqLen * numHeads * headDim);
            void** args = stackalloc void*[]
            {
                &sourceArg, &outputArg, &seqLenArg, &sourceColsArg, &colOffsetArg, &numHeadsArg, &headDimArg
            };
            Launch(splitQkvHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchCopyHeadFirstToCacheF32(
            IntPtr source,
            IntPtr cache,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            int circular,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr cacheArg = cache;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &sourceArg, &cacheArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg, &circularArg
            };
            Launch(copyHeadFirstToCacheF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchCopyHeadFirstToCacheF16(
            IntPtr source,
            IntPtr cache,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            int circular,
            IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr cacheArg = cache;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int circularArg = circular;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &sourceArg, &cacheArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg, &circularArg
            };
            Launch(copyHeadFirstToCacheF16, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchGatherCircularHeadFirstF32(
            IntPtr cache,
            IntPtr output,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            IntPtr stream)
        {
            IntPtr cacheArg = cache;
            IntPtr outputArg = output;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &cacheArg, &outputArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg
            };
            Launch(gatherCircularHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchGatherCircularHeadFirstF16(
            IntPtr cache,
            IntPtr output,
            int numHeads,
            int seqLen,
            int headDim,
            int startPos,
            int cacheSize,
            IntPtr stream)
        {
            IntPtr cacheArg = cache;
            IntPtr outputArg = output;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int startPosArg = startPos;
            int cacheSizeArg = cacheSize;
            int count = checked(numHeads * seqLen * headDim);
            void** args = stackalloc void*[]
            {
                &cacheArg, &outputArg, &numHeadsArg, &seqLenArg, &headDimArg, &startPosArg, &cacheSizeArg
            };
            Launch(gatherCircularHeadFirstF16, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchConcatHeadFirstF32(
            IntPtr a,
            IntPtr b,
            IntPtr output,
            int numHeads,
            int lenA,
            int lenB,
            int headDim,
            IntPtr stream)
        {
            IntPtr aArg = a;
            IntPtr bArg = b;
            IntPtr outputArg = output;
            int numHeadsArg = numHeads;
            int lenAArg = lenA;
            int lenBArg = lenB;
            int headDimArg = headDim;
            int count = checked(numHeads * (lenA + lenB) * headDim);
            void** args = stackalloc void*[] { &aArg, &bArg, &outputArg, &numHeadsArg, &lenAArg, &lenBArg, &headDimArg };
            Launch(concatHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchNeoXRoPEHeadFirstF32(
            IntPtr data,
            IntPtr cosTable,
            IntPtr sinTable,
            int numHeads,
            int seqLen,
            int headDim,
            int ropeHalf,
            IntPtr stream)
        {
            IntPtr dataArg = data;
            IntPtr cosTableArg = cosTable;
            IntPtr sinTableArg = sinTable;
            int numHeadsArg = numHeads;
            int seqLenArg = seqLen;
            int headDimArg = headDim;
            int ropeHalfArg = ropeHalf;
            int count = checked(numHeads * seqLen * ropeHalf);
            void** args = stackalloc void*[]
            {
                &dataArg, &cosTableArg, &sinTableArg, &numHeadsArg, &seqLenArg, &headDimArg, &ropeHalfArg
            };
            Launch(neoxRopeHeadFirstF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchIndexSelectF32(IntPtr source, IntPtr indices, IntPtr output, int rows, int cols, int sourceRows, int indicesAreInt32, int isAdd, IntPtr stream)
        {
            IntPtr sourceArg = source;
            IntPtr indicesArg = indices;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            int sourceRowsArg = sourceRows;
            int indicesAreInt32Arg = indicesAreInt32;
            int isAddArg = isAdd;
            void** args = stackalloc void*[] { &sourceArg, &indicesArg, &outputArg, &rowsArg, &colsArg, &sourceRowsArg, &indicesAreInt32Arg, &isAddArg };
            Launch(indexSelectF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchAddCausalMaskF32(IntPtr tensor, int totalRows, int cols, int seqLen, int startPos, float maskedValue, IntPtr stream)
        {
            IntPtr tensorArg = tensor;
            int totalRowsArg = totalRows;
            int colsArg = cols;
            int seqLenArg = seqLen;
            int startPosArg = startPos;
            float maskedValueArg = maskedValue;
            void** args = stackalloc void*[] { &tensorArg, &totalRowsArg, &colsArg, &seqLenArg, &startPosArg, &maskedValueArg };
            Launch(addCausalMaskF32, (uint)totalRows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchRoPEF32(IntPtr input, IntPtr output, int rows, int cols, int seqLen, int rowOffset, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            int seqLenArg = seqLen;
            int rowOffsetArg = rowOffset;
            int pairCount = cols / 2;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &rowsArg, &colsArg, &seqLenArg, &rowOffsetArg };
            Launch(ropeF32, Grid(rows * pairCount), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchRoPEExF32(
            IntPtr input,
            IntPtr positions,
            IntPtr output,
            int rows,
            int cols,
            int ropeDim,
            int mode,
            int positionsAreInt32,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            int addToResult,
            IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr positionsArg = positions;
            IntPtr outputArg = output;
            int rowsArg = rows;
            int colsArg = cols;
            int ropeDimArg = ropeDim;
            int modeArg = mode;
            int positionsAreInt32Arg = positionsAreInt32;
            int originalContextLengthArg = originalContextLength;
            float freqBaseArg = freqBase;
            float freqScaleArg = freqScale;
            float extFactorArg = extFactor;
            float attnFactorArg = attnFactor;
            float betaFastArg = betaFast;
            float betaSlowArg = betaSlow;
            int addToResultArg = addToResult;
            int pairCount = Math.Min(ropeDim, cols) / 2;
            int copyWork = addToResult == 0 ? rows * cols : rows;
            void** args = stackalloc void*[]
            {
                &inputArg, &positionsArg, &outputArg, &rowsArg, &colsArg, &ropeDimArg, &modeArg,
                &positionsAreInt32Arg, &originalContextLengthArg, &freqBaseArg, &freqScaleArg,
                &extFactorArg, &attnFactorArg, &betaFastArg, &betaSlowArg, &addToResultArg
            };
            Launch(ropeExF32, Grid(Math.Max(rows * pairCount, copyWork)), 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulF32(IntPtr weights, IntPtr input, IntPtr output, int type, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int typeArg = type;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &typeArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            Launch(quantMatmulF32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulIq2XxsQ81F32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            uint sharedBytes = checked((uint)((inDim / 32) * 36));
            Launch(quantMatmulIq2XxsQ81F32, gridX, (uint)rows, 1, BlockSize, 1, 1, sharedBytes, stream, args);
        }

        public void LaunchQuantMatmulQ40F32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            Launch(quantMatmulQ40F32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantMatmulQ80F32(IntPtr weights, IntPtr input, IntPtr output, int inDim, int outDim, int rows, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int inDimArg = inDim;
            int outDimArg = outDim;
            int rowsArg = rows;
            void** args = stackalloc void*[] { &weightsArg, &inputArg, &outputArg, &inDimArg, &outDimArg, &rowsArg };
            uint gridX = (uint)((outDim + 3) / 4);
            if (rows < 4)
                Launch(quantMatmulQ80SingleF32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
            else
                Launch(quantMatmulQ80F32, gridX, (uint)((rows + 3) / 4), 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void LaunchQuantGetRowsF32(IntPtr weights, IntPtr indices, IntPtr output, int type, int cols, int rows, int indicesAreInt32, IntPtr stream)
        {
            IntPtr weightsArg = weights;
            IntPtr indicesArg = indices;
            IntPtr outputArg = output;
            int typeArg = type;
            int colsArg = cols;
            int rowsArg = rows;
            int indicesAreInt32Arg = indicesAreInt32;
            void** args = stackalloc void*[] { &weightsArg, &indicesArg, &outputArg, &typeArg, &colsArg, &rowsArg, &indicesAreInt32Arg };
            Launch(quantGetRowsF32, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args);
        }

        public void Dispose()
        {
            module.Dispose();
        }

        private static void Launch(IntPtr function, uint gx, uint gy, uint gz, int bx, int by, int bz, uint sharedBytes, IntPtr stream, void** args)
        {
            CudaDriverApi.cuLaunchKernel(
                function,
                gx,
                gy,
                gz,
                (uint)bx,
                (uint)by,
                (uint)bz,
                sharedBytes,
                stream,
                (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
        }

        private static uint Grid(int count)
        {
            return (uint)Math.Max(1, (count + BlockSize - 1) / BlockSize);
        }

        private static string LocatePtxPath()
        {
            string baseDirectory = AppContext.BaseDirectory;
            string[] candidates =
            {
                Path.Combine(baseDirectory, "cuda_kernels", "tensorsharp_kernels.ptx"),
                Path.Combine(baseDirectory, "tensorsharp_kernels.ptx"),
                Path.Combine(baseDirectory, "..", "..", "..", "native", "ptx", "tensorsharp_kernels.ptx"),
            };

            foreach (string candidate in candidates)
            {
                string fullPath = Path.GetFullPath(candidate);
                if (File.Exists(fullPath))
                    return fullPath;
            }

            DirectoryInfo dir = new DirectoryInfo(baseDirectory);
            while (dir != null)
            {
                string sourceTreePath = Path.Combine(dir.FullName, "TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_kernels.ptx");
                if (File.Exists(sourceTreePath))
                    return sourceTreePath;

                string localProjectPath = Path.Combine(dir.FullName, "native", "ptx", "tensorsharp_kernels.ptx");
                if (File.Exists(localProjectPath))
                    return localProjectPath;

                dir = dir.Parent;
            }

            return null;
        }
    }
}
