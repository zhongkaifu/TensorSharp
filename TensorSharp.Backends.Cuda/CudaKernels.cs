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
        private readonly IntPtr unaryF32;
        private readonly IntPtr binaryActivationF32;
        private readonly IntPtr siluMulSplitF32;
        private readonly IntPtr rmsNormF32;
        private readonly IntPtr softmaxF32;
        private readonly IntPtr indexSelectF32;
        private readonly IntPtr addCausalMaskF32;
        private readonly IntPtr ropeF32;
        private readonly IntPtr ropeExF32;
        private readonly IntPtr quantMatmulF32;
        private readonly IntPtr quantMatmulQ80F32;
        private readonly IntPtr quantGetRowsF32;

        private CudaKernels(CudaModule module)
        {
            this.module = module;
            fillF32 = module.GetFunction("ts_fill_f32");
            unaryF32 = module.GetFunction("ts_unary_f32");
            binaryActivationF32 = module.GetFunction("ts_binary_activation_f32");
            siluMulSplitF32 = module.GetFunction("ts_silu_mul_split_f32");
            rmsNormF32 = module.GetFunction("ts_rmsnorm_f32");
            softmaxF32 = module.GetFunction("ts_softmax_f32");
            indexSelectF32 = module.GetFunction("ts_index_select_f32");
            addCausalMaskF32 = module.GetFunction("ts_add_causal_mask_f32");
            ropeF32 = module.GetFunction("ts_rope_f32");
            ropeExF32 = module.GetFunction("ts_rope_ex_f32");
            quantMatmulF32 = module.GetFunction("ts_quant_matmul_f32");
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

        public void LaunchUnaryF32(IntPtr input, IntPtr output, int count, int op, IntPtr stream)
        {
            IntPtr inputArg = input;
            IntPtr outputArg = output;
            int countArg = count;
            int opArg = op;
            void** args = stackalloc void*[] { &inputArg, &outputArg, &countArg, &opArg };
            Launch(unaryF32, Grid(count), 1, 1, BlockSize, 1, 1, 0, stream, args);
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
            Launch(quantMatmulQ80F32, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, args);
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
