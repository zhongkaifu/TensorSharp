using System;
using System.IO;
using System.Text.Json;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>F32 device projections and transforms for MiniMax-H3.</summary>
public static class CudaMiniMaxH3Ops
{
    private static readonly string LinearTraceFile=Environment.GetEnvironmentVariable("TS_H3_LINEAR_TRACE_FILE");
    // BF16 weights do not justify rounding the F32 activations to TF32. That
    // rounding in H3's condition projection accumulates across all 50 blocks.
    // Keep the historical approximate mode only as an explicit diagnostic.
    private static readonly bool Bf16Pedantic=Environment.GetEnvironmentVariable("TS_H3_CUDA_BF16_PEDANTIC") != "0";
    private static readonly string QuantBandMode=Environment.GetEnvironmentVariable("TS_H3_CUDA_F32_BANDS");
    private static void TraceLinear(string path,int storageType,int rows,int inner,int columns,int compute=0)
    {
        if(string.IsNullOrEmpty(LinearTraceFile))return;
        string directory=Path.GetDirectoryName(Path.GetFullPath(LinearTraceFile));
        Directory.CreateDirectory(directory);
        File.AppendAllText(LinearTraceFile,JsonSerializer.Serialize(new{path,storageType,rows,inner,columns,compute,
            activationStorage="F32",outputStorage="F32"})+Environment.NewLine);
    }
    /// <summary>H3 projections retain raw F32 activations at every row count.
    /// Supported quantized weights use bounded conversion and pedantic F32 GEMM.
    /// TS_H3_CUDA_F32_BANDS=0 selects the original F32 kernels; 1 or tf32 opts
    /// into approximate TF32 products. No activation q8_1 or Half conversion occurs.</summary>
    public static bool TryLinear(Tensor output, Tensor input, IntPtr host, int type, long inner, long columns, long bytes)
    {
        if (type is 0 or 1 or 30) return TryMappedDense(output,input,host,type,inner,columns,bytes);
        if (QuantBandMode != "0" && (type is 2 or 8 or 12 or 14) &&
            TryQuantizedBands(output,input,host,type,inner,columns,bytes,
                forcePedantic:QuantBandMode is not ("1" or "tf32"))) return true;
        if (host == IntPtr.Zero || !CudaQuantizedOps.SupportsQuantizedType(type) || bytes <= 0 ||
            inner <= 0 || columns <= 0 || inner > int.MaxValue || columns > int.MaxValue ||
            !CudaKernelOps.TryGetContiguousFloat(output,out var os,out var yp,out _) ||
            !CudaKernelOps.TryGetContiguousFloat(input,out var xs,out var xp,out _) ||
            input.DimensionCount != 2 || output.DimensionCount != 2 || input.Sizes[1] != inner || output.Sizes[1] != columns ||
            output.Sizes[0] != input.Sizes[0] || input.Sizes[0] <= 0 || !ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl)) return false;
        var allocator=os.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator,out var kernels)) return false;
        allocator.Context.MakeCurrent();
        CudaQuantizedOps.PreloadQuantizedWeight(allocator,host,host,type,inner,columns,bytes);
        if (!CudaQuantizedOps.TryGetResidentDevicePtr(allocator,host,out var resident,out int residentType) || residentType != type) return false;
        xs.EnsureDeviceCurrent();
        int rows=checked((int)input.Sizes[0]), width=checked((int)inner), total=checked((int)columns);
        TraceLinear(type == 8 ? "Q8 scalar F32" : type == 2 ? "Q4_0 scalar F32" : rows >= 2 ? "quant batched F32" : "quant vector F32",
            type,rows,width,total);
        if (type == 8) kernels.LaunchQuantMatmulQ80F32(resident,xp,yp,width,total,rows,allocator.Stream.Handle);
        else if (type == 2) kernels.LaunchQuantMatmulQ40F32(resident,xp,yp,width,total,rows,allocator.Stream.Handle);
        else if (rows >= 2) kernels.LaunchQuantMatmulBatchedF32(resident,xp,yp,type,width,total,rows,allocator.Stream.Handle);
        else kernels.LaunchQuantMatmulVecF32(resident,xp,yp,type,width,total,allocator.Stream.Handle);
        os.MarkDeviceModified();
        return true;
    }

    /// <summary>Bounded Q4_0/Q8_0/Q4_K/Q6_K conversion and GEMM.
    /// Raw weights use the existing lease-owned cache; converted storage is at
    /// most 64MiB. Inputs/results retain F32 storage and range. Pedantic F32
    /// products are the default; TF32 requires forcePedantic=false.</summary>
    public static bool TryQuantizedBands(Tensor output,Tensor input,IntPtr host,int type,long inner,long columns,long bytes,
        int maximumConvertedWeightBytes=64<<20,bool forcePedantic=true)
    {
        int block=type is 2 or 8 ? 32 : type is 12 or 14 ? 256 : 0;
        int blockBytes=type switch {2=>18,8=>34,12=>144,14=>210,_=>0};
        if(host == IntPtr.Zero || block == 0 || inner <= 0 || columns <= 0 || inner > int.MaxValue || columns > int.MaxValue ||
            inner%block != 0 || bytes <= 0 || bytes%columns != 0 || bytes/columns != inner/block*blockBytes ||
            maximumConvertedWeightBytes < sizeof(float)*inner || maximumConvertedWeightBytes > (64<<20) ||
            !CudaKernelOps.TryGetContiguousFloat(output,out var os,out var yp,out _) ||
            !CudaKernelOps.TryGetContiguousFloat(input,out var xs,out var xp,out _) ||
            input.DimensionCount != 2 || output.DimensionCount != 2 || input.Sizes[1] != inner || output.Sizes[1] != columns ||
            output.Sizes[0] != input.Sizes[0] || input.Sizes[0] <= 0 || input.Sizes[0] > int.MaxValue ||
            !ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl)) return false;
        var allocator=os.AllocatorImpl;
        if(!CudaKernelOps.TryGetKernels(allocator,out var kernels) || !kernels.HasH3QuantBandKernel) return false;
        allocator.Context.MakeCurrent();
        CudaQuantizedOps.PreloadQuantizedWeight(allocator,host,host,type,inner,columns,bytes);
        if(!CudaQuantizedOps.TryGetResidentDevicePtr(allocator,host,out var resident,out int residentType) || residentType != type) return false;
        xs.EnsureDeviceCurrent(); os.EnsureDeviceCurrent(); allocator.Blas.SetStream(allocator.Stream.Handle);
        int rows=checked((int)input.Sizes[0]),width=checked((int)inner),total=checked((int)columns);
        int band=checked((int)Math.Min(columns,maximumConvertedWeightBytes/(sizeof(float)*inner)));
        long rowBytes=bytes/columns;
        TraceLinear("quantized bounded F32 bands",type,rows,width,total);
        using var converted=new Tensor(allocator,DType.Float32,band,inner);
        var storage=(CudaStorage)converted.Storage;
        for(int start=0;start<total;start+=band)
        {
            int count=Math.Min(band,total-start);
            IntPtr convertedPointer=storage.DevicePtrAtElement(0);
            kernels.LaunchH3WeightQuantF32(new IntPtr(resident.ToInt64()+start*rowBytes),convertedPointer,type,width,
                checked(count*width),allocator.Stream.Handle);
            storage.MarkDeviceModified();
            GemmMapped(allocator,convertedPointer,xp,new IntPtr(yp.ToInt64()+start*sizeof(float)),rows,width,count,total,type,forcePedantic);
        }
        os.MarkDeviceModified(); return true;
    }

    /// <summary>Mapped F16/BF16/F32 weights stay resident in their original storage.
    /// Only weight slices are converted; inputs/results retain F32 storage and range.
    /// BF16 weights use pedantic F32 by default, preserving activation precision.
    /// Other dense weight types retain their existing GEMM policy.</summary>
    public static bool TryMappedDense(Tensor output,Tensor input,IntPtr host,int type,long inner,long columns,long bytes,
        int maximumConvertedWeightBytes=64<<20)
    {
        if(host == IntPtr.Zero || type is not (0 or 1 or 30) || inner <= 0 || columns <= 0 || inner > int.MaxValue || columns > int.MaxValue ||
            maximumConvertedWeightBytes < sizeof(float) || bytes != checked(inner*columns*(type == 0 ? sizeof(float) : sizeof(ushort))) ||
            !CudaKernelOps.TryGetContiguousFloat(output,out var os,out var yp,out _) ||
            !CudaKernelOps.TryGetContiguousFloat(input,out var xs,out var xp,out _) ||
            input.DimensionCount != 2 || output.DimensionCount != 2 || input.Sizes[1] != inner || output.Sizes[1] != columns ||
            output.Sizes[0] != input.Sizes[0] || !ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl))return false;
        var allocator=os.AllocatorImpl;
        if(!CudaKernelOps.TryGetKernels(allocator,out var kernels) || !kernels.HasH3Kernels)return false;
        allocator.Context.MakeCurrent();
        CudaQuantizedOps.PreloadQuantizedWeight(allocator,host,host,type,inner,columns,bytes);
        if(!CudaQuantizedOps.TryGetResidentDevicePtr(allocator,host,out var resident,out int residentType) || residentType != type)return false;
        xs.EnsureDeviceCurrent(); os.EnsureDeviceCurrent(); allocator.Blas.SetStream(allocator.Stream.Handle);
        int rows=checked((int)input.Sizes[0]), width=checked((int)inner), total=checked((int)columns);
        if(type == 0)
        {
            GemmMapped(allocator,resident,xp,yp,rows,width,total,total,type);
        }
        else
        {
            int band=checked((int)Math.Max(1,Math.Min(columns,maximumConvertedWeightBytes/(sizeof(float)*inner))));
            using var converted=new Tensor(allocator,DType.Float32,band,inner);
            var storage=(CudaStorage)converted.Storage;
            for(int start=0;start<total;start+=band)
            {
                int count=Math.Min(band,total-start);
                IntPtr convertedPointer=storage.DevicePtrAtElement(0);
                IntPtr bandPointer=new IntPtr(resident.ToInt64()+start*inner*sizeof(ushort));
                if(type == 30) kernels.LaunchH3WeightBf16F32(bandPointer,convertedPointer,checked(count*width),allocator.Stream.Handle);
                else kernels.LaunchH3WeightF16F32(bandPointer,convertedPointer,checked(count*width),allocator.Stream.Handle);
                storage.MarkDeviceModified();
                // ldc retains the full output width while each GEMM fills only
                // this band of columns. No strided-copy or output scratch is needed.
                GemmMapped(allocator,convertedPointer,xp,new IntPtr(yp.ToInt64()+start*sizeof(float)),rows,width,count,total,type);
            }
        }
        os.MarkDeviceModified();return true;
    }

    private static void GemmMapped(CudaAllocator allocator,IntPtr weight,IntPtr input,IntPtr output,int rows,int inner,int columns,int outputStride,int storageType,bool forcePedantic=false)
    {
        float alpha=1,beta=0;
        // CUDA 13.4 cublas_api.h:208-212: 69 = COMPUTE_32F_PEDANTIC,
        // 77 = COMPUTE_32F_FAST_TF32. This choice is per-call; shared math mode
        // remains unchanged. TF32 retains F32's exponent range.
        int compute=forcePedantic || (long)rows*inner*columns <= 16_000_000 || (storageType == 30 && Bf16Pedantic) ? 69 : 77;
        TraceLinear("dense F32 GEMM",storageType,rows,inner,columns,compute);
        CublasApi.cublasGemmEx(allocator.Blas.Handle,CublasApi.CUBLAS_OP_T,CublasApi.CUBLAS_OP_N,
            columns,rows,inner,ref alpha,weight,CublasApi.CUDA_R_32F,inner,input,CublasApi.CUDA_R_32F,inner,ref beta,
            output,CublasApi.CUDA_R_32F,outputStride,compute,CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
    }

    /// <summary>Unmasked attention over token-major [sequence, heads*headDim]
    /// tensors, using bounded score chunks and pedantic F32 batched GEMMs.
    /// Q/K/V are consumed with their existing head strides, without head copies.
    /// Full F32 products retain H3's activation range; reduction order differs
    /// from streaming attention, so this remains a diagnostic candidate.</summary>
    public static bool TryBatchedAttention(Tensor output, Tensor query, Tensor key, Tensor value,
        int heads, int headDim, float scale, int maximumScoreBytes = 64 << 20, int maximumQueryRows = 0)
    {
        long width = (long)heads * headDim;
        if (heads <= 0 || headDim <= 0 || width > int.MaxValue || !float.IsFinite(scale) || scale <= 0 ||
            maximumScoreBytes <= 0 || maximumScoreBytes > (64 << 20) || maximumQueryRows < 0 ||
            !CudaKernelOps.TryGetContiguousFloat(output, out var os, out var yp, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(query, out var qs, out var qp, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(key, out var ks, out var kp, out _) ||
            !CudaKernelOps.TryGetContiguousFloat(value, out var vs, out var vp, out _) ||
            output.DimensionCount != 2 || query.DimensionCount != 2 || key.DimensionCount != 2 || value.DimensionCount != 2 ||
            output.Sizes[0] != query.Sizes[0] || key.Sizes[0] != value.Sizes[0] ||
            query.Sizes[0] <= 0 || query.Sizes[0] > int.MaxValue || key.Sizes[0] <= 0 || key.Sizes[0] > int.MaxValue ||
            output.Sizes[1] != width || query.Sizes[1] != width || key.Sizes[1] != width || value.Sizes[1] != width ||
            !ReferenceEquals(os.AllocatorImpl, qs.AllocatorImpl) || !ReferenceEquals(os.AllocatorImpl, ks.AllocatorImpl) ||
            !ReferenceEquals(os.AllocatorImpl, vs.AllocatorImpl) || ReferenceEquals(os, qs) || ReferenceEquals(os, ks) || ReferenceEquals(os, vs))
            return false;
        int sequences = checked((int)query.Sizes[0]), keys = checked((int)key.Sizes[0]), inner = checked((int)width);
        long rowBytes = sizeof(float) * (long)heads * keys;
        if (rowBytes > maximumScoreBytes) return false;
        int chunk = (int)Math.Min(sequences, maximumScoreBytes / rowBytes);
        if (maximumQueryRows > 0) chunk = Math.Min(chunk, maximumQueryRows);
        var allocator = os.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out _)) return false;
        allocator.Context.MakeCurrent();
        qs.EnsureDeviceCurrent(); ks.EnsureDeviceCurrent(); vs.EnsureDeviceCurrent();
        // A contiguous output may be a subview of a larger host-dirty storage.
        // Preserve its untouched prefix/suffix before marking the storage current.
        os.EnsureDeviceCurrent();
        allocator.Blas.SetStream(allocator.Stream.Handle);
        const int pedanticFloat32 = 69; // CUBLAS_COMPUTE_32F_PEDANTIC, independent of shared math mode.
        float zero = 0, one = 1;
        for (int start = 0; start < sequences; start += chunk)
        {
            int rows = Math.Min(chunk, sequences - start);
            using var scores = new Tensor(allocator, DType.Float32, checked(heads * rows), keys);
            var ss = (CudaStorage)scores.Storage;
            IntPtr sp = ss.DevicePtrAtElement(0);
            // Each head's K is [headDim, keys] column-major with lda=inner;
            // scores are [keys, rows] column-major, then flattened for softmax.
            CublasApi.cublasGemmStridedBatchedEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                keys, rows, headDim, ref scale, kp, CublasApi.CUDA_R_32F, inner, headDim,
                new IntPtr(qp.ToInt64() + (long)start * inner * sizeof(float)), CublasApi.CUDA_R_32F, inner, headDim,
                ref zero, sp, CublasApi.CUDA_R_32F, keys, (long)rows * keys, heads,
                pedanticFloat32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            ss.MarkDeviceModified();
            if (!CudaKernelOps.TrySoftmax(scores, scores))
                throw new NotSupportedException("H3 batched attention requires the CUDA F32 softmax kernel.");
            // A head writes only its headDim elements in each token row. The
            // interleaved C matrices share padding, never output elements.
            CublasApi.cublasGemmStridedBatchedEx(allocator.Blas.Handle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                headDim, rows, keys, ref one, vp, CublasApi.CUDA_R_32F, inner, headDim,
                sp, CublasApi.CUDA_R_32F, keys, (long)rows * keys,
                ref zero, new IntPtr(yp.ToInt64() + (long)start * inner * sizeof(float)), CublasApi.CUDA_R_32F, inner, headDim,
                heads, pedanticFloat32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }
        os.MarkDeviceModified();
        return true;
    }

    /// <summary>Copy F32 permutations of up to four dimensions on the device.</summary>
    /// <summary>Any-stride copy of 4-byte elements (F32, or Int32 moved bit for bit: rotary positions broadcast to heads).</summary>
    public static bool TryCopyStrided(Tensor output,Tensor input)
    {
        if(output?.Storage is not CudaStorage os || input?.Storage is not CudaStorage xs ||
            output.ElementType is not (DType.Float32 or DType.Int32) || input.ElementType != output.ElementType ||
            output.DimensionCount != input.DimensionCount || output.DimensionCount is < 1 or > 4 ||
            output.ElementCount() != input.ElementCount() || output.ElementCount() > int.MaxValue ||
            !ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl) || ReferenceEquals(os,xs)) return false;
        long[] sizes=[1,1,1,1], source=[0,0,0,0], target=[0,0,0,0];
        for(int d=0;d<input.DimensionCount;d++)
        {
            if(input.Sizes[d] != output.Sizes[d]) return false;
            int axis=4-input.DimensionCount+d;
            sizes[axis]=input.Sizes[d]; source[axis]=input.Strides[d]; target[axis]=output.Strides[d];
        }
        if(input.ElementCount()==0)return true;
        var allocator=os.AllocatorImpl;
        if(!CudaKernelOps.TryGetKernels(allocator,out var kernels) || !kernels.HasH3Kernels)return false;
        allocator.Context.MakeCurrent(); xs.EnsureDeviceCurrent(); os.EnsureDeviceCurrent();
        kernels.LaunchH3CopyStrided(xs.DevicePtrAtElement(input.StorageOffset),os.DevicePtrAtElement(output.StorageOffset),
            checked((int)input.ElementCount()),sizes[0],sizes[1],sizes[2],sizes[3],source[0],source[1],source[2],source[3],
            target[0],target[1],target[2],target[3],allocator.Stream.Handle);
        os.MarkDeviceModified();return true;
    }

    public static bool TryClamp(Tensor output,Tensor input,float lower,float upper)
    {
        if(lower>upper || !CudaKernelOps.TryGetContiguousFloat(output,out var os,out var yp,out int count) ||
            !CudaKernelOps.TryGetContiguousFloat(input,out var xs,out var xp,out int inputCount) || count != inputCount ||
            !ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl))return false;
        var allocator=os.AllocatorImpl;
        if(!CudaKernelOps.TryGetKernels(allocator,out var kernels) || !kernels.HasH3Kernels)return false;
        allocator.Context.MakeCurrent(); xs.EnsureDeviceCurrent(); os.EnsureDeviceCurrent();
        kernels.LaunchH3Clamp(xp,yp,count,lower,upper,allocator.Stream.Handle);
        os.MarkDeviceModified();return true;
    }

    /// <summary>F32 input, accumulation and output, without TF32 rounding or
    /// changes to the allocator's shared cuBLAS math mode. Weight is [out,in].</summary>
    public static bool TryMatmul(Tensor output, Tensor input, Tensor weight, float beta = 0)
    {
        if (output?.Storage is not CudaStorage os || input?.Storage is not CudaStorage xs || weight?.Storage is not CudaStorage ws ||
            output.ElementType != DType.Float32 || input.ElementType != DType.Float32 || weight.ElementType != DType.Float32 ||
            !output.IsContiguous() || !input.IsContiguous() || !weight.IsContiguous() ||
            output.DimensionCount != 2 || input.DimensionCount != 2 || weight.DimensionCount != 2 ||
            output.Sizes[0] != input.Sizes[0] || output.Sizes[1] != weight.Sizes[0] || input.Sizes[1] != weight.Sizes[1] ||
            !ReferenceEquals(os.AllocatorImpl,xs.AllocatorImpl) || !ReferenceEquals(os.AllocatorImpl,ws.AllocatorImpl)) return false;
        var allocator=os.AllocatorImpl;
        allocator.Context.MakeCurrent();
        xs.EnsureDeviceCurrent(); ws.EnsureDeviceCurrent(); os.EnsureDeviceCurrent();
        allocator.Blas.SetStream(allocator.Stream.Handle);
        float alpha=1;
        int rows=checked((int)input.Sizes[0]), inner=checked((int)input.Sizes[1]), columns=checked((int)weight.Sizes[0]);
        // CUDA 13.4 cublas_api.h:208-212: CUBLAS_COMPUTE_32F_PEDANTIC = 69.
        CublasApi.cublasGemmEx(allocator.Blas.Handle,CublasApi.CUBLAS_OP_T,CublasApi.CUBLAS_OP_N,
            columns,rows,inner,ref alpha,ws.DevicePtrAtElement(weight.StorageOffset),CublasApi.CUDA_R_32F,inner,
            xs.DevicePtrAtElement(input.StorageOffset),CublasApi.CUDA_R_32F,inner,ref beta,
            os.DevicePtrAtElement(output.StorageOffset),CublasApi.CUDA_R_32F,columns,69,CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        os.MarkDeviceModified(); return true;
    }

    public static bool TryTransform(Tensor output, Tensor input, Tensor a, Tensor b, int op,
        int p0, int p1 = 0, int p2 = 0, int p3 = 0, int p4 = 0, int p5 = 0,
        int p6 = 0, int p7 = 0, int p8 = 0, int p9 = 0)
    {
        if (!CudaKernelOps.TryGetContiguousFloat(output, out var os, out var yp, out int count) ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var xp, out _) ||
            !ReferenceEquals(os.AllocatorImpl, xs.AllocatorImpl)) return false;
        IntPtr ap = IntPtr.Zero, bp = IntPtr.Zero;
        if (a != null && (!CudaKernelOps.TryGetContiguousFloat(a, out var ast, out ap, out _) ||
            !ReferenceEquals(os.AllocatorImpl, ast.AllocatorImpl))) return false;
        if (b != null && (!CudaKernelOps.TryGetContiguousFloat(b, out var bst, out bp, out _) ||
            !ReferenceEquals(os.AllocatorImpl, bst.AllocatorImpl))) return false;
        var allocator = os.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasH3Kernels) return false;
        xs.EnsureDeviceCurrent();
        (a?.Storage as CudaStorage)?.EnsureDeviceCurrent(); (b?.Storage as CudaStorage)?.EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        kernels.LaunchH3Transform(xp, ap, bp, yp, count, op, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, allocator.Stream.Handle);
        os.MarkDeviceModified(); return true;
    }

    public static bool TryGroupNormSilu(Tensor output, Tensor input, Tensor gain, Tensor bias, int groups, float eps)
    {
        if (!CudaKernelOps.TryGetContiguousFloat(output, out var os, out var yp, out int count) ||
            !CudaKernelOps.TryGetContiguousFloat(input, out var xs, out var xp, out int inputCount) ||
            input.DimensionCount != 4 || output.DimensionCount != 4 || count <= 0 || count != inputCount ||
            groups <= 0 || input.Sizes[3] % groups != 0 || !float.IsFinite(eps) || eps <= 0 ||
            !ReferenceEquals(os.AllocatorImpl, xs.AllocatorImpl)) return false;
        for (int dimension=0; dimension<4; ++dimension)
            if (input.Sizes[dimension] != output.Sizes[dimension]) return false;
        IntPtr gp = IntPtr.Zero, bp = IntPtr.Zero;
        if (gain != null && (!CudaKernelOps.TryGetContiguousFloat(gain, out var gs, out gp, out int gainCount) ||
            gain.DimensionCount != 1 || gainCount != input.Sizes[3] || !ReferenceEquals(os.AllocatorImpl,gs.AllocatorImpl))) return false;
        if (bias != null && (!CudaKernelOps.TryGetContiguousFloat(bias, out var bs, out bp, out int biasCount) ||
            bias.DimensionCount != 1 || biasCount != input.Sizes[3] || !ReferenceEquals(os.AllocatorImpl,bs.AllocatorImpl))) return false;
        var allocator = os.AllocatorImpl;
        if (!CudaKernelOps.TryGetKernels(allocator, out var kernels) || !kernels.HasH3Kernels) return false;
        xs.EnsureDeviceCurrent(); (gain?.Storage as CudaStorage)?.EnsureDeviceCurrent(); (bias?.Storage as CudaStorage)?.EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        kernels.LaunchH3GroupNorm(xp, gp, bp, yp, checked((int)input.Sizes[0]), checked((int)(input.Sizes[1]*input.Sizes[2])),
            checked((int)input.Sizes[3]), groups, eps, allocator.Stream.Handle);
        os.MarkDeviceModified(); return true;
    }
}
