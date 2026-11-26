// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Core;

namespace TensorSharp.MLX;

/// <summary>Device-resident F32 video/audio VAE transforms.</summary>
public static class MlxMiniMaxH3Ops
{
    /// <summary>Copy an immutable H3 weight directly into an owned F32 MLX array.
    /// Unlike a host write followed by upload, this does not retain a second host
    /// buffer. Ordinary storage reads can still materialize a host copy later.</summary>
    public static Tensor CreateFloat32Weight(MlxAllocator allocator, float[] values, long rows, long columns)
    {
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentNullException.ThrowIfNull(values);
        if (rows <= 0 || columns <= 0 || rows > int.MaxValue || columns > int.MaxValue ||
            rows * columns != values.LongLength)
            throw new ArgumentException("H3 MLX weight values must match positive matrix dimensions.");

        return MlxWorker.Shared.Invoke(() =>
        {
            var tensor = new Tensor(allocator, DType.Float32, rows, columns);
            MlxNative.MlxArray array = default;
            try
            {
                // NewArrayFromHost copies before returning. Do not replace it
                // with the no-copy variant: the caller owns the source array.
                unsafe
                {
                    fixed (float* source = values)
                        array = MlxNative.NewArrayFromHost((IntPtr)source, [values.Length], DType.Float32);
                }
                ((MlxStorage)tensor.Storage).ReplaceDeviceArray(array);
                array = default;
                return tensor;
            }
            catch { tensor.Dispose(); throw; }
            finally { MlxNative.FreeArray(array); }
        });
    }

    public static bool TryClamp(Tensor output, Tensor input, float lower, float upper)
    {
        if (lower > upper || output?.Storage is not MlxStorage os || input?.Storage is not MlxStorage xs ||
            output.ElementType != DType.Float32 || input.ElementType != DType.Float32 || !output.IsContiguous() ||
            output.DimensionCount != input.DimensionCount || output.ElementCount() > int.MaxValue ||
            !ReferenceEquals(os.AllocatorImpl, xs.AllocatorImpl)) return false;
        for (int dimension=0; dimension<input.DimensionCount; dimension++)
            if (output.Sizes[dimension] != input.Sizes[dimension]) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray view=default, contiguous=default, minScalar=default, maxScalar=default, min=default, max=default, result=default;
            try
            {
                view=xs.CreateArrayView(input);
                contiguous=MlxNative.Contiguous(view);
                minScalar=MlxNative.NewScalar(lower); maxScalar=MlxNative.NewScalar(upper);
                min=MlxNative.Reshape(minScalar,[1]); max=MlxNative.Reshape(maxScalar,[1]);
                result=MlxNative.H3Transform(contiguous,min,max,Array.ConvertAll(output.Sizes.ToArray(), v => checked((int)v)),6,
                    [1,1,1,1,1,1,1,1,1,1]);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { os.ReplaceDeviceArray(result); result=default; }
                else os.UpdateDeviceSlice(output,result);
                return true;
            }
            finally
            {
                MlxNative.FreeArray(view); MlxNative.FreeArray(contiguous); MlxNative.FreeArray(minScalar);
                MlxNative.FreeArray(maxScalar); MlxNative.FreeArray(min);
                MlxNative.FreeArray(max); MlxNative.FreeArray(result);
            }
        });
    }

    public static bool TryTransform(Tensor output, Tensor input, Tensor a, Tensor b, int op,
        int p0, int p1=0, int p2=0, int p3=0, int p4=0, int p5=0,
        int p6=0, int p7=0, int p8=0, int p9=0)
    {
        if (output?.Storage is not MlxStorage os || input?.Storage is not MlxStorage xs ||
            output.ElementType != DType.Float32 || input.ElementType != DType.Float32 ||
            !output.IsContiguous() || !input.IsContiguous() ||
            !ReferenceEquals(os.AllocatorImpl, xs.AllocatorImpl) ||
            (a != null && (a.Storage is not MlxStorage || a.ElementType != DType.Float32)) ||
            (b != null && (b.Storage is not MlxStorage || b.ElementType != DType.Float32))) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x=default, av=default, bv=default, result=default;
            try
            {
                x=xs.CreateArrayView(input);
                av=((MlxStorage)(a ?? input).Storage).CreateArrayView(a ?? input);
                bv=((MlxStorage)(b ?? input).Storage).CreateArrayView(b ?? input);
                result=MlxNative.H3Transform(x, av, bv, Array.ConvertAll(output.Sizes.ToArray(), v => checked((int)v)), op,
                    [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9]);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { os.ReplaceDeviceArray(result); result=default; }
                else os.UpdateDeviceSlice(output, result);
                return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(av); MlxNative.FreeArray(bv); MlxNative.FreeArray(result); }
        });
    }

    public static bool TryGroupNormSilu(Tensor output, Tensor input, Tensor gain, Tensor bias, int groups, float eps)
    {
        if (input?.Storage is not MlxStorage xs || output?.Storage is not MlxStorage os ||
            input.ElementType != DType.Float32 || output.ElementType != DType.Float32 ||
            input.DimensionCount != 4 || output.DimensionCount != 4 || !input.IsContiguous() || !output.IsContiguous() ||
            !ReferenceEquals(xs.AllocatorImpl, os.AllocatorImpl) || groups <= 0 || !float.IsFinite(eps) || eps <= 0 ||
            input.ElementCount() <= 0 || input.ElementCount() > int.MaxValue || input.Sizes[3] % groups != 0) return false;
        for (int dimension=0; dimension<4; ++dimension)
            if (input.Sizes[dimension] != output.Sizes[dimension]) return false;
        int channels=checked((int)input.Sizes[3]);
        if (channels <= 0 || !ValidVector(gain) || !ValidVector(bias)) return false;
        // Narrow groups traverse NHWC with poorly coalesced loads. Retain MLX's
        // contiguous reduction for large planes in this geometry. H3's shipped
        // encoder starts at 128 channels and uses the fused path throughout.
        if (channels < 128 && input.Sizes[1]*input.Sizes[2] >= 4096)
            return GroupNormSiluExpression(output,input,gain,bias,groups,eps);
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x=default, g=default, b=default, result=default;
            try
            {
                x=xs.CreateArrayView(input);
                g=((MlxStorage)(gain ?? input).Storage).CreateArrayView(gain ?? input);
                b=((MlxStorage)(bias ?? input).Storage).CreateArrayView(bias ?? input);
                result=MlxNative.H3GroupNormSilu(x,g,b,checked((int)input.Sizes[0]),
                    checked((int)(input.Sizes[1]*input.Sizes[2])),channels,groups,eps,gain != null,bias != null);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { os.ReplaceDeviceArray(result); result=default; }
                else os.UpdateDeviceSlice(output,result);
                return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(g); MlxNative.FreeArray(b); MlxNative.FreeArray(result); }
        });

        bool ValidVector(Tensor vector) => vector == null || (vector.Storage is MlxStorage storage &&
            vector.ElementType == DType.Float32 && vector.DimensionCount == 1 && vector.Sizes[0] == channels &&
            vector.IsContiguous() && ReferenceEquals(xs.AllocatorImpl,storage.AllocatorImpl));
    }

    private static bool GroupNormSiluExpression(Tensor output,Tensor input,Tensor gain,Tensor bias,int groups,float eps)
    {
        long frames=input.Sizes[0],rows=input.Sizes[1]*input.Sizes[2],channels=input.Sizes[3],width=channels/groups;
        using var shaped=input.View(frames,rows,groups,width);
        using var permuted=shaped.Permute(0,2,1,3);
        using var contiguous=Ops.NewContiguous(permuted);
        using var normalizedInput=contiguous.View(frames*groups,rows*width);
        using var ones=new Tensor(input.Allocator,DType.Float32,rows*width); Ops.Fill(ones,1);
        using var normalized=Ops.LayerNorm(null,normalizedInput,ones,null,eps);
        using var groupsView=normalized.View(frames,groups,rows,width);
        using var restored=groupsView.Permute(0,2,1,3);
        using var restoredContiguous=Ops.NewContiguous(restored);
        using var values=restoredContiguous.View(frames*rows,channels);
        if(gain != null) { using var g=gain.View(1,channels); Ops.Mul(values,values,g); }
        if(bias != null) { using var b=bias.View(1,channels); Ops.Add(values,values,b); }
        using var activated=Ops.Sigmoid(null,values); Ops.Mul(values,values,activated);
        using var result=values.View(input.Sizes); Ops.Copy(output,result);
        return true;
    }
}

[OpsClass]
public sealed class MlxH3ClampOps
{
    [RegisterOpStorageType("clamp", typeof(MlxStorage))]
    public static Tensor Clamp(Tensor result, Tensor src, float lower, float upper)
    {
        Tensor output=TensorResultBuilder.GetWriteTarget(result,src,true,src.Sizes);
        if (MlxMiniMaxH3Ops.TryClamp(output,src,lower,upper)) return output;
        throw new NotSupportedException("MLX F32 clamp requires matching tensors and a contiguous output view.");
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel h3TransformKernel;
    private static MlxFastMetalKernel h3GroupNormSiluKernel;

    // Each group belongs to one frame, as in upstream H3 TemporalGroupNorm.
    // Work directly in NHWC so normalization, affine and SiLU need no layout
    // copies. Reduce centered values twice to retain small variance around a
    // large F32 offset; E[x*x]-E[x]*E[x] loses that variance by cancellation.
    private const string H3GroupNormSiluSource="""
    const uint tid=thread_position_in_threadgroup.x;
    const uint lane=tid & 31u, simd=tid >> 5;
    const uint group=threadgroup_position_in_grid.x;
    const uint frame=group/Groups, channelStart=(group%Groups)*Width;
    const uint count=Rows*Width, base=frame*Rows*Channels+channelStart;
    threadgroup float partial[8];
    const float anchor=x[base];
    float sum=0.0f;
    for (uint i=tid; i<count; i+=256)
        sum+=x[base+(i/Width)*Channels+i%Width]-anchor;
    float total=simd_sum(sum);
    if (lane == 0) partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane < 8u ? partial[lane] : 0.0f);
    const float mean=anchor+total/float(count);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float squares=0.0f;
    for (uint i=tid; i<count; i+=256) {
        const float centered=x[base+(i/Width)*Channels+i%Width]-mean;
        squares+=centered*centered;
    }
    total=simd_sum(squares);
    if (lane == 0) partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane < 8u ? partial[lane] : 0.0f);
    const float inverse=rsqrt(total/float(count)+as_type<float>(uint(EpsBits)));
    for (uint i=tid; i<count; i+=256) {
        const uint channel=channelStart+i%Width;
        const uint index=base+(i/Width)*Channels+i%Width;
        float value=(x[index]-mean)*inverse;
        if (HasGain) value*=gain[channel];
        if (HasBias) value+=bias[channel];
        y[index]=value/(1.0f+exp(-value));
    }
    """;

    internal static MlxArray H3GroupNormSilu(MlxArray input,MlxArray gain,MlxArray bias,
        int frames,int rows,int channels,int groups,float eps,bool hasGain,bool hasBias)
    {
        lock(fastKernelSync)
            if (!h3GroupNormSiluKernel.IsValid)
                h3GroupNormSiluKernel=CreateFastMetalKernel("tensorsharp_h3_groupnorm_silu_f32",["x","gain","bias"],["y"],
                    H3GroupNormSiluSource,string.Empty);
        var config=mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs=default,outputs=default;
        try
        {
            AddTemplateInt(config,"Rows",rows); AddTemplateInt(config,"Channels",channels);
            AddTemplateInt(config,"Groups",groups); AddTemplateInt(config,"Width",channels/groups);
            AddTemplateInt(config,"EpsBits",BitConverter.SingleToInt32Bits(eps));
            AddTemplateInt(config,"HasGain",hasGain ? 1 : 0); AddTemplateInt(config,"HasBias",hasBias ? 1 : 0);
            int[] shape=[frames,rows,channels];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config,shape,(nuint)shape.Length,ToMlxDtype(DType.Float32)),"configuring H3 group norm output");
            Check(mlx_fast_metal_kernel_config_set_grid(config,checked(frames*groups*256),1,1),"configuring H3 group norm grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config,256,1,1),"configuring H3 group norm threadgroup");
            inputs=CreateVectorArray(input,gain,bias); outputs=mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs,h3GroupNormSiluKernel,inputs,config,DefaultStream()),"running H3 group norm");
            Check(mlx_vector_array_get(out MlxArray result,outputs,0),"reading H3 group norm");
            return result;
        }
        finally
        {
            if(inputs.IsValid) _=mlx_vector_array_free(inputs);
            if(outputs.IsValid) _=mlx_vector_array_free(outputs);
            if(config.IsValid) _=mlx_fast_metal_kernel_config_free(config);
        }
    }
    private const string H3TransformSource="""
    const uint i = thread_position_in_grid.x;
    const int count=Count, op=Op, p0=P0, p1=P1, p2=P2, p3=P3, p4=P4, p5=P5, p6=P6, p7=P7, p8=P8, p9=P9;
    if (i >= count) return;
    if (op == 0) { // reflect im2col: x[H,W,C], y[bandRows*OW,C*K*K]
        int dim = p0*p3*p3, row = i/dim, tap = i%dim;
        int channel = tap/(p3*p3), ky = (tap/p3)%p3, kx = tap%p3;
        int sy = (p8 + row/p4)*p7 - p6 + ky, sx = (row%p4)*p7 - p5 + kx;
        sy = p1 <= 1 ? 0 : (sy < 0 ? -sy : (sy >= p1 ? 2*(p1-1)-sy : sy));
        sx = p2 <= 1 ? 0 : (sx < 0 ? -sx : (sx >= p2 ? 2*(p2-1)-sx : sx));
        y[i] = x[(sy*p2+sx)*p0+channel];
    } else if (op == 1) { // audio im2col: x[T,C], y[band,C*K]
        int dim = p0*p2, row = i/dim+p6, tap = i%dim, channel = tap/p2, k = tap%p2;
        int st = p7 ? row-k : row*p3-p4+k*p5;
        bool valid = !p7 || st%p3 == 0;
        if (p7) st /= p3;
        y[i] = valid && st >= 0 && st < p1 ? x[st*p0+channel] : 0.0f;
    } else if (op == 2) { // audio shared filter, optional zero-stuff + replicate
        int ch = i%p0, ot = i/p0;
        float sum = 0.0f;
        for (int k = 0; k < p2; ++k) {
            int st = ot*p3+p7-p4+k;
            if (p5) { // upsample: pad source before zero-stuff, zero beyond it
                if (st < 0 || st >= (p1+2*p6)*2 || st%2) continue;
                st = st/2-p6;
            }
            st = max(0, min(p1-1, st));
            sum += a[k]*x[st*p0+ch];
        }
        y[i] = sum;
    } else if (op == 3) { // Snake / SnakeBeta; b is the precomputed reciprocal
        int ch = i%p0;
        float value = x[i], s = sin(a[ch]*value);
        y[i] = value + (s*s)*b[ch];
    } else if (op == 4) { // mean over heads, then adaptive mean over head dim
        int row = i/p2, channel = i%p2, group = p1/p2;
        float pooled = 0.0f;
        for (int j = 0; j < group; ++j) {
            float heads = 0.0f;
            for (int h = 0; h < p0; ++h) heads += x[row*p0*p1+h*p1+channel*group+j];
            pooled += heads/p0;
        }
        y[i] = pooled/group;
    } else if (op == 5) { // exact-form GELU with the managed A&S erf polynomial
        const float value=x[i], scaled=value*0.7071067811865475f;
        const float absolute=abs(scaled), t=1.0f/(1.0f+0.3275911f*absolute);
        const float polynomial=((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
        const float sign=scaled > 0.0f ? 1.0f : (scaled < 0.0f ? -1.0f : 0.0f);
        const float erf=sign*(1.0f-polynomial*exp(-absolute*absolute));
        y[i]=value*(0.5f*(1.0f+erf));
    } else if (op == 6) {
        y[i]=clamp(x[i],a[0],b[0]);
    }
""";
    internal static MlxArray H3Transform(MlxArray input, MlxArray a, MlxArray b, int[] shape, int op, int[] parameters)
    {
        int count=1; foreach(int dimension in shape) count=checked(count*dimension);
        lock(fastKernelSync)
            if (!h3TransformKernel.IsValid)
                h3TransformKernel=CreateFastMetalKernel("tensorsharp_h3_transform_f32",["x","a","b"],["y"],H3TransformSource,string.Empty);
        var config=mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs=default, outputs=default;
        try
        {
            AddTemplateInt(config,"Count",count); AddTemplateInt(config,"Op",op);
            for(int i=0;i<parameters.Length;i++) AddTemplateInt(config,"P"+i,parameters[i]);
            Check(mlx_fast_metal_kernel_config_add_output_arg(config,shape,(nuint)shape.Length,ToMlxDtype(DType.Float32)),"configuring H3 transform output");
            Check(mlx_fast_metal_kernel_config_set_grid(config,count,1,1),"configuring H3 transform grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config,256,1,1),"configuring H3 transform threadgroup");
            inputs=CreateVectorArray(input,a,b); outputs=mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs,h3TransformKernel,inputs,config,DefaultStream()),"running H3 transform");
            Check(mlx_vector_array_get(out MlxArray result,outputs,0),"reading H3 transform");
            return result;
        }
        finally
        {
            if(inputs.IsValid) _=mlx_vector_array_free(inputs);
            if(outputs.IsValid) _=mlx_vector_array_free(outputs);
            if(config.IsValid) _=mlx_fast_metal_kernel_config_free(config);
        }
    }
}
