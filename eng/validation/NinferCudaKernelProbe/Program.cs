// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

// Synthetic component checks. No model weights, token quality or performance are certified.
if(args.Length!=2)throw new ArgumentException("Usage: NinferCudaKernelProbe MODULE_PATH OUTPUT_JSON");
CudaBackend.Register();
CudaDriverApi.cuInit(0).ThrowOnError();
using var device=CudaContext.Create(0);
using var kernels=CudaNinferKernels.Create(args[0]);
var results=new List<object>();
using var memory=new Memory();

void Check(string name,float[] actual,float[] expected,bool exact=true,float absolute=0,float relative=0)
{
    if(actual.Length!=expected.Length)throw new Exception(name+": different sizes");
    double max=0; int mismatch=0;
    for(int i=0;i<actual.Length;i++)
    {
        if(!float.IsFinite(actual[i])||!float.IsFinite(expected[i]))throw new Exception(name+": nonfinite at "+i);
        double error=Math.Abs((double)actual[i]-expected[i]);max=Math.Max(max,error);
        bool equal=exact?BitConverter.SingleToInt32Bits(actual[i])==BitConverter.SingleToInt32Bits(expected[i]):error<=absolute+relative*Math.Abs(expected[i]);
        if(!equal)mismatch++;
    }
    results.Add(new{name,count=actual.Length,exact,absolute,relative,maxAbsoluteError=max,mismatch,passed=mismatch==0});
    if(mismatch!=0)throw new Exception($"{name}: {mismatch} mismatches, max {max}");
}
float[] Download(IntPtr p,int n)=>memory.Read<float>(p,n);

void CheckBounded(string name,float[] actual,double[] expected,double[] bounds,string boundPolicy,bool requireBf16)
{
    if(actual.Length!=expected.Length||bounds.Length!=expected.Length)throw new Exception(name+": inconsistent sizes");
    double maxError=0,maxBound=0,maxFraction=0;int mismatches=0;
    for(int i=0;i<actual.Length;i++)
    {
        if(!float.IsFinite(actual[i])||!double.IsFinite(expected[i])||!double.IsFinite(bounds[i])||bounds[i]<0)
            throw new Exception(name+": nonfinite or invalid bound at "+i);
        if(requireBf16&&(BitConverter.SingleToInt32Bits(actual[i])&0xffff)!=0)
            throw new Exception(name+": output did not cross its BF16 boundary at "+i);
        double error=Math.Abs(actual[i]-expected[i]);maxError=Math.Max(maxError,error);maxBound=Math.Max(maxBound,bounds[i]);
        maxFraction=Math.Max(maxFraction,bounds[i]==0?(error==0?0:double.PositiveInfinity):error/bounds[i]);
        if(error>bounds[i])mismatches++;
    }
    results.Add(new{name,count=actual.Length,exact=false,boundPolicy,requireBf16,maxAbsoluteError=maxError,maxBound,maxFraction,mismatches,passed=mismatches==0});
    if(mismatches!=0)throw new Exception($"{name}: {mismatches} out-of-bound values, maximum fraction {maxFraction}");
}

// BF16 ties and adjacent representable values, plus negative and signed zero.
{
    float[] x={0,-0f,1,1+1f/256,1+3f/256,-1-1f/256,123.25f,0.000123f};
    var p=memory.Put(x); kernels.Round(p,x.Length,IntPtr.Zero);
    Check("round_bf16_ties",Download(p,x.Length),x.Select(Bf).ToArray());
}
// W8 format is raw codes[N,K], then FP16 scales[N,K/32].
{
    const int n=4,k=128; sbyte[] codes=Enumerable.Range(0,n*k).Select(i=>(sbyte)(i%255-127)).ToArray();
    Half[] scales=Enumerable.Range(0,n*k/32).Select(i=>(Half)MathF.ScaleB(1,(i%5)-3)).ToArray();
    byte[] blob=new byte[codes.Length+scales.Length*2]; Buffer.BlockCopy(codes,0,blob,0,codes.Length);
    MemoryMarshal.AsBytes(scales.AsSpan()).CopyTo(blob.AsSpan(codes.Length));
    int[] ids={3,0,2}; float[] want=new float[ids.Length*k];
    for(int t=0;t<ids.Length;t++)for(int c=0;c<k;c++)want[t*k+c]=Bf(codes[ids[t]*k+c]*(float)scales[ids[t]*(k/32)+c/32]);
    var output=memory.Zeros<float>(want.Length);kernels.EmbedW8(memory.Put(blob),memory.Put(ids),output,ids.Length,k,n,IntPtr.Zero);
    Check("w8_embedding",Download(output,want.Length),want);
}
// Uniform exact norms avoid reducing-order ambiguity while exercising residual BF16 ties.
{
    const int rows=3,k=256; float[] x=new float[rows*k],d=new float[x.Length],w=new float[k],want=new float[x.Length],res=new float[x.Length];
    for(int t=0;t<rows;t++)for(int c=0;c<k;c++) { x[t*k+c]=t+1;d[t*k+c]=1f/256; }
    for(int c=0;c<k;c++)w[c]=c%2==0?1:0.5f;
    for(int t=0;t<rows;t++)
    {
        float v=Bf(t+1+1f/256),inv=1/MathF.Sqrt(v*v+1e-6f);
        for(int c=0;c<k;c++) { res[t*k+c]=v;want[t*k+c]=Bf(v*inv*w[c]); }
    }
    var xp=memory.Put(x);var output=memory.Zeros<float>(x.Length);
    kernels.AddRmsNorm(xp,memory.Put(d),memory.Put(w),output,rows,k,1e-6f,IntPtr.Zero);
    Check("residual_round",Download(xp,x.Length),res);Check("residual_norm",Download(output,x.Length),want);
}
// Exact per16 calibration check, with all FP4 midpoints and FP8 scale transitions.
{
    const int rows=3,k=256;float divisor=1.25f;
    float[] x=Enumerable.Range(0,rows*k).Select(i=>Bf(((i*37)%257-128)/16f)).ToArray();
    byte[] codes=new byte[x.Length/2],scales=new byte[x.Length/16];
    for(int g=0;g<scales.Length;g++)
    {
        float max=0;for(int j=0;j<16;j++)max=Math.Max(max,Math.Abs(x[g*16+j]));
        byte s=EncodeFp8((divisor*max)/6f);scales[g]=s;float scale=DecodeFp8(s);
        for(int j=0;j<8;j++)
        {
            byte lo=EncodeFp4(s==0?0:(x[g*16+j]*divisor)/scale);
            byte hi=EncodeFp4(s==0?0:(x[g*16+j+8]*divisor)/scale);
            codes[g*8+j]=(byte)(lo|(hi<<4));
        }
    }
    var cp=memory.Zeros<byte>(codes.Length);var sp=memory.Zeros<byte>(scales.Length);var rp=memory.Zeros<float>(rows);
    kernels.QuantizeNvfp4(memory.Put(x),cp,sp,rp,k,rows,divisor,IntPtr.Zero);
    var ca=memory.Read<byte>(cp,codes.Length);var sa=memory.Read<byte>(sp,scales.Length);
    if(!ca.SequenceEqual(codes)||!sa.SequenceEqual(scales))throw new Exception("calibrated NVFP4 byte mismatch");
    Check("calibrated_nvfp4_row_scale",Download(rp,rows),Enumerable.Repeat(1f,rows).ToArray());
    results.Add(new{name="calibrated_nvfp4_codes_scales",codes=codes.Length,scales=scales.Length,passed=true});
}
// Original BF16 projection: exact small integer dot, including noncontiguous output rows.
{
    const int n=96,k=128,rows=3,stride=104;ushort[] w=Enumerable.Range(0,n*k).Select(i=>BfBits((i%7)-3)).ToArray();
    float[] x=Enumerable.Range(0,rows*k).Select(i=>(float)(i%5-2)).ToArray(),want=new float[rows*stride];
    for(int t=0;t<rows;t++)for(int r=0;r<n;r++)
    { float sum=0;for(int c=0;c<k;c++)sum+=FromBf(w[r*k+c])*x[t*k+c];want[t*stride+r]=sum; }
    var output=memory.Zeros<float>(want.Length);kernels.ProjectBf16(memory.Put(w),memory.Put(x),output,n,k,rows,IntPtr.Zero,stride);
    Check("bf16_projection_output_stride",Download(output,want.Length),want);
}
// Weight-only NVFP4: exact scaled power-of-two/integer fixture isolates representation/order.
{
    const int n=13,k=256,rows=3,stride=16;float inv=0.125f;
    byte[] codes=Enumerable.Range(0,n*k/2).Select(i=>(byte)(((i*7)&15)|(((i*11)&15)<<4))).ToArray();
    byte[] scales=Enumerable.Range(0,n*k/16).Select(i=>(byte)(48+(i%3)*8)).ToArray();
    float[] x=Enumerable.Range(0,rows*k).Select(i=>(float)(i%7-3)).ToArray(),want=new float[rows*stride];
    for(int t=0;t<rows;t++)for(int r=0;r<n;r++)
    {
        float sum=0;for(int c=0;c<k;c++)
        { byte b=codes[r*(k/2)+(c/16)*8+c%8];byte code=(byte)(c%16<8?b&15:b>>4);
          sum+=DecodeFp4(code)*(DecodeFp8(scales[r*(k/16)+c/16])*inv)*x[t*k+c]; }
        want[t*stride+r]=sum;
    }
    var output=memory.Zeros<float>(want.Length);kernels.ProjectA16(memory.Put(codes),memory.Put(scales),memory.Put(x),output,n,k,rows,inv,IntPtr.Zero,outputStride:stride);
    Check("nvfp4_a16_planes_stride",Download(output,want.Length),want);
}
// Fused SwiGLU keeps projection accumulators in F32 until its BF16 output.
{
    const int rows=3,ff=256;
    float[] p=Enumerable.Range(0,rows*2*ff).Select(i=>(i%97-48)/37f).ToArray(),want=new float[rows*ff];
    for(int t=0;t<rows;t++)for(int i=0;i<ff;i++)
    { int gate=(i/8)*16+i%8;float a=p[t*2*ff+gate],b=p[t*2*ff+gate+8];want[t*ff+i]=Bf((a/(1+MathF.Exp(-a)))*b); }
    var output=memory.Zeros<float>(want.Length);kernels.SwiGlu(memory.Put(p),output,ff,rows,IntPtr.Zero);
    Check("fused_swiglu_f32_accumulators",Download(output,want.Length),want);
}
// Conv is BF16 at the output; padding may not advance a sequence's state.
{
    const int rows=3,sequences=2,conv=10240,packedWidth=16384,dconv=4;
    float[] packed=new float[rows*sequences*packedWidth],weights=new float[conv*dconv],want=new float[rows*sequences*conv];
    int[] count={1,3};float[][] old=Enumerable.Range(0,sequences).Select(_=>new float[(dconv-1)*conv]).ToArray();
    for(int c=0;c<conv;c++)
    {weights[c*dconv]=0.125f;weights[c*dconv+1]=0.25f;weights[c*dconv+2]=0.5f;weights[c*dconv+3]=1;}
    for(int seq=0;seq<sequences;seq++)
    {
        for(int t=0;t<dconv-1;t++)for(int c=0;c<conv;c++)old[seq][t*conv+c]=(t+1)*0.25f;
        for(int t=0;t<rows;t++)for(int c=0;c<conv;c++)packed[(seq*rows+t)*packedWidth+c]=(t+1)*0.5f;
        for(int t=0;t<count[seq];t++)for(int c=0;c<conv;c++)
        {
            float v=0;for(int tap=0;tap<dconv;tap++)
            {int r=t-(dconv-1)+tap;float x=r>=0?packed[(seq*rows+r)*packedWidth+c]:old[seq][(dconv-1+r)*conv+c];v=MathF.FusedMultiplyAdd(weights[c*dconv+tap],x,v);}
            want[(seq*rows+t)*conv+c]=Bf(v/(1+MathF.Exp(-v)));
        }
    }
    IntPtr[] state=old.Select(memory.Put).ToArray();var table=memory.Put(state);var output=memory.Zeros<float>(want.Length);
    var raw=memory.Zeros<float>(want.Length);var counts=memory.Put(count);var source=memory.Put(packed);var wp=memory.Put(weights);
    kernels.GdnConv(source,packedWidth,rows,sequences,conv,dconv,wp,table,output,raw,false,IntPtr.Zero,counts);
    Check("gdn_bf16_causal_conv",Download(output,want.Length),want);
    for(int seq=0;seq<sequences;seq++)Check("gdn_conv_noadvance_"+seq,Download(state[seq],old[seq].Length),old[seq]);
    kernels.GdnConvState(source,packedWidth,0,rows,counts,sequences,conv,dconv,table,0,1,IntPtr.Zero);
    for(int seq=0;seq<sequences;seq++)
    {
        float[] expected=new float[(dconv-1)*conv];
        for(int i=0;i<dconv-1;i++)for(int c=0;c<conv;c++)
        {int r=count[seq]-(dconv-1)+i;expected[i*conv+c]=r>=0?packed[(seq*rows+r)*packedWidth+c]:old[seq][(dconv-1+r)*conv+c];}
        Check("gdn_conv_counted_state_"+seq,Download(state[seq],expected.Length),expected);
    }
}
// q/k normalization stays F32. Raw A_log requires -expf; AB has no BF16 reround.
{
    const int rows=2,hk=16,hv=48,conv=10240;float[] qkv=new float[rows*conv],ab=new float[rows*96],dt=new float[hv],aLog=new float[hv];
    for(int t=0;t<rows;t++)for(int h=0;h<2*hk;h++)qkv[t*conv+h*128]=(h%7+1)*0.5f;
    for(int t=0;t<rows;t++)for(int h=0;h<hv;h++){ab[t*96+h]=0.5017f;ab[t*96+hv+h]=-0.2507f;aLog[h]=0.125f;dt[h]=0.25f;}
    float[] expected=qkv.ToArray(),gd=new float[rows*hv],beta=new float[rows*hv];
    for(int t=0;t<rows;t++)for(int h=0;h<2*hk;h++) {int i=t*conv+h*128;float x=qkv[i];expected[i]=x/MathF.Sqrt(x*x+1e-6f);}
    for(int t=0;t<rows;t++)for(int h=0;h<hv;h++)
    {gd[t*hv+h]=-MathF.Exp(aLog[h])*MathF.Log(1+MathF.Exp(ab[t*96+h]+dt[h]));beta[t*hv+h]=1/(1+MathF.Exp(-ab[t*96+hv+h]));}
    var qp=memory.Put(qkv);var gp=memory.Zeros<float>(gd.Length);var bp=memory.Zeros<float>(beta.Length);
    kernels.GdnPrep(qp,memory.Put(ab),96,rows,hk,hv,conv,memory.Put(dt),memory.Put(aLog),gp,bp,IntPtr.Zero);
    // Fixed bounds apply only to these fp32 transcendentals, not to BF16 weight/model quality.
    Check("gdn_qk_normalization",Download(qp,expected.Length),expected,false,1e-6f,1e-6f);
    Check("gdn_raw_a_log_decay",Download(gp,gd.Length),gd,false,1e-6f,1e-6f);
    Check("gdn_ab_f32_beta",Download(bp,beta.Length),beta,false,1e-6f,1e-6f);
}
// GDN gated norm: an independent FP64 formula with a declared BF16 rounding
// envelope, not a copy of the device reduction. Arbitrary signs, norms and z vary
// with token/head/channel so a wrong offset, gate or norm convention is visible.
{
    const int rows=3,hv=48,dim=128,conv=10240,packedWidth=16384;
    const float epsilon=1e-6f;
    float[] recurrent=Enumerable.Range(0,rows*hv*dim).Select(i=>Bf(((i*29)%257-128)/23f)).ToArray();
    float[] packed=new float[rows*packedWidth];
    float[] norm=Enumerable.Range(0,dim).Select(i=>Bf(0.5f+(i%17)/16f)).ToArray();
    for(int t=0;t<rows;t++)for(int h=0;h<hv;h++)for(int d=0;d<dim;d++)
        packed[t*packedWidth+conv+h*dim+d]=Bf(((t*11+h*7+d*3)%113-56)/19f);
    double[] reference=new double[recurrent.Length],bounds=new double[recurrent.Length];
    // gamma_1024 comfortably covers the 128-wide positive sum, reciprocal sqrt,
    // epilogue multiplications and scalar transcendentals for this finite fixture.
    // The separately declared BF16 half-ULP covers the final storage rounding.
    const double unitRoundoff=1.0/16777216.0;
    double gamma=1024*unitRoundoff/(1-1024*unitRoundoff);
    for(int t=0;t<rows;t++)for(int h=0;h<hv;h++)
    {
        double ss=0;for(int d=0;d<dim;d++){double x=recurrent[(t*hv+h)*dim+d];ss+=x*x;}
        double inv=1/Math.Sqrt(ss/dim+(double)epsilon);
        for(int d=0;d<dim;d++)
        {
            int i=(t*hv+h)*dim+d;double z=packed[t*packedWidth+conv+h*dim+d];
            reference[i]=recurrent[i]*inv*norm[d]*(z/(1+Math.Exp(-z)));
            double magnitude=Math.Abs(reference[i]);
            double halfBfUlp=magnitude==0?0:Math.ScaleB(1,Math.Max(-134,Math.ILogB(magnitude)-8));
            bounds[i]=halfBfUlp+gamma*magnitude;
        }
    }
    var output=memory.Zeros<float>(reference.Length);
    kernels.GdnOut(memory.Put(recurrent),memory.Put(packed),packedWidth,conv,rows,hv,memory.Put(norm),epsilon,output,IntPtr.Zero);
    CheckBounded("gdn_out_plain_norm_nontrivial_z",Download(output,reference.Length),reference,bounds,
        "FP64 ideal RMSNorm*plainNorm*SiLU(z); BF16 half-ULP + gamma_1024*abs(reference)",true);
}
// Integrated calibrated activation pack -> existing block-scaled FP4 tensor core
// GEMM -> original alpha. This is a synthetic component gate, not a whole-model
// comparison. The reference decodes independent CPU-quantized payloads in FP64.
{
    string mainPath=CudaKernels.LocatePtxPath("tensorsharp_q35_kernels.ptx")
        ??throw new FileNotFoundException("Original Q35 tensor-core module is required for the integrated component check.");
    var mainModule=CudaModule.LoadFromFile(mainPath,identify:true);
    using var q35=Q35Kernels.FromModule(mainModule);
    CudaDriverApi.cuDeviceGetAttribute(out int sms,16,0).ThrowOnError();
    const int outputRows=128,columns=512,guard=128;
    const float inputDivisor=1.25f,weightDivisor=1.75f;
    float alpha=1.0f/(inputDivisor*weightDivisor);
    const double unitRoundoff=1.0/16777216.0;
    const int operationBudget=3*columns+32;
    double gamma=operationBudget*unitRoundoff/(1-operationBudget*unitRoundoff);
    byte[] wCodes=Enumerable.Range(0,outputRows*columns/2)
        .Select(i=>(byte)((i*7+3)%16|(((i*11+5)%16)<<4))).ToArray();
    byte[] wScales=Enumerable.Range(0,outputRows*columns/16)
        .Select(i=>(byte)(32+(i*13)%40)).ToArray();
    var wc=memory.Put(wCodes);var ws=memory.Put(wScales);
    foreach(int rows in new[]{1,4,7,8,9,33,129})
    {
        using var caseMemory=new Memory();
        float[] input=Enumerable.Range(0,rows*columns).Select(i=>Bf(((i*37+i/columns*11)%521-260)/31f)).ToArray();
        var (referenceCodes,referenceScales)=QuantizeReference(input,columns,inputDivisor);
        var xc=caseMemory.Zeros<byte>(referenceCodes.Length);var xs=caseMemory.Zeros<byte>(referenceScales.Length);var xr=caseMemory.Zeros<float>(rows);
        kernels.QuantizeNvfp4(caseMemory.Put(input),xc,xs,xr,columns,rows,inputDivisor,IntPtr.Zero);
        if(!caseMemory.Read<byte>(xc,referenceCodes.Length).SequenceEqual(referenceCodes)
            ||!caseMemory.Read<byte>(xs,referenceScales.Length).SequenceEqual(referenceScales))
            throw new Exception("integrated calibrated payload mismatch, T="+rows);
        Check("integrated_quant_row_scale_T"+rows,caseMemory.Read<float>(xr,rows),Enumerable.Repeat(1f,rows).ToArray());
        double[] reference=new double[rows*outputRows],bounds=new double[reference.Length];
        for(int t=0;t<rows;t++)for(int r=0;r<outputRows;r++)
        {
            double dot=0,sumAbs=0;
            for(int k=0;k<columns;k++)
            {
                double weight=DecodePlaneValue(wCodes,wScales,r,k,columns);
                double activation=DecodePlaneValue(referenceCodes,referenceScales,t,k,columns);
                double product=weight*activation;dot+=product;sumAbs+=Math.Abs(product);
            }
            int i=t*outputRows+r;
            reference[i]=(double)alpha*dot;
            bounds[i]=gamma*Math.Abs(alpha)*sumAbs+2*unitRoundoff*Math.Abs(reference[i]);
        }
        int elements=reference.Length;
        float[] poison=Enumerable.Repeat(123456f,elements+2*guard).ToArray();
        var target=caseMemory.Put(poison);var output=target+guard*sizeof(float);
        var partial=caseMemory.Zeros<float>(8*elements);var counters=caseMemory.Zeros<int>(Math.Max(1,(rows+31)/32));
        foreach(int splitLimit in new[]{1,8})
        {
            q35.GemmW4A4(wc,ws,outputRows,columns,xc,xs,xr,rows,alpha,output,partial,counters,sms,splitLimit,IntPtr.Zero);
            float[] actual=caseMemory.Read<float>(target,poison.Length);
            if(!actual.AsSpan(0,guard).SequenceEqual(poison.AsSpan(0,guard))
                ||!actual.AsSpan(guard+elements,guard).SequenceEqual(poison.AsSpan(guard+elements,guard)))
                throw new Exception("integrated MMA output guard modified");
            CheckBounded($"calibrated_w4a4_T{rows}_splitLimit{splitLimit}",actual.AsSpan(guard,elements).ToArray(),reference,bounds,
                "FP64 decoded quantized dot; gamma_(3K+32)*abs(alpha)*sumAbsProducts + 2u*abs(reference); u=2^-24",false);
        }
        // K=512 has only two 256-value stages, so even a limit of eight resolves
        // to one actual K partition. This checks the normal production resolver;
        // it does not claim coverage of a multi-partition split-K reduction.
        results.Add(new{name="integrated_calibrated_geometry",rows,outputRows,columns,inputDivisor,weightDivisor,
            alphaBits=BitConverter.SingleToInt32Bits(alpha),operationBudget,actualKPartitions=1,mainModuleSha256=mainModule.ImageSha256,
            outputGuards=2*guard,payloadBytes=referenceCodes.Length+referenceScales.Length,passed=true});
    }
}
// Grouped h/3 GDN recurrent transition, state commit and speculative nonmutation.
{
    const int hk=16,hv=48,dim=128,conv=(2*hk+hv)*dim,rows=3,sequences=2;
    float[] qkv=new float[sequences*rows*conv],g=new float[sequences*rows*hv],beta=Enumerable.Repeat(0.5f,g.Length).ToArray();
    for(int t=0;t<sequences*rows;t++)for(int h=0;h<hk;h++)
    { qkv[t*conv+h*dim]=h+1;qkv[t*conv+(hk+h)*dim]=1; }
    for(int t=0;t<sequences*rows;t++)for(int h=0;h<hv;h++)for(int v=0;v<dim;v++)qkv[t*conv+2*hk*dim+h*dim+v]=(h%4+1)*0.25f;
    float[][] original=Enumerable.Range(0,sequences).Select(s=>new float[hv*dim*dim]).ToArray();
    IntPtr[] statePointers=original.Select(memory.Put).ToArray();var states=memory.Put(statePointers);
    var qp=memory.Put(qkv);var gp=memory.Put(g);var bp=memory.Put(beta);var output=memory.Zeros<float>(sequences*rows*hv*dim);
    var want=new float[sequences*rows*hv*dim];var expectedStates=original.Select(v=>v.ToArray()).ToArray();
    for(int s=0;s<sequences;s++)for(int t=0;t<rows;t++)for(int h=0;h<hv;h++)for(int v=0;v<dim;v++)
    {
        int j=(h*dim+v)*dim;float old=expectedStates[s][j],val=qkv[(s*rows+t)*conv+2*hk*dim+h*dim+v];
        float next=old+(val-old)*0.5f;expectedStates[s][j]=next;
        want[((s*rows+t)*hv+h)*dim+v]=Bf(next*(h/3+1)*0.08838834764831845f);
    }
    kernels.GdnScan(qp,conv,hk,hv,rows,sequences,gp,bp,states,output,true,IntPtr.Zero);
    Check("gdn_grouped_verify_output",Download(output,want.Length),want);
    for(int s=0;s<sequences;s++)Check("gdn_verify_preserves_state_"+s,Download(statePointers[s],original[s].Length),original[s]);
    kernels.GdnScan(qp,conv,hk,hv,rows,sequences,gp,bp,states,output,false,IntPtr.Zero);
    Check("gdn_grouped_commit_output",Download(output,want.Length),want);
    for(int s=0;s<sequences;s++)Check("gdn_committed_state_"+s,Download(statePointers[s],original[s].Length),expectedStates[s]);
}
results.AddRange(AttentionChecks.Run());
File.WriteAllText(args[1],JsonSerializer.Serialize(new{schema="ninfer-cuda-kernel-probe/1",moduleSha256=kernels.ImageSha256,scope="synthetic-component-only",passed=true,checks=results},new JsonSerializerOptions{WriteIndented=true}));
Console.WriteLine("PASS synthetic NInfer CUDA components: "+results.Count);

static float Bf(float x)=>FromBf(BfBits(x));
static ushort BfBits(float x)
{
    uint u=(uint)BitConverter.SingleToInt32Bits(x);
    return (ushort)((u+0x7fffu+((u>>16)&1))>>16);
}
static float FromBf(ushort x)=>BitConverter.Int32BitsToSingle(x<<16);
static (byte[] Codes,byte[] Scales) QuantizeReference(float[] input,int columns,float inputDivisor)
{
    byte[] codes=new byte[input.Length/2],scales=new byte[input.Length/16];
    for(int group=0;group<scales.Length;group++)
    {
        float maximum=0;for(int i=0;i<16;i++)maximum=Math.Max(maximum,Math.Abs(input[group*16+i]));
        byte scale=EncodeFp8((inputDivisor*maximum)/6f);scales[group]=scale;float decoded=DecodeFp8(scale);
        for(int i=0;i<8;i++)
        {
            byte low=EncodeFp4(scale==0?0:(input[group*16+i]*inputDivisor)/decoded);
            byte high=EncodeFp4(scale==0?0:(input[group*16+i+8]*inputDivisor)/decoded);
            codes[group*8+i]=(byte)(low|(high<<4));
        }
    }
    return(codes,scales);
}
static double DecodePlaneValue(byte[] codes,byte[] scales,int row,int column,int columns)
{
    int within=column%16;
    byte packed=codes[row*(columns/2)+(column/16)*8+within%8];
    byte code=(byte)(within<8?packed&15:packed>>4);
    return(double)DecodeFp4(code)*DecodeFp8(scales[row*(columns/16)+column/16]);
}
static float DecodeFp4(byte x) { float a=(x&7) switch {0=>0,1=>0.5f,2=>1,3=>1.5f,4=>2,5=>3,6=>4,_=>6};return (x&8)!=0?-a:a; }
static byte EncodeFp4(float x)
{
    bool sign=(BitConverter.SingleToInt32Bits(x)&int.MinValue)!=0;float v=Math.Abs(x),best=float.PositiveInfinity;byte chosen=0;
    for(byte i=0;i<8;i++) { float d=Math.Abs(v-DecodeFp4(i));if(d<best||(d==best&&(i&1)==0)) {best=d;chosen=i;} }
    return (byte)(chosen|(sign?8:0));
}
static float DecodeFp8(byte b)
{
    int e=(b>>3)&15,m=b&7;return e==0?m/512f:MathF.ScaleB(1+m/8f,e-7);
}
static byte EncodeFp8(float x)
{
    float best=float.PositiveInfinity;byte chosen=0;
    for(byte i=0;i<127;i++) { float d=Math.Abs(x-DecodeFp8(i));if(d<best||(d==best&&(i&1)==0)) {best=d;chosen=i;} }
    return chosen;
}
sealed unsafe class Memory:IDisposable
{
    private readonly List<IntPtr> _allocations=new();
    public IntPtr Put<T>(T[] values)where T:unmanaged
    {
        long bytes=checked((long)values.Length*sizeof(T));var p=Alloc(bytes);
        fixed(T* v=values)CudaDriverApi.cuMemcpyHtoD(p,(IntPtr)v,(UIntPtr)bytes).ThrowOnError();return p;
    }
    public IntPtr Zeros<T>(int count)where T:unmanaged=>Put(new T[count]);
    public IntPtr Alloc(long bytes) { CudaDriverApi.cuMemAlloc(out var p,(UIntPtr)bytes).ThrowOnError();_allocations.Add(p);return p; }
    public T[] Read<T>(IntPtr p,int count)where T:unmanaged
    { T[] result=new T[count];fixed(T* v=result)CudaDriverApi.cuMemcpyDtoH((IntPtr)v,p,(UIntPtr)checked((long)count*sizeof(T))).ThrowOnError();return result; }
    public void Dispose() {foreach(var p in _allocations)CudaDriverApi.cuMemFree(p).ThrowOnError();}
}
