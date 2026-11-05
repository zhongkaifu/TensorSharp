// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;

internal static partial class Program
{
    private const int NormWidth=5120;
    private const float Epsilon=1e-6f;
    private static readonly string[] FixtureNames={"exact-binary","broad-finite","ties-zero-and-last-column","fma-contraction","bf16-ties-adjacent"};
    private static float[] NormWeights()=>Enumerable.Range(0,NormWidth).Select(c=>1+Bf((c%17-8)/2048f)).ToArray();
    private static float[] Seed(Fixture f,int rotation=0)
    {
        var x=new float[OutputCount];
        for(int t=0;t<Rows;t++)for(int c=0;c<N;c++)
        {
            int i=t*N+c,r=(c+rotation)%N;
            x[i]=f.Name switch{
                "fma-contraction"=>-1f,
                "bf16-ties-adjacent"=>r%2==0?1f:-1f,
                "broad-finite"=>-Bf((float)f.Ideal[t*N+r]),
                "ties-zero-and-last-column"=>BitConverter.UInt32BitsToSingle(r%2==0?0x80000000u:0u),
                _=>Bf(((r*37+t*13)%257-128)/16f)};
        }
        return x;
    }
    private static Fixture ResidualFixture(string name)
    {
        if(name is not ("fma-contraction" or "bf16-ties-adjacent"))return Make(name);
        bool fma=name=="fma-contraction";var input=new float[Count];var wc=new byte[N*Width/2];var ws=new byte[N*Width/16];var codes=new byte[Count/2];var scales=new byte[Count/16];
        float weightD=fma?3:8,alpha=1f/weightD;
        for(int t=0;t<Rows;t++){input[t*Width]=fma?3:1;input[t*Width+15]=6;if(!fma){input[t*Width+16]=MathF.ScaleB(1,-10);input[t*Width+31]=MathF.ScaleB(6,-9);}}
        for(int g=0;g<scales.Length;g++)
        {float max=0;for(int j=0;j<16;j++)max=Math.Max(max,Math.Abs(input[g*16+j]));byte s=Encode8(max/6f);scales[g]=s;for(int j=0;j<8;j++)codes[g*8+j]=(byte)(Encode4(s==0?0:input[g*16+j]/R8(s))|(Encode4(s==0?0:input[g*16+j+8]/R8(s))<<4));}
        var ideal=new double[OutputCount];var bound=new double[OutputCount];var exact=new float[OutputCount];double u=Math.ScaleB(1d,-24),gamma=(3*Width+32)*u/(1-(3*Width+32)*u);
        for(int r=0;r<N;r++)
        {
            int codeBase=r*(Width/2),scaleBase=r*(Width/16),sign=fma?1:r%2==0?1:-1,extra=fma||r%3==0?0:r%3==1?1:-1;
            wc[codeBase]=(byte)(fma?2:sign>0?1:9);ws[scaleBase]=(byte)(fma?0x38:0x18);
            if(extra!=0){wc[codeBase+8]=(byte)(sign*extra>0?1:9);ws[scaleBase+1]=1;}
            double dot=fma?3:sign*(1d/32+extra*Math.ScaleB(1d,-20)),absolute=fma?3:1d/32+Math.Abs(extra)*Math.ScaleB(1d,-20);
            for(int t=0;t<Rows;t++){int i=t*N+r;ideal[i]=dot*alpha;exact[i]=(float)ideal[i];bound[i]=gamma*Math.Abs(alpha)*absolute+2*u*Math.Abs(ideal[i]);}
        }
        return new(name,input,wc,ws,codes,scales,1,weightD,ideal,bound,exact);
    }
    private static float R8(byte s)=>TensorSharp.Validation.NinferFlashInferA16Reference.Fp8(s);
    private sealed record NormOracle(float[] Residual,double[] Ideal,double[] Bound);
    private static NormOracle NormReference(float[] seed,byte[] projection,float[] weight,bool nullDelta=false)
    {
        Need(seed.Length==OutputCount&&projection.Length==OutputCount*4&&weight.Length==NormWidth,"Complete norm operands required.");
        var p=MemoryMarshal.Cast<byte,float>(projection);var residual=new float[OutputCount];
        for(int i=0;i<residual.Length;i++)residual[i]=Bf(seed[i]+(nullDelta?0f:p[i]));
        return CanonicalNormReference(residual,weight);
    }
    private static NormOracle CanonicalNormReference(float[] residual,float[] weight)
    {
        var ideal=new double[OutputCount];var bound=new double[OutputCount];
        // Independent FP64 norm of the exact represented residual. Positive square
        // sums: gamma(K+64); 8*gamma covers F32 reduction/division/products, 2^-18
        // conservatively covers normal rsqrt. BF16 RNE adds <=2^-8 relative plus
        // a subnormal half-ulp. The separate projection oracle covers GEMM error.
        double u=Math.ScaleB(1d,-24),gamma=(NormWidth+64)*u/(1-(NormWidth+64)*u),rho=8*gamma+Math.ScaleB(1d,-18);
        for(int t=0;t<Rows;t++)
        {
            double sum=0;for(int c=0;c<NormWidth;c++){double v=residual[t*NormWidth+c];Need(double.IsFinite(v),"Nonfinite canonical residual.");sum+=v*v;}
            Need(double.IsFinite(sum)&&sum<1e30,"Norm fixture exceeds the declared nonoverflow envelope.");double inv=1/Math.Sqrt(sum/NormWidth+Epsilon);
            for(int c=0;c<NormWidth;c++){int i=t*NormWidth+c;double v=residual[i]*inv*weight[c];ideal[i]=v;bound[i]=Math.Abs(v)*(rho+(1+rho)/256)+Math.ScaleB(1d,-133);}
        }
        return new(residual,ideal,bound);
    }
    private sealed record NormNumeric(int Values,int Nonfinite,int NonBf16,int Violations,double MaxFraction){public bool Passed=>Nonfinite+NonBf16+Violations==0;}
    private static NormNumeric CheckNorm(byte[] bytes,NormOracle o)
    {
        Need(bytes.Length==OutputCount*4,"Full norm output required.");int nf=0,bf=0,bad=0;double fraction=0;var a=MemoryMarshal.Cast<byte,float>(bytes);
        for(int i=0;i<a.Length;i++){if(!float.IsFinite(a[i])){nf++;continue;}if((BitConverter.SingleToUInt32Bits(a[i])&0xffff)!=0)bf++;double e=Math.Abs(a[i]-o.Ideal[i]);if(e>o.Bound[i])bad++;fraction=Math.Max(fraction,e/o.Bound[i]);}return new(a.Length,nf,bf,bad,fraction);
    }
    private static void ResidualSelfTest()
    {
        Need(N==5120&&Geometry.Name is "output" or "down","Only the two fixed residual projection roles are admitted.");
        Need(PtxTypes(".visible .entry example(.param .u64 .ptr .align 1 example_0, .param .u32 example_1, .param .f32 example_2)","example").SequenceEqual(new[]{"u64","u32","f32"}),"Actual NVCC pointer-qualified PTX ABI parser failed.");
        var original=Console.Out;using var discarded=new StringWriter();try{Console.SetOut(discarded);SelfTest();}finally{Console.SetOut(original);}
        float[] w=NormWeights();Need(w.Any(v=>BitConverter.SingleToInt32Bits(v)!=BitConverter.SingleToInt32Bits(Bf(v))),"Effective F32 weight boundary missing.");
        foreach(string name in FixtureNames)
        {
            var f=ResidualFixture(name);Need(Check(Bytes(f.Ideal.Select(v=>(float)v).ToArray()),f,false).Passed,"Projection FP64 fixture failed.");
            var o=NormReference(Seed(f),Bytes(f.Exact??f.Ideal.Select(v=>(float)v).ToArray()),w);Need(CheckNorm(Bytes(o.Ideal.Select(v=>Bf((float)v)).ToArray()),o).Passed,"FP64 norm bound excludes rounded ideal.");
        }
        float alpha=1f/3,product=3*alpha,separate=-1f+product,fused=MathF.FusedMultiplyAdd(3,alpha,-1);Need(separate==0&&fused!=0&&Bf(fused)!=Bf(separate),"FMA contraction witness ineffective.");
        float z=BitConverter.UInt32BitsToSingle(0x80000000u);Need(BitConverter.SingleToUInt32Bits(Bf(z+z))==0x80000000u&&BitConverter.SingleToUInt32Bits(Bf(z+0f))==0,"Initial-null and fused-canonical signedzero semantics collapsed.");
        var tie=ResidualFixture("bf16-ties-adjacent");var r=NormReference(Seed(tie),Bytes(tie.Exact!),w).Residual;Need(r[0]==1&&r[1]==-1-1f/128&&r[2]==1,"Even tie/above/below fixtures differ.");
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new{passed=true,CPU="PASS",GPU="NOT_RUN",shape=Geometry.Name,fixtures=5,rows=8,n=N,k=Width,splits=8,rawByteBudget=RawBudget,originalReferenceUnchanged=true,performance="NOT_RUN"}));
    }
}
