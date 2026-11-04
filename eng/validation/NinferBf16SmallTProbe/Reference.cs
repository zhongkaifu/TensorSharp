// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static partial class Program
{
    private const int Rows=8;
    private const float Epsilon=1e-6f;
    private static readonly byte[] Poisons={0xc1,0xd2,0xc1};
    private sealed record Shape(string Name,int N,int K,int RealCount);
    private static readonly Shape[] Shapes={new("input",14336,5120,6),new("output",5120,6144,3)};
    private sealed record Case(string Id,string Shape,string Kind,string? Tensor,long AbsoluteOffset,int WeightBytes,bool Permuted,bool Exact,Pin Input,Pin Seed,Pin Norm);
    private sealed record ArtifactStat(long Bytes,long MtimeTicks);
    private sealed record Prepared(string Schema,string Artifact,string DirectorySha256,ArtifactStat ArtifactStat,Case[] Cases,string GeneratorPolicy,Pin ProducerAssembly);
    private sealed record OracleCase(string Id,string WeightSha256,string OriginalWeightSha256,Pin Ideal,Pin L1);
    private sealed record OracleSet(string Schema,Pin Prepared,OracleCase[] Cases);
    private static float Bf(float v){uint b=BitConverter.SingleToUInt32Bits(v);return BitConverter.UInt32BitsToSingle(unchecked(b+0x7fffu+((b>>16)&1))&0xffff0000u);}
    private static byte[] Bytes<T>(T[] x)where T:unmanaged=>MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
    private static float[] Floats(Pin p){Check(p);return MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(p.Path)).ToArray();}
    private static int SourceRow(int r)=>r<12288?r/512*256+r%256+(r%512>=256?7168:0):r<13312?6144+r-12288:r;
    private static byte[] Weight(Case c,Shape s,NinferArtifactReader? reader)
    {
        var b=new byte[c.WeightBytes];var w=MemoryMarshal.Cast<byte,ushort>(b.AsSpan());
        if(c.Kind=="real")
        {
            Need(reader!=null,"Original reader required.");var d=reader!.Tensors.Single(x=>x.Name==c.Tensor);Need(d.Format==NinferTensorFormat.BFloat16&&d.Layout==NinferTensorLayout.ContiguousLittleEndianV1&&d.AbsoluteOffset==c.AbsoluteOffset&&d.ByteLength==(ulong)c.WeightBytes,"Actual BF16 tensor differs.");
            for(int r=0;r<s.N;){int run=c.Permuted?Math.Min(s.N-r,r<12288?256-r%256:r<13312?13312-r:s.N-r):s.N-r;reader.ReadObjectRange(d,(ulong)((c.Permuted?SourceRow(r):r)*s.K*2L),b.AsSpan(r*s.K*2,run*s.K*2));r+=run;}
        }
        else for(int r=0;r<s.N;r++)for(int k=0;k<s.K;k++)
        {
            float v=c.Kind=="dense"?((r*17+k*13)%31-15)/32f:c.Kind=="cancellation"?(r==0?BitConverter.UInt32BitsToSingle(0x80000000u):((r*17+(k/2)*13)%31-15)/32f):k==0?(r%2==0?1f:-1f):k==1?(r%2==0?1f:-1f)*(MathF.ScaleB(1,-8)+(r%3==0?0:r%3==1?MathF.ScaleB(1,-15):-MathF.ScaleB(1,-15))):k==s.K-1?MathF.ScaleB(1,16):k==s.K-2?-MathF.ScaleB(1,-16):0;
            w[r*s.K+k]=(ushort)(BitConverter.SingleToUInt32Bits(v)>>16);
        }
        foreach(ushort q in w)Need((q&0x7f80)!=0x7f80,"Nonfinite BF16 weight.");return b;
    }
    private static float[] Input(Shape s,string kind,int salt)
    {
        var x=new float[Rows*s.K];for(int t=0;t<Rows;t++)for(int k=0;k<s.K;k++)x[t*s.K+k]=kind=="rounding-lowbits"?(k<2?1:k==s.K-1?MathF.ScaleB(1,-16):k==s.K-2?MathF.ScaleB(1,16):0):kind=="cancellation"?(t==0?0:t==1?BitConverter.UInt32BitsToSingle(0x80000000u):(k%4 switch{0=>16f,1=>MathF.ScaleB(1,-8),2=>-16f,_=>-MathF.ScaleB(1,-8)})):Bf(((t*11+k*7+salt)%29-14)/16f);return x;
    }
    private static float[] Seed(Shape s,string kind)
    {var x=new float[Rows*s.N];for(int t=0;t<Rows;t++)for(int r=0;r<s.N;r++)x[t*s.N+r]=kind=="rounding-lowbits"?(r%2==0?-1:1):Bf(((t*17+r*3)%41-20)/8f);return x;}
    private static void Prepare(string artifact,string expected,string output)
    {
        Fresh(output);using var reader=new NinferArtifactReader(artifact);Need(reader.DirectorySha256==expected,"Original artifact directory differs.");var plan=NinferQwen38TargetPlan.Create(reader);Directory.CreateDirectory(output);var cases=new List<Case>();
        Pin Put(string name,float[] x){string p=Path.Combine(output,name+".f32");File.WriteAllBytes(p,Bytes(x));return FilePin(p);}
        foreach(var s in Shapes)
        {
            var real=plan.Layers.Select(l=>(Layer:l,Binding:s.Name=="input"?l.InputProjection:l.OutputProjection)).Where(x=>x.Binding.Descriptor.Format==NinferTensorFormat.BFloat16&&x.Binding.Descriptor.Shape.SequenceEqual(new ulong[]{(ulong)s.N,(ulong)s.K})).ToArray();Need(real.Length==s.RealCount,"All actual BF16 exception matrices required.");
            foreach(var item in real)
            {
                var d=item.Binding.Descriptor;string id=s.Name+"-layer"+item.Layer.Index;float[] norm=Enumerable.Range(0,5120).Select(k=>1f+Bf((k%17-8)/2048f)).ToArray();
                if(s.Name=="output"){var nd=item.Layer.PostAttentionNorm.Descriptor;var raw=new byte[checked((int)nd.ByteLength)];reader.ReadObjectRange(nd,0,raw);var bits=MemoryMarshal.Cast<byte,ushort>(raw);Need(bits.Length==5120,"Original postnorm width differs.");for(int k=0;k<5120;k++)norm[k]=1f+BitConverter.UInt32BitsToSingle((uint)bits[k]<<16);}
                cases.Add(new(id,s.Name,"real",d.Name,d.AbsoluteOffset,checked(s.N*s.K*2),s.Name=="input",false,Put(id+"-input",Input(s,"real",item.Layer.Index)),Put(id+"-seed",Seed(s,"real")),Put(id+"-norm",norm)));
            }
            foreach(string kind in new[]{"dense","cancellation","rounding-lowbits"})
            {string id=s.Name+"-"+kind;cases.Add(new(id,s.Name,kind,null,0,checked(s.N*s.K*2),false,kind!="cancellation",Put(id+"-input",Input(s,kind,0)),Put(id+"-seed",Seed(s,kind)),Put(id+"-norm",Enumerable.Range(0,5120).Select(k=>1f+Bf((k%17-8)/2048f)).ToArray())));}
        }
        Save(Path.Combine(output,"prepared-inputs.json"),new Prepared("ninfer-bf16-small-t-prepared/1",artifact,expected,new(reader.FileBytes,reader.FileLastWriteUtcTicks),cases.ToArray(),"bf16-smallt-fixtures-v1",FilePin(typeof(Program).Assembly.Location)),true);
    }
    private sealed record Numeric(int Values,int Nonfinite,int Violations,double MaxBoundFraction){internal bool Passed=>Nonfinite==0&&Violations==0;}
    private static Numeric Dot(byte[] bytes,double[] ideal,double[] l1,int k,bool exact)
    {
        var a=MemoryMarshal.Cast<byte,float>(bytes);Need(a.Length==ideal.Length&&l1.Length==a.Length,"Full oracle shape required.");double u=Math.ScaleB(1d,-24),gamma=(k+64)*u/(1-(k+64)*u),u64=Math.ScaleB(1d,-53);int nf=0,bad=0;double fraction=0;
        for(int i=0;i<a.Length;i++){if(!float.IsFinite(a[i])){nf++;continue;}double bound=(gamma+32*k*u64)*l1[i]+Math.ScaleB(1d,-149)*k;double error=Math.Abs(a[i]-ideal[i]);if(error>bound||exact&&BitConverter.SingleToUInt32Bits(a[i])!=BitConverter.SingleToUInt32Bits((float)ideal[i]))bad++;fraction=Math.Max(fraction,bound==0?error:error/bound);}return new(a.Length,nf,bad,fraction);
    }
    private static bool Consumer(Case c,Shape s,byte[] projection,byte[] output,byte[]? residual)
    {
        var a=MemoryMarshal.Cast<byte,float>(projection);var y=MemoryMarshal.Cast<byte,float>(output);if(s.Name=="input"){for(int i=0;i<a.Length;i++)if(BitConverter.SingleToUInt32Bits(y[i])!=BitConverter.SingleToUInt32Bits(Bf(a[i])))return false;return true;}
        Need(residual!=null,"Canonical residual capture required.");float[] seed=Floats(c.Seed),weight=Floats(c.Norm);var x=MemoryMarshal.Cast<byte,float>(residual);double u=Math.ScaleB(1d,-24),g=(5120+64)*u/(1-(5120+64)*u),rho=8*g+Math.ScaleB(1d,-18);
        for(int t=0;t<Rows;t++)
        {double sum=0;for(int r=0;r<s.N;r++){int i=t*s.N+r;if(BitConverter.SingleToUInt32Bits(x[i])!=BitConverter.SingleToUInt32Bits(Bf(seed[i]+a[i])))return false;sum+=(double)x[i]*x[i];}Need(sum<1e30&&double.IsFinite(sum),"Norm fixture outside declared range.");double inv=1/Math.Sqrt(sum/s.N+Epsilon);
         for(int r=0;r<s.N;r++){int i=t*s.N+r;double ideal=x[i]*inv*weight[r],bound=Math.Abs(ideal)*(rho+(1+rho)/256)+Math.ScaleB(1d,-133);if(!float.IsFinite(y[i])||(BitConverter.SingleToUInt32Bits(y[i])&65535)!=0||Math.Abs(y[i]-ideal)>bound)return false;}}
        return true;
    }
    private static object SelfTest()
    {
        Need(Enumerable.Range(0,14336).Select(SourceRow).Distinct().Count()==14336,"Attention permutation must be bijective.");float raw=1+MathF.ScaleB(1,-8);Need(raw==1.00390625f&&Bf(raw)==1&&Bf(-1f+raw)!=Bf(-1f+Bf(raw)),"Raw F32 residual boundary witness lost.");
        foreach(var s in Shapes){Need(Input(s,"dense",0).All(v=>(BitConverter.SingleToUInt32Bits(v)&65535)==0),"BF16 input fixture differs.");Need(Seed(s,"dense").Length==Rows*s.N,"Complete consumer seed required.");Need((long)s.K*15*14<(1<<24),"Dense fixed-lattice absolute integer sum must be exact independent of association.");}
        return new{CPU="PASS",GPU="NOT_RUN",shapes=Shapes,syntheticPerShape=3,realMatrices=9,resetLives=2,poisons=3,rawCap=RawCap,performance="NOT_RUN"};
    }
}
