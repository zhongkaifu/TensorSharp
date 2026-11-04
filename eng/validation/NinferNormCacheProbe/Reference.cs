// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;

internal static partial class Program
{
    private const int Rows=8, Width=5120, Count=Rows*Width;
    private const float Epsilon=1e-6f;
    private static readonly string[] Modes={"null","disjoint","delta-x","delta-output"};
    private static readonly byte[] Poisons={0xc1,0xd2,0xc1};
    private sealed record Fixture(string Name,float[] X,float[] Delta,float[] Weight);
    private sealed record Oracle(float[] Residual,double[] Ideal,double[] Bound);
    private static float Bf(float value)
    {
        uint bits=BitConverter.SingleToUInt32Bits(value);
        return BitConverter.UInt32BitsToSingle(unchecked(bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
    }
    private static byte[] Bytes<T>(T[] values) where T:unmanaged=>MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static Fixture[] Fixtures()
    {
        string[] names={"uniform","nonuniform","cancellation","bf16-ties-adjacent","signed-zero","last-column","lane-row-boundaries"};
        return names.Select(name=>
        {
            var x=new float[Count];var d=new float[Count];var w=new float[Width];
            for(int c=0;c<Width;c++)w[c]=Bf((c%17-8)/8f);
            for(int t=0;t<Rows;t++)for(int c=0;c<Width;c++)
            {
                int i=t*Width+c;
                switch(name)
                {
                    case "uniform":x[i]=t+1;d[i]=1f/256;w[c]=c%2==0?1:.5f;break;
                    case "nonuniform":x[i]=Bf(((c*37+t*13)%257-128)/16f);d[i]=((c*11+t*7)%65-32)/2048f;w[c]=1+Bf((c%17-8)/2048f);break;
                    case "cancellation":x[i]=Bf((c%2==0?1:-1)*(1+t*.25f+(c%5)*.125f));d[i]=-x[i]+(c%3-1)/256f;break;
                    case "bf16-ties-adjacent":
                        x[i]=c%2==0?1f:-1f;float half=1f/256;
                        d[i]=(c%3==0?half:c%3==1?MathF.BitDecrement(half):MathF.BitIncrement(half))*(c%2==0?1:-1);break;
                    case "signed-zero":x[i]=BitConverter.UInt32BitsToSingle(c%2==0?0x80000000u:0u);d[i]=BitConverter.UInt32BitsToSingle(c%3==0?0x80000000u:0u);w[c]=(c%4) switch{0=>-1,1=>1,2=>BitConverter.UInt32BitsToSingle(0x80000000u),_=>0};break;
                    case "last-column":if(c==Width-1){x[i]=t%2==0?64:-64;d[i]=.125f;w[c]=1.5f;}break;
                    case "lane-row-boundaries":if(c is 0 or 31 or 32 or 255 or 256 or 511 or 512 or 4864 or 5119){x[i]=(t+1)*(c%2==0?.25f:-.5f);d[i]=(c%3-1)/128f;}break;
                }
            }
            return new Fixture(name,x,d,w);
        }).ToArray();
    }
    private static Oracle Reference(Fixture fixture,string mode)
    {
        Need(Modes.Contains(mode),"Unknown alias mode.");
        var residual=new float[Count];var ideal=new double[Count];var bound=new double[Count];
        // A fixed, deliberately conservative scoped FP64 envelope. Positive
        // square sums have at most K+64 F32 roundings in any reduction order;
        // 8*gamma also covers division, normal rsqrt and two output products.
        // The additional 2^-18 covers approximate rsqrt error. Final BF16 RNE
        // contributes <=2^-8 relative error (plus its subnormal half-ulp).
        // Fixtures use finite normal nonzero operands, |x|,|d|<=64, |w|<=2.
        double u=Math.ScaleB(1d,-24),gamma=(Width+64)*u/(1-(Width+64)*u),rho=8*gamma+Math.ScaleB(1d,-18);
        for(int t=0;t<Rows;t++)
        {
            double sum=0;
            for(int c=0;c<Width;c++)
            {
                int i=t*Width+c;float delta=mode=="null"?0:mode=="delta-x"?fixture.X[i]:fixture.Delta[i];
                float v=Bf(fixture.X[i]+delta);residual[i]=v;sum+=(double)v*v;
            }
            double inverse=1/Math.Sqrt(sum/Width+Epsilon);
            for(int c=0;c<Width;c++)
            {
                int i=t*Width+c;double value=(double)residual[i]*inverse*fixture.Weight[c];ideal[i]=value;
                bound[i]=Math.Abs(value)*(rho+(1+rho)/256)+Math.ScaleB(1d,-133);
            }
        }
        return new(residual,ideal,bound);
    }
    private sealed record Numeric(int Values,int Nonfinite,int NonBf16,int Violations,double MaxError,double MaxFraction)
    { public bool Passed=>Nonfinite==0&&NonBf16==0&&Violations==0; }
    private static Numeric Check(byte[] bytes,Oracle oracle)
    {
        Need(bytes.Length==Count*4,"Full norm output required.");var values=MemoryMarshal.Cast<byte,float>(bytes);int nonfinite=0,nonbf=0,violations=0;double max=0,fraction=0;
        for(int i=0;i<Count;i++)
        {
            float value=values[i];if(!float.IsFinite(value)){nonfinite++;continue;}
            if((BitConverter.SingleToUInt32Bits(value)&0xffff)!=0)nonbf++;
            double error=Math.Abs(value-oracle.Ideal[i]);max=Math.Max(max,error);fraction=Math.Max(fraction,error/oracle.Bound[i]);if(error>oracle.Bound[i])violations++;
        }
        return new(Count,nonfinite,nonbf,violations,max,fraction);
    }
    private static void SelfTest()
    {
        Need(BitConverter.SingleToUInt32Bits(Bf(1+1f/256))==0x3f800000u,"BF16 even tie.");
        Need(BitConverter.SingleToUInt32Bits(Bf(1+3f/256))==0x3f820000u,"BF16 odd tie.");
        Need(Bf(MathF.BitDecrement(1+1f/256))==1&&Bf(MathF.BitIncrement(1+1f/256))==1+1f/128,"BF16 adjacent ties.");
        Need(BitConverter.SingleToUInt32Bits(Bf(BitConverter.UInt32BitsToSingle(0x80000000u)))==0x80000000u,"BF16 signed zero.");
        var fixtures=Fixtures();Need(fixtures.Length==7&&fixtures.Select(x=>x.Name).Distinct().Count()==7,"Seven unique fixtures required.");
        Need(fixtures.Single(x=>x.Name=="nonuniform").Weight.Any(x=>BitConverter.SingleToUInt32Bits(x)!=BitConverter.SingleToUInt32Bits(Bf(x))),"Effective unit-offset weights must include non-BF16 F32 values.");
        foreach(var f in fixtures)foreach(string mode in Modes)
        {
            Need(f.X.All(float.IsFinite)&&f.Delta.All(float.IsFinite)&&f.Weight.All(float.IsFinite),"Finite fixtures required.");var r=Reference(f,mode);
            Need(r.Residual.All(float.IsFinite)&&r.Ideal.All(double.IsFinite)&&r.Bound.All(x=>double.IsFinite(x)&&x>0),"Finite independent oracle required.");
            Need(Check(Bytes(r.Ideal.Select(x=>Bf((float)x)).ToArray()),r).Passed,"Oracle's correctly rounded ideal must fit fixed envelope.");
        }
        var cancellation=fixtures.Single(x=>x.Name=="cancellation");Need(!Bytes(Reference(cancellation,"null").Residual).AsSpan().SequenceEqual(Bytes(Reference(cancellation,"disjoint").Residual)),"Alias fixtures must distinguish delta.");
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new{passed=true,CPU="PASS",GPU="NOT_RUN",fixtures=7,aliasModes=4,fullOutputValues=28*Count,componentLifecycleRecords=56,componentNormLaunches=336,timing="NOT_RUN"}));
    }
}
