// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using R=TensorSharp.Validation.NinferFlashInferA16Reference;

internal static partial class Program
{
    private sealed record Shape(string Name,int N,int K,int Splits,string Consumer);
    private static readonly Shape[] Shapes={new("gdn-input",16384,5120,2,"round"),new("gate-up",34816,5120,1,"swiglu"),new("down",5120,17408,8,"none"),new("output",5120,6144,8,"none"),new("gqa-input",14336,5120,2,"round")};
    private static Shape Geometry=Shapes[0];
    private const int Rows=8;
    private static int Width=>Geometry.K;
    private static int Count=>Rows*Width;
    private static int N=>Geometry.N;
    private static int OutputCount=>Rows*N;
    private static int FinalCount=>Geometry.Consumer=="swiglu"?OutputCount/2:OutputCount;
    private static int NodesPerChain=>Geometry.Consumer=="none"?2:3;
    private static readonly byte[] Poisons={0xc1,0xd2,0xc1};
    private static byte[] Bytes<T>(T[] x)where T:unmanaged=>MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
    private sealed record Fixture(string Name,float[] Input,byte[] WeightCodes,byte[] WeightScales,byte[] Codes,byte[] Scales,float InputD,float WeightD,double[] Ideal,double[] Bound,float[]? Exact)
    {internal float Alpha=>1f/(InputD*WeightD);internal bool WitnessOnly{get;init;}}
    private static int[] Active
    {
        get
        {
            int[] basis={0,1,7,8,31,32,63,64,127,128,255,256,511,512,1023,1024,2047,2048,3071,3072,4095,4096,6143,6144,8191,8192,10239,10240,12287,12288,14335,14336,16382,16383};
            var rows=basis.Where(r=>r<N).Concat(new[]{N-2,N-1});
            if(Geometry.Consumer=="swiglu")rows=rows.SelectMany(r=>new[]{r,r^8});
            return rows.Distinct().Order().ToArray();
        }
    }
    private sealed record Format(int Values,int Nonfinite,int NonBf16){public bool Passed=>Nonfinite==0&&NonBf16==0;}
    private static Format ConsumerFormat(byte[] bytes)
    {
        Need(bytes.Length==FinalCount*4,"Full consumer output required.");int nonfinite=0,bf=0;foreach(float x in MemoryMarshal.Cast<byte,float>(bytes)){if(!float.IsFinite(x))nonfinite++;if(Geometry.Consumer!="none"&&(BitConverter.SingleToUInt32Bits(x)&0xffff)!=0)bf++;}return new(FinalCount,nonfinite,bf);
    }
    private static float Bf(float x)=>R.Float(R.Bf(x));
    private static byte Encode4(float x)
    {
        byte chosen=0;float best=float.PositiveInfinity,v=Math.Abs(x);
        for(byte i=0;i<8;i++){float d=Math.Abs(v-R.Fp4(i));if(d<best||(d==best&&(i&1)==0)){chosen=i;best=d;}}
        return(byte)(chosen|((BitConverter.SingleToUInt32Bits(x)>>28)&8));
    }
    private static byte Encode8(float x)
    {
        byte chosen=0;float best=float.PositiveInfinity;
        for(byte i=0;i<127;i++){float d=Math.Abs(x-R.Fp8(i));if(d<best||(d==best&&(i&1)==0)){chosen=i;best=d;}}
        return chosen;
    }
    private static double Decode(byte[] codes,byte[] scales,int row,int c)
    {
        int within=c%16;byte b=codes[row*(Width/2)+(c/16)*8+within%8];int q=within<8?b&15:b>>4;
        return(double)R.Fp4(q)*R.Fp8(scales[row*(Width/16)+c/16]);
    }
    private static Fixture Make(string name)
    {
        Need(name is "exact-binary" or "broad-finite" or "ties-zero-and-last-column","Fixed three fixtures only.");bool exact=name=="exact-binary";
        float inputD=name=="broad-finite"?1.25f:1,weightD=exact?2:1.75f,alpha=1f/(inputD*weightD);
        var input=new float[Count];var wc=new byte[N*Width/2];var ws=new byte[N*Width/16];var codes=new byte[Count/2];var scales=new byte[Count/16];
        for(int t=0;t<Rows;t++)for(int c=0;c<Width;c++)
        {
            float v=exact?R.Fp4((c+t*3)%16):name=="broad-finite"?Bf(((c*37+t*11)%521-260)/31f):new[]{0f,BitConverter.UInt32BitsToSingle(0x80000000u),.25f,.75f,1.25f,1.75f,2.5f,3.5f,5f,-.25f,-.75f,-1.25f,-1.75f,-2.5f,-3.5f,6f}[c%16];
            input[t*Width+c]=c%16==15&&name!="broad-finite"?6:v;
        }
        for(int g=0;g<scales.Length;g++)
        {
            float max=0;for(int j=0;j<16;j++)max=Math.Max(max,Math.Abs(input[g*16+j]));byte s=Encode8((inputD*max)/6f);scales[g]=s;
            for(int j=0;j<8;j++){float a=s==0?0:(input[g*16+j]*inputD)/R.Fp8(s),b=s==0?0:(input[g*16+j+8]*inputD)/R.Fp8(s);codes[g*8+j]=(byte)(Encode4(a)|(Encode4(b)<<4));}
        }
        foreach(int r in Active)for(int g=0;g<Width/16;g++)
        {
            ws[r*(Width/16)+g]=exact?(byte)32:(byte)(32+(r*7+g*3)%40);
            for(int j=0;j<8;j++)wc[r*(Width/2)+g*8+j]=(byte)((r*7+g*3+j*11)%16|(((r*13+g*5+j*7)%16)<<4));
        }
        var ideal=new double[OutputCount];var bounds=new double[OutputCount];float[]? exactValues=exact?new float[OutputCount]:null;
        double u=1d/16777216,gamma=(3*Width+32)*u/(1-(3*Width+32)*u);
        foreach(int r in Active)for(int t=0;t<Rows;t++)
        {
            double dot=0,absolute=0;for(int c=0;c<Width;c++){double p=Decode(wc,ws,r,c)*Decode(codes,scales,t,c);dot+=p;absolute+=Math.Abs(p);}
            int i=t*N+r;ideal[i]=dot*alpha;bounds[i]=gamma*Math.Abs(alpha)*absolute+2*u*Math.Abs(ideal[i]);
            if(exact){Need(absolute*64<16777216&&alpha==.5f,"Fixed binary lattice proof failed.");exactValues![i]=(float)ideal[i];Need(exactValues[i]==ideal[i],"Exact dot not F32 representable.");}
        }
        return new(name,input,wc,ws,codes,scales,inputD,weightD,ideal,bounds,exactValues);
    }
    private static Fixture DenseTiming()
    {
        var f=Make("broad-finite");uint state=0x6da3b791;
        uint Next(){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
        for(int i=0;i<f.WeightCodes.Length;i++)f.WeightCodes[i]=(byte)Next();for(int i=0;i<f.WeightScales.Length;i++)f.WeightScales[i]=(byte)(32+Next()%40);
        Array.Clear(f.Ideal);Array.Clear(f.Bound);double u=1d/16777216,gamma=(3*Width+32)*u/(1-(3*Width+32)*u);
        foreach(int r in Active)for(int t=0;t<Rows;t++)
        {double dot=0,absolute=0;for(int c=0;c<Width;c++){double p=Decode(f.WeightCodes,f.WeightScales,r,c)*Decode(f.Codes,f.Scales,t,c);dot+=p;absolute+=Math.Abs(p);}int i=t*N+r;f.Ideal[i]=dot*f.Alpha;f.Bound[i]=gamma*Math.Abs(f.Alpha)*absolute+2*u*Math.Abs(f.Ideal[i]);}
        return f with{Name="dense-pseudorandom-performance",WitnessOnly=true};
    }
    private sealed record Numeric(int Values,int Fp64Values,int Nonfinite,int Violations,int NonBf16,int ExactMismatches,double MaxFraction){public bool Passed=>Nonfinite+Violations+NonBf16+ExactMismatches==0;}
    private static Numeric Check(byte[] bytes,Fixture f,bool rounded)
    {
        Need(bytes.Length==OutputCount*4,"Every output must be present.");var values=MemoryMarshal.Cast<byte,float>(bytes);int nf=0,bad=0,bf=0,exact=0,checkedFp64=0;double fraction=0;
        for(int i=0;i<OutputCount;i++)
        {
            float v=values[i];if(!float.IsFinite(v)){nf++;continue;}if(rounded&&(BitConverter.SingleToUInt32Bits(v)&0xffff)!=0)bf++;
            if(f.WitnessOnly&&!Active.Contains(i%N))continue;checkedFp64++;
            double bound=f.Bound[i]+(rounded?(Math.Abs(f.Ideal[i])+f.Bound[i])/256+Math.ScaleB(1d,-133):0),error=Math.Abs(v-f.Ideal[i]);if(error>bound)bad++;fraction=Math.Max(fraction,bound==0?(error==0?0:double.PositiveInfinity):error/bound);
            if(f.Exact!=null&&BitConverter.SingleToUInt32Bits(v)!=BitConverter.SingleToUInt32Bits(rounded?Bf(f.Exact[i]):f.Exact[i]))exact++;
        }
        return new(OutputCount,checkedFp64,nf,bad,bf,exact,fraction);
    }
    private static void SelfTest()
    {
        Need(Active.Distinct().Count()==Active.Length&&Active.All(r=>r>=0&&r<N),"Active rows must be exact.");
        Need(Width%256==0&&N%128==0&&Geometry.Splits>0&&Width/256>=2*Geometry.Splits,"Only the fixed production partition geometries are admitted.");
        int splits=1;while(N/128*splits<170&&splits*2<=Width/256/2&&splits*2<=8)splits*=2;Need(splits==Geometry.Splits,"Fixed split must equal the actual production resolver.");
        foreach(string name in new[]{"exact-binary","broad-finite","ties-zero-and-last-column"})
        {var f=Make(name);Need(f.Codes.Length==Count/2&&f.Scales.Length==Count/16&&f.WeightCodes.Length+f.WeightScales.Length==N*Width*9/16,"Fixed payload dimensions differ.");Need(Check(Bytes(f.Ideal.Select(x=>(float)x).ToArray()),f,false).Passed,"Independent beforeconsumer FP64 oracle envelope failed.");}
        var dense=DenseTiming();Need(dense.WeightCodes.Count(x=>x!=0)>dense.WeightCodes.Length*.99&&dense.WeightScales.All(x=>x>=32&&x<72),"Dense timing payload must not be zero sparse.");
        var witness=Check(Bytes(dense.Ideal.Select(x=>(float)x).ToArray()),dense,false);Need(witness.Passed&&witness.Fp64Values==Rows*Active.Length,"Dense FP64 witness coverage must be explicit.");
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new{passed=true,CPU="PASS",GPU="NOT_RUN",role=Geometry.Name,fixtures=3,activeRows=Active.Length,fullValuesPerOutput=OutputCount,denseTimingFp64WitnessValues=Rows*Active.Length,shape=new[]{Rows,N,Width},splits=Geometry.Splits,consumer=Geometry.Consumer,componentKernelLaunches=54*NodesPerChain,oldDynamicShared=76800,candidateDynamicShared=43520}));
    }
}
