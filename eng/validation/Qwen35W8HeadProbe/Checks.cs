using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
namespace W8HeadProbe;
internal static class Checks
{
    internal const int K=5120,Vocab=248320;
    internal const long Codes=1271398400,ScaleOffset=Codes,WeightBytes=1350860800;
    internal const string OriginalRunSha="b6086e5b6a4f060f2f953ab9023d8ed3c4a418619ad25ea0fcc2ca4495d77e61";
    internal const string OriginalInventorySha="97851b4998227808ac856644555e04f595b2aa6bcb3b58d6939e4027b2986c22";
    internal const string OriginalBenchSha="d36c77b1b3828971a9cd16cd652bc7877772ad957aad57d13f77d0de09fd729a";
    internal static readonly int[] DirectRows={1,7,8,14,16,28,32,33};
    internal static void Need(bool x,string message){if(!x)throw new InvalidDataException(message);}
    internal static string Sha(string path){using var f=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant();}
    internal static string Sha(ReadOnlySpan<byte>x)=>Convert.ToHexString(SHA256.HashData(x)).ToLowerInvariant();
    internal static uint Hash(uint v){unchecked{v^=v>>16;v*=0x7feb352d;v^=v>>15;v*=0x846ca68b;return v^(v>>16);}}
    internal static byte Code(int r,int c)=>unchecked((byte)Hash((uint)(r*131+c)));
    internal static ushort Scale(int r,int g)=>(ushort)(0x3000+Hash((uint)(r*173+g))%0x0800);
    internal static uint InputBits(int i){uint bits=0x3d000000+Hash((uint)i)%0x01000000;return (i&1)!=0?bits|0x80000000:bits;}
    internal static ushort Bf16(uint bits)
    {
        Need((bits&0x7f800000)!=0x7f800000,"nonfinite input");
        ushort result=unchecked((ushort)((bits+0x7fff+((bits>>16)&1))>>16));
        Need((result&0x7f80)!=0x7f80&&((result&0x7f80)!=0||(result&127)==0),"unsupported BF16 cast");return result;
    }
    internal static float Value(ushort b)=>BitConverter.UInt32BitsToSingle((uint)b<<16);
    internal static int OracleRows(int n)
    {
        Need(n>=1&&n<=33,"unsupported rows");if(DirectRows.Contains(n))return n;
        return n<=6?7:n<=13?14:n==15?16:n<=27?28:32;
    }
    internal static double Allowed(double reference,double sumAbs)
    {
        Need(double.IsFinite(reference)&&double.IsFinite(sumAbs)&&sumAbs>=0,"invalid reference");
        const double u32=1.0/16777216,u64=1.0/9007199254740992,n=224;
        double error=((n*u32)/(1-n*u32)+(K*u64)/(1-K*u64))*sumAbs+n*Math.ScaleB(1,-149);
        return error+Math.ScaleB(1,-8)*(Math.Abs(reference)+error)+Math.ScaleB(1,-134);
    }
    internal static void Output(ReadOnlySpan<byte> actual,ReadOnlySpan<byte> expected,ReadOnlySpan<byte> reference,ReadOnlySpan<byte> sumAbs)
    {
        Need(actual.Length>0&&actual.Length==expected.Length&&actual.Length%2==0&&reference.Length==actual.Length*4&&sumAbs.Length==reference.Length,"output full lengths");
        Need(actual.SequenceEqual(expected),"full BF16 bits differ from frozen original C++ oracle");
        for(int i=0;i<actual.Length/2;i++)
        {
            float value=Value(BinaryPrimitives.ReadUInt16LittleEndian(actual[(2*i)..]));
            double r=BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(reference[(8*i)..]));
            double s=BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(sumAbs[(8*i)..]));
            Need(float.IsFinite(value)&&Math.Abs((double)value-r)<=Allowed(r,s),"unchanged preregistered W8 bound failed");
        }
    }
    internal static void Guard(ReadOnlySpan<byte> x){foreach(byte b in x)Need(b==0xd3,"device uint-painted guard changed");}
    internal static void SelfTest()
    {
        int assertions=0;void Eq(bool b){Need(b,"CPU self-test");assertions++;}
        foreach(int n in Enumerable.Range(1,33)){int r=OracleRows(n);Eq(r>=n&&DirectRows.Contains(r)&&((n<=32&&r<=32)||n==33));}
        Eq(Bf16(0x3f808000)==0x3f80);Eq(Bf16(0x3f818000)==0x3f82);Eq(Bf16(0x80000000)==0x8000);
        foreach(uint bad in new uint[]{0x7f800000,0x7fc00000,0x00010000}){bool reject=false;try{Bf16(bad);}catch(InvalidDataException){reject=true;}Eq(reject);}
        byte[] output={0x80,0x3f};byte[] reference=BitConverter.GetBytes(1.0),abs=BitConverter.GetBytes(1.0);Output(output,output,reference,abs);assertions++;
        foreach(byte[] bad in new[]{new byte[]{0x81,0x3f},new byte[]{0xc0,0x7f}}){bool reject=false;try{Output(bad,output,reference,abs);}catch(InvalidDataException){reject=true;}Eq(reject);}
        Guard(new byte[]{0xd3,0xd3});bool guard=false;try{Guard(new byte[]{0xd3,0});}catch(InvalidDataException){guard=true;}Eq(guard);
        foreach(int n in new[]{0,34}){bool rejected=false;try{OracleRows(n);}catch(InvalidDataException){rejected=true;}Eq(rejected);}
        foreach(double bad in new[]{double.NaN,double.PositiveInfinity,-1.0}){bool rejected=false;try{Allowed(0,bad);}catch(InvalidDataException){rejected=true;}Eq(rejected);}
        bool bound=false;try{Output(output,output,BitConverter.GetBytes(2.0),abs);}catch(InvalidDataException){bound=true;}Eq(bound);
        bool length=false;try{Output(output,output,reference,Array.Empty<byte>());}catch(InvalidDataException){length=true;}Eq(length);
        Eq(WeightBytes==Codes+248320L*160*2);Eq(Code(0,0)==0);Eq(Scale(0,0)==0x3000);Eq(InputBits(0)==0x3d000000);
        Console.WriteLine(JsonSerializer.Serialize(new{self_test_pass=true,CPU_assertions=assertions,gpu_not_run=true,complete=false}));
    }
}
