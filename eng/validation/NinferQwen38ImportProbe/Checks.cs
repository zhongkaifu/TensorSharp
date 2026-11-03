using System.Buffers.Binary;
using System.Security.Cryptography;

internal static class Checks
{
    internal static void Need(bool ok,string message){if(!ok)throw new InvalidDataException(message);}
    internal static string Hash(ReadOnlySpan<byte> bytes)=>Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    internal static string HashFile(string path){using var file=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();}
    internal static string Finish(IncrementalHash hash)=>Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    internal static float PositiveF32(uint bits)
    {
        float value=BitConverter.UInt32BitsToSingle(bits);Need(float.IsFinite(value)&&value>0,"Input calibration is not positive finite; raw bits are never repaired.");return value;
    }
    // Independent inverse, expressed in logical K positions and original five-axis storage order.
    // Does not call the converter's row mapping, nibble helper, or scale offset expression.
    internal static (byte[] Codes,byte[] Scales) Inverse(ReadOnlySpan<byte> nativeCodes,ReadOnlySpan<byte> nativeScales,int columns)
    {
        Need(columns>0&&columns%64==0&&columns<=65536&&nativeCodes.Length==128*(columns/2)&&nativeScales.Length==128*(columns/16),"Inverse tile geometry differs.");
        byte[] codes=new byte[nativeCodes.Length],scales=new byte[nativeScales.Length];int rowBytes=columns/2;
        for(int row=0;row<128;row++)for(int k=0;k<columns;k++)
        {
            int nativeAt=row*rowBytes+(k/16)*8+k%8;int nibble=(nativeCodes[nativeAt]>>(k%16<8?0:4))&15;
            codes[row*rowBytes+k/2]|=(byte)(nibble<<(k%2*4));
        }
        int at=0,groups=columns/16;
        for(int groupBlock=0;groupBlock<groups/4;groupBlock++)for(int lowRow=0;lowRow<32;lowRow++)for(int highRow=0;highRow<4;highRow++)for(int lowGroup=0;lowGroup<4;lowGroup++)
            scales[at++]=nativeScales[(highRow*32+lowRow)*groups+groupBlock*4+lowGroup];
        Need(at==scales.Length,"Inverse scale coverage differs.");return(codes,scales);
    }
    internal static void TokenFrame(IncrementalHash hash,int id,ReadOnlySpan<byte> bytes)
    {
        Span<byte> header=stackalloc byte[8];BinaryPrimitives.WriteInt32LittleEndian(header,id);BinaryPrimitives.WriteInt32LittleEndian(header[4..],bytes.Length);hash.AppendData(header);hash.AppendData(bytes);
    }
    internal static void SelfTest()
    {
        int assertions=0;void Yes(bool value,string message){Need(value,message);assertions++;}
        void No(Action action){bool rejected=false;try{action();}catch(InvalidDataException){rejected=true;}Yes(rejected,"Negative control accepted.");}
        foreach(int k in new[]{64,128,192})
        {
            int groups=k/16;byte[] nativeCodes=new byte[128*k/2],nativeScales=new byte[128*groups],expectedCodes=new byte[nativeCodes.Length],expectedScales=new byte[nativeScales.Length];
            for(int r=0;r<128;r++)for(int c=0;c<k;c++)
            {
                int q=(r*7+c*3)&15;expectedCodes[r*k/2+c/2]|=(byte)(q<<(4*(c&1)));
                nativeCodes[r*k/2+(c/16)*8+(c&7)]|=(byte)(q<<(c%16>=8?4:0));
            }
            for(int r=0;r<128;r++)for(int g=0;g<groups;g++)nativeScales[r*groups+g]=(byte)((r*13+g*17)%127);
            for(int r=0;r<128;r++)for(int g=0;g<groups;g++)expectedScales[(g/4)*512+(r%32)*16+(r/32)*4+g%4]=nativeScales[r*groups+g];
            var actual=Inverse(nativeCodes,nativeScales,k);Yes(actual.Codes.AsSpan().SequenceEqual(expectedCodes),"Logical-nibble inverse failed.");Yes(actual.Scales.AsSpan().SequenceEqual(expectedScales),"Five-axis scale inverse failed.");
            nativeCodes[7]^=16;Yes(!Inverse(nativeCodes,nativeScales,k).Codes.AsSpan().SequenceEqual(expectedCodes),"Nibble corruption not detected.");
            nativeScales[17]^=1;Yes(!Inverse(nativeCodes,nativeScales,k).Scales.AsSpan().SequenceEqual(expectedScales),"Scale corruption not detected.");
        }
        No(()=>Inverse(new byte[4095],new byte[512],64));No(()=>Inverse(new byte[4096],new byte[511],64));No(()=>Inverse(Array.Empty<byte>(),Array.Empty<byte>(),63));
        foreach(uint bits in new uint[]{1,0x00800000,0x3f800000,0x7f7fffff})Yes(float.IsFinite(PositiveF32(bits)),"Positive finite raw bits refused.");
        foreach(uint bits in new uint[]{0,0x80000000,0xbf800000,0x7f800000,0xff800000,0x7fc00001})No(()=>PositiveF32(bits));
        using var hash=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);TokenFrame(hash,3,new byte[]{0,255,1});
        Yes(Finish(hash)==Hash(new byte[]{3,0,0,0,3,0,0,0,0,255,1}),"Token frame hasher differs from explicit framing.");
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new{self_test_pass=true,CPU_assertions=assertions,GPU="NOT_RUN",model="NOT_RUN"}));
    }
}
