using System.Runtime.InteropServices;
using TensorSharp.Cuda;
namespace W8HeadProbe;
internal static partial class Native
{
    private const string Driver="libcuda.so.1";
    [LibraryImport(Driver,EntryPoint="cuMemAlloc_v2")]internal static partial int Alloc(out nint p,nuint bytes);
    [LibraryImport(Driver,EntryPoint="cuMemFree_v2")]internal static partial int Free(nint p);
    [LibraryImport(Driver,EntryPoint="cuMemcpyHtoD_v2")]internal static partial int HtoD(nint dst,nint src,nuint bytes);
    [LibraryImport(Driver,EntryPoint="cuMemcpyDtoH_v2")]internal static partial int DtoH(nint dst,nint src,nuint bytes);
    [LibraryImport(Driver,EntryPoint="cuMemsetD8_v2")]internal static partial int Fill(nint dst,byte value,nuint bytes);
    [LibraryImport(Driver,EntryPoint="cuMemGetAddressRange_v2")]internal static partial int Range(out nint allocation,out nuint bytes,nint pointer);
    [LibraryImport(Driver,EntryPoint="cuCtxSetCurrent")]internal static partial int SetContext(nint context);
    [LibraryImport(Driver,EntryPoint="cuCtxSynchronize")]internal static partial int SyncContext();
    [LibraryImport(Driver)]internal static partial int cuEventCreate(out nint e,uint flags);
    [LibraryImport(Driver)]internal static partial int cuEventRecord(nint e,nint stream);
    [LibraryImport(Driver)]internal static partial int cuEventSynchronize(nint e);
    [LibraryImport(Driver)]internal static partial int cuEventElapsedTime(out float ms,nint a,nint b);
    [LibraryImport(Driver,EntryPoint="cuEventDestroy_v2")]internal static partial int DestroyEvent(nint e);
    [LibraryImport(Driver)]internal static partial int cuDeviceGet(out int device,int ordinal);
    [LibraryImport(Driver)]internal static partial int cuDeviceGetUuid_v2(out Uuid uuid,int device);
    [LibraryImport(Driver)]internal static partial int cuDriverGetVersion(out int version);
    [StructLayout(LayoutKind.Sequential)]internal unsafe struct Uuid{internal fixed byte Bytes[16];}
    internal static unsafe object DeviceIdentity(){Ok(cuDeviceGet(out int device,0));Ok(cuDeviceGetUuid_v2(out Uuid uuid,device));Ok(cuDriverGetVersion(out int driver));byte* p=uuid.Bytes;return new{visible_ordinal=0,device,driver_version=driver,uuid_hex=Convert.ToHexString(new ReadOnlySpan<byte>(p,16)).ToLowerInvariant()};}
    internal static void Ok(int r){if(r!=0)throw new InvalidOperationException("CUDA driver result "+r);}
}
internal sealed unsafe class DeviceBuffer:IDisposable
{
    private nint _base;internal const int Halo=256;internal long Length{get;}
    internal nint Pointer{get{ObjectDisposedException.ThrowIf(_base==0,this);return _base+Halo;}}
    internal DeviceBuffer(long n){Checks.Need(n>0&&n<long.MaxValue-2*Halo,"allocation arithmetic");Length=n;Native.Ok(Native.Alloc(out _base,(nuint)(n+2*Halo)));try{Native.Ok(Native.Fill(_base,0xd3,(nuint)(n+2*Halo)));VerifyPhysicalRange();}catch{Native.Free(_base);_base=0;throw;}}
    internal void VerifyPhysicalRange(){Native.Ok(Native.Range(out nint first,out nuint bytes,Pointer));Checks.Need(first==_base&&(ulong)bytes>=(ulong)Length+2*Halo,"actual driver allocation does not own buffer+guards");}
    internal void Upload(ReadOnlySpan<byte> data){Checks.Need(data.Length==Length,"upload length");fixed(byte*p=data)Native.Ok(Native.HtoD(Pointer,(nint)p,(nuint)Length));}
    internal void Upload(nint data){Native.Ok(Native.HtoD(Pointer,data,(nuint)Length));}
    internal void Paint(byte p){Native.Ok(Native.Fill(Pointer,p,(nuint)Length));}
    internal byte[] Read(){var result=new byte[checked((int)Length)];fixed(byte*p=result)Native.Ok(Native.DtoH((nint)p,Pointer,(nuint)Length));Guard();return result;}
    internal void Guard(){Span<byte> data=stackalloc byte[Halo*2];fixed(byte*p=data){Native.Ok(Native.DtoH((nint)p,_base,Halo));Native.Ok(Native.DtoH((nint)(p+Halo),Pointer+(nint)Length,Halo));}Checks.Guard(data);VerifyPhysicalRange();}
    internal void NegativeGuardControl(){Native.Ok(Native.Fill(Pointer+(nint)Length,0,1));bool rejected=false;try{Guard();}catch(InvalidDataException){rejected=true;}Checks.Need(rejected,"guard negative control accepted corrupt bytes");Native.Ok(Native.Fill(Pointer+(nint)Length,0xd3,1));Guard();}
    public void Dispose(){if(_base==0)return;Native.Ok(Native.Free(_base));_base=0;}
}
internal sealed class EventInterval:IDisposable
{
    private nint _a,_b;internal EventInterval(){Native.Ok(Native.cuEventCreate(out _a,0));try{Native.Ok(Native.cuEventCreate(out _b,0));}catch{Native.DestroyEvent(_a);throw;}}
    internal void Start(nint stream)=>Native.Ok(Native.cuEventRecord(_a,stream));
    internal float Finish(nint stream){Native.Ok(Native.cuEventRecord(_b,stream));Native.Ok(Native.cuEventSynchronize(_b));Native.Ok(Native.cuEventElapsedTime(out float ms,_a,_b));Checks.Need(float.IsFinite(ms)&&ms>0,"invalid full pipeline event interval");return ms;}
    public void Dispose(){if(_a!=0){Native.Ok(Native.DestroyEvent(_a));_a=0;}if(_b!=0){Native.Ok(Native.DestroyEvent(_b));_b=0;}}
}
