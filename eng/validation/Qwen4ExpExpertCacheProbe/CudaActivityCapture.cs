// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;

// Optional diagnostic tool; never loaded by product/runtime builds.
internal sealed class CudaActivityCapture
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int StringCall([MarshalAs(UnmanagedType.LPUTF8Str)] string text);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int StopCall();
    private readonly StringCall _mark;
    private readonly StopCall _stop;
    private bool _stopped;

    public CudaActivityCapture(string library, string output)
    {
        if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("CUPTI recorder currently validated on Linux only.");
        // CUPTI retains callback addresses until process shutdown. Deliberately
        // retain the library handle for this short-lived probe's whole lifetime.
        IntPtr handle = NativeLibrary.Load(Path.GetFullPath(library));
        var start = Marshal.GetDelegateForFunctionPointer<StringCall>(NativeLibrary.GetExport(handle, "TsCudaTraceStart"));
        _mark = Marshal.GetDelegateForFunctionPointer<StringCall>(NativeLibrary.GetExport(handle, "TsCudaTraceMark"));
        _stop = Marshal.GetDelegateForFunctionPointer<StopCall>(NativeLibrary.GetExport(handle, "TsCudaTraceStop"));
        if (start(Path.GetFullPath(output)) != 1)
        {
            _stop();
            throw new InvalidOperationException("CUPTI activity recording could not start; see diagnostic output.");
        }
    }
    public void Mark(string label)
    {
        if (_stopped || _mark(label) != 1) throw new InvalidOperationException("CUPTI marker failed.");
    }
    public void Stop()
    {
        if (_stopped) return;
        _stopped = true;
        if (_stop() != 1) throw new InvalidOperationException("CUPTI recording failed or dropped records; do not use partial data as a passing profile.");
    }
}
