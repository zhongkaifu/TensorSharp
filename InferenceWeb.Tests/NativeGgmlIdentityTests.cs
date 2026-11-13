using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class NativeGgmlIdentityTests
{
    [Fact]
    public void MappedIdentityFindsLibraryUsedByNativeExecution()
    {
        // Exercise the production assembly's import resolver without allocating
        // a model or selecting a GPU. The identity helper must find that mapping.
        Assert.Equal(0, GgmlDeepSeek4Native.NPast(IntPtr.Zero));
        string mapped = TestGates.MappedNativeGgmlOpsPath();
        Assert.True(Path.IsPathFullyQualified(mapped), mapped);
        Assert.True(File.Exists(mapped), mapped);

        // Reopen the observed mapping, then independently ask dladdr which image
        // owns its actual exported function. No guessed search path participates.
        IntPtr module = NativeLibrary.Load(mapped);
        try
        {
            IntPtr symbol = NativeLibrary.GetExport(module, "TSGgml_Dsv4NPast");
            Assert.NotEqual(IntPtr.Zero, symbol);
            if (OperatingSystem.IsMacOS())
            {
                Assert.NotEqual(0, DlAddr(symbol, out DlInfo info));
                Assert.Equal(mapped, Marshal.PtrToStringUTF8(info.FileName));
            }
        }
        finally { NativeLibrary.Free(module); }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DlInfo
    {
        public IntPtr FileName;
        public IntPtr BaseAddress;
        public IntPtr SymbolName;
        public IntPtr SymbolAddress;
    }

    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "dladdr")]
    private static extern int DlAddr(IntPtr address, out DlInfo info);
}
