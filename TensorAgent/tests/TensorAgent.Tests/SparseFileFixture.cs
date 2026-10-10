// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace TensorAgent.Tests;

/// <summary>Catalog metadata fixtures need the real logical length, not model payload storage.</summary>
internal static class SparseFileFixture
{
    internal static void Create(string path, long length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        using FileStream stream = File.Create(path);
        // Unix SetLength leaves an unwritten hole. NTFS requires explicit sparse
        // marking first; otherwise a catalog test can reserve tens of GB on disk.
        if (OperatingSystem.IsWindows()
            && !DeviceIoControl(stream.SafeFileHandle, 0x000900C4, IntPtr.Zero, 0,
                IntPtr.Zero, 0, out _, IntPtr.Zero))
            throw new IOException("Could not create a sparse catalog fixture.",
                new Win32Exception(Marshal.GetLastWin32Error()));
        stream.SetLength(length);
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool DeviceIoControl(SafeFileHandle device, uint controlCode,
        IntPtr input, uint inputBytes, IntPtr output, uint outputBytes,
        out uint returnedBytes, IntPtr overlapped);
}
