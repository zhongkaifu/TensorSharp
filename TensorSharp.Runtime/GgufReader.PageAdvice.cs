// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Runtime;

public partial class GgufFile
{
    /// <summary>Best-effort Linux release of a tensor's source pages after its
    /// caller has completed a persistent device upload. Keeps the mapping and
    /// file handle alive: subsequent host reads still return the original bytes.
    /// Both the tensor identity and the exact mapped pointer/length must match.
    /// This never accepts an anonymous or copied weight buffer.</summary>
    public bool TryReleaseMappedTensorPages(GgufTensorInfo info, IntPtr data, long bytes)
        => TryReleaseMappedTensorPages(info, data, bytes, out _);

    public unsafe bool TryReleaseMappedTensorPages(GgufTensorInfo info, IntPtr data, long bytes, out long advisedBytes)
    {
        advisedBytes = 0;
        if (!OperatingSystem.IsLinux() ||
            !TryGetMappedTensorSourcePages(info, data, bytes, out var owner, out long start, out long length))
            return false;
        // Drop this process's PTE references before asking to drop file-cache
        // pages. fadvise alone cannot reclaim pages still mapped by a reader;
        // madvise alone need not release the cgroup's file-cache charge.
        // Round inward so a neighboring CPU tensor's shared page stays hot.
        try
        {
            if (ReleaseMappedPages(owner._mappedBase + start, (nuint)length, 4) != 0) return false;
            if (ReleaseFilePages(owner._stream.SafeFileHandle, start, length, 4) != 0) return false;
            advisedBytes = length;
            return true;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
    }

    /// <summary>Where a tensor's bytes live: the shard file that holds it, the byte offset of its data in that
    /// file and its byte count. False for a tensor this model does not have.</summary>
    public bool TryGetTensorFileRange(string name, out string path, out long offset, out long bytes)
    {
        path = string.Empty;
        offset = bytes = 0;
        if (name == null || !Tensors.TryGetValue(name, out var info))
            return false;
        GgufFile owner = OwnerOf(info);
        path = owner._path;
        offset = checked(owner.DataOffset + (long)info.Offset);
        bytes = GetTensorByteCount(info);
        return true;
    }

    internal unsafe bool TryGetMappedTensorSourcePages(GgufTensorInfo info, IntPtr data, long bytes,
        out GgufFile owner, out long start, out long length)
    {
        owner = null;
        start = length = 0;
        if (info == null || data == IntPtr.Zero || bytes <= 0 ||
            !Tensors.TryGetValue(info.Name, out var registered) || !ReferenceEquals(info, registered))
            return false;
        owner = OwnerOf(info);
        if (owner._mappedBase == null || owner._lockedBase != null || owner._stream == null ||
            bytes != GetTensorByteCount(info)) return false;
        long offset = checked(owner.DataOffset + (long)info.Offset);
        if (offset < 0 || offset > owner._stream.Length || bytes > owner._stream.Length - offset ||
            data != (IntPtr)(owner._mappedBase + offset)) return false;
        return TryGetCompletePageRange(offset, bytes, Environment.SystemPageSize, out start, out length);
    }

    internal static bool TryGetCompletePageRange(long offset, long bytes, int pageSize,
        out long start, out long length)
    {
        start = length = 0;
        if (offset < 0 || bytes <= 0 || pageSize <= 0 || offset > long.MaxValue - bytes) return false;
        long end = offset + bytes;
        long remainder = offset % pageSize;
        long padding = remainder == 0 ? 0 : pageSize - remainder;
        if (padding >= bytes) return false;
        start = offset + padding;
        length = end - end % pageSize - start;
        return length > 0;
    }

    [LibraryImport("libc", EntryPoint = "madvise", SetLastError = true)]
    private static unsafe partial int ReleaseMappedPages(void* address, nuint length, int advice);

    [LibraryImport("libc", EntryPoint = "posix_fadvise")]
    private static partial int ReleaseFilePages(SafeFileHandle handle, long offset, long length, int advice);
}
