// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using TensorSharp.Cuda;
using static Support;

internal static class State
{
    internal sealed record Metadata(int TargetPosition, int Frontier, int FirstPosition, int PendingBytes, string PendingSha256, string Namespace);
    internal sealed record Part(string Role, long ByteOffset, long Bytes, string Sha256);
    internal sealed record Snapshot(int Position, int RopeDelta, int WrittenKeys, Metadata Draft, Part[] Parts);
    // Reflection is intentional: the same probe binary must run with the old
    // runtime, which cannot grant a new probe an InternalsVisibleTo declaration.
    private delegate int Copy(IntPtr destination, IntPtr source, UIntPtr bytes);
    private static readonly Lazy<Copy> CopyToHost = new(() =>
    {
        Type type = typeof(Q35CudaEngine).Assembly.GetType("TensorSharp.Cuda.Interop.CudaDriverApi", true)!;
        var method = type.GetMethod("cuMemcpyDtoH", BindingFlags.Public | BindingFlags.Static)!;
        return method.CreateDelegate<Copy>();
    });
    internal static unsafe byte[] Read(IntPtr p, int bytes)
    {
        Need(p != IntPtr.Zero && bytes > 0, "Readable nonempty device range required."); var b = new byte[bytes];
        fixed (byte* dst = b) Need(CopyToHost.Value((IntPtr)dst, p, (UIntPtr)bytes) == 0, "Device diagnostic read failed."); return b;
    }
    private static unsafe Part Digest(string role, IntPtr p, long bytes, byte[] buffer, long byteOffset = 0)
    {
        Need(p != IntPtr.Zero && bytes > 0, "Invalid written-state range."); using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (long offset = 0; offset < bytes;)
        {
            int count = (int)Math.Min(buffer.Length, bytes - offset);
            fixed (byte* dst = buffer) Need(CopyToHost.Value((IntPtr)dst, p + (nint)offset, (UIntPtr)count) == 0, "Device state read failed.");
            hash.AppendData(buffer, 0, count); offset += count;
        }
        return new(role, byteOffset, bytes, Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant());
    }
    internal static Metadata Meta(Q35CudaEngine e, Q35CudaEngine.Slot s)
    {
        var capture = typeof(Q35CudaEngine).GetMethod("CaptureNinferDraftSlotState", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidDataException("Original native draft metadata diagnostic unavailable.");
        object capsule = capture.Invoke(e, new object[] { s }) ?? throw new InvalidDataException("Native draft capsule absent.");
        var pending = Member<ReadOnlyMemory<byte>>(capsule, "Pending");
        var m = new Metadata(Member<int>(capsule, "TargetPosition"), Member<int>(capsule, "Frontier"), Member<int>(capsule, "FirstPosition"), pending.Length, Hash(pending.Span), Member<string>(capsule, "Namespace"));
        Need(m.TargetPosition == s.NPast && m.FirstPosition >= Math.Max(0, m.Frontier - 2048) && m.FirstPosition <= m.Frontier
            && (m.PendingBytes == 0 ? m.Frontier == s.NPast : m.PendingBytes == 51200 && m.Frontier == s.NPast - 1), "Native pending/frontier invariant differs.");
        return m;
    }
    internal static Snapshot Capture(Q35CudaEngine e, Q35CudaEngine.Slot s, int writtenKeys, bool includeKv = true)
    {
        e.Synchronize(); Metadata meta = Meta(e, s); var parts = new List<Part>(); byte[] buffer = new byte[1 << 20];
        var conv = Member<IntPtr[]>(s, "Conv"); var recurrent = Member<IntPtr[]>(s, "State");
        Need(conv.Length == 48 && recurrent.Length == 48, "Original recurrent geometry required.");
        for (int l = 0; l < 48; l++)
        {
            parts.Add(Digest($"conv/{l}", conv[l], 3L * 10240 * 4, buffer));
            parts.Add(Digest($"state/{l}", recurrent[l], 48L * 128 * 128 * 4, buffer));
        }
        if (includeKv)
        {
            var blocks = Member<List<IntPtr>>(s, "Blocks");
            Need(writtenKeys >= s.NPast && writtenKeys > 0 && (writtenKeys + 255) / 256 <= blocks.Count
                && e.SlotKvBytes(s) / blocks.Count == 16L * 2 * 4 * 256 * 256 * 2, "Original BF16 KV geometry/defined extent differs.");
            for (int b = 0; b * 256 < writtenKeys; b++)
            {
                int live = Math.Min(256, writtenKeys - b * 256);
                if (live == 256) parts.Add(Digest($"kv/{b}/complete", blocks[b], 16L * 2 * 4 * 256 * 256 * 2, buffer));
                else for (int l = 0; l < 16; l++) for (int p = 0; p < 2; p++) for (int h = 0; h < 4; h++)
                {
                    long offset = (((long)l * 2 + p) * 4 + h) * 256 * 256 * 2;
                    parts.Add(Digest($"kv/{b}/{l}/{p}/{h}/tokens0-{live}", blocks[b] + (nint)offset, (long)live * 256 * 2, buffer, offset));
                }
            }
        }
        var keys = Member<IntPtr[]>(s, "RingK"); var values = Member<IntPtr[]>(s, "RingV");
        Need(keys.Length == 5 && values.Length == 5 && meta.Frontier - meta.FirstPosition <= 2048, "Original native ring geometry differs.");
        // Only encoded logical history is defined. RingPos is not written/read
        // by this native DFlash adapter and is deliberately not certified.
        for (int l = 0; l < 5; l++) for (int p = 0; p < 2; p++) for (int h = 0; h < 8; h++)
        {
            int first = meta.FirstPosition;
            while (first < meta.Frontier)
            {
                int count = Math.Min(meta.Frontier - first, 2048 - (first & 2047));
                long offset = ((long)h * 2048 + (first & 2047)) * 128 * 2;
                parts.Add(Digest($"ring/{l}/{p}/{h}/positions{first}-{first + count}", (p == 0 ? keys[l] : values[l]) + (nint)offset, (long)count * 128 * 2, buffer, offset)); first += count;
            }
        }
        return new(s.NPast, s.RopeDelta, includeKv ? writtenKeys : 0, meta, parts.ToArray());
    }
    internal static void Taps(Q35CudaEngine e, Store store, string id, int pass, int rows)
    {
        e.Synchronize(); byte[] bytes = Read(Member<IntPtr>(e, "_ndFeatures"), checked(rows * 51200));
        foreach (ushort v in MemoryMarshal.Cast<byte, ushort>(bytes)) Need((v & 0x7f80) != 0x7f80, "Nonfinite native BF16 tap.");
        store.Raw(id + "/native-taps", pass, 0, "bf16", bytes); store.TapCaptures++;
    }
    internal sealed record Counter(bool? Enabled, long? Capture, long? Replay, long? Gemm, long? Plain,
        long? Initial, long? LegacyPost, long? Completed, long TrunkCapture, long TrunkReplay);
    internal static Counter Counters(Q35CudaEngine e) => new(
        Optional(e, "NinferResidualNormEnabled") as bool?, Optional(e, "NinferResidualNormCaptureCount") as long?,
        Optional(e, "NinferResidualNormReplayCount") as long?, Optional(e, "NinferResidualNormGemmHostSubmissions") as long?,
        Optional(e, "NinferResidualNormPlainHostSubmissions") as long?, Optional(e, "NinferResidualNormInitialHostSubmissions") as long?,
        Optional(e, "NinferResidualNormLegacyPostHostSubmissions") as long?, Optional(e, "NinferResidualNormCompletedVerifies") as long?,
        e.NinferDecodeGraphCaptureCount, e.NinferDecodeGraphReplayCount);
    internal static void VerifyTransition(Counter a, Counter b, string mode, int rows)
    {
        Need(b.TrunkReplay == a.TrunkReplay + 1, "Verify did not replay its actual trunk graph.");
        if (mode == "old") { Need(a.Enabled == null && b.Enabled == null && a.Capture == null && b.Replay == null, "Old runtime unexpectedly contains candidate diagnostics."); return; }
        Need(a.Enabled == (mode == "on") && b.Enabled == a.Enabled && a.Capture.HasValue && a.Replay.HasValue
            && a.Gemm.HasValue && a.Plain.HasValue && a.Initial.HasValue && a.LegacyPost.HasValue && a.Completed.HasValue, "Candidate diagnostics absent/wrong mode.");
        long c = b.Capture!.Value - a.Capture!.Value, v = mode == "on" && rows == 8 ? 1 : 0;
        Need(c >= 0 && c <= v && b.Replay == a.Replay + v && b.Completed == a.Completed + v
            && b.Gemm == a.Gemm + 125*c && b.Plain == a.Plain + 124*c + v
            && b.Initial == a.Initial + c && b.LegacyPost == a.LegacyPost + 3*c, "Candidate pass/capture/final norm counters differ.");
        if (mode == "off") Need(b.Capture == 0 && b.Replay == 0 && b.Gemm == 0 && b.Plain == 0 && b.Completed == 0, "OFF path submitted candidate work.");
    }
    internal static Snapshot Numeric(Snapshot s) => s with { Draft = s.Draft with { Namespace = "" } };
}
