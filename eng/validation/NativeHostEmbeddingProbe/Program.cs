// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Validation;

// This is a synthetic placement/row-byte check. It does not run model sampling,
// establish serving correctness, measure model VRAM savings, or time inference.
internal static class Program
{
    private const int GuardBytes = 256, GuardBits = 0x4e48a17e, SourcePrefix = 18;
    private static readonly int[] Poisons = { unchecked((int)0x7fc148a1), unchecked((int)0x7fc248a1), unchecked((int)0x7fc148a1) };
    private static readonly ushort[] ScaleBits =
    {
        0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x8002, 0x03ff, 0x83ff,
        0x0400, 0x8400, 0x3555, 0xb555, 0x3c00, 0xbc00, 0x3c01, 0xbc01,
        0x3c04, 0xbc04, 0x3c0c, 0xbc0c, 0x0003, 0x8003, 0x0200, 0x8200,
        0x3bff, 0xbbff, 0x4000, 0xc000, 0x7bff, 0xfbff, 0x7bfe, 0xfbfe
    };
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Fixture(string Id, int Width, int Vocabulary, byte[] Payload, int[] TokenIds, float[] Expected);

    private static void Need(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static byte[] Bytes<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static void Save(string path, object value)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10);
    }
    private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, parts.Length - 1).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored evidence output required.");
        Need(!Path.Exists(path) && !IsLink(path), "Fresh output required.");
        for (string? parent = Path.GetDirectoryName(path); parent != null; parent = Path.GetDirectoryName(parent))
            Need(!IsLink(parent) && !File.Exists(parent), "Aliased output parent refused.");
    }

    // Decode binary16 using its mathematical integer/exponent representation;
    // do not call Half conversion or the implementation's BF16 rounding routine.
    private static double HalfValue(ushort bits)
    {
        int exponent = (bits >> 10) & 31, fraction = bits & 1023;
        Need(exponent != 31, "The finite-scale oracle rejects NaN and infinity.");
        double magnitude = exponent == 0 ? Math.ScaleB(fraction, -24) : Math.ScaleB(1024 + fraction, exponent - 25);
        return (bits & 0x8000) == 0 ? magnitude : -magnitude;
    }
    private static float Oracle(sbyte code, ushort scale, out bool midpoint)
    {
        double exact = code * HalfValue(scale); midpoint = false;
        if (exact == 0)
        {
            bool negative = (code < 0) ^ ((scale & 0x8000) != 0);
            return BitConverter.UInt32BitsToSingle(negative ? 0x80000000u : 0u);
        }
        // Every finite int8 * binary16 product is exactly representable in F32.
        float f32 = (float)exact;
        Need((double)f32 == exact && float.IsFinite(f32), "The independent exact product premise failed.");
        uint firstBits = BitConverter.SingleToUInt32Bits(f32) & 0xffff0000u;
        float first = BitConverter.UInt32BitsToSingle(firstBits);
        float second = BitConverter.UInt32BitsToSingle(firstBits + 0x10000u);
        double firstDistance = Math.Abs(exact - first), secondDistance = Math.Abs(exact - second);
        midpoint = firstDistance == secondDistance;
        return firstDistance < secondDistance || (midpoint && (firstBits & 0x10000u) == 0) ? first : second;
    }

    private static byte[] Payload(int width, ushort[] scales, bool perGroup)
    {
        int vocabulary = scales.Length, codeBytes = checked(vocabulary * width), groups = width / 32;
        var payload = new byte[checked(codeBytes + vocabulary * groups * 2)];
        for (int row = 0; row < vocabulary; row++)
        {
            for (int c = 0; c < width; c++)
                payload[row * width + c] = (byte)(perGroup ? (c * 37 + row * 17 + (c / 256) * 73) & 255 : c ^ 0x80);
            for (int group = 0; group < groups; group++)
            {
                ushort scale = perGroup ? scales[(row * 11 + group * 7) % scales.Length] : scales[row];
                BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(codeBytes + (row * groups + group) * 2), scale);
            }
        }
        return payload;
    }
    private static float[] Expected(byte[] payload, int width, int vocabulary, int[] ids)
    {
        var values = new float[checked(ids.Length * width)]; int codeBytes = checked(vocabulary * width), groups = width / 32;
        for (int row = 0; row < ids.Length; row++) for (int c = 0; c < width; c++)
        {
            sbyte code = unchecked((sbyte)payload[ids[row] * width + c]);
            ushort scale = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(codeBytes + (ids[row] * groups + c / 32) * 2));
            values[row * width + c] = Oracle(code, scale, out _);
        }
        return values;
    }
    private static Fixture[] Fixtures()
    {
        var fixtures = new List<Fixture>();
        foreach (int width in new[] { 128, 256, 5120 })
        {
            byte[] payload = Payload(width, ScaleBits, perGroup: width != 256);
            foreach (int batch in new[] { 1, 7, 33 })
            {
                int[] ids = batch switch
                {
                    1 => new[] { 31 },
                    7 => new[] { 31, 0, 15, 15, 1, 30, 2 },
                    _ => Enumerable.Range(0, 32).Select(i => (i * 13 + 7) & 31).Append(7).ToArray()
                };
                fixtures.Add(new($"w{width}-t{batch}", width, ScaleBits.Length, payload, ids, Expected(payload, width, ScaleBits.Length, ids)));
            }
        }
        return fixtures.ToArray();
    }

    private sealed class Source : ICudaWeightSource, IDisposable
    {
        private readonly byte[] _guarded;
        internal int Reads { get; private set; }
        internal bool Disposed { get; private set; }
        internal Source(byte[] payload)
        {
            _guarded = Enumerable.Repeat((byte)0xa5, checked(SourcePrefix + payload.Length + 30)).ToArray();
            payload.CopyTo(_guarded, SourcePrefix);
        }
        public void Read(long offset, IntPtr destination, long bytes)
        {
            ObjectDisposedException.ThrowIf(Disposed, this);
            if (offset < SourcePrefix || bytes < 0 || offset + bytes > _guarded.Length - 30) throw new ArgumentOutOfRangeException(nameof(offset));
            Reads++; Marshal.Copy(_guarded, checked((int)offset), destination, checked((int)bytes));
        }
        public void Dispose() => Disposed = true;
    }
    private static CudaNinferProjectionDesc Descriptor(Source source, int width, int vocabulary, int payloadBytes) =>
        new(source, SourcePrefix, payloadBytes, vocabulary, width, CudaNinferProjectionFormat.W8G32, allowA4: false);
    private static byte[] Guard(byte[] body)
    {
        var bytes = new byte[checked(body.Length + 2 * GuardBytes)];
        for (int i = 0; i < GuardBytes / 4; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), GuardBits);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(GuardBytes + body.Length + i * 4), GuardBits);
        }
        body.CopyTo(bytes, GuardBytes); return bytes;
    }
    private static bool GuardsMatch(ReadOnlySpan<byte> actual, ReadOnlySpan<byte> expected) =>
        actual[..GuardBytes].SequenceEqual(expected[..GuardBytes]) && actual[^GuardBytes..].SequenceEqual(expected[^GuardBytes..]);
    private static int Mismatches(ReadOnlySpan<byte> actual, ReadOnlySpan<byte> expected)
    {
        Need(actual.Length == expected.Length && actual.Length % 4 == 0, "F32 vectors have inconsistent sizes."); int count = 0;
        for (int i = 0; i < actual.Length; i += 4)
            if (BinaryPrimitives.ReadInt32LittleEndian(actual[i..]) != BinaryPrimitives.ReadInt32LittleEndian(expected[i..])) count++;
        return count;
    }
    private static object Coverage(Fixture fixture)
    {
        var codes = new HashSet<byte>(); var scales = new HashSet<ushort>(); var pairs = new HashSet<int>();
        int codeBytes = fixture.Vocabulary * fixture.Width, groups = fixture.Width / 32;
        long positiveZero = 0, negativeZero = 0, midpoint = 0, subnormalScaleValues = 0;
        foreach (int token in fixture.TokenIds) for (int c = 0; c < fixture.Width; c++)
        {
            byte code = fixture.Payload[token * fixture.Width + c];
            ushort scale = BinaryPrimitives.ReadUInt16LittleEndian(fixture.Payload.AsSpan(codeBytes + (token * groups + c / 32) * 2));
            codes.Add(code); scales.Add(scale); pairs.Add((scale << 8) | code);
            float value = Oracle(unchecked((sbyte)code), scale, out bool tie);
            uint bits = BitConverter.SingleToUInt32Bits(value);
            if (bits == 0) positiveZero++; if (bits == 0x80000000u) negativeZero++; if (tie) midpoint++;
            if ((scale & 0x7c00) == 0 && (scale & 1023) != 0) subnormalScaleValues++;
        }
        return new { distinctW8Codes = codes.Count, distinctF16ScaleBits = scales.Count, distinctCodeScalePairs = pairs.Count,
            positiveZeroResults = positiveZero, negativeZeroResults = negativeZero, exactMidpoints = midpoint, subnormalScaleValues,
            duplicateIds = fixture.TokenIds.Length - fixture.TokenIds.Distinct().Count() };
    }
    private static object CpuSelfTest()
    {
        var fixtures = Fixtures(); long fixtureValues = 0;
        foreach (var fixture in fixtures)
        {
            using var source = new Source(fixture.Payload);
            using var host = new CudaNinferHostEmbedding(Descriptor(source, fixture.Width, fixture.Vocabulary, fixture.Payload.Length));
            Need(ValidationSha256.Matches(host.PayloadSha256, Hash(fixture.Payload)), "CPU fixture payload digest mismatch.");
            int reads = source.Reads; source.Dispose();
            byte[] guarded = Guard(new byte[fixture.Expected.Length * 4]); byte[] initialGuard = guarded.ToArray();
            host.GatherRows(fixture.TokenIds, MemoryMarshal.Cast<byte, float>(guarded.AsSpan(GuardBytes, fixture.Expected.Length * 4)));
            Need(GuardsMatch(guarded, initialGuard) && source.Reads == reads, "CPU fixture touched a guard or borrowed source.");
            Need(Mismatches(guarded.AsSpan(GuardBytes, fixture.Expected.Length * 4), Bytes(fixture.Expected)) == 0, "CPU fixture differs from independent nearest-even oracle.");
            fixtureValues += fixture.Expected.Length;
        }
        // This CPU-only sweep is deliberately broader than GPU fixture coverage:
        // every finite binary16 scale crossed with every signed int8 code.
        ushort[] finite = Enumerable.Range(0, 65536).Where(x => (x & 0x7c00) != 0x7c00).Select(x => (ushort)x).ToArray();
        byte[] payload = Payload(256, finite, perGroup: false);
        using var exhaustiveSource = new Source(payload);
        using var exhaustive = new CudaNinferHostEmbedding(Descriptor(exhaustiveSource, 256, finite.Length, payload.Length));
        int exhaustiveReads = exhaustiveSource.Reads; exhaustiveSource.Dispose();
        var id = new int[1]; var values = new float[256]; long allValues = 0, ties = 0, negativeZeros = 0;
        for (int row = 0; row < finite.Length; row++)
        {
            id[0] = row; exhaustive.GatherRows(id, values);
            for (int c = 0; c < 256; c++)
            {
                float expected = Oracle(unchecked((sbyte)(c ^ 0x80)), finite[row], out bool tie);
                if (BitConverter.SingleToInt32Bits(values[c]) != BitConverter.SingleToInt32Bits(expected))
                    throw new InvalidDataException($"CPU exhaustive failure at scale {finite[row]:x4}, code {c - 128}.");
                if (tie) ties++; if (BitConverter.SingleToUInt32Bits(values[c]) == 0x80000000u) negativeZeros++; allValues++;
            }
        }
        Need(exhaustiveSource.Reads == exhaustiveReads, "CPU exhaustive gather read its disposed source.");
        var grid = fixtures.Single(f => f.Width == 256 && f.TokenIds.Length == 33);
        // The 33-row fixture visits every row, so all 32 scales see all 256 codes.
        Need(grid.TokenIds.Distinct().Count() == ScaleBits.Length, "The GPU scale grid does not visit all selected scales.");
        return new { CPU = "PASS", uniqueGpuFixturesPrepared = fixtures.Length, fixtureValues,
            exhaustiveFiniteF16Scales = finite.Length, exhaustiveSignedW8Codes = 256, exhaustiveValues = allValues,
            exhaustiveMidpoints = ties, exhaustiveNegativeZeros = negativeZeros, gpuGridCoverage = Coverage(grid),
            GPU = "NOT_RUN", modelQuality = "NOT_RUN", modelMemorySaving = "NOT_MEASURED", performance = "NOT_RUN" };
    }

    private sealed unsafe class DeviceMemory : IDisposable
    {
        private readonly List<IntPtr> _owned = new(); private readonly IntPtr _stream;
        internal DeviceMemory(IntPtr stream) => _stream = stream;
        internal IntPtr Put(byte[] bytes)
        {
            CudaDriverApi.cuMemAlloc(out var pointer, (UIntPtr)bytes.Length).ThrowOnError(); _owned.Add(pointer);
            Write(pointer, bytes); return pointer;
        }
        internal byte[] Read(IntPtr pointer, int bytes)
        {
            var output = new byte[bytes]; fixed (byte* p = output) CudaDriverApi.cuMemcpyDtoH((IntPtr)p, pointer, (UIntPtr)bytes).ThrowOnError(); return output;
        }
        internal static void Write(IntPtr pointer, byte[] bytes)
        {
            fixed (byte* p = bytes) CudaDriverApi.cuMemcpyHtoD(pointer, (IntPtr)p, (UIntPtr)bytes.Length).ThrowOnError();
        }
        public void Dispose()
        {
            CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
            foreach (var pointer in _owned) CudaDriverApi.cuMemFree(pointer).ThrowOnError();
        }
    }
    private sealed unsafe class Pinned : IDisposable
    {
        internal IntPtr Pointer { get; private set; }
        private readonly int _bytes; private readonly IntPtr _stream;
        internal Pinned(int bytes, IntPtr stream)
        {
            _bytes = bytes; _stream = stream;
            CudaDriverApi.cuMemHostAlloc(out var pointer, (UIntPtr)bytes, 0).ThrowOnError(); Pointer = pointer;
        }
        internal Span<byte> Span => new((void*)Pointer, _bytes);
        public void Dispose()
        {
            if (Pointer == IntPtr.Zero) return;
            CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
            CudaDriverApi.cuMemFreeHost(Pointer).ThrowOnError(); Pointer = IntPtr.Zero;
        }
    }

    private static void RunCase(CudaNinferKernels kernels, Fixture fixture, int lifetime, IntPtr stream,
        string output, List<object> records, Dictionary<string, string> firstLifetimeSha)
    {
        string id = $"life{lifetime}-{fixture.Id}"; int bodyBytes = fixture.Expected.Length * 4;
        using var source = new Source(fixture.Payload);
        using var host = new CudaNinferHostEmbedding(Descriptor(source, fixture.Width, fixture.Vocabulary, fixture.Payload.Length));
        Need(ValidationSha256.Matches(host.PayloadSha256, Hash(fixture.Payload)), "Host payload differs from the original GPU payload.");
        int sourceReads = source.Reads; source.Dispose();
        using var memory = new DeviceMemory(stream);
        byte[] weights = Guard(fixture.Payload), tokenIds = Guard(Bytes(fixture.TokenIds)), expected = Bytes(fixture.Expected);
        var gpuWeights = memory.Put(weights); var gpuIds = memory.Put(tokenIds);
        var gpuGather = memory.Put(Guard(new byte[bodyBytes])); var gpuEmbed = memory.Put(Guard(new byte[bodyBytes]));
        using var staging = new Pinned(bodyBytes + 2 * GuardBytes, stream);
        using var pinnedIds = new Pinned(tokenIds.Length, stream); tokenIds.CopyTo(pinnedIds.Span);
        string fixtureOutput = Path.Combine(output, fixture.Id); Directory.CreateDirectory(fixtureOutput);
        if (lifetime == 0)
        {
            File.WriteAllBytes(Path.Combine(fixtureOutput, "original-w8-payload.bin"), fixture.Payload);
            File.WriteAllBytes(Path.Combine(fixtureOutput, "token-ids.i32"), Bytes(fixture.TokenIds));
            File.WriteAllBytes(Path.Combine(fixtureOutput, "independent-nearest-even.f32"), expected);
        }
        var passes = new List<object>();
        var record = new Dictionary<string, object?>
        {
            ["id"] = id, ["fixtureId"] = fixture.Id, ["lifetime"] = lifetime, ["passed"] = false,
            ["width"] = fixture.Width, ["vocabulary"] = fixture.Vocabulary, ["batch"] = fixture.TokenIds.Length,
            ["payloadBytes"] = fixture.Payload.Length, ["payloadSha256"] = host.PayloadSha256,
            ["idsSha256"] = Hash(Bytes(fixture.TokenIds)), ["expectedSha256"] = Hash(expected),
            ["oracle"] = "binary16 integer/exponent decode, exact FP64 int8 product, adjacent-BF16 distance and nearest-even tie; preserve signed zero",
            ["exactF32Tolerance"] = 0, ["sourceDisposedBeforeGather"] = source.Disposed,
            ["coverage"] = Coverage(fixture), ["passes"] = passes
        };
        records.Add(record); byte[]? first = null;
        for (int pass = 0; pass < Poisons.Length; pass++)
        {
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            var poisonBody = Enumerable.Repeat(BitConverter.Int32BitsToSingle(Poisons[pass]), fixture.Expected.Length).ToArray();
            byte[] poisoned = Guard(Bytes(poisonBody)); poisoned.CopyTo(staging.Span);
            DeviceMemory.Write(gpuGather, poisoned); DeviceMemory.Write(gpuEmbed, poisoned);
            // The production host helper writes the same pinned F32 staging and
            // uses the same asynchronous H2D API as this isolated component path.
            host.GatherRows(MemoryMarshal.Cast<byte, int>(pinnedIds.Span.Slice(GuardBytes, fixture.TokenIds.Length * 4)),
                MemoryMarshal.Cast<byte, float>(staging.Span.Slice(GuardBytes, bodyBytes)));
            byte[] actualHost = staging.Span.ToArray();
            bool hostGuards = GuardsMatch(actualHost, poisoned), pinnedIdsUnchanged = pinnedIds.Span.SequenceEqual(tokenIds);
            File.WriteAllBytes(Path.Combine(output, id + $"-pass{pass}.host.guarded.f32"), actualHost);
            CudaDriverApi.cuMemcpyHtoDAsync(gpuGather + GuardBytes, staging.Pointer + GuardBytes, (UIntPtr)bodyBytes, stream).ThrowOnError();
            kernels.EmbedW8(gpuWeights + GuardBytes, gpuIds + GuardBytes, gpuEmbed + GuardBytes,
                fixture.TokenIds.Length, fixture.Width, fixture.Vocabulary, stream);
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            byte[] gathered = memory.Read(gpuGather, poisoned.Length), embedded = memory.Read(gpuEmbed, poisoned.Length);
            string gatherFile = id + $"-pass{pass}.host-upload.guarded.f32", embedFile = id + $"-pass{pass}.gpu-embed.guarded.f32";
            File.WriteAllBytes(Path.Combine(output, gatherFile), gathered); File.WriteAllBytes(Path.Combine(output, embedFile), embedded);
            var gatherValues = gathered.AsSpan(GuardBytes, bodyBytes); var embedValues = embedded.AsSpan(GuardBytes, bodyBytes);
            int hostMismatches = Mismatches(actualHost.AsSpan(GuardBytes, bodyBytes), expected),
                gatherMismatches = Mismatches(gatherValues, expected), embedMismatches = Mismatches(embedValues, expected),
                hostGpuMismatches = Mismatches(gatherValues, embedValues);
            bool gatherGuards = GuardsMatch(gathered, poisoned), embedGuards = GuardsMatch(embedded, poisoned);
            bool finiteAndBfloat = true;
            foreach (float value in MemoryMarshal.Cast<byte, float>(gatherValues))
                finiteAndBfloat &= float.IsFinite(value) && (BitConverter.SingleToUInt32Bits(value) & 0xffffu) == 0;
            foreach (float value in MemoryMarshal.Cast<byte, float>(embedValues))
                finiteAndBfloat &= float.IsFinite(value) && (BitConverter.SingleToUInt32Bits(value) & 0xffffu) == 0;
            bool sameAsFirst = first == null || first.AsSpan().SequenceEqual(gatherValues); first ??= gatherValues.ToArray();
            string activeSha = Hash(gatherValues);
            bool sameAsFirstLifetime = !firstLifetimeSha.TryGetValue(fixture.Id, out var priorSha) || ValidationSha256.Matches(priorSha, activeSha);
            if (lifetime == 0) firstLifetimeSha.TryAdd(fixture.Id, activeSha);
            bool sourceAndIdsUnchanged = memory.Read(gpuWeights, weights.Length).AsSpan().SequenceEqual(weights)
                && memory.Read(gpuIds, tokenIds.Length).AsSpan().SequenceEqual(tokenIds)
                && pinnedIdsUnchanged && source.Reads == sourceReads && staging.Span.SequenceEqual(actualHost);
            passes.Add(new { pass, poisonBits = Poisons[pass].ToString("x8"),
                hostFile = id + $"-pass{pass}.host.guarded.f32", hostSha256 = Hash(actualHost),
                gatherFile, gatherSha256 = Hash(gathered), embedFile, embedSha256 = Hash(embedded), activeSha256 = activeSha,
                hostMismatches, gatherMismatches, embedMismatches, hostGpuMismatches,
                hostGuardsIntact = hostGuards, gatherGuardsIntact = gatherGuards, embedGuardsIntact = embedGuards,
                finiteAndBfloat, sameAsFirst, sameAsFirstLifetime, sourceAndIdsUnchanged });
            Need(hostMismatches == 0 && gatherMismatches == 0 && embedMismatches == 0 && hostGpuMismatches == 0
                && hostGuards && gatherGuards && embedGuards && finiteAndBfloat && sameAsFirst && sameAsFirstLifetime && sourceAndIdsUnchanged,
                id + ": independent exact/placement/guard/replay/lifetime gate failed; captured outputs retained.");
        }
        record["passed"] = true;
    }

    public static int Main(string[] args)
    {
        string? output = null; bool owned = false; var records = new List<object>();
        var state = new Dictionary<string, object?>
        {
            ["schema"] = "ninfer-native-host-embedding-component/1", ["complete"] = false,
            ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_RUN",
            ["realModelMemorySaving"] = "NOT_MEASURED", ["records"] = records
        };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian captures required.");
            if (args.Length == 1 && args[0] == "--self-test")
            {
                Console.WriteLine(JsonSerializer.Serialize(CpuSelfTest(), Json)); return 0;
            }
            if (args.Length == 0)
            {
                Console.WriteLine("PLAN: --execute --module ABSOLUTE_FATBIN --out FRESH_IGNORED_DIRECTORY; --self-test (CPU only). 9 fixtures, 2 host/module lifetimes, 3 A/B/A poison passes, exact host/gather-upload/GPU-embed bytes. No model timing or sampling."); return 0;
            }
            Need(args.Length == 5 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--out", "Use --execute --module ABSOLUTE_FATBIN --out FRESH_IGNORED_DIRECTORY.");
            string modulePath = args[2]; output = args[4]; Fresh(output);
            Need(Path.IsPathFullyQualified(modulePath) && File.Exists(modulePath) && Path.GetExtension(modulePath).Equals(".fatbin", StringComparison.OrdinalIgnoreCase), "An explicit absolute fatbin is required to identify the exact loaded module image.");
            Directory.CreateDirectory(output); owned = true;
            state["modulePath"] = modulePath; state["moduleFileSha256"] = Hash(File.ReadAllBytes(modulePath));
            state["toolAssemblySha256"] = Hash(File.ReadAllBytes(typeof(Program).Assembly.Location));
            state["backendAssemblySha256"] = Hash(File.ReadAllBytes(typeof(CudaBackend).Assembly.Location));
            state["uniqueFixtures"] = 9; state["hostAndModuleLifetimes"] = 2; state["outputPoisonPassesPerFixtureAndLifetime"] = 3;
            state["selectedFiniteF16ScaleBits"] = ScaleBits.Select(x => x.ToString("x4")).ToArray();
            state["realTargetW8PayloadBytesArithmeticOnly"] = checked(248320L * 5120 * 17 / 16);
            state["componentScope"] = "original W8 int8 codes + row-major group32 FP16 scales; CPU gather -> pinned F32 H2D vs original GPU EmbedW8; no dequantized full-matrix copy, model, sampling, or timing";
            Save(Path.Combine(output, "result.json"), state);
            state["cpuSelfTest"] = CpuSelfTest(); Save(Path.Combine(output, "result.json"), state);
            var fixtures = Fixtures(); var firstLifetime = new Dictionary<string, string>();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using var device = CudaContext.Create(0); state["GPU"] = "EXECUTED";
            CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); var name = new byte[256];
            CudaDriverApi.cuDeviceGetName(name, name.Length, 0).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int major, 75, 0).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int minor, 76, 0).ThrowOnError();
            state["device"] = new { ordinal = 0, name = Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver, computeMajor = major, computeMinor = minor };
            for (int lifetime = 0; lifetime < 2; lifetime++)
            {
                CudaDriverApi.cuStreamCreate(out var stream, 1).ThrowOnError();
                try
                {
                    using var kernels = new CudaNinferKernels(modulePath);
                    Need(ValidationSha256.Matches(kernels.ImageSha256, state["moduleFileSha256"]!.ToString()!), "Loaded module image differs from the pinned input fatbin.");
                    foreach (var fixture in fixtures)
                    {
                        RunCase(kernels, fixture, lifetime, stream, output, records, firstLifetime);
                        Save(Path.Combine(output, "result.json"), state);
                    }
                }
                finally { CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError(); CudaDriverApi.cuStreamDestroy(stream).ThrowOnError(); }
            }
            Need(ValidationSha256.Matches(Hash(File.ReadAllBytes(modulePath)), state["moduleFileSha256"]!.ToString()!), "Module file changed during execution.");
            Need(records.Count == 18 && firstLifetime.Count == 9, "Incomplete fixture/lifetime coverage.");
            state["complete"] = true; state["passed"] = true; Save(Path.Combine(output, "result.json"), state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, uniqueFixtures = 9, lifecycleRecords = records.Count,
                nativeEmbeddingLaunches = 54, pinnedEmbeddingUploads = 54, modelQuality = "NOT_RUN", performance = "NOT_RUN", realModelMemorySaving = "NOT_MEASURED" })); return 0;
        }
        catch (Exception error)
        {
            state["passed"] = false; state["error"] = error.ToString();
            if (owned && output != null) Save(Path.Combine(output, "result.json"), state);
            Console.Error.WriteLine(error); return 1;
        }
    }
}
