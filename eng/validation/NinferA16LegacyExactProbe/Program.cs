// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Validation;

internal static class Program
{
    private const int GuardBytes = 256, GuardBits = 0x4e56789a;
    private static readonly int[] Poisons = { unchecked((int)0x7fc1a16f), unchecked((int)0x7fc2a16f), unchecked((int)0x7fc1a16f) };
    private const double UnitRoundoff = 1.0 / 16777216.0;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private sealed record Fixture(string Id, int Width, int OutputRows, int Tokens, int InputStride, int OutputStride,
        float InverseDivisor, byte[] OriginalCodes, byte[] OriginalScales, byte[] Codes, byte[] Scales, float[] Input,
        float[] Expected, double[] Fp64, double[] SumAbsolute, double[] Bounds);
    private static void Need(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static float Float(int bits) => BitConverter.Int32BitsToSingle(bits);
    private static float Bf(float value) { uint x = (uint)BitConverter.SingleToInt32Bits(value); return Float((int)((x + 0x7fffU + ((x >> 16) & 1U)) & 0xffff0000U)); }
    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static byte[] Bytes<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static void Save(string file, object value) { using var stream = new FileStream(file, FileMode.Create, FileAccess.Write, FileShare.None); JsonSerializer.Serialize(stream, value, Json); stream.WriteByte(10); }
    private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        string[] p = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(p.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, p.Length - 1).Any(i => p[i] == "docs" && p[i + 1] == "validation"), "Ignored evidence output required.");
        Need(!Path.Exists(path) && !IsLink(path), "Fresh output required.");
        for (string? parent = Path.GetDirectoryName(path); parent != null; parent = Path.GetDirectoryName(parent)) Need(!IsLink(parent) && !File.Exists(parent), "Aliased output parent refused.");
    }
    private static int Code(int row, int column, int pattern) => pattern == 2
        ? ((column * 3 + row) & 7) | (((column / 2 + row) & 1) << 3)
        : (row * 7 + column * 11 + column / 16) & 15;
    private static byte Scale(int row, int group, int pattern) => pattern == 0 ? (byte)(48 + ((row + group) % 3) * 8)
        : (byte)((row * 17 + group * 29 + group / 7) % 127);
    private static Fixture Make(int width, int outputRows, int tokens, int pattern)
    {
        int storedRows = (outputRows + 127) / 128 * 128;
        int inputStride = width + 8, outputStride = outputRows + 7;
        float inverse = new[] { 0.125f, 0.13f, 3.7f }[pattern];
        byte[] originalCodes = new byte[storedRows * width / 2], originalScales = new byte[storedRows * width / 16];
        byte[] codes = new byte[outputRows * width / 2], scales = new byte[outputRows * width / 16];
        // Independently materialize the original adjacent-nibble encoding.
        for (int row = 0; row < storedRows; row++) for (int c = 0; c < width; c += 2)
            originalCodes[row * (width / 2) + c / 2] = (byte)(Code(row, c, pattern) | (Code(row, c + 1, pattern) << 4));
        // Explicit five-axis directory order, rather than the CPU oracle's offset helper.
        int scaleOffset = 0;
        for (int tile = 0; tile < storedRows / 128; tile++) for (int kg = 0; kg < width / 64; kg++)
            for (int r32 = 0; r32 < 32; r32++) for (int quarter = 0; quarter < 4; quarter++) for (int g = 0; g < 4; g++)
                originalScales[scaleOffset++] = Scale(tile * 128 + quarter * 32 + r32, kg * 4 + g, pattern);
        // Independently pack the same logical values into the candidate's low8/high8 planes.
        for (int row = 0; row < outputRows; row++) for (int g = 0; g < width / 16; g++)
        {
            scales[row * (width / 16) + g] = Scale(row, g, pattern);
            for (int j = 0; j < 8; j++) codes[row * (width / 2) + g * 8 + j] =
                (byte)(Code(row, g * 16 + j, pattern) | (Code(row, g * 16 + j + 8, pattern) << 4));
        }
        float[] input = Enumerable.Repeat(Float(unchecked((int)0x7fc3a16f)), tokens * inputStride).ToArray();
        for (int t = 0; t < tokens; t++) for (int c = 0; c < width; c++)
        {
            int signed = ((c * 19 + t * 31 + c / 7) % 127) - 63;
            float value = pattern == 0 ? signed / 16f : MathF.ScaleB(signed / 32f, (c * 7 + t * 3) % 41 - 20);
            input[t * inputStride + c] = c % 251 == 0 ? Float(unchecked((int)0x80000000)) : Bf(value);
        }
        float[] expected = NinferA16LegacyExactReference.Compute(originalCodes, originalScales, input,
            outputRows, width, tokens, inverse, inputStride, outputStride);
        double[] fp64 = new double[tokens * outputRows], sumAbsolute = new double[fp64.Length], bounds = new double[fp64.Length];
        // A priori forward-error bound includes both coefficient multiplications,
        // every lane's FMA chain, chain merge and five shuffle reductions. These
        // finite fixtures have no overflow/underflow. Do not fit this to GPU output.
        int operations = width / 32 + 16;
        double gamma = operations * UnitRoundoff / (1.0 - operations * UnitRoundoff);
        for (int t = 0; t < tokens; t++) for (int row = 0; row < outputRows; row++)
        {
            double sum = 0, absolute = 0;
            for (int c = 0; c < width; c++)
            {
                double weight = (double)NinferA16LegacyExactReference.DecodeCode(Code(row, c, pattern))
                    * NinferA16LegacyExactReference.DecodeScale(Scale(row, c / 16, pattern)) * inverse;
                double product = weight * input[t * inputStride + c]; sum += product; absolute += Math.Abs(product);
            }
            int at = t * outputRows + row; fp64[at] = sum; sumAbsolute[at] = absolute; bounds[at] = gamma * absolute;
            Need(double.IsFinite(sum) && double.IsFinite(bounds[at]), "FP64 oracle is nonfinite.");
            Need(Math.Abs((double)expected[t * outputStride + row] - sum) <= bounds[at], "CPU schedule exceeds its predeclared FP64 bound.");
        }
        return new($"k{width}-n{outputRows}-t{tokens}-f{pattern}", width, outputRows, tokens, inputStride, outputStride,
            inverse, originalCodes, originalScales, codes, scales, input, expected, fp64, sumAbsolute, bounds);
    }
    private static Fixture Analytical(int width, int tokens)
    {
        var f = Make(width, 1, tokens, 0);
        Array.Fill(f.OriginalCodes, (byte)0x22); Array.Fill(f.Codes, (byte)0x22);
        Array.Fill(f.OriginalScales, (byte)0x38); Array.Fill(f.Scales, (byte)0x38);
        for (int t = 0; t < tokens; t++) for (int c = 0; c < width; c++)
            f.Input[t * f.InputStride + c] = c >= 512 ? 0 : (c % 4) switch { 0 => 16777216f, 2 => -16777216f, _ => 1f };
        float[] expected = NinferA16LegacyExactReference.Compute(f.OriginalCodes, f.OriginalScales, f.Input,
            1, width, tokens, 1f, f.InputStride, f.OutputStride);
        const float known = 64f;
        for (int t = 0; t < tokens; t++) Need(expected[t * f.OutputStride] == known, "Analytical legacy low/high single-chain witness differs.");
        double operations = width / 32 + 16, gamma = operations * UnitRoundoff / (1 - operations * UnitRoundoff);
        return f with { Id = $"k{width}-n1-t{tokens}-analytical-cancellation", InverseDivisor = 1f, Expected = expected,
            Fp64 = Enumerable.Repeat(256d, tokens).ToArray(), SumAbsolute = Enumerable.Repeat(4294967552d, tokens).ToArray(),
            Bounds = Enumerable.Repeat(gamma * 4294967552d, tokens).ToArray() };
    }
    private static Fixture[] Fixtures() =>
        (from width in new[] { 5120, 6144, 17408 } from n in new[] { 1, 17, 129 }
         from t in new[] { 1, 2, 3 } from pattern in new[] { 0, 1, 2 } select Make(width, n, t, pattern))
        .Concat(from width in new[] { 5120, 6144, 17408 } from t in new[] { 1, 2, 3 } select Analytical(width, t)).ToArray();
    private sealed unsafe class DeviceMemory : IDisposable
    {
        private readonly List<IntPtr> _owned = new();
        internal IntPtr Put(byte[] bytes)
        {
            CudaDriverApi.cuMemAlloc(out var p, (UIntPtr)bytes.Length).ThrowOnError(); _owned.Add(p);
            fixed (byte* src = bytes) CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, (UIntPtr)bytes.Length).ThrowOnError(); return p;
        }
        internal byte[] Read(IntPtr p, int count)
        {
            var bytes = new byte[count]; fixed (byte* dst = bytes) CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, p, (UIntPtr)count).ThrowOnError(); return bytes;
        }
        internal static void Write(IntPtr p, byte[] bytes) { fixed (byte* src = bytes) CudaDriverApi.cuMemcpyHtoD(p, (IntPtr)src, (UIntPtr)bytes.Length).ThrowOnError(); }
        public void Dispose() { foreach (var p in _owned) CudaDriverApi.cuMemFree(p).ThrowOnError(); }
    }
    private static byte[] Guard(byte[] body)
    {
        byte[] bytes = new byte[body.Length + 2 * GuardBytes];
        for (int i = 0; i < GuardBytes / 4; i++)
        { BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), GuardBits); BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(GuardBytes + body.Length + i * 4), GuardBits); }
        body.CopyTo(bytes, GuardBytes); return bytes;
    }
    private static void RunCase(CudaNinferA16LegacyExactModule module, CudaNinferKernels legacy, Fixture f, int cycle, string output, List<object> records)
    {
        using var memory = new DeviceMemory();
        byte[] codeGuard = Guard(f.Codes), scaleGuard = Guard(f.Scales), inputGuard = Guard(Bytes(f.Input));
        var code = memory.Put(codeGuard); var scale = memory.Put(scaleGuard); var input = memory.Put(inputGuard);
        var destination = memory.Put(Guard(new byte[f.Expected.Length * 4]));
        var legacyDestination = memory.Put(Guard(new byte[f.Expected.Length * 4]));
        string name = $"cycle{cycle}-{f.Id}";
        var result = new Dictionary<string, object?> { ["id"] = name, ["fixtureId"] = f.Id, ["cycle"] = cycle,
            ["passed"] = false, ["width"] = f.Width, ["outputRows"] = f.OutputRows, ["tokens"] = f.Tokens,
            ["inputStride"] = f.InputStride, ["outputStride"] = f.OutputStride, ["inverseWeightDivisor"] = f.InverseDivisor,
            ["originalCodesSha256"] = Hash(f.OriginalCodes), ["originalScalesSha256"] = Hash(f.OriginalScales),
            ["candidateCodesSha256"] = Hash(f.Codes), ["candidateScalesSha256"] = Hash(f.Scales),
            ["inputSha256"] = Hash(Bytes(f.Input)), ["expectedF32Sha256"] = Hash(Bytes(f.Expected)),
            ["fp64BoundRule"] = "gamma_(K/32+16)*sum(abs(original_fp4*fp8*inverseD*BF16input)); unit_roundoff=2^-24",
            ["exactF32Tolerance"] = 0, ["outputs"] = new List<object>() };
        records.Add(result); var outputs = (List<object>)result["outputs"]!;
        var fixtureDirectory = Path.Combine(output, f.Id);
        if (!Directory.Exists(fixtureDirectory))
        {
            Directory.CreateDirectory(fixtureDirectory);
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "original-codes.bin"), f.OriginalCodes);
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "original-scales.bin"), f.OriginalScales);
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "candidate-codes.bin"), f.Codes);
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "candidate-scales.bin"), f.Scales);
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "input.f32"), Bytes(f.Input));
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "expected.f32"), Bytes(f.Expected));
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "ideal.f64"), Bytes(f.Fp64));
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "sum-absolute.f64"), Bytes(f.SumAbsolute));
            File.WriteAllBytes(Path.Combine(fixtureDirectory, "bounds.f64"), Bytes(f.Bounds));
        }
        byte[]? firstValues = null;
        for (int pass = 0; pass < Poisons.Length; pass++)
        {
            float[] poisoned = Enumerable.Repeat(Float(Poisons[pass]), f.Expected.Length).ToArray();
            byte[] poisonGuard = Guard(Bytes(poisoned)); DeviceMemory.Write(destination, poisonGuard);
            DeviceMemory.Write(legacyDestination, poisonGuard);
            legacy.ProjectA16(code + GuardBytes, scale + GuardBytes, input + GuardBytes, legacyDestination + GuardBytes,
                f.OutputRows, f.Width, f.Tokens, f.InverseDivisor, IntPtr.Zero, f.InputStride, f.OutputStride);
            byte[] legacyActual = memory.Read(legacyDestination, poisonGuard.Length);
            string legacyFile = name + $"-pass{pass}.legacy.guarded.f32";
            File.WriteAllBytes(Path.Combine(output, legacyFile), legacyActual);
            module.Project(code + GuardBytes, scale + GuardBytes, input + GuardBytes, destination + GuardBytes,
                f.OutputRows, f.Width, f.Tokens, f.InverseDivisor, IntPtr.Zero, f.InputStride, f.OutputStride);
            byte[] actual = memory.Read(destination, poisonGuard.Length);
            string file = name + $"-pass{pass}.guarded.f32"; File.WriteAllBytes(Path.Combine(output, file), actual);
            int bitMismatches = 0, legacyCpuBitMismatches = 0, legacyGpuBitMismatches = 0, fp64Failures = 0, paddingMismatches = 0; double maxError = 0, maximumBoundFraction = 0;
            var written = new byte[f.Tokens * f.OutputRows * 4];
            for (int t = 0; t < f.Tokens; t++) for (int row = 0; row < f.OutputStride; row++)
            {
                int index = t * f.OutputStride + row, bits = BinaryPrimitives.ReadInt32LittleEndian(actual.AsSpan(GuardBytes + index * 4));
                int legacyBits = BinaryPrimitives.ReadInt32LittleEndian(legacyActual.AsSpan(GuardBytes + index * 4));
                if (row >= f.OutputRows) { if (bits != Poisons[pass] || legacyBits != Poisons[pass]) paddingMismatches++; continue; }
                float value = Float(bits); int at = t * f.OutputRows + row;
                Need(float.IsFinite(value), name + ": nonfinite result; original output retained.");
                int expectedBits = BitConverter.SingleToInt32Bits(f.Expected[index]);
                if (bits != expectedBits) bitMismatches++;
                if (legacyBits != expectedBits) legacyCpuBitMismatches++;
                if (bits != legacyBits) legacyGpuBitMismatches++;
                double error = Math.Abs(value - f.Fp64[at]); maxError = Math.Max(maxError, error);
                if (error > f.Bounds[at]) fp64Failures++;
                if (f.Bounds[at] > 0) maximumBoundFraction = Math.Max(maximumBoundFraction, error / f.Bounds[at]);
                BinaryPrimitives.WriteInt32LittleEndian(written.AsSpan(at * 4), bits);
            }
            bool guards = actual.AsSpan(0, GuardBytes).SequenceEqual(poisonGuard.AsSpan(0, GuardBytes))
                && actual.AsSpan(actual.Length - GuardBytes).SequenceEqual(poisonGuard.AsSpan(poisonGuard.Length - GuardBytes));
            bool legacyGuards = legacyActual.AsSpan(0, GuardBytes).SequenceEqual(poisonGuard.AsSpan(0, GuardBytes))
                && legacyActual.AsSpan(legacyActual.Length - GuardBytes).SequenceEqual(poisonGuard.AsSpan(poisonGuard.Length - GuardBytes));
            bool repeat = firstValues == null || firstValues.AsSpan().SequenceEqual(written); firstValues ??= written;
            outputs.Add(new { pass, poisonBits = Poisons[pass].ToString("x8"), file, sha256 = Hash(actual), writtenSha256 = Hash(written),
                legacyFile, legacySha256 = Hash(legacyActual), legacyCpuBitMismatches, legacyGpuBitMismatches,
                legacyGuardsIntact = legacyGuards, bitMismatches, fp64Failures, paddingMismatches, guardsIntact = guards, sameWrittenBitsAsFirst = repeat,
                maxAbsoluteErrorToIdeal = maxError, maximumBoundFraction });
            Need(legacyCpuBitMismatches == 0 && legacyGpuBitMismatches == 0 && legacyGuards && bitMismatches == 0 && fp64Failures == 0 && paddingMismatches == 0 && guards && repeat,
                name + ": exact/bound/poison/guard/replay gate failed; complete outputs retained.");
        }
        Need(memory.Read(code, codeGuard.Length).AsSpan().SequenceEqual(codeGuard), "Weight code allocation changed.");
        Need(memory.Read(scale, scaleGuard.Length).AsSpan().SequenceEqual(scaleGuard), "Weight scale allocation changed.");
        Need(memory.Read(input, inputGuard.Length).AsSpan().SequenceEqual(inputGuard), "Activation/stride-padding allocation changed.");
        result["sourceAndInputAllocationsUnchanged"] = true; result["passed"] = true;
    }
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false;
        var records = new List<object>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-a16-legacy-exact-component/1", ["complete"] = false,
            ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_QUALIFIED", ["records"] = records };
        try
        {
            Need(BitConverter.IsLittleEndian, "The captured binary-vector format requires a little-endian host.");
            if (args.Length == 1 && args[0] == "--self-test")
            {
                var cpuFixtures = Fixtures();
                Need(cpuFixtures.All(f => f.Expected.Where((_, i) => i % f.OutputStride < f.OutputRows).All(float.IsFinite)), "CPU fixture nonfinite.");
                Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", uniqueFixtures = cpuFixtures.Length,
                    fullResults = cpuFixtures.Sum(f => f.OutputRows * f.Tokens),
                    analyticalSchedules = new[] { 64f, 64f, 64f }, GPU = "NOT_RUN", model = "NOT_RUN" })); return 0;
            }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --module ABSOLUTE_PTX_OR_FATBIN --legacy ABSOLUTE_R5_HELPER --out FRESH_IGNORED_DIRECTORY; --self-test. 90 unique fixtures, two module lifetimes, three output poison passes. No GPU work without --execute."); return 0; }
            Need(args.Length == 7 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--legacy" && args[5] == "--out", "Use --execute --module FILE --legacy R5_HELPER --out FRESH_IGNORED_DIRECTORY.");
            string modulePath = Path.GetFullPath(args[2]), legacyPath = Path.GetFullPath(args[4]); output = args[6]; Fresh(output);
            Need(File.Exists(legacyPath), "Frozen legacy helper is unavailable.");
            Need(File.Exists(modulePath), "Candidate module is unavailable.");
            Directory.CreateDirectory(output); owned = true;
            state["legacyModulePath"] = legacyPath; state["legacyModuleFileSha256"] = Hash(File.ReadAllBytes(legacyPath));
            state["modulePath"] = modulePath; state["moduleFileSha256"] = Hash(File.ReadAllBytes(modulePath));
            state["toolAssemblySha256"] = Hash(File.ReadAllBytes(typeof(Program).Assembly.Location));
            state["backendAssemblySha256"] = Hash(File.ReadAllBytes(typeof(CudaBackend).Assembly.Location));
            state["uniqueFixtures"] = 90; state["moduleLifetimes"] = 2; state["outputPoisonPassesPerFixtureAndLifetime"] = 3;
            Save(Path.Combine(output, "result.json"), state);
            var fixtures = Fixtures();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError(); using var device = CudaContext.Create(0); state["GPU"] = "EXECUTED";
            for (int cycle = 0; cycle < 2; cycle++)
            {
                using var module = new CudaNinferA16LegacyExactModule(modulePath);
                using var legacy = new CudaNinferKernels(legacyPath);
                Need(ValidationSha256.Matches(legacy.ImageSha256, state["legacyModuleFileSha256"]!.ToString()!), "Loaded legacy image differs.");
                Need(ValidationSha256.Matches(module.ImageSha256, state["moduleFileSha256"]!.ToString()!), "Loaded module image differs.");
                foreach (var fixture in fixtures) { RunCase(module, legacy, fixture, cycle, output, records); Save(Path.Combine(output, "result.json"), state); }
            }
            Need(Hash(File.ReadAllBytes(modulePath)) == state["moduleFileSha256"]?.ToString(), "Candidate module changed during execution.");
            Need(Hash(File.ReadAllBytes(legacyPath)) == state["legacyModuleFileSha256"]?.ToString(), "Frozen legacy module changed during execution.");
            state["complete"] = true; state["passed"] = true; Save(Path.Combine(output, "result.json"), state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, uniqueFixtures = fixtures.Length, lifecycleRecords = records.Count, modelQuality = "NOT_RUN", performance = "NOT_QUALIFIED" })); return 0;
        }
        catch (Exception error)
        {
            state["passed"] = false; state["error"] = error.ToString(); if (owned && output != null) Save(Path.Combine(output, "result.json"), state);
            Console.Error.WriteLine(error); return 1;
        }
    }
}
