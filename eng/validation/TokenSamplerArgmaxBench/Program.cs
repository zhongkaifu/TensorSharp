// CPU-only full TokenSampler.Sample temp0 same-binary ABBA validation.
using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Runtime;

string Required(string name) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : throw new ArgumentException("Missing " + name); }
int Number(string name, int fallback) { int i = Array.IndexOf(args, name); return i < 0 ? fallback : int.Parse(args[i + 1]); }
string output = Path.GetFullPath(Required("--out")), source = Path.GetFullPath(Required("--sampler-source"));
int rounds = Number("--rounds", 20), iterations = Number("--iterations", 100);
if (rounds < 20 || iterations < 1 || File.Exists(output) || Directory.Exists(output) || new FileInfo(output).LinkTarget != null) throw new ArgumentException("Fresh nonaliased output, >=20 rounds and positive iterations required.");
string[] parts = output.Split(Path.DirectorySeparatorChar);
if (!parts.Contains("artifacts") && !output.Replace('\\', '/').Contains("/docs/validation/")) throw new ArgumentException("Output must be in artifacts/ or docs/validation/.");
for (DirectoryInfo? p = new(Path.GetDirectoryName(output)!); p != null; p = p.Parent)
    if (p.LinkTarget != null) throw new ArgumentException("Output ancestor is aliased.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
string HashFile(string p) { using var f = File.OpenRead(p); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
string HashBits(float[] data) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(data.AsSpan()))).ToLowerInvariant();
int Scalar(float[] a) { int best = 0; float max = a[0]; for (int i = 1; i < a.Length; i++) if (a[i] > max) { max = a[i]; best = i; } return best; }
double Median(IEnumerable<double> x) { double[] a = x.Order().ToArray(); return (a[(a.Length - 1) / 2] + a[a.Length / 2]) * .5; }
var ctor = typeof(TokenSampler).GetConstructor(BindingFlags.Instance | BindingFlags.NonPublic, null, new[] { typeof(SamplingConfig), typeof(bool) }, null)
    ?? throw new InvalidOperationException("Explicit same-binary constructor unavailable.");
var config = new SamplingConfig { Temperature = 0, RepetitionPenalty = 1, PresencePenalty = 0, FrequencyPenalty = 0 };
TokenSampler Create(bool vector) => (TokenSampler)ctor.Invoke(new object[] { config, vector });
var scalar = Create(false); var vector = Create(true);
var policy = typeof(TokenSampler).GetField("_vectorizeArgmax", BindingFlags.Instance | BindingFlags.NonPublic)!;
if ((bool)policy.GetValue(scalar)! || !(bool)policy.GetValue(vector)!) throw new InvalidOperationException("Actual controls differ.");
string sourceBefore = HashFile(source), runtime = typeof(TokenSampler).Assembly.Location, runtimeBefore = HashFile(runtime);
var state = new Dictionary<string, object?>
{
    ["schema"] = "token-sampler-argmax-ABBA/1", ["complete"] = false, ["quality_passed"] = false,
    ["scope"] = "CPU full TokenSampler.Sample temp0; no model, CUDA, server, network or end-to-end inference benchmark",
    ["framework"] = RuntimeInformation.FrameworkDescription, ["os"] = RuntimeInformation.OSDescription,
    ["process_architecture"] = RuntimeInformation.ProcessArchitecture.ToString(), ["processor_count"] = Environment.ProcessorCount,
    ["ISA"] = new { Vector.IsHardwareAccelerated, portable_float_lanes = Vector<float>.Count,
        Vector128 = Vector128.IsHardwareAccelerated, Vector256 = Vector256.IsHardwareAccelerated,
        Vector512 = Vector512.IsHardwareAccelerated, Avx2 = Avx2.IsSupported, Avx512F = Avx512F.IsSupported, AdvSimd = AdvSimd.IsSupported },
    ["rounds"] = rounds, ["iterations"] = iterations, ["sequence"] = "A1 scalar / B1 vector / B2 vector / A2 scalar, each round",
    ["source"] = new { path = source, sha256 = sourceBefore }, ["runtime"] = new { path = runtime, sha256 = runtimeBefore },
    ["bench_sha256"] = HashFile(Assembly.GetExecutingAssembly().Location),
    ["process_controls"] = new { tiered_compilation = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"), inherited_public_control = Environment.GetEnvironmentVariable("TS_TOKEN_SAMPLER_SIMD"), public_default = "scalar; only exact TS_TOKEN_SAMPLER_SIMD=1 opts in", mode = "explicit internal constructor, outside timed loops; independent of public default" },
    ["limitations"] = new[] { "Same binary, managed arrays and CPU microbench only; no inference-speed or 472ms attribution claim.", "Warm arrays and sequential ABBA do not characterize PCIe transfers, host scheduling, cold caches or different CPUs/ISAs.", "Scalar oracle and all measured token IDs plus unchanged input bits gate every sample; no timing result is a full model-quality proof." }
};
var rows = new List<object>(); state["fixtures"] = rows;
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Save();
try
{
    foreach ((int length, int position) in new[] { (248320, -1), (248320, 248319), (248320, 124160), (248323, 248322), (128, 127) })
    {
        var random = new Random(9131 + length + position);
        float[] data = Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 20 - 10)).ToArray();
        if (position >= 0) data[position] = 20;
        string before = HashBits(data); int expected = Scalar(data);
        for (int warm = 0; warm < 128; warm++)
            if (scalar.Sample(data) != expected || vector.Sample(data) != expected) throw new InvalidOperationException("Warmup ID mismatch.");
        (double us, long bytes) Measure(TokenSampler sampler)
        {
            long allocated = GC.GetAllocatedBytesForCurrentThread(); long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < iterations; i++)
                if (sampler.Sample(data) != expected) throw new InvalidOperationException("Measured ID differs from independent scalar.");
            return (Stopwatch.GetElapsedTime(start).TotalMicroseconds / iterations, GC.GetAllocatedBytesForCurrentThread() - allocated);
        }
        var samples = new List<(double A1, double B1, double B2, double A2)>(); var raw = new List<object>();
        for (int round = 0; round < rounds; round++)
        {
            var a = Measure(scalar); var b = Measure(vector); var b2 = Measure(vector); var a2 = Measure(scalar);
            samples.Add((a.us, b.us, b2.us, a2.us));
            raw.Add(new { round, A1_us = a.us, B1_us = b.us, B2_us = b2.us, A2_us = a2.us,
                allocated_A1 = a.bytes, allocated_B1 = b.bytes, allocated_B2 = b2.bytes, allocated_A2 = a2.bytes });
        }
        if (HashBits(data) != before) throw new InvalidOperationException("Logits changed.");
        double aMedian = Median(samples.SelectMany(x => new[] { x.A1, x.A2 }));
        double bMedian = Median(samples.SelectMany(x => new[] { x.B1, x.B2 }));
        rows.Add(new { length, forced_max_position = position, actual_max = expected, input_sha256 = before, full_input_bits_unchanged = true,
            all_measured_token_ids_equal = true, scalar_median_us = aMedian, vector_median_us = bMedian,
            scalar_over_vector = aMedian / bMedian, sample_count_per_arm = rounds * 2, raw });
        Save();
        Console.WriteLine($"n={length} max={expected}: scalar={aMedian:F3}us vector={bMedian:F3}us ratio={aMedian / bMedian:F3}x");
    }
    bool unchanged = sourceBefore == HashFile(source) && runtimeBefore == HashFile(runtime);
    if (!unchanged) throw new InvalidOperationException("Source/runtime changed during measurement.");
    state["identities_unchanged"] = true; state["complete"] = true; state["quality_passed"] = true;
}
finally { Save(); }
