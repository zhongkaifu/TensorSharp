// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Fixed input histories distinguish state leakage from autoregressive divergence.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

try
{
var valueOptions = new HashSet<string>(StringComparer.Ordinal) { "--model", "--fixture", "--output", "--native-sha256", "--backend", "--session", "--sequence", "--reference" };
var seenOptions = new HashSet<string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    string option = args[i];
    if (option is "--help" or "--record-only") continue;
    if (!valueOptions.Contains(option) || i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException("Unknown option or missing value: " + option);
    if (option != "--reference" && !seenOptions.Add(option)) throw new ArgumentException("Duplicate option: " + option);
    i++;
}
string? Option(string name)
{
    int at = Array.IndexOf(args, name);
    return at < 0 ? null : at + 1 < args.Length ? args[at + 1] : throw new ArgumentException("Missing value for " + name);
}
string Required(string name) => Option(name) ?? throw new ArgumentException("Missing " + name);
if (args.Length == 0 || args.Contains("--help"))
{
    Console.WriteLine("ResetLogitProbe --model FILE --fixture JSON --output NEW_DIRECTORY --native-sha256 HASH [--backend ggml_cuda|ggml_cpu] [--session reused|fresh] [--sequence p32,p32,p512,p512,p32] [--reference METRICS_JSON ...] [--record-only]");
    Console.WriteLine("Every requested row is exported. No numeric tolerance option. Exit 0: exact comparisons cover every requested case (or recording completed); 1: a difference or unverified case remains; 2: incomplete/invalid run. Fresh means a fresh model in this process; use separate invocations for fresh-process references.");
    return 0;
}
var json = new JsonSerializerOptions { WriteIndented = true, PropertyNameCaseInsensitive = true };
string modelPath = Path.GetFullPath(Required("--model"));
string fixturePath = Path.GetFullPath(Required("--fixture"));
string output = Path.GetFullPath(Required("--output"));
string backendName = Option("--backend") ?? "ggml_cuda";
BackendType backend = backendName switch { "ggml_cuda" => BackendType.GgmlCuda, "ggml_cpu" => BackendType.GgmlCpu, _ => throw new ArgumentException("Unsupported backend") };
string session = Option("--session") ?? "reused";
if (session is not ("reused" or "fresh")) throw new ArgumentException("--session must be reused or fresh");
bool recordOnly = args.Contains("--record-only");
if (Directory.Exists(output) || File.Exists(output)) throw new IOException("Refusing to overwrite existing evidence: " + output);
if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Evidence format is little-endian float32");
var fixture = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(fixturePath), json) ?? throw new InvalidDataException("Invalid fixture");
if (fixture.Version != 1 || fixture.Steps < 1 || fixture.Cases.Length < 1 || fixture.Context < 1 || fixture.VocabSize < 2) throw new InvalidDataException("Invalid fixture dimensions/version");
var cases = fixture.Cases.ToDictionary(c => c.Name, StringComparer.Ordinal);
foreach (var c in cases.Values)
{
    if (string.IsNullOrWhiteSpace(c.Name) || Path.GetFileName(c.Name) != c.Name || c.Name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 ||
        c.PromptTokens.Length < 1 || c.ForcedTokens.Length != fixture.Steps || c.PromptTokens.Length + fixture.Steps > fixture.Context ||
        c.PromptTokens.Concat(c.ForcedTokens).Any(t => t < 0 || t >= fixture.VocabSize)) throw new InvalidDataException("Invalid case " + c.Name);
}
string[] sequence = (Option("--sequence") ?? string.Join(',', fixture.Sequence)).Split(',', StringSplitOptions.RemoveEmptyEntries);
if (sequence.Length == 0 || sequence.Any(n => !cases.ContainsKey(n))) throw new ArgumentException("Sequence contains an unknown case");
string fixtureHash = Hash(fixturePath);
var modelIdentity = ValidateModelIdentity(modelPath, fixture);
string modelHash = modelIdentity.Sha256;
string nativeName = OperatingSystem.IsWindows() ? "GgmlOps.dll" : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so";
string nativePath = Path.Combine(AppContext.BaseDirectory, nativeName), nativeHash = Hash(nativePath);
if (!nativeHash.Equals(Required("--native-sha256"), StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Unexpected app-local native library");
Environment.SetEnvironmentVariable("MAX_CONTEXT", fixture.Context.ToString());
string[] envKeys = ["MAX_CONTEXT", "KV_CACHE_DTYPE", "TS_CPU_MOE", "TS_N_CPU_MOE", "TS_CPU_MOE_THREADS", "TS_GGML_CPU_THREADS", "TS_SPEC", "TS_HOST_MOE_DECODE", "TS_HOST_MOE_DEVICE_MIN_BATCH", "TS_HOST_MOE_METAL_OVERLAP", "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "GGML_CUDA_DISABLE_GRAPHS", "TS_HOST_MOE_TRACE_DIR", "TS_HOST_MOE_TRACE_CALLS", "TS_HOST_MOE_TRACE_START", "TS_GGUF_PREFAULT", "TS_HOST_MOE_EXPERT_CACHE_MB", "TS_Q4E_RETAINED_CACHE_MB"];
var environment = envKeys.ToDictionary(k => k, Environment.GetEnvironmentVariable);
KvCacheDtypeConfig.ConfigureFromEnvironment();
MoeCpuOffloadConfig.ConfigureFromEnvironment();
var references = new Dictionary<string, float[][]>(StringComparer.Ordinal);
var external = new Dictionary<string, float[][]>(StringComparer.Ordinal);
var externalFiles = new List<object>();
string[] referencePaths = Enumerable.Range(0, args.Length).Where(i => args[i] == "--reference")
    .Select(i => i + 1 < args.Length ? Path.GetFullPath(args[i + 1]) : throw new ArgumentException("Missing --reference path")).ToArray();
foreach (string referencePath in referencePaths)
{
    using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
    var r = reference.RootElement;
    if (!r.GetProperty("complete").GetBoolean() || !r.GetProperty("allFinite").GetBoolean() || !r.GetProperty("allExact").GetBoolean() || r.GetProperty("fixtureSha256").GetString() != fixtureHash ||
        r.GetProperty("modelSha256").GetString() != modelHash || r.GetProperty("backend").GetString() != backendName)
        throw new InvalidDataException("Reference is incomplete/self-inconsistent/nonfinite or uses a different fixture/model/backend; record fresh-process cases separately");
    var validated = ProbeEvidenceValidation.LoadReferenceRequests(referencePath, r, fixture.Steps, fixture.VocabSize);
    foreach (var entry in validated.Cases)
    {
        string name = entry.Key;
        if (!cases.ContainsKey(name)) throw new InvalidDataException("Reference contains unknown case " + name);
        if (external.ContainsKey(name)) throw new InvalidDataException("Multiple reference files define case " + name);
        external[name] = entry.Value;
    }
    externalFiles.AddRange(validated.Files.Select(file => (object)new { caseName = file.CaseName, path = file.Path, sha256 = file.Sha256, bytes = file.Bytes }));
}
if (referencePaths.Length > 0 && sequence.Any(n => !external.ContainsKey(n))) throw new InvalidDataException("References do not cover every requested case");
Directory.CreateDirectory(output);
string metricsPath = Path.Combine(output, "metrics.json");
var requests = new List<object>();
var loads = new List<object>();
bool allFinite = true, allExact = true;
int processId = Environment.ProcessId;
DateTime startedUtc = DateTime.UtcNow;
int comparisonCount = 0;
var comparedCases = new HashSet<string>(StringComparer.Ordinal);
string[] UnverifiedCases() => ProbeEvidenceValidation.UnverifiedCases(sequence, comparedCases);
string? error = null;
ModelBase? model = null;
void Write(bool complete) => File.WriteAllText(metricsPath, JsonSerializer.Serialize(new
{
    complete, allFinite, allExact, comparisonCount, unverifiedCases = UnverifiedCases(), recordOnly, processId, startedUtc,
    passed = ProbeEvidenceValidation.StrictPassed(complete, recordOnly, allFinite, allExact, comparisonCount, UnverifiedCases()),
    error, session, sequence, backend = backendName, modelPath, modelSha256 = modelHash, modelIdentity,
    fixturePath, fixtureSha256 = fixtureHash, fixture, environment, nativePath, nativeSha256 = nativeHash,
    managedModelsSha256 = Hash(typeof(ModelBase).Assembly.Location),
    probeAssemblySha256 = Hash(typeof(ProbeEvidenceValidation).Assembly.Location),
    referenceFiles = referencePaths.Select(path => new { path, sha256 = Hash(path) }), externalFiles,
    binaryFormat = "little-endian float32, contiguous [step][vocabulary], no header",
    meaning = "Exact repeated-history consistency only. A captured reference is a comparator, not proof of mathematical model correctness. No tolerance hides discrepancies.",
    loads, requests
}, json));
Write(false);
try
{
    for (int index = 0; index < sequence.Length; index++)
    {
        string name = sequence[index]; var c = cases[name];
        if (model == null)
        {
            var clock = Stopwatch.StartNew(); model = ModelBase.Create(modelPath, backend); clock.Stop();
            if (model.Config.VocabSize != fixture.VocabSize || model.MaxContextLength != fixture.Context) throw new InvalidDataException("Runtime vocabulary/context differs from fixture");
            using var process = Process.GetCurrentProcess();
            var paths = process.Modules.Cast<ProcessModule>().Where(m => m.ModuleName.Equals(nativeName, StringComparison.OrdinalIgnoreCase)).Select(m => Path.GetFullPath(m.FileName)).Distinct().ToArray();
            if (paths.Length != 1 || !paths[0].Equals(nativePath, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)) throw new InvalidOperationException("Loaded native library does not match the requested app-local file");
            loads.Add(new { beforeRequest = index + 1, seconds = clock.Elapsed.TotalSeconds, modelClass = model.GetType().FullName, nativePaths = paths, kvDtype = model.KvCacheDtype.ToString(), cpuMoeAll = MoeCpuOffloadConfig.AllLayers, cpuMoeLayers = MoeCpuOffloadConfig.CpuMoeLayers, cpuMoeThreads = MoeCpuOffloadConfig.CpuThreads });
        }
        int beforeReset = model.CacheSeqLen; model.ResetKVCache(); int afterReset = model.CacheSeqLen;
        if (afterReset != 0) throw new InvalidOperationException("ResetKVCache did not clear the logical sequence length");
        float[][] rows = new float[fixture.Steps][];
        double[] forwardSeconds = new double[fixture.Steps];
        var forwardClock = Stopwatch.StartNew();
        float[] logits = model.Forward((int[])c.PromptTokens.Clone());
        forwardClock.Stop(); forwardSeconds[0] = forwardClock.Elapsed.TotalSeconds;
        for (int step = 0; step < fixture.Steps; step++)
        {
            if (logits.Length != fixture.VocabSize) throw new InvalidDataException("Unexpected logit width");
            rows[step] = (float[])logits.Clone(); // Forward may reuse its host-visible array.
            if (step + 1 < fixture.Steps)
            {
                forwardClock.Restart();
                logits = model.Forward([c.ForcedTokens[step]]);
                forwardClock.Stop(); forwardSeconds[step + 1] = forwardClock.Elapsed.TotalSeconds;
            }
        }
        int afterForward = model.CacheSeqLen;
        if (afterForward != c.PromptTokens.Length + fixture.Steps - 1) throw new InvalidOperationException("Unexpected forward sequence accounting");
        bool finite = rows.All(row => row.All(float.IsFinite)); allFinite &= finite;
        string binaryFile = $"r{index + 1:D3}-{name}.f32", binaryPath = Path.Combine(output, binaryFile);
        using (var file = new FileStream(binaryPath, FileMode.CreateNew, FileAccess.Write))
            foreach (var row in rows) file.Write(MemoryMarshal.AsBytes(row.AsSpan()));
        Comparison[]? within = null, againstExternal = null;
        if (finite)
        {
            if (references.TryGetValue(name, out var own)) within = rows.Select((row, i) => Compare(row, own[i], i)).ToArray();
            else references[name] = rows;
            if (external.TryGetValue(name, out var other)) againstExternal = rows.Select((row, i) => Compare(row, other[i], i)).ToArray();
        }
        foreach (var comparisons in new[] { within, againstExternal }) if (comparisons != null)
        {
            comparisonCount += comparisons.Length;
            comparedCases.Add(name);
            allExact &= comparisons.All(row => row.DifferentBits == 0);
        }
        var topRows = finite ? rows.Select((row, step) => { var t = TopTwo(row); return new { step, top1 = t.Top, top2 = t.Second, top1Value = row[t.Top], top2Value = row[t.Second], top2Margin = t.Margin, forcedToken = c.ForcedTokens[step] }; }).ToArray() : null;
        requests.Add(new { request = index + 1, caseName = name, promptTokens = c.PromptTokens.Length, steps = fixture.Steps, beforeReset, afterReset, afterForward, finite, forwardSeconds, prefillSeconds = forwardSeconds[0], decodeSeconds = forwardSeconds.Skip(1).Sum(), totalForwardSeconds = forwardSeconds.Sum(), binaryFile, binarySha256 = Hash(binaryPath), binaryBytes = new FileInfo(binaryPath).Length, topRows, within, againstExternal });
        Write(false);
        Console.WriteLine($"[reset-logits] request={index + 1} case={name} session={session} finite={finite} first_within_difference={within?.FirstOrDefault(r => r.DifferentBits != 0)?.Step} max_relative_l2={within?.Max(r => r.RelativeL2):G8}");
        if (!finite) throw new InvalidDataException("Nonfinite logits retained in binary evidence");
        if (session == "fresh") { model.Dispose(); model = null; }
    }
    model?.Dispose(); model = null;
    Write(true);
    return recordOnly ? 0 : ProbeEvidenceValidation.StrictPassed(true, false, allFinite, allExact, comparisonCount, UnverifiedCases()) == true ? 0 : 1;
}
catch (Exception exception)
{
    error = exception.ToString(); Write(false); Console.Error.WriteLine(error); return 2;
}
finally { model?.Dispose(); }
}
catch (Exception exception)
{
    Console.Error.WriteLine(exception);
    return 2;
}

static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
static ModelIdentity ValidateModelIdentity(string modelPath, Fixture fixture)
{
    // The same header-only reader used by ModelBase discovers the complete set.
    // A fixture that hashes only a metadata shard must never identify a model.
    string[] actualPaths;
    using (var gguf = new GgufFile(modelPath)) actualPaths = gguf.FilePaths.Select(Path.GetFullPath).ToArray();
    var pathComparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
    if (!pathComparer.Equals(actualPaths[0], modelPath)) throw new InvalidDataException("--model must name the metadata-bearing first shard");
    bool multiple = fixture.ModelFiles != null;
    var expected = fixture.ModelFiles ?? [new ModelFile(Path.GetFileName(modelPath), fixture.ModelBytes, fixture.ModelSha256)];
    if (expected.Length == 0 || actualPaths.Length != expected.Length)
        throw new InvalidDataException("Fixture ModelFiles must identify every GGUF shard discovered by the model loader");
    var names = new HashSet<string>(pathComparer);
    long total = 0;
    // Check the entire manifest and all lengths before expensive payload reads.
    for (int i = 0; i < expected.Length; i++)
    {
        var e = expected[i];
        if (e == null || string.IsNullOrWhiteSpace(e.FileName) || e.FileName is "." or ".." ||
            e.FileName.IndexOfAny(['/', '\\']) >= 0 || e.FileName.Any(char.IsControl) ||
            Path.GetFileName(e.FileName) != e.FileName || !names.Add(e.FileName) || e.Bytes < 1 ||
            e.Sha256 == null || e.Sha256.Length != 64 || !e.Sha256.All(Uri.IsHexDigit))
            throw new InvalidDataException("Invalid model file identity at index " + i);
        string path = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(modelPath)!, e.FileName));
        if (!pathComparer.Equals(path, actualPaths[i]) || new FileInfo(path).Length != e.Bytes)
            throw new InvalidDataException("Model shard name/order/length differs from fixture: " + e.FileName);
        total = checked(total + e.Bytes);
    }
    if (total != fixture.ModelBytes) throw new InvalidDataException("ModelBytes must equal the total length of every shard");
    var evidence = new List<ModelFileEvidence>();
    var canonical = new StringBuilder("ResetLogitProbe model-files v1\n");
    foreach (var e in expected)
    {
        string path = Path.Combine(Path.GetDirectoryName(modelPath)!, e.FileName);
        var before = new FileInfo(path); long bytes = before.Length; DateTime modifiedUtc = before.LastWriteTimeUtc;
        DateTime startedUtc = DateTime.UtcNow; string hash = Hash(path);
        var after = new FileInfo(path);
        if (after.Length != bytes || after.LastWriteTimeUtc != modifiedUtc || !hash.Equals(e.Sha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Model payload hash changed or differs from fixture: " + e.FileName);
        evidence.Add(new(e.FileName, path, bytes, hash, modifiedUtc, startedUtc, DateTime.UtcNow));
        canonical.Append(e.FileName).Append('\t').Append(bytes.ToString(System.Globalization.CultureInfo.InvariantCulture)).Append('\t').Append(hash).Append('\n');
        Console.WriteLine($"[model-identity] verified {e.FileName} bytes={bytes} sha256={hash}");
    }
    string aggregate = multiple ? Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString()))) : evidence[0].Sha256;
    if (!aggregate.Equals(fixture.ModelSha256, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("ModelSha256 differs from the complete model identity");
    return new(multiple ? "sha256-utf8-model-files-v1" : "single-file-sha256", total, aggregate, evidence.ToArray());
}
static (int Top, int Second, double Margin) TopTwo(float[] row)
{
    int top = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[top]) top = i;
    int second = top == 0 ? 1 : 0; for (int i = 0; i < row.Length; i++) if (i != top && row[i] > row[second]) second = i;
    return (top, second, (double)row[top] - row[second]);
}
static Comparison Compare(float[] actual, float[] reference, int step)
{
    var a = TopTwo(actual); var r = TopTwo(reference);
    double maxAbs = 0, squaredError = 0, normA = 0, normR = 0, dot = 0; int bits = 0, rank = 1;
    for (int i = 0; i < actual.Length; i++)
    {
        double x = actual[i], y = reference[i], delta = x - y;
        maxAbs = Math.Max(maxAbs, Math.Abs(delta)); squaredError += delta * delta; normA += x * x; normR += y * y; dot += x * y;
        if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(reference[i])) bits++;
        if (actual[i] > actual[r.Top]) rank++;
    }
    return new(step, bits, a.Top == r.Top, a.Top, r.Top, a.Margin, r.Margin, rank, maxAbs, Math.Sqrt(squaredError / actual.Length),
        Math.Sqrt(squaredError / Math.Max(normR, 1e-300)), dot / Math.Sqrt(Math.Max(normA * normR, 1e-300)));
}
sealed record Comparison(int Step, int DifferentBits, bool Top1Equal, int ActualTop, int ReferenceTop, double ActualTop2Margin, double ReferenceTop2Margin, int ReferenceTopRankInActual, double MaxAbs, double Rms, double RelativeL2, double Cosine);
sealed record Fixture(int Version, int Steps, int Context, int VocabSize, long ModelBytes, string ModelSha256, string[] Sequence, ProbeCase[] Cases, ModelFile[]? ModelFiles = null);
sealed record ModelFile(string FileName, long Bytes, string Sha256);
sealed record ModelFileEvidence(string FileName, string Path, long Bytes, string Sha256, DateTime LastWriteTimeUtc, DateTime HashStartedUtc, DateTime HashCompletedUtc);
sealed record ModelIdentity(string Scheme, long TotalBytes, string Sha256, ModelFileEvidence[] Files);
sealed record ProbeCase(string Name, int[] PromptTokens, int[] ForcedTokens);
