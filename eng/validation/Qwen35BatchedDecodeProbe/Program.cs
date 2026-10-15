using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Validation;

string Required(string name) => Environment.GetEnvironmentVariable(name)
    ?? throw new ArgumentException($"Set {name}.");
int Number(string name, int fallback) => int.TryParse(Environment.GetEnvironmentVariable(name), out int value) ? value : fallback;
string modelPath = Required("TS_TEST_QWEN35_MODEL");
string output = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_OUT") ?? "artifacts/qwen35-batched-decode/report.json";
bool allowFallback = args.Contains("--allow-fallback", StringComparer.Ordinal);
bool independentPrefill = args.Contains("--independent-prefill", StringComparer.Ordinal);
bool collectErrors = args.Contains("--collect-errors", StringComparer.Ordinal);
bool greedyReference = args.Contains("--greedy-reference", StringComparer.Ordinal);
bool heldOut = args.Contains("--held-out", StringComparer.Ordinal);
if (greedyReference && independentPrefill)
    throw new ArgumentException("--greedy-reference requires matched state; omit --independent-prefill.");
string[]? promptSources = heldOut ? new[]
{
    "A violin has four strings tuned in fifths. An orchestra follows the conductor through changes in rhythm. ",
    "Bread dough rises as yeast releases carbon dioxide. The baker folds the dough before its final rest. ",
    "A lunar eclipse occurs when Earth casts its shadow on the Moon. Observers record its duration and colour. ",
    "Why does the Moon show different phases during a month? Explain how sunlight and its orbit produce the changing appearance. "
} : null;
string? snapshotDirectory = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_SNAPSHOT_DIR");
bool snapshotDiagnostic = !string.IsNullOrWhiteSpace(snapshotDirectory);
string? promptLengthsOverride = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS");
int steps = Number("TS_QWEN35_PROBE_STEPS", 32);
int pairs = Number("TS_QWEN35_PROBE_PAIRS", 3);
int context = Number("MAX_CONTEXT", 2048);
int initial = Number("TS_KV_INITIAL_TOKENS", 128);
int[] widths = (Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_WIDTHS") ?? "2,3,4").Split(',').Select(int.Parse).ToArray();
if (steps is < 2 or > 512 || pairs is < 1 or > 100 || widths.Length == 0 || widths.Any(width => width is < 2 or > 4))
    throw new ArgumentOutOfRangeException("Use 2..512 steps, 1..100 pairs and widths from 2,3,4.");
string backendName = Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cuda";
BackendType backend = backendName switch
{
    "cuda" => BackendType.GgmlCuda,
    "metal" => BackendType.GgmlMetal,
    _ => throw new ArgumentException("This arena probe requires cuda or metal.")
};
Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", initial.ToString());
KvCacheDtypeConfig.ConfigureFromEnvironment();
var runs = new List<Qwen35DecodeComparison>();
var warmups = new List<Qwen35DecodeComparison>();
bool passed = false;
bool completed = false;
bool numericalPassed = false;
bool distributionPassed = false;
string? error = null;
string? nativePath = null;
string? kvDtype = null;
Qwen35DecodeComparison? serialControl = null;
IReadOnlyList<Qwen35ArgmaxFlip>? failureFlips = null;
IReadOnlyList<Qwen35NumericalFailure>? failureRows = null;
try
{
    using var model = ModelBase.Create(modelPath, backend);
    if (model is not Qwen35Model qwen) throw new InvalidOperationException("Requires a Qwen3.5/3.6/3.8 model.");
    kvDtype = model.KvCacheDtype.ToString();
    if (args.Contains("--serial-control", StringComparer.Ordinal))
    {
        Console.WriteLine("[qwen35-probe] serial repeat control (no batching)");
        serialControl = Qwen35DecodeProbe.Compare(qwen, widths.Max(), Math.Max(32, steps), batchedFirst: false, serialControl: true, independentPrefill: independentPrefill, collectNumericalFailures: collectErrors, greedyReference: greedyReference, promptSources: promptSources);
        Console.WriteLine($"[qwen35-probe] serial control width={serialControl.Width} steps={serialControl.Steps} cosine={serialControl.MinCosine:F8} nrmse={serialControl.MaxNormalizedRmse:F6} softmax_kl={serialControl.MaxSoftmaxKl:F8} argmax_differences={serialControl.TopTokenDifferences}");
    }
    foreach (int width in widths)
    {
        Console.WriteLine($"[qwen35-probe] warmup width={width}");
        warmups.Add(Qwen35DecodeProbe.Compare(qwen, width, Math.Min(steps, 4), batchedFirst: true, allowFallback, independentPrefill: independentPrefill, collectNumericalFailures: collectErrors, greedyReference: greedyReference, promptSources: promptSources));
        for (int pair = 0; pair < pairs; pair++)
        {
            var row = Qwen35DecodeProbe.Compare(qwen, width, steps, pair % 2 == 0, allowFallback, independentPrefill: independentPrefill, collectNumericalFailures: collectErrors, greedyReference: greedyReference, promptSources: promptSources);
            runs.Add(row);
            Console.WriteLine($"[qwen35-probe] width={width} pair={pair + 1} serial_ms={row.Serial.Milliseconds:F2} batch_ms={row.Batched.Milliseconds:F2} fused={row.Batched.FusedSteps}/{steps} cosine={row.MinCosine:F8} nrmse={row.MaxNormalizedRmse:F6} centered_nrmse={row.MaxCenteredNormalizedRmse:F6} mean_error_maxabs={row.MaxAbsoluteMeanError:F6} softmax_kl={row.MaxSoftmaxKl:F8} argmax_differences={row.TopTokenDifferences} numerical_failures={row.NumericalFailures.Count} distribution_passed={row.DistributionPassed} greedy_match={row.GreedyContinuationsMatch}");
        }
    }
    completed = true;
    numericalPassed = runs.All(row => row.NumericalPassed) && warmups.All(row => row.NumericalPassed)
        && (serialControl?.NumericalPassed ?? true);
    distributionPassed = runs.All(row => row.DistributionPassed) && warmups.All(row => row.DistributionPassed)
        && (serialControl?.DistributionPassed ?? true);
    passed = !allowFallback && !snapshotDiagnostic && numericalPassed;
    if (!numericalPassed) Environment.ExitCode = 1;
}
catch (Exception exception)
{
    error = exception.ToString();
    failureFlips = exception.Data["ArgmaxFlips"] as Qwen35ArgmaxFlip[];
    failureRows = exception.Data["NumericalFailures"] as Qwen35NumericalFailure[];
    Console.Error.WriteLine(error);
    Environment.ExitCode = 1;
}
finally
{
    string nativeName = OperatingSystem.IsWindows() ? "GgmlOps.dll" : "libGgmlOps.dylib";
    nativePath = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .FirstOrDefault(module => Path.GetFileName(module.FileName).Equals(nativeName, StringComparison.OrdinalIgnoreCase))?.FileName;
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, JsonSerializer.Serialize(new
    {
        Model = Path.GetFullPath(modelPath), ModelBytes = new FileInfo(modelPath).Length, ModelSha256 = Hash(modelPath),
        Backend = backendName, KvDtype = kvDtype, Context = context, InitialCacheTokens = initial,
        Steps = steps, Pairs = pairs, Widths = widths, AllowFallback = allowFallback,
        PromptLengthsOverride = promptLengthsOverride,
        InitialStateMode = independentPrefill ? "independent-prefill" : "shared-checkpoint-clones",
        GreedyReference = greedyReference, Order = greedyReference ? "serial-reference-first" : "alternating-pairs",
        PromptFixture = heldOut ? "held-out-music-baking-astronomy" : "original-short-text", PromptSources = promptSources,
        ValidationPassed = passed, Error = error, WarmupsExcluded = true,
        CollectNumericalFailures = collectErrors, NumericalPassed = numericalPassed, DistributionPassed = distributionPassed,
        GreedyContinuationsMatch = greedyReference ? (bool?)(completed && runs.All(row => row.GreedyContinuationsMatch == true)
            && warmups.All(row => row.GreedyContinuationsMatch == true)
            && (serialControl == null || serialControl.GreedyContinuationsMatch == true)) : null,
        SnapshotDiagnostic = snapshotDiagnostic, SnapshotDirectory = snapshotDirectory,
        PerformanceQualified = !snapshotDiagnostic && !allowFallback && !greedyReference && passed,
        NativePath = nativePath, NativeSha256 = nativePath is null ? null : Hash(nativePath),
        ManagedSha256 = Hash(typeof(Qwen35Model).Assembly.Location),
        ProbeSha256 = Hash(typeof(Qwen35DecodeProbe).Assembly.Location),
        CudaGraphsDisabled = Environment.GetEnvironmentVariable("GGML_CUDA_DISABLE_GRAPHS"),
        CudaFusionDisabled = Environment.GetEnvironmentVariable("GGML_CUDA_DISABLE_FUSION"),
        WeightFusionCopies = Environment.GetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES"),
        Q8ParallelVector = Environment.GetEnvironmentVariable("TS_GGML_Q8_PARALLEL_VECTOR"),
        Q8ParallelSmallBatch = Environment.GetEnvironmentVariable("TS_GGML_Q8_PARALLEL_SMALL_BATCH"),
        Q8PrefillTile = Environment.GetEnvironmentVariable("TS_GGML_Q8_PREFILL_TILE"),
        GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
        Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
        Runs = runs, Warmups = warmups, SerialControl = serialControl,
        FailureArgmaxFlips = failureFlips, FailureNumericalRows = failureRows,
        Summary = runs.GroupBy(row => row.Width).Select(group => new
        {
            Width = group.Key,
            NumericalPassed = group.All(row => row.NumericalPassed),
            DistributionPassed = group.All(row => row.DistributionPassed),
            NumericalFailureRows = group.Sum(row => row.NumericalFailures.Count),
            SerialMedianMs = Median(group.Select(row => row.Serial.Milliseconds)),
            BatchedMedianMs = Median(group.Select(row => row.Batched.Milliseconds)),
            Speedup = Median(group.Select(row => row.Serial.Milliseconds)) / Median(group.Select(row => row.Batched.Milliseconds)),
            SerialTokensPerSecond = group.Key * steps * 1000 / Median(group.Select(row => row.Serial.Milliseconds)),
            BatchedTokensPerSecond = group.Key * steps * 1000 / Median(group.Select(row => row.Batched.Milliseconds)),
            FusedSteps = group.Sum(row => row.Batched.FusedSteps), FallbackSteps = group.Sum(row => row.Batched.FallbackSteps)
        }),
        Tolerances = new { MinimumCosine = 0.995, MaximumSoftmaxKl = 0.02,
            ConfidentArgmaxReferenceLogitMargin = 0.25,
            RawNormalizedRmse = "reported diagnostic, not an acceptance threshold",
            Basis = "Explicit empirical budgets; not a proof of general model quality." },
        Limitations = "Direct teacher-forced decode microbenchmark. Default arms clone identical production checkpoints after one shared prefill and descriptor-initialization token per row; independent-prefill mode is separately labeled. Loading, prefill/checkpoint copying, capacity reservation and final solo continuation are excluded from timing. Clones start host-authoritative; timing includes device reseeding, graph capture and managed logits copies, so it is not steady-state-only throughput. Distribution checks include the final solo continuation; empirical cosine/KL budgets do not prove quality. Raw NRMSE/max errors and every argmax flip remain reported. A prior independently prefilled 4x32 serial-only control failed cosine tolerance without batching; matched-state results do not resolve or erase that limitation. No speedup is assumed or asserted. Allow-fallback results are diagnostic, not proof of batching. Only reported model/backend/cache scenarios are covered; independent request prefill, multi-agent reasoning and HTTP latency require separate end-to-end checks. Dependency revision/device are operator-supplied metadata."
    }, new JsonSerializerOptions { WriteIndented = true }));
}

static string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream));
}
static double Median(IEnumerable<double> values)
{
    double[] sorted = values.Order().ToArray();
    return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
}
