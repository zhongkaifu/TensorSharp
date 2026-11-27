using System.Diagnostics;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models.Ltx25;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
var allowed = new HashSet<string>(["output", "shapes", "warmup", "repeats"]);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--") || !allowed.Contains(args[i][2..]) ||
        !options.TryAdd(args[i][2..], args[i + 1])) throw new ArgumentException("Expected unique known --option value pairs.");
string output = Path.GetFullPath(options.GetValueOrDefault("output") ?? throw new ArgumentException("Missing --output."));
if (File.Exists(output)) throw new IOException("Refusing to overwrite a report.");
int Number(string name, int fallback) => int.Parse(options.GetValueOrDefault(name, fallback.ToString()), System.Globalization.CultureInfo.InvariantCulture);
int warmup = Number("warmup", 2), repeats = Number("repeats", 7);
if (warmup < 0 || repeats < 1) throw new ArgumentException("Warmup must be nonnegative and repeats positive.");
int[][] shapes = options.GetValueOrDefault("shapes",
    "192:1024:32:128,768:1024:32:128,18:1024:32:64,3072:126:32:64,126:3072:32:64,3:10752:32:64")
    .Split(',').Select(spec => spec.Split(':').Select(int.Parse).ToArray()).ToArray();
if (shapes.Any(shape => shape.Length != 4 || shape.Any(value => value < 1) || shape[2] > 128 || shape[3] > 1024))
    throw new ArgumentException("Shapes must be queries:keys:heads:headDim, with positive supported dimensions.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var cases = new List<object>();
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["created_utc"] = DateTimeOffset.UtcNow,
    ["warmup"] = warmup, ["repeats"] = repeats,
    ["scope"] = "Resident synthetic Float32 unbiased attention only: previous DirectOps.Attention dispatch versus CudaLtx25AttentionSession. Both routes allocate their output inside each measured call. Alternating synchronized samples include managed dispatch, output allocation, and GPU execution; exclude input upload, first JIT/module initialization, output readback, projection, sampling, decoding, and media export. This is not end-to-end generation or perceptual validation.",
    ["baseline_invocation"] = "A compiled expression calls the existing internal DirectOps.Attention method; reflection is used only once outside measurements to bind the method, without changing library visibility.",
    ["environment"] = new[] { "TS_WAN_DIRECT_FUSED_ATTN", "TS_WAN_DIRECT_TIMING" }
        .ToDictionary(name => name, Environment.GetEnvironmentVariable),
    ["managed_binaries"] = new[] { Assembly.GetExecutingAssembly(), typeof(Ltx25Denoiser).Assembly, typeof(CudaAllocator).Assembly, typeof(Tensor).Assembly }
        .Select(assembly => new { name = assembly.GetName().Name, path = assembly.Location, sha256 = HashFile(assembly.Location) }).ToArray(),
    ["cases"] = cases,
};
var json = new JsonSerializerOptions { WriteIndented = true };
void Save() => File.WriteAllText(output, JsonSerializer.Serialize(report, json) + "\n");
try
{
    // Resolve the exact module the backend will use, rather than accepting a
    // caller-supplied path that may differ from the actual loaded binary.
    var locate = typeof(CudaKernels).GetMethod("LocatePtxPath", BindingFlags.Static | BindingFlags.NonPublic,
        null, [typeof(string)], null) ?? throw new MissingMethodException("CUDA PTX resolver not found.");
    string primaryPath = (string?)locate.Invoke(null, ["tensorsharp_kernels.ptx"])
        ?? throw new FileNotFoundException("CUDA primary PTX unavailable.");
    report["primary_ptx"] = new { path = primaryPath, sha256 = HashFile(primaryPath) };
    using var allocator = new CudaAllocator();
    allocator.Context.MakeCurrent();
    var deviceName = new byte[256];
    CudaDriverApi.cuDeviceGet(out int device, 0).ThrowOnError();
    CudaDriverApi.cuDeviceGetName(deviceName, deviceName.Length, device).ThrowOnError();
    report["device"] = Encoding.UTF8.GetString(deviceName).TrimEnd('\0');
    Type contextType = typeof(Ltx25Denoiser).Assembly.GetType("TensorSharp.Models.Direct.DirectContext", true)!;
    using var context = (IDisposable)Activator.CreateInstance(contextType, allocator)!;
    var baseline = BindBaseline(context, contextType);
    CudaDriverApi.cuEventCreate(out IntPtr startEvent, 0).ThrowOnError();
    IntPtr endEvent = IntPtr.Zero;
    try
    {
        CudaDriverApi.cuEventCreate(out endEvent, 0).ThrowOnError();
        foreach (int[] shape in shapes)
        {
            int queries = shape[0], keys = shape[1], heads = shape[2], dim = shape[3], width = checked(heads * dim);
            float scale = 1f / MathF.Sqrt(dim);
            float[] qValues = Values(checked(queries * width), 0x19201024),
                kValues = Values(checked(keys * width), 0x76801024), vValues = Values(checked(keys * width), 0x12603072);
            using var query = new Tensor(allocator, DType.Float32, queries, width);
            using var key = new Tensor(allocator, DType.Float32, keys, width);
            using var value = new Tensor(allocator, DType.Float32, keys, width);
            query.SetElementsAsFloat(qValues); key.SetElementsAsFloat(kValues); value.SetElementsAsFloat(vValues);
            using var owned = new CudaLtx25AttentionSession(allocator);
            Tensor Run(bool optimized)
            {
                if (!optimized) return baseline(query, key, value, heads, dim, scale);
                var result = new Tensor(allocator, DType.Float32, queries, width);
                try
                {
                    if (!owned.TryAttention(result, query, key, value, heads, dim, scale))
                        throw new InvalidOperationException("Owned attention refused a requested shape.");
                    return result;
                }
                catch { result.Dispose(); throw; }
            }
            for (int i = 0; i < warmup; ++i)
            {
                using var old = Run(false); using var current = Run(true); allocator.Synchronize();
            }
            // Uploads, JIT/module initialization and reference readback are all
            // completed before the measured launches, even with --warmup 0.
            float[] Read(bool optimized)
            {
                using var result = Run(optimized); allocator.Synchronize();
                return result.GetElementsAsFloat(checked(queries * width));
            }
            float[] expected = Read(false), actual = Read(true);
            var oldWall = new List<double>(); var ownedWall = new List<double>();
            var oldGpu = new List<double>(); var ownedGpu = new List<double>();
            void Time(bool optimized)
            {
                allocator.Synchronize();
                var watch = Stopwatch.StartNew();
                CudaDriverApi.cuEventRecord(startEvent, allocator.Stream.Handle).ThrowOnError();
                using var result = Run(optimized);
                CudaDriverApi.cuEventRecord(endEvent, allocator.Stream.Handle).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(endEvent).ThrowOnError();
                watch.Stop();
                CudaDriverApi.cuEventElapsedTime(out float milliseconds, startEvent, endEvent).ThrowOnError();
                (optimized ? ownedWall : oldWall).Add(watch.Elapsed.TotalMilliseconds);
                (optimized ? ownedGpu : oldGpu).Add(milliseconds);
            }
            for (int round = 0; round < repeats; ++round) { Time(round % 2 != 0); Time(round % 2 == 0); }
            float[] oldRepeated = Read(false), ownedRepeated = Read(true);
            bool oldStable = expected.AsSpan().SequenceEqual(oldRepeated), ownedStable = actual.AsSpan().SequenceEqual(ownedRepeated);
            double squareError = 0, squareReference = 0, maxError = 0;
            for (int i = 0; i < expected.Length; ++i)
            {
                if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i])) throw new InvalidOperationException("Nonfinite attention output.");
                double error = (double)actual[i] - expected[i];
                squareError += error * error; squareReference += (double)expected[i] * expected[i]; maxError = Math.Max(maxError, Math.Abs(error));
            }
            double rmse = Math.Sqrt(squareError / expected.Length), relativeL2 = Math.Sqrt(squareError / Math.Max(squareReference, 1e-30));
            bool matches = maxError <= 2e-5 && relativeL2 <= 2e-5;
            cases.Add(new
            {
                queries, keys, heads, head_dim = dim, scale, query_key_pairs = (long)queries * keys,
                query_sha256 = HashFloats(qValues), key_sha256 = HashFloats(kValues), value_sha256 = HashFloats(vValues),
                baseline_output_sha256 = HashFloats(expected), owned_output_sha256 = HashFloats(actual),
                all_finite = true, maximum_absolute_error = maxError, rmse, relative_l2 = relativeL2,
                maximum_absolute_tolerance = 2e-5, relative_l2_tolerance = 2e-5, within_tolerance = matches,
                baseline_exact_repeat = oldStable, owned_exact_repeat = ownedStable,
                baseline_gpu_ms = oldGpu, owned_gpu_ms = ownedGpu,
                baseline_gpu_median_ms = Median(oldGpu), owned_gpu_median_ms = Median(ownedGpu), gpu_speedup = Median(oldGpu) / Median(ownedGpu),
                baseline_wall_ms = oldWall, owned_wall_ms = ownedWall,
                baseline_wall_median_ms = Median(oldWall), owned_wall_median_ms = Median(ownedWall), wall_speedup = Median(oldWall) / Median(ownedWall),
                owned_executions = owned.ExecutedAttentions, workspace_bytes = owned.ScoreCapacityBytes,
                query_tile_rows = owned.LastQueryTileRows, head_tile_count = owned.LastHeadTileCount,
            });
            Save();
            if (owned.ExecutedAttentions != warmup + repeats + 2 || !matches || !oldStable || !ownedStable)
                throw new InvalidOperationException("Attention numerical, determinism or execution-count gate failed.");
            Console.WriteLine($"{queries}x{keys} h={heads} d={dim}: {Median(oldWall):F3} -> {Median(ownedWall):F3} ms, " +
                $"speedup={Median(oldWall) / Median(ownedWall):F3}x, maxError={maxError:G6}, relativeL2={relativeL2:G6}");
        }
        report["completed"] = true; Save(); return 0;
    }
    finally
    {
        allocator.Synchronize(); CudaDriverApi.cuEventDestroy(startEvent);
        if (endEvent != IntPtr.Zero) CudaDriverApi.cuEventDestroy(endEvent);
    }
}
catch (Exception error)
{
    report["error_type"] = error.GetType().FullName; report["error"] = error.Message;
    Save(); Console.Error.WriteLine(error); return 1;
}

static Func<Tensor, Tensor, Tensor, int, int, float, Tensor> BindBaseline(object context, Type contextType)
{
    Type operations = typeof(Ltx25Denoiser).Assembly.GetType("TensorSharp.Models.Direct.DirectOps", true)!;
    MethodInfo method = operations.GetMethod("Attention", BindingFlags.Public | BindingFlags.Static)
        ?? throw new MissingMethodException("Direct attention method not found.");
    var q = Expression.Parameter(typeof(Tensor), "query"); var k = Expression.Parameter(typeof(Tensor), "key");
    var v = Expression.Parameter(typeof(Tensor), "value"); var h = Expression.Parameter(typeof(int), "heads");
    var d = Expression.Parameter(typeof(int), "headDim"); var s = Expression.Parameter(typeof(float), "scale");
    var call = Expression.Call(method, Expression.Constant(context, contextType), q, k, v, h, d, s, Expression.Constant(null, typeof(Tensor)));
    return Expression.Lambda<Func<Tensor, Tensor, Tensor, int, int, float, Tensor>>(call, q, k, v, h, d, s).Compile();
}

static float[] Values(int count, uint seed)
{
    var result = new float[count];
    for (int i = 0; i < count; ++i)
    {
        // Defined integer generator makes input bits independent of framework
        // Random implementations and host trigonometric libraries.
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        result[i] = ((seed >> 8) * (1f / 8388608f) - 1f) * .75f;
    }
    return result;
}
static double Median(List<double> samples)
{
    double[] sorted = samples.Order().ToArray();
    return (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;
}
static string HashFloats(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
static string HashFile(string path)
{
    using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream));
}
