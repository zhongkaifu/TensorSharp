// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 >= args.Length) throw new ArgumentException("Use --name value pairs.");
    options.Add(args[i], args[i + 1]);
}
if (options.GetValueOrDefault("--self-test") == "true") { MoeMath.SelfTest(); Capture.SelfTest(); return 0; }
string captureRoot = Path.GetFullPath(options["--captures"]);
string modelsPath = Path.GetFullPath(options["--models-assembly"]);
string outputPath = Path.GetFullPath(options["--output"]);
if (File.Exists(outputPath)) throw new IOException("Use a fresh report path.");
long maxBytes = long.Parse(options.GetValueOrDefault("--max-bytes", "67108864"));
if (maxBytes <= 0 || maxBytes > 1L << 30) throw new ArgumentException("Explicit positive capture limit <=1GiB required.");
string indexPath = Path.Combine(captureRoot, "index.jsonl");
if (new FileInfo(indexPath).Length > 8 << 20) throw new InvalidDataException("Capture index exceeds bounded diagnostic size.");
var captures = File.ReadLines(indexPath).Where(x => !string.IsNullOrWhiteSpace(x))
    .Select(x => JsonSerializer.Deserialize<Capture>(x) ?? throw new InvalidDataException("Invalid capture index.")).ToArray();
if (captures.Sum(x => x.NBytes) > maxBytes) throw new InvalidDataException("Capture payload exceeds explicit CPU diagnostic limit.");
foreach (var capture in captures) capture.Load(captureRoot);
// Cached graph addresses and per-object counters can be reused after teardown.
// Explicit ordinary-input / selected-boundary markers identify ordered frames;
// check the full recorded context remains constant inside each such frame.
Capture.AssignFrames(captures);
var groups = captures.GroupBy(x => (x.Frame, x.Mode, x.Model, x.Graph, x.Evaluation, x.Layer))
    .Where(g => g.Any(c => c.Name == "input") && g.Any(c => c.Name == "router_ids"))
    .OrderBy(g => g.Min(c => c.Id)).ToArray();
var ordinary = groups.First(g => g.Key.Mode == "ordinary").ToArray();
var compact = groups.First(g => g.Key.Mode == "compact").ToArray();
Capture One(Capture[] group, string name) => group.Single(x => x.Name == name);
var originalInput = One(ordinary, "input");
int hidden = checked((int)originalInput.Ne[0]), rows = originalInput.Tokens;
int used = checked((int)One(ordinary, "router_ids").Ne[0]);
int ffn = checked((int)One(ordinary, "gate_weight").Ne[1]);
long decodedBytes = new[] { "gate_weight", "up_weight", "down_weight" }.Sum(name => {
    var shape = One(ordinary, name).Ne;
    return checked(shape[0] * shape[1] * shape[2] * 4);
});
long maxDecoded = long.Parse(options.GetValueOrDefault("--max-decoded-bytes", "134217728"));
long dotProducts = checked((long)rows * used * hidden * ffn * 3);
long maxDots = long.Parse(options.GetValueOrDefault("--max-dot-products", "67108864"));
if (maxDecoded <= 0 || maxDecoded > 1L << 30 || decodedBytes > maxDecoded || maxDots <= 0 || dotProducts > maxDots)
    throw new InvalidDataException("Decoded weight storage or independent CPU arithmetic exceeds its explicit diagnostic limit.");
double clamp = originalInput.Clamp;
double[] input = originalInput.Matrix(hidden, rows);
int[] ids = One(ordinary, "router_ids").RoutingIds(used, rows);
double[] routes = One(ordinary, "route_weights").Routed(1, used, rows);
bool BitwiseEqual(IEnumerable<double> a, IEnumerable<double> b) =>
    a.Select(BitConverter.DoubleToInt64Bits).SequenceEqual(b.Select(BitConverter.DoubleToInt64Bits));
bool matchedInput = BitwiseEqual(input, One(compact, "input").Matrix(hidden, rows));
bool matchedIds = ids.SequenceEqual(One(compact, "router_ids").RoutingIds(used, rows));
bool matchedRoutes = BitwiseEqual(routes, One(compact, "route_weights").Routed(1, used, rows));
bool matchedSource = new[] { "gate_weight", "up_weight", "down_weight" }
    .All(n => One(ordinary, n).Sha256 == One(compact, n).Sha256 && One(ordinary, n).Type == One(compact, n).Type
        && One(ordinary, n).Ne.SequenceEqual(One(compact, n).Ne) && One(ordinary, n).Nb.SequenceEqual(One(compact, n).Nb));
if (!matchedInput || !matchedIds || !matchedRoutes || !matchedSource || clamp != One(compact, "input").Clamp)
    throw new InvalidDataException("First captured paths have different exact inputs/IDs/routes/source/clamp; mathematical ranking refused.");

// Use TensorSharp's independent managed GGUF block decoder, not a native dot
// kernel or ggml dequantization entry point. Assembly/dependencies are pinned.
string assemblyRoot = Path.GetDirectoryName(modelsPath)!;
AssemblyLoadContext.Default.Resolving += (_, name) =>
    File.Exists(Path.Combine(assemblyRoot, name.Name + ".dll")) ? Assembly.LoadFrom(Path.Combine(assemblyRoot, name.Name + ".dll")) : null;
Assembly models = Assembly.LoadFrom(modelsPath);
var decode = models.GetType("TensorSharp.Models.ManagedQuantizedOps", true)!
    .GetMethod("DequantizeToFloat32", BindingFlags.Public | BindingFlags.Static, null,
        [typeof(int), typeof(byte[]), typeof(int), typeof(float[]), typeof(int), typeof(long)], null)!
    .CreateDelegate<Action<int, byte[], int, float[], int, long>>();
float[][][] Decode(Capture weight)
{
    if (weight.TypeName is not ("q8_0" or "iq1_s" or "iq2_xxs" or "iq3_xxs" or "iq4_xs") || weight.Ne[3] != 1)
        throw new NotSupportedException("Only qualified original Q8_0/IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS blocks are accepted.");
    int width = checked((int)weight.Ne[0]), columns = checked((int)weight.Ne[1]), experts = checked((int)weight.Ne[2]);
    var values = new float[experts][][];
    for (int expert = 0; expert < experts; ++expert)
    {
        values[expert] = new float[columns][];
        for (int column = 0; column < columns; ++column)
        {
            values[expert][column] = new float[width];
            int offset = checked((int)(expert * weight.Nb[2] + column * weight.Nb[1]));
            decode(weight.Type, weight.Bytes, offset, values[expert][column], 0, width);
            if (values[expert][column].Any(x => !float.IsFinite(x))) throw new ArithmeticException("Nonfinite independent weight decode.");
        }
    }
    return values;
}
float[][][] gate = Decode(One(ordinary, "gate_weight")), up = Decode(One(ordinary, "up_weight")), down = Decode(One(ordinary, "down_weight"));
MoeMath.Stages Actual(Capture[] group, bool cached)
{
    if (!cached) return new(One(group, "gate_projection").Routed(ffn, used, rows),
        One(group, "up_projection").Routed(ffn, used, rows), One(group, "activation").Routed(ffn, used, rows),
        One(group, "down_projection").Routed(hidden, used, rows), One(group, "output").Matrix(hidden, rows));
    var result = new MoeMath.Stages(new double[rows * used * ffn], new double[rows * used * ffn],
        new double[rows * used * ffn], new double[rows * used * hidden], One(group, "output").Matrix(hidden, rows));
    var covered = new bool[rows];
    foreach (var chunk in group.Where(x => x.Name == "compact_input").OrderBy(x => x.Row))
    {
        int start = chunk.Row, count = chunk.Rows;
        bool bulkLayout = chunk.Ne[1] == 1 && chunk.Ne[2] == count;
        if (count <= 0 || start < 0 || start + count > rows || covered.AsSpan(start, count).Contains(true))
            throw new InvalidDataException("Overlapping/out-of-range compact chunk.");
        var chunkGroup = group.Where(x => x.Row == start && x.Rows == count && x.Name.StartsWith("compact_", StringComparison.Ordinal)).ToArray();
        if (!BitwiseEqual(chunk.TokenRows(hidden, count), input.Skip(start * hidden).Take(count * hidden)))
            throw new InvalidDataException("Compact bridge changed exact F32 input rows.");
        if (!BitwiseEqual(One(chunkGroup, "compact_weights").Routed(1, used, count), routes.Skip(start * used).Take(count * used)))
            throw new InvalidDataException("Compact bridge changed route weights.");
        var remapped = One(chunkGroup, "compact_remapped_ids");
        foreach (string part in new[] { "gate", "up", "down" })
        {
            var source = One(group, part + "_weight");
            var resident = One(chunkGroup, "compact_resident_" + part + "_weight");
            if (resident.Type != source.Type || resident.Ne[0] != source.Ne[0] || resident.Ne[1] != source.Ne[1])
                throw new InvalidDataException("Compact resident quantization geometry differs.");
            int stride = checked((int)source.Nb[2]);
            for (int route = 0; route < used; ++route) for (int row = 0; row < count; ++row)
            {
                int slot = bulkLayout ? remapped.I32(route, row) : remapped.I32(route * count + row);
                int expert = ids[(start + row) * used + route];
                if (slot < 0 || slot >= resident.Ne[2] || !resident.Bytes.AsSpan(checked(slot * (int)resident.Nb[2]), stride)
                    .SequenceEqual(source.Bytes.AsSpan(checked(expert * stride), stride)))
                    throw new InvalidDataException("Compact resident selected block identity mismatch.");
            }
        }
        void Copy(string name, int width, double[] destination)
        {
            var values = One(chunkGroup, name).Routed(width, used, count, !bulkLayout);
            values.CopyTo(destination, start * used * width);
        }
        Copy("compact_gate_projection", ffn, result.Gate); Copy("compact_up_projection", ffn, result.Up);
        Copy("compact_activation", ffn, result.Activation); Copy("compact_down_projection", hidden, result.Down);
        if (!BitwiseEqual(One(chunkGroup, "compact_output").Matrix(hidden, count), result.Output.Skip(start * hidden).Take(count * hidden)))
            throw new InvalidDataException("Compact chunk/main output mismatch.");
        Array.Fill(covered, true, start, count);
    }
    if (covered.Any(x => !x)) throw new InvalidDataException("Incomplete compact row coverage.");
    return result;
}
var ordinaryValues = Actual(ordinary, false);
var compactValues = Actual(compact, true);
var ideal = MoeMath.Evaluate(input, ids, routes, gate, up, down, hidden, ffn, used, rows, clamp);
var rounded = MoeMath.Evaluate(input, ids, routes, gate, up, down, hidden, ffn, used, rows, clamp, true);
object Metrics(MoeMath.Stages reference, MoeMath.Stages actual) => new {
    gate = MoeMath.Compare(reference.Gate, actual.Gate), up = MoeMath.Compare(reference.Up, actual.Up),
    activation = MoeMath.Compare(reference.Activation, actual.Activation), down = MoeMath.Compare(reference.Down, actual.Down),
    output = MoeMath.Compare(reference.Output, actual.Output) };
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
File.WriteAllText(outputPath, JsonSerializer.Serialize(new {
    diagnosis_completed = true, hidden, ffn, used, rows, clamp,
    captured_payload_bytes = captures.Sum(c => c.NBytes), decoded_weight_bytes = decodedBytes,
    independent_reference_dot_terms_per_evaluation = dotProducts,
    exact_inputs_equal = matchedInput, exact_full_router_ids_equal = matchedIds, exact_route_weights_equal = matchedRoutes,
    original_quant_blocks_equal = matchedSource, compact_selected_resident_blocks_equal = true,
    ordinary_vs_fp64_math = Metrics(ideal, ordinaryValues), compact_vs_fp64_math = Metrics(ideal, compactValues),
    ordinary_vs_explicit_fp32_stages = Metrics(rounded, ordinaryValues), compact_vs_explicit_fp32_stages = Metrics(rounded, compactValues),
    ordinary_vs_compact = Metrics(ordinaryValues, compactValues),
    ordinary_local_stage_error = MoeMath.LocalStageMetrics(ordinaryValues, ids, routes, down, hidden, ffn, used, rows, clamp),
    compact_local_stage_error = MoeMath.LocalStageMetrics(compactValues, ids, routes, down, hidden, ffn, used, rows, clamp),
    index_sha256 = Hash(indexPath), models_sha256 = Hash(modelsPath),
    probe_sha256 = Hash(Assembly.GetExecutingAssembly().Location),
    loaded_managed_assemblies = AppDomain.CurrentDomain.GetAssemblies().Where(a => !a.IsDynamic && a.GetName().Name!.StartsWith("TensorSharp", StringComparison.Ordinal))
        .Select(a => new { path = a.Location, sha256 = Hash(a.Location) }).ToArray(),
    captures = captures.Select(c => new { c.Id, c.Frame, c.Name, c.Mode, c.Model, c.Graph, c.Evaluation, c.Layer, c.Row, c.Rows, c.TypeName, c.Ne, c.Nb, c.NBytes, c.Sha256 }).ToArray(),
    scope = "First captured ordinary/compact MoE layer on identical full F32 inputs, IDs, routing weights and original quantized blocks. Independent managed block decode followed by scalar FP64 dots/clamp/SiLU/down/route-ordered fold.",
    limitations = new[] { "Diagnostic OUTPUT lifetimes can alter fusion; no production timing or full language-quality qualification.",
        "FP64 math defines accuracy for the captured exact F32 input and independently decoded F32 weights. Explicit FP32 stages are a supplemental rounding model, not a CUDA kernel oracle.",
        "No strict gate is relaxed. Diagnosis completion is not bitwise identity or shared physical memory-budget acceptance.",
        "Captured payload and decoded CPU weights are bounded diagnostic allocations outside production shared-budget scope." }
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Completed MoE diagnosis N={rows}, H={hidden}, FF={ffn}, K={used}: {outputPath}");
return 0;
