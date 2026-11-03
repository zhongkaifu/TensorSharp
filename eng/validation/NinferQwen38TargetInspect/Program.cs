using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static class Program
{
    private const string Usage = "Read-only known-family target plan, NOT inference. --artifact FILE --output FRESH_IGNORED_JSON --expect-directory-sha SHA [--known-sha256 SUPPLIED_UNVERIFIED_SHA]";
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase, Converters = { new JsonStringEnumConverter() } };
    private static int Main(string[] args)
    {
        if (args.Length == 1 && args[0] is "--help" or "-h") { Console.WriteLine(Usage); return 0; }
        Dictionary<string, string> options;
        string output;
        try
        {
            options = Parse(args); output = Path.GetFullPath(options["--output"]); CheckOutput(output);
        }
        catch (Exception e) when (e is ArgumentException or IOException or UnauthorizedAccessException)
        { Console.Error.WriteLine(e.Message + "\n" + Usage); return 2; }
        var report = new Dictionary<string, object?>
        {
            ["schema"] = "tensorsharp-ninfer-qwen38-target-plan-inspection/1", ["complete"] = false,
            ["knownFullArtifactSha256"] = options.GetValueOrDefault("--known-sha256"),
            ["knownShaStatus"] = "SUPPLIED_NOT_VERIFIED_BY_THIS_TOOL", ["fullArtifactSha256Recomputed"] = false,
            ["inference"] = "NOT_IMPLEMENTED", ["tokenizerBinding"] = "NOT_IMPLEMENTED", ["draftBinding"] = "NOT_IMPLEMENTED",
            ["visionBinding"] = "NOT_IMPLEMENTED", ["modelQuality"] = "NOT_RUN", ["GPU"] = "NOT_RUN", ["performance"] = "NOT_RUN"
        };
        int code = 1;
        try
        {
            report["execution"] = new { tool = AssemblyIdentity(Assembly.GetExecutingAssembly()), models = AssemblyIdentity(typeof(NinferQwen38TargetPlan).Assembly), runtime = AssemblyIdentity(typeof(NinferArtifactReader).Assembly) };
            report["artifact"] = Path.GetFullPath(options["--artifact"]); object before = Snapshot(options["--artifact"]); report["fileStatBefore"] = before;
            using var reader = new NinferArtifactReader(options["--artifact"]);
            Require(reader.DirectorySha256 == options["--expect-directory-sha"], "Actual directory SHA differs.");
            long bytesBeforePlan = reader.TotalBytesRead, callsBeforePlan = reader.ReadOperations;
            var plan = NinferQwen38TargetPlan.Create(reader);
            Require(reader.TotalBytesRead == bytesBeforePlan && reader.ReadOperations == callsBeforePlan, "Target binding unexpectedly read tensor/resource contents.");
            Require(plan.Tensors.Count == 906 && plan.Layers.Count == 64 && plan.TargetNvfp4WeightDivisorsValidated == 247, "Target plan counts differ.");
            Require(plan.Layers.Count(x => x.Kind == NinferQwen38LayerKind.GatedDeltaNet) == 48 && plan.Layers.Count(x => x.Kind == NinferQwen38LayerKind.GroupedQueryAttention) == 16, "Layer pattern differs.");
            Require(plan.RequiresGroupedGdnKernel && Enumerable.Range(0, 48).All(h => NinferQwen38TargetPlan.GdnQueryKeyHead(h) == h / 3), "Source grouped head semantics differ.");
            Require(!plan.InferenceSupported && !plan.TokenizerBound && !plan.DraftBound && !plan.VisionBound, "Metadata plan must not enable inference/frontends.");
            Require(plan.TokenEmbedding.Descriptor.Format == NinferTensorFormat.W8G32Float16Scale && plan.OutputHead.Descriptor.Format == NinferTensorFormat.W8G32Float16Scale, "W8 endpoints differ.");
            var exceptions = plan.Layers.SelectMany(x => new[] { x.InputProjection, x.OutputProjection }).Where(x => x.Descriptor.Format == NinferTensorFormat.BFloat16).Select(x => x.Descriptor.Name).Order().ToArray();
            string[] expectedExceptions = { "text/layers/3/attention/query_key_gate_value", "text/layers/7/attention/query_key_gate_value", "text/layers/11/attention/query_key_gate_value", "text/layers/15/attention/query_key_gate_value", "text/layers/19/attention/query_key_gate_value", "text/layers/23/attention/query_key_gate_value", "text/layers/3/attention/output", "text/layers/7/attention/output", "text/layers/4/gdn/output" };
            Require(exceptions.SequenceEqual(expectedExceptions.Order()), "Nine BF16 exceptions differ.");
            foreach (var tensor in plan.Tensors)
            {
                Require(ReferenceEquals(reader.GetObject(tensor.Descriptor.Name), tensor.Descriptor), "Target tensor lost original reader ownership.");
                if (tensor.InputScaleDivisor is { } input)
                    Require(ReferenceEquals(reader.GetObject(input.Name), input) && input.WeightDivisor == null && input.WeightDivisorBits == null
                        && tensor.InputScaleDivisorContentValidation == NinferQwen38ContentValidation.NotRead, "Activation divisor must remain separate unread descriptor.");
            }
            Require(plan.Tensors.Count(x => x.InputScaleDivisor != null) == 247, "Activation divisor association count differs.");
            string[] resourceNames = { "frontend/tokenizer.json", "frontend/tokenizer_config.json", "frontend/chat_template.jinja", "frontend/generation_config.json", "frontend/preprocessor_config.json", "frontend/video_preprocessor_config.json" };
            Require(plan.Resources.Count == 6 && plan.Resources.Select(x => x.Name).Order().SequenceEqual(resourceNames.Order()), "Six original resource descriptors differ.");
            report["plan"] = new
            {
                plan.Identity, plan.ArtifactPath, plan.DirectorySha256, bindingVersion = NinferQwen38TargetPlan.BindingVersion,
                targetTensors = plan.Tensors.Count, layers = plan.Layers.Count, gdnLayers = 48, attentionLayers = 16,
                weightDivisorsPositiveFiniteAtReaderInitialization = plan.TargetNvfp4WeightDivisorsValidated,
                activationDivisorDescriptors = 247, activationDivisorContents = "NOT_READ_NOT_VALIDATED",
                resourceContents = "NOT_READ_NOT_BOUND", readerOwnershipPreserved = true,
                plan.RequiresGroupedGdnKernel, headMap = "value_head / 3; legacy h%16 IS NOT COMPATIBLE",
                largeBf16Exceptions = exceptions,
                tensors = plan.Tensors.Select(t => new
                {
                    t.Descriptor.Name, t.Descriptor.Format, t.Descriptor.Layout, t.Descriptor.Shape, t.Descriptor.PayloadOffset, t.Descriptor.AbsoluteOffset,
                    t.Descriptor.ByteLength, t.Descriptor.Geometry, weightDivisorRawBits = t.WeightDivisorBits?.ToString("x8"),
                    t.Role, t.NormConvention, t.RowOrder, t.ParameterSemantics, t.ContentValidation,
                    inputScaleDivisor = t.InputScaleDivisor == null ? null : new { t.InputScaleDivisor.Name, t.InputScaleDivisor.AbsoluteOffset, t.InputScaleDivisor.ByteLength, content = "NOT_READ_NOT_VALIDATED" }
                }).ToArray(),
                layerBindings = plan.Layers.Select(x => new { x.Index, x.Kind, input = x.InputProjection.Descriptor.Name, output = x.OutputProjection.Descriptor.Name, gateUp = x.GateUp.Descriptor.Name, down = x.Down.Descriptor.Name, alphaBeta = x.AlphaBeta?.Descriptor.Name }).ToArray(),
                resources = plan.Resources.Select(x => new { x.Name, x.Encoding, x.PayloadOffset, x.AbsoluteOffset, x.ByteLength }).ToArray()
            };
            report["readStatistics"] = new { reader.InitializationBytesRead, reader.InitializationReadOperations, reader.TotalBytesRead, reader.ReadOperations, targetPlanExtraBytes = reader.TotalBytesRead - bytesBeforePlan, targetPlanExtraReadCalls = reader.ReadOperations - callsBeforePlan };
            object after = Snapshot(options["--artifact"]); report["fileStatAfter"] = after;
            Require(JsonSerializer.Serialize(before, Json) == JsonSerializer.Serialize(after, Json), "Artifact stat/path changed.");
            report["fileIdentityScope"] = "Length/mtime/creation/link-target only; external full SHA/inode guard required.";
            report["complete"] = true; code = 0;
        }
        catch (Exception e) { report["error"] = new { type = e.GetType().Name, e.Message }; }
        try
        {
            CheckOutput(output); Directory.CreateDirectory(Path.GetDirectoryName(output)!); CheckOutput(output);
            using var file = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.None); JsonSerializer.Serialize(file, report, Json); file.WriteByte((byte)'\n');
        }
        catch (Exception e) { Console.Error.WriteLine("Output refused: " + e.Message); return 2; }
        Console.WriteLine(JsonSerializer.Serialize(new { output, complete = code == 0, exitCode = code }, Json)); return code;
    }
    private static Dictionary<string, string> Parse(string[] args)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i];
            if (key is not ("--artifact" or "--output" or "--expect-directory-sha" or "--known-sha256") || ++i == args.Length || string.IsNullOrWhiteSpace(args[i]) || !result.TryAdd(key, args[i])) throw new ArgumentException("Unknown/duplicate/missing option: " + key);
        }
        foreach (string key in new[] { "--artifact", "--output", "--expect-directory-sha" }) if (!result.ContainsKey(key)) throw new ArgumentException("Required: " + key);
        foreach (string key in new[] { "--expect-directory-sha", "--known-sha256" }) if (result.TryGetValue(key, out string? value))
        { if (value.Length != 64 || !value.All(Uri.IsHexDigit)) throw new ArgumentException("Invalid SHA: " + key); result[key] = value.ToLowerInvariant(); }
        return result;
    }
    private static void CheckOutput(string path)
    {
        string[] parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        if (!parts.Contains("artifacts", StringComparer.Ordinal) && !Enumerable.Range(0, Math.Max(0, parts.Length - 1)).Any(i => parts[i] == "docs" && parts[i + 1] == "validation")) throw new ArgumentException("Output must be under artifacts/ or docs/validation/.");
        if (File.Exists(path) || Directory.Exists(path) || IsLink(path)) throw new IOException("Fresh output required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p)) if (IsLink(p) || File.Exists(p)) throw new IOException("Aliased/non-directory parent refused.");
    }
    private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static object Snapshot(string path) { var f = new FileInfo(Path.GetFullPath(path)); return new { f.FullName, f.Length, f.LastWriteTimeUtc, f.CreationTimeUtc, f.LinkTarget }; }
    private static object AssemblyIdentity(Assembly assembly) { using var file = File.OpenRead(assembly.Location); return new { path = assembly.Location, sha256 = Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant() }; }
    private static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
}
