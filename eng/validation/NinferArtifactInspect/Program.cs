using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp.Runtime;

internal static class Program
{
    private const string Usage = """
        Read-only NInfer v2 typed inventory; no factory, inference, CUDA or full-weight hash.
        Usage: dotnet NinferArtifactInspect.dll --artifact FILE --output NEW_IGNORED_JSON
          [--expect-directory-sha SHA256] [--known-sha256 SUPPLIED_NOT_VERIFIED_SHA256]
          [--resource EXACT_NAME]... [--max-resource-bytes N]
          [--max-json-bytes N] [--max-objects N]
        --output must be fresh under artifacts/ or docs/validation/, without symlink parents.
        Resources are read only on explicit request, bounded, and reported as SHA/byte counts.
        Resource names are lookup keys, never filesystem extraction paths. No resource content is emitted.
        Exit 0: v2 metadata/divisor inspection completed; 1: inspection failed; 2: argument/output refusal.
        A completed inspection does not establish native import compatibility or model quality.
        """;
    private sealed class Options
    {
        internal string Artifact = "", Output = "";
        internal string? ExpectedDirectorySha, KnownSha;
        internal int MaxJsonBytes = 16 << 20, MaxObjects = 100_000, MaxResourceBytes = 16 << 20;
        internal readonly List<string> Resources = new();
    }
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() }
    };
    private static int Main(string[] args)
    {
        if (args.Length == 1 && args[0] is "--help" or "-h") { Console.WriteLine(Usage); return 0; }
        Options options;
        try
        {
            options = Parse(args);
            // Validate and open the same canonical path; never reopen an alias/../ raw spelling.
            options.Output = Path.GetFullPath(options.Output);
            CheckOutput(options.Output);
        }
        catch (Exception error) when (error is ArgumentException or IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine(error.Message); Console.Error.WriteLine(Usage); return 2;
        }
        var report = new Dictionary<string, object?>
        {
            ["schema"] = "tensorsharp-ninfer-v2-typed-inspection/1", ["complete"] = false,
            ["artifact"] = Path.GetFullPath(options.Artifact), ["createdUtc"] = DateTimeOffset.UtcNow,
            ["knownFullArtifactSha256"] = options.KnownSha, ["knownShaStatus"] = "SUPPLIED_NOT_VERIFIED_BY_THIS_TOOL",
            ["fullArtifactSha256Recomputed"] = false, ["factoryBinding"] = "NOT_IMPLEMENTED",
            ["inferenceSupport"] = "NOT_IMPLEMENTED", ["importCompatibility"] = "NOT_ESTABLISHED",
            ["modelQuality"] = "NOT_RUN", ["gpu"] = "NOT_RUN",
            ["payloadContentScope"] = "INITIALIZATION_READS_DIRECTORY_AND_NVFP4_DIVISORS_ONLY; NOT_CODE_OR_SCALE_VALUES",
            ["resourceContentOutput"] = "NEVER; EXPLICIT_SELECTION_SHA_AND_LENGTH_ONLY",
            ["geometrySpecificationRevision"] = NinferArtifactReader.GeometrySpecificationRevision,
            ["geometrySpecificationSourceSha256"] = NinferArtifactReader.GeometrySpecificationSourceSha256
        };
        int exitCode = 1;
        try
        {
            report["execution"] = new
            {
                toolAssembly = AssemblyIdentity(Assembly.GetExecutingAssembly()),
                runtimeAssembly = AssemblyIdentity(typeof(NinferArtifactReader).Assembly)
            };
            object before = FileSnapshot(options.Artifact);
            report["fileStatBefore"] = before;
            using var reader = new NinferArtifactReader(options.Artifact, options.MaxJsonBytes, options.MaxObjects);
            report["directorySha256"] = reader.DirectorySha256;
            report["prefixSha256"] = reader.PrefixSha256;
            report["expectedDirectorySha256"] = options.ExpectedDirectorySha;
            if (options.ExpectedDirectorySha != null && options.ExpectedDirectorySha != reader.DirectorySha256)
                throw new InvalidDataException("Expected directory SHA256 does not match the inspected directory.");
            report["inventory"] = new
            {
                format = "NINFER_V2", reader.Identity, reader.FileBytes, reader.MetadataBytes, reader.PayloadStart,
                objectCount = reader.Objects.Count, tensorCount = reader.Tensors.Count, resourceCount = reader.Resources.Count,
                nvfp4DivisorCount = reader.Tensors.Count(t => t.WeightDivisorBits.HasValue),
                formatCounts = reader.Tensors.GroupBy(t => t.Format.ToString()).ToDictionary(g => g.Key, g => g.Count()),
                reader.TrailingUnclaimedPayloadBytes,
                tensors = reader.Tensors.Select(t => new
                {
                    t.Name, t.Format, t.Layout, t.Shape, t.PayloadOffset, t.AbsoluteOffset, t.ByteLength, t.Geometry,
                    t.WeightDivisor, weightDivisorRawBits = t.WeightDivisorBits?.ToString("x8", CultureInfo.InvariantCulture)
                }).ToArray(),
                resources = reader.Resources.Select(r => new { r.Name, r.Encoding, r.PayloadOffset, r.AbsoluteOffset, r.ByteLength }).ToArray()
            };
            var selectedResources = new List<object>();
            foreach (string name in options.Resources)
            {
                if (reader.GetObject(name) is not NinferResourceDescriptor resource)
                    throw new InvalidOperationException("Selected name is a tensor, not a raw resource: " + name);
                byte[] bytes = reader.ReadResource(resource, options.MaxResourceBytes);
                selectedResources.Add(new { name, bytes = bytes.Length, sha256 = Hash(bytes), content = "NOT_EMITTED" });
            }
            report["explicitResourceReads"] = selectedResources;
            report["readStatistics"] = new
            {
                reader.InitializationBytesRead, reader.InitializationReadOperations,
                reader.TotalBytesRead, reader.ReadOperations,
                note = "Actual positional read calls; partial reads can require retries. Assembly hashes are separate non-artifact I/O."
            };
            object after = FileSnapshot(options.Artifact);
            report["fileStatAfter"] = after;
            if (JsonSerializer.Serialize(before, JsonOptions) != JsonSerializer.Serialize(after, JsonOptions))
                throw new IOException("Artifact path/stat changed during inspection.");
            report["fileStatUnchanged"] = true;
            report["fileIdentityScope"] = "Length/mtime/creationTime/link-target snapshot; not inode identity or whole-payload checksum. External full SHA/stat guard still required.";
            report["complete"] = true;
            report["status"] = "V2_METADATA_GEOMETRY_AND_DIVISORS_VALIDATED";
            exitCode = 0;
        }
        catch (Exception error)
        {
            report["complete"] = false;
            report["status"] = error is NotSupportedException ? "UNSUPPORTED" : "INSPECTION_ERROR";
            report["error"] = new { type = error.GetType().Name, error.Message };
        }
        try
        {
            CheckOutput(options.Output);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.Output))!);
            CheckOutput(options.Output);
            using var file = new FileStream(options.Output, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            JsonSerializer.Serialize(file, report, JsonOptions);
            file.WriteByte((byte)'\n');
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("Output refused: " + error.Message); return 2;
        }
        Console.WriteLine(JsonSerializer.Serialize(new { output = Path.GetFullPath(options.Output), complete = exitCode == 0, exitCode }, JsonOptions));
        return exitCode;
    }
    private static Options Parse(string[] args)
    {
        var options = new Options(); var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string key = args[i];
            if (key is not ("--artifact" or "--output" or "--known-sha256" or "--expect-directory-sha" or "--resource" or "--max-resource-bytes" or "--max-json-bytes" or "--max-objects"))
                throw new ArgumentException("Unknown option: " + key);
            if (key != "--resource" && !seen.Add(key)) throw new ArgumentException("Duplicate option: " + key);
            if (++i == args.Length || string.IsNullOrWhiteSpace(args[i])) throw new ArgumentException("Missing value: " + key);
            string value = args[i];
            switch (key)
            {
                case "--artifact": options.Artifact = value; break;
                case "--output": options.Output = value; break;
                case "--known-sha256": options.KnownSha = ShaArgument(value); break;
                case "--expect-directory-sha": options.ExpectedDirectorySha = ShaArgument(value); break;
                case "--resource":
                    if (options.Resources.Count >= 32 || options.Resources.Contains(value, StringComparer.Ordinal)) throw new ArgumentException("Duplicate/excessive resource selection.");
                    options.Resources.Add(value); break;
                case "--max-resource-bytes": options.MaxResourceBytes = PositiveInt(value, key); break;
                case "--max-json-bytes": options.MaxJsonBytes = PositiveInt(value, key); break;
                case "--max-objects": options.MaxObjects = PositiveInt(value, key); break;
            }
        }
        if (options.Artifact.Length == 0 || options.Output.Length == 0) throw new ArgumentException("--artifact and --output are required.");
        return options;
    }
    private static int PositiveInt(string value, string key) =>
        int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int n) && n > 0 ? n : throw new ArgumentException("Positive integer required: " + key);
    private static string ShaArgument(string value) => value.Length == 64 && value.All(Uri.IsHexDigit)
        ? value.ToLowerInvariant() : throw new ArgumentException("SHA256 must contain exactly 64 hexadecimal characters.");
    private static void CheckOutput(string path)
    {
        string full = Path.GetFullPath(path);
        string[] parts = full.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        bool ignored = parts.Contains("artifacts", StringComparer.Ordinal) || Enumerable.Range(0, Math.Max(0, parts.Length - 1))
            .Any(i => parts[i] == "docs" && parts[i + 1] == "validation");
        if (!ignored) throw new ArgumentException("Output must be under artifacts/ or docs/validation/.");
        if (File.Exists(full) || Directory.Exists(full) || IsLink(full)) throw new IOException("Fresh output required.");
        for (string? p = Path.GetDirectoryName(full); p != null; p = Path.GetDirectoryName(p))
        {
            if (IsLink(p)) throw new IOException("Symlinked output parent refused.");
            if (File.Exists(p)) throw new IOException("Output parent is not a directory.");
        }
    }
    private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;
    private static object FileSnapshot(string path)
    {
        var f = new FileInfo(Path.GetFullPath(path));
        return new { f.FullName, f.Length, mtimeUtc = f.LastWriteTimeUtc, creationTimeUtc = f.CreationTimeUtc, f.LinkTarget };
    }
    private static object AssemblyIdentity(Assembly assembly)
    {
        string path = assembly.Location;
        using var file = File.OpenRead(path);
        return new { path, sha256 = Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant() };
    }
    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}
