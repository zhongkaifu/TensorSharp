using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace QwenVisionEmbeddingProbe;

internal static class Program
{
    private static string Hash(string path)
    {
        using var file = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
    }
    private static object Pin(string path) => new { Path = Path.GetFullPath(path), Sha256 = Hash(path) };
    private static object Field(object owner, string name) => owner.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(owner)
        ?? throw new InvalidDataException("Missing prepared-media field: " + name);
    private static T Property<T>(object owner, string name) => owner.GetType().GetProperty(name)?.GetValue(owner) is T result
        ? result : throw new InvalidDataException("Missing/invalid prepared-media property: " + name);

    private static unsafe object ExportPrepared(object injector, string id, int[] expanded, string output)
    {
        lock (Field(injector, "_bucketLock"))
        {
            var buckets = Field(injector, "_visionByRequest") as IDictionary
                ?? throw new InvalidDataException("Prepared vision bucket type changed.");
            var positionBuckets = Field(injector, "_mropePositionsByRequest") as IDictionary
                ?? throw new InvalidDataException("Prepared MRoPE bucket type changed.");
            if (buckets[id] is not IEnumerable spans || positionBuckets[id] is not int[] positions)
                throw new InvalidDataException("Actual request has no prepared image/positions.");
            ProbePolicy.ValidatePositions(positions, expanded.Length);
            List<object> exports = [];
            foreach (object span in spans)
            {
                if (exports.Count != 0) throw new InvalidDataException("Expected exactly one projected image tensor.");
                object entry = Property<object>(span, "CacheEntry");
                var tensor = Property<Tensor>(entry, "Embeddings");
                if (tensor.ElementType != DType.Float32) throw new InvalidDataException("Projected tensor is not F32.");
                long[] shape = tensor.Sizes.ToArray();
                long count = tensor.ElementCount();
                ProbePolicy.ValidateLayout(shape, count, tensor.StorageOffset, tensor.Storage.ByteLength, tensor.IsContiguous());
                int start = Property<int>(span, "PromptTokenStart"), end = Property<int>(span, "PromptTokenEndExclusive");
                if (start < 0 || end > expanded.Length || end - start != shape[0])
                    throw new InvalidDataException("Prepared span does not exactly cover the actual image placeholder rows.");
                int tokenCount = Property<int>(entry, "TokenCount"), mergedHeight = Property<int>(entry, "Extra0"), mergedWidth = Property<int>(entry, "Extra1");
                if (tokenCount != shape[0] || mergedHeight <= 0 || mergedWidth <= 0
                    || checked((long)mergedHeight * mergedWidth) != tokenCount)
                    throw new InvalidDataException("Prepared image token count/grid differs from projected tensor rows.");
                for (int i = start; i < end; i++)
                    if (expanded[i] != 248056) throw new InvalidDataException("Prepared span contains a nonimage token.");
                // The cache owns this tensor throughout this synchronous lock.
                // Pull its actual cached device copy into its existing admitted
                // host storage, then hash/write spans directly: no media-sized
                // float[], byte[], tensor clone or diagnostic payload allocation.
                tensor.Storage.EnsureHostReadable();
                IntPtr basePointer = TensorComputePrimitives.GetStorageBasePointer(tensor);
                if (basePointer == IntPtr.Zero) throw new InvalidDataException("Null tensor backing storage.");
                GgmlBasicOps.SyncHostBuffer(basePointer, tensor.Storage.ByteLength);
                IntPtr pointer = TensorComputePrimitives.GetStoragePointer(tensor);
                var values = new ReadOnlySpan<float>(pointer.ToPointer(), checked((int)count));
                double sum = 0, squares = 0; float minimum = float.PositiveInfinity, maximum = float.NegativeInfinity;
                foreach (float value in values)
                {
                    if (!float.IsFinite(value)) throw new InvalidDataException("Projected tensor contains nonfinite actual values.");
                    sum += value; squares += (double)value * value;
                    minimum = Math.Min(minimum, value); maximum = Math.Max(maximum, value);
                }
                ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(values);
                string sha = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
                string name = "projected-000.f32", filePath = Path.Combine(output, name);
                using (var file = new FileStream(filePath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.SequentialScan))
                {
                    file.Write(bytes); file.Flush(flushToDisk: true);
                }
                if (new FileInfo(filePath).Length != bytes.Length || Hash(filePath) != sha)
                    throw new InvalidDataException("Exported file differs from synchronous actual tensor bytes.");
                exports.Add(new { File = name, Shape = shape, Elements = count, Bytes = bytes.Length, Sha256 = sha,
                    Finite = true, Minimum = minimum, Maximum = maximum, Mean = sum / count, L2 = Math.Sqrt(squares),
                    ElementType = "Float32", Layout = "row-major [image-token, language-hidden]", ByteOrder = "little-endian",
                    PromptTokenStart = start, PromptTokenEndExclusive = end, InsertPosition = Property<int>(span, "InsertPosition"),
                    TokenCount = tokenCount, MergedHeight = mergedHeight, MergedWidth = mergedWidth,
                    ContentId = Property<string>(entry, "ContentId"),
                    ExistingStorageBytes = tensor.Storage.ByteLength, StorageOffsetElements = tensor.StorageOffset,
                    ExportScratch = "Existing admitted host tensor; fixed-size FileStream/hash internals only, no media-sized copy." });
            }
            if (exports.Count != 1) throw new InvalidDataException("Exactly one actual projected image tensor was not found.");
            return new { Embeddings = exports, MropePositions = positions, MropeLayout = "interleaved [T0,H0,W0,T1,H1,W1,...] for all expanded prompt tokens",
                Scope = "Actual prepared projected F32 tensor and absolute T/H/W positions before any language-model forward." };
        }
    }

    private static int Main(string[] args)
    {
        if (args.SequenceEqual(["--self-test"])) return ProbePolicy.SelfTest();
        var options = ProbePolicy.Parse(args);
        string output = options["output"];
        if (File.Exists(output) || Directory.Exists(output)) throw new ArgumentException("Fresh output directory required.");
        ProbePolicy.NoReparseAncestors(output);
        foreach (string name in new[] { "model", "mmproj", "image", "input" }) ProbePolicy.Stamp(options[name]);
        if (new FileInfo(options["input"]).Length > 1 << 20)
            throw new InvalidDataException("Bounded diagnostic input JSON must not exceed 1 MiB.");
        string native = Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "GgmlOps.dll"
            : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so");
        ProbePolicy.Stamp(native);
        var modelFiles = ProbePolicy.Shards(options["model"]).Append(options["mmproj"]).ToArray();
        FileStamp[] modelBefore = modelFiles.Select(ProbePolicy.Stamp).ToArray();
        if (modelFiles.Append(options["input"]).Append(options["image"]).Any(x => ProbePolicy.SamePath(x, output)))
            throw new ArgumentException("Output may not replace an input.");
        Directory.CreateDirectory(output);
        var report = new Dictionary<string, object?>
        {
            ["Schema"] = "qwen-vision-embedding-probe.v1", ["Completed"] = false,
            ["PreparationCompleted"] = false, ["LanguageModelForwards"] = 0, ["TeacherForcing"] = false,
            ["SemanticPassed"] = null, ["StartedUtc"] = DateTime.UtcNow, ["ProcessId"] = Environment.ProcessId,
            ["Arguments"] = options, ["ModelMetadataBefore"] = modelBefore,
            ["Scope"] = "CUDA unified-memory image-preparation diagnostic; no LLM execution, generated-answer quality, throughput or full physical-hard-limit qualification."
        };
        IUnifiedMemoryModelSession? session = null;
        int exit = 1;
        try
        {
            if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("This raw-F32 comparison format requires little endian.");
            foreach (DictionaryEntry variable in Environment.GetEnvironmentVariables())
                if (variable.Key is string key && ProbePolicy.IsOverride(key) && !string.IsNullOrEmpty(variable.Value as string))
                    throw new InvalidOperationException("Inherited model/runtime override is not permitted: " + key);
            if (Hash(native) != options["native-sha256"].ToLowerInvariant()) throw new InvalidDataException("Native binary identity mismatch.");
            report["NativeBefore"] = Pin(native);
            string inputHash = Hash(options["input"]), imageHash = Hash(options["image"]);
            report["InputSha256"] = inputHash; report["ImageSha256"] = imageHash;
            VisionInput input = JsonSerializer.Deserialize<VisionInput>(File.ReadAllText(options["input"]), ProbePolicy.Json)
                ?? throw new InvalidDataException("Missing input JSON.");
            ProbePolicy.ValidateInput(input);
            if (imageHash != input.ImageSha256.ToLowerInvariant()) throw new InvalidDataException("Actual image SHA differs from original input binding.");
            report["Input"] = input;
            var core = new[] { typeof(ModelBase).Assembly, typeof(KVCachePromptRenderer).Assembly, typeof(Tensor).Assembly,
                typeof(TensorSharp.Memory.MemoryBudget).Assembly, typeof(GgmlBasicOps).Assembly }.Distinct().ToArray();
            Dictionary<string, string> coreBefore = core.ToDictionary(a => a.Location, a => Hash(a.Location), StringComparer.Ordinal);
            report["ManagedCoreBefore"] = coreBefore;
            Environment.SetEnvironmentVariable("TS_HOST_MOE_DEVICE_MIN_BATCH", "0");
            Environment.SetEnvironmentVariable("TS_Q4E_PREFILL_COMBINE", "0");
            Environment.SetEnvironmentVariable("TS_Q4E_VISION_MERGER_ERF", options["merger"] == "legacy" ? "0" : "1");
            report["ChildControls"] = new { TS_HOST_MOE_DEVICE_MIN_BATCH = "0", TS_Q4E_PREFILL_COMBINE = "0",
                TS_Q4E_VISION_MERGER_ERF = options["merger"] == "legacy" ? "0" : "1", CpuThreads = 8 };
            MoeCpuOffloadConfig.Reset(); MoeCpuOffloadConfig.SetCpuThreads(8);
            var load = Stopwatch.StartNew();
            session = UnifiedMemoryModelSession.Create(options["model"], options["mmproj"], BackendType.GgmlCuda,
                new UnifiedMemoryModelOptions { ContextTokens = 1024, PrefillChunkTokens = 32, MaximumRunningRequests = 1,
                    PrefixCheckpointCount = 0, RetainedConversationCount = 0, MaximumExpertCacheBytes = 0,
                    ExpertCachePrefillRows = 32, EnableAsyncExpertPrefill = false, EnableExpertCacheCapacityGrowth = false });
            report["LoadMs"] = load.Elapsed.TotalMilliseconds; report["FeatureReport"] = session.FeatureReport;
            var model = session.Model as Qwen4ExpModel ?? throw new InvalidOperationException("Require actual Qwen4Exp model.");
            if (model.Config.Architecture != "qwen4exp" || model.VisionEncoder is null || session.ContextTokens != 1024)
                throw new InvalidOperationException("Model/projector/context diagnostic contract differs.");
            if (input.ExpectedPromptTokens.Any(x => x >= model.Config.VocabSize)) throw new InvalidDataException("Expected prompt ID exceeds loaded vocabulary.");
            var placement = typeof(Qwen4ExpModel).GetField("_expertOnHost", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model) as bool[];
            if (placement is null || placement.Length == 0 || placement.Any(x => !x))
                throw new InvalidOperationException("Diagnostic requires all routed expert layers host-resident.");
            report["HostRoutedLayers"] = placement.Length;
            var cacheBefore = GgmlBasicOps.GetHostMoeExpertCacheUsage();
            if (cacheBefore.AccountedBytes != 0 || cacheBefore.BudgetBytes != 0) throw new InvalidOperationException("Expert cache not disabled.");
            report["ExpertCacheBefore"] = cacheBefore;
            string id = "vision-export-" + Guid.NewGuid().ToString("N");
            List<ChatMessage> messages = ProbePolicy.Messages(input, options["image"]);
            int[] raw = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(model.Tokenizer,
                model.Config.ChatTemplate, messages, model.Config.Architecture, addGenerationPrompt: true, enableThinking: false).ToArray();
            report["RenderedUnexpandedIds"] = raw;
            var prepare = Stopwatch.StartNew();
            int[] expanded = model.MultimodalInjector.ProcessPromptTokens(messages, raw.ToList(), id).ToArray();
            report["PreparationMs"] = prepare.Elapsed.TotalMilliseconds; report["ExpandedIds"] = expanded;
            if (!expanded.SequenceEqual(input.ExpectedPromptTokens)) throw new InvalidDataException("Rendered/expanded IDs differ from the actual original repaired HTTP history.");
            report["PreparedMedia"] = ExportPrepared(model.MultimodalInjector, id, expanded, output);
            report["BudgetBeforeDispose"] = session.Budget.Snapshot(); report["BudgetHighWater"] = session.Budget.HighWatermarks();
            var modules = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
                .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase)).Select(m => m.FileName).ToArray();
            report["LoadedNativeModules"] = modules;
            if (modules.Length != 1 || !ProbePolicy.SamePath(modules[0], native) || Hash(modules[0]) != options["native-sha256"].ToLowerInvariant())
                throw new InvalidDataException("Loaded native module differs from the explicit pinned private runtime.");
            if (Hash(options["input"]) != inputHash || Hash(options["image"]) != imageHash)
                throw new InvalidDataException("Input/image changed during preparation.");
            var coreAfter = core.ToDictionary(a => a.Location, a => Hash(a.Location), StringComparer.Ordinal);
            report["ManagedCoreAfter"] = coreAfter;
            if (coreBefore.Any(p => !coreAfter.TryGetValue(p.Key, out string? value) || value != p.Value))
                throw new InvalidDataException("Managed runtime changed during preparation.");
            report["PreparationCompleted"] = true; exit = 0;
        }
        catch (Exception error) { report["Error"] = error.ToString(); }
        finally
        {
            try { session?.Dispose(); report["SessionDisposed"] = session is not null; }
            catch (Exception error) { report["CleanupError"] = error.ToString(); exit = 1; }
            if (session is not null)
            {
                var final = session.Budget.Snapshot(); report["FinalBudget"] = final;
                if (final.Any(p => p.Reserved != 0 || p.Committed != 0)) { report["LedgerZero"] = false; exit = 1; }
                else report["LedgerZero"] = true;
            }
            try
            {
                FileStamp[] after = modelFiles.Select(ProbePolicy.Stamp).ToArray();
                report["ModelMetadataAfter"] = after; report["ModelMetadataUnchanged"] = modelBefore.SequenceEqual(after);
                report["NativeAfter"] = Pin(native);
                if (!modelBefore.SequenceEqual(after) || Hash(native) != options["native-sha256"].ToLowerInvariant()) exit = 1;
            }
            catch (Exception error) { report["BookendError"] = error.ToString(); exit = 1; }
            report["FinishedUtc"] = DateTime.UtcNow; report["Completed"] = exit == 0;
            report["ExitCode"] = exit;
            File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, ProbePolicy.Json) + "\n");
        }
        return exit;
    }
}
