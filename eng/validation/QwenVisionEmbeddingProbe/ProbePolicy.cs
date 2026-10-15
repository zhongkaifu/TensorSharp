using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;

namespace QwenVisionEmbeddingProbe;

internal sealed record MessageInput(string Role, string Content, string[]? ImagePaths = null);
internal sealed record VisionInput(string Schema, List<MessageInput> Messages, int[] ExpectedPromptTokens, string ImageSha256);
internal sealed record FileStamp(string Path, long Bytes, long LastWriteTimeUtcTicks);

internal static class ProbePolicy
{
    internal static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true, PropertyNameCaseInsensitive = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow
    };
    internal static Dictionary<string, string> Parse(string[] args)
    {
        Dictionary<string, string> values = new(StringComparer.Ordinal);
        string[] required = ["model", "mmproj", "image", "input", "output", "merger", "native-sha256"];
        for (int i = 0; i < args.Length; i += 2)
            if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal)
                || !values.TryAdd(args[i][2..], args[i + 1]))
                throw new ArgumentException("Require unique --name value pairs.");
        if (values.Count != required.Length || required.Except(values.Keys).Any())
            throw new ArgumentException("Require exactly model, mmproj, image, input, output, merger, native-sha256.");
        if (values["merger"] is not ("legacy" or "reference") || !IsSha(values["native-sha256"]))
            throw new ArgumentException("Require merger legacy/reference and explicit SHA256 native identity.");
        foreach (string name in required.Except(["merger", "native-sha256"]))
        {
            if (!Path.IsPathFullyQualified(values[name])) throw new ArgumentException("Require absolute " + name + " path.");
            values[name] = Path.GetFullPath(values[name]);
        }
        return values;
    }
    internal static bool IsSha(string? value) => value is not null
        && Regex.IsMatch(value, "\\A[0-9a-fA-F]{64}\\z", RegexOptions.CultureInvariant);
    internal static bool SamePath(string a, string b) => string.Equals(Path.GetFullPath(a), Path.GetFullPath(b),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    internal static bool IsOverride(string key)
    {
        // CUDA installers publish SDK discovery locations. These do not select
        // devices or change execution. Keep behavior controls (visibility,
        // ordering, module loading, launch blocking, cache policy, etc.) refused.
        bool installedCudaLocation = key.Equals("CUDA_PATH", StringComparison.OrdinalIgnoreCase)
            || key.Equals("CUDA_HOME", StringComparison.OrdinalIgnoreCase)
            || Regex.IsMatch(key, @"\ACUDA_PATH_V[0-9]+_[0-9]+\z", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        return key.StartsWith("TS_", StringComparison.OrdinalIgnoreCase)
            || key.StartsWith("TENSORSHARP_", StringComparison.OrdinalIgnoreCase)
            || key.StartsWith("GGML_", StringComparison.OrdinalIgnoreCase)
            || key.StartsWith("LLAMA_ARG_", StringComparison.OrdinalIgnoreCase)
            || key.StartsWith("CUDA_", StringComparison.OrdinalIgnoreCase) && !installedCudaLocation
            || key is "OMP_NUM_THREADS" or "OMP_THREAD_LIMIT" or "OMP_DYNAMIC" or "MAX_CONTEXT" or "KV_CACHE_DTYPE";
    }
    internal static void NoReparseAncestors(string path)
    {
        FileSystemInfo? item = File.Exists(path) ? new FileInfo(path) : new DirectoryInfo(path);
        for (; item is not null; item = item is FileInfo f ? f.Directory : ((DirectoryInfo)item).Parent)
            if (item.Exists && (item.Attributes & FileAttributes.ReparsePoint) != 0)
                throw new InvalidDataException("Reparse/symlink paths are not accepted: " + item.FullName);
    }
    internal static FileStamp Stamp(string path)
    {
        NoReparseAncestors(path);
        var file = new FileInfo(path);
        if (!file.Exists || file.Length <= 0) throw new InvalidDataException("Missing/empty required file: " + path);
        return new(Path.GetFullPath(path), file.Length, file.LastWriteTimeUtc.Ticks);
    }
    internal static string[] Shards(string first)
    {
        Match split = Regex.Match(first, @"-00001-of-([0-9]{5})\.gguf\z", RegexOptions.CultureInvariant);
        if (!split.Success)
        {
            if (Regex.IsMatch(first, @"-[0-9]{5}-of-[0-9]{5}\.gguf\z", RegexOptions.CultureInvariant))
                throw new ArgumentException("Supply the first model shard.");
            return [first];
        }
        int count = int.Parse(split.Groups[1].Value, CultureInfo.InvariantCulture);
        if (count is < 1 or > 100) throw new ArgumentException("Unsupported split count.");
        return Enumerable.Range(1, count).Select(i => first[..split.Index] + $"-{i:D5}-of-{count:D5}.gguf").ToArray();
    }
    internal static void ValidateInput(VisionInput input)
    {
        if (input.Schema != "qwen-vision-embedding-input.v1" || !IsSha(input.ImageSha256)
            || input.Messages is null || input.ExpectedPromptTokens is null
            || input.Messages.Count is not (1 or 3)
            || input.ExpectedPromptTokens.Length != (input.Messages.Count == 1 ? 318 : 354))
            throw new InvalidDataException("Require the repaired 318/354-token original image history and input schema v1.");
        if (input.ExpectedPromptTokens.Any(x => x < 0) || input.ExpectedPromptTokens.Count(x => x == 248056) != 300)
            throw new InvalidDataException("Require all original 300 expanded image placeholders and nonnegative IDs.");
        for (int i = 0; i < input.Messages.Count; i++)
        {
            MessageInput? message = input.Messages[i];
            if (message is null || message.Role != (i == 1 ? "assistant" : "user")
                || string.IsNullOrWhiteSpace(message.Content) || message.Content.Length > 16384)
                throw new InvalidDataException("Require canonical user / optional assistant / user text history.");
            if (i == 0)
            {
                if (message.ImagePaths is not { Length: 1 }
                    || !string.Equals(message.ImagePaths[0], "sha256:" + input.ImageSha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("First user image path must be the explicit sha256 content identity.");
            }
            else if (message.ImagePaths is { Length: > 0 }) throw new InvalidDataException("Only the first user message may carry the one image.");
        }
    }
    internal static List<ChatMessage> Messages(VisionInput input, string image) => input.Messages.Select((m, i) =>
        new ChatMessage { Role = m.Role, Content = m.Content, ImagePaths = i == 0 ? [image] : null }).ToList();
    internal static void ValidateLayout(long[] shape, long count, long offset, long storageBytes, bool contiguous)
    {
        if (shape.Length != 2 || shape.Any(x => x <= 0) || shape[0] != 300 || !contiguous || count <= 0
            || checked(shape[0] * shape[1]) != count || count > int.MaxValue / sizeof(float)
            || offset < 0 || checked(offset * sizeof(float) + count * sizeof(float)) > storageBytes)
            throw new InvalidDataException("Require one contiguous bounded [300, hidden] projected F32 tensor.");
    }
    internal static void ValidatePositions(int[] positions, int promptLength)
    {
        if (positions.Length != checked(3 * promptLength) || positions.Any(x => x < 0))
            throw new InvalidDataException("Require complete nonnegative absolute T/H/W positions for every expanded token.");
    }
    internal static int SelfTest()
    {
        int checks = 0;
        void Check(bool value) { if (!value) throw new InvalidOperationException("Self-test failed at " + checks); checks++; }
        void Refuses(Action action)
        {
            try { action(); } catch (ArgumentException) { checks++; return; }
            catch (InvalidDataException) { checks++; return; }
            catch (JsonException) { checks++; return; }
            throw new InvalidOperationException("Self-test refusal missing at " + checks);
        }
        string sha = new('a', 64);
        var original = new VisionInput("qwen-vision-embedding-input.v1", [new("user", "请详细描述这幅图", ["sha256:" + sha])],
            Enumerable.Repeat(248056, 300).Concat(Enumerable.Repeat(1, 18)).ToArray(), sha);
        ValidateInput(original); checks++;
        ValidateInput(original with { Messages = [original.Messages[0], new("assistant", "图像中央上方显示黑色数字“4821”，下方是一个醒目的红色矩形块。"), new("user", "请继续")],
            ExpectedPromptTokens = original.ExpectedPromptTokens.Concat(Enumerable.Repeat(2, 36)).ToArray() }); checks++;
        Refuses(() => ValidateInput(original with { Schema = "old" }));
        Refuses(() => ValidateInput(original with { ExpectedPromptTokens = original.ExpectedPromptTokens[..317] }));
        Refuses(() => ValidateInput(original with { ExpectedPromptTokens = Enumerable.Repeat(1, 318).ToArray() }));
        Refuses(() => ValidateInput(original with { Messages = [new("system", "x", ["sha256:" + sha])] }));
        Refuses(() => ValidateInput(original with { Messages = [new("user", "x", ["foreign.png"])] }));
        Refuses(() => ValidateInput(original with { ImageSha256 = "x" }));
        Refuses(() => JsonSerializer.Deserialize<VisionInput>(JsonSerializer.Serialize(original).TrimEnd('}') + ",\"TeacherTokens\":[1]}", Json));
        ValidateLayout([300, 4096], 300L * 4096, 0, 300L * 4096 * 4, true); checks++;
        Refuses(() => ValidateLayout([299, 4096], 299L * 4096, 0, 299L * 4096 * 4, true));
        Refuses(() => ValidateLayout([300, 4096], 300L * 4096, 0, 300L * 4096 * 4, false));
        Refuses(() => ValidateLayout([300, 4096], 300L * 4096, 1, 300L * 4096 * 4, true));
        ValidatePositions(new int[3 * 318], 318); checks++;
        Refuses(() => ValidatePositions(new int[3 * 318 - 1], 318));
        Refuses(() => ValidatePositions([-1, 0, 0], 1));
        Check(IsSha(sha) && !IsSha(sha + "a") && !IsSha("z" + sha[1..]));
        Check(IsOverride("TS_X") && IsOverride("cuda_visible_devices") && !IsOverride("PATH"));
        Check(!IsOverride("CUDA_PATH") && !IsOverride("CUDA_HOME"));
        Check(!IsOverride("CUDA_PATH_V12_1") && !IsOverride("cuda_path_v13_0"));
        Check(IsOverride("CUDA_VISIBLE_DEVICES") && IsOverride("CUDA_DEVICE_ORDER"));
        Check(IsOverride("CUDA_MODULE_LOADING") && IsOverride("CUDA_LAUNCH_BLOCKING"));
        Check(IsOverride("CUDA_CACHE_PATH") && IsOverride("CUDA_CACHE_DISABLE"));
        Check(IsOverride("CUDA_PATH_V12_1_FORCE") && IsOverride("CUDA_PATH_V_INVALID"));
        Check(Shards(Path.GetFullPath("model-00001-of-00003.gguf")).Length == 3);
        Refuses(() => Shards(Path.GetFullPath("model-00002-of-00003.gguf")));
        Refuses(() => Parse(["--model", "relative.gguf"]));
        Refuses(() => Parse(["--model", "one", "--model", "two"]));
        Refuses(() => Parse(["--unknown", "one"]));
        string[] names = ["model", "mmproj", "image", "input", "output"];
        string[] valid = names.SelectMany(name => new[] { "--" + name, Path.GetFullPath(name) })
            .Concat(["--merger", "legacy", "--native-sha256", sha]).ToArray();
        Check(Parse(valid).Count == 7);
        string[] badMerger = (string[])valid.Clone(); badMerger[^3] = "auto";
        Refuses(() => Parse(badMerger));
        int native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Count(x => Path.GetFileName(x.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase));
        Check(native == 0);
        Console.WriteLine(JsonSerializer.Serialize(new { Schema = "qwen-vision-embedding-selftest.v1", Passed = checks, NativeModules = native, ModelCalls = 0 }));
        return 0;
    }
}
