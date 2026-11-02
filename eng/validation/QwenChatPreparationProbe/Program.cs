// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// CPU-only prompt preparation microprobe; it never loads weights or starts a GPU backend.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;

string Required(string name)
{
    int index = Array.IndexOf(args, name);
    if (index < 0 || index + 1 == args.Length) throw new ArgumentException($"Missing {name}");
    return args[index + 1];
}
int Number(string name, int fallback)
{
    int index = Array.IndexOf(args, name);
    return index < 0 ? fallback : int.Parse(args[index + 1]);
}
string templatePath = Required("--template");
string tokenizerPath = Required("--tokenizer");
int repetitions = Number("--repeats", 15);
int lines = Number("--lines", 700);
if (repetitions < 1 || lines < 1) throw new ArgumentException("Positive --repeats and --lines are required.");
string template = File.ReadAllText(templatePath);
long loadStarted = Stopwatch.GetTimestamp();
var tokenizer = LoadTokenizer(tokenizerPath);
double tokenizerLoadMs = Stopwatch.GetElapsedTime(loadStarted).TotalMilliseconds;
var system = new StringBuilder();
for (int i = 0; i < lines; i++)
    system.AppendLine($"Policy {i:D4}: Preserve exact tool parameters, task facts, image coordinates, permissions, and conversation boundaries.");
List<ToolFunction> tools = Enumerable.Range(0, 16).Select(i => new ToolFunction
{
    Name = $"inspect_resource_{i}",
    Description = $"Inspect resource {i} and return its current state. Preserve all values exactly; report failures explicitly.",
    Parameters = new()
    {
        ["path"] = new() { Type = "string", Description = "Workspace-relative resource path, including Unicode file names." },
        ["mode"] = new() { Type = "string", Description = "Read mode.", Enum = ["summary", "complete"] },
    },
    Required = ["path"],
}).ToList();
List<ChatMessage> history =
[
    new() { Role = "system", Content = system.ToString() },
    new() { Role = "user", Content = "Inspect the current screenshot and return the exact requested coordinate as JSON. Do not invent unseen details." },
];
string Render() => ChatTemplate.RenderFromGgufTemplate(template, history, true, "qwen35", tools, false);
long coldStarted = Stopwatch.GetTimestamp();
string rendered = Render();
double coldRenderMs = Stopwatch.GetElapsedTime(coldStarted).TotalMilliseconds;
coldStarted = Stopwatch.GetTimestamp();
List<int> expected = tokenizer.Encode(rendered);
double coldEncodeMs = Stopwatch.GetElapsedTime(coldStarted).TotalMilliseconds;
var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
List<int> RenderTokens() => renderer.RenderToTokens(tokenizer, template, history, "qwen35", true, tools, false);
if (!RenderTokens().SequenceEqual(expected)) throw new InvalidOperationException("KV renderer changed the prompt tokens.");
var measurements = new Dictionary<string, object>
{
    ["render_warm_ast"] = Measure(() => Encoding.UTF8.GetByteCount(Render()), repetitions),
    ["encode_warm_pieces"] = Measure(() => tokenizer.Encode(rendered).Count, repetitions),
    ["render_and_encode"] = Measure(() => RenderTokens().Count, repetitions),
    ["construct_and_render_template"] = Measure(() => new Jinja2Template(template).Render(
        new Dictionary<string, object>
        {
            ["messages"] = new List<object>
            {
                new Dictionary<string, object> { ["role"] = "system", ["content"] = system.ToString() },
                new Dictionary<string, object> { ["role"] = "user", ["content"] = "inspect" },
            },
            ["add_generation_prompt"] = true, ["enable_thinking"] = false,
            ["tools"] = new List<object>(),
        }).Length, repetitions),
};
// This template deliberately calls one macro many times, isolating repeat parsing
// of static parameter declarations from the real Qwen prompt's other work.
const string macroTemplate = "{% macro emit(value, label='x') %}{{ label }}={{ value }};{% endmacro %}{% for item in items %}{{ emit(item) }}{% endfor %}";
var macro = new Jinja2Template(macroTemplate);
var macroContext = new Dictionary<string, object> { ["items"] = Enumerable.Range(0, 1000).Cast<object>().ToList() };
string macroOutput = macro.Render(macroContext);
measurements["macro_1000_calls"] = Measure(() => macro.Render(macroContext).Length, repetitions);
Console.WriteLine(JsonSerializer.Serialize(new
{
    schema_version = 1,
    framework = RuntimeInformation.FrameworkDescription,
    os = RuntimeInformation.OSDescription,
    architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    tiered_compilation = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
    runtime_sha256 = FileHash(typeof(ChatTemplate).Assembly.Location),
    template_sha256 = FileHash(templatePath),
    tokenizer_sha256 = FileHash(tokenizerPath),
    lines,
    prompt_characters = rendered.Length,
    prompt_tokens = expected.Count,
    prompt_sha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(rendered))),
    tokens_sha256 = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(expected)))),
    macro_output_sha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(macroOutput))),
    cold = new { tokenizer_load_ms = tokenizerLoadMs, first_render_ms = coldRenderMs, first_encode_ms = coldEncodeMs },
    measurements,
    limitations = new[] { "Synthetic public text/tools, no images encoded or model generation.",
        "CPU phase timings only; not end-to-end TTFT or a reproduction of the older screenshot.",
        "construct_and_render_template has no tools and is a distinct control workload." },
}, new JsonSerializerOptions { WriteIndented = true }));

static string FileHash(string path)
{
    using var source = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(source));
}

static object Measure(Func<int> action, int repetitions)
{
    for (int i = 0; i < 4; i++) action();
    var samples = new List<double>();
    long allocated = 0;
    int result = 0;
    for (int i = 0; i < repetitions; i++)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        result = action();
        samples.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        allocated += GC.GetAllocatedBytesForCurrentThread() - before;
    }
    samples.Sort();
    return new { median_ms = samples[samples.Count / 2], min_ms = samples[0], max_ms = samples[^1],
        mean_allocated_bytes = allocated / repetitions, result, repetitions, samples_ms = samples };
}

static BpeTokenizer LoadTokenizer(string path)
{
    using var document = JsonDocument.Parse(File.ReadAllBytes(path));
    JsonElement root = document.RootElement;
    JsonElement model = root.GetProperty("model");
    var vocabulary = new Dictionary<int, string>();
    foreach (var item in model.GetProperty("vocab").EnumerateObject())
        vocabulary[item.Value.GetInt32()] = item.Name;
    var special = new HashSet<int>();
    foreach (var item in root.GetProperty("added_tokens").EnumerateArray())
    {
        int id = item.GetProperty("id").GetInt32();
        vocabulary[id] = item.GetProperty("content").GetString()!;
        if (item.GetProperty("special").GetBoolean()) special.Add(id);
    }
    string[] vocab = Enumerable.Range(0, vocabulary.Keys.Max() + 1)
        .Select(id => vocabulary.GetValueOrDefault(id, $"[UNUSED_{id}]")).ToArray();
    int[] types = Enumerable.Range(0, vocab.Length).Select(id => special.Contains(id) ? 3 : 1).ToArray();
    string[] merges = model.GetProperty("merges").EnumerateArray().Select(item => item.ValueKind == JsonValueKind.Array
        ? string.Join(" ", item.EnumerateArray().Select(piece => piece.GetString())) : item.GetString()!).ToArray();
    int eos = Array.IndexOf(vocab, "<|im_end|>");
    return new BpeTokenizer(vocab, types, merges, -1, eos < 0 ? [] : [eos], false, false, "qwen35");
}
