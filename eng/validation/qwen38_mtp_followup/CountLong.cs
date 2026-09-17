// Metadata/tokenizer only. Compiled with the existing production test-friend assembly name.
using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Chat;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.RequestParsers;

if (args.Length != 5) throw new ArgumentException("first-shard request.json requestSHA output-dir expectedManagedManifest.json");
static string Sha(byte[] b) => Convert.ToHexStringLower(SHA256.HashData(b));
static string FileSha(string p) { using var f = File.OpenRead(p); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static void Guard()
{
    if (Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => new[] { "ggmlops", "nvcuda", "cudart", "cublas", "cudnn", "mlxops" }
        .Any(s => Path.GetFileName(m.FileName).Contains(s, StringComparison.OrdinalIgnoreCase)))) throw new InvalidOperationException("Native inference/CUDA library unexpectedly mapped");
}
Guard();
string model = args[0], output = args[3];
const string checkpoint = "a4f3b21e77353999829f2f767e9ac21ce9c71d29a74f2cc9eda48c9bf23c8b86";
if (FileSha(model) != checkpoint || FileSha(args[1]) != args[2] || Directory.Exists(output)) throw new InvalidDataException("Input identity/output path rejected");
var pins = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllBytes(args[4]))!;
void CheckManaged() { foreach (var (name, hash) in pins) if (FileSha(Path.Combine(AppContext.BaseDirectory, name)) != hash) throw new InvalidDataException("Managed identity changed: " + name); }
CheckManaged();
using var gguf = GgufFile.OpenWithoutSiblingShards(model);
if (gguf.FilePaths.Count != 1 || gguf.GetString("general.architecture") != "qwen4exp") throw new InvalidDataException("Wrong single-shard metadata");
var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
using var doc = JsonDocument.Parse(File.ReadAllBytes(args[1])); var body = doc.RootElement;
if (body.TryGetProperty("tools", out _) || body.TryGetProperty("response_format", out _) || body.GetProperty("think").GetBoolean()) throw new InvalidDataException("Unexpected preprocessing branch");
var history = ChatMessageParser.ParseOpenAI(body.GetProperty("messages"), null!, architecture: "qwen4exp");
history = StructuredOutputPrompt.Apply(history, null);
history = ChatHistoryPreparer.PrepareHistoryForInference(history, "qwen4exp");
var recording = new Recorder(tokenizer);
var tokens = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(recording, gguf.GetString("tokenizer.chat_template"), history, "qwen4exp", true,
    out var breakpoints, out var boundary, null, false);
if (recording.Calls.Count != 1 || !tokens.SequenceEqual(recording.Calls[0].Tokens)) throw new InvalidDataException("Unexpected prompt splice/multi-encode");
int max = body.GetProperty("max_tokens").GetInt32();
var budget = ChatGenerationPipeline.CompactHistoryForContextBudget(history, tokens.Count, 65536, max, false,
    _ => throw new InvalidOperationException("The single latest user message must not be compacted"));
if (budget.RemovedMessages != 0 || budget.FinalPromptTokens != tokens.Count || tokens.Count > 65535 || tokens.Count + max + 4 > 65536)
    throw new InvalidDataException("New input is not admissible with full output+verify reserve");
Directory.CreateDirectory(output);
byte[] utf8 = new UTF8Encoding(false, true).GetBytes(recording.Calls[0].Text), ids = new byte[tokens.Count * 4];
for (int i = 0; i < tokens.Count; i++) BinaryPrimitives.WriteInt32LittleEndian(ids.AsSpan(i * 4, 4), tokens[i]);
File.WriteAllBytes(Path.Combine(output, "rendered.utf8"), utf8); File.WriteAllBytes(Path.Combine(output, "tokens.i32"), ids);
Guard(); CheckManaged();
if (FileSha(model) != checkpoint || FileSha(args[1]) != args[2]) throw new InvalidDataException("After input identity drift");
File.WriteAllText(Path.Combine(output, "count.json"), JsonSerializer.Serialize(new { status = "metadata-only-count-passed", release_qualified = false,
    checkpoint_first_shard_sha256 = checkpoint, body_sha256 = args[2], managed = pins, native_unmapped_before = true, native_unmapped_after = true,
    prompt_tokens = tokens.Count, requested_max_tokens = max, verify_reserve = 4, maximum_position_exclusive = tokens.Count + max + 4,
    hard_prompt_limit = 65535, removed_messages = budget.RemovedMessages, rendered_utf8_sha256 = Sha(utf8), tokens_i32_sha256 = Sha(ids),
    limitations = "Exact referenced frozen v5 production tokenizer/render. Future v7 application must reproduce the exact bytes/IDs or retain a failed identity gate; no HTTP/model inference executed." }, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"metadata-only count {tokens.Count}; output+verify {tokens.Count + max + 4}; native unmapped");

sealed class Recorder(ITokenizer inner) : ITokenizer
{
    public List<(string Text, List<int> Tokens)> Calls = [];
    public List<int> Encode(string text, bool addSpecial = true) { var ids = inner.Encode(text, addSpecial); Calls.Add((text, ids.ToList())); return ids; }
    public string[] Vocab => inner.Vocab;
    public int BosTokenId => inner.BosTokenId;
    public int[] EosTokenIds => inner.EosTokenIds;
    public int VocabSize => inner.VocabSize;
    public string Decode(List<int> ids) => inner.Decode(ids);
    public void AppendTokenBytes(int id, List<byte> b) => inner.AppendTokenBytes(id, b);
    public bool IsEos(int id) => inner.IsEos(id);
    public int LookupToken(string text) => inner.LookupToken(text);
}
