using System.Net;
using System.Security.Cryptography;
using System.Text.Json;
using InferenceWeb.Tests;
using TensorSharp;
using TensorSharp.Distributed;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length == 2 && args[0] is "write" or "write-q4xl")
{
    string directory = Path.GetFullPath(args[1]);
    if (Directory.Exists(directory)) throw new ArgumentException("Fixture directory must be fresh.");
    Directory.CreateDirectory(directory);
    string model = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(directory, "qwen4exp-fixture.gguf"),
        hiddenSize: args[0] == "write-q4xl" ? 512 : 256, q4kxlExperts: args[0] == "write-q4xl");
    int[] tokens = Enumerable.Range(0, 84).Select(i => i % 19 == 0 ? 255 : (i * 17 + 3) % 240).ToArray();
    File.WriteAllText(Path.Combine(directory, "tokens.json"), JsonSerializer.Serialize(tokens));
    File.WriteAllText(Path.Combine(directory, "fixture.json"), JsonSerializer.Serialize(new
    {
        synthetic = true, architecture = "qwen4exp", layers = 8, tokens = tokens.Length,
        sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(model)))
    }));
    Console.WriteLine(model);
    return 0;
}
if (args.Length != 5 || args[0] is not ("cpu-node" or "cuda-tp-node" or "cuda-layer-node"))
{
    Console.Error.WriteLine("write|write-q4xl FRESH_DIRECTORY | cpu-node FIXTURE_DIRECTORY FRESH_OUTPUT NODE_ID ENDPOINTS (NODE_ID=-1 and ENDPOINTS=- for unsplit control)");
    return 2;
}
// Explicit synthetic-only CPU harness. Production capability gates continue to
// admit only qualified CUDA backends; the public model constructor permits this
// direct native-CPU adapter qualification without advertising CPU deployment.
string fixture = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
if (!JsonDocument.Parse(File.ReadAllText(Path.Combine(fixture, "fixture.json"))).RootElement.GetProperty("synthetic").GetBoolean())
    throw new ArgumentException("CPU probe requires the synthetic fixture marker.");
if (Directory.Exists(output)) throw new ArgumentException("Output directory must be fresh.");
Directory.CreateDirectory(output);
Environment.SetEnvironmentVariable("MAX_CONTEXT", "512");
Environment.SetEnvironmentVariable("TS_GGML_THREADS", "2");
KvCacheDtypeConfig.Set(KvCacheDtype.F16);
MoeCpuOffloadConfig.SetLayers(0);
int node = int.Parse(args[3]);
bool cuda = args[0] != "cpu-node", tensorParallel = args[0] == "cuda-tp-node";
ITensorParallelGroup group = null;
if (node >= 0)
{
    IPEndPoint[] endpoints = args[4].Split(',').Select(IPEndPoint.Parse).ToArray();
    var context = new GgmlContext([0], cuda ? GgmlBackendType.Cuda : GgmlBackendType.Cpu, enableCollectives: tensorParallel);
    group = tensorParallel
        ? new DistributedTensorParallelGroup(new GgmlTensorParallelGroup(context), node, endpoints)
        : new DistributedLayerSplitGroup(new GgmlLayerSplitDevices(context), node, endpoints);
}
using var modelInstance = ModelBase.Create(Path.Combine(fixture, "qwen4exp-fixture.gguf"), cuda ? BackendType.GgmlCuda : BackendType.GgmlCpu, tpGroup: group);
group?.Barrier();
if (modelInstance.IsDistributedWorker)
{
    modelInstance.RunDistributedWorkerLoop();
    File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new { node, worker_returned = true }));
    return 0;
}
modelInstance.BeginDistributedDriver();
int[] all = JsonSerializer.Deserialize<int[]>(File.ReadAllText(Path.Combine(fixture, "tokens.json")));
int[] prefix = all[..80];
var vectors = new List<float[]>();
void Forward(int[] tokens, bool refill = false)
{
    float[] logits = refill ? modelInstance.ForwardRefill(tokens) : modelInstance.Forward(tokens);
    if (logits.Length != 256 || logits.Any(value => !float.IsFinite(value))) throw new InvalidOperationException("Invalid logits.");
    vectors.Add((float[])logits.Clone());
}
// Grow a live cache after the first chunk when TS_KV_INITIAL_TOKENS=16.
Forward(prefix[..13]);
Forward(prefix[13..]);
Forward([all[80]]);
modelInstance.ResetKVCache();
Forward(prefix);
foreach (int token in all[80..]) Forward([token]);
modelInstance.ResetKVCache();
foreach (int[] chunk in prefix.Chunk(13)) Forward(chunk);
foreach (int token in all[80..]) Forward([token]);
modelInstance.ResetKVCache();
Forward(prefix, refill: true);
Forward([all[80]]);
modelInstance.ResetKVCache();
Forward(prefix[..1]);
Forward(prefix[1..]);
File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(new
{
    node, vectors, finite = true, resets = 4, prefix_tokens = prefix.Length, explicit_chunk = 13,
    coverage = "Synthetic separate-process native execution over loopback TCP on one host; CUDA modes share the same physical GPU. Not production topology, model quality or throughput."
}));
return 0;
