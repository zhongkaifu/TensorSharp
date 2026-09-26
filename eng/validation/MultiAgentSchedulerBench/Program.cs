using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

var values = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i++)
{
    if (args[i] is "--help" or "-h")
    {
        Console.WriteLine("MultiAgentSchedulerBench [--iterations 8] [--warmup 1] [--work-ms 40] [--out artifacts/multi-agent/scheduler.json]");
        return;
    }
    if (args[i] is not ("--iterations" or "--warmup" or "--work-ms" or "--out") || i + 1 == args.Length)
        throw new ArgumentException("Unknown or incomplete option: " + args[i]);
    values[args[i]] = args[++i];
}
int Number(string key, int fallback, int minimum) => !values.TryGetValue(key, out string? text) ? fallback
    : int.TryParse(text, out int value) && value >= minimum ? value : throw new ArgumentException(key + " is out of range.");
int iterations = Number("--iterations", 8, 1), warmup = Number("--warmup", 1, 0), workMs = Number("--work-ms", 40, 0);
string output = Path.GetFullPath(values.GetValueOrDefault("--out") ?? "artifacts/multi-agent/scheduler.json");
Node[] graph =
[
    new("a", []), new("b", []), new("c", []), new("d", []), new("e", []), new("f", []),
    new("ab", ["a", "b"]), new("cd", ["c", "d"]), new("ef", ["e", "f"]),
    new("final", ["ab", "cd", "ef"]),
];
var rows = new List<Measurement>();
Console.WriteLine("Scripted DAG scheduling only. Injected delays are not model latency or reasoning quality.");
for (int iteration = -warmup; iteration < iterations; iteration++)
{
    (string Mode, int Capacity, int Delay)[] modes = [("sequential", 1, workMs), ("parallel", 3, workMs), ("overhead", 3, 0)];
    if (iteration % 2 != 0) Array.Reverse(modes);
    foreach (var mode in modes)
    {
        Measurement row = await Run(iteration + 1, mode.Mode, mode.Capacity, mode.Delay);
        Console.WriteLine($"{(iteration < 0 ? "warmup" : "sample")} {iteration + 1,2} {row.Mode,-10} {row.ElapsedMilliseconds,8:F2} ms; peak={row.PeakGenerators}; completed={row.Completed}/{graph.Length}; dependency_errors={row.DependencyErrors}; evidence_errors={row.EvidenceErrors}");
        if (iteration >= 0) rows.Add(row);
    }
}
var summaries = rows.GroupBy(row => row.Mode).Select(group => new
{
    mode = group.Key, samples = group.Count(),
    wall_ms_p50 = Percentile(group.Select(row => row.ElapsedMilliseconds), .50),
    wall_ms_p95 = Percentile(group.Select(row => row.ElapsedMilliseconds), .95),
    peak_generators_min = group.Min(row => row.PeakGenerators),
    peak_generators_max = group.Max(row => row.PeakGenerators),
    dependency_errors = group.Sum(row => row.DependencyErrors), evidence_errors = group.Sum(row => row.EvidenceErrors),
}).ToArray();
double speedup = summaries.Single(row => row.mode == "sequential").wall_ms_p50 / summaries.Single(row => row.mode == "parallel").wall_ms_p50;
var report = new
{
    schema_version = 1, measured_at_utc = DateTimeOffset.UtcNow, mode = "scripted_dependency_scheduler",
    runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
    processor_count = Environment.ProcessorCount,
    harness_assembly_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(Node).Assembly.Location))),
    agent_host_assembly_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(MultiAgentSession).Assembly.Location))),
    iterations, warmup, simulated_work_ms = workMs, graph, p50_speedup = speedup,
    limitations = "Ten fixed tasks with deterministic generators and injected Task.Delay. Tests queue capacity, dependency ordering, report handoff, completion and scheduler overhead; does not measure inference, autonomous decomposition, reasoning quality, file merging or device throughput. Timer resolution and other machine activity affect timing. No native dependencies are modified.",
    summaries, rows,
};
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
await File.WriteAllTextAsync(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Paired p50 scheduling speedup: {speedup:F2}x; report: {output}");
if (rows.Any(row => row.Completed != graph.Length || row.DependencyErrors != 0 || row.EvidenceErrors != 0 || row.PeakGenerators > row.Capacity))
    Environment.ExitCode = 1;

async Task<Measurement> Run(int iteration, string mode, int capacity, int delay)
{
    int active = 0, peak = 0, dependencyErrors = 0, evidenceErrors = 0;
    var finished = new ConcurrentDictionary<string, bool>();
    using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    await using var session = new MultiAgentSession(
        [new() { Role = "system", Content = "Use prerequisite evidence and complete the assigned node." }],
        SkillTools.BuiltIn(), new SkillToolContext([]), id => async (messages, _, ct) =>
        {
            Node node = graph.Single(candidate => id == "/root/" + candidate.Name);
            int count = Interlocked.Increment(ref active);
            int previous;
            do { previous = Volatile.Read(ref peak); }
            while (count > previous && Interlocked.CompareExchange(ref peak, count, previous) != previous);
            try
            {
                string context = string.Join("\n", messages.Select(message => message.Content));
                foreach (string dependency in node.Dependencies)
                {
                    if (!finished.ContainsKey(dependency)) Interlocked.Increment(ref dependencyErrors);
                    if (!context.Contains("EVIDENCE:" + dependency + ":VERIFIED", StringComparison.Ordinal))
                        Interlocked.Increment(ref evidenceErrors);
                }
                if (delay > 0) await Task.Delay(delay, ct);
                finished[node.Name] = true;
                return new(new ParsedOutput { Content = "EVIDENCE:" + node.Name + ":VERIFIED" });
            }
            finally { Interlocked.Decrement(ref active); }
        }, new() { Enabled = true, MaxConcurrentAgents = capacity, MaxAgents = graph.Length }, cancellationToken: timeout.Token);
    Stopwatch timer = Stopwatch.StartNew();
    foreach (Node node in graph)
    {
        SkillToolResult result = await session.ExecuteAsync(new ToolCall
        {
            Id = Guid.NewGuid().ToString("N"), Name = "spawn_agent",
            Arguments = new Dictionary<string, object?>
            {
                ["task_name"] = node.Name, ["task"] = "Complete " + node.Name, ["agent_type"] = "explorer",
                ["depends_on"] = string.Join(",", node.Dependencies.Select(dependency => "/root/" + dependency)),
            },
        }, cancellationToken: timeout.Token);
        if (!result.Ok) throw new InvalidOperationException(result.Content);
    }
    await session.CollectResultsAsync(cancellationToken: timeout.Token);
    timer.Stop();
    int completed = session.GetProgress().Count(agent => agent.Status == "completed");
    return new(iteration, mode, capacity, delay, timer.Elapsed.TotalMilliseconds, peak, completed, dependencyErrors, evidenceErrors);
}

static double Percentile(IEnumerable<double> values, double percentile)
{
    double[] sorted = values.Order().ToArray();
    return sorted[Math.Clamp((int)Math.Ceiling(sorted.Length * percentile) - 1, 0, sorted.Length - 1)];
}
sealed record Node(string Name, string[] Dependencies);
sealed record Measurement(int Iteration, string Mode, int Capacity, int SimulatedWorkMilliseconds,
    double ElapsedMilliseconds, int PeakGenerators, int Completed, int DependencyErrors, int EvidenceErrors);
