// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Memory;
using TensorSharp.Runtime.Scheduling;

string Option(string name, string fallback) => Array.IndexOf(args, name) is int i && i >= 0 ? args[i + 1] : fallback;
double durationSeconds = double.Parse(Option("--seconds", "7200"), System.Globalization.CultureInfo.InvariantCulture);
if (durationSeconds < 1 || durationSeconds > 86400) throw new ArgumentOutOfRangeException("--seconds");
bool admissionHook = bool.Parse(Option("--admission-hook", "false"));
string directory = Path.GetFullPath(Option("--output", "artifacts/default-readiness-20261010/cpu-soak"));
Directory.CreateDirectory(directory);
Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
const int lanes = 4, perLane = 8, bytes = 64 * 1024;
var budget = new MemoryBudget([new("host", 2 * bytes)]);
var models = Enumerable.Range(0, lanes).Select(_ => new SoakModel()).ToArray();
long[] admissionCalls = new long[lanes];
var engines = models.Select((model, lane) => new InferenceEngine(model, new SchedulerConfig
{
    BlockSize = 8, NumBlocks = 64, MaxNumRunningSequences = 1,
    EnablePrefixCaching = false, StopRepetition = false, DecodeQuantumTokens = 1,
    MaxNumBatchedTokens = 16, SoloPrefillChunkSize = 16, MaxPrefillChunkSize = 8,
    MemoryAdmission = new(budget, _ => [new("host", bytes)], 32)
    {
        EnterSerialExecution = model.Enter,
        BeforeAdmission = admissionHook ? sequence =>
        {
            Require(Monitor.IsEntered(((IModelArchitecture)model).GpuComputeLock), "Admission hook ran outside model lock");
            Require(sequence.MemoryEnvelope == null, "Admission hook ran for an already admitted request");
            Interlocked.Increment(ref admissionCalls[lane]);
            // A synthetic, fixed constraint exercises the atomic admission path.
            // It is deliberately not a hardware observation or capacity refresh.
            return [new("host", 2 * bytes)];
        } : null
    }
})).ToArray();
var clock = Stopwatch.StartNew();
DateTimeOffset start = DateTimeOffset.UtcNow;
long cycles = 0, submitted = 0, cancelled = 0, completed = 0, gateCycles = 0;
long[] completions = new long[lanes];
double[] maximumWaitMs = new double[lanes];
long peakPrivate = 0, peakWorkingSet = 0;
string? failure = null;
double nextReport = 0;
using var progress = new StreamWriter(Path.Combine(directory, "progress.jsonl"), append: false) { AutoFlush = true };
try
{
    while (clock.Elapsed.TotalSeconds < durationSeconds)
    {
        using var external = budget.Reserve([new("host", 2 * bytes)]);
        var cts = new List<CancellationTokenSource>();
        var handles = new List<(int Lane, int Index, InferenceRequestHandle Handle)>();
        ComputeGate? gate = null;
        if (cycles % 20 == 0)
        {
            gate = new ComputeGate(); gate.Close(); engines[0].ComputeGate = gate; gateCycles++;
        }
        try
        {
            for (int lane = 0; lane < lanes; lane++)
                for (int index = 0; index < perLane; index++)
                {
                    var cancellation = new CancellationTokenSource(); cts.Add(cancellation);
                    if (index == 0) cancellation.Cancel();
                    int[] prompt = Enumerable.Range(1, 8 + (int)(cycles % 3) * 8).Select(t => t + lane * 11).ToArray();
                    var sequence = new SequenceState($"lane{lane}-slot{index}", prompt, 6, 8, SamplingConfig.Greedy,
                        userTag: index == 6 ? cancellation : null);
                    handles.Add((lane, index, engines[lane].SubmitRequest(sequence, cancellation.Token)));
                    submitted++;
                }
            for (int lane = 0; lane < lanes; lane++)
                foreach (int index in new[] { 2, 4 }) cts[lane * perLane + index].Cancel();
            foreach (var row in handles.Where(h => h.Index is 0 or 2 or 4))
            {
                var sequence = await row.Handle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
                Require(sequence.Status == SequenceStatus.FinishedAborted, "Queued cancellation did not abort");
            }
            gate?.Open();
            external.Dispose();
            await Task.WhenAll(handles.Select(h => h.Handle.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
            foreach (var row in handles)
            {
                var sequence = row.Handle.Sequence;
                if (row.Index is 0 or 2 or 4 or 6)
                {
                    var completion = await row.Handle.Completion;
                    Require(completion.Status == SequenceStatus.FinishedAborted,
                        $"Cancellation {row.Lane}/{row.Index}: {completion.Status}; {sequence.Error}");
                    cancelled++;
                }
                else
                {
                    Require(sequence.OutputTokens.SequenceEqual(SoakModel.Expected(sequence.PromptTokens, 6)), "Stateful output mismatch");
                    Require(sequence.Error == null && sequence.Status.IsFinished(), "Request failed");
                    maximumWaitMs[row.Lane] = Math.Max(maximumWaitMs[row.Lane],
                        (sequence.FirstScheduledAt!.Value - sequence.SubmittedAt).TotalMilliseconds);
                    completions[row.Lane]++; completed++;
                }
            }
            // Completion is published just before release hooks; hold model locks to
            // observe after each worker finishes that same step, without polling.
            foreach (var model in models) lock (((IModelArchitecture)model).GpuComputeLock) { }
            Require(budget.Snapshot().All(p => p.Reserved == 0 && p.Committed == 0), "Credit remained after a drained wave");
            Require(models.All(m => m.LiveBytes == 0), "Physical buffer remained after a drained wave");
            Require(engines.All(e => e.RunningCount == 0 && e.WaitingCount == 0), "Sequence remained after a drained wave");
            cycles++;
            if (clock.Elapsed.TotalSeconds >= nextReport)
            {
                using var process = Process.GetCurrentProcess(); process.Refresh();
                peakPrivate = Math.Max(peakPrivate, process.PrivateMemorySize64);
                peakWorkingSet = Math.Max(peakWorkingSet, process.WorkingSet64);
                progress.WriteLine(JsonSerializer.Serialize(Snapshot("running")));
                Console.WriteLine($"{clock.Elapsed.TotalSeconds:F1}s cycles={cycles} completed={completed} cancelled={cancelled}");
                nextReport = clock.Elapsed.TotalSeconds + 30;
            }
            await Task.Delay(10);
        }
        finally
        {
            gate?.Open();
            foreach (var cancellation in cts) cancellation.Dispose();
        }
    }
}
catch (Exception error) { failure = error.ToString(); }
finally
{
    foreach (var engine in engines) try { engine.Dispose(); } catch (Exception error) { failure ??= error.ToString(); }
    foreach (var model in models) try { model.Dispose(); } catch (Exception error) { failure ??= error.ToString(); }
}
if (budget.Snapshot().Any(p => p.Reserved != 0 || p.Committed != 0) || models.Any(m => m.LiveBytes != 0))
    failure ??= "Shutdown leaked charged or physical ownership";
if (admissionHook && admissionCalls.Any(count => count == 0))
    failure ??= "An engine never exercised the configured admission hook";
var final = Snapshot(failure == null ? "passed" : "failed");
File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(final, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(JsonSerializer.Serialize(final));
return failure == null ? 0 : 1;

object Snapshot(string status) => new
{
    status, start, now = DateTimeOffset.UtcNow, requestedSeconds = durationSeconds, elapsedSeconds = clock.Elapsed.TotalSeconds,
    cycles, submitted, completed, cancelled, gateCycles, completionsPerEngine = completions,
    admissionHook, admissionHookCalls = admissionCalls.Sum(), admissionHookCallsPerEngine = admissionCalls,
    admissionConstraintScope = admissionHook ? "Synthetic fixed 128 KiB host capacity; no hardware observation or capacity refresh." : "Disabled",
    processCpuSeconds = CpuSeconds(),
    maximumAdmissionWaitMsPerEngine = maximumWaitMs, maximumWaveSeconds = 10,
    physicalAllocations = models.Sum(m => m.Allocations), physicalReleases = models.Sum(m => m.Releases),
    livePhysicalBytes = models.Sum(m => m.LiveBytes), budget = budget.Snapshot(), peaks = budget.HighWatermarks(),
    sampledPeakPrivateBytes = peakPrivate, sampledPeakWorkingSetBytes = peakWorkingSet,
    runtimeSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(InferenceEngine).Assembly.Location))),
    memorySha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(MemoryBudget).Assembly.Location))),
    toolSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(SoakModel).Assembly.Location))),
    scope = "CPU runtime workers and real bounded unmanaged host allocations; deterministic stateful model. No GPU, trained-model, SSD, RSS-hard-limit or wall-clock throughput claim.", failure
};
double CpuSeconds() { using var process = Process.GetCurrentProcess(); return process.TotalProcessorTime.TotalSeconds; }
static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }

sealed class SoakModel : IModelArchitecture, IBatchedPagedModel
{
    readonly List<int> history = new();
    readonly Dictionary<string, (IntPtr Pointer, BudgetReservation Credit)> owners = new();
    SequenceState? current;
    public long Allocations, Releases, LiveBytes;
    public ModelConfig Config { get; } = new() { Architecture = "cpu-soak", VocabSize = 251 };
    public ITokenizer Tokenizer => null!;
    public IMultimodalInjector MultimodalInjector => null!;
    public IBackendExecutionPlan ExecutionPlan => null!;
    public bool SupportsKVStateSnapshot => false;
    public bool SupportsCrossSequenceKvReuse => false;
    public bool SupportsKVCacheTruncation => true;
    public void TruncateKVCache(int count) => history.RemoveRange(count, history.Count - count);
    public IDisposable Enter(SequenceState sequence) { current = sequence; return new Exit(() => current = null); }
    public float[] Forward(int[] tokens)
    {
        var sequence = current ?? throw new InvalidOperationException("No execution envelope");
        if (!owners.ContainsKey(sequence.RequestId))
        {
            var credit = sequence.MemoryEnvelope!.TryTake([new("host", 64 * 1024)])
                ?? throw new InvalidOperationException("Missing envelope credit");
            IntPtr pointer = IntPtr.Zero;
            try
            {
                pointer = Marshal.AllocHGlobal(64 * 1024);
                for (int offset = 0; offset < 64 * 1024; offset += 4096) Marshal.WriteByte(pointer, offset, 37);
                credit.Commit(); owners.Add(sequence.RequestId, (pointer, credit));
                Allocations++; LiveBytes += 64 * 1024;
            }
            catch { if (pointer != IntPtr.Zero) Marshal.FreeHGlobal(pointer); credit.Dispose(); throw; }
        }
        history.AddRange(tokens);
        if (sequence.UserTag is CancellationTokenSource cancellation) cancellation.Cancel();
        Thread.Sleep(1); // Model-work stand-in; keeps overlapping engine workers observable.
        var logits = new float[251]; logits[Next(history)] = 100; return logits;
    }
    public void ResetKVCache() => history.Clear();
    static int Next(IEnumerable<int> tokens)
    { uint hash = 2166136261; foreach (int token in tokens) hash = unchecked((hash ^ (uint)token) * 16777619); return (int)(hash % 251); }
    public static int[] Expected(IEnumerable<int> prompt, int count)
    { var state = prompt.ToList(); var output = new int[count]; for (int i = 0; i < count; i++) { output[i] = Next(state); state.Add(output[i]); } return output; }
    public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context) => throw new NotSupportedException();
    public void OnSequenceReleased(string requestId)
    {
        if (!owners.Remove(requestId, out var owner)) return;
        Marshal.FreeHGlobal(owner.Pointer); owner.Credit.Dispose(); Releases++; LiveBytes -= 64 * 1024;
    }
    public void Dispose() { foreach (string id in owners.Keys.ToArray()) OnSequenceReleased(id); }
    sealed class Exit(Action exit) : IDisposable { public void Dispose() => exit(); }
}
