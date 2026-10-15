using System.Text.Json;
using TensorSharp.Runtime.Speculative;

int checks = 0;
void Equal(double expected, double actual)
{
    checks++;
    if (!double.IsFinite(expected) || !double.IsFinite(actual) || Math.Abs(expected - actual) > 1e-10)
        throw new Exception($"Expected {expected:R}, got {actual:R}");
}
void True(bool value) { checks++; if (!value) throw new Exception("Assertion failed"); }
void Invalid(Action action)
{
    checks++;
    try { action(); } catch (ArgumentException) { return; }
    throw new Exception("Expected invalid observation to be rejected");
}
ConcurrentDelivery R(double offset, params double[] times) => new(offset, times);

// Early requests can decode before the last first delivery. Those earlier
// decoded tokens must be excluded from a denominator starting at that boundary.
var simultaneous = ConcurrentDecodeMetrics.Calculate(new[] { R(0, 10, 20, 30, 60), R(0, 40, 50, 70) }, 80);
True(simultaneous.Established);
Equal(40, simultaneous.LastFirstTokenOffsetMs);
Equal(3, simultaneous.TokensAfterLastFirst);
Equal(40, simultaneous.WindowMs);
Equal(75, simultaneous.TokensPerSecond);

// Per-request TTFT is10/40ms, while the wave's last first token is110ms.
var staggered = ConcurrentDecodeMetrics.Calculate(new[] { R(0, 40, 60, 120, 140), R(100, 10, 20, 50) }, 160);
Equal(110, staggered.LastFirstTokenOffsetMs);
Equal(4, staggered.TokensAfterLastFirst);
Equal(50, staggered.WindowMs);
Equal(80, staggered.TokensPerSecond);

// Exact boundary ties and an EOS-only tail do not fabricate extra deliveries.
var ties = ConcurrentDecodeMetrics.Calculate(new[] { R(0, 10, 30, 40), R(20, 10, 20) }, 50);
Equal(2, ties.TokensAfterLastFirst);
Equal(100, ties.TokensPerSecond);
var noTail = ConcurrentDecodeMetrics.Calculate(new[] { R(0, 10), R(0, 10) }, 10);
Equal(0, noTail.WindowMs); Equal(0, noTail.TokensPerSecond);
True(!ConcurrentDecodeMetrics.Calculate(new[] { R(0, 1), R(3) }, 5).Established);
True(!ConcurrentDecodeMetrics.Calculate(Array.Empty<ConcurrentDelivery>(), 0).Established);
Invalid(() => ConcurrentDecodeMetrics.Calculate(new[] { R(-1, 1) }, 5));
Invalid(() => ConcurrentDecodeMetrics.Calculate(new[] { R(0, 2, 1) }, 5));
Invalid(() => ConcurrentDecodeMetrics.Calculate(new[] { R(0, double.NaN) }, 5));
Invalid(() => ConcurrentDecodeMetrics.Calculate(new[] { R(4, 2) }, 5));
Invalid(() => ConcurrentDecodeMetrics.Calculate(new[] { R(0, 1) }, double.PositiveInfinity));

// A partial output followed by an error must fail, even when tokens exist.
True(RequestCompletionChecks.ConcurrentFailed(4, "error", "allocation failure"));
True(RequestCompletionChecks.ConcurrentFailed(4, "error", null));
True(RequestCompletionChecks.ConcurrentFailed(4, "eos", "late failure"));
True(RequestCompletionChecks.ConcurrentFailed(0, "eos", null));
True(!RequestCompletionChecks.ConcurrentFailed(4, "eos", null));
True(RequestCompletionChecks.HasError("error", null));

var engaged = new RequestSpeculationCounters(20, 12, 4, 2, 1, 3, 1, 2, 3);
var fallback = new RequestSpeculationCounters(0, 0, 0, 8, 0, 4, 0, 1, 4);
var sum = RequestSpeculationCounters.Sum(new[] { engaged, fallback });
Equal(20, sum.Drafted); Equal(12, sum.Accepted); Equal(4, sum.VerifySteps);
Equal(10, sum.PlainSteps); Equal(1, sum.Rollbacks); Equal(7, sum.ParkedSteps);
Equal(1, sum.GovernorWins); Equal(3, sum.GovernorLosses); Equal(7, sum.GovernorParkedSteps);
// Retaining individual records prevents an aggregate >0 from hiding fallback.
var timelines = new[]
{
    new RequestTimeline("a", 1700000000000, 0, 2, new() { 10, 20 }, "eos", null!, engaged),
    new RequestTimeline("b", 1700000000100, 100, 1, new() { 10 }, "error", "late failure", fallback),
};
var restored = JsonSerializer.Deserialize<RequestTimeline[]>(JsonSerializer.Serialize(timelines))!;
Equal(100, restored[1].SubmissionOffsetMs); Equal(0, restored[1].Speculation.VerifySteps);
True(restored[1].Error == "late failure"); True(restored[1].TokenTimesMs.SequenceEqual(new[] { 10.0 }));

// Strict CLI values and actual arm enumeration preserve the old default while
// allowing a real reversed control. Every arm gets one request per warmup/pass.
True(SpeculativeBenchmarkOrdering.Parse("plain-first") == SpeculativeRunOrder.PlainFirst);
True(SpeculativeBenchmarkOrdering.Parse("spec-first") == SpeculativeRunOrder.SpecFirst);
Invalid(() => SpeculativeBenchmarkOrdering.Parse("auto"));
Invalid(() => SpeculativeBenchmarkOrdering.Parse(""));
Invalid(() => SpeculativeBenchmarkOrdering.Parse("Plain-First"));
var candidates = new[] { "ngram", "auto" };
True(SpeculativeBenchmarkOrdering.Arrange(SpeculativeRunOrder.PlainFirst, "plain", candidates)
    .SequenceEqual(new[] { "plain", "ngram", "auto" }));
True(SpeculativeBenchmarkOrdering.Arrange(SpeculativeRunOrder.SpecFirst, "plain", candidates)
    .SequenceEqual(new[] { "ngram", "auto", "plain" }));
foreach (SpeculativeRunOrder order in Enum.GetValues<SpeculativeRunOrder>())
{
    True(SpeculativeBenchmarkOrdering.Arrange(order, "plain", Array.Empty<string>()).SequenceEqual(new[] { "plain" }));
    var passes = Enumerable.Range(0, 3).SelectMany(_ => SpeculativeBenchmarkOrdering.Arrange(order, "plain", candidates));
    True(passes.GroupBy(x => x).All(group => group.Count() == 3));
}
Invalid(() => SpeculativeBenchmarkOrdering.Arrange((SpeculativeRunOrder)99, "plain", candidates).ToArray());

// Link the real SpeculationStats source: mapping must retain fractional phase
// values and calibrated rates, independent of the rounded/truncated log string.
var stats = new SpeculationStats
{
    DraftTicks = 12345, VerifyTicks = 23456, SnapshotTicks = 34567,
    RollbackTicks = 45678, CatchUpTicks = 56789, PlainTicks = 67891,
    PlainMsPerToken = 12.345678901234, SpecMsPerToken = 23.456789012345,
};
var timing = SpeculativePhaseTimings.From(stats)!;
Equal(stats.DraftMs, timing.DraftMs); Equal(stats.VerifyMs, timing.VerifyMs);
Equal(stats.SnapshotMs, timing.SnapshotMs); Equal(stats.RollbackMs, timing.RollbackMs);
Equal(stats.CatchUpMs, timing.CatchUpMs); Equal(stats.PlainMs, timing.PlainMs);
Equal(stats.PlainMsPerToken, timing.PlainMsPerToken); Equal(stats.SpecMsPerToken, timing.SpecMsPerToken);
True(SpeculativePhaseTimings.From(null) == null);
var timedTimeline = timelines[0] with { SpeculationTimings = timing };
var timedRestored = JsonSerializer.Deserialize<RequestTimeline>(JsonSerializer.Serialize(timedTimeline))!;
True(timedRestored.SpeculationTimings == timing);
stats.Reset();
True(timing.DraftMs > 0 && timing.PlainMsPerToken > 0); // immutable snapshot survives reuse
var untimedJson = JsonSerializer.Serialize(timelines[0], new JsonSerializerOptions
{
    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
});
True(!untimedJson.Contains("SpeculationTimings")); // absent is not a measured zero

True(SpeculativeDiagnosticPrefill.Parse("refill") == DiagnosticPrefillMode.Refill);
True(SpeculativeDiagnosticPrefill.Parse("forward") == DiagnosticPrefillMode.Forward);
Invalid(() => SpeculativeDiagnosticPrefill.Parse("Forward"));
Invalid(() => SpeculativeDiagnosticPrefill.Parse(""));
int[] diagnosticTokens = { 42, 17 };
float[] forwardResult = { 1 }, refillResult = { 2 };
int forwardCalls = 0, refillCalls = 0;
float[] Forward(int[] input) { True(ReferenceEquals(input, diagnosticTokens)); forwardCalls++; return forwardResult; }
float[] Refill(int[] input) { True(ReferenceEquals(input, diagnosticTokens)); refillCalls++; return refillResult; }
True(ReferenceEquals(SpeculativeDiagnosticPrefill.Run(default, diagnosticTokens, Forward, Refill), refillResult));
Equal(0, forwardCalls); Equal(1, refillCalls);
True(ReferenceEquals(SpeculativeDiagnosticPrefill.Run(DiagnosticPrefillMode.Forward, diagnosticTokens, Forward, Refill), forwardResult));
Equal(1, forwardCalls); Equal(1, refillCalls);
Invalid(() => SpeculativeDiagnosticPrefill.Run((DiagnosticPrefillMode)99, diagnosticTokens, Forward, Refill));

Console.WriteLine($"PASS {checks} model-free metric, ordering, precise timing, serialization and error checks");
