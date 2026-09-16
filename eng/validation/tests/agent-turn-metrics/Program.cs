using System.Text.Json;

int checks = 0;
void Equal(double expected, double actual)
{
    checks++;
    if (Math.Abs(expected - actual) > 1e-10) throw new Exception($"Expected {expected:R}, got {actual:R}");
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
Console.WriteLine($"PASS {checks} model-free concurrent metric, counter, serialization and error checks");
