using System;
using System.Collections.Generic;

internal sealed record ConcurrentDelivery(double SubmissionOffsetMs, IReadOnlyList<double> TokenTimesMs);

internal static class RequestCompletionChecks
{
    internal static bool HasError(string? finish, string? error) => error != null || finish == "error";
    internal static bool ConcurrentFailed(int tokens, string? finish, string? error) => tokens == 0 || HasError(finish, error);
}

internal sealed record RequestTimeline(string Id, long StartedUnixMilliseconds, double SubmissionOffsetMs, int OutTokens,
    List<double> TokenTimesMs, string Finish, string Error, RequestSpeculationCounters Speculation);

internal sealed record RequestSpeculationCounters(long Drafted, long Accepted, long VerifySteps, long PlainSteps,
    long Rollbacks, long ParkedSteps, int GovernorWins, int GovernorLosses, int GovernorParkedSteps)
{
    internal static RequestSpeculationCounters Sum(IEnumerable<RequestSpeculationCounters> requests)
    {
        long drafted = 0, accepted = 0, verify = 0, plain = 0, rollback = 0, parked = 0;
        int wins = 0, losses = 0, governorParked = 0;
        foreach (var request in requests)
        {
            drafted = checked(drafted + request.Drafted); accepted = checked(accepted + request.Accepted);
            verify = checked(verify + request.VerifySteps); plain = checked(plain + request.PlainSteps);
            rollback = checked(rollback + request.Rollbacks); parked = checked(parked + request.ParkedSteps);
            wins = checked(wins + request.GovernorWins); losses = checked(losses + request.GovernorLosses);
            governorParked = checked(governorParked + request.GovernorParkedSteps);
        }
        return new(drafted, accepted, verify, plain, rollback, parked, wins, losses, governorParked);
    }
}

internal sealed record ConcurrentDecodeMetrics(
    bool Established, double LastFirstTokenOffsetMs, int TokensAfterLastFirst, double WindowMs, double TokensPerSecond)
{
    /// <summary>
    /// All offsets use the same monotonic wave origin. Request token times are
    /// relative to that request's submission. Count only tokens delivered after
    /// every request has delivered its first token, through wave completion.
    /// Individual request TTFT is not the last first-token offset of a wave.
    /// </summary>
    internal static ConcurrentDecodeMetrics Calculate(IReadOnlyList<ConcurrentDelivery> requests, double completedOffsetMs)
    {
        if (!double.IsFinite(completedOffsetMs) || completedOffsetMs < 0)
            throw new ArgumentOutOfRangeException(nameof(completedOffsetMs));
        bool hasEmpty = requests.Count == 0;
        double lastFirst = 0;
        foreach (var request in requests)
        {
            if (!double.IsFinite(request.SubmissionOffsetMs) || request.SubmissionOffsetMs < 0
                || request.SubmissionOffsetMs > completedOffsetMs)
                throw new ArgumentException("Request submission is outside the measured wave.", nameof(requests));
            double previous = 0;
            foreach (double delivered in request.TokenTimesMs)
            {
                if (!double.IsFinite(delivered) || delivered < previous
                    || request.SubmissionOffsetMs + delivered > completedOffsetMs)
                    throw new ArgumentException("Token deliveries must be ordered inside the measured wave.", nameof(requests));
                previous = delivered;
            }
            if (request.TokenTimesMs.Count == 0) hasEmpty = true;
            else lastFirst = Math.Max(lastFirst, request.SubmissionOffsetMs + request.TokenTimesMs[0]);
        }
        if (hasEmpty) return new(false, 0, 0, 0, 0);
        int count = 0;
        foreach (var request in requests)
            foreach (double delivered in request.TokenTimesMs)
                if (request.SubmissionOffsetMs + delivered > lastFirst) count++;
        double window = completedOffsetMs - lastFirst;
        return new(true, lastFirst, count, window, window > 0 ? count * 1000.0 / window : 0);
    }
}
