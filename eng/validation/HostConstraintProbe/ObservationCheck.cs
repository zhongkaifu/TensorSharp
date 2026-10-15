// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

internal sealed record ObservationCheck(bool Passed, bool? RequestedCommitCeilingObserved,
    bool CommitUnknown, IReadOnlyList<string> Failures)
{
    public static ObservationCheck Evaluate(HostMemorySnapshot sample, bool windows, long? expectedCeiling)
    {
        var failures = new List<string>();
        if (sample.Total <= 0 || sample.Available < 0 || sample.Available > sample.Total)
            failures.Add("Physical capacity is invalid.");
        bool unknown = !sample.CommitLimit.HasValue && !sample.CommitAvailable.HasValue;
        bool validCommit = sample.CommitLimit is > 0 && sample.CommitAvailable is >= 0
            && sample.CommitAvailable <= sample.CommitLimit;
        if (windows && !validCommit) failures.Add("Windows did not provide a valid process-commit allowance.");
        if (!windows && !unknown) failures.Add("This provider currently declares commit unknown on non-Windows platforms.");
        bool? respectsCeiling = null;
        if (expectedCeiling.HasValue)
        {
            respectsCeiling = windows && validCommit && expectedCeiling > 0
                && sample.CommitLimit <= expectedCeiling && sample.CommitAvailable <= expectedCeiling;
            if (respectsCeiling != true)
                failures.Add("The API observation does not establish both commit limit and available bytes within the requested ceiling.");
        }
        return new(failures.Count == 0, respectsCeiling, unknown, failures);
    }
}
