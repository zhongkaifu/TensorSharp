// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

internal static class SelfTests
{
    public static int Run()
    {
        int checks = 0;
        void Check(bool ok) { if (!ok) throw new InvalidOperationException($"Self-test {checks + 1} failed."); checks++; }
        bool Reject(params string[] args)
        {
            try { ProbeOptions.Parse(args); return false; }
            catch (ArgumentException) { return true; }
        }
        Check(ProbeOptions.Parse(["--output", "unused.json"]).AllocationBytes == 0);
        Check(ProbeOptions.Parse(["--output", "unused.json", "--allocation-bytes", "67108864"]).AllocationBytes == 64L << 20);
        Check(ProbeOptions.Parse(["--output", "unused.json", "--expect-commit-ceiling-bytes", "536870912"]).ExpectedCommitCeiling == 512L << 20);
        foreach (string bad in new[] { "", "-1", "1.5", "64MiB", " 1", "１", "67108865", "9223372036854775808" })
            Check(Reject("--output", "unused.json", "--allocation-bytes", bad));
        Check(Reject()); Check(Reject("--output")); Check(Reject("--output", ""));
        Check(Reject("--output", "a", "--output", "b"));
        Check(Reject("--output", "a", "--model", "never-read.gguf"));
        Check(Reject("--output", "a", "--expect-commit-ceiling-bytes", "0"));
        var good = new HostMemorySnapshot(32L << 30, 17L << 30, 512L << 20, 400L << 20);
        Check(ObservationCheck.Evaluate(good, true, 512L << 20).Passed);
        Check(!ObservationCheck.Evaluate(good with { CommitLimit = 40L << 30 }, true, 512L << 20).Passed);
        Check(!ObservationCheck.Evaluate(good with { CommitAvailable = 513L << 20 }, true, 512L << 20).Passed);
        Check(ObservationCheck.Evaluate(good with { CommitAvailable = 0 }, true, 512L << 20).Passed);
        Check(!ObservationCheck.Evaluate(good with { Available = good.Total + 1 }, true, 512L << 20).Passed);
        Check(!ObservationCheck.Evaluate(good with { CommitLimit = null }, true, null).Passed);
        var linux = new HostMemorySnapshot(8L << 30, 4L << 30);
        var unknown = ObservationCheck.Evaluate(linux, false, null);
        Check(unknown.Passed && unknown.CommitUnknown && unknown.RequestedCommitCeilingObserved is null);
        Check(!ObservationCheck.Evaluate(linux, false, 512L << 20).Passed);
        Check(!ObservationCheck.Evaluate(good, false, null).Passed);
        Console.WriteLine($"PASS {checks} HostConstraintProbe parameter/classification checks; no API, native allocation, model or device executed.");
        return 0;
    }
}
