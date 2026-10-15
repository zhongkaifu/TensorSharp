// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

/// <summary>Probe-only text prefill plan. All slices exist before the first stateful call.</summary>
internal sealed class PrefillChunkPlan
{
    private readonly int[][] _chunks;
    public int RequestedChunkSize { get; }
    public int[] ChunkSizes { get; }
    public string Entry => RequestedChunkSize == 0 ? "ForwardRefill(prompt)" : "ForwardRefill(chunk)";

    private PrefillChunkPlan(int requestedChunkSize, int[][] chunks)
    {
        RequestedChunkSize = requestedChunkSize;
        _chunks = chunks;
        ChunkSizes = chunks.Select(chunk => chunk.Length).ToArray();
    }

    public static PrefillChunkPlan Create(int[] prompt, int requestedChunkSize)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(requestedChunkSize);
        if (prompt.Length == 0) throw new ArgumentException("Prefill prompt must not be empty.", nameof(prompt));
        // Zero deliberately retains the original array and single-call entry.
        if (requestedChunkSize == 0) return new PrefillChunkPlan(0, [prompt]);
        // Avoid (length + chunk - 1), which overflows for valid large chunk values.
        int count = 1 + (prompt.Length - 1) / requestedChunkSize;
        var chunks = new int[count][];
        int offset = 0;
        for (int i = 0; i < count; i++)
        {
            int length = Math.Min(requestedChunkSize, prompt.Length - offset);
            chunks[i] = prompt.AsSpan(offset, length).ToArray();
            offset = checked(offset + length);
        }
        if (offset != prompt.Length) throw new InvalidOperationException("Prefill plan did not consume the prompt.");
        return new PrefillChunkPlan(requestedChunkSize, chunks);
    }

    public float[] Execute(Func<int[], float[]> forwardRefill)
    {
        ArgumentNullException.ThrowIfNull(forwardRefill);
        float[]? logits = null;
        foreach (int[] chunk in _chunks)
            logits = forwardRefill(chunk);
        // The probe captures/hashes only this prediction row, preserving its teacher convention.
        return logits!;
    }

    public static void SelfTest()
    {
        int checks = 0;
        void Require(bool condition, string message)
        {
            if (!condition) throw new InvalidOperationException(message);
            checks++;
        }
        var prompt = Enumerable.Range(0, 63).ToArray();
        foreach (int size in new[] { 0, 1, 8, 63, 64, int.MaxValue })
        {
            var plan = Create(prompt, size);
            var seen = new List<int>();
            int calls = 0;
            float[]? last = null;
            float[] result = plan.Execute(chunk =>
            {
                calls++;
                if (size == 0) Require(ReferenceEquals(chunk, prompt), "Default entry copied the prompt.");
                seen.AddRange(chunk);
                last = [chunk[^1]];
                return last;
            });
            Require(seen.SequenceEqual(prompt), "Chunk plan changed prompt order or token count.");
            Require(ReferenceEquals(result, last) && result[0] == 62, "Chunk plan returned an intermediate prediction row.");
            Require(calls == plan.ChunkSizes.Length && plan.ChunkSizes.Sum() == 63, "Chunk plan reported incorrect sizes.");
            Require(size == 0 || plan.ChunkSizes.All(length => length > 0 && length <= size), "Invalid chunk length.");
            if (size == 8) Require(plan.ChunkSizes.SequenceEqual(new[] { 8, 8, 8, 8, 8, 8, 8, 7 }), "Trailing chunk was lost.");
        }
        foreach (var invalid in new[] { (Array.Empty<int>(), 1), (prompt, -1) })
        {
            bool rejected = false;
            try { Create(invalid.Item1, invalid.Item2); }
            catch (ArgumentException) { rejected = true; }
            Require(rejected, "Invalid prefill plan was accepted.");
        }
        int attempted = 0;
        var expected = new InvalidOperationException("injected stateful failure");
        try
        {
            Create(prompt, 8).Execute(_ =>
            {
                if (++attempted == 2) throw expected;
                return [1];
            });
            throw new InvalidOperationException("Stateful failure was swallowed.");
        }
        catch (InvalidOperationException error) when (ReferenceEquals(error, expected))
        {
            Require(attempted == 2, "A failed stateful chunk was retried or later chunks executed.");
        }
        Console.WriteLine($"PrefillChunkPlan: {checks} CPU checks passed; no model/backend was initialized.");
    }
}
