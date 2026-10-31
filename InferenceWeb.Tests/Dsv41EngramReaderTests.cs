// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// How the direct-CUDA engine reads DeepSeek V4.1's host-mapped Engram tables (Dsv41EngramReader): the gather in
// parallel over (token, column), the prefetch pool, the hash preview the prefetch reads ahead with, and the settings
// that select them. None of them may change a result: the gather writes the same bytes whatever its parallelism,
// the preview names the same rows the forward will hash, and on the GPU the 1017-shaped fixture answers to the bit
// with every setting, including the S9 gather (TS_DSV41_ENGRAM_THREADS=1).
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed unsafe class Dsv41EngramReaderTests : IDisposable
{
    private const int Q5K = 13, HeadDim = 256, Columns = 24;
    private readonly List<IntPtr> _native = new();
    private readonly NativeEnvScope _env = new();

    public void Dispose()
    {
        foreach (IntPtr p in _native)
            Marshal.FreeHGlobal(p);
        _env.Dispose();
    }

    // ================================================================ a synthetic Q5_K table

    /// <summary><paramref name="rows"/> Q5_K rows of 256 values with random quants and finite scales, in native memory.</summary>
    private Dsv41EngramReader.Table Table(int rows, int seed)
    {
        long rowBytes = ManagedQuantizedOps.RowSize(Q5K, HeadDim);
        Assert.Equal(176, rowBytes);
        var bytes = new byte[rows * rowBytes];
        var rng = new Random(seed);
        rng.NextBytes(bytes);
        for (int r = 0; r < rows; r++)
        {
            // d and dmin (the block's two F16 scales) finite and modest, so every value is a number.
            BitConverter.TryWriteBytes(bytes.AsSpan((int)(r * rowBytes)), (System.Half)(rng.NextDouble() * 0.02 + 0.001));
            BitConverter.TryWriteBytes(bytes.AsSpan((int)(r * rowBytes) + 2), (System.Half)(rng.NextDouble() * 0.02));
        }
        IntPtr p = Marshal.AllocHGlobal(bytes.Length);
        _native.Add(p);
        Marshal.Copy(bytes, 0, p, bytes.Length);
        return new Dsv41EngramReader.Table((byte*)p, Q5K, rowBytes, rows);
    }

    /// <summary>Row ids for <paramref name="tokens"/> tokens, every one in range, with repeats (two columns of one token,
    /// and two tokens, reading the same row) and the table's first and last row.</summary>
    private static int[] Ids(int tokens, int rows, int seed)
    {
        var rng = new Random(seed);
        var ids = new int[tokens * Columns];
        for (int i = 0; i < ids.Length; i++)
            ids[i] = rng.Next(rows);
        ids[0] = 0;
        ids[^1] = rows - 1;
        if (ids.Length > 3)
            ids[2] = ids[1];
        if (tokens > 1)
            ids[Columns + 5] = ids[5];
        return ids;
    }

    private static float[] Gather(Dsv41EngramReader reader, Dsv41EngramReader.Table table, int[] ids, int tokens)
    {
        var dst = new float[tokens * Columns * HeadDim];
        Array.Fill(dst, float.NaN);
        fixed (int* i = ids)
        fixed (float* d = dst)
            reader.Gather(table, i, tokens, Columns, HeadDim, d);
        return dst;
    }

    private static void AssertSameBits(float[] expected, float[] actual, string what, bool rows = true)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail(rows
                    ? $"{what}: value {i} (token {i / (Columns * HeadDim)}, column {i / HeadDim % Columns}) is {actual[i]:R}, " +
                      $"the S9 gather wrote {expected[i]:R}"
                    : $"{what}: logit {i} is {actual[i]:R}, with the S9 gather {expected[i]:R} " +
                      $"(max |d| over the vector {expected.Zip(actual, (a, b) => MathF.Abs(a - b)).Max():G4})");
        }
    }

    // ================================================================ the gather

    /// <summary>
    /// The parallel gather writes the S9 gather's bytes at every parallelism, for a decode step (one token, 24 rows), a
    /// partial ubatch and a full one, with repeated rows: each row is dequantized by the same function into the same
    /// place, whatever thread does it. Every value is written (the destination starts as NaN, and none of the rows has one).
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(256)]
    public void TheGather_WritesTheS9Bytes_AtEveryParallelism(int tokens)
    {
        var table = Table(rows: 4099, seed: tokens);
        int[] ids = Ids(tokens, 4099, seed: 7 * tokens);
        using var s9 = new Dsv41EngramReader(threads: 1, prefetch: false, randomAdvice: false);
        float[] expected = Gather(s9, table, ids, tokens);
        Assert.DoesNotContain(expected, float.IsNaN);
        foreach (int threads in new[] { 2, 3, 16, 32 })
        {
            using var reader = new Dsv41EngramReader(threads, prefetch: false, randomAdvice: false);
            Assert.False(reader.Legacy);
            AssertSameBits(expected, Gather(reader, table, ids, tokens), $"{threads} threads, {tokens} tokens");
        }
    }

    /// <summary>A row id past the table is refused (it would read past the mapping), in either gather.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    public void ARowPastTheTable_IsRefused(int threads)
    {
        var table = Table(rows: 64, seed: 3);
        int[] ids = Ids(2, 64, seed: 4);
        ids[30] = 64;
        using var reader = new Dsv41EngramReader(threads, prefetch: false, randomAdvice: false);
        Exception error = Record.Exception(() => Gather(reader, table, ids, 2));
        Assert.NotNull(error);
        Assert.Contains("Engram lookup is out of bounds", error.ToString());
    }

    /// <summary>What an owning node sends: the named rows' raw bytes, back to back, at every parallelism.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    public void CopyRaw_CopiesTheNamedRows(int threads)
    {
        var table = Table(rows: 300, seed: 11);
        int[] ids = Ids(9, 300, seed: 12);
        var raw = new byte[ids.Length * table.RowBytes];
        using var reader = new Dsv41EngramReader(threads, prefetch: false, randomAdvice: false);
        fixed (int* i = ids)
        fixed (byte* r = raw)
            reader.CopyRaw(table, i, ids.Length, r);
        for (int k = 0; k < ids.Length; k++)
        {
            var expected = new ReadOnlySpan<byte>(table.Base + ids[k] * table.RowBytes, (int)table.RowBytes);
            Assert.True(expected.SequenceEqual(raw.AsSpan((int)(k * table.RowBytes), (int)table.RowBytes)), $"row {k}");
        }
    }

    // ================================================================ the prefetch pool

    /// <summary>
    /// The pool touches every row it is handed (a forward's rows, over several segments), a new forward replaces what
    /// is left of the last, and Dispose waits for every thread, after which nothing is read.
    /// </summary>
    [Fact]
    public void ThePrefetchPool_TouchesEveryRow_AndStopsOnDispose()
    {
        var table = Table(rows: 2000, seed: 21);
        int[] ids = Ids(64, 2000, seed: 22);
        var reader = new Dsv41EngramReader(threads: 4, prefetch: true, randomAdvice: false);
        try
        {
            var segments = new List<Dsv41EngramReader.Segment>
            {
                new(table, ids, 0, 100),
                new(table, ids, 100, 0),
                new(table, ids, 100, ids.Length - 100),
            };
            reader.PrefetchRows(segments);
            Assert.True(SpinWait.SpinUntil(() => reader.RowsTouched == ids.Length, TimeSpan.FromSeconds(20)),
                $"touched {reader.RowsTouched} of {ids.Length} rows");
            reader.PrefetchRows(new[] { new Dsv41EngramReader.Segment(table, ids, 0, 10) });
            Assert.True(SpinWait.SpinUntil(() => reader.RowsTouched == ids.Length + 10, TimeSpan.FromSeconds(20)));
            Assert.Equal(2, reader.Batches);
        }
        finally
        {
            reader.Dispose();
        }
        long touched = reader.RowsTouched;
        reader.PrefetchRows(new[] { new Dsv41EngramReader.Segment(table, ids, 0, 10) });
        Thread.Sleep(50);
        Assert.Equal(touched, reader.RowsTouched);
    }

    /// <summary>A reader with the prefetch off reads nothing ahead (the S9 setting).</summary>
    [Fact]
    public void WithThePrefetchOff_NothingIsReadAhead()
    {
        var table = Table(rows: 50, seed: 31);
        using var reader = new Dsv41EngramReader(threads: 1, prefetch: false, randomAdvice: false);
        reader.PrefetchRows(new[] { new Dsv41EngramReader.Segment(table, Ids(1, 50, 32), 0, Columns) });
        Thread.Sleep(50);
        Assert.Equal(0, reader.RowsTouched);
        Assert.Equal(0, reader.Batches);
    }

    // ================================================================ the reader's own threads

    /// <summary>Spins for about <paramref name="microseconds"/>: a stand-in for a page fault on network storage.</summary>
    private static void Busy(double microseconds)
    {
        long until = System.Diagnostics.Stopwatch.GetTimestamp() + (long)(microseconds * System.Diagnostics.Stopwatch.Frequency / 1e6);
        while (System.Diagnostics.Stopwatch.GetTimestamp() < until)
            Thread.SpinWait(20);
    }

    /// <summary>
    /// The gather and the raw copy run on the reader's own threads and the calling thread, never on the .NET thread
    /// pool, which a busy server's request threads could starve (design section 4.2): every row, whichever thread reads
    /// it, is read on a thread that is not a pool thread, and more than one thread takes part. The bytes are the S9
    /// gather's.
    /// </summary>
    [Fact]
    public void TheGather_RunsOnTheReadersOwnThreads_NeverOnTheThreadPool()
    {
        var table = Table(rows: 4099, seed: 51);
        int[] ids = Ids(8, 4099, seed: 52);
        using var s9 = new Dsv41EngramReader(threads: 1, prefetch: false, randomAdvice: false);
        float[] expected = Gather(s9, table, ids, 8);
        using var reader = new Dsv41EngramReader(threads: 4, prefetch: false, randomAdvice: false);
        var threads = new System.Collections.Concurrent.ConcurrentDictionary<int, bool>();
        int onPool = 0;
        reader.ReadHookForTests = kind =>
        {
            threads[Environment.CurrentManagedThreadId] = true;
            if (Thread.CurrentThread.IsThreadPoolThread)
                Interlocked.Increment(ref onPool);
            Busy(200);
        };
        float[] actual = null;
        Exception failure = null;
        var caller = new Thread(() =>
        {
            try
            {
                actual = Gather(reader, table, ids, 8);
                var raw = new byte[ids.Length * table.RowBytes];
                fixed (int* i = ids)
                fixed (byte* r = raw)
                    reader.CopyRaw(table, i, ids.Length, r);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        caller.Start();
        Assert.True(caller.Join(TimeSpan.FromSeconds(60)), "the gather did not finish");
        Assert.Null(failure);
        AssertSameBits(expected, actual, "4 threads with a slow read");
        Assert.Equal(0, onPool);
        Assert.InRange(threads.Count, 2, reader.Threads);
    }

    /// <summary>
    /// With the read-ahead busy on many slow rows, a gather still starts at once and finishes long before the read-ahead,
    /// the read-ahead's threads join it (each gather's rows are read by more than the calling thread), and the reader
    /// never has more rows in flight than <see cref="Dsv41EngramReader.Threads"/>, the gather's calling thread included.
    /// (S10 review: up to about 32 faults at once on a mount that serves about 16 usefully.)
    /// </summary>
    [Fact]
    public void AGather_GoesBeforeTheReadAhead_AndTheReadsInFlightStayAtTheThreads()
    {
        var table = Table(rows: 4099, seed: 61);
        int[] ahead = Ids(800, 4099, seed: 62);       // 19,200 rows at ~300 us on 3 threads: about 2 s
        int[] ids = Ids(4, 4099, seed: 63);
        using var reader = new Dsv41EngramReader(threads: 4, prefetch: true, randomAdvice: false);
        int inFlight = 0, most = 0;
        var gatherThreads = new System.Collections.Concurrent.ConcurrentDictionary<int, bool>();
        reader.ReadHookForTests = kind =>
        {
            if (kind == 1)
                gatherThreads[Environment.CurrentManagedThreadId] = true;
            int now = Interlocked.Increment(ref inFlight);
            int seen;
            while (now > (seen = Volatile.Read(ref most)) && Interlocked.CompareExchange(ref most, now, seen) != seen)
            {
            }
            Busy(300);
            Interlocked.Decrement(ref inFlight);
        };
        reader.PrefetchRows(new[] { new Dsv41EngramReader.Segment(table, ahead, 0, ahead.Length) });
        Assert.True(SpinWait.SpinUntil(() => reader.RowsTouched > 30, TimeSpan.FromSeconds(20)), "the read-ahead never started");
        for (int round = 0; round < 3; round++)
        {
            gatherThreads.Clear();
            float[] rows = Gather(reader, table, ids, 4);
            Assert.DoesNotContain(rows, float.IsNaN);
            Assert.True(gatherThreads.Count >= 2, $"gather {round}: only {gatherThreads.Count} thread read its rows; the read-ahead did not join it");
        }
        long touchedAfterGathers = reader.RowsTouched;
        Assert.True(touchedAfterGathers < ahead.Length,
            $"the read-ahead ({touchedAfterGathers} of {ahead.Length} rows) finished before three 96-row gathers did");
        reader.CancelPrefetch(wait: true);
        Assert.Equal(0, reader.ReadingAheadThreads);
        Assert.InRange(Volatile.Read(ref most), 2, reader.Threads);
    }

    /// <summary>
    /// CancelPrefetch abandons the forward's read-ahead; with wait it returns only when no thread is still reading it
    /// (each finishes the row it is on), after which nothing more is read. A reset, a failed forward and a page-cache
    /// eviction call it.
    /// </summary>
    [Fact]
    public void CancelPrefetch_StopsTheReadAhead_AndWaitsForItsThreads()
    {
        var table = Table(rows: 2000, seed: 71);
        int[] ids = Ids(400, 2000, seed: 72);
        using var reader = new Dsv41EngramReader(threads: 3, prefetch: true, randomAdvice: false);
        reader.ReadHookForTests = _ => Busy(2000);
        reader.PrefetchRows(new[] { new Dsv41EngramReader.Segment(table, ids, 0, ids.Length) });
        Assert.True(SpinWait.SpinUntil(() => reader.ReadingAheadThreads > 0 && reader.RowsTouched > 0, TimeSpan.FromSeconds(20)));
        reader.CancelPrefetch(wait: true);
        Assert.Equal(0, reader.ReadingAheadThreads);
        long touched = reader.RowsTouched;
        Assert.True(touched < ids.Length, $"{touched} of {ids.Length} rows were read ahead before the cancel");
        Thread.Sleep(100);
        Assert.Equal(touched, reader.RowsTouched);
        // The threads stay for the next forward.
        reader.ReadHookForTests = null;
        reader.PrefetchRows(new[] { new Dsv41EngramReader.Segment(table, ids, 0, 10) });
        Assert.True(SpinWait.SpinUntil(() => reader.RowsTouched == touched + 10, TimeSpan.FromSeconds(20)));
    }

    /// <summary>A failed row fails the whole gather with every failure inside one AggregateException (as Parallel.For
    /// did), and the reader's threads serve the next gather; a disposed reader still gathers, on the calling thread.</summary>
    [Fact]
    public void AFailedGather_LeavesTheReaderUsable()
    {
        var table = Table(rows: 64, seed: 81);
        int[] bad = Ids(4, 64, seed: 82);
        bad[3] = 64;
        bad[70] = -1;
        int[] good = Ids(4, 64, seed: 83);
        using var s9 = new Dsv41EngramReader(threads: 1, prefetch: false, randomAdvice: false);
        float[] expected = Gather(s9, table, good, 4);
        var reader = new Dsv41EngramReader(threads: 8, prefetch: false, randomAdvice: false);
        var error = Assert.Throws<AggregateException>(() => Gather(reader, table, bad, 4));
        Assert.All(error.InnerExceptions, e => Assert.Contains("Engram lookup is out of bounds", e.Message));
        AssertSameBits(expected, Gather(reader, table, good, 4), "after a failed gather");
        reader.Dispose();
        AssertSameBits(expected, Gather(reader, table, good, 4), "after Dispose");
    }

    // ================================================================ the settings

    private static Func<string, string> Env(params (string Name, string Value)[] values)
        => name => values.FirstOrDefault(v => v.Name == name).Value;

    [Fact]
    public void TheSettings_ReadAsDocumented()
    {
        int auto = Math.Min(16, Math.Max(1, Environment.ProcessorCount));
        using (var r = Dsv41EngramReader.FromEnvironment(Env()))
        {
            // The defaults: parallel reads and the advice on (bit-identical, low risk); the prefetch is opt-in.
            Assert.Equal(auto, r.Threads);
            Assert.False(r.Prefetch);
            Assert.Equal(auto > 1, r.RandomAdvice);
            Assert.Equal(Math.Max(1, auto - 1), r.PoolThreads);
        }
        using (var r = Dsv41EngramReader.FromEnvironment(Env(("TS_DSV41_ENGRAM_THREADS", "1"))))
        {
            Assert.True(r.Legacy);
            Assert.False(r.Prefetch);
            Assert.False(r.RandomAdvice);
            Assert.StartsWith("S9 gather", r.Describe());
        }
        using (var r = Dsv41EngramReader.FromEnvironment(Env(("TS_DSV41_ENGRAM_THREADS", "1"), ("TS_DSV41_ENGRAM_PREFETCH", "1"),
                   ("TS_DSV41_ENGRAM_RANDOM", "1"))))
        {
            Assert.True(r.Legacy);
            Assert.True(r.Prefetch);
            Assert.True(r.RandomAdvice);
        }
        using (var r = Dsv41EngramReader.FromEnvironment(Env(("TS_DSV41_ENGRAM_THREADS", " 32 "), ("TS_DSV41_ENGRAM_PREFETCH", "0"))))
        {
            Assert.Equal(32, r.Threads);
            Assert.Equal(31, r.PoolThreads);
            Assert.False(r.Prefetch);
            Assert.True(r.RandomAdvice);
        }
        using (var r = Dsv41EngramReader.FromEnvironment(Env(("TS_DSV41_ENGRAM_THREADS", "2"), ("TS_DSV41_ENGRAM_PREFETCH", "1"))))
        {
            Assert.True(r.Prefetch);
            Assert.True(r.RandomAdvice);
            Assert.Equal(1, r.PoolThreads);
        }
        foreach (var bad in new[] { ("TS_DSV41_ENGRAM_THREADS", "0"), ("TS_DSV41_ENGRAM_THREADS", "33"), ("TS_DSV41_ENGRAM_THREADS", "x"),
                     ("TS_DSV41_ENGRAM_PREFETCH", "yes"), ("TS_DSV41_ENGRAM_RANDOM", "2") })
            Assert.Contains(bad.Item1, Assert.Throws<NotSupportedException>(() => Dsv41EngramReader.FromEnvironment(Env(bad))).Message);
    }

    /// <summary>
    /// TS_DSV4_LOAD_DROP_CACHE with the native loader's rule: unset drops exactly when the upload plus the host-mapped
    /// tables plus 8 GiB exceed a known allowance (the 1017 node: ~76 GiB of weights and a 63 GiB slice in a 93.1 GiB
    /// cgroup), keeps otherwise and when the allowance is unknown; a set value is read as atoi reads it.
    /// </summary>
    [Fact]
    public void TheLoadDropRule_IsTheNativeLoaders()
    {
        const long GiB = 1L << 30;
        long cgroup = 99_999_997_952;
        var node = LoadPageCachePolicy.Decide(null, 76 * GiB, 63 * GiB, cgroup);
        Assert.True(node.Drop);
        Assert.True(node.Automatic);
        Assert.Contains("76.0 GiB upload + 63.0 GiB host-mapped + 8.0 GiB headroom exceeds the 93.1 GiB allowance; " +
            "TS_DSV4_LOAD_DROP_CACHE=0 overrides", LoadPageCachePolicy.Describe(node, 76 * GiB, 63 * GiB, cgroup));
        Assert.False(LoadPageCachePolicy.Decide(null, 20 * GiB, 63 * GiB, cgroup).Drop);
        var unknown = LoadPageCachePolicy.Decide(null, 500 * GiB, 0, 0);
        Assert.False(unknown.Drop);
        Assert.Contains("allowance unknown", LoadPageCachePolicy.Describe(unknown, 500 * GiB, 0, 0));
        Assert.False(LoadPageCachePolicy.Decide("0", 500 * GiB, 0, cgroup).Drop);
        Assert.True(LoadPageCachePolicy.Decide("1", 1, 0, 0).Drop);
        Assert.True(LoadPageCachePolicy.Decide(" 2x", 1, 0, 0).Drop);
        Assert.False(LoadPageCachePolicy.Decide("on", 500 * GiB, 0, cgroup).Drop);
        Assert.False(LoadPageCachePolicy.Decide("on", 500 * GiB, 0, cgroup).Automatic);
        // The allowance: the cgroup limit when there is one, else (here) none known.
        Assert.Equal(cgroup, LoadPageCachePolicy.Allowance(new CgroupMemory(1, "/x", cgroup, 0, 0, 0, 0)));
    }

    /// <summary>
    /// The loader's side of TS_DSV4_LOAD_DROP_CACHE on Linux: with DropPagesAfterRead every range the shard set reads
    /// (here the fixture's small F32/I32 tensors, which BeginStreaming reads ahead) has its whole pages dropped right
    /// after the read, and the kernel accepts every drop; without it nothing is dropped.
    /// </summary>
    [LinuxFact]
    public void TheShardSet_DropsWhatItRead()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        using (var kept = new GgufShardSet(path, "test"))
        {
            kept.BeginStreaming();
            Assert.NotEmpty(kept.PrefetchedNames);
            kept.EndLoad();
            Assert.Equal(0, kept.DroppedBytes);
        }
        using var dropped = new GgufShardSet(path, "test") { DropPagesAfterRead = true };
        dropped.BeginStreaming();
        var names = dropped.PrefetchedNames.ToHashSet();
        long read = dropped.TensorBytes().Where(t => names.Contains(t.Name)).Sum(t => t.Bytes);
        long during = dropped.DroppedBytes;
        Assert.True(during > 0 && during <= read, $"dropped {during} of {read} bytes read");
        dropped.EndLoad();
        Assert.Equal(during, dropped.DroppedBytes);
    }

    [Fact]
    public void TheWarmThreads_ReadAsDocumented()
    {
        _env.Set("TS_DSV41_ENGRAM_WARM_THREADS", null);
        Assert.Equal(16, DeepSeek4CudaExecutor.EngramWarmThreads());
        _env.Set("TS_DSV41_ENGRAM_WARM_THREADS", "4");
        Assert.Equal(4, DeepSeek4CudaExecutor.EngramWarmThreads());
        _env.Set("TS_DSV41_ENGRAM_WARM_THREADS", "65");
        Assert.Throws<NotSupportedException>(() => DeepSeek4CudaExecutor.EngramWarmThreads());
    }

    // ================================================================ the hash preview

    /// <summary>
    /// The prefetch reads ahead with <see cref="Dsv41EngramData.PreviewHashes"/>, which must name exactly the rows the
    /// forward's own hashing will, without touching the history: on the 1017-shaped fixture's Engram metadata, a
    /// sequence of forwards (a prompt over several ubatches, decode steps, an append from an odd position, image positions
    /// and the raw ids the token map sends to the pad, then a rewind) is previewed whole before each forward and compared
    /// with the hashes of each of its ubatches, laid out as the preview lays out the forward.
    /// </summary>
    [Fact]
    public void ThePreview_NamesTheRowsTheForwardHashes()
    {
        using var shards = new GgufShardSet(Dsv41SyntheticModelBuilder.Fixture(), "test");
        Dsv41EngramData data = Dsv41EngramData.Load(shards.First, shards.InfoOf);
        int tables = data.Layers.Length, columns = (int)data.HashColumns;
        var rng = new Random(41);
        int[] Tokens(int n) => Enumerable.Range(0, n).Select(_ => rng.Next(10) switch
        {
            0 => -1,                               // an image position
            1 => Dsv41SyntheticModelBuilder.RawPad,
            _ => rng.Next(Dsv41SyntheticModelBuilder.Vocab),
        }).ToArray();

        var history = new Dsv41EngramHistory();
        int position = 0;
        void Forward(int[] tokens, int ubatch)
        {
            int[] before = history.Tokens == null ? null : (int[])history.Tokens.Clone();
            int beforeLength = history.Length;
            int[] preview = data.PreviewHashes(tokens, position, history.Tokens, history.Length);
            Assert.Equal(beforeLength, history.Length);
            if (before != null)
                Assert.Equal(before, history.Tokens);
            for (int u = 0; u < tokens.Length; u += ubatch)
            {
                int nt = Math.Min(ubatch, tokens.Length - u);
                int[] real = data.HashTokens(tokens.AsSpan(u, nt), position + u, ref history.Tokens, ref history.Length);
                for (int t = 0; t < tables; t++)
                    for (int i = 0; i < nt * columns; i++)
                        Assert.True(real[(t * nt) * columns + i] == preview[(t * tokens.Length + u) * columns + i],
                            $"table {t}, position {position + u + i / columns}, column {i % columns}");
            }
            position += tokens.Length;
        }

        Forward(Tokens(150), 64);
        for (int i = 0; i < 6; i++)
            Forward(Tokens(1), 64);
        Forward(Tokens(21), 64);
        Forward(Tokens(1), 64);
        // A rewind shortens the history; the next forward previews from the new end.
        position = 40;
        history.Length = 40;
        Forward(Tokens(300), 128);
        Assert.Throws<InvalidOperationException>(() => data.PreviewHashes(Tokens(1), position + 1, history.Tokens, history.Length));
    }

    // ================================================================ GPU: the engine, every setting

    private static readonly int[] Prompt = Seeded(901, 150), DecodeTokens = Seeded(902, 18), Append = Seeded(903, 21);

    private static int[] Seeded(int seed, int count)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => rng.Next(0, Dsv41SyntheticModelBuilder.Vocab)).ToArray();
    }

    /// <summary>A 150-token prompt (64 + 64 + 22), 15 decode steps, a 21-token append, 3 more steps, a reset and the
    /// prompt again with 3 steps.</summary>
    private static float[][] Script(ModelBase model)
    {
        var rows = new List<float[]> { (float[])model.Forward(Prompt).Clone() };
        for (int i = 0; i < DecodeTokens.Length; i++)
        {
            if (i == 15)
                rows.Add((float[])model.Forward(Append).Clone());
            rows.Add((float[])model.Forward(new[] { DecodeTokens[i] }).Clone());
        }
        model.ResetKVCache();
        rows.Add((float[])model.Forward(Prompt).Clone());
        for (int i = 0; i < 3; i++)
            rows.Add((float[])model.Forward(new[] { DecodeTokens[i] }).Clone());
        return rows.ToArray();
    }

    /// <summary>
    /// The direct engine with host-mapped Engram tables (TS_DSV41_ENGRAM_DEVICE=0) on the 1017-shaped fixture answers to
    /// the bit with the S9 gather (TS_DSV41_ENGRAM_THREADS=1, no prefetch, no advice), the default reader (16 rows at
    /// once, MADV_RANDOM, no prefetch) and two other settings, over a multi-ubatch prefill, decode steps, an append and a
    /// reset. The default run read nothing ahead; the run that asked for the prefetch did.
    /// </summary>
    [CudaFact]
    public void TheHostMappedEngine_AnswersToTheBit_AtEverySetting()
    {
        _env.Set("MAX_CONTEXT", "1024");
        _env.Set("TS_DSV4_UBATCH", "64");
        _env.Set("TS_DSV41_YOCO_KEEP", "0");
        _env.Set("TS_DSV41_ENGRAM_DEVICE", "0");
        foreach (string name in new[] { "TS_DSV4_MMAP", "TS_DSV4_CUDA_TRACE_DIR", "TS_DSV4_PERF", "KV_CACHE_DTYPE",
                     "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TS_N_CPU_MOE", "TS_CPU_MOE",
                     "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_PREFETCH", "TS_DSV41_ENGRAM_RANDOM" })
            _env.Set(name, null);
        string path = Dsv41SyntheticModelBuilder.Fixture();

        (float[][] Logits, int Threads, bool Prefetch, long Touched) Run(string threads, string prefetch, string random)
        {
            _env.Set("TS_DSV41_ENGRAM_THREADS", threads);
            _env.Set("TS_DSV41_ENGRAM_PREFETCH", prefetch);
            _env.Set("TS_DSV41_ENGRAM_RANDOM", random);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            var exec = ((DeepSeek4Model)model).CudaExecutor;
            Assert.False(exec.Engine.EngramResident, "TS_DSV41_ENGRAM_DEVICE=0 left the tables in VRAM");
            float[][] logits = Script(model);
            return (logits, exec.EngramReader.Threads, exec.EngramReader.Prefetch, exec.EngramReader.RowsTouched);
        }

        var s9 = Run("1", null, null);
        Assert.Equal(1, s9.Threads);
        Assert.False(s9.Prefetch);
        Assert.Equal(0, s9.Touched);
        var settings = new[] { (null as string, null as string, null as string), ("3", "0", "0"), ("32", "1", "1") };
        foreach (var (threads, prefetch, random) in settings)
        {
            var run = Run(threads, prefetch, random);
            for (int i = 0; i < s9.Logits.Length; i++)
                AssertSameBits(s9.Logits[i], run.Logits[i], $"TS_DSV41_ENGRAM_THREADS={threads ?? "<default>"}, PREFETCH={prefetch ?? "<default>"}, " +
                    $"RANDOM={random ?? "<default>"}: logit vector {i}", rows: false);
            if (threads == null)
            {
                Assert.True(run.Threads > 1);
                Assert.False(run.Prefetch);
                Assert.Equal(0, run.Touched);
            }
            if (prefetch == "1")
            {
                Assert.True(run.Prefetch);
                Assert.True(run.Touched > 0, "the reader that asked for the prefetch read nothing ahead");
            }
        }
    }
}
