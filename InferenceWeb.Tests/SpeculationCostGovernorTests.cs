using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>The cost governor's verdicts, driven with synthetic step timings.</summary>
public class SpeculationCostGovernorTests
{
    private static long Ticks(double ms) => (long)(ms * Stopwatch.Frequency / 1000.0);

    /// <summary>Drive a full first round: a probe that wins (4 tokens per 20 ms step
    /// against a 15 ms plain step) and the plain calibration behind it.</summary>
    private static SpeculationCostGovernor WinningRound()
    {
        var g = new SpeculationCostGovernor { Enabled = true };
        g.Reset();
        // The first speculative sample is discarded (graph builds), then 8 are measured.
        for (int i = 0; i < 9; i++)
        {
            Assert.True(g.AllowsSpeculation(), $"probe step {i}");
            g.Record(speculated: true, tokensEmitted: 4, elapsedTicks: Ticks(20));
        }
        // Then the plain baseline: one discarded, three measured.
        for (int i = 0; i < 4; i++)
        {
            Assert.False(g.AllowsSpeculation(), $"calibration step {i}");
            g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
        }
        Assert.True(g.AllowsSpeculation(), "the round should have ended in a win");
        Assert.False(g.IsParked);
        return g;
    }

    [Fact]
    public void AWinKeepsHoldingWhileItsStepsStayCheap()
    {
        var g = WinningRound();
        for (int i = 0; i < 40; i++)
        {
            Assert.True(g.AllowsSpeculation(), $"held step {i}");
            g.Record(speculated: true, tokensEmitted: 3, elapsedTicks: Ticks(20));   // 6.7 ms/token vs 15 plain
        }
        Assert.False(g.IsParked);
    }

    [Fact]
    public void AWinParksOnceTheEvidenceOfALossOutweighsIt()
    {
        var g = WinningRound();
        // The text turned into free prose: 2 tokens per 63 ms step = 31 ms/token against 15 plain. The decayed
        // evidence still holds the probe's eight 5 ms/token steps, so the verdict takes eleven of these.
        for (int i = 0; i < 10; i++)
        {
            Assert.True(g.AllowsSpeculation(), $"losing step {i}");
            g.Record(speculated: true, tokensEmitted: 2, elapsedTicks: Ticks(63));
        }
        Assert.False(g.IsParked, "ten losing steps do not yet outweigh the probe");
        g.Record(speculated: true, tokensEmitted: 2, elapsedTicks: Ticks(63));
        Assert.True(g.IsParked, "the eleventh losing step parks the hold");
        Assert.False(g.AllowsSpeculation());
        Assert.True(g.SpecMsPerToken > 15 * 1.15);
    }

    [Fact]
    public void AProvenWinSurvivesAShortUnluckyStretch()
    {
        var g = WinningRound();
        for (int i = 0; i < 40; i++)
            g.Record(speculated: true, tokensEmitted: 3, elapsedTicks: Ticks(20));   // 6.7 ms/token
        // Eight windows that accept nothing (sampling, or a hard sentence): alone they read 45 ms/token, which the
        // old eight-step window parked on.
        for (int i = 0; i < 8; i++)
        {
            Assert.True(g.AllowsSpeculation(), $"unlucky step {i}");
            g.Record(speculated: true, tokensEmitted: 1, elapsedTicks: Ticks(45));
        }
        Assert.False(g.IsParked);
        Assert.True(g.AllowsSpeculation());
    }

    [Fact]
    public void AWinThatComesInBurstsKeepsHolding()
    {
        var g = WinningRound();
        // Sampled decoding on a MoE trunk: seven windows that accept nothing, then three that accept everything,
        // over and over - 11.3 ms/token in all against 15 plain, while a run of seven bad and one good reads 17.5.
        // Steps the governor gives to plain decoding (its periodic re-check's baseline) run plain, as in the executor.
        int step = 0;
        for (int i = 0; i < 200; i++)
        {
            if (!g.AllowsSpeculation())
            {
                Assert.False(g.IsParked, $"step {i} parked");
                g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
                continue;
            }
            bool good = step++ % 10 >= 7;
            g.Record(speculated: true, tokensEmitted: good ? 7 : 1, elapsedTicks: Ticks(good ? 35 : 30));
        }
        Assert.Equal(0, g.Losses);
        Assert.Equal(0, g.ParkedSteps);
    }

    /// <summary>
    /// Steps run as the executor runs them (a step the governor declines is plain), each speculative one costing
    /// <paramref name="specMs"/> for <paramref name="specTokens"/> tokens and each plain one 15 ms: the lengths of the runs
    /// of speculative and of plain steps, in order.
    /// </summary>
    private static (List<int> Spec, List<int> Plain) Runs(SpeculationCostGovernor g, int steps, int specTokens, double specMs)
    {
        var spec = new List<int>(); var plain = new List<int>();
        bool? last = null;
        for (int i = 0; i < steps; i++)
        {
            bool speculate = g.AllowsSpeculation();
            if (speculate) g.Record(speculated: true, tokensEmitted: specTokens, elapsedTicks: Ticks(specMs));
            else g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
            var runs = speculate ? spec : plain;
            if (last != speculate) runs.Add(0);
            runs[^1]++;
            last = speculate;
        }
        return (spec, plain);
    }

    [Fact]
    public void AWideWinHoldsLongerEachTimeItIsConfirmedAndReChecksItWithOnePlainStep()
    {
        // 5 ms a token against 15 plain: wide. The first round's verdict holds 4 x 64 steps, each confirmation 4x longer up
        // to 4096; a re-check is the nine-step probe (its first sample dropped) and two plain steps (the first dropped).
        var g = WinningRound();
        var (spec, plain) = Runs(g, 6000, specTokens: 4, specMs: 20);
        Assert.Equal(new[] { 256 + 9, 1024 + 9, 4096 + 9 }, spec.Take(3).ToArray());
        Assert.Equal(new[] { 2, 2 }, plain.Take(2).ToArray());
        Assert.Equal(0, g.Losses);
    }

    [Fact]
    public void ANarrowWinKeepsTheShortHoldAndTheFullBaseline()
    {
        // 12.5 ms a token against 15 plain: a win inside the margin, not a wide one. Every round holds 64 steps and measures
        // three plain steps behind the one it drops, as before.
        var g = new SpeculationCostGovernor { Enabled = true };
        g.Reset();
        var (spec, plain) = Runs(g, 400, specTokens: 4, specMs: 50);
        Assert.Equal(9, spec[0]);
        Assert.All(spec.Skip(1).Take(3), n => Assert.Equal(64 + 9, n));
        Assert.All(plain.Take(4), n => Assert.Equal(4, n));
    }

    [Fact]
    public void AWideWinThatStopsWinningParksAndStartsTheHoldOver()
    {
        // Confirmed twice (a 4096-step hold under way), the text turns to prose: 2 tokens a 63 ms step, 31.5 ms a token. The
        // hold parks on the long-run evidence as a 64-step hold's window does (the evidence decays per sample, not per hold,
        // so the same ~33 steps outweigh a long run of 5 ms tokens), and the hold starts over: the re-probe's win, with the
        // loss still in the evidence, is not a wide one.
        var g = WinningRound();
        Runs(g, 256 + 9 + 2 + 1024 + 9 + 2 + 100, specTokens: 4, specMs: 20);
        int losing = 0;
        while (!g.IsParked && losing < 64)
        {
            Assert.True(g.AllowsSpeculation());
            g.Record(speculated: true, tokensEmitted: 2, elapsedTicks: Ticks(63));
            losing++;
        }
        Assert.True(g.IsParked, "the hold should have parked on the loss");
        Assert.InRange(losing, 8, 40);
        Assert.Equal(1, g.Losses);
        // The 32-step park (its plain steps are the baseline), a three-step re-probe that wins (its first sample dropped),
        // a 64-step hold, the next round's nine-step probe and its full baseline (one dropped, three measured); that round's
        // win is wide again and holds 4 x 64.
        var (spec, plain) = Runs(g, 32 + 4 + 64 + 9 + 4 + 256, specTokens: 4, specMs: 20);
        Assert.Equal(new[] { 32, 4 }, plain.ToArray());
        Assert.Equal(new[] { 4 + 64 + 9, 256 }, spec.ToArray());
    }

    [Fact]
    public void PlainStepsInsideAHoldAreNotCountedAgainstIt()
    {
        var g = WinningRound();
        // A head that refuses to draft below its gate produces plain steps; they
        // say nothing about speculation's cost and must not trip the window.
        for (int i = 0; i < 30; i++)
            g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
        Assert.False(g.IsParked);
        Assert.True(g.AllowsSpeculation());
    }
}

public class SpeculationCostGovernorNewSequenceTests
{
    private static long Ticks(double ms) => (long)(ms * System.Diagnostics.Stopwatch.Frequency / 1000.0);

    [Fact]
    public void ANewRequestCapsAnInheritedParkAtTheFirstInterval()
    {
        var g = new SpeculationCostGovernor { Enabled = true };
        g.Reset();
        // Two losing rounds in a row: the second park is 64 steps.
        for (int round = 0; round < 2; round++)
        {
            while (g.AllowsSpeculation())
                g.Record(speculated: true, tokensEmitted: 1, elapsedTicks: Ticks(60));
            while (!g.AllowsSpeculation() && !g.IsParked)
                g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
            Assert.True(g.IsParked, $"round {round} should end parked");
            if (round == 0)
                for (int i = 0; i < 32; i++) g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
        }
        // Parked for 64; a new request shortens that to 32 and forgets the backoff.
        g.NewSequence();
        for (int i = 0; i < 32; i++)
        {
            Assert.False(g.AllowsSpeculation(), $"parked step {i}");
            g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
        }
        Assert.True(g.AllowsSpeculation(), "the new turn re-probes after thirty-two steps");
        // That re-probe loses too: the park restarts at the first interval, not at 128.
        while (g.AllowsSpeculation())
            g.Record(speculated: true, tokensEmitted: 1, elapsedTicks: Ticks(60));
        while (!g.AllowsSpeculation() && !g.IsParked)
            g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15));
        int parked = 0;
        while (!g.AllowsSpeculation()) { g.Record(speculated: false, tokensEmitted: 1, elapsedTicks: Ticks(15)); parked++; }
        Assert.Equal(64, parked);
        Assert.Equal(3, g.Losses);
        Assert.Equal(0, g.Wins);
    }
}

public class SpeculationCostGovernorZeroAcceptTests
{
    private static long Ticks(double ms) => (long)(ms * System.Diagnostics.Stopwatch.Frequency / 1000.0);

    [Fact]
    public void AProbeThatAcceptsNothingEndsAfterFourSteps()
    {
        var g = new SpeculationCostGovernor { Enabled = true };
        g.Reset();
        // One discarded first sample, then four measured one-token speculative steps.
        for (int i = 0; i < 5; i++)
        {
            Assert.True(g.AllowsSpeculation(), $"probe step {i}");
            g.Record(speculated: true, tokensEmitted: 1, elapsedTicks: Ticks(45));
        }
        Assert.True(g.IsParked, "four measured zero-acceptance steps are a verdict");
        Assert.Equal(1, g.Losses);
    }

    [Fact]
    public void AProbeThatAcceptsSomethingRunsItsFullLength()
    {
        var g = new SpeculationCostGovernor { Enabled = true };
        g.Reset();
        for (int i = 0; i < 5; i++)
            g.Record(speculated: true, tokensEmitted: i == 2 ? 2 : 1, elapsedTicks: Ticks(45));
        Assert.False(g.IsParked);
        Assert.True(g.AllowsSpeculation(), "the probe is still measuring");
    }
}
