// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Models.Telemetry;
using TensorSharp.Runtime.Telemetry;
using Xunit;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The <c>tensorsharp.moe.*</c> instruments: routed-expert placements, the host expert kernel's time, adaptive swaps and
/// expert I/O, read from a registered source; and the native counters behind them, driven through the host expert
/// kernel itself.
/// </summary>
public sealed class MoeOffloadTelemetryTests
{
    private const string Routes = "tensorsharp.moe.expert.routes";
    private const string HostTime = "tensorsharp.moe.host.compute.time";
    private const string Swaps = "tensorsharp.moe.expert.swaps";
    private const string Io = "tensorsharp.moe.expert.io";

    [Fact]
    public void ARegisteredSource_ReportsEveryPlacementAndTotal_UnderItsArchitecture()
    {
        string architecture = "moe-telemetry-test-" + Guid.NewGuid().ToString("N");
        TensorSharpMetrics.EnsureInitialized();
        var readings = new ConcurrentQueue<(string Instrument, double Value, object Architecture, object Placement)>();
        using var meters = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == TensorSharpTelemetry.MeterName && instrument.Name is Routes or HostTime or Swaps or Io)
                    listener.EnableMeasurementEvents(instrument);
            },
        };
        static object Tag(ReadOnlySpan<KeyValuePair<string, object>> tags, string key)
        {
            foreach (var t in tags) if (t.Key == key) return t.Value;
            return null;
        }
        meters.SetMeasurementEventCallback<long>((i, v, tags, _) => readings.Enqueue((i.Name, v,
            Tag(tags, TelemetryAttributes.ModelArchitecture), Tag(tags, TelemetryAttributes.MoePlacement))));
        meters.SetMeasurementEventCallback<double>((i, v, tags, _) => readings.Enqueue((i.Name, v,
            Tag(tags, TelemetryAttributes.ModelArchitecture), Tag(tags, TelemetryAttributes.MoePlacement))));
        meters.Start();
        Assert.True(TensorSharpMetrics.MoeOffloadObserved);

        using (TensorSharpMetrics.RegisterMoeOffloadSource(architecture,
                   () => new MoeOffloadReading(DeviceRoutes: 900, StreamedRoutes: 60, HostRoutes: 40, HostComputeSeconds: 0.25, Swaps: 7, IoBytes: 1 << 20)))
        {
            meters.RecordObservableInstruments();
        }
        var mine = readings.Where(r => Equals(r.Architecture, architecture)).ToList();
        double Value(string instrument, string placement = null)
            => Assert.Single(mine, r => r.Instrument == instrument && Equals(r.Placement, placement)).Value;
        Assert.Equal(900, Value(Routes, TelemetryAttributes.MoePlacementDevice));
        Assert.Equal(60, Value(Routes, TelemetryAttributes.MoePlacementStreamed));
        Assert.Equal(40, Value(Routes, TelemetryAttributes.MoePlacementHost));
        Assert.Equal(0.25, Value(HostTime));
        Assert.Equal(7, Value(Swaps));
        Assert.Equal(1 << 20, Value(Io));

        // Unregistered: the next collection reports nothing for it.
        readings.Clear();
        meters.RecordObservableInstruments();
        Assert.DoesNotContain(readings, r => Equals(r.Architecture, architecture));
    }

    /// <summary>The host expert kernel (every offloaded MoE layer, and the direct CUDA engine's host experts) adds its
    /// computed slots and its time to the process totals the instruments report.</summary>
    [Fact]
    public unsafe void TheHostExpertKernel_CountsItsSlotsAndTime()
    {
        const int hidden = 256, ff = 256, experts = 2, used = 2;
        int rowBytes = hidden / 32 * 34, downRowBytes = ff / 32 * 34;
        byte[] Q8(int rows, int cols)
        {
            int blocks = cols / 32;
            var b = new byte[rows * blocks * 34];
            var rng = new Random(rows * 31 + cols);
            for (int i = 0; i < rows * blocks; i++)
            {
                BitConverter.TryWriteBytes(b.AsSpan(i * 34), BitConverter.HalfToUInt16Bits((System.Half)0.01f));
                for (int j = 0; j < 32; j++) b[i * 34 + 2 + j] = (byte)(sbyte)rng.Next(-60, 60);
            }
            return b;
        }
        byte[] gate = Q8(experts * ff, hidden), up = Q8(experts * ff, hidden), down = Q8(experts * hidden, ff);
        float[] x = Enumerable.Range(0, hidden).Select(i => MathF.Sin(i * 0.1f)).ToArray();
        int[] ids = { 0, 1 };
        float[] weights = { 0.6f, 0.4f };
        byte[] skip = { 0, 1 };   // the second slot is the accelerator's
        float[] output = new float[hidden];

        MoeOffloadReading before = MoeOffloadTelemetry.Read() ?? default;
        bool ok;
        fixed (byte* g = gate, u = up, d = down, s = skip)
        fixed (float* px = x, pw = weights, po = output)
        fixed (int* pi = ids)
            ok = GgmlBasicOps.HostExpertRows((IntPtr)g, (IntPtr)u, (IntPtr)d, 8, 8, 8, experts, hidden, ff, used, 1, px, pi, pw, s, po);
        Assert.True(ok, "the host expert kernel declined a Q8_0 layer");
        Assert.Contains(output, v => v != 0f);
        MoeOffloadReading after = Assert.NotNull(MoeOffloadTelemetry.Read());
        Assert.Equal(1, after.HostRoutes - before.HostRoutes);
        Assert.True(after.HostComputeSeconds > before.HostComputeSeconds);
        _ = rowBytes; _ = downRowBytes;
    }
}
