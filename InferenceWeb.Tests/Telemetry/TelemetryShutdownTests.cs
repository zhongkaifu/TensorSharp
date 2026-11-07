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
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using TensorSharp.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The flush when the host stops (<see cref="TensorSharpTelemetryServiceCollectionExtensions.StartTensorSharpTelemetry"/>).
/// The engine and KV-cache gauges read only live engines, and dependency injection disposes
/// the models (and with them the engine) before the meter provider, whose own flush therefore
/// never sees them. Each test uses its own meter, and no process-wide switch is touched.
/// </summary>
public sealed class TelemetryShutdownTests
{
    private const string GaugeName = "tensorsharp.test.engine.requests.running";

    [Fact]
    public async Task StoppingTheHostExportsGaugesBeforeTheServicesTheyReadAreDisposed()
    {
        var run = await RunHost(registerStopFlush: true);

        // Exported by the time StopAsync returned, while the engine stand-in was alive.
        Assert.Contains(run.ExportedWhenStopped, p => p.Name == GaugeName && p.Value == 1);
        Assert.True(run.EngineDisposed);
    }

    [Fact]
    public async Task WithoutTheStopFlushTheGaugeOfADisposedEngineIsNeverExported()
    {
        // Control: what happened before the stop flush. The provider's flush at disposal runs
        // after the engine stand-in was disposed, so the gauge has nothing to report.
        var run = await RunHost(registerStopFlush: false);

        Assert.DoesNotContain(run.ExportedWhenStopped, p => p.Name == GaugeName);
        Assert.DoesNotContain(run.ExportedAtTheEnd, p => p.Name == GaugeName);
        Assert.True(run.EngineDisposed);
    }

    [Fact]
    public void FlushWithNothingToFlushReturnsAtOnce()
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        TensorSharpTelemetrySetup.Flush(meter: null, tracer: null, logger: null, timeoutMilliseconds: 5000);
        Assert.True(watch.ElapsedMilliseconds < 1000);
    }

    [Fact]
    public void AStalledExporterDoesNotHoldTheStopFlushPastItsDeadline()
    {
        // A meter provider exports on the flushing thread and checks its timeout only before
        // the export starts; the flush must still return on time.
        using var meter = new Meter("TensorSharp.Tests.StopFlush.Stalled." + Guid.NewGuid().ToString("N"));
        meter.CreateCounter<long>("tensorsharp.test.counter").Add(1);
        using var release = new System.Threading.ManualResetEventSlim(false);
        var exporter = new StallingMetricExporter(release);
        using var provider = OpenTelemetry.Sdk.CreateMeterProviderBuilder()
            .AddMeter(meter.Name)
            .AddReader(new BaseExportingMetricReader(exporter))
            .Build();
        meter.CreateCounter<long>("tensorsharp.test.counter2").Add(1);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        bool finished = TensorSharpTelemetrySetup.Flush(provider, tracer: null, logger: null, timeoutMilliseconds: 300);
        watch.Stop();
        release.Set();

        Assert.False(finished);
        Assert.True(exporter.Entered.Wait(TimeSpan.FromSeconds(5)), "the export never started");
        Assert.True(watch.ElapsedMilliseconds < 3000, $"the flush returned after {watch.ElapsedMilliseconds} ms");
    }

    private sealed class StallingMetricExporter : BaseExporter<Metric>
    {
        private readonly System.Threading.ManualResetEventSlim _release;

        public StallingMetricExporter(System.Threading.ManualResetEventSlim release) => _release = release;

        public System.Threading.ManualResetEventSlim Entered { get; } = new(false);

        public override ExportResult Export(in Batch<Metric> batch)
        {
            Entered.Set();
            // Stalls like an unreachable endpoint, until the test lets it go.
            _release.Wait(TimeSpan.FromSeconds(30));
            return ExportResult.Success;
        }
    }

    private sealed record Run(List<(string Name, long Value)> ExportedWhenStopped,
        List<(string Name, long Value)> ExportedAtTheEnd, bool EngineDisposed);

    private static async Task<Run> RunHost(bool registerStopFlush)
    {
        using var meter = new Meter("TensorSharp.Tests.StopFlush." + Guid.NewGuid().ToString("N"));
        var engine = new EngineStandIn();
        meter.CreateObservableGauge(GaugeName,
            () => engine.Alive ? new[] { new Measurement<long>(1) } : Array.Empty<Measurement<long>>());

        var exporter = new CapturingMetricExporter();
        var builder = Host.CreateEmptyApplicationBuilder(new HostApplicationBuilderSettings());
        builder.Services.AddOpenTelemetry().WithMetrics(m => m
            .AddMeter(meter.Name)
            // No periodic export: only a flush collects, so what is exported says which flush ran.
            .AddReader(new BaseExportingMetricReader(exporter)));
        // A factory, not the instance: the container disposes only what it created, and it
        // creates the server's model services.
        builder.Services.AddSingleton(_ => engine);

        var host = builder.Build();
        if (registerStopFlush)
            host.Services.StartTensorSharpTelemetry();
        else
            _ = host.Services.GetService<MeterProvider>();
        // Resolved after the providers, as the server's models are, so disposed before them.
        _ = host.Services.GetRequiredService<EngineStandIn>();

        await host.StartAsync();
        await host.StopAsync();
        var whenStopped = exporter.Snapshot();
        host.Dispose();
        return new Run(whenStopped, exporter.Snapshot(), !engine.Alive);
    }

    private sealed class EngineStandIn : IDisposable
    {
        public volatile bool Alive = true;
        public void Dispose() => Alive = false;
    }

    private sealed class CapturingMetricExporter : BaseExporter<Metric>
    {
        private readonly List<(string Name, long Value)> _points = new();

        public List<(string Name, long Value)> Snapshot()
        {
            lock (_points)
                return new List<(string Name, long Value)>(_points);
        }

        public override ExportResult Export(in Batch<Metric> batch)
        {
            lock (_points)
            {
                foreach (var metric in batch)
                {
                    foreach (ref readonly var point in metric.GetMetricPoints())
                        _points.Add((metric.Name, metric.MetricType == MetricType.LongGauge ? point.GetGaugeLastValueLong() : 0));
                }
            }
            return ExportResult.Success;
        }
    }
}
