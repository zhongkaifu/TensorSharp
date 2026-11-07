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
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using OpenTelemetry.Resources;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server.Host.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The resource: TensorSharp's defaults, with the operator's OTEL_SERVICE_NAME and
/// OTEL_RESOURCE_ATTRIBUTES winning. These read the real process environment, so they run in
/// the non-parallel telemetry collection and restore what they change.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public class TelemetryResourceTests
{
    private static Dictionary<string, object> Build(TelemetryHost host)
    {
        var options = TelemetryOptions.FromEnvironment(host, n => n == TelemetryOptions.EnableEnvironmentVariable ? "1" : null);
        return TensorSharpTelemetrySetup.ConfigureResource(ResourceBuilder.CreateEmpty(), options)
            .Build().Attributes.ToDictionary(a => a.Key, a => a.Value);
    }

    private static T WithEnvironment<T>(string name, string? value, Func<T> body)
    {
        string? previous = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
        try { return body(); }
        finally { Environment.SetEnvironmentVariable(name, previous); }
    }

    [Fact]
    public void DefaultsNameTheHostAndTheVersionAndTheInstance()
    {
        var attributes = WithEnvironment("OTEL_SERVICE_NAME", null, () =>
            WithEnvironment("OTEL_RESOURCE_ATTRIBUTES", null, () => Build(TelemetryHost.Server)));
        Assert.Equal("tensorsharp-server", attributes["service.name"]);
        Assert.Equal(TensorSharpTelemetry.Version, attributes["service.version"]);
        Assert.Equal($"{Environment.MachineName}:{Environment.ProcessId}", attributes["service.instance.id"]);

        var cli = WithEnvironment("OTEL_SERVICE_NAME", null, () => Build(TelemetryHost.Cli));
        Assert.Equal("tensorsharp-cli", cli["service.name"]);
    }

    [Fact]
    public void OtelServiceNameWins()
    {
        var attributes = WithEnvironment("OTEL_SERVICE_NAME", "inference-eu", () => Build(TelemetryHost.Server));
        Assert.Equal("inference-eu", attributes["service.name"]);
    }

    [Fact]
    public void OtelResourceAttributesWin()
    {
        var attributes = WithEnvironment("OTEL_SERVICE_NAME", null, () =>
            WithEnvironment("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attributes,deployment.environment=staging",
                () => Build(TelemetryHost.Server)));
        Assert.Equal("from-attributes", attributes["service.name"]);
        Assert.Equal("staging", attributes["deployment.environment"]);
    }
}

/// <summary>Which HTTP requests get a server span.</summary>
public class ServerTelemetryFilterTests
{
    private static HttpContext Request(string path, IServiceProvider? services = null)
    {
        var context = new DefaultHttpContext();
        context.Request.Path = path;
        if (services != null) context.RequestServices = services;
        return context;
    }

    [Fact]
    public void HealthAndLowNoisePathsAreNotTracedAndEverythingElseIs()
    {
        var services = new ServiceCollection()
            .AddTensorSharpRequestLogging(o => o.LowNoisePaths.Add("/api/queue/status"))
            .BuildServiceProvider();

        Assert.False(ServerTelemetry.ShouldTrace(Request("/health", services)));
        Assert.False(ServerTelemetry.ShouldTrace(Request("/HEALTH", services)));
        Assert.False(ServerTelemetry.ShouldTrace(Request("/api/queue/status", services)));
        Assert.True(ServerTelemetry.ShouldTrace(Request("/v1/chat/completions", services)));
        Assert.True(ServerTelemetry.ShouldTrace(Request("/api/chat", services)));
        // No request logging registered (an embedder): only /health is left out.
        Assert.True(ServerTelemetry.ShouldTrace(Request("/api/queue/status", new ServiceCollection().BuildServiceProvider())));
    }
}
