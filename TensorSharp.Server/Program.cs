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
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Server.ProtocolAdapters;

const string ListenAddress = "http://0.0.0.0:5000";
const long MaxRequestBodyBytes = 500L * 1024L * 1024L;

string baseDirectory = AppContext.BaseDirectory;
ServerHostingOptions hostingOptions = ServerOptionsBuilder.Build(args, baseDirectory);
LogLevel resolvedLogLevel = LoggingSetup.ResolveMinimumLevel();
string configuredBackendInput = ServerOptionsBuilder.ReadConfiguredBackendInput(args);

var builder = WebApplication.CreateBuilder(args);
LoggingSetup.Configure(builder.Logging, hostingOptions, resolvedLogLevel);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = MaxRequestBodyBytes;
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = MaxRequestBodyBytes;
});

builder.Services.AddSingleton(hostingOptions);
builder.Services.AddSingleton<ModelService>();
builder.Services.AddSingleton<InferenceQueue>();
builder.Services.AddSingleton<SessionManager>();

// Demote the high-frequency status-polling endpoints to Debug so the
// default Information-level log isn't dominated by their request entries.
// Set TENSORSHARP_LOG_LEVEL=Debug to see them when troubleshooting.
builder.Services.AddTensorSharpRequestLogging(options =>
{
    options.LowNoisePaths.Add("/api/queue/status");
});

// One adapter per protocol; instances are stateless and free to share between requests.
builder.Services.AddSingleton<WebUiAdapter>();
builder.Services.AddSingleton<OllamaAdapter>();
builder.Services.AddSingleton<OpenAIChatAdapter>();

WebRootSetup.Resolve(builder.Environment, baseDirectory);

var app = builder.Build();

ILogger startupLogger = app.Services.GetRequiredService<ILoggerFactory>()
    .CreateLogger("TensorSharp.Server.Startup");
startupLogger.LogInformation(LogEventIds.LoggingInitialized,
    "Logging initialized: minimumLevel={MinimumLevel} fileLogging={FileLogging} logDir={LogDir}",
    resolvedLogLevel, hostingOptions.FileLoggingEnabled,
    hostingOptions.FileLoggingEnabled ? hostingOptions.LogDirectory : "(disabled)");

StartupBanner.EmitBackendFallback(startupLogger, hostingOptions, configuredBackendInput);

app.UseTensorSharpRequestLogging();
app.UseStaticFiles();
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(hostingOptions.UploadDirectory),
    RequestPath = "/uploads",
});

app.MapHealthEndpoints(app.Environment);
app.MapSessionEndpoints();
app.MapUploadEndpoints();
app.MapWebUiEndpoints();
app.MapOllamaEndpoints();
app.MapOpenAIEndpoints();

StartupModelLoader.LoadIfConfigured(
    hostingOptions,
    app.Services.GetRequiredService<ModelService>(),
    configuredBackendInput,
    startupLogger);

StartupBanner.Emit(startupLogger, hostingOptions, ListenAddress);

app.Run(ListenAddress);
