// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class ConfigFileArgsEnvironmentTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-config-env-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string, string?> _original = new();

    public ConfigFileArgsEnvironmentTests() => Directory.CreateDirectory(_dir);
    public void Dispose()
    {
        foreach (var entry in _original) Environment.SetEnvironmentVariable(entry.Key, entry.Value);
        Directory.Delete(_dir, recursive: true);
    }
    private string Key() => "TS_CONFIG_TEST_" + Guid.NewGuid().ToString("N");
    private void Set(string key, string? value)
    {
        _original.TryAdd(key, Environment.GetEnvironmentVariable(key));
        Environment.SetEnvironmentVariable(key, value);
    }
    private string Config(object value, string name = "config.json")
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, JsonSerializer.Serialize(value));
        return path;
    }
    private string Defaults(string key, object? value, string name = "config.json")
        => Config(new { environment = new Dictionary<string, object?> { [key] = value } }, name);

    [Fact]
    public void Environment_IsReserved_AndExpandsVariablesWithoutEmittingArguments()
    {
        string key = Key(); Set(key, null);
        string file = Config(new
        {
            variables = new { root = "${appBaseDirectory}/cuda_kernels" },
            environment = new Dictionary<string, string> { [key] = "${root}/draft.ptx" }
        });
        Assert.Empty(ConfigFileArgs.Expand(["--config", file]));
        Assert.Equal(Path.Combine(AppContext.BaseDirectory, "cuda_kernels", "draft.ptx"),
            Path.GetFullPath(Environment.GetEnvironmentVariable(key)!));
    }

    /// <summary>Many of these switches are read by the native library with getenv, and on Unix
    /// .NET keeps its own copy of the environment: a default published only there never reached
    /// the C runtime (config/qwen3.8-flash-next.json's TS_Q4E_VERIFY_ROW_KERNELS was ignored).</summary>
    [Fact]
    public void Defaults_ReachTheNativeEnvironment()
    {
        if (OperatingSystem.IsWindows()) return;
        string key = Key(); Set(key, null);
        try
        {
            ConfigFileArgs.Expand(["--config", Defaults(key, "native")]);
            Assert.Equal("native", System.Runtime.InteropServices.Marshal.PtrToStringAnsi(NativeGetEnv(key)));
        }
        finally { NativeEnvironment.Set(key, null); }
    }

    [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "getenv")]
    private static extern IntPtr NativeGetEnv(string name);

    [Fact]
    public void LaterFileWinsDefaults_ButExistingProcessValueWinsBothFiles()
    {
        string key = Key(); Set(key, null);
        string first = Defaults(key, "first", "first.json"), second = Defaults(key, "second", "second.json");
        ConfigFileArgs.Expand(["--config", first, "--config", second]);
        Assert.Equal("second", Environment.GetEnvironmentVariable(key));
        Set(key, "process");
        ConfigFileArgs.Expand(["--config", first, "--config", second]);
        Assert.Equal("process", Environment.GetEnvironmentVariable(key));
    }

    [Theory]
    [InlineData(123)]
    [InlineData(true)]
    [InlineData(null)]
    public void NonStringValue_RejectsBeforePublishingAnyDefault(object? invalid)
    {
        string key = Key(); Set(key, null);
        string first = Defaults(key, "must-not-leak", "first.json"), bad = Defaults(Key(), invalid, "bad.json");
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", first, "--config", bad]));
        Assert.Null(Environment.GetEnvironmentVariable(key));
    }

    [Fact]
    public void EnvironmentMustBeAnObject_AndArraysAreNotStringValues()
    {
        foreach (object? invalid in new object?[] { null, "value", 1, new[] { "value" } })
        {
            string file = Config(new { environment = invalid });
            Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", file]));
        }
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", Defaults(Key(), new[] { "value" })]));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("X=Y")]
    [InlineData("X\0Y")]
    [InlineData("X\nY")]
    public void InvalidName_RejectsBeforePublishingEarlierValidEntry(string invalid)
    {
        string key = Key(); Set(key, null);
        string file = Config(new { environment = new Dictionary<string, string> { [key] = "must-not-leak", [invalid] = "bad" } });
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", file]));
        Assert.Null(Environment.GetEnvironmentVariable(key));
    }

    [Fact]
    public void InvalidValueOrVariableOrOption_DoesNotPublishDefaults()
    {
        string key = Key(); Set(key, null);
        foreach (string invalid in new[] { "contains\0nul", "${undefined_config_variable}" })
        {
            Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", Defaults(key, invalid)]));
            Assert.Null(Environment.GetEnvironmentVariable(key));
        }
        string file = Config(new Dictionary<string, object>
        {
            ["environment"] = new Dictionary<string, string> { [key] = "must-not-leak" },
            ["temperature"] = new[] { new[] { 1 } }
        });
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", file]));
        Assert.Null(Environment.GetEnvironmentVariable(key));
    }

    [Fact]
    public void PreflightSeesNoMutation_AndItsExplicitProcessValueWins()
    {
        string key = Key(); Set(key, null);
        string file = Defaults(key, "default");
        Assert.Empty(ConfigFileArgs.Expand(["--config", file], _ =>
        {
            Assert.Null(Environment.GetEnvironmentVariable(key));
            Set(key, "host-preflight");
        }));
        Assert.Equal("host-preflight", Environment.GetEnvironmentVariable(key));
    }

    [Fact]
    public void PreflightFailure_AndMalformedLaterFile_DoNotPublishDefaults()
    {
        string key = Key(); Set(key, null);
        string file = Defaults(key, "must-not-leak");
        Assert.Throws<InvalidOperationException>(() => ConfigFileArgs.Expand(["--config", file], _ => throw new InvalidOperationException("refused")));
        Assert.Null(Environment.GetEnvironmentVariable(key));
        string bad = Path.Combine(_dir, "bad.json"); File.WriteAllText(bad, "{");
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", file, "--config", bad]));
        Assert.Null(Environment.GetEnvironmentVariable(key));
    }

    [Fact]
    public void AppBaseDirectory_CannotBeReplacedByConfigOrProcessVariable()
    {
        Set("appBaseDirectory", "foreign");
        Assert.Equal(["--model", AppContext.BaseDirectory], ConfigFileArgs.Expand(["--config", Config(new { model = "${appBaseDirectory}" })]));
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", Config(new { variables = new { appBaseDirectory = "foreign" } })]));
    }
}
