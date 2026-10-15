using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

public sealed class PortablePdfProcessRunnerTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "pdf-process-" + Guid.NewGuid().ToString("N"));

    public PortablePdfProcessRunnerTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public async Task ArgumentsRemainLiteralWithoutShellExpansion()
    {
        string[] arguments = { "path with spaces.pdf", "\"quoted\"", "$(touch never-created)", "x;y&z", "中文扫描", "--flag" };
        var result = await RunAsync(new[] { "arguments" }.Concat(arguments).ToArray());
        Assert.Equal(0, result.ExitCode);
        Assert.Equal(arguments, JsonSerializer.Deserialize<string[]>(result.StandardOutput));
        Assert.False(File.Exists(Path.Combine(_directory, "never-created")));
    }

    [Fact]
    public async Task ConcurrentLargeInputAndBothOutputStreamsDoNotDeadlock()
    {
        byte[] input = new byte[1024 * 1024];
        new Random(742).NextBytes(input);
        var result = await RunAsync(new[] { "duplex" }, maxOutputBytes: input.Length, input: input);
        Assert.Equal(0, result.ExitCode);
        Assert.Equal(input, result.StandardOutput);
        Assert.True(result.ErrorTruncated);
        Assert.Equal(PortablePdfProcessRunner.MaximumErrorBytes, Encoding.UTF8.GetByteCount(result.StandardError));
    }

    [Fact]
    public async Task OutputLimitStopsAndReapsTheWriter()
    {
        string pidPath = Path.Combine(_directory, "pid");
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => RunAsync(
            new[] { "overflow", pidPath, "1000000" }, maxOutputBytes: 1024));
        Assert.NotEmpty(await File.ReadAllTextAsync(pidPath));
        Assert.False(IsRunning(int.Parse(await File.ReadAllTextAsync(pidPath))));
    }

    [Fact]
    public async Task CancellationStopsParentAndDescendant()
    {
        string pidPath = Path.Combine(_directory, "pids");
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        Task run = RunAsync(new[] { "tree", pidPath }, ct: cancellation.Token);
        int[] ids = Array.Empty<int>();
        try
        {
            for (int attempt = 0; attempt < 200 && !File.Exists(pidPath); attempt++)
                await Task.Delay(25);
            Assert.True(File.Exists(pidPath), "Fixture process did not initialize.");
            ids = (await File.ReadAllTextAsync(pidPath)).Split(',').Select(int.Parse).ToArray();
            Assert.Equal(2, ids.Length);
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
            for (int attempt = 0; attempt < 100 && ids.Any(IsRunning); attempt++)
                await Task.Delay(20);
            Assert.All(ids, id => Assert.False(IsRunning(id), $"Process {id} survived cancellation."));
        }
        finally
        {
            cancellation.Cancel();
            foreach (int id in ids)
                try { using var process = Process.GetProcessById(id); process.Kill(entireProcessTree: true); }
                catch (ArgumentException) { }
                catch (InvalidOperationException) { }
        }
    }

    [Fact]
    public async Task RejectedInputPreservesDiagnosticExitStatus()
    {
        var result = await RunAsync(new[] { "reject" }, input: new byte[2 * 1024 * 1024]);
        Assert.Equal(17, result.ExitCode);
        Assert.Equal("fixture rejected request", result.StandardError);
    }

    [Fact]
    public async Task InvalidDiagnosticUtf8IsReplacedWithoutBreakingExtraction()
    {
        var result = await RunAsync(new[] { "invalid-utf8" });
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("\uFFFD\uFFFDa", result.StandardError);
        Assert.False(result.ErrorTruncated);
    }

    [Fact]
    public async Task ConfiguredEnvironmentIsPassedWithoutMutatingParent()
    {
        string name = "TS_PDF_PROCESS_TEST_" + Guid.NewGuid().ToString("N");
        var result = await RunAsync(new[] { "environment", name }, environment: new Dictionary<string, string> { [name] = "a value with spaces" });
        Assert.Equal("a value with spaces", Encoding.UTF8.GetString(result.StandardOutput));
        Assert.Null(Environment.GetEnvironmentVariable(name));
    }

    [Fact]
    public async Task PreCanceledCallDoesNotLaunchChild()
    {
        string pidPath = Path.Combine(_directory, "must-not-exist");
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => RunAsync(new[] { "overflow", pidPath, "10" }, ct: cancellation.Token));
        Assert.False(File.Exists(pidPath));
    }

    private async Task<PortablePdfProcessResult> RunAsync(string[] arguments, int maxOutputBytes = 65536,
        ReadOnlyMemory<byte> input = default, CancellationToken ct = default,
        IReadOnlyDictionary<string, string>? environment = null)
    {
        string fixture = Path.Combine(AppContext.BaseDirectory, "PdfProcessFixture", "PdfProcessFixture.dll");
        Assert.True(File.Exists(fixture), "The test fixture must be built and copied before test execution.");
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(TimeSpan.FromSeconds(20));
        string host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ??
            Path.GetFullPath(Path.Combine(RuntimeEnvironment.GetRuntimeDirectory(), "..", "..", "..", OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet"));
        return await PortablePdfProcessRunner.RunAsync(host, new[] { fixture }.Concat(arguments).ToArray(),
            _directory, maxOutputBytes, deadline.Token, input, environment);
    }

    private static bool IsRunning(int id)
    {
        try { using var process = Process.GetProcessById(id); return !process.HasExited; }
        catch (ArgumentException) { return false; }
    }

    public void Dispose() => PortablePdfScratchDirectory.Delete(_directory);
}
