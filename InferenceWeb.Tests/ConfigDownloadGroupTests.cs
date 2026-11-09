using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class ConfigDownloadGroupTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-config-group-" + Guid.NewGuid().ToString("N"));

    public ConfigDownloadGroupTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void Expand_CachedPrimaryStillDownloadsShards_EmitsOnlyPrimary_AndReusesEachFile()
    {
        using var server = new FileServer();
        var shards = new[] { "model-00001-of-00003.gguf", "model-00002-of-00003.gguf", "model-00003-of-00003.gguf" };
        File.WriteAllBytes(Path.Combine(_directory, shards[0]), FileServer.Payload);
        var config = new
        {
            variables = new { root = ".", source = server.Url },
            model = new
            {
                path = "${root}/" + shards[0],
                url = "${source}/" + shards[0],
                files = shards.Skip(1).Select(name => new
                {
                    path = "${root}/" + name,
                    url = "${source}/" + name,
                    sha256 = Convert.ToHexStringLower(SHA256.HashData(FileServer.Payload))
                })
            }
        };
        string path = Write(config);

        string[] expanded = ConfigFileArgs.Expand(["--config", path], TextWriter.Null, false);

        Assert.Equal(new[] { "--model", Path.Combine(_directory, shards[0]) }, expanded);
        Assert.Equal(0, server.RequestCount(shards[0]));
        foreach (string shard in shards.Skip(1))
        {
            Assert.Equal(FileServer.Payload, File.ReadAllBytes(Path.Combine(_directory, shard)));
            Assert.Equal(1, server.RequestCount(shard));
        }

        ConfigFileArgs.Expand(["--config", path], TextWriter.Null, false);
        Assert.All(shards.Skip(1), name => Assert.Equal(1, server.RequestCount(name)));
        File.Delete(Path.Combine(_directory, shards[2]));
        Assert.Equal(Path.Combine(_directory, shards[0]), ConfigFileArgs.ResolveFileEntry(path, "model"));
        Assert.Equal(1, server.RequestCount(shards[1]));
        Assert.Equal(2, server.RequestCount(shards[2]));
    }

    [Fact]
    public void Expand_BadShardChecksumRefusesGroup_AndDoesNotPublishPartialFile()
    {
        using var server = new FileServer();
        File.WriteAllText(Path.Combine(_directory, "first.gguf"), "cached");
        string path = Write(new
        {
            model = new
            {
                path = "first.gguf",
                files = new[] { new { path = "second.gguf", url = server.Url + "/second.gguf", sha256 = new string('0', 64) } }
            }
        });

        IOException error = Assert.Throws<IOException>(() => ConfigFileArgs.Expand(["--config", path], TextWriter.Null, false));

        Assert.Contains("SHA-256 mismatch", error.Message);
        Assert.Contains("model.files[0]", error.Message);
        Assert.True(File.Exists(Path.Combine(_directory, "first.gguf")));
        Assert.False(File.Exists(Path.Combine(_directory, "second.gguf")));
        Assert.False(File.Exists(Path.Combine(_directory, "second.gguf.part")));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[\"second.gguf\"]")]
    [InlineData("[{\"url\": \"https://example.invalid/second.gguf\"}]")]
    public void Expand_InvalidShardManifestIsRefusedBeforePrimaryDownload(string files)
    {
        using var server = new FileServer();
        string path = Path.Combine(_directory, "invalid.json");
        File.WriteAllText(path, $$"""
            { "model": { "path": "first.gguf", "url": "{{server.Url}}/first.gguf", "files": {{files}} } }
            """);

        Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(["--config", path], TextWriter.Null, false));
        Assert.Equal(0, server.RequestCount("first.gguf"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Expand_ModelOverrideSkipsEntireGroupIncludingUnresolvedVariables(bool laterConfig)
    {
        string path = Write(new
        {
            model = new
            {
                path = "unused.gguf",
                files = new[] { new { path = "${TS_GROUP_UNDEFINED}/second.gguf", url = "https://example.invalid/second.gguf" } }
            }
        });
        string[] args = laterConfig
            ? ["--config", path, "--config", Write(new { model = "override.gguf" }, "later.json")]
            : ["--config", path, "--model", "override.gguf"];

        Assert.Equal(new[] { "--model", "override.gguf" }, ConfigFileArgs.Expand(args, TextWriter.Null, false));
    }

    /// <summary>The Flash Next preset is the fastest measured setup on Apple Silicon: ISTA-DASLab's GSQ-RCO Q2_0 (two
    /// shards), the DFlash2 drafter and the validated projector, on ggml_metal, with its environment (MAX_CONTEXT and
    /// the Q8_0 hyper-connection copy) reaching native getenv. The UD-Q2_K_XL preset keeps the MTP head.</summary>
    [Fact]
    public void FlashNextPreset_ResolvesItsShardsProjectorAndDrafter_ThroughServerParser()
    {
        DirectoryInfo? repo = new(AppContext.BaseDirectory);
        while (repo != null && !File.Exists(Path.Combine(repo.FullName, "TensorSharp.slnx"))) repo = repo.Parent;
        Assert.NotNull(repo);
        var parse = (string name) => JsonNode.Parse(File.ReadAllText(Path.Combine(repo.FullName, "config", name)),
            documentOptions: new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true })!;
        var config = parse("qwen3.8-flash-next.json");
        // Resolve the real manifest locally, without downloading the multi-gigabyte artifacts.
        config["variables"]!["modelRoot"] = _directory;
        var modelFiles = new[] { config["model"]! }.Concat(config["model"]!["files"]!.AsArray().Select(node => node!)).ToArray();
        Assert.Equal(2, modelFiles.Length);
        string Local(JsonNode entry) => entry["path"]!.GetValue<string>()
            .Replace("${root}", Path.Combine(_directory, "qwen3.8-flash-next-ista-q2_0"))
            .Replace("${draftRoot}", Path.Combine(_directory, "qwen3.8-flash-next-dflash2"))
            .Replace("${projectorRoot}", Path.Combine(_directory, "qwen3.8-flash-next"));
        foreach (JsonNode entry in modelFiles.Append(config["mmproj"]!).Append(config["draft-model"]!))
        {
            string local = Local(entry);
            Directory.CreateDirectory(Path.GetDirectoryName(local)!);
            File.WriteAllText(local, "cached fixture");
        }

        string[] variables = { "MAX_CONTEXT", "TS_Q4E_HC_Q8", "TS_Q4E_VERIFY_ROW_KERNELS" };
        string?[] before = variables.Select(Environment.GetEnvironmentVariable).ToArray();
        foreach (string name in variables) NativeEnvironment.Set(name, null);
        try
        {
            string path = Write(config);
            string[] expanded = ConfigFileArgs.Expand(["--config", path], TextWriter.Null, false);
            Assert.Single(expanded, value => value == "--model");
            Assert.EndsWith("GSQ-RCO-Q2_0-00001-of-00002.gguf", expanded[Array.IndexOf(expanded, "--model") + 1]);
            Assert.Equal(Local(config["mmproj"]!), expanded[Array.IndexOf(expanded, "--mmproj") + 1]);
            // Naming the DFlash2 drafter is the request to speculate; its block drafts are the engine's defaults.
            Assert.Equal(Local(config["draft-model"]!), expanded[Array.IndexOf(expanded, "--draft-model") + 1]);
            Assert.DoesNotContain("--spec-draft", expanded);
            Assert.Equal("ggml_metal", expanded[Array.IndexOf(expanded, "--backend") + 1]);
            Assert.Equal("32768", Environment.GetEnvironmentVariable("MAX_CONTEXT"));
            Assert.Equal("1", Environment.GetEnvironmentVariable("TS_Q4E_HC_Q8"));
            // Batched verify is the ggml_metal default; the preset does not set the switch.
            Assert.Null(Environment.GetEnvironmentVariable("TS_Q4E_VERIFY_ROW_KERNELS"));
            var options = ServerOptionsBuilder.Build(expanded, _directory);
            Assert.Equal(1.0f, options.DefaultSamplingConfig.Temperature);

            // The UD-Q2_K_XL preset is the previous one: three shards and the shared MTP head, three drafts.
            var udq2 = parse("qwen3.8-flash-next-ud-q2_k_xl.json");
            Assert.Equal(2, udq2["model"]!["files"]!.AsArray().Count);
            Assert.EndsWith("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", udq2["draft-model"]!["path"]!.GetValue<string>());
            Assert.Equal(3, udq2["spec-draft"]!.GetValue<int>());
        }
        finally
        {
            // Never leave the preset's environment to the other tests in this process.
            for (int i = 0; i < variables.Length; i++) NativeEnvironment.Set(variables[i], before[i]);
        }
    }

    private string Write(object config, string filename = "config.json")
    {
        string path = Path.Combine(_directory, filename);
        File.WriteAllText(path, JsonSerializer.Serialize(config));
        return path;
    }

    private sealed class FileServer : IDisposable
    {
        internal static readonly byte[] Payload = Encoding.UTF8.GetBytes("small local test shard");
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        private readonly ConcurrentDictionary<string, int> _requests = new();
        private readonly Task _worker;
        private volatile bool _stopped;
        public string Url { get; }

        public FileServer()
        {
            _listener.Start();
            Url = $"http://127.0.0.1:{((IPEndPoint)_listener.LocalEndpoint).Port}";
            _worker = Task.Run(async () =>
            {
                try
                {
                    while (!_stopped)
                    {
                        using TcpClient client = await _listener.AcceptTcpClientAsync();
                        using NetworkStream stream = client.GetStream();
                        using var reader = new StreamReader(stream, Encoding.ASCII, leaveOpen: true);
                        string? request = await reader.ReadLineAsync();
                        while (!string.IsNullOrEmpty(await reader.ReadLineAsync())) { }
                        string name = request!.Split(' ')[1].TrimStart('/');
                        _requests.AddOrUpdate(name, 1, (_, count) => count + 1);
                        byte[] headers = Encoding.ASCII.GetBytes($"HTTP/1.1 200 OK\r\nContent-Length: {Payload.Length}\r\nConnection: close\r\n\r\n");
                        await stream.WriteAsync(headers);
                        await stream.WriteAsync(Payload);
                    }
                }
                catch (SocketException) { }
                catch (ObjectDisposedException) { }
                catch (InvalidOperationException) when (_stopped) { }
            });
        }

        public int RequestCount(string filename) => _requests.GetValueOrDefault(filename);
        public void Dispose()
        {
            _stopped = true;
            _listener.Stop();
            _worker.GetAwaiter().GetResult();
        }
    }
}
