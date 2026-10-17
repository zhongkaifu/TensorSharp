// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

/// <summary>
/// The real host processes, launched the way an operator launches them, with a model
/// the load refuses. Every refusal used to leave through an unhandled exception: the
/// stack trace was printed after the refusal and the runtime called abort(), exit code
/// 134 (the campaign logs of DeepSeek V4 on too few GPUs, V4.1 with KV_CACHE_DTYPE=q8_0,
/// and GLM with --tp). The contract is one line on stderr and exit code 2.
/// </summary>
/// <remarks>
/// Deliberately written against the process boundary only (exit code, stdout, stderr)
/// and not against any type the fix introduced, so the same file runs against a build
/// from before the fix and fails there.
/// </remarks>
public class HostLoadRefusalProcessTests : IDisposable
{
    /// <summary>USAGE.md "Exit codes": the model load was refused.</summary>
    private const int ModelLoadRefusedExitCode = 2;

    /// <summary>USAGE.md "Exit codes": a command-line or configuration-file mistake.</summary>
    private const int ConfigurationErrorExitCode = 1;

    private const string ErrorPrefix = "error: model load refused: ";

    private readonly string _dir;

    public HostLoadRefusalProcessTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-load-refusal-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Server_MissingModel_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        string missing = Path.Combine(_dir, "missing.gguf");
        HostRun run = RunServer(missing);

        AssertRefused(run, "missing.gguf");
    }

    [Fact]
    public void Server_FileThatIsNotAGguf_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        HostRun run = RunServer(WriteJunkModel());

        AssertRefused(run, "Not a GGUF file");
        // Refused, not crashed: no stack trace anywhere at the default log level.
        Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("   at ", run.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void Server_DebugLogLevel_StillOneStderrLine_ButTheStackTraceIsLogged()
    {
        HostRun run = RunServer(WriteJunkModel(), logLevel: "Debug");

        AssertRefused(run, "Not a GGUF file");
        Assert.Contains("   at ", run.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void Cli_MissingModel_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        HostRun run = RunCli(Path.Combine(_dir, "missing.gguf"));

        // This one used to exit 0 after printing the usage lines, so a script could not
        // tell a missing model from a completed run.
        AssertRefused(run, "missing.gguf");
    }

    [Fact]
    public void Cli_FileThatIsNotAGguf_ExitsWithTheRefusalCodeAndOneStderrLine()
    {
        HostRun run = RunCli(WriteJunkModel());

        AssertRefused(run, "Not a GGUF file");
        Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
    }

    public static IEnumerable<object[]> InvalidParallelismLines() => new[]
    {
        new object[] { new[] { "--layer-split" }, "--layer-split" },
        new object[] { new[] { "--layer-split=0" }, "--layer-split" },
        new object[] { new[] { "--tp=two" }, "--tp" },
        new object[] { new[] { "--tp=2", "--layer-split=2" }, "cannot both" },
        new object[] { new[] { "--tp-node-id=0" }, "--tp-peers" },
        new object[] { new[] { "--layer-split=2", "--tp-node-id=0",
            "--tp-peers=127.0.0.1:9500,127.0.0.1:9501" }, "one node only" },
    };

    [Theory]
    [MemberData(nameof(InvalidParallelismLines))]
    public void BothHosts_RejectInvalidParallelismBeforeLoadingOrConnecting(string[] args, string reason)
    {
        string missing = Path.Combine(_dir, "must-not-be-loaded.gguf");
        foreach (HostRun run in new[] { RunCliWith(missing, args), RunServerWith(missing, args) })
        {
            Assert.Equal(ConfigurationErrorExitCode, run.ExitCode);
            Assert.Contains("Configuration error:", run.Stderr);
            Assert.Contains(reason, run.Stderr);
            Assert.DoesNotContain("Unhandled exception", run.Stderr);
            Assert.DoesNotContain(ErrorPrefix, run.Stderr);
        }
    }

    // ---- malformed values and removed variables ---------------------------------------
    // The server applies most option families after Build, and those appliers ran outside
    // its configuration-error handler: a malformed --kv-cache-dtype, --spec-draft,
    // --n-cpu-moe, --gpu-device or --prefill-chunk-size, a missing --draft-model file, and a
    // removed speculation variable (TS_DSV4_DSPARK, TS_MTP_SPEC ...) each aborted it with a
    // stack trace, exit code 134. The CLI reported every one of them already.

    public static IEnumerable<object[]> MalformedOptionLines() => new[]
    {
        new object[] { new[] { "--kv-cache-dtype", "bogus" }, "--kv-cache-dtype", true },
        new object[] { new[] { "--spec-draft", "abc" }, "--spec-draft", true },
        new object[] { new[] { "--n-cpu-moe", "abc" }, "--n-cpu-moe", true },
        new object[] { new[] { "--gpu-device", "abc" }, "--gpu-device", true },
        new object[] { new[] { "--draft-model", "missing-drafter.gguf" }, "--draft-model", true },
        new object[] { new[] { "--prefill-chunk-size", "abc" }, "--prefill-chunk-size", false },   // server only
    };

    [Theory]
    [MemberData(nameof(MalformedOptionLines))]
    public void Hosts_MalformedOptionValue_IsAConfigurationError(string[] args, string option, bool cliToo)
    {
        string missing = Path.Combine(_dir, "must-not-be-loaded.gguf");
        AssertOneConfigurationErrorLine(RunServerWith(missing, args), option);
        if (cliToo)
            AssertOneConfigurationErrorLine(RunCliWith(missing, args), option);
    }

    [Theory]
    [InlineData("TS_ENCODER_YIELD", "0", "was removed:")]                                   // RemovedCliFlags
    [InlineData("TS_DSV4_DSPARK", "drafter.gguf", "was removed; set TS_SPEC_DRAFT_MODEL instead")] // SpeculationEnvVars
    [InlineData("TS_MTP_SPEC", "1", "was removed; set TS_SPEC instead")]
    public void BothHosts_RemovedEnvironmentVariable_IsAConfigurationError(string name, string value, string advice)
    {
        string missing = Path.Combine(_dir, "must-not-be-loaded.gguf");
        var environment = new Dictionary<string, string> { [name] = value };
        foreach (HostRun run in new[] { RunServerWith(missing, Array.Empty<string>(), environment: environment),
                                         RunCliWith(missing, Array.Empty<string>(), environment) })
        {
            string line = AssertOneConfigurationErrorLine(run, name);
            Assert.Contains(advice, line, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void BothHosts_RefuseLocalOnlyArchitectureWithoutWaitingForPeers()
    {
        // The architecture is refused before the backend is judged, so a backend every machine has keeps the
        // hosts' own backend selection (which rejects ggml_cuda where there is no CUDA) out of the way.
        string model = GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "local-glm.gguf"));
        string[] args = { "--backend", "ggml_cpu", "--tp", "2", "--tp-node-id", "0",
            "--tp-peers", "127.0.0.1:49500,127.0.0.1:49501" };
        foreach (HostRun run in new[] { RunCliWith(model, args), RunServerWith(model, args) })
        {
            Assert.Equal(ModelLoadRefusedExitCode, run.ExitCode);
            Assert.Contains("does not implement distributed tensor parallelism", run.Stderr);
            Assert.DoesNotContain("waiting for rank", run.Stdout);
            Assert.DoesNotContain("Unhandled exception", run.Stderr);
        }
    }

    [Theory]
    [InlineData("--tp=2")]
    [InlineData("--layer-split=2")]
    [InlineData("--tp-node-id=0")]
    public void BothHosts_KeepPlacementLookingStopValuesAsText(string literal)
    {
        string model = WriteJunkModel();
        foreach (HostRun run in new[] { RunCliWith(model, new[] { "--stop", literal }),
            RunServerWith(model, new[] { "--stop", literal }) })
            AssertRefused(run, "Not a GGUF file");
    }

    [Theory]
    [InlineData("--tp=2")]
    [InlineData("--layer-split")]
    public void Cli_KeepsPlacementLookingSystemPromptAsText(string literal)
        => AssertRefused(RunCliWith(WriteJunkModel(), new[] { "--system", literal }), "Not a GGUF file");

    // ---- the Qwen-Image-Edit-2511 pipeline's options and checkpoints ----------------
    // Removed with the pipeline. Each host must stop at startup with one line saying why,
    // never ignore the flag (the CLI's switch has no unknown-flag trap) and never answer
    // with a bare "Unknown option" (the server's has, but it cannot say what happened).

    public static IEnumerable<object[]> RemovedFlagLines() => new[]
    {
        new object[] { "--offload-cpu", new[] { "--offload-cpu" } },
        new object[] { "--qwen-image-lora", new[] { "--qwen-image-lora", "lightning.safetensors" } },
        new object[] { "--qwen-image-lora", new[] { "--qwen-image-lora=lightning.safetensors" } },
    };

    [Theory]
    [MemberData(nameof(RemovedFlagLines))]
    public void Server_RemovedQwenImageFlag_IsAConfigurationErrorThatSaysWhy(string flag, string[] line)
    {
        HostRun run = RunServerWith(WriteJunkModel(), line);

        // Refused before the model is even opened: the junk model is never reported.
        AssertConfigurationError(run, flag + " was removed:");
    }

    [Theory]
    [MemberData(nameof(RemovedFlagLines))]
    public void Cli_RemovedQwenImageFlag_IsAConfigurationErrorThatSaysWhy(string flag, string[] line)
    {
        HostRun run = RunCliWith(WriteJunkModel(), line);

        AssertConfigurationError(run, flag + " was removed:");
    }

    [Theory]
    [InlineData("{ \"offload-cpu\": true }", "--offload-cpu")]
    [InlineData("{ \"qwen-image-lora\": \"lightning.safetensors\" }", "--qwen-image-lora")]
    public void BothHosts_RemovedQwenImageConfigKey_IsTheSameConfigurationError(string json, string flag)
    {
        string config = Path.Combine(_dir, "stale-qwen-image-edit.json");
        File.WriteAllText(config, json);

        AssertConfigurationError(RunServerWith(WriteJunkModel(), new[] { "--config", config }), flag + " was removed:");
        AssertConfigurationError(RunCliWith(WriteJunkModel(), new[] { "--config", config }), flag + " was removed:");
    }

    [Fact]
    public void Server_EarlierQwenImageCheckpoint_IsRefusedWithAMigrationNote()
    {
        HostRun run = RunServer(WriteLegacyQwenImageModel());

        AssertRefused(run, "is not a Qwen-Image-2.1 diffusion transformer");
        Assert.Contains("docs/models/qwenimage21.md", run.Stderr, StringComparison.Ordinal);
    }

    [Fact]
    public void Cli_EarlierQwenImageCheckpoint_IsRefusedWithAMigrationNote()
    {
        HostRun run = RunCli(WriteLegacyQwenImageModel());

        AssertRefused(run, "is not a Qwen-Image-2.1 diffusion transformer");
        Assert.Contains("docs/models/qwenimage21.md", run.Stderr, StringComparison.Ordinal);
    }

    // ---- the Qwen-Image-2.1 checkpoint declaration (--qwen-image-variant) ---------------
    // It means nothing to another model: the CLI refuses the flag before loading (the server
    // loads and warns, QwenImage21TurboTests). A declaration the model cannot read is the
    // model's refusal, through the variable both hosts publish.

    [Theory]
    [InlineData("--qwen-image-variant", "turbo")]
    [InlineData("--qwen-image-variant=base", null)]
    public void Cli_QwenImageVariantForAnotherModel_IsAConfigurationError(string flag, string? value)
    {
        string model = WriteGguf("Qwen3.5-9B-Q8_0.gguf", "qwen35", new[] { "token_embd.weight" });
        string[] line = value == null ? new[] { flag } : new[] { flag, value };

        string error = AssertOneConfigurationErrorLine(RunCliWith(model, line), "--qwen-image-variant applies to Qwen-Image-2.1 models only");
        Assert.Contains("'Qwen3.5-9B-Q8_0.gguf' (qwen35) is not one.", error, StringComparison.Ordinal);
    }

    [Fact]
    public void BothHosts_AnUnknownQwenImageVariantVariable_RefusesTheLoad()
    {
        // A Qwen-Image-2.1 transformer by its tensor names (the GGUFs carry no other sign).
        string model = WriteGguf("Qwen-Image-2.1-Turbo-AD-Q4_K.gguf", "qwen_image", new[]
        {
            "img_in.weight", "txt_norm.weight", "txt_in.text_norm.weight", "txt_in.weight", "proj_out.weight",
        });
        var environment = new Dictionary<string, string> { ["TS_QWEN_IMAGE_VARIANT"] = "lightning" };
        foreach (HostRun run in new[] { RunCliWith(model, Array.Empty<string>(), environment),
                                         RunServerWith(model, Array.Empty<string>(), environment: environment) })
        {
            AssertRefused(run, "TS_QWEN_IMAGE_VARIANT expects one of base, turbo " +
                "(which Qwen-Image-2.1 checkpoint the GGUF holds), not 'lightning'.");
            Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
        }
    }

    // ---- LoRA plug-ins: the shared startup preflight -----------------------------------------
    // A plug-in the model does not take, or a MiniMax-H3 Turbo plug-in beside a sampling flag its
    // recipe contradicts, stops the host with one configuration-error line before any plug-in or
    // model file is fetched; only the family of a model a --config has yet to download waits for
    // that download (its header is what names it). The MiniMax-H3 stand-ins are what the published GGUFs are to the
    // registry: no metadata, two patch projections, the partition in the file name. A plug-in's
    // weights (and a --config's model) would download from a loopback server that counts requests.

    private string H3Model(string partition = "fl2va") =>
        LoraHostFixtures.WriteH3Model(_dir, $"minimax_h3_{partition}_pruned-Q4_K.gguf");

    private string TurboManifest(LoopbackCountingServer server) =>
        LoraHostFixtures.WriteH3Manifest(Path.Combine(_dir, "loras"), url: server.UrlFor("/" + LoraHostFixtures.Weights8));

    public static IEnumerable<object[]> TurboContradictions() => new[]
    {
        new object[] { "fl2va", new[] { "--diffusion-steps", "20" }, "--diffusion-steps 20 contradicts" },
        new object[] { "fl2va", new[] { "--flow-shift", "12" }, "flow shift 12 contradicts it" },
        new object[] { "fl2va", new[] { "--sampler", "unipc" }, "sampler 'unipc' contradicts it" },
        new object[] { "fl2va", new[] { "--cfg", "2" }, "requires guidance 1.0" },
        new object[] { "ref2va", Array.Empty<string>(), "is the Ref2VA checkpoint, and Ref2VA Turbo is not supported" },
    };

    [Theory]
    [MemberData(nameof(TurboContradictions))]
    public void Cli_H3TurboPlugin_ContradictedOrOnRef2va_IsRefusedBeforeAnyDownload(string partition, string[] flags, string reason)
    {
        using var server = new LoopbackCountingServer();
        string[] line = new[] { "--lora", TurboManifest(server) }.Concat(flags).ToArray();

        AssertRefusedBeforeDownload(RunCliWith(H3Model(partition), line), reason, server);
    }

    [Theory]
    [InlineData("qwen35", "--lora applies to Qwen-Image-2.1 and MiniMax-H3 (FL2VA) models only")]
    [InlineData("qwen_image", "--lora turbo-8step.json is a minimax-h3-lora plug-in;")]
    public void Cli_H3TurboPlugin_OnAnotherFamily_IsRefusedBeforeAnyDownload(string architecture, string reason)
    {
        using var server = new LoopbackCountingServer();
        string model = architecture == "qwen35"
            ? LoraHostFixtures.WriteTextModel(_dir)
            : LoraHostFixtures.WriteQwenImageModel(_dir);

        AssertRefusedBeforeDownload(RunCliWith(model, new[] { "--lora", TurboManifest(server) }), reason, server);
    }

    [Fact]
    public void Cli_H3_RefusesBareWeights()
    {
        string weights = LoraHostFixtures.Touch(_dir, Path.Combine("loras", LoraHostFixtures.Weights8));

        AssertOneLoraRefusal(RunCliWith(H3Model(), new[] { "--lora", weights }),
            $"--lora {LoraHostFixtures.Weights8}: MiniMax-H3 takes LoRA plug-ins only as a minimax-h3-lora config");
    }

    [Fact]
    public void Cli_TypoInThePluginConfigPath_IsReportedMissing()
    {
        string typo = Path.Combine(_dir, "loras", "minimax-h3-fl2v-turbo-8stp.json");

        AssertOneLoraRefusal(RunCliWith(H3Model(), new[] { "--lora", typo }), "--lora file not found: " + typo);
    }

    [Fact]
    public void Cli_InheritedTsLoras_H3Plugin_IsHeldToTheFlags()
    {
        using var server = new LoopbackCountingServer();
        var environment = new Dictionary<string, string>
        {
            ["TS_LORAS"] = LoraCliFlags.ToJson(new[] { new LoraSpec(TurboManifest(server)) }),
        };

        AssertRefusedBeforeDownload(RunCliWith(H3Model(), new[] { "--diffusion-steps", "20" }, environment),
            "--diffusion-steps 20 contradicts", server);
    }

    [Fact]
    public void BothHosts_ConfigNamedModel_IsNotDownloaded_WhenThePluginRefusesTheLine()
    {
        using var server = new LoopbackCountingServer();
        const string ModelFile = "minimax_h3_fl2va_pruned-Q4_K.gguf";
        server.AddFile("/" + ModelFile, File.ReadAllBytes(LoraHostFixtures.WriteH3Model(Path.Combine(_dir, "source"), ModelFile)));
        string model = Path.Combine(_dir, "downloads", ModelFile);
        string config = LoraHostFixtures.Write(_dir, "h3.json",
            $$"""{ "model": { "path": {{JsonSerializer.Serialize(model)}}, "urls": [ {{JsonSerializer.Serialize(server.UrlFor("/" + ModelFile))}} ] } }""");
        string manifest = TurboManifest(server);

        AssertRefusedBeforeDownload(RunCliWith(null, new[] { "--config", config, "--lora", manifest, "--diffusion-steps", "20" }),
            "--diffusion-steps 20 contradicts", server);
        AssertRefusedBeforeDownload(RunServerWith(null, new[] { "--config", config, "--lora", manifest, "--video-steps", "20" }),
            "--video-steps 20 contradicts", server);
        Assert.False(File.Exists(model));
    }

    /// <summary>
    /// The model's family is read from its GGUF header, so a model a <c>--config</c> still has to
    /// download cannot be checked against the plug-in's type before that download. It is checked
    /// right after it, on the final command line: the plug-in another family's is refused before
    /// its own weights are fetched (the usage pages and docs say exactly this).
    /// </summary>
    [Theory]
    [InlineData("qwen_image", "--lora turbo-8step.json is a minimax-h3-lora plug-in; Qwen-Image-2.1-Q4_K.gguf (qwen_image) takes qwen-image-2.1-lora plug-ins only.")]
    [InlineData("minimax_h3", "--lora qwen-plugin.json is a qwen-image-2.1-lora plug-in; minimax_h3_fl2va_pruned-Q4_K.gguf (minimax-h3) takes minimax-h3-lora plug-ins only.")]
    public void BothHosts_ConfigModelStillToDownload_IsCheckedAfterItsDownload_BeforeThePluginsWeights(string family, string reason)
    {
        using var server = new LoopbackCountingServer();
        string modelFile = family == "qwen_image" ? "Qwen-Image-2.1-Q4_K.gguf" : "minimax_h3_fl2va_pruned-Q4_K.gguf";
        string source = family == "qwen_image"
            ? LoraHostFixtures.WriteQwenImageModel(Path.Combine(_dir, "source"), modelFile)
            : LoraHostFixtures.WriteH3Model(Path.Combine(_dir, "source"), modelFile);
        server.AddFile("/" + modelFile, File.ReadAllBytes(source));
        const string QwenWeights = "qwen-lora.safetensors";
        string plugin = family == "qwen_image"
            ? TurboManifest(server)
            : LoraHostFixtures.WriteQwenManifest(Path.Combine(_dir, "loras"), weights: QwenWeights, url: server.UrlFor("/" + QwenWeights));
        string weightsPath = "/" + (family == "qwen_image" ? LoraHostFixtures.Weights8 : QwenWeights);

        foreach (string host in new[] { "cli", "server" })
        {
            // Each host downloads into a folder of its own, so each run starts with the model pending.
            string model = Path.Combine(_dir, "downloads", host, modelFile);
            string config = LoraHostFixtures.Write(_dir, $"{host}-{family}.json",
                $$"""{ "model": { "path": {{JsonSerializer.Serialize(model)}}, "urls": [ {{JsonSerializer.Serialize(server.UrlFor("/" + modelFile))}} ] } }""");
            int before = server.RequestCount("/" + modelFile);
            string[] line = { "--config", config, "--lora", plugin };
            HostRun run = host == "cli" ? RunCliWith(null, line) : RunServerWith(null, line);

            AssertOneConfigurationErrorAmongDownloadLines(run, reason);
            Assert.Equal(before + 1, server.RequestCount("/" + modelFile));
            Assert.True(File.Exists(model), $"{host}: the model was not downloaded before the check.");
            Assert.Equal(0, server.RequestCount(weightsPath));
        }
    }

    /// <summary>A <c>--config</c> file that cannot be downloaded from any of its URLs is a
    /// configuration error on both hosts, not an unhandled exception.</summary>
    [Fact]
    public void BothHosts_ConfigFileThatCannotBeDownloaded_IsAConfigurationError()
    {
        using var server = new LoopbackCountingServer();   // serves nothing: every request is a 404
        foreach (string host in new[] { "cli", "server" })
        {
            string model = Path.Combine(_dir, "downloads", host, "missing.gguf");
            string config = LoraHostFixtures.Write(_dir, $"{host}-missing.json",
                $$"""{ "model": { "path": {{JsonSerializer.Serialize(model)}}, "urls": [ {{JsonSerializer.Serialize(server.UrlFor("/missing.gguf"))}} ] } }""");
            string[] line = { "--config", config };
            HostRun run = host == "cli" ? RunCliWith(null, line) : RunServerWith(null, line);

            string error = AssertOneConfigurationErrorAmongDownloadLines(run, "Configuration error: Failed to download model to '");
            Assert.Contains("missing.gguf' from all 1 source(s):", error, StringComparison.Ordinal);
            Assert.False(File.Exists(model));
        }
        Assert.Equal(2, server.RequestCount("/missing.gguf"));
    }

    /// <summary>Exit code 1 and exactly one "Configuration error: ..." line naming
    /// <paramref name="reason"/>, after the <c>--config</c> download's own progress lines on
    /// stderr; never a stack trace.</summary>
    private static string AssertOneConfigurationErrorAmongDownloadLines(HostRun run, string reason)
    {
        string context = $"exit={run.ExitCode}\n--- stderr ---\n{run.Stderr}\n--- stdout (tail) ---\n{Tail(run.Stdout)}";
        string[] errors = run.Stderr.Split('\n').Select(l => l.TrimEnd('\r'))
            .Where(l => l.StartsWith("Configuration error: ", StringComparison.Ordinal)).ToArray();
        Assert.True(run.ExitCode == ConfigurationErrorExitCode, context);
        Assert.True(errors.Length == 1, context);
        Assert.Contains(reason, errors[0], StringComparison.Ordinal);
        Assert.Contains("[model-download]", run.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
        return errors[0];
    }

    [Fact]
    public void Server_H3TurboPlugin_ContradictingVideoSteps_IsRefusedBeforeAnyDownload()
    {
        using var server = new LoopbackCountingServer();

        AssertRefusedBeforeDownload(RunServerWith(H3Model(), new[] { "--lora", TurboManifest(server), "--video-steps", "20" }),
            "--video-steps 20 contradicts", server);
    }

    [Theory]
    [InlineData("qwen_image", "--lora turbo-8step.json is a minimax-h3-lora plug-in;")]
    [InlineData("ref2va", "is the Ref2VA checkpoint, and Ref2VA Turbo is not supported")]
    public void Server_H3TurboPlugin_ForTheWrongModel_IsRefusedBeforeAnyDownload(string model, string reason)
    {
        using var server = new LoopbackCountingServer();
        string path = model == "qwen_image" ? LoraHostFixtures.WriteQwenImageModel(_dir) : H3Model(model);

        AssertRefusedBeforeDownload(RunServerWith(path, new[] { "--lora", TurboManifest(server) }), reason, server);
    }

    [Fact]
    public void BothHosts_ShippedTurboConfig_IsRefusedByItsOwnRecipe()
    {
        string manifest = Path.Combine(LoraHostFixtures.RepoRoot(), "config", "lora", "minimax-h3-fl2v-turbo-8step.json");
        string model = H3Model();
        const string Recipe = "contradicts minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors: " +
            "LightX2V Turbo 8-step (FL2V v1.0, 768p) samples exactly 8 steps on its own schedule.";

        Assert.StartsWith("Configuration error: --diffusion-steps 4 " + Recipe,
            AssertOneLoraRefusal(RunCliWith(model, new[] { "--lora", manifest, "--diffusion-steps", "4" }), Recipe), StringComparison.Ordinal);
        Assert.StartsWith("Configuration error: --video-steps 4 " + Recipe,
            AssertOneLoraRefusal(RunServerWith(model, new[] { "--lora", manifest, "--video-steps", "4" }), Recipe), StringComparison.Ordinal);
    }

    [Fact]
    public void Cli_InheritedQwenPlugin_OnH3_IsAWarning_AndTheModelLoads()
    {
        string qwen = LoraHostFixtures.WriteQwenManifest(Path.Combine(_dir, "loras"));
        var environment = new Dictionary<string, string> { ["TS_LORAS"] = LoraCliFlags.ToJson(new[] { new LoraSpec(qwen) }) };
        // A complete (tiny) FL2VA transformer, so the load gets as far as the companions it lacks.
        // One folder down: the model looks for them beside it and in the parent folder, which
        // must be this test's own and not the system's temporary folder.
        using var fixtures = new MiniMaxH3LoraFixtures();
        string model = Path.Combine(_dir, "h3", MiniMaxH3LoraFixtures.Fl2vaModel);
        Directory.CreateDirectory(Path.GetDirectoryName(model)!);
        File.Copy(fixtures.WriteTransformer(), model);

        HostRun run = RunCliWith(model, Array.Empty<string>(), environment);

        Assert.Contains("TS_LORAS names LoRA plug-ins qwen-plugin.json of type qwen-image-2.1-lora; " +
            $"{MiniMaxH3LoraFixtures.Fl2vaModel} (minimax-h3) takes minimax-h3-lora plug-ins and runs without them.",
            run.Stdout + run.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("Configuration error", run.Stderr, StringComparison.Ordinal);
        // It went on to the load, which refuses the transformer for the companions it lacks.
        Assert.Contains($"Loading model {MiniMaxH3LoraFixtures.Fl2vaModel}", run.Stdout, StringComparison.Ordinal);
        AssertRefused(run, "MiniMax-H3 needs companion models beside the denoiser GGUF");
    }

    /// <summary>One configuration-error line naming <paramref name="reason"/>, and nothing
    /// fetched: no request reached <paramref name="server"/> and no download was announced.</summary>
    private static string AssertRefusedBeforeDownload(HostRun run, string reason, LoopbackCountingServer server)
    {
        string line = AssertOneLoraRefusal(run, reason);
        Assert.Equal(0, server.Requests);
        return line;
    }

    private static string AssertOneLoraRefusal(HostRun run, string reason)
    {
        string line = AssertOneConfigurationErrorLine(run, reason);
        Assert.DoesNotContain("[model-download]", run.Stdout + run.Stderr, StringComparison.Ordinal);
        return line;
    }

    private static void AssertConfigurationError(HostRun run, string messageFragment)
    {
        string line = AssertOneConfigurationErrorLine(run, messageFragment);
        Assert.Contains("Qwen-Image-2.1", line, StringComparison.Ordinal);
    }

    /// <summary>Exit code 1 and exactly one stderr line, "Configuration error: ..." naming
    /// <paramref name="messageFragment"/>: never a stack trace, never "Unknown option".</summary>
    private static string AssertOneConfigurationErrorLine(HostRun run, string messageFragment)
    {
        string[] lines = run.Stderr
            .Split('\n')
            .Select(l => l.TrimEnd('\r'))
            .Where(l => l.Length > 0)
            .ToArray();
        string context = $"exit={run.ExitCode}\n--- stderr ---\n{run.Stderr}\n--- stdout (tail) ---\n{Tail(run.Stdout)}";

        Assert.True(run.ExitCode == ConfigurationErrorExitCode, context);
        Assert.True(lines.Length == 1, context);
        Assert.StartsWith("Configuration error: ", lines[0], StringComparison.Ordinal);
        Assert.Contains(messageFragment, lines[0], StringComparison.Ordinal);
        Assert.DoesNotContain("Unknown option", lines[0], StringComparison.Ordinal);
        Assert.DoesNotContain("Unhandled exception", run.Stdout + run.Stderr, StringComparison.Ordinal);
        return lines[0];
    }

    /// <summary>
    /// A tiny but complete GGUF carrying tensor names from the Qwen-Image-Edit-2511 layout
    /// (double-stream blocks, a Qwen2.5-VL text input, no <c>txt_in.text_norm</c>) and
    /// tagged <c>general.architecture=qwen_image</c>, so the registry routes it to the
    /// Qwen-Image loader, which must refuse it by name.
    /// </summary>
    private string WriteLegacyQwenImageModel() => WriteGguf("qwen-image-edit-2511-Q4_K_M.gguf", "qwen_image", new[]
    {
        "img_in.weight", "txt_norm.weight", "txt_in.weight", "proj_out.weight",
        "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.add_q_proj.weight",
    });

    /// <summary>
    /// A tiny but complete GGUF tagged <c>general.architecture=<paramref name="architecture"/></c>
    /// with tensors named <paramref name="names"/>, enough for the registry to route it and a
    /// loader to refuse it by those names. The tensors are small F32 stubs: the server checks
    /// that a file holds every byte its tensors claim before it loads.
    /// </summary>
    private string WriteGguf(string fileName, string architecture, string[] names)
    {
        const int Alignment = 32;          // GGUF default (general.alignment absent)
        const int Elements = 32;           // 128 bytes per tensor, already aligned
        string path = Path.Combine(_dir, fileName);
        using var writer = new BinaryWriter(File.Create(path));
        void WriteString(string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }
        writer.Write(0x46554747u);           // "GGUF"
        writer.Write(3u);                    // version
        writer.Write((ulong)names.Length);
        writer.Write(1UL);                   // one metadata pair
        WriteString("general.architecture");
        writer.Write(8u);                    // string
        WriteString(architecture);
        for (int i = 0; i < names.Length; i++)
        {
            WriteString(names[i]);
            writer.Write(1u);                // one dimension
            writer.Write((ulong)Elements);
            writer.Write(0u);                // F32
            writer.Write((ulong)(i * Elements * sizeof(float)));
        }
        writer.Flush();
        long pad = (Alignment - writer.BaseStream.Position % Alignment) % Alignment;
        writer.Write(new byte[pad]);
        writer.Write(new byte[names.Length * Elements * sizeof(float)]);
        return path;
    }

    private static void AssertRefused(HostRun run, string reasonFragment)
    {
        string[] lines = run.Stderr
            .Split('\n')
            .Select(l => l.TrimEnd('\r'))
            .Where(l => l.Length > 0)
            .ToArray();
        string context = $"exit={run.ExitCode}\n--- stderr ---\n{run.Stderr}\n--- stdout (tail) ---\n{Tail(run.Stdout)}";

        Assert.True(run.ExitCode == ModelLoadRefusedExitCode, context);
        Assert.True(lines.Length == 1, context);
        Assert.StartsWith(ErrorPrefix, lines[0], StringComparison.Ordinal);
        Assert.Contains(reasonFragment, lines[0], StringComparison.Ordinal);
    }

    private string WriteJunkModel()
    {
        string path = Path.Combine(_dir, "junk.gguf");
        File.WriteAllText(path, "not a gguf file at all");
        return path;
    }

    private HostRun RunServer(string modelPath, string logLevel = "Information") =>
        RunServerWith(modelPath, Array.Empty<string>(), logLevel);

    /// <param name="modelPath">The <c>--model</c>, or null when a <c>--config</c> in
    /// <paramref name="extra"/> names it.</param>
    private HostRun RunServerWith(string? modelPath, string[] extra, string logLevel = "Information",
        IReadOnlyDictionary<string, string>? environment = null)
    {
        string dll = HostAssembly("TensorSharp.Server.Host", "TensorSharp.Server.Host.dll");
        return Run(dll, logLevel, environment,
            ModelArgs(modelPath).Concat(new[] { "--backend", "cpu", "--no-webui", "--no-skills", "--no-prefix-cache" })
                .Concat(extra).ToArray());
    }

    private HostRun RunCli(string modelPath) => RunCliWith(modelPath, Array.Empty<string>());

    /// <param name="modelPath">The <c>--model</c>, or null when a <c>--config</c> in
    /// <paramref name="extra"/> names it.</param>
    private HostRun RunCliWith(string? modelPath, string[] extra, IReadOnlyDictionary<string, string>? environment = null)
    {
        string dll = HostAssembly("TensorSharp.Cli", "TensorSharp.Cli.dll");
        string input = Path.Combine(_dir, "prompt.txt");
        File.WriteAllText(input, "hello");
        return Run(dll, "Information", environment,
            ModelArgs(modelPath).Concat(new[] { "--backend", "cpu", "--input", input, "--log-dir", Path.Combine(_dir, "cli-logs") })
                .Concat(extra).ToArray());
    }

    private static string[] ModelArgs(string? modelPath) =>
        modelPath == null ? Array.Empty<string>() : new[] { "--model", modelPath };

    private HostRun Run(string dll, string logLevel, IReadOnlyDictionary<string, string>? environment, params string[] args)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = DotnetHost(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = _dir,
        };
        startInfo.ArgumentList.Add(dll);
        foreach (string arg in args)
            startInfo.ArgumentList.Add(arg);
        startInfo.Environment["TENSORSHARP_LOG_LEVEL"] = logLevel;
        startInfo.Environment["TENSORSHARP_LOG_DIR"] = Path.Combine(_dir, "logs");
        // Nothing inherited from the test run may change what the host loads.
        foreach (string name in new[] { "KV_CACHE_DTYPE", "MAX_CONTEXT", "TENSORSHARP_TP_DEGREE",
                     "TENSORSHARP_LAYER_SPLIT_DEGREE", "TENSORSHARP_TP_NODE_ID", "TENSORSHARP_TP_PEERS",
                     "TS_SPEC_DRAFT_MODEL", "TS_QWEN_IMAGE_VARIANT", "TS_LORAS" })
            startInfo.Environment.Remove(name);
        foreach ((string name, string value) in environment ?? new Dictionary<string, string>())
            startInfo.Environment[name] = value;

        using var process = new Process { StartInfo = startInfo };
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) => { if (e.Data != null) lock (stdout) stdout.AppendLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (stderr) stderr.AppendLine(e.Data); };
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        if (!process.WaitForExit(120_000))
        {
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
            Assert.Fail($"{Path.GetFileName(dll)} did not exit within 120 s.\n{stderr}\n{Tail(stdout.ToString())}");
        }
        process.WaitForExit();
        return new HostRun(process.ExitCode, stdout.ToString(), stderr.ToString());
    }

    private static string DotnetHost()
    {
        string? fromSdk = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        if (!string.IsNullOrWhiteSpace(fromSdk) && File.Exists(fromSdk))
            return fromSdk;
        string? root = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        if (!string.IsNullOrWhiteSpace(root))
        {
            string candidate = Path.Combine(root, OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet");
            if (File.Exists(candidate))
                return candidate;
        }
        return "dotnet";
    }

    /// <summary>
    /// The host's own build output (both projects build into <c>&lt;project&gt;/bin/</c>),
    /// not the copy beside the test assembly, whose deps.json is the test project's.
    /// Both hosts are project references; a missing output is a failed prerequisite.
    /// </summary>
    private static string HostAssembly(string project, string fileName)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        string path = Path.Combine(dir.FullName, project, "bin", fileName);
        Assert.True(File.Exists(path), $"Required host build output not found: {path}");
        return path;
    }

    private static string Tail(string text)
    {
        string[] lines = text.Split('\n');
        return string.Join('\n', lines.Skip(Math.Max(0, lines.Length - 25)));
    }

    private sealed record HostRun(int ExitCode, string Stdout, string Stderr);
}
