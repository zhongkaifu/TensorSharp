// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="LoraHostPreflight"/>: the check TensorSharp.Cli and TensorSharp.Server run on their
/// LoRA plug-ins before any is resolved or downloaded, rule by rule: every named file exists
/// (a path a <c>--config</c> expansion will download is skipped), the model is probed, the
/// command line's plug-ins must be of the model's family, inherited <c>TS_LORAS</c> entries the
/// model skips are warnings, and the MiniMax-H3 plug-ins the model takes are checked against the
/// host's sampling flags. Then <see cref="LoraHostPreflight.CheckPreview"/> on a real
/// <c>--config</c> expansion, against a loopback server that counts every request. The refusal
/// and warning texts are spelled out here, not taken from the code under test.
/// </summary>
public sealed class LoraHostPreflightTests : IDisposable
{
    private const string W8 = LoraHostFixtures.Weights8;
    private const string W4 = LoraHostFixtures.Weights4;
    private const string Name8 = LoraHostFixtures.H3Name8;
    private const string H3Fl2va = "minimax_h3_fl2va_pruned-Q4_K.gguf";
    private const string H3Ref2va = "minimax_h3_ref2va_pruned-Q4_K.gguf";
    private const string QwenImage = "Qwen-Image-2.1-Q4_K.gguf";
    private const string TextModel = "Qwen3.5-9B-Q8_0.gguf";

    private const string BareWeightsText =
        "MiniMax-H3 takes LoRA plug-ins only as a minimax-h3-lora config (for example config/lora/minimax-h3-fl2v-turbo-8step.json), " +
        "which declares the checkpoint partition and the sampling contract. Pass that config instead of the bare weights.";

    private readonly string _dir;

    public LoraHostPreflightTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-lora-host-preflight-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- helpers ------------------------------------------------------------------------

    private static MiniMaxH3HostSampling Cli(int steps = 0, float shift = 0f, float cfg = 0f, string? sampler = null) =>
        LoraHostPreflight.CliSampling(steps, shift, cfg, sampler);

    private static LoraHostVerdict Check(IReadOnlyList<LoraSpec> flags, string? model, MiniMaxH3HostSampling? sampling = null,
        IReadOnlyList<LoraSpec>? env = null, Func<string, bool>? pending = null, LoraHost host = LoraHost.Cli) =>
        LoraHostPreflight.Check(new LoraHostRequest(host, flags, env ?? Array.Empty<LoraSpec>(), model, sampling ?? Cli(), pending));

    private static LoraSpec[] Specs(params LoraSpec[] specs) => specs;

    private string H3Model(string fileName = H3Fl2va) => LoraHostFixtures.WriteH3Model(_dir, fileName);
    private string QwenModel() => LoraHostFixtures.WriteQwenImageModel(_dir, QwenImage);
    private string Text() => LoraHostFixtures.WriteTextModel(_dir, TextModel);
    private string Manifest8() => LoraHostFixtures.WriteH3Manifest(_dir);
    private string Manifest4() => LoraHostFixtures.WriteH3Manifest(_dir, "turbo-4step.json", W4, steps: 4, name: "Test Turbo 4");
    private string QwenManifest() => LoraHostFixtures.WriteQwenManifest(_dir);
    private string Touch(string relative) => LoraHostFixtures.Touch(_dir, relative);
    private string Peft() => LoraHostFixtures.Write(_dir, "adapter_config.json", """{ "lora_alpha": 16, "r": 16 }""");

    private static string StepsText(string flag, int steps) =>
        $"{flag} {steps} contradicts {W8}: {Name8} samples exactly 8 steps on its own schedule. Drop {flag} or pass 8.";

    // ---- the sampling flags each host checks ----------------------------------------------

    [Fact]
    public void SamplingFactories_SpellEachHostsOwnLiveFlags()
    {
        var cli = LoraHostPreflight.CliSampling(8, 6f, 1f, "euler");
        Assert.Equal(new MiniMaxH3HostSampling(8, "--diffusion-steps", 6f, "--flow-shift", 1f, "--cfg", "euler", "--sampler"), cli);
        var server = LoraHostPreflight.ServerSampling(20);
        Assert.Equal(20, server.Steps);
        Assert.Equal("--video-steps", server.StepsFlag);
        Assert.Equal(0f, server.FlowShift);
        Assert.Equal(0f, server.Cfg);
        Assert.Null(server.Sampler);

        // Each spelling is a live flag of its host, so a refusal never names one the user
        // cannot type (and a rename on either side fails here).
        var cliFlags = TensorSharp.Cli.CliUsage.DocumentedFlags().ToHashSet(StringComparer.Ordinal);
        foreach (string flag in new[] { cli.StepsFlag, cli.ShiftFlag, cli.CfgFlag, cli.SamplerFlag })
            Assert.Contains(flag, cliFlags);
        var serverFlags = TensorSharp.Server.Host.Hosting.ServerUsage.DocumentedFlags().ToHashSet(StringComparer.Ordinal);
        Assert.Contains(server.StepsFlag, serverFlags);
        // The server has no flow-shift, guidance or sampler flag: a request carries them.
        foreach (string flag in new[] { cli.ShiftFlag, cli.CfgFlag, cli.SamplerFlag, cli.StepsFlag })
            Assert.DoesNotContain(flag, serverFlags);
    }

    // ---- rule 1: every named file exists, unless it is still to be downloaded --------------

    [Fact]
    public void MissingPluginConfig_IsReportedMissing_NotAsBareWeights()
    {
        string typo = Path.Combine(_dir, "minimax-h3-fl2v-turbo-8stp.json");

        var ex = Assert.Throws<FileNotFoundException>(() => Check(Specs(new LoraSpec(typo)), H3Model()));

        Assert.Equal($"--lora file not found: {typo}", ex.Message);
    }

    [Fact]
    public void MissingWeightsOrCompanionConfig_IsReportedMissing()
    {
        string qwen = QwenModel();
        string weights = Touch("w.safetensors");
        string config = Path.Combine(_dir, "absent_config.json");
        string missingWeights = Path.Combine(_dir, "absent.safetensors");

        Assert.Equal($"--lora-config file not found: {config}",
            Assert.Throws<FileNotFoundException>(() => Check(Specs(new LoraSpec(weights, null, config)), qwen)).Message);
        Assert.Equal($"--lora file not found: {missingWeights}",
            Assert.Throws<FileNotFoundException>(() => Check(Specs(new LoraSpec(missingWeights)), qwen)).Message);
    }

    [Fact]
    public void PendingPluginConfig_IsSkipped_ItsTypeIsUnknownUntilDownloaded()
    {
        string pending = Path.Combine(_dir, "downloads", "turbo.json");

        var verdict = Check(Specs(new LoraSpec(pending)), H3Model(), Cli(steps: 20), pending: p => p == pending);

        // The final command line, after the download, is checked again by the host.
        Assert.Equal(MiniMaxH3Model.ArchitectureId, verdict.Architecture?.Id);
        Assert.Empty(verdict.EnvWarnings);
    }

    [Fact]
    public void PendingWeights_CountAsBareWeights()
    {
        string pending = Path.Combine(_dir, "downloads", "style.safetensors");
        Func<string, bool> isPending = p => p == pending;

        var ex = Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(pending)), H3Model(), pending: isPending));
        Assert.Equal("--lora style.safetensors: " + BareWeightsText, ex.Message);
        // Qwen-Image-2.1 takes bare weights.
        Assert.Equal("qwen_image", Check(Specs(new LoraSpec(pending)), QwenModel(), pending: isPending).Architecture?.Id);
    }

    [Fact]
    public void PendingWeights_BesideALocalH3Config_StillMeetTheRecipeAndTheTie()
    {
        string model = H3Model();
        string manifest = Manifest8();
        string w8 = Path.Combine(_dir, "downloads", W8);
        string w4 = Path.Combine(_dir, "downloads", W4);
        Func<string, bool> isPending = p => p == w8 || p == w4;

        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(w8, null, manifest)), model, Cli(steps: 20), pending: isPending)).Message);
        Assert.NotNull(Check(Specs(new LoraSpec(w8, null, manifest)), model, Cli(steps: 8), pending: isPending).Architecture);
        Assert.StartsWith("--lora-config turbo-8step.json is the Test Turbo 8 plug-in for ", Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(w4, null, manifest)), model, pending: isPending)).Message);
    }

    [Fact]
    public void PendingCompanionConfig_LeavesTheTypeUnknown()
    {
        string weights = Touch(W8);
        string pending = Path.Combine(_dir, "downloads", "turbo.json");

        var verdict = Check(Specs(new LoraSpec(weights, null, pending)), H3Model(), Cli(steps: 20), pending: p => p == pending);

        Assert.Equal(MiniMaxH3Model.ArchitectureId, verdict.Architecture?.Id);
    }

    // ---- rule 2: the model is probed when its file exists ---------------------------------

    [Fact]
    public void ModelNotYetOnDisk_IsNotProbed_ButItsH3PluginsAreStillChecked()
    {
        string missingFl2va = Path.Combine(_dir, "downloads", H3Fl2va);
        string missingRef2va = Path.Combine(_dir, "downloads", H3Ref2va);
        string manifest = Manifest8();

        // No family is known, so another family's plug-in is left to the final check ...
        Assert.Null(Check(Specs(new LoraSpec(QwenManifest())), missingFl2va).Architecture);
        // ... while a MiniMax-H3 plug-in's recipe and partition (from the file name) are checked now.
        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(manifest)), missingFl2va, Cli(steps: 20))).Message);
        Assert.Contains($"{H3Ref2va} is the Ref2VA checkpoint", Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(manifest)), missingRef2va)).Message);
        Assert.Null(Check(Specs(new LoraSpec(manifest)), null, Cli(steps: 8)).Architecture);
    }

    [Fact]
    public void UnreadableModel_IsLeftToTheLoad()
    {
        string junk = LoraHostFixtures.Write(_dir, "junk.gguf", "not a gguf file at all");

        var verdict = Check(Specs(new LoraSpec(Touch("w.safetensors"))), junk);

        Assert.Null(verdict.Architecture);
    }

    // ---- rule 3: the command line's plug-ins against the model's family --------------------

    [Fact]
    public void ModelWithoutPlugins_RefusesEveryPlugin()
    {
        string text = Text();
        string expected = $"--lora applies to Qwen-Image-2.1 and MiniMax-H3 (FL2VA) models only; '{TextModel}' (qwen35) does not take LoRA plug-ins.";

        Assert.Equal(expected, Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(Manifest8())), text)).Message);
        Assert.Equal(expected, Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(Touch("w.safetensors"))), text)).Message);
    }

    [Fact]
    public void QwenImage_RefusesAnH3Plugin()
    {
        var ex = Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(Manifest8())), QwenModel()));

        Assert.Equal($"--lora turbo-8step.json is a minimax-h3-lora plug-in; {QwenImage} (qwen_image) takes qwen-image-2.1-lora plug-ins only.",
            ex.Message);
    }

    [Fact]
    public void H3_RefusesAQwenPlugin_InEitherForm()
    {
        string model = H3Model();
        string qwen = QwenManifest();
        string expected = $"is a qwen-image-2.1-lora plug-in; {H3Fl2va} (minimax-h3) takes minimax-h3-lora plug-ins only.";

        Assert.Equal("--lora qwen-plugin.json " + expected,
            Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(qwen)), model)).Message);
        Assert.Equal("--lora qwen-lora.safetensors " + expected, Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(Touch("qwen-lora.safetensors"), null, qwen)), model)).Message);
    }

    [Fact]
    public void H3_RefusesBareWeights_AndWeightsWithAThirdPartyConfig()
    {
        string model = H3Model();
        string weights = Touch(W8);

        Assert.Equal($"--lora {W8}: " + BareWeightsText,
            Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(weights)), model)).Message);
        Assert.Equal($"--lora {W8}: " + BareWeightsText,
            Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(weights, null, Peft())), model)).Message);
    }

    [Fact]
    public void EachFamily_TakesItsOwnPlugins()
    {
        var qwen = Check(Specs(new LoraSpec(Touch("a.safetensors")), new LoraSpec(QwenManifest()),
            new LoraSpec(Touch("b.safetensors"), 0.5f, Peft())), QwenModel());
        Assert.Equal("qwen_image", qwen.Architecture?.Id);
        Assert.Empty(qwen.EnvWarnings);

        string model = H3Model();
        string manifest = Manifest8();
        Assert.Equal(MiniMaxH3Model.ArchitectureId, Check(Specs(new LoraSpec(manifest)), model).Architecture?.Id);
        // TensorAgent's form: the weights, with the plug-in config as their --lora-config.
        Assert.Equal(MiniMaxH3Model.ArchitectureId,
            Check(Specs(new LoraSpec(Touch(W8), null, manifest)), model, Cli(steps: 8, shift: 6f, cfg: 1f, sampler: "euler")).Architecture?.Id);
    }

    // ---- rule 4: inherited plug-ins the model skips are warnings --------------------------

    [Fact]
    public void InheritedPlugins_OnAModelWithoutPlugins_AreOneWarning_AndAreNotRead()
    {
        var env = Specs(new LoraSpec(Path.Combine(_dir, "absent.json")), new LoraSpec(Path.Combine(_dir, "absent.safetensors")));

        var verdict = Check(Array.Empty<LoraSpec>(), Text(), env: env);

        Assert.Equal(new[]
        {
            $"TS_LORAS names LoRA plug-ins (absent.json, absent.safetensors), but {TextModel} (qwen35) takes none; it runs without them.",
        }, verdict.EnvWarnings);
    }

    [Fact]
    public void InheritedPlugins_OnH3_WarnOncePerForeignType_AndItsOwnAreChecked()
    {
        var env = Specs(new LoraSpec(QwenManifest()), new LoraSpec(Touch("bare.safetensors")), new LoraSpec(Manifest8()),
            new LoraSpec(Touch("peft.safetensors"), null, Peft()));

        var verdict = Check(Array.Empty<LoraSpec>(), H3Model(), Cli(steps: 8), env: env);

        Assert.Equal(new[]
        {
            $"TS_LORAS names LoRA plug-ins qwen-plugin.json of type qwen-image-2.1-lora; {H3Fl2va} (minimax-h3) takes minimax-h3-lora plug-ins and runs without them.",
            $"TS_LORAS names LoRA plug-ins bare.safetensors, peft.safetensors of type none (bare weights); {H3Fl2va} (minimax-h3) takes minimax-h3-lora plug-ins and runs without them.",
        }, verdict.EnvWarnings);
        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(
            () => Check(Array.Empty<LoraSpec>(), H3Model(), Cli(steps: 20), env: env)).Message);
    }

    [Fact]
    public void InheritedPlugins_OnQwenImage_WarnAboutH3Plugins_AndKeepBareWeights()
    {
        var env = Specs(new LoraSpec(Manifest8()), new LoraSpec(Touch("style.safetensors")));

        // An H3 plug-in Qwen-Image skips is never held to the CLI's sampling flags.
        var verdict = Check(Array.Empty<LoraSpec>(), QwenModel(), Cli(steps: 40), env: env);

        Assert.Equal(new[]
        {
            $"TS_LORAS names LoRA plug-ins turbo-8step.json of type minimax-h3-lora; {QwenImage} (qwen_image) takes qwen-image-2.1-lora plug-ins and runs without them.",
        }, verdict.EnvWarnings);
    }

    [Fact]
    public void InheritedPlugins_OnAModelThatTakesPlugins_MustExist()
    {
        string missing = Path.Combine(_dir, "absent.json");

        var ex = Assert.Throws<FileNotFoundException>(
            () => Check(Array.Empty<LoraSpec>(), H3Model(), env: Specs(new LoraSpec(missing))));

        Assert.Equal($"--lora file not found: {missing}", ex.Message);
    }

    [Fact]
    public void CommandLinePlugins_ReplaceTheInheritedOnes()
    {
        var env = Specs(new LoraSpec(Path.Combine(_dir, "absent.json")), new LoraSpec(QwenManifest()), new LoraSpec(Manifest4()));

        var verdict = Check(Specs(new LoraSpec(Manifest8())), H3Model(), Cli(steps: 8), env: env);

        // Neither read (the missing file) nor checked (the 4-step recipe) nor warned about.
        Assert.Empty(verdict.EnvWarnings);
    }

    [Fact]
    public void InheritedH3Plugin_IsHeldToTheFlags_UnlessTheModelSkipsIt()
    {
        var env = Specs(new LoraSpec(Manifest8()));

        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(
            () => Check(Array.Empty<LoraSpec>(), null, Cli(steps: 20), env: env)).Message);
        var verdict = Check(Array.Empty<LoraSpec>(), Text(), Cli(steps: 20), env: env);
        Assert.Single(verdict.EnvWarnings);
    }

    // ---- rule 5: the MiniMax-H3 plug-in against the host's sampling flags ------------------

    public static IEnumerable<object[]> ContradictingCliFlags() => new[]
    {
        new object[] { 20, 0f, 0f, null!, StepsText("--diffusion-steps", 20) },
        new object[] { 0, 12f, 0f, null!,
            $"{Name8} ({W8}) samples with video flow shift 6 (audio 3); flow shift 12 contradicts it. " +
            "Leave it unset (--flow-shift, \"flowShift\") or set it to 6." },
        new object[] { 0, 0f, 0f, "unipc",
            $"{Name8} ({W8}) samples with Euler; sampler 'unipc' contradicts it. Leave it unset (--sampler, \"sampler\") or pass euler." },
        new object[] { 0, 0f, 2f, null!, "MiniMax-H3 is CFG-distilled and requires guidance 1.0. Use --cfg 1.0 (the default)." },
    };

    [Theory]
    [MemberData(nameof(ContradictingCliFlags))]
    public void H3Plugin_RefusesAContradictingCliFlag(int steps, float shift, float cfg, string? sampler, string expected)
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(Manifest8())), H3Model(), Cli(steps, shift, cfg, sampler)));

        Assert.Equal(expected, ex.Message);
    }

    [Theory]
    [InlineData(0, 0f, 0f, null)]
    [InlineData(8, 6f, 1f, "euler")]
    [InlineData(8, 0f, 0f, "Euler")]
    public void H3Plugin_AcceptsUnsetFlagsAndTheRecipesOwnValues(int steps, float shift, float cfg, string? sampler)
    {
        var verdict = Check(Specs(new LoraSpec(Manifest8())), H3Model(), Cli(steps, shift, cfg, sampler));

        Assert.Equal(MiniMaxH3Model.ArchitectureId, verdict.Architecture?.Id);
    }

    [Theory]
    [InlineData(0, true)]
    [InlineData(8, true)]
    [InlineData(20, false)]
    [InlineData(4, false)]
    public void Server_VideoSteps_MeetsTheRecipe(int videoSteps, bool accepted)
    {
        void Run() => Check(Specs(new LoraSpec(Manifest8())), H3Model(), LoraHostPreflight.ServerSampling(videoSteps), host: LoraHost.Server);

        if (accepted)
            Run();
        else
            Assert.Equal(StepsText("--video-steps", videoSteps), Assert.Throws<ArgumentException>(Run).Message);
    }

    [Fact]
    public void Ref2vaModel_RefusesTheFl2vaPlugin()
    {
        var ex = Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(Manifest8())), H3Model(H3Ref2va)));

        Assert.Equal($"LoRA plug-in turbo-8step.json is for the MiniMax-H3 FL2VA checkpoint; {H3Ref2va} is the Ref2VA checkpoint, " +
            "and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.", ex.Message);
    }

    [Fact]
    public void TwoH3Plugins_AreRefused()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => Check(Specs(new LoraSpec(Manifest8()), new LoraSpec(Manifest4())), H3Model()));

        Assert.Equal("Two MiniMax-H3 plug-ins (turbo-8step.json, turbo-4step.json) each fix the sampling schedule; use one.", ex.Message);
    }

    [Theory]
    [InlineData(-0.5f, false)]
    [InlineData(0f, true)]
    [InlineData(1.5f, true)]
    public void H3Plugin_RefusesANegativeStrength(float strength, bool accepted)
    {
        void Run() => Check(Specs(new LoraSpec(Manifest8(), strength)), H3Model());

        if (accepted)
            Run();
        else
            Assert.Equal($"{Name8} ({W8}): strength -0.5 is negative; the Turbo schedule is distilled at strength 1 " +
                "(0 keeps the adapter's terms with zero up factors).", Assert.Throws<ArgumentException>(Run).Message);
    }

    [Fact]
    public void H3PluginConfig_IsTiedToItsWeights()
    {
        var ex = Assert.Throws<ArgumentException>(() => Check(Specs(new LoraSpec(Touch(W4), null, Manifest8())), H3Model()));

        Assert.Equal($"--lora-config turbo-8step.json is the {Name8} plug-in for {W8}; --lora names {W4}. " +
            "Pass the plug-in alone with --lora (it names its own weights), or the weights it was written for.", ex.Message);
    }

    [Fact]
    public void UnknownModel_HoldsOnlyItsH3PluginsToTheFlags()
    {
        var specs = Specs(new LoraSpec(QwenManifest()), new LoraSpec(Manifest8()));

        Assert.Null(Check(specs, null, Cli(steps: 8)).Architecture);
        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(() => Check(specs, null, Cli(steps: 20))).Message);
    }

    // ---- CheckPreview on a real --config expansion -------------------------------------------

    private string Config(string json) => LoraHostFixtures.Write(_dir, "run-" + Guid.NewGuid().ToString("N") + ".json", json);

    private static string Q(string value) => JsonSerializer.Serialize(value);

    /// <summary>Expand <paramref name="args"/> with the host's preview check as the preflight;
    /// returns the expanded line, or throws what the check threw.</summary>
    private static string[] Expand(string[] args, LoraHost host, string? inherited = null) =>
        ConfigFileArgs.Expand(args, TextWriter.Null, interactiveProgress: false,
            preview => LoraHostPreflight.CheckPreview(preview, host, inherited));

    [Fact]
    public void Preview_RefusesAContradictingFlag_BeforeAnyDownload()
    {
        using var server = new LoopbackCountingServer();
        server.AddFile("/model.gguf", new byte[] { 1, 2, 3, 4 });
        server.AddFile("/" + W8, new byte[] { 5, 6, 7, 8 });
        string model = Path.Combine(_dir, "downloads", H3Fl2va);
        string manifest = LoraHostFixtures.WriteH3Manifest(_dir, url: server.UrlFor("/" + W8));
        string cfg = Config($$"""{ "model": { "path": {{Q(model)}}, "urls": [ {{Q(server.UrlFor("/model.gguf"))}} ] } }""");

        var ex = Assert.Throws<ArgumentException>(
            () => Expand(new[] { "--config", cfg, "--lora", manifest, "--diffusion-steps=20" }, LoraHost.Cli));

        Assert.Equal(StepsText("--diffusion-steps", 20), ex.Message);
        Assert.Equal(0, server.Requests);
        Assert.False(File.Exists(model));
    }

    public static IEnumerable<object[]> PreviewLines() => new[]
    {
        new object[] { LoraHost.Cli, new[] { "--Flow-Shift", "12" }, "flow shift 12 contradicts it" },
        new object[] { LoraHost.Cli, new[] { "--sampler=unipc" }, "sampler 'unipc' contradicts it" },
        new object[] { LoraHost.Cli, new[] { "--cfg", "2" }, "requires guidance 1.0. Use --cfg 1.0" },
        new object[] { LoraHost.Cli, new[] { "--diffusion-steps", "8", "--diffusion-steps", "4" }, "--diffusion-steps 4 contradicts" },
        new object[] { LoraHost.Server, new[] { "--video-steps", "20" }, "--video-steps 20 contradicts" },
        new object[] { LoraHost.Server, new[] { "--VIDEO-STEPS=4" }, "--video-steps 4 contradicts" },
    };

    [Theory]
    [MemberData(nameof(PreviewLines))]
    public void Preview_ReadsEachHostsSamplingFlags_InEverySpelling(LoraHost host, string[] flags, string expected)
    {
        string cfg = Config($$"""{ "lora": {{Q(Manifest8())}} }""");

        var ex = Assert.Throws<ArgumentException>(() => Expand(new[] { "--config", cfg }.Concat(flags).ToArray(), host));

        Assert.Contains(expected, ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(LoraHost.Cli, "--diffusion-steps", "abc")]
    [InlineData(LoraHost.Cli, "--flow-shift", "fast")]
    [InlineData(LoraHost.Server, "--video-steps", "-3")]
    [InlineData(LoraHost.Server, "--diffusion-steps", "20")]   // not a server flag: its parser refuses it
    public void Preview_LeavesValuesItsHostRefusesToTheHostsParser(LoraHost host, string flag, string value)
    {
        string cfg = Config($$"""{ "lora": {{Q(Manifest8())}} }""");

        string[] line = Expand(new[] { "--config", cfg, flag, value }, host);

        Assert.Contains(value, line);
    }

    [Fact]
    public void Preview_InheritedTsLoras_CountOnlyWithoutCommandLinePlugins()
    {
        string inherited = LoraCliFlags.ToJson(new[] { new LoraSpec(Manifest8()) });
        string cfg = Config("""{ "max-tokens": 16 }""");

        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(
            () => Expand(new[] { "--config", cfg, "--diffusion-steps", "20" }, LoraHost.Cli, inherited)).Message);
        // A command-line plug-in replaces the variable, so its 8-step recipe is not held to 20.
        Expand(new[] { "--config", cfg, "--lora", QwenManifest(), "--diffusion-steps", "20" }, LoraHost.Cli, inherited);
    }

    [Fact]
    public void Preview_ProbesTheLastModel()
    {
        string cfg = Config($$"""{ "lora": {{Q(Manifest8())}} }""");
        string fl2va = H3Model();
        string ref2va = H3Model(H3Ref2va);

        var ex = Assert.Throws<ArgumentException>(() => Expand(new[] { "--config", cfg, "--model", fl2va, "--model", ref2va }, LoraHost.Cli));

        Assert.Contains($"{H3Ref2va} is the Ref2VA checkpoint", ex.Message, StringComparison.Ordinal);
        Assert.Equal(fl2va, Expand(new[] { "--config", cfg, "--model", ref2va, "--model", fl2va }, LoraHost.Cli)[^1]);
    }

    [Fact]
    public void Preview_SkipsAPluginConfigStillToBeDownloaded_TheFinalCheckCatchesIt()
    {
        using var server = new LoopbackCountingServer();
        string source = Manifest8();
        server.AddFile("/turbo-8step.json", File.ReadAllBytes(source));
        string pending = Path.Combine(_dir, "downloads", "turbo-8step.json");
        string cfg = Config($$"""{ "lora": { "path": {{Q(pending)}}, "urls": [ {{Q(server.UrlFor("/turbo-8step.json"))}} ] } }""");

        string[] line = Expand(new[] { "--config", cfg, "--diffusion-steps", "20" }, LoraHost.Cli);

        Assert.Equal(1, server.RequestCount("/turbo-8step.json"));
        var specs = LoraCliFlags.Parse(line);
        Assert.Equal(StepsText("--diffusion-steps", 20), Assert.Throws<ArgumentException>(
            () => Check(specs, H3Model(), Cli(steps: 20))).Message);
    }

    [Fact]
    public void Preview_WithoutAnyPlugin_DoesNothing()
    {
        string cfg = Config($$"""{ "model": {{Q(Path.Combine(_dir, "absent.gguf"))}}, "diffusion-steps": 20 }""");

        Assert.Contains("--diffusion-steps", Expand(new[] { "--config", cfg }, LoraHost.Cli, inherited: null));
    }
}

/// <summary>
/// Fixtures shared by the host-side LoRA tests (<see cref="LoraHostPreflightTests"/>,
/// <see cref="MiniMaxH3TurboHostTests"/>, <see cref="HostLoadRefusalProcessTests"/>): model files
/// the architecture registry routes by their header alone, and plug-in configs.
/// </summary>
internal static class LoraHostFixtures
{
    internal const string Weights8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors";
    internal const string Weights4 = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors";
    internal const string H3Name8 = "Test Turbo 8";

    /// <summary>A MiniMax-H3 transformer as the published GGUFs are: no metadata at all,
    /// recognised by its two patch projections (the partition comes from the file name).</summary>
    internal static string WriteH3Model(string dir, string fileName = "minimax_h3_fl2va_pruned-Q4_K.gguf") =>
        WriteGguf(Path.Combine(dir, fileName), null, new[] { "video_patch_proj.weight", "audio_patch_proj.weight" });

    /// <summary>A Qwen-Image-2.1 transformer by its tensor names.</summary>
    internal static string WriteQwenImageModel(string dir, string fileName = "Qwen-Image-2.1-Q4_K.gguf") =>
        WriteGguf(Path.Combine(dir, fileName), "qwen_image", new[]
        {
            "img_in.weight", "txt_norm.weight", "txt_in.text_norm.weight", "txt_in.weight", "proj_out.weight",
        });

    /// <summary>A text model, which takes no LoRA plug-ins.</summary>
    internal static string WriteTextModel(string dir, string fileName = "Qwen3.5-9B-Q8_0.gguf") =>
        WriteGguf(Path.Combine(dir, fileName), "qwen35", new[] { "token_embd.weight" });

    /// <summary>A tiny but complete GGUF v3 with <paramref name="names"/> as small F32 stubs and
    /// <c>general.architecture</c> set, or no metadata when <paramref name="architecture"/> is null.</summary>
    internal static string WriteGguf(string path, string? architecture, string[] names)
    {
        const int Alignment = 32;
        const int Elements = 32;
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
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
        writer.Write(architecture == null ? 0UL : 1UL);
        if (architecture != null)
        {
            WriteString("general.architecture");
            writer.Write(8u);                // string
            WriteString(architecture);
        }
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

    /// <summary>
    /// A <c>minimax-h3-lora</c> plug-in config in the shipped manifests' schema, naming
    /// <paramref name="weights"/> beside it: a plain path, or a download entry when
    /// <paramref name="url"/> is set.
    /// </summary>
    internal static string WriteH3Manifest(string dir, string fileName = "turbo-8step.json", string weights = Weights8,
        int steps = 8, string? url = null, string name = H3Name8, string partition = "fl2va")
    {
        string weightsEntry = url == null
            ? JsonSerializer.Serialize(weights)
            : $$"""{ "path": {{JsonSerializer.Serialize(weights)}}, "urls": [ {{JsonSerializer.Serialize(url)}} ] }""";
        return Write(dir, fileName, $$"""
        {
          "type": "minimax-h3-lora",
          "name": {{JsonSerializer.Serialize(name)}},
          "weights": {{weightsEntry}},
          "partition": {{JsonSerializer.Serialize(partition)}},
          "key_format": "minimax-h3-diffusers",
          "alpha": 8,
          "scale": 1.0,
          "sampling": { "steps": {{steps}}, "schedule": "linspace", "video_shift": 6.0, "audio_shift": 3.0, "sampler": "euler", "cfg": 1.0 }
        }
        """);
    }

    /// <summary>A Qwen-Image-2.1 plug-in config naming <paramref name="weights"/> beside it.</summary>
    internal static string WriteQwenManifest(string dir, string fileName = "qwen-plugin.json", string weights = "qwen-lora.safetensors",
        string? url = null)
    {
        string weightsEntry = url == null
            ? $$"""{ "path": {{JsonSerializer.Serialize(weights)}} }"""
            : $$"""{ "path": {{JsonSerializer.Serialize(weights)}}, "urls": [ {{JsonSerializer.Serialize(url)}} ] }""";
        return Write(dir, fileName, $$"""
        { "type": "qwen-image-2.1-lora", "weights": {{weightsEntry}}, "sampling": { "steps": 8, "sigmas": [1.0, 0.5] } }
        """);
    }

    internal static string Write(string dir, string relative, string text)
    {
        string path = Path.Combine(dir, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        return path;
    }

    internal static string Touch(string dir, string relative) => Write(dir, relative, "weights");

    /// <summary>The repository root (the folder holding TensorSharp.slnx).</summary>
    internal static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        return dir!.FullName;
    }
}

/// <summary>
/// A loopback HTTP server that serves registered files and counts every request, so "nothing
/// was downloaded" is measured rather than assumed. Unregistered paths get 404.
/// </summary>
internal sealed class LoopbackCountingServer : IDisposable
{
    private readonly HttpListener _listener = new();
    private readonly string _prefix;
    private readonly Dictionary<string, byte[]> _files = new();
    private readonly Dictionary<string, int> _hits = new();
    private readonly object _gate = new();
    private int _requests;

    public LoopbackCountingServer()
    {
        var probe = new TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        int port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        _prefix = $"http://127.0.0.1:{port}/";
        _listener.Prefixes.Add(_prefix);
        _listener.Start();
        Task.Run(Loop);
    }

    public int Requests
    {
        get { lock (_gate) return _requests; }
    }

    public void AddFile(string path, byte[] content)
    {
        lock (_gate) _files[path] = content;
    }

    public string UrlFor(string path) => _prefix.TrimEnd('/') + path;

    public int RequestCount(string path)
    {
        lock (_gate) return _hits.TryGetValue(path, out int n) ? n : 0;
    }

    private void Loop()
    {
        while (_listener.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = _listener.GetContext(); }
            catch { return; }

            string path = ctx.Request.Url!.AbsolutePath;
            byte[]? body;
            lock (_gate)
            {
                _requests++;
                _hits[path] = (_hits.TryGetValue(path, out int n) ? n : 0) + 1;
                _files.TryGetValue(path, out body);
            }
            try
            {
                if (body == null)
                {
                    ctx.Response.StatusCode = 404;
                    ctx.Response.Close();
                    continue;
                }
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentLength64 = body.Length;
                ctx.Response.OutputStream.Write(body, 0, body.Length);
                ctx.Response.OutputStream.Close();
            }
            catch { /* the client went away */ }
        }
    }

    public void Dispose()
    {
        try { _listener.Stop(); } catch { }
        try { _listener.Close(); } catch { }
    }
}
