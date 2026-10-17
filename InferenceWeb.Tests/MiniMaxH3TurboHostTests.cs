// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// The server's startup check of a MiniMax-H3 Turbo plug-in
/// (<see cref="ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(string[], int, string, ICollection{string})"/>
/// with the startup model and <c>--video-steps</c>). A <c>--video-steps</c> the plug-in's recipe
/// contradicts, a Ref2VA startup model, or a plug-in of the other family is refused before
/// the plug-in's weights are fetched (a loopback server counts every request) and before
/// <c>TS_LORAS</c> is published; a line the recipe allows goes on to Resolve, which downloads
/// the weights and publishes them.
/// </summary>
public sealed class MiniMaxH3TurboHostTests : IDisposable
{
    private const string W8 = LoraHostFixtures.Weights8;

    private readonly string _dir;
    private readonly EnvScope _env = new();
    private readonly LoopbackCountingServer _server = new();

    public MiniMaxH3TurboHostTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-h3-turbo-host-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        _server.AddFile("/" + W8, new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        _server.AddFile("/qwen-lora.safetensors", new byte[] { 9, 9, 9, 9 });
    }

    public void Dispose()
    {
        _server.Dispose();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string LoraDir => Path.Combine(_dir, "loras");

    /// <summary>The 8-step plug-in, whose weights download from the counting server into
    /// <see cref="LoraDir"/>.</summary>
    private string H3Manifest() => LoraHostFixtures.WriteH3Manifest(LoraDir, url: _server.UrlFor("/" + W8));

    private string QwenManifest() =>
        LoraHostFixtures.WriteQwenManifest(LoraDir, url: _server.UrlFor("/qwen-lora.safetensors"));

    private string H3Model(string fileName = "minimax_h3_fl2va_pruned-Q4_K.gguf") => LoraHostFixtures.WriteH3Model(_dir, fileName);

    private static bool Apply(string[] args, int videoSteps, string? model, ICollection<string>? warnings = null) =>
        ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args, videoSteps, model, warnings);

    private static string? PublishedLoras => Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable);

    /// <summary>Refused with <paramref name="expected"/> in the message, nothing requested,
    /// nothing written, nothing published.</summary>
    private void AssertRefusedBeforeAnyDownload(Action apply, string expected)
    {
        var ex = Assert.Throws<ArgumentException>(apply);
        Assert.Contains(expected, ex.Message, StringComparison.Ordinal);
        Assert.Equal(0, _server.Requests);
        Assert.False(File.Exists(Path.Combine(LoraDir, W8)));
        Assert.Null(PublishedLoras);
    }

    [Fact]
    public void ContradictingVideoSteps_IsRefused_BeforeTheWeightsDownload()
    {
        string manifest = H3Manifest();
        string model = H3Model();

        AssertRefusedBeforeAnyDownload(() => Apply(new[] { "--model", model, "--lora", manifest }, 20, model),
            $"--video-steps 20 contradicts {W8}: {LoraHostFixtures.H3Name8} samples exactly 8 steps on its own schedule. " +
            "Drop --video-steps or pass 8.");
    }

    [Theory]
    [InlineData(8)]
    [InlineData(0)]
    public void VideoStepsTheRecipeAllows_GoesOnToResolve(int videoSteps)
    {
        string manifest = H3Manifest();
        string model = H3Model();

        Assert.True(Apply(new[] { "--model", model, "--lora", manifest }, videoSteps, model));

        Assert.Equal(1, _server.RequestCount("/" + W8));
        string weights = Path.Combine(LoraDir, W8);
        Assert.True(File.Exists(weights));
        var published = Assert.Single(LoraCliFlags.FromJson(PublishedLoras));
        Assert.Equal(new LoraSpec(Path.GetFullPath(weights), null, Path.GetFullPath(manifest)), published);
    }

    [Fact]
    public void Ref2vaStartupModel_IsRefused()
    {
        string manifest = H3Manifest();
        string model = H3Model("minimax_h3_ref2va_pruned-Q4_K.gguf");

        AssertRefusedBeforeAnyDownload(() => Apply(new[] { "--model", model, "--lora", manifest }, 0, model),
            "LoRA plug-in turbo-8step.json is for the MiniMax-H3 FL2VA checkpoint; minimax_h3_ref2va_pruned-Q4_K.gguf is the " +
            "Ref2VA checkpoint, and Ref2VA Turbo is not supported.");
    }

    [Fact]
    public void H3StartupModel_RefusesAQwenImagePlugin()
    {
        string manifest = QwenManifest();
        string model = H3Model();

        AssertRefusedBeforeAnyDownload(() => Apply(new[] { "--model", model, "--lora", manifest }, 0, model),
            "--lora qwen-plugin.json is a qwen-image-2.1-lora plug-in; minimax_h3_fl2va_pruned-Q4_K.gguf (minimax-h3) " +
            "takes minimax-h3-lora plug-ins only.");
    }

    [Fact]
    public void QwenImageStartupModel_RefusesAnH3Plugin()
    {
        string manifest = H3Manifest();
        string model = LoraHostFixtures.WriteQwenImageModel(_dir);

        AssertRefusedBeforeAnyDownload(() => Apply(new[] { "--model", model, "--lora", manifest }, 0, model),
            "--lora turbo-8step.json is a minimax-h3-lora plug-in; Qwen-Image-2.1-Q4_K.gguf (qwen_image) " +
            "takes qwen-image-2.1-lora plug-ins only.");
    }

    [Fact]
    public void TextStartupModel_RefusesEveryPlugin()
    {
        string manifest = H3Manifest();
        string model = LoraHostFixtures.WriteTextModel(_dir);

        AssertRefusedBeforeAnyDownload(() => Apply(new[] { "--model", model, "--lora", manifest }, 0, model),
            "--lora applies to Qwen-Image-2.1 and MiniMax-H3 (FL2VA) models only; 'Qwen3.5-9B-Q8_0.gguf' (qwen35) " +
            "does not take LoRA plug-ins.");
    }

    [Fact]
    public void InheritedH3Plugin_IsHeldToVideoSteps_AndLeftAsInherited()
    {
        string inherited = LoraCliFlags.ToJson(new[] { new LoraSpec(H3Manifest()) });
        _env.Set(LoraCliFlags.EnvironmentVariable, inherited);
        string model = H3Model();

        var ex = Assert.Throws<ArgumentException>(() => Apply(new[] { "--model", model }, 20, model));

        Assert.StartsWith("--video-steps 20 contradicts", ex.Message, StringComparison.Ordinal);
        Assert.Equal(0, _server.Requests);
        Assert.Equal(inherited, PublishedLoras);
        // The recipe's own count passes, and the variable is left for the model to read.
        Assert.False(Apply(new[] { "--model", model }, 8, model));
        Assert.Equal(inherited, PublishedLoras);
    }

    [Fact]
    public void InheritedForeignPlugin_IsAWarningForTheCaller_NotARefusal()
    {
        string inherited = LoraCliFlags.ToJson(new[] { new LoraSpec(QwenManifest()) });
        _env.Set(LoraCliFlags.EnvironmentVariable, inherited);
        string model = H3Model();
        var warnings = new List<string>();

        Assert.False(Apply(new[] { "--model", model }, 20, model, warnings));

        Assert.Equal(new[]
        {
            "TS_LORAS names LoRA plug-ins qwen-plugin.json of type qwen-image-2.1-lora; minimax_h3_fl2va_pruned-Q4_K.gguf " +
            "(minimax-h3) takes minimax-h3-lora plug-ins and runs without them.",
        }, warnings);
        Assert.Equal(0, _server.Requests);
        Assert.Equal(inherited, PublishedLoras);
        // Without a collection the warnings are dropped (the lifecycle says them at load).
        Assert.False(Apply(new[] { "--model", model }, 20, model));
    }

    [Fact]
    public void OneArgumentOverload_KnowsNoModel_ButStillRefusesAMissingPlugin()
    {
        string typo = Path.Combine(LoraDir, "minimax-h3-fl2v-turbo-8stp.json");

        var ex = Assert.Throws<FileNotFoundException>(() => ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(new[] { "--lora", typo }));

        Assert.Equal($"--lora file not found: {typo}", ex.Message);
        Assert.Null(PublishedLoras);
    }
}
