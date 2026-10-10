// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using TensorSharp.Cli;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen-Image-2.1-Turbo: its published 8-step schedule, the defaults it gives a request, what
/// it refuses, and how a load decides that a GGUF holds it (<see cref="QwenImage21Turbo"/>).
/// The class writes <c>TS_QWEN_IMAGE_VARIANT</c>, which every Qwen-Image load reads (and the
/// server's load checks for any other model), so it runs alone.
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class QwenImage21TurboTests : IDisposable
{
    private readonly EnvScope _env = new();

    public void Dispose() => _env.Dispose();

    /// <summary>Qwen/Qwen-Image-2.1-Turbo model_index.json sample_sigmas (revision d65dbc9),
    /// which the AtomicChat GGUF card passes to stable-diffusion.cpp with a final 0.</summary>
    private static readonly float[] Published =
        { 1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f, 0.0f };

    [Theory]
    [InlineData(64 * 64)]       // 1024x1024
    [InlineData(128 * 128)]     // 2048x2048, the native area
    [InlineData(108 * 38)]      // 1728x608, an edit at a reference's aspect ratio
    [InlineData(16)]            // 64x64
    public void TheScheduleIsThePublishedSigmasUsedAsTheyAreAtEverySize(int imageTokens)
    {
        Assert.Equal(Published, QwenImage21Turbo.Recipe.Sigmas(8, imageTokens));
    }

    [Fact]
    public void TheRecipeIsEightStepsAtCfgOneWithoutAShift()
    {
        var recipe = QwenImage21Turbo.Recipe;
        Assert.Equal(8, recipe.DefaultSteps);
        Assert.Equal(1f, recipe.Cfg);
        Assert.Equal(QwenImage21SigmaShift.None, recipe.Shift);
        Assert.False(recipe.TimestepBf16);
        Assert.True(recipe.HasSchedule);
        Assert.Equal(new[] { 8 }, recipe.Nodes.Keys);
        Assert.Equal(Published[..^1], QwenImage21Turbo.SampleSigmas);
    }

    [Fact]
    public void TurboDefaultsToItsOwnScheduleAndTheBaseCheckpointToFortyShiftedSteps()
    {
        var (steps, cfg, sigmas) = QwenImage21Pipeline.ResolveSampling(new QwenImageParams(), QwenImage21Turbo.Recipe, 4096);
        Assert.Equal(8, steps);
        Assert.Equal(1f, cfg);
        Assert.Equal(Published, sigmas);

        (steps, cfg, sigmas) = QwenImage21Pipeline.ResolveSampling(new QwenImageParams(), null, 4096);
        Assert.Equal(40, steps);
        Assert.Equal(1f, cfg);
        Assert.Equal(QwenImage21Sampling.Sigmas(40, 4096), sigmas);
        // The base schedule is shifted by resolution; Turbo's is not.
        Assert.NotEqual(QwenImage21Sampling.Sigmas(8, 4096), Published);
    }

    [Fact]
    public void ExplicitSettingsWinWhereTheScheduleAllowsThem()
    {
        // The one count Turbo has, named explicitly, and an explicit CFG (a second, negative pass).
        var (steps, cfg, sigmas) = QwenImage21Pipeline.ResolveSampling(
            new QwenImageParams { Steps = 8, CfgScale = 2.5f }, QwenImage21Turbo.Recipe, 16384);
        Assert.Equal(8, steps);
        Assert.Equal(2.5f, cfg);
        Assert.Equal(Published, sigmas);

        // The base checkpoint takes any count.
        (steps, _, sigmas) = QwenImage21Pipeline.ResolveSampling(new QwenImageParams { Steps = 12 }, null, 4096);
        Assert.Equal(12, steps);
        Assert.Equal(13, sigmas.Length);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(12)]
    [InlineData(40)]
    public void AnotherStepCountIsRefusedRatherThanResampled(int requested)
    {
        var ex = Assert.Throws<ArgumentException>(() => QwenImage21Pipeline.ResolveSampling(
            new QwenImageParams { Steps = requested }, QwenImage21Turbo.Recipe, 4096));
        Assert.StartsWith("Qwen-Image-2.1-Turbo, which was distilled for one published 8-step schedule", ex.Message, StringComparison.Ordinal);
        Assert.Contains($"defines schedules for 8 step(s), not {requested}", ex.Message, StringComparison.Ordinal);
        Assert.Contains("its default (8)", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheRunLogsTheScheduleItSamples()
    {
        Assert.Equal("  [qwen21] Turbo schedule (model_index.json sample_sigmas): 8 steps on fixed sigmas " +
            "[1, 0.97845, 0.95418, 0.92663, 0.89508, 0.84515, 0.70453, 0.41457, 0]",
            QwenImage21Turbo.Recipe.LogLine(8, 4096));
    }

    // ---- identification -----------------------------------------------------------------

    [Theory]
    [InlineData("turbo", QwenImageVariant.Turbo)]
    [InlineData(" Turbo ", QwenImageVariant.Turbo)]
    [InlineData("BASE", QwenImageVariant.Base)]
    public void ADeclarationDecides(string declared, QwenImageVariant expected)
    {
        // Whatever the file is called: the declaration is the identification.
        foreach (string file in new[] { "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf", "qwen_image_2.1_Q4_K_M.gguf" })
        {
            Assert.Equal(expected, QwenImage21Turbo.Resolve(declared, Path.Combine("models", file), out string note));
            Assert.Equal($"{QwenImageVariantFlag.Name(expected)} (declared)", note);
        }
    }

    [Theory]
    [InlineData("Qwen-Image-2.1-Turbo-AD-Q4_K.gguf", QwenImageVariant.Turbo)]   // AtomicChat
    [InlineData("Qwen-Image-2.1-Turbo-Q8_0.gguf", QwenImageVariant.Turbo)]
    [InlineData("qwen_image_2.1_turbo_Q4_K_M.gguf", QwenImageVariant.Turbo)]    // Abiray, Trilogix1
    [InlineData("qwen-image-2.1-turbo-Q4_K.gguf", QwenImageVariant.Turbo)]      // DogukanUrker
    [InlineData("qwen_image_2.1_Q4_K_M.gguf", QwenImageVariant.Base)]
    [InlineData("qwen-image-2.1-Q8_0.gguf", QwenImageVariant.Base)]
    [InlineData("turbocharged-qwen-image-2.1.gguf", QwenImageVariant.Base)]     // not the word on its own
    public void WithoutADeclarationTheFileNameIsAGuessThatTheLoadSays(string file, QwenImageVariant expected)
    {
        foreach (string? undeclared in new[] { null, "", "  " })
        {
            Assert.Equal(expected, QwenImage21Turbo.Resolve(undeclared, Path.Combine("models", file), out string note));
            if (expected == QwenImageVariant.Turbo)
            {
                Assert.StartsWith("turbo, ASSUMED from the word \"turbo\" in the file name", note, StringComparison.Ordinal);
                Assert.Contains("--qwen-image-variant turbo or base (TS_QWEN_IMAGE_VARIANT)", note, StringComparison.Ordinal);
            }
            else
            {
                Assert.Equal("base", note);
            }
        }
    }

    [Theory]
    [InlineData("lightning")]
    [InlineData("8step")]
    [InlineData("1")]
    public void AnUnknownDeclarationIsRefused(string declared)
    {
        var ex = Assert.Throws<ArgumentException>(() => QwenImage21Turbo.Resolve(declared, "qwen_image_2.1_Q4_K_M.gguf", out _));
        Assert.Equal($"TS_QWEN_IMAGE_VARIANT expects one of base, turbo (which Qwen-Image-2.1 checkpoint the GGUF holds), not '{declared}'.",
            ex.Message);
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(null)] // the GGML backend this test process pins
    public void AnUnknownDeclarationRefusesTheLoad(BackendType? requested)
    {
        // A header-only Qwen-Image-2.1 transformer (Turbo's layout, no payload, no companions):
        // the declaration is read before anything else could fail.
        BackendType backend = requested ?? TestGates.PinnedGgmlBackend;
        foreach (string variable in new[] { "TS_QWEN_IMAGE_VAE", "TS_QWEN_IMAGE_TE", "TS_QWEN_IMAGE_MMPROJ", "TS_QWEN_IMAGE_LORA", LoraCliFlags.EnvironmentVariable })
            _env.Set(variable, null);
        _env.Set(QwenImageVariantFlag.EnvironmentVariable, "lightning");
        using var fixture = new QwenImageArchitectureTests.TensorHeader(
            QwenImageArchitectureTests.Version21Tensors("model.diffusion_model.", fused: true), "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf");

        bool preferManaged = NativeDequant.PreferManaged;
        try
        {
            var error = Assert.Throws<ModelLoadRefusedException>(() => new QwenImageModel(fixture.Path, backend));
            Assert.Equal("TS_QWEN_IMAGE_VARIANT expects one of base, turbo (which Qwen-Image-2.1 checkpoint the GGUF holds), not 'lightning'.",
                error.Message);
            Assert.True(ModelLoadRefusal.TryDescribe(error, out _));
        }
        finally
        {
            // Constructing a model pins the process-global dequant route; restore it.
            NativeDequant.PreferManaged = preferManaged;
        }
    }

    // ---- the model's wiring -----------------------------------------------------------------

    /// <summary>
    /// A <see cref="QwenImageModel"/> no constructor ran for, holding <paramref name="variant"/>
    /// the way a load resolves it (<paramref name="declared"/> or assumed from the file name), on
    /// <paramref name="backend"/>, with <paramref name="dit"/> as its transformer when given. The
    /// sampling and LoRA wiring read nothing else before their checks.
    /// </summary>
    internal static QwenImageModel ModelOf(QwenImageVariant variant, bool declared = true,
        GgufFile? dit = null, BackendType backend = BackendType.Cpu)
    {
        var model = (QwenImageModel)RuntimeHelpers.GetUninitializedObject(typeof(QwenImageModel));
        static void Set(Type owner, object target, string field, object? value) =>
            (owner.GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(owner.Name, field)).SetValue(target, value);
        Set(typeof(QwenImageModel), model, $"<{nameof(QwenImageModel.Variant)}>k__BackingField", variant);
        Set(typeof(QwenImageModel), model, "_variantDeclared", declared);
        Set(typeof(ModelBase), model, "_backend", backend);
        if (dit != null) Set(typeof(ModelBase), model, "_gguf", dit);
        return model;
    }

    [Fact]
    public void TheModelSamplesWithItsCheckpointsRecipe()
    {
        QwenImageModel turbo = ModelOf(QwenImageVariant.Turbo);
        Assert.Same(QwenImage21Turbo.Recipe, turbo.CheckpointRecipe);
        Assert.Same(QwenImage21Turbo.Recipe, turbo.SamplingRecipe);

        QwenImageModel baseModel = ModelOf(QwenImageVariant.Base);
        Assert.Null(baseModel.CheckpointRecipe);
        Assert.Null(baseModel.SamplingRecipe);
    }

    [Fact]
    public void ATurboModelRefusesAnotherStepCountBeforeAnyEncoderWork()
    {
        // Through the pipeline the model runs, not ResolveSampling alone: a run on Turbo must
        // sample with Turbo's recipe. The model has no encoder, transformer or VAE, so the
        // schedule is the first thing it can fail on.
        QwenImageModel turbo = ModelOf(QwenImageVariant.Turbo);
        var ex = Assert.Throws<ArgumentException>(() =>
            turbo.GenerateImage("a red fox", new QwenImageParams { Steps = 12, Width = 512, Height = 512 }));
        Assert.StartsWith("Qwen-Image-2.1-Turbo, which was distilled for one published 8-step schedule", ex.Message, StringComparison.Ordinal);
        Assert.Contains("defines schedules for 8 step(s), not 12", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>Server log entries, for the lifecycle's warnings.</summary>
    private sealed class RecordingLogger : ILogger
    {
        public List<(LogLevel Level, string Message)> Entries { get; } = new();
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter) => Entries.Add((logLevel, formatter(state, exception)));
    }

    /// <summary>A model of another architecture, for the server's load.</summary>
    private sealed class OtherModel : ModelBase
    {
        public OtherModel(string path) : base(path, BackendType.Cpu) { }
        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();
        protected override void ResetKVCacheCore() { }
    }

    [Theory]
    [InlineData("turbo")]
    [InlineData("base")]
    public void TheServerSaysWhenTheDeclarationMeetsAnotherModel(string declared)
    {
        // The server hosts the model it is given, and a declaration means nothing to any other
        // kind: it loads, and says the declaration is ignored, as it does for --lora (the CLI
        // refuses the flag, HostLoadRefusalProcessTests). TensorAgent clears the variable
        // for its entries that are not Qwen-Image.
        _env.Set(QwenImageVariantFlag.EnvironmentVariable, declared);
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        string path = Path.Combine(Path.GetTempPath(), $"ts-turbo-other-{Guid.NewGuid():N}.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(path);
        var logger = new RecordingLogger();
        try
        {
            using var lifecycle = new ModelLifecycleService(logger, (p, _, _, _) => new OtherModel(p));
            lifecycle.LoadModel(path, null, "cpu");

            var warning = Assert.Single(logger.Entries, e => e.Level == LogLevel.Warning);
            Assert.Equal($"{QwenImageVariantFlag.Flag} / {QwenImageVariantFlag.EnvironmentVariable} ({declared}) applies to " +
                $"Qwen-Image-2.1 models only; {Path.GetFileName(path)} (unknown architecture) is not one and ignores it.", warning.Message);

            // Without a declaration there is nothing to say.
            _env.Set(QwenImageVariantFlag.EnvironmentVariable, null);
            logger.Entries.Clear();
            lifecycle.LoadModel(path, null, "cpu");
            Assert.DoesNotContain(logger.Entries, e => e.Level == LogLevel.Warning);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TheFlagTheVariableAndTheValuesAreSpelledOnce()
    {
        Assert.Equal("--qwen-image-variant", QwenImageVariantFlag.Flag);
        Assert.Equal("TS_QWEN_IMAGE_VARIANT", QwenImageVariantFlag.EnvironmentVariable);
        Assert.Equal(new[] { "base", "turbo" }, QwenImageVariantFlag.Values);
        foreach (QwenImageVariant variant in Enum.GetValues<QwenImageVariant>())
            Assert.Equal(variant, QwenImageVariantFlag.Parse(QwenImageVariantFlag.Name(variant), "x"));
    }

    // ---- hosts ----------------------------------------------------------------------------

    [Fact]
    public void TheCliTakesEitherSpellingCaseInsensitivelyAndTheLastWins()
    {
        // The CLI's own switch ignores what it does not know, so every spelling the server
        // accepts is taken out before it.
        var remaining = new List<string>();
        QwenImageVariant? variant = QwenImageVariantFlag.Take(
            new[] { "--model", "m.gguf", "--qwen-image-variant", "base", "--prompt", "x", "--QWEN-IMAGE-VARIANT=Turbo" }, remaining);
        Assert.Equal(QwenImageVariant.Turbo, variant);
        Assert.Equal(new[] { "--model", "m.gguf", "--prompt", "x" }, remaining);

        remaining.Clear();
        Assert.Null(QwenImageVariantFlag.Take(new[] { "--model", "m.gguf" }, remaining));
        Assert.Equal(new[] { "--model", "m.gguf" }, remaining);

        Assert.Contains("requires a value", Assert.Throws<ArgumentException>(
            () => QwenImageVariantFlag.Take(new[] { "--qwen-image-variant" }, new List<string>())).Message, StringComparison.Ordinal);
        Assert.Contains("--qwen-image-variant expects one of base, turbo", Assert.Throws<ArgumentException>(
            () => QwenImageVariantFlag.Take(new[] { "--qwen-image-variant=fast" }, new List<string>())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BothUsagePagesDocumentTheDeclaration()
    {
        Assert.Contains(QwenImageVariantFlag.Flag, CliUsage.DocumentedFlags());
        var sw = new StringWriter();
        ServerUsage.PrintUsage(sw);
        Assert.Contains(QwenImageVariantFlag.Flag + " <base|turbo>", sw.ToString(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("turbo", "turbo")]
    [InlineData("Base", "base")]
    public void TheServerPublishesTheDeclarationAndItsValidationPassAcceptsTheFlag(string value, string published)
    {
        _env.Set(QwenImageVariantFlag.EnvironmentVariable, null);
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        string[] args = { QwenImageVariantFlag.Flag, value };

        Assert.True(ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(args));
        Assert.Equal(published, Environment.GetEnvironmentVariable(QwenImageVariantFlag.EnvironmentVariable));

        string baseDir = Path.Combine(Path.GetTempPath(), "ts-turbo-variant-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(baseDir);
        try
        {
            Assert.NotNull(ServerOptionsBuilder.Build(args, baseDir));
        }
        finally
        {
            try { Directory.Delete(baseDir, recursive: true); } catch { /* best effort */ }
        }
    }

    [Fact]
    public void TheServerRefusesAnUnknownDeclarationAtStartup()
    {
        _env.Set(QwenImageVariantFlag.EnvironmentVariable, null);
        var ex = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags(
            new[] { QwenImageVariantFlag.Flag, "lightning" }));
        Assert.Contains("--qwen-image-variant expects one of base, turbo", ex.Message, StringComparison.Ordinal);
        Assert.Null(Environment.GetEnvironmentVariable(QwenImageVariantFlag.EnvironmentVariable));
    }
}
