// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

/// <summary>
/// The Qwen-Image-2.1 entry the diffusion tests exercise: the built-in one.
///
/// <para>
/// The catalog used to offer no diffusion checkpoint, and this was then a test-only copy of
/// the files <c>config/qwen-image-2.1.json</c> downloads, so the companion publisher and the
/// live media routes had a complete multi-file model to run against. The built-in entry is
/// now that same set, so the tests check what a user actually downloads; live media tests
/// stage explicitly supplied local files under its role-aware names.
/// </para>
/// </summary>
internal static class DiffusionModelFixture
{
    internal static CatalogModel QwenImage21 { get; } =
        ModelCatalog.Find("qwen-image-2.1-q4km")
        ?? throw new InvalidOperationException("the built-in Qwen-Image-2.1 entry is missing");
}

/// <summary>
/// Tests that write the process's own environment, kept out of everything else's way.
///
/// <para>
/// <c>DiffusionCompanions</c> publishes to environment variables because that is the
/// only channel the Qwen-Image DiT reads, and an environment is one object shared by
/// every test in the assembly. Any class that constructs an <c>AgentAppHost</c>
/// publishes too, so without this the two race and the loser fails somewhere else
/// entirely — which is exactly what happened before it was added.
/// </para>
/// </summary>
[CollectionDefinition(ProcessEnvironmentCollection.Name, DisableParallelization = true)]
public sealed class ProcessEnvironmentCollection
{
    public const string Name = "process environment";
}

/// <summary>
/// Whether the built-in image-generation model's four files are the four files the
/// pipeline goes looking for, under names it will recognise.
///
/// <para>
/// A companion whose name the pipeline's directory scan does not match can be present
/// and valid yet still be silently ignored: the download verifies, and the picture is
/// made without it or not at all.
/// </para>
/// <para>
/// The scans are stated here rather than called, because they are private to
/// <c>QwenImageModel</c>. <see cref="TheScansThisFileMirrorsAreStillTheOnesTheModelPerforms"/>
/// is what keeps the two from drifting apart.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class DiffusionCatalogTests
{
    private static CatalogModel QwenImage21 =>
        DiffusionModelFixture.QwenImage21;

    /// <summary>What <c>DiffusionCompanions</c> publishes, and all it publishes: the three
    /// companions and the checkpoint variant the entry declares.</summary>
    private static readonly string[] CompanionVariables =
    [
        "TS_QWEN_IMAGE_VAE", "TS_QWEN_IMAGE_TE", "TS_QWEN_IMAGE_MMPROJ", "TS_QWEN_IMAGE_VARIANT",
    ];

    /// <summary>The Qwen-Image 2.1 Turbo entries (the 4-bit default and the 8-bit one).</summary>
    private static CatalogModel[] Turbos =>
        ModelCatalog.BuiltIn.Where(m => m.Family == CatalogFamily.QwenImage && m.ImageVariant == QwenImageVariant.Turbo).ToArray();

    /// <summary>
    /// Variables only the retired Qwen-Image-Edit-2511 pipeline read. Nothing may publish
    /// them: nothing reads the area cap any more, and a set <c>TS_QWEN_IMAGE_LORA</c> makes
    /// Qwen-Image-2.1 refuse to run at all.
    /// </summary>
    private static readonly string[] RetiredVariables =
    [
        "TS_QWEN_IMAGE_LORA", "TS_QWEN_IMAGE_MAX_AREA",
    ];

    private static string NameOf(CatalogModel model, CatalogFileRole role) =>
        model.Files.Single(f => f.Role == role).FileName;

    /// <summary>The VAE scan in <c>QwenImageModel</c>'s constructor: a safetensors file
    /// naming the 2.1 VAE.</summary>
    private static bool VaeScanFinds(string fileName)
    {
        string n = fileName.ToLowerInvariant();
        return n.Contains("qwen_image_2.1_vae") && n.EndsWith(".safetensors");
    }

    /// <summary>The text-encoder scan: the largest GGUF naming Qwen3-VL-8B that is not
    /// a projector.</summary>
    private static bool TextEncoderScanFinds(string fileName)
    {
        string n = fileName.ToLowerInvariant();
        return (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && !n.Contains("mmproj") && n.EndsWith(".gguf");
    }

    /// <summary>The vision-projector scan: a GGUF naming both a projector and
    /// Qwen3-VL-8B.</summary>
    private static bool VisionProjectorScanFinds(string fileName)
    {
        string n = fileName.ToLowerInvariant();
        return n.Contains("mmproj") && (n.Contains("qwen3vl-8b") || n.Contains("qwen3-vl-8b")) && n.EndsWith(".gguf");
    }

    [Fact]
    public void TheQwenImage21FixtureCarriesEveryNetworkTheDiTDoesNotContain()
    {
        CatalogModel model = QwenImage21;
        foreach (CatalogFileRole role in new[]
                 {
                     CatalogFileRole.Weights, CatalogFileRole.TextEncoder,
                     CatalogFileRole.Vae, CatalogFileRole.VisionProjector,
                 })
        {
            Assert.True(model.Files.Any(f => f.Role == role), $"{model.Id} has no {role}");
        }

        // The VAE and the text encoder are not optional: QwenImageModel's constructor
        // throws FileNotFoundException without them, so an entry that marks either one
        // optional is an entry that can finish downloading and then refuse to load.
        Assert.False(model.Files.Single(f => f.Role == CatalogFileRole.Vae).Optional);
        Assert.False(model.Files.Single(f => f.Role == CatalogFileRole.TextEncoder).Optional);
        // Nor the vision projector: an edit refuses to run without it, and nothing in the
        // app can fetch this role after the fact (the Models page adds a chat model's
        // Projector), so an optional one made an editor that could not edit.
        Assert.False(model.Files.Single(f => f.Role == CatalogFileRole.VisionProjector).Optional);
    }

    [Fact]
    public void EveryFixtureCompanionIsNamedSomethingThePipelinesOwnScanWillMatch()
    {
        CatalogModel model = QwenImage21;

        Assert.True(VaeScanFinds(NameOf(model, CatalogFileRole.Vae)),
            $"the VAE '{NameOf(model, CatalogFileRole.Vae)}' is not a name the 2.1 VAE scan looks for");

        Assert.True(TextEncoderScanFinds(NameOf(model, CatalogFileRole.TextEncoder)),
            $"the text encoder '{NameOf(model, CatalogFileRole.TextEncoder)}' is not a name the Qwen3-VL-8B scan looks for");

        Assert.True(VisionProjectorScanFinds(NameOf(model, CatalogFileRole.VisionProjector)),
            $"the vision projector '{NameOf(model, CatalogFileRole.VisionProjector)}' is not a name the mmproj scan "
            + "looks for — it needs 'mmproj' AND 'qwen3vl-8b' (or 'qwen3-vl-8b') in it, or editing refuses to run "
            + "for want of a projector that is sitting right there");
    }

    [Fact]
    public void NoTwoFixtureCompanionsAnswerToTheSameScan()
    {
        // The three scans run over one directory, so a name that satisfies two of them
        // hands one network to the wrong loader. The text-encoder scan in particular
        // takes the LARGEST matching GGUF, which is what the projector would be if its
        // name did not say "mmproj".
        CatalogModel model = QwenImage21;
        string weights = NameOf(model, CatalogFileRole.Weights);
        string textEncoder = NameOf(model, CatalogFileRole.TextEncoder);
        string projector = NameOf(model, CatalogFileRole.VisionProjector);

        Assert.False(TextEncoderScanFinds(projector), $"the projector '{projector}' would be loaded as the text encoder");
        Assert.False(TextEncoderScanFinds(weights), $"the DiT '{weights}' would be loaded as the text encoder");
        Assert.False(VisionProjectorScanFinds(textEncoder), $"the text encoder '{textEncoder}' would be loaded as the projector");
        Assert.False(VisionProjectorScanFinds(weights), $"the DiT '{weights}' would be loaded as the projector");
        Assert.False(VaeScanFinds(weights));
        Assert.False(VaeScanFinds(textEncoder));
        Assert.False(VaeScanFinds(projector));
    }

    /// <summary>
    /// The Turbo entries take the base entry's VAE, text encoder and projector as the same
    /// files -- name, size, hash and pinned URL -- so an install that holds one entry links
    /// them for the other instead of downloading 6.9 GB again (ModelStore.SharedCopy matches
    /// on size and hash). Only the denoiser is their own, and no scan mistakes it for a companion.
    /// </summary>
    [Fact]
    public void TheTurboEntriesShareTheBaseEntrysCompanionsAndDeclareTheirCheckpoint()
    {
        Assert.Equal(QwenImageVariant.Base, QwenImage21.ImageVariant);
        Assert.Equal(new[] { "qwen-image-2.1-turbo-adq4k", "qwen-image-2.1-turbo-q8" }, Turbos.Select(m => m.Id));
        foreach (CatalogModel turbo in Turbos)
        {
            Assert.True(turbo.IsImageGenerator);
            Assert.Equal(CatalogArchitectureKind.Diffusion, turbo.Kind);
            Assert.Equal(QwenImage21.Modalities, turbo.Modalities);
            foreach (CatalogFileRole role in new[] { CatalogFileRole.TextEncoder, CatalogFileRole.Vae, CatalogFileRole.VisionProjector })
                Assert.Equal(QwenImage21.Files.Single(f => f.Role == role), turbo.Files.Single(f => f.Role == role));
            Assert.Equal(4, turbo.Files.Count);
            Assert.Contains("/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/bb25d06bc74119c12207243d68917951e6d9c232/", turbo.Weights.Url);
            string weights = turbo.Weights.FileName;
            Assert.False(VaeScanFinds(weights) || TextEncoderScanFinds(weights) || VisionProjectorScanFinds(weights), weights);
        }
        CatalogModel fast = ModelCatalog.Find("qwen-image-2.1-turbo-adq4k")!, best = ModelCatalog.Find("qwen-image-2.1-turbo-q8")!;
        Assert.Equal(("Qwen-Image-2.1-Turbo-AD-Q4_K.gguf", 4_201_694_944L, "4bb73c53cbe284bbd6d69b9fdc59539c531f0389dd2aa567663d777f8130bc59"),
            (fast.Weights.FileName, fast.Weights.Bytes, fast.Weights.Sha256));
        Assert.Equal(("Qwen-Image-2.1-Turbo-Q8_0.gguf", 7_591_554_784L, "99f498fb7188be7eac9eb5f345a9e074d30eef3ca235b419e92563e5b48073c4"),
            (best.Weights.FileName, best.Weights.Bytes, best.Weights.Sha256));
    }

    /// <summary>
    /// The sampling schedule follows the entry, never the file name: every Qwen-Image selection
    /// publishes its variant (the base entry too, so a Turbo declaration does not outlive its
    /// selection), and any other selection clears it.
    /// </summary>
    [Fact]
    public void EveryQwenImageSelectionPublishesTheCheckpointItsEntryDeclares()
    {
        string variable = QwenImageVariantFlag.EnvironmentVariable;
        foreach (CatalogModel model in Turbos.Prepend(QwenImage21))
        {
            using var installation = new FakeInstall(model, install: m => m.Files);
            IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(model, installation.Store);
            string expected = model.ImageVariant == QwenImageVariant.Turbo ? "turbo" : "base";
            Assert.Equal(expected, published[variable]);
            Assert.Equal(expected, Environment.GetEnvironmentVariable(variable));
            Assert.Equal(CompanionVariables.Order(StringComparer.Ordinal), published.Keys.Order(StringComparer.Ordinal));

            DiffusionCompanions.Publish(ModelCatalog.BuiltIn.First(m => m.Family == CatalogFamily.Wan), installation.Store);
            Assert.Null(Environment.GetEnvironmentVariable(variable));
        }
    }

    [Fact]
    public void TheScansThisFileMirrorsAreStillTheOnesTheModelPerforms()
    {
        // A weak guard and an honest one: it cannot tell that a predicate's logic
        // changed, only that the strings it matches on are still there. That is enough
        // to catch the rename this file exists to prevent, and the alternative — making
        // the resolvers public on a shared assembly so a phone test can call them — is
        // a worse trade.
        string model = ReadSource("TensorSharp.Models/Models/QwenImage/QwenImageModel.cs");
        foreach (string literal in new[]
                 {
                     "\"TS_QWEN_IMAGE_VAE\"", "\"TS_QWEN_IMAGE_TE\"", "\"TS_QWEN_IMAGE_MMPROJ\"",
                     "n.Contains(\"qwen_image_2.1_vae\") && n.EndsWith(\".safetensors\")",
                     "(n.Contains(\"qwen3vl-8b\") || n.Contains(\"qwen3-vl-8b\")) && !n.Contains(\"mmproj\") && n.EndsWith(\".gguf\")",
                     "n.Contains(\"mmproj\") && (n.Contains(\"qwen3vl-8b\") || n.Contains(\"qwen3-vl-8b\")) && n.EndsWith(\".gguf\")",
                 })
        {
            Assert.Contains(literal, model, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EveryInstalledCompanionIsPublishedUnderTheVariableTheModelReads()
    {
        using var installation = new FakeInstall(QwenImage21, install: model => model.Files);

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(QwenImage21, installation.Store);

        Assert.Equal(installation.PathOf(CatalogFileRole.Vae), published["TS_QWEN_IMAGE_VAE"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.TextEncoder), published["TS_QWEN_IMAGE_TE"]);
        Assert.Equal(installation.PathOf(CatalogFileRole.VisionProjector), published["TS_QWEN_IMAGE_MMPROJ"]);
        foreach (string variable in CompanionVariables)
            Assert.Equal(published[variable], Environment.GetEnvironmentVariable(variable));
        Assert.Equal(CompanionVariables.Order(StringComparer.Ordinal), published.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void NothingOnlyTheRetiredPipelineReadIsEverPublished()
    {
        // A host that still published the LoRA path would stop every Qwen-Image-2.1 run
        // with "does not support LoRA adapters" the moment such a file was installed.
        // The variables are process-wide and a developer shell may still export them, so
        // start from a known state and hand the caller's values back afterwards.
        var saved = RetiredVariables.ToDictionary(v => v, Environment.GetEnvironmentVariable);
        try
        {
            foreach (string variable in RetiredVariables)
                Environment.SetEnvironmentVariable(variable, null);
            using var installation = new FakeInstall(QwenImage21, install: model => model.Files);

            IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(QwenImage21, installation.Store);

            foreach (string variable in RetiredVariables)
            {
                Assert.False(published.ContainsKey(variable), $"{variable} was published");
                Assert.Null(Environment.GetEnvironmentVariable(variable));
            }
        }
        finally
        {
            foreach ((string variable, string? value) in saved)
                Environment.SetEnvironmentVariable(variable, value);
        }
    }

    [Fact]
    public void ACompanionThatWasNeverDownloadedIsClearedRatherThanPointedAt()
    {
        // Companion files may be absent from a partial or deliberately minimal local
        // installation (a sideloaded copy without the projector), and the environment is
        // process-wide. Leaving a variable from a previous selection would point the
        // next load at unrelated state.
        Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ", "/somewhere/from/before.gguf");
        using var installation = new FakeInstall(QwenImage21,
            install: model => model.Files.Where(f => f.Role != CatalogFileRole.VisionProjector));

        IReadOnlyDictionary<string, string> published = DiffusionCompanions.Publish(QwenImage21, installation.Store);

        Assert.Null(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MMPROJ"));
        Assert.False(published.ContainsKey("TS_QWEN_IMAGE_MMPROJ"));
        Assert.Equal(installation.PathOf(CatalogFileRole.Vae), published["TS_QWEN_IMAGE_VAE"]);
    }

    [Fact]
    public void SelectingSomethingThatIsNotADiffusionModelLeavesNothingBehind()
    {
        using var installation = new FakeInstall(QwenImage21, install: model => model.Files);
        DiffusionCompanions.Publish(QwenImage21, installation.Store);
        DiffusionCompanions.Publish(null, installation.Store);

        foreach (string variable in CompanionVariables)
            Assert.Null(Environment.GetEnvironmentVariable(variable));
    }

    /// <summary>The repo's own copy of a file, so a test can read the source it mirrors.</summary>
    private static string ReadSource(string relativePath)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;

        Assert.True(directory is not null,
            $"no repository root above {AppContext.BaseDirectory}; this test reads {relativePath} from the working tree");
        string full = Path.Combine(directory!.FullName, relativePath);
        Assert.True(File.Exists(full), $"{full} is missing");
        return File.ReadAllText(full);
    }

    /// <summary>
    /// A model directory holding a chosen subset of an entry's files, so the publisher
    /// can be exercised against what is really on disk. The files are empty: nothing
    /// here loads them, and the point is which paths exist.
    /// </summary>
    private sealed class FakeInstall : IDisposable
    {
        private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-diffusion-" + Guid.NewGuid().ToString("N"));
        private readonly CatalogModel _model;

        public FakeInstall(CatalogModel model, Func<CatalogModel, IEnumerable<CatalogFile>> install)
        {
            _model = model;
            Store = new ModelStore(_root);
            Directory.CreateDirectory(Store.DirectoryFor(model));
            foreach (CatalogFile file in install(model))
                File.WriteAllBytes(Store.PathFor(model, file), []);
        }

        public ModelStore Store { get; }

        public string PathOf(CatalogFileRole role) =>
            Store.PathFor(_model, _model.Files.Single(f => f.Role == role));

        public void Dispose()
        {
            // The publisher writes process-wide state; a test that left it set would
            // decide what the next one sees.
            DiffusionCompanions.Publish(null, Store);
            try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
        }
    }

    /// <summary>
    /// A remembered selection can outlive the catalog entry it names. Constructing a
    /// host for that state must clear process-wide companion paths rather than leave a
    /// previous diffusion run wired into an unrelated model.
    /// </summary>
    [Fact]
    public void BuildingTheHostClearsCompanionsForARemovedDiffusionSelection()
    {
        CatalogModel model = QwenImage21;
        // The id the catalog's Qwen-Image-Edit entry used before it was withdrawn: a
        // device that installed it still has it saved as its selection.
        const string removedId = "qwen-image-edit-2511-q2k";
        Assert.Null(ModelCatalog.Find(removedId));

        string root = Path.Combine(Path.GetTempPath(), "tensoragent-wired-" + Guid.NewGuid().ToString("N"));
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache")) { DeviceMemoryGB = 16 };
        paths.EnsureCreated();

        var settings = new SettingsStore(paths.SettingsFile);
        AppSettings chosen = settings.Load();
        chosen.SelectedModelId = removedId;
        settings.Save(chosen);

        using var installation = new FakeInstall(model, install: candidate => candidate.Files);
        Assert.NotEmpty(DiffusionCompanions.Publish(model, installation.Store));

        using (var host = new AgentAppHost(paths))
        {
            // Only the published companions are the host's to clear; nothing sets the
            // retired variables any more (NothingOnlyTheRetiredPipelineReadIsEverPublished).
            foreach (string variable in CompanionVariables)
                Assert.Null(Environment.GetEnvironmentVariable(variable));
        }

        try { Directory.Delete(root, true); } catch (Exception) { /* scratch */ }
    }
}
