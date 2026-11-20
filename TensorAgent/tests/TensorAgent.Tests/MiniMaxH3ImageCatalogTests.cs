// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using System.Text.RegularExpressions;
using TensorAgent.Core.Catalog;
using TensorSharp.Models.MiniMaxH3Image;

namespace TensorAgent.Tests;

/// <summary>
/// The MiniMax-H3-Image entries: their files are the configs' (pinned, hashed), the text encoder and tokenizer
/// files are the MiniMax-H3 entry's byte for byte, and the companions reach the engine under the variables it
/// reads. No test here writes a process environment variable: <see cref="DiffusionCompanions.Resolve"/> is
/// what <see cref="DiffusionCompanions.Publish"/> would write.
/// </summary>
public sealed class MiniMaxH3ImageCatalogTests
{
    private static readonly string[] Ids = { "minimax-h3-image-q4km", "minimax-h3-image-q8-0" };

    private static CatalogModel Entry(string id) => Assert.IsType<CatalogModel>(ModelCatalog.Find(id));

    private static CatalogModel Video => Entry("minimax-h3-fl2va-q4k");

    [Fact]
    public void TwoEntriesMakePicturesAndEditPhotos()
    {
        Assert.Equal(Ids, ModelCatalog.BuiltIn.Where(m => m.Family == CatalogFamily.MiniMaxH3Image).Select(m => m.Id));
        foreach (string id in Ids)
        {
            CatalogModel model = Entry(id);
            Assert.Equal("MiniMax-H3-Image", model.DisplayName);
            Assert.Equal(CatalogArchitectureKind.Diffusion, model.Kind);
            Assert.Equal(CatalogModalities.Image | CatalogModalities.ImageOutput, model.Modalities);
            Assert.True(model.IsImageGenerator);
            Assert.False(model.IsVideoGenerator);
            Assert.Equal(0, model.ContextLength);
            Assert.All(model.Files, f => Assert.False(f.Optional, $"{id}/{f.FileName} is optional"));
            Assert.Equal(new[] { CatalogFileRole.Weights, CatalogFileRole.TextEncoder, CatalogFileRole.Vae,
                    CatalogFileRole.Tokenizer, CatalogFileRole.Tokenizer, CatalogFileRole.Tokenizer },
                model.Files.Select(f => f.Role));
            Assert.Same(ModelCatalog.MiniMaxH3ImageTerms, model.LicenseTerms);
            Assert.Equal("MiniMax-H3-Image Community License", model.License);
        }
        Assert.Equal(32, Entry("minimax-h3-image-q4km").MinDeviceMemoryGB);
        Assert.Equal(48, Entry("minimax-h3-image-q8-0").MinDeviceMemoryGB);
    }

    /// <summary>
    /// The sizes and hashes were read from the Hugging Face tree API (LFS object ids) at the pinned commits on
    /// 2026-11-19 and are the bytes the validation runs downloaded; stated here once more, so an edit to one
    /// place shows up as a failure rather than a download that fails its hash check on a user's machine.
    /// </summary>
    [Theory]
    [InlineData("minimax-h3-image-q4km", "minimax_h3_image_pruned-Q4_K_M.gguf", 12_721_119_968L,
        "ef784687ff25a974191d00f941388060ef971286fbd2e98b2769fa78c8d599a8")]
    [InlineData("minimax-h3-image-q8-0", "minimax_h3_image_pruned-Q8_0.gguf", 22_026_760_928L,
        "ff5ed88d968e9fc3c3352101592ba83f963879ae3118ce9ab9eb7946aad70255")]
    public void TheDenoiserIsUnslothsGgufAtItsPinnedCommit(string id, string file, long bytes, string sha256)
    {
        CatalogFile weights = Entry(id).Weights;
        Assert.Equal(file, weights.FileName);
        Assert.Equal(bytes, weights.Bytes);
        Assert.Equal(sha256, weights.Sha256);
        Assert.Equal("https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF/resolve/0d25c7349e007529691bbc0f4426b579fce7de01/" + file,
            weights.Url);
        // The DiT is recognised by its tensors (the GGUFs say hyvid); the catalog only names the file.
        Assert.EndsWith(".gguf", weights.FileName, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTextEncoderAndTokenizerAreTheMiniMaxH3EntrysFilesSoTheStoreLinksThem()
    {
        foreach (string id in Ids)
        {
            CatalogModel model = Entry(id);
            foreach (CatalogFileRole role in new[] { CatalogFileRole.TextEncoder, CatalogFileRole.Tokenizer })
                Assert.Equal(Video.Files.Where(f => f.Role == role), model.Files.Where(f => f.Role == role));
            CatalogFile vae = Assert.Single(model.Files, f => f.Role == CatalogFileRole.Vae);
            Assert.Equal(new CatalogFile(CatalogFileRole.Vae, "minimax_h3_image_vae_fp16.safetensors",
                "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/vae/minimax_h3_image_vae_fp16.safetensors",
                1_168_505_312, "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"), vae);
        }
        // The two picture entries share every companion with each other as well.
        Assert.Equal(Entry(Ids[0]).Files.Skip(1), Entry(Ids[1]).Files.Skip(1));
        CatalogFile encoder = Assert.Single(Video.Files, f => f.Role == CatalogFileRole.TextEncoder);
        Assert.Equal(18_218_065_024, encoder.Bytes);
        Assert.Equal("11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc", encoder.Sha256);
    }

    [Fact]
    public void EveryFileIsPinnedToACommit()
    {
        foreach (CatalogFile file in Ids.SelectMany(id => Entry(id).Files)
                     .Concat(Video.Files.Where(f => f.Role is CatalogFileRole.TextEncoder or CatalogFileRole.Tokenizer)))
            Assert.Matches(@"^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/", file.Url);
    }

    /// <summary>
    /// The app downloads what the CLI and server download: each entry's files are those of the config for its
    /// quantization, under the same URLs and hashes.
    /// </summary>
    [Theory]
    [InlineData("minimax-h3-image-q4km", "minimax-h3-image-q4km.json")]
    [InlineData("minimax-h3-image-q8-0", "minimax-h3-image-q8.json")]
    public void TheEntryDownloadsWhatItsConfigDownloads(string id, string config)
    {
        Dictionary<string, (string Url, string Sha256)> downloads = ConfigDownloads(config);
        foreach (CatalogFile file in Entry(id).Files)
        {
            Assert.True(downloads.TryGetValue(file.FileName, out var entry), $"{config} does not download {file.FileName}");
            Assert.Equal(file.Url, entry.Url);
            Assert.Equal(file.Sha256, entry.Sha256);
        }
        Assert.Equal(Entry(id).Files.Count, downloads.Count);
    }

    /// <summary>The variables the engine resolves the companions from, as the model reads them.</summary>
    [Fact]
    public void TheNamesAndVariablesThisCatalogReliesOnAreTheOnesTheEngineUses()
    {
        string model = ReadSource("TensorSharp.Models/Models/MiniMaxH3Image/MiniMaxH3ImageDiffusionModel.cs");
        CatalogModel entry = Entry(Ids[0]);
        foreach (string literal in new[]
                 {
                     "\"TS_VIDEO_TEXT_ENCODER\"", "\"TS_VIDEO_VAE\"",
                     $"\"{entry.Files.Single(f => f.Role == CatalogFileRole.TextEncoder).FileName}\"",
                     $"\"{entry.Files.Single(f => f.Role == CatalogFileRole.Vae).FileName}\"",
                 })
            Assert.Contains(literal, model, StringComparison.Ordinal);
        Assert.Equal("minimax-h3-image", MiniMaxH3ImageFamily.ArchitectureId);
    }

    [Fact]
    public void EveryInstalledCompanionResolvesUnderTheVariableTheModelReads()
    {
        string root = Path.Combine(Path.GetTempPath(), "tensoragent-h3img-" + Guid.NewGuid().ToString("N"));
        try
        {
            CatalogModel model = Entry(Ids[0]);
            var store = new ModelStore(root);
            Directory.CreateDirectory(store.DirectoryFor(model));
            foreach (CatalogFile file in model.Files)
                File.WriteAllBytes(store.PathFor(model, file), []);

            IReadOnlyDictionary<string, string?> resolved = DiffusionCompanions.Resolve(model, store);

            Assert.Equal(store.PathFor(model, model.Files.Single(f => f.Role == CatalogFileRole.TextEncoder)),
                resolved["TS_VIDEO_TEXT_ENCODER"]);
            Assert.Equal(store.PathFor(model, model.Files.Single(f => f.Role == CatalogFileRole.Vae)), resolved["TS_VIDEO_VAE"]);
            Assert.Equal(store.DirectoryFor(model), resolved["TS_VIDEO_TOKENIZER"]);
            // Its own entry has no audio VAE, and no other family's variable keeps a path.
            Assert.Null(resolved["TS_VIDEO_AUDIO_VAE"]);
            Assert.Equal(new[] { "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_TOKENIZER", "TS_VIDEO_VAE" },
                resolved.Where(p => p.Value is not null).Select(p => p.Key).Order(StringComparer.Ordinal));

            // A tokenizer folder missing a file is not published at all.
            File.Delete(store.PathFor(model, model.Files.Single(f => f.FileName == "tokenizer_config.json")));
            Assert.Null(DiffusionCompanions.Resolve(model, store)["TS_VIDEO_TOKENIZER"]);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (Exception) { /* scratch */ }
        }
    }

    [Fact]
    public void TheModelListGroupsItWithMiniMaxH3()
    {
        Assert.Equal("minimax-h3", ModelBrowser.FamilyId(Entry(Ids[0])));
        Assert.Equal("MiniMax H3", ModelBrowser.FamilyName(Entry(Ids[1])));
    }

    // ---- helpers ------------------------------------------------------------------------------------------

    private static readonly JsonDocumentOptions Jsonc = new() { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

    /// <summary>Every file a config downloads, by file name: its first URL with the config's variables
    /// substituted (an environment default reads as its default), and its SHA-256.</summary>
    private static Dictionary<string, (string Url, string Sha256)> ConfigDownloads(string config)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepoRoot(), "config", config)), Jsonc);
        JsonElement root = document.RootElement;
        var variables = root.GetProperty("variables").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!);
        string Expand(string value)
        {
            for (int i = 0; i < 8 && value.Contains("${", StringComparison.Ordinal); i++)
            {
                value = Regex.Replace(value, @"\$\{([A-Za-z0-9_]+)(:-([^}]*))?\}",
                    m => variables.TryGetValue(m.Groups[1].Value, out string? v) ? v : m.Groups[3].Value);
            }
            return value;
        }
        var downloads = new Dictionary<string, (string, string)>(StringComparer.Ordinal);
        void Add(JsonElement spec)
        {
            string path = Expand(spec.GetProperty("path").GetString()!);
            downloads.Add(Path.GetFileName(path),
                (Expand(spec.GetProperty("urls")[0].GetString()!), spec.GetProperty("sha256").GetString()!));
            if (spec.TryGetProperty("files", out JsonElement files))
                foreach (JsonElement file in files.EnumerateArray())
                    Add(file);
        }
        foreach (string key in new[] { "model", "video-text-encoder", "video-vae" })
            Add(root.GetProperty(key));
        return downloads;
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;
        Assert.True(directory is not null, $"no repository root above {AppContext.BaseDirectory}");
        return directory!.FullName;
    }

    private static string ReadSource(string relativePath) => File.ReadAllText(Path.Combine(RepoRoot(), relativePath));
}
