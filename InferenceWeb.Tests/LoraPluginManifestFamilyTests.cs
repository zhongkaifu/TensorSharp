// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// config/lora/ ships plug-ins for more than one model family. Each file belongs to exactly
/// one family by its name prefix, declares that family's <c>"type"</c> and parses with that
/// family's config reader; every plug-in type has a family, a shipped manifest and an
/// architecture that takes it. Both directions are checked, so neither a new file nor a new
/// type can drift past the other.
/// </summary>
public sealed class LoraPluginManifestFamilyTests
{
    private static readonly (string Prefix, string Type)[] Families =
    {
        ("qwen-image-2.1-", LoraCliFlags.QwenImagePluginType),
        ("minimax-h3-", LoraCliFlags.MiniMaxH3PluginType),
    };

    private static readonly JsonDocumentOptions CommentedJson = new()
    {
        CommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    private static string[] ShippedManifests()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.True(dir != null, "The tests run from a source checkout; config/lora/ must be reachable.");
        string lora = Path.Combine(dir!.FullName, "config", "lora");
        var files = Directory.GetFiles(lora, "*.json").OrderBy(f => f, StringComparer.Ordinal).ToArray();
        Assert.NotEmpty(files);
        return files;
    }

    private static string DeclaredType(string file)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(file), CommentedJson);
        return doc.RootElement.GetProperty("type").GetString();
    }

    [Fact]
    public void EveryShippedManifest_BelongsToExactlyOneFamilyAndDeclaresItsType()
    {
        foreach (string file in ShippedManifests())
        {
            string name = Path.GetFileName(file);
            var family = Assert.Single(Families, f => name.StartsWith(f.Prefix, StringComparison.Ordinal));
            Assert.True(family.Type == DeclaredType(file), $"{name} declares \"{DeclaredType(file)}\", not \"{family.Type}\".");
            Assert.Equal(family.Type, LoraCliFlags.DeclaredPluginType(new LoraSpec(file)));
        }
    }

    [Fact]
    public void EveryPluginType_HasExactlyOneFamilyAndAShippedManifest()
    {
        Assert.Equal(LoraCliFlags.PluginTypes.OrderBy(t => t, StringComparer.Ordinal),
            Families.Select(f => f.Type).OrderBy(t => t, StringComparer.Ordinal));
        string[] files = ShippedManifests();
        foreach (var (prefix, type) in Families)
            Assert.Contains(files, f => Path.GetFileName(f).StartsWith(prefix, StringComparison.Ordinal) && DeclaredType(f) == type);
    }

    [Fact]
    public void EveryShippedManifest_ParsesWithItsFamilysReader()
    {
        foreach (string file in ShippedManifests())
        {
            string type = DeclaredType(file);
            if (type == LoraCliFlags.QwenImagePluginType)
            {
                Assert.Equal("TensorSharp LoRA config", QwenImage21LoraConfig.Load(file).Format);
            }
            else
            {
                Assert.Equal(LoraCliFlags.MiniMaxH3PluginType, type);
                var config = MiniMaxH3LoraConfig.Load(file);
                Assert.Equal(MiniMaxH3Partition.FirstLastFrame, config.Partition);
                Assert.EndsWith(".safetensors", config.WeightsFileName, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void EveryPluginType_IsTakenByExactlyOneArchitecture()
    {
        var declared = ModelArchitectureRegistry.All.Where(a => a.LoraPluginType != null).ToList();

        Assert.Equal(LoraCliFlags.PluginTypes.OrderBy(t => t, StringComparer.Ordinal),
            declared.Select(a => a.LoraPluginType).OrderBy(t => t, StringComparer.Ordinal));
        Assert.Equal(LoraCliFlags.QwenImagePluginType, ModelArchitectureRegistry.All.Single(a => a.Id == "qwen_image").LoraPluginType);
        Assert.Equal(LoraCliFlags.MiniMaxH3PluginType,
            ModelArchitectureRegistry.All.Single(a => a.Id == MiniMaxH3Model.ArchitectureId).LoraPluginType);
    }
}
