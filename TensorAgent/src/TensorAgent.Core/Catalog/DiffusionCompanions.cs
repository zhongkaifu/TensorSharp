// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.Runtime;

namespace TensorAgent.Core.Catalog;

/// <summary>
/// Points a diffusion pipeline at the companion networks this installation actually
/// downloaded.
///
/// <para>
/// A diffusion GGUF is only the denoiser. Qwen-Image-2.1's VAE, its Qwen3-VL-8B text
/// encoder and that encoder's vision projector are separate files, and so are
/// MiniMax-H3's Qwen3-VL-32B text encoder, its video and audio VAEs and the tokenizer
/// files the encoder's GGUF does not carry. Each model finds its companions by scanning
/// for file names when it is not told where they are, and MiniMax-H3 scans the denoiser's
/// folder AND ITS PARENT - in this app the parent is the whole model store, where a loose
/// <c>qwen3vl</c> match is Qwen-Image's 8B encoder. So the paths are always published,
/// and the catalog's file list, not file names, decides what is loaded.
/// </para>
/// <para>
/// The desktop server does the same translation from its <c>--qwen-image-*</c> and
/// <c>--video-*</c> flags (<c>ServerOptionsBuilder.ApplyQwenImageCompanionCliFlags</c>).
/// The app has no flags, so the catalog entry and what is on disk decide instead. Every
/// variable is written on every call, and one the selected entry has no file for is
/// cleared rather than left pointing at the previous model's copy: a stale path is a load
/// that fails with a file name the user has never heard of, or one that quietly succeeds
/// with another model's network.
/// </para>
/// </summary>
public static class DiffusionCompanions
{
    /// <summary>The environment variable each family's companion role is published under,
    /// which is the name that family's model reads.</summary>
    private static readonly (CatalogFamily Family, CatalogFileRole Role, string Variable)[] Published =
    [
        (CatalogFamily.QwenImage, CatalogFileRole.Vae, "TS_QWEN_IMAGE_VAE"),
        (CatalogFamily.QwenImage, CatalogFileRole.TextEncoder, "TS_QWEN_IMAGE_TE"),
        (CatalogFamily.QwenImage, CatalogFileRole.VisionProjector, "TS_QWEN_IMAGE_MMPROJ"),
        (CatalogFamily.MiniMaxH3, CatalogFileRole.TextEncoder, "TS_VIDEO_TEXT_ENCODER"),
        (CatalogFamily.MiniMaxH3, CatalogFileRole.Vae, "TS_VIDEO_VAE"),
        (CatalogFamily.MiniMaxH3, CatalogFileRole.AudioVae, "TS_VIDEO_AUDIO_VAE"),
        // A folder, not a file: MiniMaxH3TextEncoder reads vocab.json, merges.txt and
        // tokenizer_config.json from it.
        (CatalogFamily.MiniMaxH3, CatalogFileRole.Tokenizer, "TS_VIDEO_TOKENIZER"),
        (CatalogFamily.Wan, CatalogFileRole.TextEncoder, "TS_VIDEO_TEXT_ENCODER"),
        (CatalogFamily.Wan, CatalogFileRole.Vae, "TS_VIDEO_VAE"),
        (CatalogFamily.Wan, CatalogFileRole.SecondaryWeights, "TS_VIDEO_DIT2"),
    ];

    /// <summary>
    /// Publish <paramref name="model"/>'s installed companions and, for a Qwen-Image entry, its
    /// checkpoint variant, and clear the rest. Returns what was set, variable to path (or to
    /// the variant), so the caller can log it — a startup line naming the files is the only
    /// place a user can see which copies are being used.
    /// </summary>
    /// <param name="model">The selected entry, or null when nothing is selected.</param>
    /// <param name="store">Where this installation keeps its models.</param>
    public static IReadOnlyDictionary<string, string> Publish(CatalogModel? model, ModelStore store)
    {
        ArgumentNullException.ThrowIfNull(store);

        var published = new Dictionary<string, string>(StringComparer.Ordinal);
        // Several families share the video variables. Resolve the selected family first,
        // then write each variable once so another family's row cannot clear its path.
        foreach (var group in Published.GroupBy(p => p.Variable))
        {
            string? path = null;
            if (model is not null)
                foreach (var entry in group.Where(p => p.Family == model.Family))
                    path = PathOf(model, entry.Role, store);
            Environment.SetEnvironmentVariable(group.Key, path);
            if (path is not null)
                published[group.Key] = path;
        }
        // Which checkpoint a Qwen-Image entry's weights are (CatalogModel.ImageVariant): the
        // GGUF cannot say, and without it the engine guesses from the file name. Written for
        // both variants, so a Turbo declaration never outlives its selection.
        string? variant = model?.Family == CatalogFamily.QwenImage ? QwenImageVariantFlag.Name(model.ImageVariant) : null;
        Environment.SetEnvironmentVariable(QwenImageVariantFlag.EnvironmentVariable, variant);
        if (variant is not null)
            published[QwenImageVariantFlag.EnvironmentVariable] = variant;
        return published;
    }

    private static string? PathOf(CatalogModel model, CatalogFileRole role, ModelStore store)
    {
        CatalogFile[] files = model.Files.Where(f => f.Role == role).ToArray();
        if (files.Length == 0)
            return null;
        if (role == CatalogFileRole.Tokenizer)
        {
            // All of them or nothing: a folder missing one file would be published and
            // then fail inside the first generation, after the encoder had been opened.
            string[] paths = files.Select(f => store.PathFor(model, f)).ToArray();
            return paths.All(File.Exists) ? Path.GetDirectoryName(paths[0]) : null;
        }
        string path = store.PathFor(model, files[0]);
        return File.Exists(path) ? path : null;
    }
}
