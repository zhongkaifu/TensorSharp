// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics.CodeAnalysis;
using TensorAgent.Core.Localization;
using TensorSharp.Runtime;

namespace TensorAgent.Core.Catalog;

/// <summary>
/// A catalog entry's words in the language the interface is shown in.
///
/// <para>
/// An entry names its text by a key of the catalog table (<c>Localization/en/catalog.json</c>)
/// rather than holding it, and the member that returns the text looks the key up each time it
/// is read, so a list built after the user switches language reads in the new one. A value that
/// is not a key -- a size such as "12B", a license id such as "Apache-2.0" -- reads as written.
/// </para>
/// </summary>
internal static class CatalogText
{
    private const string KeyPrefix = "catalog.";

    [return: NotNullIfNotNull(nameof(value))]
    public static string? Read(string? value) =>
        value is not null && value.StartsWith(KeyPrefix, StringComparison.Ordinal) ? Loc.T(value) : value;
}

/// <summary>What a file in a catalog entry is for. The app loads the weights and hands the
/// companions to the engine by role (the projector to <c>--mmproj</c>, a diffusion model's
/// companions to the environment variables its family reads; see DiffusionCompanions).</summary>
public enum CatalogFileRole
{
    /// <summary>The GGUF the model is loaded from.</summary>
    Weights,
    /// <summary>Vision/audio projector (mmproj) for a multimodal model.</summary>
    Projector,
    /// <summary>A diffusion model's text encoder GGUF (Qwen3-VL-8B for Qwen-Image-2.1,
    /// Qwen3-VL-32B for MiniMax-H3).</summary>
    TextEncoder,
    /// <summary>Qwen-Image-2.1 vision mmproj (image-grounded conditioning; required for editing).</summary>
    VisionProjector,
    /// <summary>A diffusion model's image or video VAE safetensors.</summary>
    Vae,
    /// <summary>Speculative-decoding draft head.</summary>
    Draft,
    /// <summary>MiniMax-H3's audio VAE safetensors, which turns the soundtrack it generates
    /// into sound (and encodes a reference recording).</summary>
    AudioVae,
    /// <summary>One of the loose tokenizer files a text encoder whose GGUF carries no
    /// tokenizer needs beside it (MiniMax-H3: vocab.json, merges.txt, tokenizer_config.json).</summary>
    Tokenizer,
    /// <summary>A later shard of a split weights GGUF (<c>-0000N-of-0000M.gguf</c>). The
    /// <see cref="Weights"/> file is shard 1, the one the engine is pointed at; it opens the
    /// others from the same folder by their exact gguf-split names, so every shard is required
    /// and stored under its published name.</summary>
    WeightsShard,
    /// <summary>The second Wan A14B denoiser, loaded after the first expert's stage.
    /// This is an independent GGUF, not a shard of the primary weights.</summary>
    SecondaryWeights,
}

/// <summary>One downloadable artifact of a catalog entry.</summary>
/// <param name="Role">What the engine uses it for.</param>
/// <param name="FileName">The name it is stored under, inside the entry's folder.</param>
/// <param name="Url">Where it is fetched from (a plain Hugging Face resolve URL), or
/// empty for an artifact whose publisher did not embed a verifiable repository and
/// which must therefore be imported by the user.</param>
/// <param name="Bytes">Exact expected artifact size (from the Hugging Face tree API for downloads,
/// or from the inspected source file for a local import).</param>
/// <param name="Sha256">Lower-case expected SHA-256 (also the LFS object id for downloads), verified
/// before a download or import is published.</param>
/// <param name="Optional">True when the model works without it (e.g. a draft head).</param>
public sealed record CatalogFile(
    CatalogFileRole Role,
    string FileName,
    string Url,
    long Bytes,
    string Sha256,
    bool Optional = false);

/// <summary>Model families the catalog knows; used for grouping in the UI and for
/// family-specific defaults (thinking, sampling).</summary>
public enum CatalogFamily
{
    // Append new families: the numeric values are part of persisted catalog data.
    Gemma4, Qwen35, Qwen36, Qwen38, QwenImage, GptOss, Bonsai, MuseGlimmer, MiniMaxH3, Qwen38FlashNext,
    DeepSeek4, DeepSeek41, Glm5, Nemotron, Mistral3, HunyuanDense, DiffusionGemma, Wan,
}

/// <summary>Dense or mixture-of-experts.</summary>
public enum CatalogArchitectureKind { Dense, MixtureOfExperts, Diffusion }

/// <summary>Input/output modalities an entry supports once its projector is installed.
/// <see cref="Image"/>, <see cref="Audio"/> and <see cref="Video"/> are what it takes in;
/// the <c>Output</c> flags are what it makes.</summary>
[Flags]
public enum CatalogModalities
{
    Text = 0,
    Image = 1,
    Audio = 2,
    Video = 4,
    ImageOutput = 8,
    VideoOutput = 16,
    AudioOutput = 32,
}

/// <summary>Sampling defaults the model card recommends; the app sends them with every
/// chat request the way the Web UI sends the server's configured defaults.</summary>
public sealed record CatalogSampling(float Temperature, int TopK, float TopP, float MinP);

/// <summary>
/// A built-in model the user can pick. Everything the app needs to obtain, size and
/// load it lives here so the catalog is data, not code: the UI shows it, the store
/// downloads or verifies an import, and the engine host turns it into load arguments.
/// </summary>
public sealed record CatalogModel
{
    private readonly string _parameters = string.Empty;
    private readonly string _quantization = string.Empty;
    private readonly string? _notes;
    private readonly string _license = string.Empty;

    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    public required CatalogFamily Family { get; init; }
    public required CatalogArchitectureKind Kind { get; init; }
    /// <summary>Human-readable parameter count, e.g. "4B effective" or "26B (4B active)",
    /// in the interface language (see <see cref="CatalogText"/>, as for the other text members).</summary>
    public required string Parameters { get => CatalogText.Read(_parameters); init => _parameters = value; }
    public required string Quantization { get => CatalogText.Read(_quantization); init => _quantization = value; }
    public required IReadOnlyList<CatalogFile> Files { get; init; }
    public required CatalogModalities Modalities { get; init; }
    /// <summary>Smallest system RAM class this entry is offered on, independent of OS.
    /// The tier allows for resident weights, KV cache, optional companions and compute
    /// buffers; accelerator placement and disk paging depend on the backend and device.</summary>
    public required int MinDeviceMemoryGB { get; init; }
    /// <summary>Context length the app configures (MAX_CONTEXT); bounds the KV cache. The
    /// phone's window, measured against its jetsam budget; a desktop is given
    /// <see cref="DesktopContextLength"/>.</summary>
    public required int ContextLength { get; init; }

    /// <summary>
    /// Bytes of K/V cache one token of context costs at F16 -- the widest precision the
    /// Settings screen offers, so a user's choice can only make it cheaper -- counted from
    /// the architecture: the layers that keep K/V for the whole context, times their KV
    /// heads, head dim, 2 (K and V) and 2 bytes. A sliding-window layer whose ring has a
    /// fixed size, and a recurrent (SSM, linear-attention) layer, does not grow with the
    /// context and is not counted. Zero when the entry states none; it then keeps
    /// <see cref="ContextLength"/> on a desktop too.
    /// </summary>
    public long KvBytesPerToken { get; init; }

    /// <summary>The window a desktop gives a chat entry when it can afford it (see
    /// <see cref="DesktopContextLength"/>).</summary>
    public const int DesktopChatContextTarget = 32768;

    /// <summary>
    /// The least window a chat entry may have on a desktop: the prompt TensorAgent shares
    /// across conversations (tool declarations, skills, instructions -- 7,219 tokens at the
    /// warm-up on Qwen3.8 27B), the 2,048-token reply reserve an 8k-16k window keeps, and
    /// as much again for the conversation itself.
    /// </summary>
    public const int MinimumDesktopChatContext = 16384;

    /// <summary>What macOS (or Windows) and the rest of a desktop keep for themselves.</summary>
    private const double DesktopSystemReserveBytes = 5e9;

    /// <summary>
    /// The context a desktop gives this entry: <see cref="DesktopChatContextTarget"/>, or
    /// the largest multiple of 4,096 below it that the entry's tier can afford, and never
    /// less than <see cref="ContextLength"/>.
    ///
    /// <para>
    /// Why: an entry's <see cref="ContextLength"/> was written against a phone's jetsam
    /// budget, and 8,192 there left a desktop with the same window. The shared prompt takes
    /// ~7.2k of it, so every follow-up compacted the whole conversation away -- an image
    /// chat forgot the image, and a thinking turn ran out at 712 tokens of thought
    /// (Nemotron-H 8B, finishReason=thinking_budget).
    /// </para>
    /// <para>
    /// Affordable: what the tier has beside the resident weights, the projector (dequantized
    /// to about twice its file) and the system, halved -- the other half is the compute
    /// buffers and the conversations the engine keeps -- must hold a whole window of K/V
    /// charged twice, the host tensor and its Metal (or CUDA) mirror. A diffusion entry is
    /// left alone: its prompt carries no shared agent prompt, and DiffusionGemma's prompt
    /// K/V is per layer and full precision, which this count does not describe.
    /// </para>
    /// </summary>
    public int DesktopContextLength
    {
        get
        {
            if (Kind == CatalogArchitectureKind.Diffusion || ContextLength <= 0
                || ContextLength >= DesktopChatContextTarget || KvBytesPerToken <= 0)
                return ContextLength;
            double spare = MinDeviceMemoryGB * 1e9 - ResidentWeightsBytes
                - 2.0 * (Projector?.Bytes ?? 0) - DesktopSystemReserveBytes;
            double affordable = spare / 2 / (2.0 * KvBytesPerToken);
            int window = affordable >= DesktopChatContextTarget
                ? DesktopChatContextTarget
                : Math.Max(0, (int)affordable / 4096 * 4096);
            return Math.Max(ContextLength, window);
        }
    }
    /// <summary>KV cache dtype to request ("f16", "q8_0"); block-quantised caches halve KV memory
    /// where the family's fused paths accept them.</summary>
    public required string KvCacheDtype { get; init; }
    /// <summary>
    /// Keep the phone's cache budget on the desktop as well (see
    /// <see cref="Hosting.EngineMemoryPolicy"/>): caches that start small and grow, at most a
    /// little of the reply reserved ahead, one finished conversation kept, nothing parked.
    /// The engine's desktop defaults are written for a machine with memory to spare; for a
    /// model whose weights take most of a device's memory there is none, and they are what
    /// fills it.
    /// </summary>
    public bool LeanCaches { get; init; }
    public required CatalogSampling Sampling { get; init; }
    /// <summary>Whether the family has a thinking channel the app may enable.</summary>
    public bool SupportsThinking { get; init; }
    /// <summary>
    /// The card describes an exact, hash-pinned artifact but has no publisher URL the
    /// app can verify. The Models page offers a file picker instead of a download and
    /// <see cref="ModelStore.ImportAsync"/> verifies the selected bytes before exposing
    /// them to the engine.
    /// </summary>
    public bool SideloadOnly { get; init; }
    /// <summary>Marked in the UI: requires a constrained memory budget or has limited
    /// device/backend validation. See the entry's notes for its tested configuration.</summary>
    public bool Experimental { get; init; }
    public string? Notes { get => CatalogText.Read(_notes); init => _notes = value; }
    public required string License { get => CatalogText.Read(_license); init => _license = value; }
    /// <summary>
    /// Bytes of the weights the engine reads from the SSD on demand instead of holding them
    /// resident: zero for every entry whose weights a token reads in full. Qwen3.8 Flash Next
    /// declares its n-gram table (a token reads 16 of its 320 M rows) and the routed experts
    /// the engine offloads on the entry's smallest device (a token reads 10 of each layer's
    /// 512), which lets its files exceed system RAM. The residency checks hold
    /// <see cref="ResidentWeightsBytes"/> to the device instead of the whole file.
    /// </summary>
    public long WeightsPagedFromDiskBytes { get; init; }

    public long TotalBytes => Files.Where(f => !f.Optional).Sum(f => f.Bytes);
    /// <summary>The weights GGUF and its later shards, in shard order.</summary>
    public IEnumerable<CatalogFile> WeightFiles =>
        Files.Where(f => f.Role is CatalogFileRole.Weights or CatalogFileRole.WeightsShard);
    /// <summary>Every byte of the weights, all shards.</summary>
    public long WeightsBytes => WeightFiles.Sum(f => f.Bytes);
    /// <summary>The weights a token reads in full: everything not paged from disk on demand.</summary>
    public long ResidentWeightsBytes => WeightsBytes - WeightsPagedFromDiskBytes;
    public long TotalBytesWithOptional => Files.Sum(f => f.Bytes);
    public CatalogFile Weights => Files.First(f => f.Role == CatalogFileRole.Weights);
    public CatalogFile? Projector => Files.FirstOrDefault(f => f.Role == CatalogFileRole.Projector);
    public bool IsImageGenerator => Family == CatalogFamily.QwenImage;

    /// <summary>
    /// For a Qwen-Image entry, which 2.1 checkpoint its weights are. The GGUFs carry no metadata
    /// and Turbo has the base checkpoint's tensors, so the entry declares it and
    /// <see cref="DiffusionCompanions"/> publishes it to the engine
    /// (<see cref="QwenImageVariantFlag.EnvironmentVariable"/>): the sampling schedule follows the
    /// entry, never the file name. Turbo also decides which LoRA plug-ins the entry takes
    /// (<see cref="CatalogLora.AppliesTo"/>). Meaningless for other families.
    /// </summary>
    public QwenImageVariant ImageVariant { get; init; } = QwenImageVariant.Base;
    public bool IsVideoGenerator => Modalities.HasFlag(CatalogModalities.VideoOutput);
}
