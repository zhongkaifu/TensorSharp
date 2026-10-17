// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Core.Catalog;

/// <summary>What a LoRA plug-in does to a picture or a clip, which decides how it may be combined.</summary>
public enum LoraKind
{
    /// <summary>A step-distillation plug-in: the same picture or clip in a few steps instead of the
    /// model's own (40 for Qwen-Image 2.1, 20 for MiniMax-H3), with its own sampling schedule. Used
    /// at the strength it was trained at, and only one per model at a time: two schedules cannot
    /// both apply (the engine refuses the pair).</summary>
    Speed,
    /// <summary>Changes how pictures look. Combines with a speed plug-in and with other styles.</summary>
    Style,
    /// <summary>Trained for one kind of edit of an attached photo, usually with a phrase the
    /// request should contain.</summary>
    Edit,
}

/// <summary>One file of a LoRA plug-in, pinned to the bytes its publisher uploaded.</summary>
/// <param name="FileName">The name it is stored under, inside the plug-in's folder.</param>
/// <param name="Url">A Hugging Face resolve URL at the commit the bytes were verified at.</param>
/// <param name="Bytes">Exact size.</param>
/// <param name="Sha256">Lower-case SHA-256, verified before the file is published.</param>
public sealed record LoraFile(string FileName, string Url, long Bytes, string Sha256);

/// <summary>
/// A LoRA plug-in for a diffusion model in <see cref="ModelCatalog"/>: a small adapter the
/// engine applies on top of the model's own weights, unmerged, to every later picture or clip.
/// </summary>
public sealed record CatalogLora
{
    private readonly string _purpose = string.Empty;
    private readonly string _license = string.Empty;

    /// <summary>Stable id, also the plug-in's folder name and the stem of the engine's own
    /// config for it in <c>config/lora/</c>.</summary>
    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    /// <summary>The <see cref="ModelCatalog"/> entry it adapts.</summary>
    public required string BaseModelId { get; init; }
    /// <summary>
    /// Other entries it applies to beside <see cref="BaseModelId"/>. Turbo style/edit adapters
    /// are checked with real pictures on at least one quantization. The same-architecture Q8
    /// entry shares compatibility; that does not imply separate image validation on Q8 (see
    /// docs/models/qwenimage21.md for measured scope). Never a speed plug-in: Turbo already
    /// supplies its own schedule, and the engine refuses a second one.
    /// </summary>
    public IReadOnlyList<string> AlsoFor { get; init; } = Array.Empty<string>();
    public required LoraKind Kind { get; init; }
    /// <summary>The adapter weights (always first) and any file the engine reads beside them.</summary>
    public required IReadOnlyList<LoraFile> Files { get; init; }
    /// <summary>The publisher's recommended strength, or a documented starting value when none
    /// is specified. A speed plug-in is always used at it.</summary>
    public float DefaultStrength { get; init; } = 1f;
    /// <summary>A speed plug-in's step count; null for the others, which keep the model's own schedule.</summary>
    public int? Steps { get; init; }
    /// <summary>
    /// The engine's plug-in config (<c>config/lora/&lt;id&gt;.json</c>, embedded in this assembly)
    /// that carries the sampling schedule, for a speed plug-in whose weights do not: the engine
    /// is handed it as the plug-in's config. A MiniMax-H3 plug-in's config also declares the
    /// checkpoint it is for and its alpha, and the engine ties it to the weights by file name.
    /// Null when nothing beyond the weights is needed.
    /// </summary>
    public string? RecipeConfig { get; init; }
    /// <summary>The engine's own config for a plug-in whose third-party config file ships with
    /// it (Fun-Acc's <c>pdd_config.json</c>): the file name in <see cref="Files"/> to pass.</summary>
    public string? ConfigFile { get; init; }
    /// <summary>What a user gets from it, in a sentence or two, in the interface language
    /// (see <see cref="CatalogText"/>, as for <see cref="License"/>).</summary>
    public required string Purpose { get => CatalogText.Read(_purpose); init => _purpose = value; }
    /// <summary>A phrase the request should contain for the plug-in to do its job, if any. Never
    /// translated: it is the wording the plug-in was trained on, and the model reads it.</summary>
    public string? Trigger { get; init; }
    /// <summary>True when it only does anything to an attached photo.</summary>
    public bool NeedsPhoto { get; init; }
    /// <summary>
    /// Its task works only on the model's own schedule, so a speed plug-in sits out the pictures
    /// it is applied to. MEASURED on Object Remover's own example photos: on the cats it removes
    /// both marked cats at the model's 40 steps and one of them with Viggle Turbo's 6; on the car
    /// park it removed none of the three marked cars at either (the boxes went, the cars stayed).
    /// It was trained and shown at 40 steps only.
    /// </summary>
    public bool NeedsModelSteps { get; init; }
    public required string License { get => CatalogText.Read(_license); init => _license = value; }

    /// <summary>Whether it applies to the catalog entry <paramref name="modelId"/>: its own base
    /// model or one of <see cref="AlsoFor"/>.</summary>
    public bool AppliesTo(string? modelId) =>
        modelId is not null
        && (string.Equals(BaseModelId, modelId, StringComparison.OrdinalIgnoreCase)
            || AlsoFor.Contains(modelId, StringComparer.OrdinalIgnoreCase));

    public LoraFile Weights => Files[0];
    public long TotalBytes => Files.Sum(f => f.Bytes);
    public bool StrengthAdjustable => Kind != LoraKind.Speed;
}

/// <summary>
/// The LoRA plug-ins the app offers: the fourteen for Qwen-Image 2.1 and the two MiniMax-H3
/// Turbo plug-ins the engine ships configs for in <c>config/lora/</c>, at the same pinned files
/// (<c>LoraCatalogTests</c> keeps the two equal). Sizes and hashes were read from the Hugging
/// Face tree API at the pinned commits on 2026-10-01 (the two added Qwen adapters on 2026-10-10,
/// the MiniMax-H3 ones on 2026-10-16) and checked against downloaded copies.
///
/// <para>
/// They are add-ons, not catalog entries: a plug-in is nothing without its base model, is a
/// tenth of its size or less, and is chosen per picture or clip rather than loaded. They live
/// beside the models (<see cref="LoraStore"/>), are downloaded one at a time on request, and
/// the ones the user turns on are applied to the loaded model before each picture or clip.
/// </para>
/// </summary>
public static class LoraCatalog
{
    public const string QwenImage = "qwen-image-2.1-q4km";

    /// <summary>The same-architecture Qwen-Image 2.1 Turbo entries. Style/edit adapters apply
    /// to both, with image-validation scope recorded separately; speed adapters never do.</summary>
    public static readonly IReadOnlyList<string> QwenImageTurbo = new[] { "qwen-image-2.1-turbo-adq4k", "qwen-image-2.1-turbo-q8" };

    /// <summary>MiniMax-H3's first-and-last-frame (FL2VA) entry, the one checkpoint its Turbo
    /// plug-ins are for: the references (Ref2VA) entry has the same tensor shapes, and the engine
    /// refuses them there by the checkpoint's file name.</summary>
    public const string MiniMaxH3Keyframes = "minimax-h3-fl2va-q4k";

    /// <summary>The commit of <c>lightx2v/Minimax-h3-Turbo</c> the MiniMax-H3 plug-ins are pinned to.</summary>
    private const string MiniMaxH3TurboCommit = "a29e57d195f80ead07c89f89e55ecf2911e893c5";

    /// <summary>The Qwen Research License of the base model and most plug-ins.</summary>
    private const string QwenResearch = "catalog.license.qwenResearch";
    private const string NoneStated = "catalog.license.noneStated";

    private static string Hf(string repo, string commit, string path) =>
        $"https://huggingface.co/{repo}/resolve/{commit}/{path}";

    public static IReadOnlyList<CatalogLora> BuiltIn { get; } = new[]
    {
        new CatalogLora
        {
            Id = "qwen-image-2.1-viggle-turbo",
            DisplayName = "Viggle Turbo",
            BaseModelId = QwenImage,
            Kind = LoraKind.Speed,
            Steps = 6,
            RecipeConfig = "qwen-image-2.1-viggle-turbo.json",
            Files = new[]
            {
                new LoraFile("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
                    Hf("Viggle/Qwen-Image-2.1-viggle-turbo", "bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13",
                        "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"),
                    679_604_800, "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b"),
            },
            Purpose = "catalog.lora.viggleTurbo.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-pruna-8step",
            DisplayName = "Pruna 8-Step",
            BaseModelId = QwenImage,
            Kind = LoraKind.Speed,
            Steps = 8,
            RecipeConfig = "qwen-image-2.1-pruna-8step.json",
            Files = new[]
            {
                new LoraFile("p_qwen_image_2.1_8step_v0.1.safetensors",
                    Hf("PrunaAI/Pruna-Qwen-Image-2.1", "113e63bb993001b3411eb3470b84fc444040cd7e",
                        "p_qwen_image_2.1_8step_v0.1.safetensors"),
                    335_606_104, "f0865d68b02511a3a0ed232d9d1aa99cac3a94166f574b38bcbab1a7a297bb15"),
            },
            Purpose = "catalog.lora.pruna8Step.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-pruna-5step",
            DisplayName = "Pruna 5-Step",
            BaseModelId = QwenImage,
            Kind = LoraKind.Speed,
            Steps = 5,
            RecipeConfig = "qwen-image-2.1-pruna-5step.json",
            Files = new[]
            {
                new LoraFile("p_qwen_image_2.1_5step_v0.1.safetensors",
                    Hf("PrunaAI/Pruna-Qwen-Image-2.1", "113e63bb993001b3411eb3470b84fc444040cd7e",
                        "p_qwen_image_2.1_5step_v0.1.safetensors"),
                    335_606_144, "021a6228a0fcd217275190b89072416a2f28548e8e18fe049531dc9b77ef89bd"),
            },
            Purpose = "catalog.lora.pruna5Step.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-fun-acc-4step",
            DisplayName = "Fun-Acc 4-Step",
            BaseModelId = QwenImage,
            Kind = LoraKind.Speed,
            Steps = 4,
            // A parallel-decoding bundle: per-step output heads and its own schedule, which the
            // engine reads from pdd_config.json beside the weights.
            ConfigFile = "pdd_config.json",
            Files = new[]
            {
                new LoraFile("Qwen-Image-2.1-Fun-Acc-4Step.safetensors",
                    Hf("alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs", "f7545234760e1847cd8e89e52bd951cb0b7e327f",
                        "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors"),
                    345_632_504, "764c56ae94f330b6d06ccc322f95e1b8ce46424ddde5899a15432f95f720d558"),
                new LoraFile("pdd_config.json",
                    Hf("alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs", "f7545234760e1847cd8e89e52bd951cb0b7e327f",
                        "models/pdd_config.json"),
                    11_356, "f798c4a8e9225350e9c46be7a60f397df35e6bb90965d64be76d11e51fd41b97"),
            },
            Purpose = "catalog.lora.funAcc4Step.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-film-stills",
            DisplayName = "Film Stills",
            BaseModelId = QwenImage,
            // Validated on Turbo at 0.7, seeds 42 and 7 (qwen-image21-turbo.py --suite style):
            // the look applies and part of the scene is redrawn
            // (docs/models/qwenimage21.md#lora-plug-ins-on-turbo).
            AlsoFor = QwenImageTurbo,
            Kind = LoraKind.Style,
            DefaultStrength = 0.7f,
            Files = new[]
            {
                new LoraFile("filmstills_qwen21.safetensors",
                    Hf("Danrisi/filmstills_qwen2.1", "29242138fcc2eea95970fe7b66e6e17c4bb373a8", "filmstills_qwen21.safetensors"),
                    79_743_888, "ad4812903c6f6b8966885d09dff23344422d0d1f8b2bebad038f8a802f55e30a"),
            },
            Purpose = "catalog.lora.filmStills.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-grainscape",
            DisplayName = "Grainscape",
            BaseModelId = QwenImage,
            // Validated on Turbo at 0.7, seeds 42 and 7 (qwen-image21-turbo.py --suite style):
            // the look applies and part of the scene is redrawn
            // (docs/models/qwenimage21.md#lora-plug-ins-on-turbo).
            AlsoFor = QwenImageTurbo,
            Kind = LoraKind.Style,
            DefaultStrength = 0.7f,
            Files = new[]
            {
                new LoraFile("grainscape_qwen21.safetensors",
                    Hf("Danrisi/grainscape_qwen2.1", "bc903faddb7b6106650859c7ddd34610d014ae8e", "grainscape_qwen21.safetensors"),
                    79_743_888, "b04b226561c5c3cc65a8e5f79a63a831ffb068d8bd3f156271650cb46e1dc2bf"),
            },
            Purpose = "catalog.lora.grainscape.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-casual-phone",
            DisplayName = "Casual Phone Photo",
            BaseModelId = QwenImage,
            AlsoFor = QwenImageTurbo,
            Kind = LoraKind.Style,
            DefaultStrength = 0.8f,
            Trigger = "lbxstyle",
            Files = new[]
            {
                new LoraFile("lbxstyle-qwen-image-2.1-step-1890.safetensors",
                    Hf("19nbrown/qwen-image-2.1-casual-phone-photo-style-lora", "3ace7de571892ac5fa6245aed4f0500ae1da795a",
                        "lbxstyle-qwen-image-2.1-step-1890.safetensors"),
                    167_830_056, "c23c23e3346ab93e0af121888b34f2176347f18dac26ade87eacf21fc11bd9dc"),
            },
            Purpose = "catalog.lora.casualPhone.purpose",
            License = NoneStated,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-voxel-style",
            DisplayName = "Voxel Style",
            BaseModelId = QwenImage,
            AlsoFor = QwenImageTurbo,
            Kind = LoraKind.Edit,
            NeedsPhoto = true,
            Trigger = "Transform the image into a voxel style",
            Files = new[]
            {
                new LoraFile("Qwen-Image-2.1-Voxel-Style-3000.safetensors",
                    Hf("prithivMLmods/Qwen-Image-2.1-Voxel-Style", "b4ccfe9333f04c2f978f2d1bfdfdafe287cbfad1",
                        "Qwen-Image-2.1-Voxel-Style-3000.safetensors"),
                    83_943_936, "165e50dfe0de46276208edf76bceb075dc65ceb1867d78c5844ad23af9a14b25"),
            },
            Purpose = "catalog.lora.voxelStyle.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-fix",
            DisplayName = "Quality Fix",
            BaseModelId = QwenImage,
            Kind = LoraKind.Style,
            Files = new[]
            {
                new LoraFile("qwen-image-2.1-fix-1.0-comfy.safetensors",
                    Hf("e-n-v-y/Qwen-Image-2.1-Fix", "5b2c7be6ced92cc2e09f9309ad00567b3a3bb6ce", "qwen-image-2.1-fix-1.0-comfy.safetensors"),
                    111_612_000, "e4a369158b957aee3a8316d94dbe00c98f7d648ef3d1926ccf7283915b6db60c"),
            },
            Purpose = "catalog.lora.qualityFix.purpose",
            License = NoneStated,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-detail-enhancer",
            DisplayName = "Detail Enhancer",
            BaseModelId = QwenImage,
            Kind = LoraKind.Edit,
            NeedsPhoto = true,
            Trigger = "Enhance this image",
            Files = new[]
            {
                new LoraFile("elusarcas-qwen2-1-detailer-v1.safetensors",
                    Hf("reverentelusarca/elusarcas-qwen-2.1-detail-enhancer-lora", "aedadc6b65a15e6b261c0ec61a4587c3e7119514",
                        "elusarcas-qwen2-1-detailer-v1.safetensors"),
                    79_744_352, "c1298f51eb090473f924314475952710a2f0322f1df65c0290f3e14997244f96"),
            },
            Purpose = "catalog.lora.detailEnhancer.purpose",
            License = NoneStated,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-natural-exposure",
            DisplayName = "Natural Exposure",
            BaseModelId = QwenImage,
            Kind = LoraKind.Edit,
            NeedsPhoto = true,
            Trigger = "Transform the image with balanced neutral exposure",
            Files = new[]
            {
                new LoraFile("Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors",
                    Hf("prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA", "382d066d079854a86c9513f3c5a4026ada21ddbe",
                        "Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors"),
                    83_943_952, "a8edea397ce55ae6e1a442f3ba963eb78beeb515d4fe7d217d496542317292b1"),
            },
            Purpose = "catalog.lora.naturalExposure.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-object-remover",
            DisplayName = "Object Remover",
            BaseModelId = QwenImage,
            Kind = LoraKind.Edit,
            NeedsPhoto = true,
            NeedsModelSteps = true,
            Trigger = "Remove the red highlighted object from the scene",
            Files = new[]
            {
                new LoraFile("Qwen-Image-2.1-Object-Remover-Bbox-turbo-4000.safetensors",
                    Hf("prithivMLmods/Qwen-Image-2.1-Object-Remover-Bbox-turbo", "90f708f53feb896a2873378629ffe414e0e584c0",
                        "Qwen-Image-2.1-Object-Remover-Bbox-turbo-4000.safetensors"),
                    83_943_952, "7035eba6f25cdbd0c780027372710c9bebebdd814baaa79d48ce9d5ca80d6449"),
            },
            Purpose = "catalog.lora.objectRemover.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-object-mover",
            DisplayName = "Object Mover",
            BaseModelId = QwenImage,
            Kind = LoraKind.Edit,
            NeedsPhoto = true,
            Trigger = "Move the object highlighted in the red box to the location indicated by the other red box in the scene.",
            Files = new[]
            {
                new LoraFile("Qwen-Image-2.1-Object-Mover-Bbox-Preview-5000.safetensors",
                    Hf("prithivMLmods/Qwen-Image-2.1-Object-Mover-Bbox-Preview", "fdae922a766c3cdae63a28caa8a123ad82f1f29f",
                        "Qwen-Image-2.1-Object-Mover-Bbox-Preview-5000.safetensors"),
                    83_944_000, "b596b5c81ea254b22acecc1801f24f3b42700c954f7ac7a333f644d217c32dcd"),
            },
            Purpose = "catalog.lora.objectMover.purpose",
            License = QwenResearch,
        },
        new CatalogLora
        {
            Id = "qwen-image-2.1-anime-consistency",
            DisplayName = "Anime Consistency",
            BaseModelId = QwenImage,
            Kind = LoraKind.Edit,
            DefaultStrength = 0.7f,
            NeedsPhoto = true,
            Files = new[]
            {
                new LoraFile("Qwen2.1_Anime_consistency.safetensors",
                    Hf("WarmBloodAban/Qwen-Image-2.1-LoRAs", "90abd1e7f590fc8ba052e0d2e6de9dc01730e260",
                        "Qwen2.1_Anime_consistency.safetensors"),
                    167_830_408, "0c171eb802ea8051b511f2d93c1743eeadd030316255aa98fa65d40809366752"),
            },
            Purpose = "catalog.lora.animeConsistency.purpose",
            License = "Apache-2.0",
        },
        // MiniMax-H3 Turbo: step distillation for the keyframes checkpoint, applied unmerged per
        // clip. The config fixes the step count, the linspace grid, video shift 6 / audio 3,
        // Euler and CFG 1, and a clip that asks for anything else is refused, never overridden;
        // the app never puts steps, a shift or a sampler in a clip's request (VideoTurns).
        new CatalogLora
        {
            Id = "minimax-h3-fl2v-turbo-8step",
            DisplayName = "LightX2V Turbo 8-Step",
            BaseModelId = MiniMaxH3Keyframes,
            Kind = LoraKind.Speed,
            Steps = 8,
            RecipeConfig = "minimax-h3-fl2v-turbo-8step.json",
            Files = new[]
            {
                new LoraFile("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
                    Hf("lightx2v/Minimax-h3-Turbo", MiniMaxH3TurboCommit, "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"),
                    1_383_677_808, "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4"),
            },
            Purpose = "catalog.lora.minimaxH3Turbo8Step.purpose",
            License = "Apache-2.0",
        },
        // Its schedule is INFERRED: upstream publishes no recipe for v1.2, so the config carries
        // the 4-step v1.0 768p row's (linspace, shift 6/3) until it is validated
        // (config/lora/minimax-h3-fl2v-turbo-4step.json).
        new CatalogLora
        {
            Id = "minimax-h3-fl2v-turbo-4step",
            DisplayName = "LightX2V Turbo 4-Step",
            BaseModelId = MiniMaxH3Keyframes,
            Kind = LoraKind.Speed,
            Steps = 4,
            RecipeConfig = "minimax-h3-fl2v-turbo-4step.json",
            Files = new[]
            {
                new LoraFile("minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors",
                    Hf("lightx2v/Minimax-h3-Turbo", MiniMaxH3TurboCommit, "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"),
                    1_383_677_808, "c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689"),
            },
            Purpose = "catalog.lora.minimaxH3Turbo4Step.purpose",
            License = "Apache-2.0",
        },
    };

    /// <summary>The strengths a user may choose for a plug-in whose strength is adjustable.</summary>
    public const float MinStrength = 0.1f;
    public const float MaxStrength = 1.5f;

    public static CatalogLora? Find(string id) =>
        BuiltIn.FirstOrDefault(l => string.Equals(l.Id, id, StringComparison.OrdinalIgnoreCase));

    /// <summary>The plug-ins that apply to the entry <paramref name="modelId"/>, in catalog order.</summary>
    public static IReadOnlyList<CatalogLora> For(string? modelId) =>
        BuiltIn.Where(l => l.AppliesTo(modelId)).ToList();

    /// <summary>
    /// The plug-ins the LoRA sheet lists: those that apply to the loaded image or video model, so
    /// a Qwen-Image 2.1 Turbo entry is never offered a speed plug-in and MiniMax-H3 References,
    /// LTX 2.5 or Wan none; with no image or video model loaded, every plug-in of a model in
    /// <paramref name="offered"/>.
    /// </summary>
    public static IReadOnlyList<CatalogLora> Offered(CatalogModel? loaded, IEnumerable<CatalogModel> offered)
    {
        ArgumentNullException.ThrowIfNull(offered);
        CatalogModel[] models = offered.ToArray();
        return BuiltIn.Where(l => loaded is { IsImageGenerator: true } or { IsVideoGenerator: true }
            ? l.AppliesTo(loaded.Id)
            : models.Any(m => l.AppliesTo(m.Id))).ToList();
    }

    /// <summary>The text of an embedded engine config (<see cref="CatalogLora.RecipeConfig"/>).</summary>
    public static string RecipeText(CatalogLora lora)
    {
        ArgumentNullException.ThrowIfNull(lora);
        if (lora.RecipeConfig is not { } name)
            throw new InvalidOperationException($"{lora.Id} has no recipe config.");
        using Stream stream = typeof(LoraCatalog).Assembly.GetManifestResourceStream("lora/" + name)
            ?? throw new InvalidOperationException($"The recipe config {name} is not embedded in this build.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }
}
