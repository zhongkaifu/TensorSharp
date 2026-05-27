// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Matrix;

/// <summary>
/// One feature / prompt type exercised by the matrix. Captures both
/// the metadata for filtering and the CLI flag template that drives
/// TensorSharp.Cli for this feature.
/// </summary>
public sealed record FeatureSpec(
    string Id,
    string DisplayName,
    FeatureKind Kind,
    string? PromptFile,
    string? MediaFile,
    string? ToolsFile,
    string? MultiTurnFile,
    int PrefillTokens,
    int DecodeTokens,
    int MaxTokens,
    bool EnableThinking,
    bool RequiresImage,
    bool RequiresAudio,
    bool RequiresVideo,
    bool RequiresTools)
{
    /// <summary>
    /// Substrings the assistant output must contain for the cell to pass
    /// correctness. Case-insensitive. Empty means "no semantic check" — only
    /// applicable to synthetic benchmark modes that don't generate real text.
    /// </summary>
    public IReadOnlyList<string> ExpectedContains { get; init; } = Array.Empty<string>();
}

public enum FeatureKind
{
    SyntheticPrefill,
    SyntheticDecode,
    Text,
    UploadedText,
    Image,
    Audio,
    Video,
    Tools,
    Thinking,
    MultiTurn,
}

public static class FeatureCatalog
{
    public const int DefaultMaxTokens = 64;

    public static readonly FeatureSpec SyntheticPrefill512 = new(
        Id: "pp512",
        DisplayName: "Synthetic prefill (512 tok)",
        Kind: FeatureKind.SyntheticPrefill,
        PromptFile: null,
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 512,
        DecodeTokens: 0,
        MaxTokens: 0,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false);

    public static readonly FeatureSpec SyntheticPrefill2048 = SyntheticPrefill512 with
    {
        Id = "pp2048",
        DisplayName = "Synthetic prefill (2048 tok)",
        PrefillTokens = 2048,
    };

    public static readonly FeatureSpec SyntheticDecode128 = new(
        Id: "tg128",
        DisplayName: "Synthetic decode (128 tok after 32 prefill)",
        Kind: FeatureKind.SyntheticDecode,
        PromptFile: null,
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 32,
        DecodeTokens: 128,
        MaxTokens: 0,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false);

    public static readonly FeatureSpec ShortText = new(
        Id: "short_text",
        DisplayName: "Short text prompt (single turn)",
        Kind: FeatureKind.Text,
        PromptFile: "prompts/short_text.txt",
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: DefaultMaxTokens,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // The prompt asks why the sky is blue. Any correct answer must
        // mention "blue" and at least gesture at scattering/wavelength.
        ExpectedContains = new[] { "blue" },
    };

    public static readonly FeatureSpec LongText = ShortText with
    {
        Id = "long_text",
        DisplayName = "Long text prompt (~1k tokens, single turn)",
        PromptFile = "prompts/long_text.txt",
        // Summary of a paged-KV-cache report. Any correct summary names
        // either the data structure or the technique.
        ExpectedContains = new[] { "paged" },
    };

    public static readonly FeatureSpec UploadedText = ShortText with
    {
        Id = "uploaded_text",
        DisplayName = "Uploaded text file (large, truncated)",
        PromptFile = "prompts/upload_text.txt",
        Kind = FeatureKind.UploadedText,
        // Log analysis; both bullets must surface the ERROR line. We assert
        // the exact error timestamp from the log.
        ExpectedContains = new[] { "08:01:12" },
    };

    public static readonly FeatureSpec MultiTurn = new(
        Id: "multi_turn",
        DisplayName: "Multi-turn chat (KV reuse)",
        Kind: FeatureKind.MultiTurn,
        PromptFile: null,
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: "multi_turn/three_turn.jsonl",
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: DefaultMaxTokens,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // Turn 1 establishes 'Alex' / 'teal'; turn 2 asks for the name back,
        // turn 3 asks for the colour back. If KV reuse is correct the model
        // must surface both facts somewhere in the combined turn-2 + turn-3
        // assistant output.
        ExpectedContains = new[] { "alex", "teal" },
    };

    public static readonly FeatureSpec Tools = new(
        Id: "tools",
        DisplayName: "Function / tool calling",
        Kind: FeatureKind.Tools,
        PromptFile: "prompts/tools_question.txt",
        MediaFile: null,
        ToolsFile: "tools/weather_tools.json",
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: 128,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: true)
    {
        // The prompt asks about Tokyo weather and exposes one tool,
        // get_current_weather. The model must name the tool and the city.
        ExpectedContains = new[] { "get_current_weather", "tokyo" },
    };

    public static readonly FeatureSpec Thinking = new(
        Id: "thinking",
        DisplayName: "Thinking / reasoning mode",
        Kind: FeatureKind.Thinking,
        PromptFile: "prompts/thinking_question.txt",
        MediaFile: null,
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: 256,
        EnableThinking: true,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // Two-train word problem.
        //   A leaves at 09:00 from station A, 60 km/h, toward B (200 km).
        //   B leaves at 09:30 from station B, 90 km/h, toward A.
        // By 09:30 A has covered 30 km, leaving 170 km. Closing speed
        // 60 + 90 = 150 km/h, so 170/150 h = 1 h 8 min from 09:30 => 10:38.
        ExpectedContains = new[] { "10:38" },
    };

    public static readonly FeatureSpec Image = new(
        Id: "image",
        DisplayName: "Image prompt",
        Kind: FeatureKind.Image,
        PromptFile: "prompts/image_question.txt",
        MediaFile: "media/apple.png",
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: DefaultMaxTokens,
        EnableThinking: false,
        RequiresImage: true,
        RequiresAudio: false,
        RequiresVideo: false,
        RequiresTools: false)
    {
        // Default media is apple.png. If a runner replaces it with something
        // else, this spec needs to be overridden — see README.
        ExpectedContains = new[] { "apple" },
    };

    public static readonly FeatureSpec Audio = new(
        Id: "audio",
        DisplayName: "Audio prompt",
        Kind: FeatureKind.Audio,
        PromptFile: "prompts/audio_question.txt",
        MediaFile: "media/sample.mp3",
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: DefaultMaxTokens,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: true,
        RequiresVideo: false,
        RequiresTools: false);
    // No default ExpectedContains: audio content depends on whatever
    // sample.mp3 the runner provides. Override per-deployment.

    public static readonly FeatureSpec Video = new(
        Id: "video",
        DisplayName: "Video prompt",
        Kind: FeatureKind.Video,
        PromptFile: "prompts/video_question.txt",
        MediaFile: "media/sample.mp4",
        ToolsFile: null,
        MultiTurnFile: null,
        PrefillTokens: 0,
        DecodeTokens: 0,
        MaxTokens: DefaultMaxTokens,
        EnableThinking: false,
        RequiresImage: false,
        RequiresAudio: false,
        RequiresVideo: true,
        RequiresTools: false);
    // No default ExpectedContains for video for the same reason as audio.

    public static readonly IReadOnlyList<FeatureSpec> All = new[]
    {
        SyntheticPrefill512,
        SyntheticPrefill2048,
        SyntheticDecode128,
        ShortText,
        LongText,
        UploadedText,
        MultiTurn,
        Tools,
        Thinking,
        Image,
        Audio,
        Video,
    };

    public static FeatureSpec? FindById(string id)
    {
        foreach (FeatureSpec f in All)
        {
            if (string.Equals(f.Id, id, StringComparison.OrdinalIgnoreCase))
            {
                return f;
            }
        }
        return null;
    }
}
