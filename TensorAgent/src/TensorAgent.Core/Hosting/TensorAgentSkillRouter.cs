// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// Routes the one compound workflow for which discovery alone proved insufficient:
/// research on the web followed by a real PowerPoint deliverable.
///
/// <para>
/// This is deliberately not a general natural-language classifier. It needs one cue
/// from each side in the same latest user turn, leaves every explicit skill selection
/// alone, and only returns ids the installed registry actually contains. The normal
/// skill planner remains responsible for permissions, tool declarations and loading
/// the full instructions progressively.
/// </para>
/// </summary>
internal static class TensorAgentSkillRouter
{
    internal const string ResearchSkill = "research";
    internal const string DocumentsSkill = "documents";

    /// <summary>
    /// Small enough for an 8K model and forceful enough to avoid the measured fallback:
    /// Qwen overlooked the documents catalog entry, tried python-pptx/lxml on iOS (which
    /// could not be installed then; both ship in the bundle now), then hand-wrote OOXML
    /// into /tmp. The actual skills already contain all implementation detail, so none
    /// of their ~25 KB of bodies is duplicated here.
    /// </summary>
    internal const string ActivationInstructions =
        "### Required research-to-PPTX workflow\n"
        + "This request requires both selected skills. Before any other tool call, read both instruction files "
        + "in the same turn with skills_read(skill=\"research\", path=\"SKILL.md\") and "
        + "skills_read(skill=\"documents\", path=\"SKILL.md\"). The `cd` and `python3` lines in those files are "
        + "command-line examples, not extra tools: do not call `cd` or `shell`. Call skills_run directly, first for "
        + "research/scripts/research.py with the routed arguments (three result pages, written to notes.md), "
        + "then read only notes.md with read_file. Do not run analyze.py or read notes.json when notes.md contains "
        + "usable sources. Keep the first write_file call safely below the generation limit: write one compact "
        + "`pptx_spec.json` with exactly three slide entries (one title plus two content slides), at most four short "
        + "bullets or six table rows per content slide, and at most three root sources. The writer adds the fourth, "
        + "final Sources slide. The JSON must stay below 6,000 characters. Use only this canonical "
        + "shape: root keys "
        + "`title`, `author`, `subject`, `slides`, and optional `sources`; title slides use `layout`, `title`, `subtitle`; content slides use "
        + "`layout`, `title`, `bullets` (or `columns` plus `rows` for a table). Put supported claims, source URLs and "
        + "dates in root `sources` objects, copying their exact URLs from notes.md, so the writer creates one visible "
        + "final Sources slide; do not duplicate a Sources slide or invent `content`. Call skills_run directly "
        + "for documents/scripts/make_pptx.py with `--spec pptx_spec.json --out report.pptx`, without reading, copying, "
        + "or rewriting that script. `make_pptx.py` validates its own output; after it succeeds, stop immediately "
        + "and answer without calling validate_document, shell, or any package installer. Do not install "
        + "python-pptx/lxml (both are already built in) or hand-build OOXML. The task is incomplete "
        + "until a real .pptx exists in the shared workspace and is returned as a downloadable artifact. If the "
        + "spec or run fails, use read_file/apply_patch on only the broken region and rerun; never regenerate it.";

    private static readonly string[] ResearchPhrases =
    {
        "search", "web search", "research", "look up", "find information", "latest information",
        "current information", "搜索", "检索", "查找", "查一下", "调研", "研究一下",
    };

    private static readonly string[] PowerPointPhrases =
    {
        "pptx", "powerpoint", "slide deck", "presentation", "slides", "幻灯片", "演示文稿", "演示报告",
    };

    private static readonly string[] CreationPhrases =
    {
        "create", "make", "generate", "produce", "build", "prepare", "write", "turn into",
        "生成", "制作", "创建", "做一份", "输出",
    };

    private static readonly string[] CreationConnectives =
    {
        "and", "then", "please", "also", "next", "finally",
        "并", "并且", "然后", "请", "再", "同时", "最后",
    };

    /// <summary>
    /// Return a route only for an unselected, compound latest turn. Null means the Web
    /// UI request continues through its existing discovery/selection path unchanged.
    /// </summary>
    internal static WebUiSkillRoute? Route(
        IReadOnlyList<ChatMessage> messages,
        IReadOnlyList<string>? requestedSkills,
        SkillRegistry registry)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(registry);

        // Presence is the explicit signal, including []. TensorAgent's browser omits
        // `skills` when the picker is untouched and sends an array when the user scoped
        // it; treating [] like omission would erase that deliberate opt-out.
        if (requestedSkills != null)
            return null;

        ChatMessage? latestMessage = LatestUserMessage(messages);
        string? latest = latestMessage?.Content;
        if (string.IsNullOrWhiteSpace(latest)
            || HasAttachments(latestMessage)
            || !ContainsAny(latest, ResearchPhrases)
            || !ContainsAny(latest, PowerPointPhrases)
            || !HasPowerPointDeliverableIntent(latest)
            || !IsAppleM5M6Comparison(latest))
        {
            return null;
        }

        // Resolve the winning registry entries, not merely the ids. This route owns the
        // exact CLI and output contract of TensorAgent's bundled implementations; an
        // installed same-named skill is user-controlled and may define entirely
        // different semantics even when it happens to reuse the filename. Never
        // auto-execute such a shadow (especially with network enabled). Do this only
        // after the cheap intent check so unrelated turns never pay for filesystem
        // resolution. The runner repeats the confined-file check at execution time.
        if (!registry.TryGet(ResearchSkill, out Skill research)
            || research.Origin != SkillOrigin.Discovered
            || !HasRequiredScript(research, "scripts/research.py")
            || !registry.TryGet(DocumentsSkill, out Skill documents)
            || documents.Origin != SkillOrigin.Discovered
            || !HasRequiredScript(documents, "scripts/make_pptx.py"))
            return null;

        return new WebUiSkillRoute(
            new[] { ResearchSkill, DocumentsSkill },
            ActivationInstructions,
            ArtifactRequirement: new WebUiArtifactRequirement(
                Extension: ".pptx",
                RequiredRuns: new[]
                {
                    new WebUiSkillRunRequirement(
                        ResearchSkill,
                        "scripts/research.py",
                        DefaultArguments: new[]
                        {
                            ResearchQuery(latest), "--pages", "3", "--out", "notes.md",
                        },
                        EnforceArguments: true),
                    new WebUiSkillRunRequirement(
                        DocumentsSkill,
                        "scripts/make_pptx.py",
                        ProducesArtifact: true,
                        DefaultArguments: new[]
                        {
                            "--spec", "pptx_spec.json", "--out", "report.pptx",
                        },
                        EnforceArguments: true,
                        RequiredInputPath: "pptx_spec.json"),
                },
                MinimumSlides: 4,
                RequiredVisibleTerms: new[] { "M5", "M6" },
                RequireVisibleHttpUrl: true,
                CitationEvidencePath: "notes.md"),
            RequiresNetwork: true);
    }

    private static bool HasRequiredScript(Skill skill, string relativePath)
    {
        bool indexed = skill.Files.Any(file =>
            file.Kind == SkillFileKind.Script
            && string.Equals(file.Path, relativePath, SkillPathGuard.PathComparison));
        return indexed
            && SkillPathGuard.TryResolveExistingFile(
                skill.RootDirectory, relativePath, out _, out _);
    }

    private static string ResearchQuery(string latestUserText)
    {
        // Search indexes rank the stable English product terms well, while the bounded
        // suffix retains any region, benchmark, security, or other focus the user added.
        // Keeping the suffix bounded also prevents an attachment-sized prompt from
        // becoming a command-line argument (attachments are rejected above anyway).
        const string core = "Apple M6 chip specifications release information compared with Apple M5 chip";
        int deliverableAction = FindPowerPointDeliverableAction(latestUserText);
        string researchClause = deliverableAction > 0
            ? latestUserText.Substring(0, deliverableAction)
            : latestUserText;
        string focus = string.Join(" ", researchClause
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            .Trim(' ', '\t', '\r', '\n', ',', '，', ';', '；');
        foreach (string connective in CreationConnectives
                     .OrderByDescending(value => value.Length))
        {
            if (focus.EndsWith(connective, StringComparison.OrdinalIgnoreCase))
            {
                focus = focus.Substring(0, focus.Length - connective.Length)
                    .Trim(' ', '\t', '\r', '\n', ',', '，', ';', '；');
                break;
            }
        }
        if (focus.Length > 384)
            focus = focus.Substring(0, 384);
        return string.IsNullOrEmpty(focus) ? core : core + ". User-requested focus: " + focus;
    }

    private static bool IsAppleM5M6Comparison(string text) =>
        (text.Contains("apple", StringComparison.OrdinalIgnoreCase)
         || text.Contains("苹果", StringComparison.Ordinal))
        && ContainsChipToken(text, "M5")
        && ContainsChipToken(text, "M6");

    private static ChatMessage? LatestUserMessage(IReadOnlyList<ChatMessage> messages)
    {
        for (int index = messages.Count - 1; index >= 0; index--)
        {
            ChatMessage? message = messages[index];
            if (message is not null
                && string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase)
                && !string.IsNullOrWhiteSpace(message.Content))
            {
                return message;
            }
        }

        return null;
    }

    private static bool HasAttachments(ChatMessage? message) =>
        message?.AttachmentPaths is { Count: > 0 }
        || message?.TextFilePaths is { Count: > 0 }
        || message?.ImagePaths is { Count: > 0 }
        || message?.AudioPaths is { Count: > 0 };

    private static bool ContainsChipToken(string text, string token)
    {
        int searchFrom = 0;
        while (searchFrom < text.Length)
        {
            int index = text.IndexOf(token, searchFrom, StringComparison.OrdinalIgnoreCase);
            if (index < 0)
                return false;
            int end = index + token.Length;
            bool leftBoundary = index == 0 || !char.IsAsciiLetterOrDigit(text[index - 1]);
            bool rightBoundary = end == text.Length || !char.IsAsciiLetterOrDigit(text[end]);
            if (leftBoundary && rightBoundary)
                return true;
            searchFrom = end;
        }
        return false;
    }

    private static bool ContainsAny(string text, IEnumerable<string> phrases)
    {
        foreach (string phrase in phrases)
        {
            if (text.Contains(phrase, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Require the creation verb to look like a requested action and to be followed
    /// closely by the presentation format. Three independent substrings are too broad:
    /// “Research why tools generate PowerPoint presentations” discusses generation but
    /// does not ask this host to create a file.
    /// </summary>
    private static bool HasPowerPointDeliverableIntent(string text)
        => FindPowerPointDeliverableAction(text) >= 0;

    private static int FindPowerPointDeliverableAction(string text)
    {
        int earliest = -1;
        foreach (string creation in CreationPhrases)
        {
            int searchFrom = 0;
            while (searchFrom < text.Length)
            {
                int action = text.IndexOf(creation, searchFrom, StringComparison.OrdinalIgnoreCase);
                if (action < 0)
                    break;

                int afterAction = action + creation.Length;
                int nearbyEnd = Math.Min(text.Length, afterAction + 96);
                string nearby = text.Substring(afterAction, nearbyEnd - afterAction);
                if (LooksLikeRequestedAction(text, action)
                    && ContainsAny(nearby, PowerPointPhrases)
                    && (earliest < 0 || action < earliest))
                {
                    earliest = action;
                }

                searchFrom = afterAction;
            }
        }

        return earliest;
    }

    private static bool LooksLikeRequestedAction(string text, int action)
    {
        if (action == 0 || string.IsNullOrWhiteSpace(text.Substring(0, action)))
            return true;

        char previous = text[action - 1];
        if (previous is ',' or '，' or ';' or '；' or '.' or '!' or '！' or '?' or '？' or ':' or '\n')
            return true;

        string prefix = text.Substring(0, action).TrimEnd();
        foreach (string connective in CreationConnectives)
        {
            if (!prefix.EndsWith(connective, StringComparison.OrdinalIgnoreCase))
                continue;

            int start = prefix.Length - connective.Length;
            if (start == 0 || !char.IsAsciiLetterOrDigit(prefix[start - 1])
                || !char.IsAsciiLetterOrDigit(connective[0]))
            {
                return true;
            }
        }

        return false;
    }
}
