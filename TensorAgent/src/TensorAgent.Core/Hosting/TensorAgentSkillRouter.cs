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
/// research on the web followed by a real PowerPoint deliverable, on any subject.
///
/// <para>
/// This is deliberately not a general natural-language classifier. It needs one cue
/// from each side in the same latest user turn, leaves every explicit skill selection
/// alone, and only returns ids the installed registry actually contains. The normal
/// skill planner remains responsible for permissions, tool declarations and loading
/// the full instructions progressively.
/// </para>
/// <para>
/// Nothing here depends on what the deck is about. The research query is the user's
/// own words, and the evidence contract checks the deck's structure and that it cites
/// this turn's sources, never that it names particular terms.
/// </para>
/// <para>
/// Precision comes first, because a routed turn is committed to a network-only
/// workflow (refused outright while network access is off) and a missed one only falls
/// back to ordinary discovery. So a research word must be asked for where it stands
/// ("Search X", "can you research", "帮我搜索"), not merely appear ("our research
/// results", "搜索引擎"); a format name must be a word of its own ("slides", never the
/// end of "landslides"); and it must be what the verb makes ("create a presentation",
/// not "write a summary of the clinical presentation").
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

    /// <summary>
    /// Asking for research in so many words. A cue counts only where it is requested - at
    /// the start of a clause, after a connective or a lead-in such as "you" or "帮我" - so
    /// "a presentation about our research" or "关于搜索引擎优化的演示文稿" is a subject, not
    /// a request to search.
    /// </summary>
    private static readonly string[] ResearchPhrases =
    {
        "search", "web search", "research", "look up", "find information",
        "搜索", "检索", "查找", "查一下", "调研", "研究一下",
    };

    /// <summary>
    /// Asking for fresh information by what is wanted rather than by a verb. Specific enough
    /// to count wherever it appears.
    /// </summary>
    private static readonly string[] FreshInformationPhrases =
    {
        "latest information", "current information",
    };

    /// <summary>
    /// Words after which a research verb is being asked for rather than named: "can you
    /// search", "I want you to research", "help me look up", "帮我搜索", "先调研".
    /// </summary>
    private static readonly string[] ResearchLeadIns =
    {
        "you", "to", "me", "us", "first", "pls", "let's", "lets", "kindly",
        "帮我", "帮忙", "给我", "先", "你", "想", "要", "去",
    };

    /// <summary>
    /// Words that, between a creation verb and the format, make the format the subject of
    /// something else: "write a summary of the clinical presentation" does not ask for a
    /// presentation.
    /// </summary>
    private static readonly string[] DeliverableGapBreakers =
    {
        "of", "about", "on", "for", "from", "with", "in", "into", "by", "regarding", "through", "that", "which",
    };

    /// <summary>
    /// The most words an English request may put between its verb and the format ("make me
    /// a Rust vs Go presentation" has five).
    /// </summary>
    private const int MaxDeliverableGapWords = 6;

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
    /// The connectives that can stand before another one ("and then create", "然后再生成").
    /// Only these are stripped as a second word before the verb, so "release next and
    /// create" keeps its "next" and "企业合并然后生成" keeps its 合并. Single-character CJK
    /// connectives are left out on purpose: they end ordinary words (合并, 申请).
    /// </summary>
    private static readonly string[] Coordinators =
    {
        "and", "then", "并且", "然后", "同时",
    };

    /// <summary>At most this many URLs the user wrote are handed to research.py to read.</summary>
    private const int MaxResearchUrls = 3;

    /// <summary>A URL longer than this is not passed on (the artifact contract caps each argument).</summary>
    private const int MaxResearchUrlLength = 2048;

    /// <summary>
    /// Words that only introduce the deck ("make me a presentation", "生成一份演示文稿")
    /// and would be left dangling in the research query where the deck request is cut.
    /// </summary>
    private static readonly string[] DeliverableDeterminers =
    {
        "a", "an", "the", "me", "us", "一份", "一个",
    };

    /// <summary>
    /// Longest query the route hands research.py. It is a search query, not a document:
    /// the bound keeps a pasted essay from becoming a command-line argument (attachments
    /// are rejected anyway) and stays far inside the 4,096 characters the artifact
    /// contract allows each routed argument.
    /// </summary>
    internal const int MaxResearchQueryLength = 384;

    private static readonly char[] ClauseTrim = { ' ', '\t', '\r', '\n', ',', '，', ';', '；' };

    /// <summary>
    /// Also trims the sentence punctuation left where the deck request was cut, and a
    /// leading '-': research.py reads its question with argparse, which takes a one-word
    /// argument that starts with '-' for an unknown option and fails the run.
    /// </summary>
    private static readonly char[] QueryTrim =
    {
        ' ', '\t', '\r', '\n', ',', '，', ';', '；', '.', '。', '!', '！', '?', '？', ':', '：', '-',
    };

    /// <summary>Where the requested deck is named: its creation verb and its format.</summary>
    private readonly record struct PowerPointDeliverable(
        int ActionStart, int ActionLength, int FormatStart, int FormatLength);

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
            || !HasResearchRequest(latest)
            || !ContainsCue(latest, PowerPointPhrases, allowPlural: true)
            || !TryFindPowerPointDeliverable(latest, out PowerPointDeliverable deliverable))
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
                        DefaultArguments: ResearchArguments(latest, deliverable)
                            .Concat(new[] { "--pages", "3", "--out", "notes.md" })
                            .ToArray(),
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

    /// <summary>
    /// research.py's positional arguments: the user's own words with only the request for
    /// the deck cut out (its creation verb and its format name), then each URL the user
    /// wrote as an argument of its own. Nothing is added, so the route never steers the
    /// search toward a subject of its own.
    /// </summary>
    /// <remarks>
    /// The words before that request are the research clause in the usual phrasing
    /// ("Search X and compare it with Y, then create a PowerPoint report"), but the
    /// subject can equally sit inside or after it ("Search the web and make a
    /// presentation about X", "搜索最新资料，然后生成一份关于X的演示文稿"). This route
    /// enforces its arguments, so a model cannot repair a query that lost the subject;
    /// keeping the deliverable's other words costs a little noise at most. Cutting the
    /// format name keeps "PowerPoint" from pulling in pages about making decks. A URL is
    /// separated out because research.py reads every argument that starts with http(s)://
    /// as a page to read: a question that began with one used to be taken whole, spaces
    /// and all, for an address.
    /// </remarks>
    private static IEnumerable<string> ResearchArguments(string latestUserText, PowerPointDeliverable deliverable)
    {
        int actionEnd = deliverable.ActionStart + deliverable.ActionLength;
        int formatEnd = deliverable.FormatStart + deliverable.FormatLength;
        string researchClause = ResearchClause(latestUserText.Substring(0, deliverable.ActionStart));
        string deliverableWords = WithoutLeadingDeterminers(Words(
            latestUserText.Substring(actionEnd, deliverable.FormatStart - actionEnd)
            + " " + latestUserText.Substring(formatEnd)));
        string words = researchClause + " " + deliverableWords;
        if (researchClause.Length == 0 && deliverableWords.Length == 0)
            words = latestUserText;

        var urls = new List<string>();
        var question = new List<string>();
        foreach (string word in words.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            string url = word.TrimEnd('.', ',', ';', ':', '!', '?', ')', ']', '>', '"', '\'', '。', '，', '；', '！', '？');
            if (url.StartsWith("http://", StringComparison.OrdinalIgnoreCase)
                || url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            {
                if (url.Length <= MaxResearchUrlLength && urls.Count < MaxResearchUrls && !urls.Contains(url))
                    urls.Add(url);
                continue;
            }
            question.Add(word);
        }

        string query = string.Join(" ", question).Trim(QueryTrim);
        if (query.Length > MaxResearchQueryLength)
        {
            // Never split a surrogate pair: half of one cannot reach argv intact.
            int end = char.IsHighSurrogate(query[MaxResearchQueryLength - 1])
                ? MaxResearchQueryLength - 1
                : MaxResearchQueryLength;
            query = query.Substring(0, end).TrimEnd(QueryTrim);
        }

        if (query.Length > 0)
            yield return query;
        foreach (string url in urls)
            yield return url;
    }

    /// <summary>
    /// The words before the deck request, without the connective that leads into it
    /// ("Search X and then", "搜索X，然后"). Only the run directly before the verb goes: one
    /// connective, and one coordinator before that, never across clause punctuation - so
    /// "搜索美国签证申请，生成" keeps 申请 and "release next, and create" keeps "next".
    /// </summary>
    private static string ResearchClause(string prefix)
    {
        string clause = prefix.TrimEnd();
        if (TryFindTrailingConnective(clause, CreationConnectives, out int start))
        {
            clause = clause.Substring(0, start).TrimEnd();
            if (TryFindTrailingConnective(clause, Coordinators, out start))
                clause = clause.Substring(0, start).TrimEnd();
        }
        return Words(clause);
    }

    private static string Words(string text) =>
        string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            .Trim(ClauseTrim);

    /// <summary>"me a Rust vs Go" describes the deck as "Rust vs Go".</summary>
    private static string WithoutLeadingDeterminers(string words)
    {
        bool stripped;
        do
        {
            stripped = false;
            foreach (string determiner in DeliverableDeterminers)
            {
                // An English determiner must be a lower-case word of its own: "an" does
                // not start "analysis", "a" does not start "A/B", and "US" is a subject.
                // A CJK one needs no boundary.
                if (!words.StartsWith(determiner, StringComparison.Ordinal)
                    || (char.IsAsciiLetter(determiner[0])
                        && words.Length > determiner.Length
                        && !char.IsWhiteSpace(words[determiner.Length])))
                {
                    continue;
                }

                words = words.Substring(determiner.Length).Trim(ClauseTrim);
                stripped = true;
                break;
            }
        }
        while (stripped);

        return words;
    }

    /// <summary>
    /// One of <paramref name="connectives"/> that ends <paramref name="text"/> as a word of
    /// its own: "and" ends "Search X and", but not "Search Poland". CJK connectives need no
    /// word boundary.
    /// </summary>
    private static bool TryFindTrailingConnective(string text, string[] connectives, out int start)
    {
        foreach (string connective in connectives)
        {
            if (!text.EndsWith(connective, StringComparison.OrdinalIgnoreCase))
                continue;

            start = text.Length - connective.Length;
            if (start == 0 || !char.IsAsciiLetterOrDigit(text[start - 1])
                || !char.IsAsciiLetterOrDigit(connective[0]))
            {
                return true;
            }
        }

        start = -1;
        return false;
    }

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

    /// <summary>
    /// Where <paramref name="phrase"/> occurs in [<paramref name="start"/>, <paramref name="end"/>)
    /// as a word of its own, and how long the match is. An ASCII end of a phrase needs a
    /// word boundary there ("slides" is not in "landslides", "presentation" not in
    /// "representation", "search" not in "research"); a CJK end needs none. With
    /// <paramref name="allowPlural"/> a trailing "s" belongs to the match ("presentations").
    /// </summary>
    private static bool TryFindCue(
        string text, string phrase, int start, int end, bool allowPlural, out int index, out int length)
    {
        int from = start;
        while (from < end)
        {
            int found = text.IndexOf(phrase, from, end - from, StringComparison.OrdinalIgnoreCase);
            if (found < 0)
                break;

            int after = found + phrase.Length;
            bool leftBoundary = !char.IsAsciiLetterOrDigit(phrase[0])
                || found == 0
                || !char.IsAsciiLetterOrDigit(text[found - 1]);
            int matched = phrase.Length;
            bool rightBoundary = !char.IsAsciiLetterOrDigit(phrase[^1])
                || after >= text.Length
                || !char.IsAsciiLetterOrDigit(text[after]);
            if (!rightBoundary && allowPlural
                && (text[after] is 's' or 'S')
                && (after + 1 >= text.Length || !char.IsAsciiLetterOrDigit(text[after + 1])))
            {
                rightBoundary = true;
                matched++;
            }

            if (leftBoundary && rightBoundary)
            {
                index = found;
                length = matched;
                return true;
            }
            from = found + 1;
        }

        index = -1;
        length = 0;
        return false;
    }

    private static bool ContainsCue(string text, IEnumerable<string> phrases, bool allowPlural)
    {
        foreach (string phrase in phrases)
        {
            if (TryFindCue(text, phrase, 0, text.Length, allowPlural, out _, out _))
                return true;
        }

        return false;
    }

    /// <summary>
    /// The turn asks for research: a research verb where it is requested, or a phrase that
    /// asks for fresh information.
    /// </summary>
    private static bool HasResearchRequest(string text)
    {
        if (ContainsCue(text, FreshInformationPhrases, allowPlural: false))
            return true;

        foreach (string phrase in ResearchPhrases)
        {
            int from = 0;
            while (from < text.Length
                   && TryFindCue(text, phrase, from, text.Length, allowPlural: false, out int index, out int length))
            {
                if (LooksLikeRequestedResearch(text, index))
                    return true;
                from = index + length;
            }
        }

        return false;
    }

    /// <summary>
    /// A research verb opens a clause (the start of the turn, a bullet, punctuation, a line
    /// break, a connective) or follows a lead-in that asks for it ("can you search", "帮我
    /// 搜索"). Anywhere else it names a subject.
    /// </summary>
    private static bool LooksLikeRequestedResearch(string text, int cue)
    {
        string prefix = text.Substring(0, cue);
        string written = prefix.TrimEnd();
        // Nothing but bullets and punctuation before it: "- Search", "1) Research".
        if (!written.Any(char.IsLetter))
            return true;
        if (prefix.IndexOf('\n', written.Length) >= 0)
            return true;

        char previous = written[^1];
        if (previous is ',' or '，' or ';' or '；' or '.' or '。' or '!' or '！' or '?' or '？' or ':' or '：' or '、'
            or '-' or '–' or '—')
            return true;

        return TryFindTrailingConnective(written, CreationConnectives, out _)
            || TryFindTrailingConnective(written, ResearchLeadIns, out _);
    }

    /// <summary>
    /// Require the creation verb to look like a requested action and to be followed
    /// closely by the presentation format. Three independent substrings are too broad:
    /// “Research why tools generate PowerPoint presentations” discusses generation but
    /// does not ask this host to create a file. The earliest such request wins.
    /// </summary>
    private static bool TryFindPowerPointDeliverable(string text, out PowerPointDeliverable deliverable)
    {
        deliverable = default;
        bool found = false;
        foreach (string creation in CreationPhrases)
        {
            int searchFrom = 0;
            while (searchFrom < text.Length
                   && TryFindCue(text, creation, searchFrom, text.Length, allowPlural: false, out int action, out _))
            {
                int afterAction = action + creation.Length;
                int nearbyEnd = Math.Min(text.Length, afterAction + 96);
                if ((!found || action < deliverable.ActionStart)
                    && LooksLikeRequestedAction(text, action)
                    && TryFindEarliestPhrase(
                        text, afterAction, nearbyEnd, PowerPointPhrases,
                        out int format, out int formatLength)
                    && FormatIsTheObject(text, creation, afterAction, format))
                {
                    deliverable = new PowerPointDeliverable(action, creation.Length, format, formatLength);
                    found = true;
                }

                searchFrom = afterAction;
            }
        }

        return found;
    }

    /// <summary>
    /// The format is what the verb makes, not the subject of something else it makes. In
    /// English the words between them are few and none of them is a preposition ("create a
    /// short PowerPoint", but not "write a summary of the clinical presentation"). Chinese
    /// puts the whole subject there ("生成一份关于X的演示文稿"), so a CJK verb's gap is free.
    /// </summary>
    private static bool FormatIsTheObject(string text, string creation, int afterAction, int format)
    {
        if (!char.IsAsciiLetter(creation[0]))
            return true;

        string[] gap = text.Substring(afterAction, format - afterAction)
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (gap.Length > MaxDeliverableGapWords)
            return false;
        foreach (string word in gap)
        {
            string bare = word.Trim(',', '.', ';', ':', '"', '\'', '(', ')');
            foreach (string breaker in DeliverableGapBreakers)
            {
                if (string.Equals(bare, breaker, StringComparison.OrdinalIgnoreCase))
                    return false;
            }
        }
        return true;
    }

    /// <summary>The first phrase wholly inside [start, end) as a word of its own (a plural
    /// "s" included); the longest one on a tie.</summary>
    private static bool TryFindEarliestPhrase(
        string text, int start, int end, IEnumerable<string> phrases, out int index, out int length)
    {
        index = -1;
        length = 0;
        foreach (string phrase in phrases)
        {
            if (TryFindCue(text, phrase, start, end, allowPlural: true, out int found, out int matched)
                && (index < 0 || found < index || (found == index && matched > length)))
            {
                index = found;
                length = matched;
            }
        }

        return index >= 0;
    }

    private static bool LooksLikeRequestedAction(string text, int action)
    {
        string prefix = text.Substring(0, action);
        string written = prefix.TrimEnd();
        if (written.Length == 0)
            return true;

        // Judge the last character the user wrote, not the space typed after it:
        // "Search X, create a deck" asks as plainly as "Search X,create a deck". A line
        // break before the verb starts a new request as well.
        if (prefix.IndexOf('\n', written.Length) >= 0)
            return true;

        char previous = written[^1];
        if (previous is ',' or '，' or ';' or '；' or '.' or '。' or '!' or '！' or '?' or '？' or ':' or '：')
            return true;

        return TryFindTrailingConnective(written, CreationConnectives, out _);
    }
}
