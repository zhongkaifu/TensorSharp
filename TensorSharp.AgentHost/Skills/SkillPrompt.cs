// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

using TensorSharp.Runtime;
namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// How much of the context the skills block may occupy, and whether the model can
    /// fetch the rest for itself.
    /// </summary>
    public sealed class SkillPromptOptions
    {
        /// <summary>
        /// The model's context length, when the caller knows it. Used only to derive
        /// the budgets below; zero means "unknown", and the fixed defaults apply.
        /// </summary>
        public int ContextTokens { get; init; }

        /// <summary>
        /// Approximate token ceiling for the whole injected block. Zero derives it from
        /// <see cref="ContextTokens"/>.
        /// </summary>
        public int MaxBlockTokens { get; init; }

        /// <summary>
        /// Approximate token ceiling for ONE inlined <c>SKILL.md</c> body. Zero derives
        /// it. A skill whose body is larger is announced but not inlined; the model is
        /// told to read it, which is what the tools are for.
        /// </summary>
        public int MaxInlineBodyTokens { get; init; }

        /// <summary>
        /// Longest description rendered for a skill.
        ///
        /// <para>
        /// 1024 is the specification's own ceiling on the <c>description</c> field, and
        /// the same constant Codex uses (<c>MAX_CATALOG_SKILL_DESCRIPTION_CHARS</c>).
        /// It used to be 320, which quietly discarded more than half the routing signal
        /// of a well-written skill — the <c>pptx</c> description is 738 characters and
        /// the part that says "trigger whenever the user mentions deck, slides,
        /// presentation" fell off the end. The description IS the routing decision once
        /// bodies are no longer inlined, so cutting it is cutting the thing that decides.
        /// </para>
        /// </summary>
        public int MaxCatalogDescriptionChars { get; init; } = 1024;

        /// <summary>Most skills listed in the discovery catalog.</summary>
        public int MaxCatalogEntries { get; init; } = 96;

        /// <summary>
        /// False when the model has no way to fetch a skill for itself — the family's
        /// renderer discards tool declarations (see
        /// <see cref="ChatProtocol.RendersToolDeclarations"/>), or the host disabled the
        /// built-in tools. The instructions change accordingly: with no tools, telling
        /// the model to "call skills_read" is an instruction it cannot follow, and the
        /// discovery catalog is dead weight, so both are dropped and the whole budget
        /// goes to inlining the selected bodies.
        /// </summary>
        public bool ToolsAvailable { get; init; } = true;

        /// <summary>Include the "how to use a skill" guidance. Off saves ~250 tokens.</summary>
        public bool IncludeUsageInstructions { get; init; } = true;

        /// <summary>
        /// Write a SELECTED skill's whole <c>SKILL.md</c> body into the prompt up front.
        ///
        /// <para>
        /// <b>Off, and off is the specification.</b> Progressive disclosure has three
        /// tiers — metadata (~100 tokens per skill) at startup, the body "when the skill
        /// is activated", resources on demand — and activation is the MODEL's decision,
        /// not the user's. Inlining on selection collapses tiers one and two and hands
        /// the model every body whether the task needs it or not.
        /// </para>
        /// <para>
        /// Measured on a real four-skill request: the bodies came to 43,777 bytes
        /// (~10,944 tokens), of which 52% belonged to skills the model never referenced,
        /// against 1,932 bytes (~483 tokens) for the four descriptions — a 96% saving.
        /// The model then called <c>skills_read</c> for the two it did use ANYWAY, so
        /// deferring cost it nothing it was not already paying.
        /// </para>
        /// <para>
        /// Ignored when <see cref="ToolsAvailable"/> is false: a model that cannot call
        /// <c>skills_read</c> has no second chance, so its bodies are inlined regardless.
        /// </para>
        /// </summary>
        public bool InlineSelectedBodies { get; init; }

        /// <summary>Defaults for a caller that knows nothing about the model.</summary>
        public static SkillPromptOptions Default { get; } = new();

        internal int ResolvedBlockTokens
        {
            get
            {
                if (MaxBlockTokens > 0)
                    return MaxBlockTokens;
                if (ContextTokens > 0)
                    return Math.Clamp(ContextTokens / 4, 1024, 48_000);
                return 16_000;
            }
        }

        /// <summary>
        /// Ceiling on the metadata tier — the block when bodies are NOT inlined.
        ///
        /// <para>
        /// Two percent of the context window, floored at 1024 and capped at 10,000
        /// tokens: Codex's shape and very nearly its constants
        /// (<c>SKILL_METADATA_CONTEXT_WINDOW_PERCENT = 2</c>,
        /// <c>MAX_CONFIGURED_SKILL_METADATA_TOKEN_BUDGET = 10_000</c>). A quarter of the
        /// context — what <see cref="ResolvedBlockTokens"/> allows — is the right order
        /// for bodies and absurd for a list of names and descriptions: on a 262k model
        /// it reserved 48,000 tokens to say what four skills are called.
        /// </para>
        /// </summary>
        internal int ResolvedMetadataTokens
        {
            get
            {
                if (MaxBlockTokens > 0)
                    return MaxBlockTokens;
                if (ContextTokens > 0)
                    return Math.Clamp(ContextTokens * 2 / 100, 1024, 10_000);
                return 2_000;
            }
        }

        internal int ResolvedInlineBodyTokens
        {
            get
            {
                if (MaxInlineBodyTokens > 0)
                    return MaxInlineBodyTokens;
                // Three quarters of the block, so one large skill can take most of it
                // while still leaving room for the catalog and the instructions.
                return Math.Max(512, ResolvedBlockTokens * 3 / 4);
            }
        }
    }

    /// <summary>What the planner decided to put in front of the model, and why.</summary>
    /// <param name="Instructions">The rendered block, or the empty string when there is nothing to say.</param>
    /// <param name="Selected">Skills the caller explicitly asked for, sorted by id.</param>
    /// <param name="Inlined">The subset of <paramref name="Selected"/> whose body was written into the prompt.</param>
    /// <param name="Deferred">
    /// Selected skills whose body did not fit. Announced by name and description with
    /// an instruction to read them, so the model still knows they apply.
    /// </param>
    /// <param name="Catalog">Skills advertised for discovery, metadata only.</param>
    /// <param name="OmittedFromCatalog">How many registered skills did not fit the catalog.</param>
    /// <param name="ToolsAvailable">Whether the built-in skill tools were offered.</param>
    /// <param name="ApproximateTokens">Rough size of <paramref name="Instructions"/>.</param>
    public sealed record SkillPlan(
        string Instructions,
        IReadOnlyList<Skill> Selected,
        IReadOnlyList<Skill> Inlined,
        IReadOnlyList<Skill> Deferred,
        IReadOnlyList<Skill> Catalog,
        int OmittedFromCatalog,
        bool ToolsAvailable,
        int ApproximateTokens)
    {
        /// <summary>Nothing to inject.</summary>
        public static SkillPlan Empty { get; } = new(
            string.Empty,
            Array.Empty<Skill>(), Array.Empty<Skill>(), Array.Empty<Skill>(), Array.Empty<Skill>(),
            0, false, 0);

        /// <summary>True when this plan puts nothing in front of the model.</summary>
        public bool IsEmpty => Instructions.Length == 0;

        /// <summary>Every skill the model may read from — the selection plus the catalog.</summary>
        public IEnumerable<Skill> Reachable => Selected.Concat(Catalog);
    }

    /// <summary>
    /// Turns a skill selection into the block of text that goes in front of the model.
    ///
    /// <para>
    /// <b>Every byte of the output is a pure function of the selected set and the
    /// options.</b> This is not a style preference. The KV prefix cache chains a
    /// SHA-256 over 256-token blocks starting at block 0 and stops adopting at the
    /// first mismatch, and this block sits at the very front of the prompt — so a
    /// timestamp, an absolute path, a "3 skills registered" counter, or a selection
    /// rendered in whatever order the caller's JSON happened to list it would change
    /// block 0 on every turn and drop prefix reuse to zero for the entire
    /// conversation, not merely for the part that changed. Skills are sorted by id
    /// with an ordinal comparison, separators are fixed, and nothing
    /// environment-derived is rendered.
    /// </para>
    /// </summary>
    public static class SkillPrompt
    {
        /// <summary>The heading that identifies the skills block. Also how tests find it.</summary>
        public const string BlockHeading = "## Agent skills";

        /// <summary>
        /// Decide what to inject.
        /// </summary>
        /// <param name="selected">
        /// Skills the caller explicitly chose. They are ANNOUNCED — name and description
        /// — not loaded: selection scopes which skills the conversation can reach, and
        /// the model decides which of them the task actually needs and reads that one.
        ///
        /// <para>
        /// This used to inline every selected body in full, on the reasoning that "an
        /// explicit selection is the user saying use this, not consider this". The
        /// reasoning does not survive contact with a skill picker: four selected skills
        /// cost 12,848 tokens of prompt on every request, 52% of it for skills the model
        /// never referenced, and it re-read the largest one through <c>skills_read</c>
        /// anyway. Announcing them instead costs 1,050.
        /// </para>
        /// </param>
        /// <param name="catalog">
        /// Skills to advertise for discovery, metadata only. Pass the registry's other
        /// skills to let the model notice one the user did not think to name; pass an
        /// empty list to restrict the conversation to the selection.
        /// </param>
        /// <param name="options">Budgets and capabilities.</param>
        public static SkillPlan Plan(
            IReadOnlyList<Skill>? selected,
            IReadOnlyList<Skill>? catalog,
            SkillPromptOptions? options = null)
        {
            options ??= SkillPromptOptions.Default;

            List<Skill> chosen = Order(selected);
            List<Skill> discoverable = Order(catalog);

            // A skill can legitimately appear in both lists (the caller passed the whole
            // registry as the catalog). It is already fully present in the selection, so
            // listing it again would spend tokens telling the model something it has.
            if (chosen.Count > 0 && discoverable.Count > 0)
            {
                var chosenIds = new HashSet<string>(chosen.Select(s => s.Id), StringComparer.OrdinalIgnoreCase);
                discoverable.RemoveAll(s => chosenIds.Contains(s.Id));
            }

            if (chosen.Count == 0 && discoverable.Count == 0)
                return SkillPlan.Empty;

            // Without tools the model cannot fetch anything, so a discovery catalog only
            // teases skills it can never load. Drop it and give the budget to the bodies.
            if (!options.ToolsAvailable)
                discoverable.Clear();

            // Tier one or tier two. The specification puts NAME AND DESCRIPTION in the
            // prompt at startup and the body "when the skill is activated" — and
            // activation is the model's decision. A user selecting a skill is a strong
            // hint about which one to activate, not the activation itself, so selection
            // no longer buys a body.
            //
            // The exception is a model that cannot call skills_read at all. There,
            // deferring would not disclose
            // progressively, it would simply withhold: nothing could ever fetch the
            // body, so it goes in the prompt or nowhere.
            bool inlineBodies = !options.ToolsAvailable || options.InlineSelectedBodies;
            int budget = inlineBodies ? options.ResolvedBlockTokens : options.ResolvedMetadataTokens;
            int inlineCap = options.ResolvedInlineBodyTokens;

            var inlined = new List<Skill>();
            var deferred = new List<Skill>();
            int spent = 0;

            foreach (Skill skill in chosen)
            {
                if (!inlineBodies)
                {
                    // Metadata tier: the description is the whole cost, and it is never
                    // dropped for want of budget. A selected skill the model cannot see
                    // is worse than a long block — it is a silent no-op on something the
                    // user asked for by name.
                    deferred.Add(skill);
                    spent += SkillTextBudget.ApproximateTokens(DescriptionFor(skill, options.MaxCatalogDescriptionChars)) + 32;
                    continue;
                }

                // A skill installed from the hub is third-party content, and its body
                // reaches the model only as a tool result (skills_read), never as part of
                // the system prompt.
                if (skill.Source != null)
                {
                    deferred.Add(skill);
                    spent += 48;
                    continue;
                }

                int cost = SkillTextBudget.ApproximateTokens(skill.Manifest.Body) + 64;
                bool fits = cost <= inlineCap && spent + cost <= budget;

                // With no tools there is no second chance: a body that is not inlined is
                // simply unavailable, so a single oversized skill is inlined anyway
                // rather than silently doing nothing. The truncation is announced.
                if (!fits && !options.ToolsAvailable && inlined.Count == 0)
                    fits = true;

                if (fits)
                {
                    inlined.Add(skill);
                    spent += cost;
                }
                else
                {
                    deferred.Add(skill);
                    spent += 48;
                }
            }

            // How long each catalog line may be. Normally the caller's cap, but when the
            // catalog does not fit at that length the entries are SHORTENED rather than
            // dropped, because a skill the model never hears of cannot be asked for.
            //
            // This is not hypothetical. TensorAgent's bundled descriptions came to about
            // 1,450 tokens against a 1,024-token budget, and the fill below is ordinal by
            // id — so `documents` and `research`, the two the app's own router depends on,
            // were evicted by alphabetical luck while two 990-character entries ahead of
            // them took half the budget. Asked to look something up, the model listed the
            // skills it could see, found nothing that fetches a page, and refused. One
            // sentence each about all of them beats three sentences each about half; the
            // full text is one skills_read away either way. (Those two oversized entries
            // have since been unbundled for a different reason, and the catalog still
            // does not fit at full length, so this is still what makes it fit.)
            int describeChars = FitCatalogDescriptions(discoverable, options, budget - spent);

            var listed = new List<Skill>();
            int omitted = 0;
            foreach (Skill skill in discoverable)
            {
                int cost = CatalogEntryTokens(skill, describeChars);
                if (listed.Count >= options.MaxCatalogEntries || spent + cost > budget)
                {
                    omitted++;
                    continue;
                }
                listed.Add(skill);
                spent += cost;
            }

            string instructions = Render(inlined, deferred, listed, omitted, options, describeChars);
            return new SkillPlan(
                instructions,
                chosen,
                inlined,
                deferred,
                listed,
                omitted,
                options.ToolsAvailable,
                SkillTextBudget.ApproximateTokens(instructions));
        }

        /// <summary>
        /// Return a message list with <paramref name="plan"/>'s instructions in front of
        /// it, leaving the caller's list untouched.
        ///
        /// <para>
        /// The shape mirrors <see cref="StructuredOutputPrompt.Apply"/> deliberately,
        /// and for a reason that is not cosmetic: merging into an existing leading
        /// <c>system</c>/<c>developer</c> message is the only injection point every chat
        /// template in the repository handles. Appending a second system message is
        /// silently dropped by the Mistral 3 renderer, emits a duplicate system turn on
        /// GPT-OSS's Harmony format (which lifts <c>messages[0]</c> into its developer
        /// block and synthesizes its own system block), and is the shape a GGUF-embedded
        /// Jinja template is most likely to reject — which falls back to the hardcoded
        /// renderer and silently changes the whole prompt format.
        /// </para>
        /// </summary>
        public static List<ChatMessage> Apply(List<ChatMessage>? messages, SkillPlan? plan)
        {
            if (plan == null || plan.IsEmpty)
                return messages ?? new List<ChatMessage>();

            return Apply(messages, plan.Instructions);
        }

        /// <summary>
        /// Inject an arbitrary instruction block. Exposed for hosts that render their own
        /// preamble, and for tests.
        /// </summary>
        public static List<ChatMessage> Apply(List<ChatMessage>? messages, string? instructions)
        {
            if (string.IsNullOrWhiteSpace(instructions))
                return messages ?? new List<ChatMessage>();

            var result = new List<ChatMessage>((messages?.Count ?? 0) + 1);

            if (messages != null && messages.Count > 0
                && (messages[0].Role == "system" || messages[0].Role == "developer"))
            {
                ChatMessage first = Clone(messages[0]);
                string originalContent = string.IsNullOrWhiteSpace(first.Content)
                    ? string.Empty
                    : first.Content.TrimEnd();
                // A marker in removed trailing whitespace must not migrate into
                // the separator or the request-specific block appended below.
                ClampContentCacheBreakpoints(first, originalContent.Length);
                if (first.CacheControl != null)
                {
                    // The marker belongs after the caller's stable preamble,
                    // not after the request-specific skills block we append.
                    first.AddContentCacheBreakpoint(originalContent.Length);
                    first.CacheControl = null;
                }
                first.Content = originalContent.Length == 0
                    ? instructions
                    : originalContent + "\n\n" + instructions;
                result.Add(first);

                for (int i = 1; i < messages.Count; i++)
                    result.Add(Clone(messages[i]));
                return result;
            }

            result.Add(new ChatMessage { Role = "system", Content = instructions });
            if (messages != null)
            {
                foreach (ChatMessage message in messages)
                    result.Add(Clone(message));
            }
            return result;
        }

        private static void ClampContentCacheBreakpoints(ChatMessage message, int contentLength)
        {
            List<int>? breakpoints = message.ContentCacheBreakpoints;
            if (breakpoints == null)
                return;

            for (int i = 0; i < breakpoints.Count; i++)
                breakpoints[i] = System.Math.Clamp(breakpoints[i], 0, contentLength);

            breakpoints.Sort();
            int writeIndex = 0;
            for (int readIndex = 0; readIndex < breakpoints.Count; readIndex++)
            {
                if (writeIndex == 0 || breakpoints[readIndex] != breakpoints[writeIndex - 1])
                    breakpoints[writeIndex++] = breakpoints[readIndex];
            }

            if (writeIndex < breakpoints.Count)
                breakpoints.RemoveRange(writeIndex, breakpoints.Count - writeIndex);
        }

        /// <summary>
        /// A COMPLETE copy of a chat message.
        ///
        /// <para>
        /// <see cref="StructuredOutputPrompt"/>'s own cloner drops
        /// <see cref="ChatMessage.RawOutputTokens"/> and
        /// <see cref="ChatMessage.TextFilePaths"/>. On the server that loss is invisible
        /// because <c>ChatHistoryPreparer</c> re-attaches the raw tokens from the tracked
        /// session afterwards — but the CLI has no such repair, and neither does the
        /// agentic loop, which appends assistant turns carrying exactly those tokens.
        /// Dropping them makes <see cref="KVCachePromptRenderer"/> re-tokenize each
        /// assistant turn instead of splicing it, the re-rendered prefix stops matching
        /// the cache at the first assistant boundary, and every skill round-trip pays a
        /// full re-prefill while still producing a correct answer — a pure, silent
        /// slowdown.
        /// </para>
        /// </summary>
        internal static ChatMessage Clone(ChatMessage message) => new()
        {
            Role = message.Role,
            Content = message.Content,
            ImagePaths = message.ImagePaths != null ? new List<string>(message.ImagePaths) : null,
            ImageTimestamps = message.ImageTimestamps != null ? new List<double?>(message.ImageTimestamps) : null,
            AudioPaths = message.AudioPaths != null ? new List<string>(message.AudioPaths) : null,
            TextFilePaths = message.TextFilePaths != null ? new List<string>(message.TextFilePaths) : null,
            TextFileNames = message.TextFileNames != null ? new List<string>(message.TextFileNames) : null,
            AttachmentPaths = message.AttachmentPaths != null ? new List<string>(message.AttachmentPaths) : null,
            AttachmentNames = message.AttachmentNames != null ? new List<string>(message.AttachmentNames) : null,
            HasFileBackedTextAttachments = message.HasFileBackedTextAttachments,
            IsVideo = message.IsVideo,
            ToolCalls = message.ToolCalls != null ? new List<ToolCall>(message.ToolCalls) : null,
            ToolCallId = message.ToolCallId,
            Thinking = message.Thinking,
            RawOutputTokens = message.RawOutputTokens != null ? new List<int>(message.RawOutputTokens) : null,
            RawPromptTrailingWhitespace = message.RawPromptTrailingWhitespace,
            RawGenerationSuffix = message.RawGenerationSuffix,
            CacheControl = message.CacheControl != null
                ? new CacheControlMarker { Type = message.CacheControl.Type }
                : null,
            ContentCacheBreakpoints = message.ContentCacheBreakpoints != null
                ? new List<int>(message.ContentCacheBreakpoints)
                : null,
        };

        // ---- rendering ---------------------------------------------------------

        private static List<Skill> Order(IReadOnlyList<Skill>? skills)
        {
            if (skills == null || skills.Count == 0)
                return new List<Skill>();

            var ordered = new List<Skill>(skills.Count);
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (Skill skill in skills)
            {
                if (skill != null && seen.Add(skill.Id))
                    ordered.Add(skill);
            }
            ordered.Sort(static (a, b) => string.CompareOrdinal(a.Id, b.Id));
            return ordered;
        }

        private static string Render(
            IReadOnlyList<Skill> inlined,
            IReadOnlyList<Skill> deferred,
            IReadOnlyList<Skill> catalog,
            int omitted,
            SkillPromptOptions options,
            int describeChars)
        {
            var sb = new StringBuilder();
            // Establish the user's task before presenting descriptions of other work
            // the model could do. In particular, missing details call for a question,
            // not a speculative tool call against the nearest-sounding skill.
            if (options.ToolsAvailable && options.IncludeUsageInstructions)
                sb.Append(SelectionWithTools).Append("\n\n");
            sb.Append(BlockHeading).Append('\n');
            sb.Append(
                "A skill is a set of instructions stored in a SKILL.md file, together with any scripts, "
                + "references and assets it ships. Treat a skill's instructions as authoritative for the "
                + "task it covers, above your default approach. Skills are optional: listing or selecting "
                + "one does not make it relevant to every request.\n");

            if (inlined.Count > 0 || deferred.Count > 0)
            {
                sb.Append('\n').Append("### Skills selected for this conversation\n");
                sb.Append(
                    "These were chosen deliberately and are the ones to prefer when they match the task. "
                    + "Say in one short line which you are using.\n");
                if (inlined.Count == 0)
                    sb.Append("Only names and descriptions follow; the instructions are not loaded.\n");
                else
                    sb.Append("Instructions inside <skill> blocks are already loaded; read any skill marked NOT loaded before using it.\n");

                foreach (Skill skill in inlined)
                    AppendInlinedSkill(sb, skill, options);

                foreach (Skill skill in deferred)
                {
                    // One wording for both reasons a body is absent — policy or size.
                    // "Not loaded" is the fact the model has to act on; WHY it is not
                    // loaded is the host's business and spending tokens on the
                    // distinction would teach the model nothing it can use.
                    sb.Append('\n');
                    sb.Append("- ").Append(skill.Id).Append(": ")
                      .Append(DescriptionFor(skill, options.MaxCatalogDescriptionChars)).Append('\n');
                    sb.Append("  Instructions: about ")
                      .Append(skill.Manifest.ApproximateBodyTokens.ToString(CultureInfo.InvariantCulture))
                      .Append(" tokens, NOT loaded. If you decide to use this skill, read them first with ")
                      .Append("skills_read(skill=\"")
                      .Append(skill.Id).Append("\", path=\"SKILL.md\") and follow what they say.\n");
                }
            }

            if (catalog.Count > 0 || omitted > 0)
            {
                sb.Append('\n').Append("### Other available skills\n");
                sb.Append(
                    "Not loaded. If one of these matches the task better than your default approach, load it with "
                    + "skills_read(skill=\"<name>\", path=\"SKILL.md\") and follow it.\n");
                foreach (Skill skill in catalog)
                {
                    sb.Append("- ").Append(skill.Id).Append(": ")
                      .Append(DescriptionFor(skill, describeChars)).Append('\n');
                }
                if (omitted > 0)
                {
                    sb.Append("- (")
                      .Append(omitted.ToString(CultureInfo.InvariantCulture))
                      .Append(" further skills are installed but did not fit here; call skills_list to see them.)\n");
                }
            }

            if (options.IncludeUsageInstructions)
            {
                sb.Append('\n').Append("### How to use a skill\n");
                sb.Append(options.ToolsAvailable ? UsageWithTools : UsageWithoutTools);
            }

            return sb.ToString();
        }

        private static void AppendInlinedSkill(StringBuilder sb, Skill skill, SkillPromptOptions options)
        {
            sb.Append('\n').Append("<skill name=\"").Append(skill.Id).Append("\">\n");
            sb.Append(Trim(skill.Description, options.MaxCatalogDescriptionChars)).Append('\n');

            string body = skill.Manifest.Body.TrimEnd();
            int cap = options.ResolvedInlineBodyTokens * SkillTextBudget.BytesPerToken;
            if (body.Length > cap)
            {
                // Without tools there is no continuation to offer, and pointing at one
                // would be worse than silence: the model would believe the rest is
                // available and answer as though it had read it.
                string continuation = options.ToolsAvailable
                    ? "\n\n[...truncated. Read the rest with skills_read(skill=\"" + skill.Id
                      + "\", path=\"SKILL.md\", offset=" + cap.ToString(CultureInfo.InvariantCulture) + ").]"
                    : "\n\n[...truncated here; the rest of these instructions is not available "
                      + "in this conversation. Say so if the task needs it.]";
                body = body.Substring(0, cap).TrimEnd() + continuation;
            }
            if (body.Length > 0)
                sb.Append('\n').Append(body).Append('\n');

            AppendFileIndex(sb, skill, options);
            sb.Append("</skill>\n");
        }

        /// <summary>
        /// List the skill's bundled files so the model knows what it may read without
        /// guessing a path. Only the paths and sizes: the contents are the next tier of
        /// progressive disclosure and are fetched on demand.
        ///
        /// <para>
        /// Skipped entirely when the model has no way to fetch them. Listing files such
        /// a conversation can never open would spend context on an
        /// index that only invites the model to claim it read something it did not —
        /// and naming <c>skills_read</c> to a family whose renderer discards tool
        /// declarations is an instruction it cannot follow.
        /// </para>
        /// </summary>
        private static void AppendFileIndex(StringBuilder sb, Skill skill, SkillPromptOptions options)
        {
            if (!options.ToolsAvailable)
                return;

            List<SkillFile> files = skill.BundledFiles.ToList();
            if (files.Count == 0)
                return;

            // A skill with hundreds of bundled files (a fonts directory, a template set)
            // would otherwise spend more context on its file listing than on its
            // instructions; the model can always call skills_list for the rest.
            const int MaxListed = 40;
            sb.Append("\nFiles bundled with this skill (read with skills_read):\n");
            foreach (SkillFile file in files.Take(MaxListed))
            {
                sb.Append("- ").Append(file.Path);
                if (!file.IsText)
                    sb.Append(" [binary]");
                sb.Append(" (").Append(SkillTextBudget.FormatBytes(file.Bytes)).Append(")\n");
            }
            if (files.Count > MaxListed)
            {
                sb.Append("- (")
                  .Append((files.Count - MaxListed).ToString(CultureInfo.InvariantCulture))
                  .Append(" more; call skills_list to see them all.)\n");
            }
        }

        private static string Trim(string text, int maxChars) => SkillTextBudget.Truncate(text, maxChars);

        /// <summary>
        /// What one catalog line costs at a given description length. Charged against
        /// the SAME text <see cref="Render"/> will emit — the two used to disagree, so a
        /// shorter cap shortened the line without buying any room.
        /// </summary>
        private static int CatalogEntryTokens(Skill skill, int describeChars) =>
            SkillTextBudget.ApproximateTokens(DescriptionFor(skill, describeChars)) + 16;

        /// <summary>
        /// The catalog text for a skill. A hub skill's own description is third-party
        /// text, so the system prompt names where it came from instead; its description and
        /// instructions are one tool call away (skills_list, skills_read), as tool output.
        /// </summary>
        internal static string DescriptionFor(Skill skill, int maxChars) =>
            skill.Source is { } source
                ? Trim("third-party skill installed from the skill hub (" + source.Package
                    + "); skills_list shows its description and skills_read its instructions.", Math.Max(maxChars, 160))
                : Trim(skill.Description, maxChars);

        /// <summary>
        /// The longest description length at which EVERY catalog entry fits the budget,
        /// or the caller's cap when they already do.
        ///
        /// <para>
        /// A ladder rather than a solve: the lengths are few, the catalog is short, and a
        /// value that moves smoothly with the number of skills would change the rendered
        /// prompt — and so the KV-cache prefix — every time a skill is installed. Below
        /// the floor nothing is shortened further and the caller's fill runs as before,
        /// reporting what it had to omit.
        /// </para>
        /// </summary>
        private static int FitCatalogDescriptions(
            IReadOnlyList<Skill> discoverable, SkillPromptOptions options, int room)
        {
            int cap = options.MaxCatalogDescriptionChars;
            if (discoverable.Count == 0)
                return cap;

            // Only the entries that can be LISTED need to fit: past MaxCatalogEntries the
            // fill stops anyway. Bailing out to the full cap here instead was the original
            // bug restored at exactly the size where it hurts most — the more skills are
            // installed, the longer each entry was allowed to be, and the fewer got in.
            int fitting = Math.Min(discoverable.Count, options.MaxCatalogEntries);

            foreach (int candidate in new[] { cap, 512, 320, 240, 160 })
            {
                if (candidate > cap)
                    continue;
                long total = 0;
                for (int i = 0; i < fitting; i++)
                    total += CatalogEntryTokens(discoverable[i], candidate);
                if (total <= room)
                    return candidate;
            }
            return Math.Min(cap, 160);
        }

        /// <summary>
        /// Guidance for the normal case, where the model can fetch what it needs.
        ///
        /// <para>
        /// The rules that matter and why each is here: read the whole SKILL.md before
        /// acting (a half-read instruction file is worse than none); resolve relative
        /// paths against the skill, not the working directory (skills are written with
        /// relative links); prefer a shipped script over retyping its logic (that is the
        /// point of shipping it); and do not chase references the task does not need
        /// (progressive disclosure only saves context if the model exercises it).
        /// </para>
        /// </summary>
        private const string SelectionWithTools =
            "Determine the subject and requested action from the latest user message and relevant conversation "
            + "context before choosing tools. When the user supplies requested information or confirms a handoff "
            + "is ready, resume the pending task unless they change or cancel it. "
            + "Use a skill the user explicitly asks to use, or one whose description matches BOTH. "
            + "Skill availability does not imply relevance. "
            + "If required information or a choice is missing, ask a focused question and wait for the user's answer before "
            + "dependent actions. Do not invent missing details. Continue only work that does not depend on them. "
            + "Use the supplied answer to perform authorized actions yourself, such as filling a field, selecting "
            + "an option, or clicking a control; verify the result and continue the task. "
            + "If no skill matches, respond directly or use another appropriate tool.";

        private const string UsageWithTools =
            "- Decide first. Use a skill explicitly requested by name (including $name), or whose described scope matches the user's current request. "
            + "Needing current information alone does not make a skill relevant. If none matches, answer "
            + "normally or use another appropriate tool; do not call skills_list or skills_read just to start a turn.\n"
            + "- Load before using. Once you have chosen a matching skill, call "
            + "skills_read(skill=\"<name>\", path=\"SKILL.md\") and read the whole result before doing "
            + "the work that skill covers. Reading a skill is not a prerequisite for unrelated tools or answers. If several "
            + "skills apply, read them all in the same turn rather than one per turn.\n"
            + "- A description is not a skill. You have read a skill's instructions only when the result "
            + "of that call is in this conversation. Never follow, summarise or paraphrase instructions "
            + "you have not actually read, and never report a skill's steps as done when you never "
            + "opened it.\n"
            + "- Paths are relative to the skill. A path such as scripts/extract.py means that file inside that "
            + "skill's own directory; pass it to skills_read with the same skill name. Paths and home variables "
            + "shown for another agent's installation are examples, not this host's actual locations. "
            + "When skills_run is offered, run a bundled script with skills_run(skill=\"<name>\", "
            + "path=\"scripts/tool.sh\", args=[\"argument\"]); the host resolves its location. "
            + "Do not copy the wrapper into the workspace or guess a host path.\n"
            + "- Prefer what the skill ships. If it provides a script, run or adapt that script rather than "
            + "rewriting its logic. If it provides a template or asset, reuse it rather than recreating it.\n"
            + "- Read only what you need. Open the reference files the task actually calls for; do not load a "
            + "skill's whole reference directory speculatively.\n"
            + "- Read a selected SKILL.md completely: if truncated, continue from the reported offset until "
            + "the end before following its instructions. Page other resources only as far as the task needs.\n"
            + "- If a skill cannot be applied — a file is missing, the instructions do not fit the task — say so "
            + "in one line, then continue with the best alternative.\n";

        /// <summary>
        /// Guidance for a model whose chat format cannot carry tool declarations.
        /// Telling it to call <c>skills_read</c> there would be
        /// an instruction it has no way to follow, so the wording promises nothing that
        /// is not already in front of it.
        /// </summary>
        private const string UsageWithoutTools =
            "- Decide first. If a skill's description matches the task, follow its instructions above; if none "
            + "does, work normally.\n"
            + "- Everything available to you is already written above. Files a skill mentions but that are not "
            + "reproduced here cannot be opened in this conversation — work from the instructions you have, and "
            + "say so if something essential is missing.\n";
    }
}
