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

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// Names of tools TensorSharp answers itself, in one place.
    ///
    /// <para>
    /// The tool dispatch has to recognise the code-execution tools without referencing
    /// the code-execution namespace — otherwise skills and code execution would reference
    /// each other. Holding the names here rather than repeating a literal means there is
    /// one definition, and <c>ShellTools</c>'s static constructor fails loudly at startup
    /// if its own constants ever drift from these.
    /// </para>
    /// </summary>
    public static class SkillToolNames
    {
        /// <summary>
        /// The shell: the one tool through which the model does everything a workspace
        /// allows — write a file, run it, install a package, search, move, delete.
        ///
        /// <para>
        /// Named <c>shell</c> rather than <c>bash</c> or <c>run_shell</c> because that is
        /// what Codex calls it and what models therefore reach for by reflex, and because
        /// the name has to stay true on Windows, where the interpreter is PowerShell.
        /// Underscore-safe by having no separator at all: several chat templates splice a
        /// tool name into markup unescaped, so a dot would be unsafe there.
        /// </para>
        /// </summary>
        public const string Shell = "shell";

        /// <summary>
        /// Codex's multi-file patch envelope: create, update, delete and rename several
        /// files of the session's workspace in one call, all or nothing.
        ///
        /// <para>
        /// It exists alongside <see cref="Shell"/> rather than being replaced by it
        /// because a heredoc rewrites a whole file while a patch changes three lines of
        /// it, and because the HOST applies it — the bytes are placed by a deterministic
        /// program working from anchors it either finds or refuses, never by a model
        /// re-typing a file it half-remembers.
        /// </para>
        /// </summary>
        public const string ApplyPatch = "apply_patch";

        /// <summary>
        /// Claude Code's <c>Read</c>: show the model a file's real bytes, numbered.
        ///
        /// <para>
        /// The mechanical precondition for editing rather than rewriting. A model that
        /// has not seen a file's exact lines cannot write an anchor that matches one, and
        /// a model whose anchors do not match re-types the file. Until this existed the
        /// only place this host ever showed a model numbered file content was an
        /// <c>apply_patch</c> failure — reachable only by failing.
        /// </para>
        /// <para>
        /// Spelled <c>read_file</c> rather than <c>read</c> because a bare verb collides
        /// with the names models invent for looking at a PICTURE, which
        /// <c>SkillTools.LooksLikeAnImageTool</c> answers structurally; stealing them
        /// would break the only message that tells a model this host cannot show it an
        /// image.
        /// </para>
        /// </summary>
        public const string ReadFile = "read_file";

        /// <summary>
        /// Claude Code's <c>Edit</c>: replace one exact string in one file.
        ///
        /// <para>
        /// The default way to change code here, and deliberately the simplest thing on
        /// the surface: two byte strings and no envelope. The reference that ships the
        /// strongest coding model emits no diff at all, and Anthropic's published
        /// <c>str_replace_based_edit_tool</c> has this same shape — which matters most
        /// for the small local models this host serves, because a V4A envelope has half a
        /// dozen ways to be malformed and every one of them costs a round.
        /// </para>
        /// </summary>
        public const string EditFile = "edit_file";

        /// <summary>
        /// Claude Code's <c>Write</c>: create a file, or deliberately replace one whole.
        ///
        /// <para>
        /// It exists even though a whole-file write is the thing being discouraged, and
        /// for two reasons. Creating a file is a legitimate operation that
        /// <see cref="EditFile"/> structurally cannot do. And it is the SANCTIONED
        /// rewrite path, which is the only place a rewrite can be noticed by
        /// construction rather than by scanning a command line for a redirect — the
        /// scanning approach is what left half a dozen ways to rewrite a file invisible.
        /// </para>
        /// </summary>
        public const string WriteFile = "write_file";

        /// <summary>
        /// Every tool the code runner answers, in one list.
        ///
        /// <para>
        /// Classification and dispatch both read THIS rather than repeating the names,
        /// because they drifted the moment there was more than one to repeat:
        /// <c>apply_patch</c> and <c>list_files</c> were declared to the model and
        /// implemented by the adapter, but the hand-maintained predicate in
        /// <see cref="SkillTools.IsBuiltInTool"/> never learned them. Every call to either
        /// was therefore classified as the CLIENT's, handed to a Web UI that has no
        /// implementation for any tool, and the turn ended having produced nothing at all
        /// — the model's whole reply was inside its thinking channel, so the user saw a
        /// chat that simply stopped.
        /// </para>
        /// </summary>
        public static readonly IReadOnlyList<string> CodeTools =
            new[] { ReadFile, EditFile, WriteFile, Shell, ApplyPatch };

        /// <summary>
        /// Spellings of <see cref="ApplyPatch"/> that models invent, and that are
        /// therefore accepted as if they were the real name.
        ///
        /// <para>
        /// Codex's own system prompt has to say "NEVER try applypatch or apply-patch,
        /// only apply_patch" because models reach for the hyphenated and concatenated
        /// forms anyway. Repeating that warning is worth doing; refusing the call as well
        /// is not — a round spent on a spelling teaches nothing and fixes nothing.
        /// </para>
        /// </summary>
        public static readonly IReadOnlyList<string> ApplyPatchAliases =
            new[] { "apply-patch", "applypatch", "apply_patch_tool" };

        /// <summary>
        /// Spellings of the three file tools that models reach for, accepted as the real
        /// names.
        ///
        /// <para>
        /// The <c>str_replace</c> family is not a guess: those are the command names of
        /// Anthropic's own published <c>str_replace_based_edit_tool</c>, so a model
        /// trained anywhere near it will reach for them. The rest are the ordinary
        /// bare-verb and run-together forms.
        /// </para>
        /// <para>
        /// <b>Nothing here may collide with the names a model invents for looking at a
        /// PICTURE</b> — <c>view</c>, <c>open</c>, <c>display</c>, <c>show</c>,
        /// <c>read_image</c>, <c>look</c> and the rest are claimed by
        /// <c>SkillTools.LooksLikeAnImageTool</c>, which answers them with the one message
        /// that explains this host cannot show an image. Capturing one here would swap
        /// that answer for a file-not-found, and the model would keep trying.
        /// </para>
        /// </summary>
        private static readonly (string Canonical, string[] Aliases)[] FileToolAliases =
        {
            (ReadFile, new[] { "read", "readfile", "read_text_file", "view_file", "cat_file", "get_file" }),
            (EditFile, new[]
            {
                "edit", "editfile", "str_replace", "str_replace_editor",
                "str_replace_based_edit_tool", "replace_in_file", "apply_edit", "string_replace",
            }),
            (WriteFile, new[] { "write", "writefile", "create_file", "new_file", "put_file" }),
        };

        /// <summary>
        /// The canonical name for <paramref name="name"/>, or null when it is not one of
        /// the file tools.
        ///
        /// <para>
        /// One resolver rather than three predicates, because three predicates is the
        /// shape that already produced one silent outage here: a tool declared to the
        /// model and implemented by the adapter, with the hand-maintained classifier never
        /// taught about it, so every call was handed to a client that implements nothing
        /// and the turn ended having rendered nothing at all.
        /// </para>
        /// </summary>
        public static string? ResolveFileTool(string? name)
        {
            if (string.IsNullOrEmpty(name))
                return null;

            foreach ((string canonical, string[] aliases) in FileToolAliases)
            {
                if (string.Equals(canonical, name, StringComparison.Ordinal))
                    return canonical;
                for (int i = 0; i < aliases.Length; i++)
                {
                    if (string.Equals(aliases[i], name, StringComparison.OrdinalIgnoreCase))
                        return canonical;
                }
            }
            return null;
        }

        /// <summary>True when <paramref name="name"/> is one of <see cref="CodeTools"/>.</summary>
        public static bool IsCodeTool(string? name)
        {
            if (string.IsNullOrEmpty(name))
                return false;
            for (int i = 0; i < CodeTools.Count; i++)
            {
                if (string.Equals(CodeTools[i], name, StringComparison.Ordinal))
                    return true;
            }
            return IsApplyPatchAlias(name) || ResolveFileTool(name) != null;
        }

        /// <summary>True when <paramref name="name"/> is a misspelling of <see cref="ApplyPatch"/>.</summary>
        public static bool IsApplyPatchAlias(string? name)
        {
            if (string.IsNullOrEmpty(name))
                return false;
            for (int i = 0; i < ApplyPatchAliases.Count; i++)
            {
                if (string.Equals(ApplyPatchAliases[i], name, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Start a sub-agent: another copy of the model with its own context, working in
        /// parallel and reporting a final answer back.
        ///
        /// <para>
        /// The five agent names are Codex's own (<c>spawn_agent</c>, <c>send_input</c>,
        /// <c>wait_agent</c>, <c>close_agent</c> from its first multi-agent surface and
        /// <c>list_agents</c> from its second), for the same reason <see cref="Shell"/> is:
        /// they are the names models have seen and reach for by reflex.
        /// </para>
        /// </summary>
        public const string SpawnAgent = "spawn_agent";

        /// <summary>Give one of the caller's sub-agents a follow-up message.</summary>
        public const string SendInput = "send_input";

        /// <summary>Block until one of the caller's sub-agents finishes, and collect its answer.</summary>
        public const string WaitAgent = "wait_agent";

        /// <summary>Stop and discard one of the caller's sub-agents.</summary>
        public const string CloseAgent = "close_agent";

        /// <summary>List the caller's sub-agents and their status.</summary>
        public const string ListAgents = "list_agents";

        /// <summary>
        /// Every tool the sub-agent runtime answers, in declaration order.
        ///
        /// <para>
        /// Matched ORDINALLY and with no aliases, unlike the file tools. These names are
        /// only ever declared together, by one host feature, so a model that has them has
        /// just read their exact spelling; and none of them may capture a name a caller's
        /// own tool could plausibly use, which an alias list would start to do.
        /// </para>
        /// </summary>
        public static readonly IReadOnlyList<string> AgentTools =
            new[] { SpawnAgent, SendInput, WaitAgent, CloseAgent, ListAgents };

        /// <summary>True when <paramref name="name"/> is one of <see cref="AgentTools"/>.</summary>
        public static bool IsAgentTool(string? name)
        {
            if (string.IsNullOrEmpty(name))
                return false;
            for (int i = 0; i < AgentTools.Count; i++)
            {
                if (string.Equals(AgentTools[i], name, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }
    }
}
