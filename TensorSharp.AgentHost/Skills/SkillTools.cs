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
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using TensorSharp.AgentHost.Agents;

using TensorSharp.Runtime;
namespace TensorSharp.AgentHost.Skills
{
    /// <summary>
    /// The tools TensorSharp offers a model so it can pull skill content on demand,
    /// and the code that answers them.
    ///
    /// <para>
    /// These are unlike every other tool in the system: TensorSharp executes them
    /// itself rather than handing them back to the caller. That is safe because they
    /// are read-only and confined by <see cref="SkillPathGuard"/> to a directory the
    /// operator already chose to expose, and it is necessary because progressive
    /// disclosure has to work for a client that knows nothing about skills — an
    /// ordinary OpenAI client sends <c>skills: ["pdf"]</c>, and gets back a finished
    /// answer, not a tool call it has no idea how to service.
    /// </para>
    /// <para>
    /// <b>The declarations are deliberately flat.</b> <see cref="ToolParameter"/>
    /// carries only a type name, a description and an enum; <c>items</c>, nested
    /// <c>properties</c> and every other JSON Schema keyword are dropped when a tool
    /// is parsed and cannot be re-emitted, and the Harmony renderer degrades an
    /// <c>array</c> parameter to <c>any[]</c>. Every parameter here is therefore a
    /// string or an integer, and the names use underscores rather than dots because
    /// several families splice a tool's name into their markup unescaped.
    /// </para>
    /// </summary>
    public static class SkillTools
    {
        /// <summary>Lists the skills reachable in this conversation.</summary>
        public const string ListToolName = "skills_list";

        /// <summary>Reads one file out of one skill.</summary>
        public const string ReadToolName = "skills_read";

        /// <summary>Runs one of a skill's bundled scripts. Offered only when the host opts in.</summary>
        public const string RunToolName = "skills_run";

        /// <summary>Default ceiling on how much text one <see cref="ReadToolName"/> call returns.</summary>
        public const int DefaultMaxReadBytes = 48 * 1024;

        /// <summary>True when <paramref name="name"/> is one of the Agent Skills tools.</summary>
        public static bool IsSkillTool(string? name) =>
            string.Equals(name, ListToolName, StringComparison.Ordinal)
            || string.Equals(name, ReadToolName, StringComparison.Ordinal)
            || string.Equals(name, RunToolName, StringComparison.Ordinal);

        /// <summary>
        /// True when TensorSharp answers <paramref name="name"/> itself rather than
        /// handing it to the caller.
        ///
        /// <para>
        /// This, not <see cref="IsSkillTool"/>, is what a generation loop must partition
        /// on. The two came apart when code execution added a built-in that is not a skill
        /// tool: a loop still asking "is it a skill tool?" would forward <c>shell</c> to
        /// a client that has no implementation for it, which is exactly the stall the
        /// in-process loops exist to remove.
        /// </para>
        /// </summary>
        public static bool IsBuiltInTool(string? name) =>
            IsSkillTool(name) || SkillToolNames.IsCodeTool(name) || SkillToolNames.IsAgentTool(name);

        /// <summary>
        /// Split one round's tool calls three ways: the ones this host answers, the ones
        /// the CLIENT declared and must answer, and the ones NOBODY declared.
        ///
        /// <para>
        /// The third bucket is the one that matters, and it had no name before. A loop
        /// that only asks "is it mine?" hands everything else to the caller, which is
        /// right for a tool the caller actually declared and catastrophic for anything
        /// else: the Web UI registers no client tools at all and has no handler for the
        /// tool-call frame, so an unrecognised name — a built-in the classifier had not
        /// learned, a name the model invented — ended the turn with nothing rendered.
        /// A model that spent its whole reply inside &lt;think&gt; then looks, to the
        /// user, exactly like a hang.
        /// </para>
        /// <para>
        /// Answered in the loop instead, an unknown name costs one round and the model
        /// recovers, which is what already happens for every other bad call.
        /// </para>
        /// </summary>
        /// <param name="calls">The round's parsed calls; nulls are ignored.</param>
        /// <param name="clientTools">
        /// <b>The caller's OWN tools</b>, not the merged roster the model was shown.
        /// The distinction is the whole point: the merged roster contains this host's
        /// declarations too, so keying on it would put a host tool the classifier had
        /// not learned back into the client bucket — reproducing, exactly, the bug this
        /// split exists to end. Only a name the caller supplied can be the caller's to
        /// answer.
        ///
        /// <para>
        /// <b>Null means the caller does not know what it declared</b>, and then nothing
        /// can honestly be called unknown: the split falls back to the two-way one and
        /// <paramref name="unknown"/> comes back empty. That is the CLI's <c>--tools</c>
        /// contract, where a non-built-in call is shown to the operator rather than
        /// answered.
        /// </para>
        /// </param>
        public static void Partition(
            IEnumerable<ToolCall?>? calls,
            IReadOnlyCollection<ToolFunction>? clientTools,
            out List<ToolCall> builtIn,
            out List<ToolCall> client,
            out List<ToolCall> unknown)
        {
            builtIn = new List<ToolCall>();
            client = new List<ToolCall>();
            unknown = new List<ToolCall>();
            if (calls == null)
                return;

            HashSet<string>? theirs = null;
            if (clientTools != null)
            {
                // Case-insensitive, because Merge resolves a client tool shadowing a
                // built-in that way: a caller's "Run_Code" keeps its declaration, so a
                // call by that spelling is theirs to answer.
                theirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (ToolFunction tool in clientTools)
                {
                    if (!string.IsNullOrEmpty(tool?.Name))
                        theirs.Add(tool!.Name!);
                }
            }

            foreach (ToolCall? call in calls)
            {
                if (call == null)
                    continue;

                // The client's own declaration wins over ours on a shadowed name, which
                // is how Merge rendered it: the model was shown THEIR tool, so answering
                // it here would be running something else entirely.
                if (theirs != null && call.Name != null && theirs.Contains(call.Name))
                    client.Add(call);
                else if (IsBuiltInTool(call.Name))
                    builtIn.Add(call);
                else if (theirs == null)
                    client.Add(call);
                else
                    unknown.Add(call);
            }
        }

        /// <summary>
        /// What to tell the model about a call nobody can answer: that the name does not
        /// exist, and which names do.
        ///
        /// <para>
        /// The list is the useful half. A model that guessed a name is one nudge away
        /// from the right one, and small local models act on tool RESULTS far more
        /// reliably than on declarations they read thousands of tokens ago.
        /// </para>
        /// </summary>
        public static string DescribeUnknownTool(
            string? name,
            IReadOnlyList<ToolFunction>? declaredTools,
            IReadOnlyList<string>? knownSkillIds = null)
        {
            var sb = new StringBuilder();
            sb.Append("Error: there is no tool called '")
              .Append(string.IsNullOrEmpty(name) ? "(unnamed)" : name)
              .Append("', so nothing was run.");

            // The name of a skill it was just shown, called as if it were a tool. This
            // is the most useful thing an unknown name can turn out to be, and the
            // generic answer is actively harmful: told only "there is no tool called
            // 'research'. The tools you have are: …", a model concluded the skill did not
            // exist and answered the question without it. It had chosen correctly and was
            // one call away; what it needed was the calling convention, not the list.
            if (!string.IsNullOrEmpty(name) && knownSkillIds != null)
            {
                foreach (string id in knownSkillIds)
                {
                    if (!string.Equals(id, name, StringComparison.OrdinalIgnoreCase))
                        continue;
                    sb.Append(" '").Append(id).Append("' is a SKILL, not a tool: skills are not called by "
                        + "name. Read it first with ").Append(ReadToolName).Append("(skill: \"").Append(id)
                      .Append("\", path: \"SKILL.md\"), do what it says, and run any script it ships with ")
                      .Append(RunToolName).Append(". It is available — this is the wrong way to reach it, "
                        + "not a missing capability.");
                    return sb.ToString();
                }
            }

            var names = new List<string>();
            if (declaredTools != null)
            {
                foreach (ToolFunction tool in declaredTools)
                {
                    if (!string.IsNullOrEmpty(tool?.Name))
                        names.Add(tool!.Name!);
                }
            }

            if (names.Count > 0)
            {
                names.Sort(StringComparer.Ordinal);
                sb.Append(" The tools you have are: ").Append(string.Join(", ", names)).Append('.');
            }

            // The one wrong guess worth answering specifically. Several skills tell the
            // model to LOOK at what it produced — check the deck, view the gif, inspect
            // the render — and this host cannot show it an image at all. In the measured
            // logs that cost one hallucinated `view` call plus four turns totalling
            // 27,810 tokens whose reasoning is partly spent working out that the tool does
            // not exist. A correct list of the tools that do exist does not answer the
            // question the model was actually asking, which is "how do I check my work".
            if (LooksLikeAnImageTool(name))
            {
                sb.Append(" This host cannot show you an image — there is no tool for it, and there will "
                        + "not be one in this turn. Check the file STRUCTURALLY instead, with a command "
                        + "that reads it and prints what is inside (for a deck: open it with python-pptx "
                        + "and print the slide count and each slide's text). Then say in your answer that "
                        + "you could not look at it.");
                return sb.ToString();
            }

            sb.Append(" Call one of those, or answer the user directly with what you already know.");
            return sb.ToString();
        }

        /// <summary>
        /// Names a model invents when it wants to look at a picture. Matched loosely on
        /// purpose: the point is to answer the QUESTION, and the exact word it chose for a
        /// tool that does not exist carries no information.
        /// </summary>
        private static bool LooksLikeAnImageTool(string? name)
        {
            if (string.IsNullOrEmpty(name))
                return false;
            foreach (string invented in new[]
            {
                "view", "view_image", "open", "open_image", "display", "show", "show_image",
                "read_image", "look", "look_at", "see", "screenshot", "render", "preview",
                "image", "inspect_image",
            })
            {
                if (string.Equals(name, invented, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        // Per-tool aliases for the code tools used to live here. They were a fourth
        // partial copy of the name set — the shape that produced this bug — and had no
        // callers left once classification and dispatch both went through
        // SkillToolNames. Name a code tool as SkillToolNames.X.

        /// <summary>
        /// The tool declarations to offer alongside the caller's own.
        /// </summary>
        /// <param name="allowScripts">
        /// Include <see cref="RunToolName"/>. Off by default and everywhere it is not
        /// explicitly enabled: a skill is untrusted content, and running its scripts is
        /// arbitrary code execution on the host.
        /// </param>
        public static List<ToolFunction> BuiltIn(bool allowScripts = false)
        {
            var tools = new List<ToolFunction>(allowScripts ? 3 : 2)
            {
                new()
                {
                    Name = ListToolName,
                    Description =
                        "List the agent skills available in this conversation, with each skill's name, "
                        + "description and bundled files. Call this when you need a skill that was not "
                        + "already described to you, or to find the exact path of a file inside a skill.",
                    // No parameters at all. Note that Required must stay empty AND
                    // Parameters must stay empty together: the Jinja rendering path marks
                    // every parameter required when Required is empty, so a tool with
                    // optional-only arguments would be misdeclared there.
                    Parameters = new Dictionary<string, ToolParameter>(),
                    Required = new List<string>(),
                },
                new()
                {
                    Name = ReadToolName,
                    Description =
                        "Read one file from a skill whose description matches the user's current request. "
                        + "Do not read an unrelated skill as a preliminary step. Use path \"SKILL.md\" for the skill's own instructions, "
                        + "or a path relative to the skill directory such as \"references/api.md\" or "
                        + "\"scripts/extract.py\" for a file it bundles. Long files come back in pages: if the "
                        + "result says it was truncated, call again with offset set to the next offset it reports.",
                    Parameters = new Dictionary<string, ToolParameter>
                    {
                        ["skill"] = new()
                        {
                            Type = "string",
                            Description = "The skill's name, exactly as listed.",
                        },
                        ["path"] = new()
                        {
                            Type = "string",
                            Description =
                                "File path relative to the skill's own directory. \"SKILL.md\" is the skill's "
                                + "instructions. Never an absolute path and never outside the skill.",
                        },
                        ["offset"] = new()
                        {
                            Type = "integer",
                            // Two tools with a same-named parameter that means different
                            // things is exactly what this codebase refuses to let drift
                            // silently. read_file's offset is a LINE number, because a
                            // model copying text out of a numbered listing thinks in
                            // lines; this one is a BYTE offset, because it continues a
                            // truncated skill body whose cut point is a byte count.
                            // Neither is going to change, so both say which they are.
                            Description = "Byte offset to start at. Omit for the beginning of the file. "
                                + "(This counts BYTES, unlike " + SkillToolNames.ReadFile
                                + "'s offset, which is a line number.)",
                        },
                    },
                    Required = new List<string> { "skill", "path" },
                },
            };

            if (allowScripts)
            {
                tools.Add(new ToolFunction
                {
                    Name = RunToolName,
                    Description =
                        "Run one of a skill's bundled scripts on this machine and return what it printed. "
                        + "Use only a skill whose description matches the user's current request, after reading "
                        + "its SKILL.md. Use a script path documented there; do not invent a script for an unrelated task. "
                        + "Pass the script's path in 'path' and everything you would have typed after it on "
                        + "the command line in 'args'. Only files inside the skill's own directory can be "
                        + "run. A SKILL.md may show `cd <skill>/scripts` followed by `python3 tool.py ...` as a "
                        + "terminal example; do not call cd or shell and do not include python3 here—make one "
                        + "skills_run call with path=`scripts/tool.py` and only `...` in args. Follow SKILL.md; "
                        + "do not read or copy a bundled script unless a traceback proves that script is broken.",
                    Parameters = new Dictionary<string, ToolParameter>
                    {
                        ["skill"] = new()
                        {
                            Type = "string",
                            Description = "The skill's name, exactly as listed.",
                        },
                        ["path"] = new()
                        {
                            // Spelling out what does NOT belong here, because the natural mistake
                            // is to paste the whole command line: a SKILL.md writes its invocation
                            // as "RUN scripts/budget.py <payload_kg>", and a model that copies that
                            // shape lands the arguments in this parameter. See
                            // SkillScriptRunner.ExplainUnresolvedScript for the other half of this.
                            Type = "string",
                            Description =
                                "The script's file path relative to the skill directory, e.g. "
                                + "\"scripts/extract.py\". The path ONLY - never append arguments to it. "
                                + "Anything that follows the filename on a command line goes in 'args'.",
                        },
                        ["args"] = new()
                        {
                            Type = "string",
                            Description =
                                "Arguments to pass to the script, i.e. everything after the filename on the "
                                + "command line. For \"scripts/budget.py 2400\" this is \"2400\". Either one "
                                + "string quoted the way a shell would be, or a JSON array of strings (also "
                                + "accepted when encoded inside a string). Omit it only when the script takes none. No shell "
                                + "is involved, so pipes, redirection and variable expansion do not work.",
                        },
                    },
                    Required = new List<string> { "skill", "path" },
                });
            }

            return tools;
        }

        /// <summary>
        /// Merge the built-in tools into a caller-supplied list.
        /// </summary>
        /// <remarks>
        /// A client tool that already carries one of these names wins: the client has an
        /// implementation and an expectation, and shadowing it would break a working
        /// integration to add a feature it did not ask for. The collision is reported so
        /// the host can log it.
        /// </remarks>
        public static List<ToolFunction> Merge(
            List<ToolFunction>? clientTools,
            bool allowScripts,
            out IReadOnlyList<string> shadowed)
        {
            var conflicts = new List<string>();
            var merged = clientTools != null ? new List<ToolFunction>(clientTools) : new List<ToolFunction>();
            var existing = new HashSet<string>(
                merged.Select(t => t.Name ?? string.Empty), StringComparer.OrdinalIgnoreCase);

            foreach (ToolFunction tool in BuiltIn(allowScripts))
            {
                if (existing.Contains(tool.Name))
                {
                    conflicts.Add(tool.Name);
                    continue;
                }
                merged.Add(tool);
            }

            shadowed = conflicts;
            return merged;
        }

        /// <summary>
        /// Answer one skill tool call.
        /// </summary>
        /// <param name="call">The call the model made.</param>
        /// <param name="context">Which skills are reachable and what is permitted.</param>
        /// <returns>
        /// The text to feed back as the tool result. Never throws and never returns
        /// null: an error is part of the conversation, phrased so the model can correct
        /// itself — a thrown exception would abort a request that is still perfectly
        /// answerable.
        /// </returns>
        /// <summary>Answer a code-execution call, or say why this host will not.</summary>
        private static SkillToolResult ExecuteCode(ToolCall call, SkillToolContext context, Action<string>? onOutput)
        {
            if (context.CodeRunner is not { } runner)
            {
                return SkillToolResult.Failure(
                    "running code is not enabled on this host. Answer without it.");
            }

            // The reachable skills' directories go with the call, so a skill's bundled
            // package can be IMPORTED by the program rather than only read. Without this
            // a model wanting slack-gif-creator's core/ module had to copy the package
            // into its workspace file by file.
            var skillDirectories = new List<string>();
            foreach (Skill skill in context.Reachable)
            {
                if (!string.IsNullOrEmpty(skill.RootDirectory))
                    skillDirectories.Add(skill.RootDirectory);
            }

            return runner.Execute(
                call, context.CodeInputFiles, onOutput, context.Workspace, skillDirectories);
        }

        /// <param name="onToolOutput">
        /// Live tap for a tool that runs a process: called per stdout/stderr line while
        /// it executes, so a host can stream what is happening. Null taps nothing, and
        /// tools that produce no live output never call it.
        /// </param>
        public static SkillToolResult Execute(
            ToolCall? call, SkillToolContext context, Action<string>? onToolOutput = null)
        {
            ArgumentNullException.ThrowIfNull(context);
            if (call == null)
                return SkillToolResult.Failure("No tool call was supplied.");

            try
            {
                // Attachment availability is a property of the whole built-in tool
                // surface, not just shell. In particular, skills_run launches through
                // its own runner and would otherwise bypass CodeRunnerAdapter entirely.
                // Not for the agent tools: they touch no file, and a wait_agent that
                // blocks for minutes must not first take the workspace's staging path.
                if (context.Workspace != null && IsBuiltInTool(call.Name) && !SkillToolNames.IsAgentTool(call.Name))
                    CodeInputFileStager.Stage(context.CodeInputFiles, context.Workspace);

                return call.Name switch
                {
                    ListToolName => ExecuteList(context),
                    ReadToolName => ExecuteRead(call, context),
                    RunToolName => ExecuteRun(call, context, onToolOutput),

                    // Every code tool goes to the same runner, which dispatches on the
                    // name itself. Listing them one by one here is what let apply_patch
                    // and list_files be declared to the model and then refused as tools
                    // "this host answers"; asking SkillToolNames instead means a new
                    // code tool is one edit, in the place that already holds the names.
                    _ when SkillToolNames.IsCodeTool(call.Name) =>
                        ExecuteCode(call, context, onToolOutput),

                    // The loops await these directly (SubAgentScope.ExecuteAsync), so a
                    // wait never parks a thread. This synchronous path is for callers that
                    // have no await, and it is correct there too: the runtime's own turn
                    // token still bounds the wait.
                    _ when SkillToolNames.IsAgentTool(call.Name) => context.Agents is { } agents
                        ? agents.ExecuteAsync(call, onToolOutput, CancellationToken.None).GetAwaiter().GetResult()
                        : SkillToolResult.Failure("sub-agents are not enabled on this host. Do the task yourself."),

                    _ => SkillToolResult.Failure($"'{call.Name}' is not a tool this host answers."),
                };
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
            {
                // The model is mid-task; an unexpected failure here should cost it one
                // tool call, not the whole answer.
                return SkillToolResult.Failure($"The {call.Name} call failed: {ex.Message}");
            }
        }

        private static SkillToolResult ExecuteList(SkillToolContext context)
        {
            IReadOnlyList<Skill> skills = context.Reachable;
            if (skills.Count == 0)
                return SkillToolResult.Success("No skills are available in this conversation.");

            var sb = new StringBuilder();
            sb.Append(skills.Count.ToString(CultureInfo.InvariantCulture))
              .Append(skills.Count == 1 ? " skill is available.\n" : " skills are available.\n");

            foreach (Skill skill in skills)
            {
                sb.Append('\n').Append(skill.Id).Append('\n');
                sb.Append("  ").Append(skill.Description).Append('\n');
                if (!string.IsNullOrEmpty(skill.Manifest.Compatibility))
                    sb.Append("  requires: ").Append(skill.Manifest.Compatibility).Append('\n');

                List<SkillFile> files = skill.BundledFiles.ToList();
                if (files.Count == 0)
                {
                    sb.Append("  files: SKILL.md only\n");
                    continue;
                }

                sb.Append("  files: SKILL.md");
                foreach (SkillFile file in files.Take(30))
                {
                    sb.Append(", ").Append(file.Path);
                    if (!file.IsText)
                        sb.Append(" [binary]");
                }
                if (files.Count > 30)
                {
                    sb.Append(", and ")
                      .Append((files.Count - 30).ToString(CultureInfo.InvariantCulture))
                      .Append(" more");
                }
                sb.Append('\n');
            }
            return SkillToolResult.Success(sb.ToString());
        }

        private static SkillToolResult ExecuteRead(ToolCall call, SkillToolContext context)
        {
            string? skillName = ReadString(call, "skill");
            string? path = ReadString(call, "path") ?? ReadString(call, "file") ?? ReadString(call, "resource");
            long offset = ReadInt64(call, "offset") ?? 0;

            if (string.IsNullOrWhiteSpace(skillName))
                return SkillToolResult.Failure($"{ReadToolName} needs a 'skill' argument naming which skill to read.");

            if (!context.TryResolve(skillName, out Skill? skill, out string? resolveError))
                return SkillToolResult.Failure(resolveError!);

            // A model that has just been told a skill exists very often asks for it with
            // no path at all, meaning "give me the instructions". Answering that is
            // better than spending a round-trip teaching it the argument.
            if (string.IsNullOrWhiteSpace(path))
                path = SkillManifestParser.SkillFileName;

            path = StripSkillPrefix(path!, skill!.Id);

            if (!skill.TryReadResource(path, context.MaxReadBytes, offset, out SkillResourceContent content, out string? error))
                return SkillToolResult.Failure($"Cannot read '{path}' from skill '{skill.Id}': {error}");

            var sb = new StringBuilder();
            sb.Append("Skill: ").Append(skill.Id).Append('\n');
            sb.Append("File: ").Append(content.Path).Append('\n');
            if (content.Truncated || content.OffsetBytes > 0)
            {
                sb.Append("Bytes ").Append(content.OffsetBytes.ToString(CultureInfo.InvariantCulture))
                  .Append('-').Append(content.NextOffsetBytes.ToString(CultureInfo.InvariantCulture))
                  .Append(" of ").Append(content.TotalBytes.ToString(CultureInfo.InvariantCulture)).Append('\n');
            }
            sb.Append('\n').Append(content.Text);
            if (content.Truncated)
            {
                sb.Append("\n\n[Truncated. Continue with ")
                  .Append(ReadToolName).Append("(skill=\"").Append(skill.Id)
                  .Append("\", path=\"").Append(content.Path)
                  .Append("\", offset=").Append(content.NextOffsetBytes.ToString(CultureInfo.InvariantCulture))
                  .Append(").]");
            }

            // Reading SKILL.md is the moment the skill is activated, and it is the right
            // moment to say what else it ships: the file index is tier-three routing
            // information, useless to a model that has not decided to use the skill and
            // necessary to one that just has. It used to live in the system prompt, paid
            // for on every request whether the skill was touched or not.
            if (!content.Truncated
                && string.Equals(content.Path, SkillManifestParser.SkillFileName, StringComparison.Ordinal))
            {
                AppendBundledFileIndex(sb, skill);
                if (context.ScriptRunner != null)
                    AppendHostExecutionGuidance(sb, skill);
            }

            return SkillToolResult.Success(sb.ToString(), skill.Id, content.Path);
        }

        /// <summary>
        /// Place host-specific execution guidance next to the instructions it adapts.
        /// Portable skills often show another agent's installation home; repeating that
        /// path in a shell cannot locate the registered bundle. Keep this separate from
        /// the verbatim file content and offer an actual registered script path.
        /// </summary>
        private static void AppendHostExecutionGuidance(StringBuilder sb, Skill skill)
        {
            SkillFile? script = skill.BundledFiles
                .Where(file => file.Kind == SkillFileKind.Script && file.IsText)
                .Select(file => (SkillFile?)file)
                .FirstOrDefault();
            if (script == null)
                return;

            sb.Append("\n\n[TensorSharp host execution guidance; not part of SKILL.md]\n")
              .Append("This skill and its bundled scripts are available. Example installation paths and ")
              .Append("agent-home variables in the instructions do not identify their location on this host. ")
              .Append("Run bundled scripts through ").Append(RunToolName)
              .Append("; it resolves the registered skill directory. Do not probe those example home paths, ")
              .Append("copy the wrapper, or report it missing because a shell cannot find that example path.\n")
              .Append("Example tool arguments for ").Append(RunToolName).Append(": ")
              .Append(JsonSerializer.Serialize(new
              {
                  skill = skill.Id,
                  path = script.Value.Path,
                  args = Array.Empty<string>(),
              }))
              .Append("\nChoose the script documented for the task and replace args with its command arguments ")
              .Append("only, without the interpreter or script path. Follow the skill's prerequisite checks ")
              .Append("and workflow using the available host tools.\n");
        }

        /// <summary>
        /// The skill's bundled files, appended to a <c>SKILL.md</c> read so the model
        /// knows what it may open without guessing a path. Paths and sizes only — the
        /// contents are the next tier and are fetched on demand.
        /// </summary>
        private static void AppendBundledFileIndex(StringBuilder sb, Skill skill)
        {
            List<SkillFile> files = skill.BundledFiles.ToList();
            if (files.Count == 0)
                return;

            sb.Append("\n\nFiles this skill ships (read with ").Append(ReadToolName)
              .Append("(skill=\"").Append(skill.Id).Append("\", path=\"...\")):\n");
            foreach (SkillFile file in files.Take(MaxIndexedFiles))
            {
                sb.Append("  ").Append(file.Path);
                if (!file.IsText)
                    sb.Append(" [binary]");
                sb.Append(" (").Append(SkillTextBudget.FormatBytes(file.Bytes)).Append(")\n");
            }
            if (files.Count > MaxIndexedFiles)
            {
                sb.Append("  and ")
                  .Append((files.Count - MaxIndexedFiles).ToString(CultureInfo.InvariantCulture))
                  .Append(" more.\n");
            }
        }

        /// <summary>Most bundled files named in one activation. A long tail helps nobody.</summary>
        private const int MaxIndexedFiles = 40;

        private static SkillToolResult ExecuteRun(ToolCall call, SkillToolContext context, Action<string>? onOutput)
        {
            if (context.ScriptRunner == null)
            {
                return SkillToolResult.Failure(
                    "Running skill scripts is disabled on this host. Read the script with "
                    + ReadToolName + " and carry out its steps yourself, or report what it would do.");
            }

            string? skillName = ReadString(call, "skill");
            string? path = ReadString(call, "path") ?? ReadString(call, "script");
            IReadOnlyList<string>? canonicalArgs =
                ReadRunArgumentList(call, "args", context.CodeInputFiles);
            IReadOnlyList<string>? aliasedArgs = canonicalArgs is { Count: > 0 }
                ? null
                : ReadRunArgumentList(call, "arguments", context.CodeInputFiles);
            IReadOnlyList<string> args = canonicalArgs is { Count: > 0 }
                ? canonicalArgs
                : aliasedArgs is { Count: > 0 }
                    ? aliasedArgs
                    : canonicalArgs ?? aliasedArgs ?? Array.Empty<string>();

            if (string.IsNullOrWhiteSpace(skillName) || string.IsNullOrWhiteSpace(path))
                return SkillToolResult.Failure($"{RunToolName} needs both a 'skill' and a 'path' argument.");

            if (!context.TryResolve(skillName!, out Skill? skill, out string? resolveError))
                return SkillToolResult.Failure(resolveError!);

            path = StripSkillPrefix(path!, skill!.Id);
            return context.ScriptRunner.Run(skill, path, args, onOutput, ReadPackagesArgument(call));
        }

        /// <summary>
        /// Read a script argument vector, with one deliberately narrow repair for an
        /// attached filename. A scalar normally has shell-style tokenization, but local
        /// models also emit <c>args="my results.csv"</c> after copying the exact name
        /// from the attachment prompt. When the whole scalar is exactly one known input
        /// name, it is unambiguously one argument even though the model omitted quotes.
        /// </summary>
        /// <remarks>
        /// Do not apply this to an array: its item boundaries are already explicit. Do
        /// not search within a larger command line either; options and ordinary scalar
        /// arguments must retain <see cref="SkillScriptRunner.SplitArguments"/> semantics.
        /// </remarks>
        private static IReadOnlyList<string>? ReadRunArgumentList(
            ToolCall call, string name, IReadOnlyList<CodeInputFile> inputFiles)
        {
            if (call.Arguments == null || !call.Arguments.TryGetValue(name, out object? value) || value == null)
                return null;

            string? scalar = value switch
            {
                string text => text,
                JsonElement { ValueKind: JsonValueKind.String } element => element.GetString(),
                _ => null,
            };

            if (scalar != null && inputFiles != null)
            {
                string candidate = scalar.Trim();
                foreach (CodeInputFile input in inputFiles)
                {
                    string attachedName = Path.GetFileName(input.Name ?? string.Empty);
                    if (attachedName.Length > 0 && string.Equals(candidate, attachedName, StringComparison.Ordinal))
                        return new[] { attachedName };
                }
            }

            return ReadArgumentList(call, name);
        }

        /// <summary>
        /// The optional <c>packages</c> argument of <see cref="RunToolName"/>: a
        /// comma-separated string by declaration, an array in practice — the same two
        /// shapes a list-valued argument has to accept.
        /// </summary>
        private static IReadOnlyList<string>? ReadPackagesArgument(ToolCall call)
        {
            IReadOnlyList<string>? raw = ReadArgumentList(call, "packages");
            if (raw == null || raw.Count == 0)
                return null;

            List<string> names = raw
                .SelectMany(item => (item ?? string.Empty).Split(
                    new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
                .Select(name => name.Trim())
                .Where(name => name.Length > 0)
                .ToList();
            return names.Count > 0 ? names : null;
        }

        /// <summary>
        /// Drop a leading <c>&lt;skill-name&gt;/</c> from a path.
        ///
        /// <para>
        /// Models routinely answer "read pdf's reference.md" with
        /// <c>path="pdf/reference.md"</c>, because that is how the file is spelled
        /// relative to the skills directory rather than relative to the skill. The path
        /// guard would reject it as "does not exist", the model would retry with the
        /// same shape, and the loop would burn its whole iteration budget. Accepting
        /// both spellings costs nothing and cannot widen the sandbox: the result is
        /// still resolved inside the same skill.
        /// </para>
        /// <para>
        /// Only a leading <c>./</c> is stripped alongside it. A <c>..</c> segment is
        /// deliberately left in place so <see cref="SkillPathGuard"/> rejects it with
        /// the message that says what actually happened — quietly rewriting
        /// <c>../other/SKILL.md</c> into a path inside this skill would be safe but
        /// would report "does not exist", teaching the model the file is missing rather
        /// than that it may not look there. Leading dots are otherwise preserved, since
        /// a skill may legitimately ship a dotfile.
        /// </para>
        /// </summary>
        private static string StripSkillPrefix(string path, string skillId)
        {
            string normalized = path.Replace('\\', '/').Trim();
            while (normalized.StartsWith("./", StringComparison.Ordinal))
                normalized = normalized.Substring(2);

            string prefix = skillId + "/";
            return normalized.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                ? normalized.Substring(prefix.Length)
                : normalized;
        }

        /// <summary>
        /// Read a string argument. Values arrive as <see cref="JsonElement"/>, boxed
        /// primitives or strings depending on which parser produced them, so every
        /// shape a model's arguments legitimately take is accepted.
        /// </summary>
        internal static string? ReadString(ToolCall call, string name)
        {
            if (call.Arguments == null || !call.Arguments.TryGetValue(name, out object? value) || value == null)
                return null;

            return value switch
            {
                string s => s,
                JsonElement { ValueKind: JsonValueKind.String } e => e.GetString(),
                JsonElement { ValueKind: JsonValueKind.Null } => null,
                JsonElement e => e.ToString(),
                _ => Convert.ToString(value, CultureInfo.InvariantCulture),
            };
        }

        /// <summary>
        /// Read the argument vector for <see cref="RunToolName"/>, accepting the shapes
        /// a model actually produces.
        ///
        /// <para>
        /// The declaration says <c>args</c> is a string, because
        /// <see cref="ToolParameter"/> cannot express an array — and models emit it as
        /// an array anyway, because that is what an argument list looks like. Observed
        /// on Gemma 4: given a string parameter it produced
        /// <c>skills_run{args:["2400"]}</c>, the read returned nothing, and the model
        /// gave up on the tool and did the arithmetic itself — getting a different
        /// answer from the script, which is precisely the failure running the script was
        /// meant to prevent. Both shapes are therefore accepted.
        /// </para>
        /// <para>
        /// An array's elements are used VERBATIM, one argument each: they have already
        /// been separated by the model, so re-splitting them on whitespace would break
        /// any argument containing a space. Qwen's XML parser preserves a parameter
        /// declared as a string even when it contains a JSON array. Decode that vector
        /// before considering shell-style splitting; otherwise brackets and commas
        /// become part of the command's arguments.
        /// </para>
        /// </summary>
        internal static IReadOnlyList<string>? ReadArgumentList(ToolCall call, string name)
        {
            if (call.Arguments == null || !call.Arguments.TryGetValue(name, out object? value) || value == null)
                return null;

            if (value is JsonElement { ValueKind: JsonValueKind.Array } array)
            {
                var items = new List<string>();
                foreach (JsonElement item in array.EnumerateArray())
                {
                    string? text = item.ValueKind switch
                    {
                        JsonValueKind.String => item.GetString(),
                        JsonValueKind.Null => null,
                        _ => item.GetRawText(),
                    };
                    if (text != null)
                        items.Add(text);
                }
                return items;
            }

            if (value is System.Collections.IEnumerable enumerable and not string)
            {
                var items = new List<string>();
                foreach (object? item in enumerable)
                {
                    string? text = item switch
                    {
                        null => null,
                        JsonElement { ValueKind: JsonValueKind.Null } => null,
                        JsonElement { ValueKind: JsonValueKind.String } e => e.GetString(),
                        JsonElement e => e.GetRawText(),
                        _ => Convert.ToString(item, CultureInfo.InvariantCulture),
                    };
                    if (text != null)
                        items.Add(text);
                }
                return items;
            }

            string? single = ReadString(call, name);
            if (string.IsNullOrWhiteSpace(single))
                return Array.Empty<string>();

            if (single.AsSpan().TrimStart().StartsWith("[", StringComparison.Ordinal))
            {
                const string guidance = "must be a valid JSON array of strings, or shell-style arguments "
                    + "with values containing spaces quoted. To pass literal text beginning with '[', quote that argument.";
                try
                {
                    using JsonDocument document = JsonDocument.Parse(single);
                    var items = new List<string>();
                    foreach (JsonElement item in document.RootElement.EnumerateArray())
                    {
                        if (item.ValueKind != JsonValueKind.String)
                            throw new FormatException($"'{name}' {guidance}");
                        items.Add(item.GetString()!);
                    }
                    return items;
                }
                catch (JsonException)
                {
                    throw new FormatException($"'{name}' {guidance}");
                }
            }

            return SkillScriptRunner.SplitArguments(single);
        }

        internal static long? ReadInt64(ToolCall call, string name)
        {
            if (call.Arguments == null || !call.Arguments.TryGetValue(name, out object? value) || value == null)
                return null;

            switch (value)
            {
                case long l: return l;
                case int i: return i;
                case double d: return (long)d;
                case JsonElement { ValueKind: JsonValueKind.Number } e when e.TryGetInt64(out long n): return n;
                case JsonElement { ValueKind: JsonValueKind.String } e
                    when long.TryParse(e.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out long p):
                    return p;
                case string s
                    when long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out long p):
                    return p;
                default:
                    return null;
            }
        }
    }

    /// <summary>The outcome of one skill tool call.</summary>
    /// <param name="Ok">False when the model was told what went wrong instead of being given content.</param>
    /// <param name="Content">The text to feed back as the tool result.</param>
    /// <param name="SkillId">Which skill was touched, for logging and for the UI trace. Null when none was.</param>
    /// <param name="ResourcePath">Which file was read or run. Null when none was.</param>
    public readonly record struct SkillToolResult(bool Ok, string Content, string? SkillId, string? ResourcePath)
    {
        /// <summary>
        /// Files the call produced and kept for the user, when it was a <c>shell</c> command.
        /// Carried structurally so a UI can show download links without depending on the
        /// model to repeat them. Empty for every other tool.
        /// </summary>
        public IReadOnlyList<SkillProducedFile> Files { get; init; } = Array.Empty<SkillProducedFile>();

        internal static SkillToolResult Success(string content, string? skillId = null, string? path = null) =>
            new(true, content, skillId, path);

        internal static SkillToolResult Failure(string message) =>
            new(false, "Error: " + message, null, null);
    }

    /// <summary>
    /// What one conversation's skill tools may reach.
    ///
    /// <para>
    /// Scope is per-request, not global: a request that selected <c>pdf</c> can read
    /// <c>pdf</c> and anything else the host chose to advertise, and nothing else on
    /// the machine. Handing the whole registry to every request would let a prompt
    /// injected into one skill's <c>SKILL.md</c> pull the contents of another the
    /// user never enabled.
    /// </para>
    /// </summary>
    public sealed class SkillToolContext
    {
        /// <summary>Create a context over an explicit set of skills.</summary>
        public SkillToolContext(IReadOnlyList<Skill> reachable, int maxReadBytes = SkillTools.DefaultMaxReadBytes)
        {
            ArgumentNullException.ThrowIfNull(reachable);
            Reachable = reachable;
            MaxReadBytes = maxReadBytes > 0 ? maxReadBytes : SkillTools.DefaultMaxReadBytes;
            _byId = new Dictionary<string, Skill>(reachable.Count, StringComparer.OrdinalIgnoreCase);
            foreach (Skill skill in reachable)
                _byId[skill.Id] = skill;
        }

        private readonly Dictionary<string, Skill> _byId;

        /// <summary>Every skill this conversation may read from, sorted by id.</summary>
        public IReadOnlyList<Skill> Reachable { get; }

        /// <summary>Ceiling on one read's returned bytes.</summary>
        public int MaxReadBytes { get; }

        /// <summary>
        /// Set to enable <see cref="SkillTools.RunToolName"/>. Null — the default —
        /// means the tool is not offered and, if a model calls it anyway, it is told the
        /// host does not run scripts.
        /// </summary>
        public ISkillScriptRunner? ScriptRunner { get; init; }

        /// <summary>
        /// Runs code the model wrote, when the host allows it. Null when it does not, and
        /// the model is told so rather than left waiting.
        /// </summary>
        public ICodeRunner? CodeRunner { get; init; }

        /// <summary>
        /// Files the user attached to this conversation, staged into every
        /// working directory so a command can open them by
        /// name instead of re-typing their content into the source.
        /// </summary>
        public IReadOnlyList<CodeInputFile> CodeInputFiles { get; init; } = Array.Empty<CodeInputFile>();

        /// <summary>
        /// The session's persistent workspace, shared by <c>shell</c> and skill
        /// scripts so a pipeline's steps see each other's files. Null on hosts that keep
        /// the original call-scoped scratch.
        /// </summary>
        public SessionWorkspace? Workspace { get; init; }

        /// <summary>
        /// The calling agent's handle on the turn's sub-agents, when the host enabled them.
        /// Null — the default — means the agent tools are not offered and, if a model calls
        /// one anyway, it is told this host does not run sub-agents.
        /// </summary>
        public SubAgentScope? Agents { get; init; }

        /// <summary>This context, answering agent tools for <paramref name="agents"/>.</summary>
        public SkillToolContext WithAgents(SubAgentScope? agents) => new(Reachable, MaxReadBytes)
        {
            ScriptRunner = ScriptRunner,
            CodeRunner = CodeRunner,
            CodeInputFiles = CodeInputFiles,
            Workspace = Workspace,
            Agents = agents,
        };

        /// <summary>
        /// The context a sub-agent runs its tools in: the same skills, runner and
        /// attachments as its parent, answering agent tools as <paramref name="agents"/>,
        /// in its own LANE of the parent's workspace — the same files and packages, but its
        /// own shell working directory, exports and read ledger, so one agent's <c>cd</c>
        /// cannot move another's next command.
        /// </summary>
        /// <param name="agents">The sub-agent's handle.</param>
        /// <param name="laneId">
        /// Unique across the whole CONVERSATION, not just the turn: a workspace outlives
        /// the turn, and a later turn's <c>agent_1</c> must not resume an earlier one's cwd.
        /// </param>
        internal SkillToolContext ForSubAgent(SubAgentScope agents, string laneId) => new(Reachable, MaxReadBytes)
        {
            // Skill scripts keep the runner the parent's plan built. They run from the
            // shared work directory, never from a shell's cwd, so the lane changes nothing
            // about where they run; only their read records land in the owner's ledger.
            ScriptRunner = ScriptRunner,
            CodeRunner = CodeRunner,
            CodeInputFiles = CodeInputFiles,
            Workspace = Workspace?.ForAgent(laneId),
            Agents = agents,
        };

        /// <summary>Look up a skill by the name the model used, with a message it can act on when there is no match.</summary>
        public bool TryResolve(string name, out Skill? skill, out string? error)
        {
            string key = (name ?? string.Empty).Trim();
            if (_byId.TryGetValue(key, out skill))
            {
                error = null;
                return true;
            }

            skill = null;
            error = Reachable.Count == 0
                ? $"No skill called '{key}' is available; this conversation has no skills."
                : $"No skill called '{key}' is available. Available skills: "
                  + string.Join(", ", Reachable.Select(s => s.Id)) + ".";
            return false;
        }
    }

    /// <summary>
    /// Runs a skill's bundled script. Implemented by hosts that opt into script
    /// execution; the core never provides a default, so the safe behaviour is what
    /// happens when nobody wires one up.
    /// </summary>
    public interface ISkillScriptRunner
    {
        /// <summary>
        /// Run <paramref name="relativePath"/> inside <paramref name="skill"/>.
        /// Implementations must resolve the path through
        /// <see cref="SkillPathGuard"/>, bound the runtime and the captured output, and
        /// return a failure result rather than throwing.
        /// </summary>
        /// <param name="onOutput">
        /// Live tap: called per stdout/stderr line while the script runs, so a host can
        /// stream progress. May be called from other threads; null taps nothing.
        /// </param>
        /// <param name="packages">
        /// Dependencies to set up in the session environment before the script runs, on
        /// hosts that can install. Beyond these, an implementation may auto-install
        /// whatever import the script then still fails on.
        /// </param>
        SkillToolResult Run(
            Skill skill, string relativePath, IReadOnlyList<string> arguments,
            Action<string>? onOutput = null, IReadOnlyList<string>? packages = null);
    }
}
