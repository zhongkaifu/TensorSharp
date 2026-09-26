// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using System.Linq;

namespace TensorSharp.Server.ResponseSerializers
{
    /// <summary>
    /// Anonymous-typed payload builders for the Web UI's chat SSE protocol.
    /// The shapes are deliberately minimal so the JS UI can keep using
    /// <c>JSON.parse</c> + <c>typeof</c> checks instead of formal schemas.
    /// </summary>
    internal static class WebUiSseEvents
    {
        public static object QueueProgress(int position, int pending) => new
        {
            queue_position = position,
            queue_pending = pending,
        };

        public static object Token(string token) => new { token };

        public static object Thinking(string thinking) => new { thinking };

        /// <summary>Replace the entire assistant message body with <paramref name="text"/>. Used by the
        /// DiffusionGemma live denoising preview, where each step refines the whole canvas (not a
        /// left-to-right append). <paramref name="step"/>/<paramref name="totalSteps"/> drive a progress
        /// indicator; <paramref name="preview"/> marks intermediate (still-denoising) frames.</summary>
        public static object Replace(string text, int step, int totalSteps, bool preview) => new
        {
            replace = text,
            diffusionStep = step,
            diffusionTotal = totalSteps,
            preview,
        };

        public static object ToolCalls(IReadOnlyList<ToolCall> toolCalls) => new
        {
            tool_calls = toolCalls.Select(tc => (object)new
            {
                name = tc.Name,
                arguments = tc.Arguments,
            }).ToList(),
        };

        /// <summary>
        /// One Agent Skills lookup the server performed on the model's behalf.
        ///
        /// <para>
        /// The browser discriminates frames purely by WHICH KEY IS PRESENT, so
        /// <c>skill_step</c> must not collide with any other frame's discriminator. It
        /// exists because the progressive-disclosure loop deliberately withholds an
        /// intermediate round's tokens (they carry the tool-call markup); without these
        /// frames the user watches a blank composer for however long the model spends
        /// reading files.
        /// </para>
        /// </summary>
        public static object SkillStep(SkillToolInvocation invocation) => new
        {
            skill_step = invocation.Tool,
            agent_id = invocation.AgentId,
            skill = invocation.SkillId,
            detail = invocation.ResourcePath,
            ok = invocation.Ok,
            round = invocation.Round,
            // Files a shell command produced, so the UI can render download links
            // itself. The model is separately told to repeat the links in its answer,
            // but a small model does that erratically — the user's download must not
            // depend on it.
            files = invocation.Files.Count == 0 ? null : invocation.Files.Select(f => (object)new
            {
                name = f.Name,
                bytes = f.Bytes,
                url = f.Url,
            }).ToList(),
        };

        /// <summary>
        /// The one routed deliverable that passed host-side structural/content checks.
        /// This is separate from <c>skill_step</c>: provisional files remain hidden while
        /// preserving the UI's established step/finished frame adjacency.
        /// </summary>
        public static object VerifiedArtifact(SkillProducedFile artifact) => new
        {
            artifact_verified = true,
            files = new[]
            {
                new { name = artifact.Name, bytes = artifact.Bytes, url = artifact.Url },
            },
        };

        /// <summary>
        /// Live progress through an in-process tool call's two silent stretches:
        /// <c>writing</c> while the model generates the call (with the new body text),
        /// <c>running</c> while the host executes it (with elapsed seconds, one frame a
        /// second), <c>finished</c> when execution returned. Without these the user
        /// watches a frozen page for the length of a program being written plus a pip
        /// install.
        /// </summary>
        public static object ToolProgress(string phase, string tool, string text, double seconds, string detail,
            IReadOnlyList<MultiAgentProgress> agents = null) => new
        {
            tool_progress = phase,
            // The tool that runs, by its declared name: an alias the host accepts
            // (str_replace, apply-patch, ...) is reported as the tool it dispatches to, so
            // the pages label it like the real name instead of "Running str replace".
            tool = SkillToolNames.CanonicalName(tool),
            text,
            seconds,
            // What is being run, in one line (SkillChatLoop.DescribeCall): a shell call's
            // command ("pip install pandas"), a skills_run script and its arguments
            // ("scripts/extract.py 2400"), a skills_read file ("pdf/SKILL.md"), or an
            // apply_patch envelope's size ("812 chars").
            detail,
            // Non-observing snapshots refresh the expanded subagent panel during a wait.
            agents = agents?.Select(a => new
            {
                agent_id = a.AgentId,
                parent_id = a.ParentId,
                task = a.Task,
                agent_type = a.AgentType,
                permissions = a.Permissions,
                workspace_id = a.WorkspaceId,
                depends_on = a.DependsOn,
                status = a.Status,
                tool = SkillToolNames.CanonicalName(a.Tool),
                tool_status = a.ToolStatus,
                detail = a.Detail,
                result = a.Result,
                error = a.Error,
            }).ToArray(),
        };

        public static object Done(
            int tokenCount,
            double elapsedSeconds,
            double tokPerSec,
            bool aborted,
            string error,
            string sessionId,
            int promptTokens,
            int kvCacheReusedTokens,
            bool truncated = false) => new
        {
            done = true,
            tokenCount,
            elapsed = elapsedSeconds,
            tokPerSec,
            aborted,
            // The answer stops mid-thought because the max-tokens budget ran out,
            // not because the model was finished. Distinct from `aborted`, which
            // means the user (or a dropped connection) stopped it.
            truncated,
            error,
            sessionId,
            promptTokens,
            kvReusedTokens = kvCacheReusedTokens,
            kvReusePercent = promptTokens > 0 ? 100.0 * kvCacheReusedTokens / promptTokens : 0.0,
        };
    }
}
