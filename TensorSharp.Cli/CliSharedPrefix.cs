// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.Cli;

/// <summary>Find the rendered system/tool tokens that are safe to share between conversations.</summary>
internal static class CliSharedPrefix
{
    internal static List<int> Compute(string systemPrompt, bool hasTools,
        Func<IReadOnlyList<ChatMessage>, bool, List<int>> render, int minimum = 1)
    {
        if (string.IsNullOrEmpty(systemPrompt) && !hasTools)
            return new List<int>();

        var renders = new List<IReadOnlyList<int>>();
        foreach (string probe in new[] { "Hello", " indented", "\nnewline", "```code", "[Attached file: a.txt]" })
        {
            var messages = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(systemPrompt))
                messages.Add(new ChatMessage { Role = "system", Content = systemPrompt });
            messages.Add(new ChatMessage { Role = "user", Content = probe });
            renders.Add(render(messages, true));
        }

        if (!string.IsNullOrEmpty(systemPrompt))
        {
            try
            {
                var systemOnly = render(new[] { new ChatMessage { Role = "system", Content = systemPrompt } }, false);
                if (systemOnly is { Count: > 0 })
                    renders.Add(systemOnly);
            }
            catch (Exception)
            {
                // Some templates require a user turn. The divergent probes still
                // bound the prefix when a system-only render is unavailable.
            }
        }

        int count = Length(renders, minimum);
        return count == 0 ? new List<int>() : new List<int>(renders[0]).GetRange(0, count);
    }

    internal static int MatchingLength(IReadOnlyList<int> prefix, IReadOnlyList<int> prompt)
    {
        if (prefix == null || prompt.Count < prefix.Count)
            return 0;
        for (int i = 0; i < prefix.Count; i++)
            if (prefix[i] != prompt[i])
                return 0;
        return prefix.Count;
    }

    internal static int Length(IReadOnlyList<IReadOnlyList<int>> renders, int minimum)
    {
        if (renders == null || renders.Count == 0)
            return 0;
        int common = int.MaxValue;
        foreach (var render in renders)
        {
            if (render == null || render.Count == 0)
                return 0;
            common = Math.Min(common, render.Count);
        }
        for (int i = 0; i < common; i++)
        {
            for (int r = 1; r < renders.Count; r++)
                if (renders[r][i] != renders[0][i])
                {
                    common = i;
                    break;
                }
        }
        // Leave a token of margin at the boundary where tokenization may merge.
        int keep = Math.Max(0, common - 1);
        return keep < minimum ? 0 : keep;
    }
}
