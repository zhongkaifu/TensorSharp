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
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>One generation a scripted agent was asked for, captured at the moment it was asked.</summary>
internal sealed class SubAgentGeneration
{
    /// <summary>1-based: the how-manyth generation of this agent.</summary>
    public required int Index { get; init; }

    /// <summary>The prompt's messages (the same objects the loop holds, in a list of our own).</summary>
    public required List<ChatMessage> Messages { get; init; }

    /// <summary>Each message's content AS IT WAS when the generation was asked for: loops fold later deliveries into earlier results.</summary>
    public required List<string> Contents { get; init; }

    public required List<string> Roles { get; init; }

    public List<ToolFunction>? Tools { get; init; }

    public CancellationToken Token { get; init; }

    public string LastContent => Contents.Count == 0 ? string.Empty : Contents[^1];

    public string LastRole => Roles.Count == 0 ? string.Empty : Roles[^1];

    /// <summary>Every message's content, for "appears exactly once" checks.</summary>
    public string AllText => string.Join("\n␞\n", Contents);
}

/// <summary>What a scripted agent does for one generation.</summary>
internal delegate Task<SkillTurnOutput> SubAgentStep(SubAgentGeneration generation);

/// <summary>
/// A deterministic stand-in for a model: replays its steps in order (repeating the last
/// one once they run out), records every prompt it was given, and exposes signals a test
/// can await instead of sleeping — "generation N has started" and "a generation's
/// cancellation token fired".
/// </summary>
internal sealed class ScriptedAgent
{
    private readonly object _lock = new();
    private readonly List<SubAgentStep> _steps;
    private readonly List<SubAgentGeneration> _calls = new();
    private readonly Dictionary<int, TaskCompletionSource> _entered = new();
    private readonly TaskCompletionSource _cancelled = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public ScriptedAgent(params SubAgentStep[] steps) => _steps = new List<SubAgentStep>(steps);

    public IReadOnlyList<SubAgentGeneration> Calls
    {
        get { lock (_lock) return _calls.ToList(); }
    }

    public int CallCount
    {
        get { lock (_lock) return _calls.Count; }
    }

    public SubAgentGeneration Call(int index)
    {
        lock (_lock) return _calls[index - 1];
    }

    /// <summary>Completes when the token of any generation this agent was asked for is cancelled.</summary>
    public Task Cancelled => _cancelled.Task;

    /// <summary>Completes once generation <paramref name="n"/> (1-based) has started.</summary>
    public Task Entered(int n)
    {
        lock (_lock)
        {
            if (!_entered.TryGetValue(n, out TaskCompletionSource? tcs))
            {
                tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                _entered[n] = tcs;
            }
            if (_calls.Count >= n)
                tcs.TrySetResult();
            return tcs.Task;
        }
    }

    public Task<SkillTurnOutput> Generate(
        List<ChatMessage> messages, List<ToolFunction>? tools, CancellationToken cancellationToken)
    {
        SubAgentGeneration generation;
        SubAgentStep step;
        TaskCompletionSource? entered;
        lock (_lock)
        {
            generation = new SubAgentGeneration
            {
                Index = _calls.Count + 1,
                Messages = new List<ChatMessage>(messages),
                Contents = messages.Select(m => m.Content ?? string.Empty).ToList(),
                Roles = messages.Select(m => m.Role ?? string.Empty).ToList(),
                Tools = tools,
                Token = cancellationToken,
            };
            _calls.Add(generation);
            step = _steps.Count == 0
                ? Steps.Answer("(no script)")
                : _steps[Math.Min(generation.Index - 1, _steps.Count - 1)];
            _entered.TryGetValue(generation.Index, out entered);
        }

        cancellationToken.Register(() => _cancelled.TrySetResult());
        entered?.TrySetResult();
        try
        {
            return step(generation);
        }
        catch (Exception ex)
        {
            return Task.FromException<SkillTurnOutput>(ex);
        }
    }
}

/// <summary>Building blocks for <see cref="ScriptedAgent"/> scripts.</summary>
internal static class Steps
{
    private static int s_callIds;

    /// <summary>A gate a test releases; continuations never run inline on the releasing thread.</summary>
    public static TaskCompletionSource Gate() => new(TaskCreationOptions.RunContinuationsAsynchronously);

    public static SkillTurnOutput Turn(string content, params ToolCall[] calls) =>
        new(new ParsedOutput { Content = content, ToolCalls = calls.Length == 0 ? null : calls.ToList() });

    public static ToolCall Tool(string name, params (string Key, object? Value)[] args) => new()
    {
        Id = "call_" + Interlocked.Increment(ref s_callIds).ToString(System.Globalization.CultureInfo.InvariantCulture),
        Name = name,
        Arguments = args.ToDictionary(a => a.Key, a => a.Value),
    };

    /// <summary>Answer with <paramref name="text"/> and no tool calls.</summary>
    public static SubAgentStep Answer(string text) => _ => Task.FromResult(Turn(text));

    /// <summary>Make one tool call (a fresh <see cref="ToolCall"/> each time the step runs).</summary>
    public static SubAgentStep Call(string name, params (string Key, object? Value)[] args) =>
        _ => Task.FromResult(Turn(string.Empty, Tool(name, args)));

    /// <summary>Make several tool calls in one turn.</summary>
    public static SubAgentStep Calls(params Func<ToolCall>[] calls) =>
        _ => Task.FromResult(Turn(string.Empty, calls.Select(c => c()).ToArray()));

    /// <summary>Wait for <paramref name="gate"/> (or the generation's cancellation), then do <paramref name="then"/>.</summary>
    public static SubAgentStep After(Task gate, SubAgentStep then) => async generation =>
    {
        await gate.WaitAsync(generation.Token).ConfigureAwait(false);
        return await then(generation).ConfigureAwait(false);
    };

    /// <summary>Run <paramref name="before"/> (which may wait on anything), then do <paramref name="then"/>.</summary>
    public static SubAgentStep Do(Func<SubAgentGeneration, Task> before, SubAgentStep then) => async generation =>
    {
        await before(generation).ConfigureAwait(false);
        return await then(generation).ConfigureAwait(false);
    };

    /// <summary>Generate until cancelled.</summary>
    public static SubAgentStep Block() => async generation =>
    {
        await Task.Delay(Timeout.Infinite, generation.Token).ConfigureAwait(false);
        throw new UnreachableException();
    };

    /// <summary>Fail the generation, as a backend error would.</summary>
    public static SubAgentStep Throw(string message) =>
        _ => Task.FromException<SkillTurnOutput>(new InvalidOperationException(message));
}

/// <summary>
/// A <see cref="SubAgentRuntime"/> wired to scripted agents: the parent's conversation
/// and tools are bound to <see cref="SubAgentRuntime.Root"/>, and every spawned agent
/// generates through the script registered for its id (or a default one-line answer).
/// </summary>
internal sealed class SubAgentHarness : IDisposable
{
    public const string SystemPrompt = "You are a careful assistant. Use your tools.";
    public const string DeveloperPrompt = "House rules: be brief.";
    public const string UserRequest = "Please look into the build.";

    private readonly object _lock = new();
    private readonly Dictionary<string, ScriptedAgent> _scripts = new(StringComparer.Ordinal);
    private readonly List<SubAgentLaunch> _launches = new();

    public SubAgentHarness(
        int maxThreads = SubAgentOptions.DefaultMaxThreads,
        int maxDepth = SubAgentOptions.DefaultMaxDepth,
        SkillAgentLoopOptions? loopOptions = null,
        SkillToolContext? context = null,
        CancellationToken turnToken = default,
        List<ChatMessage>? conversation = null)
    {
        Conversation = conversation ?? new List<ChatMessage>
        {
            new() { Role = "system", Content = SystemPrompt },
            new() { Role = "developer", Content = DeveloperPrompt },
            new() { Role = "user", Content = UserRequest },
        };
        Tools = new List<ToolFunction>
        {
            new()
            {
                Name = "lookup",
                Description = "Look something up.",
                Parameters = new Dictionary<string, ToolParameter> { ["query"] = new() { Type = "string", Description = "What." } },
                Required = new List<string> { "query" },
            },
        };
        Tools.AddRange(SubAgentTools.Declare());

        var options = new SubAgentOptions { Enabled = true, MaxThreads = maxThreads, MaxDepth = maxDepth };
        Runtime = new SubAgentRuntime(
            options,
            new SubAgentHostBinding { CreateGenerator = CreateGenerator, LoopOptions = loopOptions },
            context ?? new SkillToolContext(Array.Empty<Skill>()),
            logger: null,
            turnToken);
        Runtime.Root.Bind(Conversation, Tools);
    }

    public SubAgentRuntime Runtime { get; }

    public List<ChatMessage> Conversation { get; }

    public List<ToolFunction> Tools { get; }

    public IReadOnlyList<SubAgentLaunch> Launches
    {
        get { lock (_lock) return _launches.ToList(); }
    }

    /// <summary>Script the agent that will get <paramref name="agentId"/>. Must be called before it is spawned.</summary>
    public ScriptedAgent Script(string agentId, params SubAgentStep[] steps)
    {
        var agent = new ScriptedAgent(steps);
        lock (_lock)
            _scripts[agentId] = agent;
        return agent;
    }

    public ScriptedAgent Agent(string agentId)
    {
        lock (_lock)
            return _scripts[agentId];
    }

    private SkillTurnGenerator CreateGenerator(SubAgentLaunch launch)
    {
        ScriptedAgent? agent;
        lock (_lock)
        {
            _launches.Add(launch);
            if (!_scripts.TryGetValue(launch.AgentId, out agent))
            {
                agent = new ScriptedAgent(Steps.Answer("default answer from " + launch.AgentId));
                _scripts[launch.AgentId] = agent;
            }
        }
        return agent.Generate;
    }

    /// <summary>Call an agent tool as the top-level agent.</summary>
    public Task<SkillToolResult> Root(string tool, params (string Key, object? Value)[] args) =>
        Runtime.Root.ExecuteAsync(Steps.Tool(tool, args), null, CancellationToken.None);

    public Task<SkillToolResult> Root(ToolCall call, Action<string>? onOutput = null, CancellationToken cancellationToken = default) =>
        Runtime.Root.ExecuteAsync(call, onOutput, cancellationToken);

    public Task<SkillToolResult> Spawn(string task) =>
        Root(SkillToolNames.SpawnAgent, ("message", task));

    public SubAgentSnapshot Snapshot(string id) => Runtime.Snapshot().Single(s => s.Id == id);

    /// <summary>
    /// Wait for an agent to reach <paramref name="status"/>. A condition poll with a
    /// generous ceiling, never a fixed sleep: the runtime sets a status under its lock
    /// right after the generator returns, and nothing else announces it.
    /// </summary>
    public async Task WaitForStatusAsync(string id, SubAgentStatus status, int timeoutSeconds = 10)
    {
        var stopwatch = Stopwatch.StartNew();
        while (true)
        {
            SubAgentSnapshot? snapshot = Runtime.Snapshot().FirstOrDefault(s => s.Id == id);
            if (snapshot?.Status == status)
                return;
            if (stopwatch.Elapsed > TimeSpan.FromSeconds(timeoutSeconds))
            {
                throw new TimeoutException(
                    $"{id} never became {status}; it is {snapshot?.Status.ToString() ?? "unknown"}. Agents: "
                    + string.Join("; ", Runtime.Snapshot().Select(s => s.Id + "=" + s.Status)));
            }
            await Task.Delay(2).ConfigureAwait(false);
        }
    }

    public void Dispose() => Runtime.Dispose();
}

internal static class SubAgentTaskExtensions
{
    /// <summary>Await with a ceiling, so a regression fails the test instead of hanging the run.</summary>
    public static async Task<T> Within<T>(this Task<T> task, int seconds = 10)
    {
        Task finished = await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(seconds))).ConfigureAwait(false);
        if (finished != task)
            throw new TimeoutException($"did not complete within {seconds} s");
        return await task.ConfigureAwait(false);
    }

    public static async Task Within(this Task task, int seconds = 10)
    {
        Task finished = await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(seconds))).ConfigureAwait(false);
        if (finished != task)
            throw new TimeoutException($"did not complete within {seconds} s");
        await task.ConfigureAwait(false);
    }

    /// <summary>How many times <paramref name="needle"/> occurs in <paramref name="text"/>.</summary>
    public static int Occurrences(this string text, string needle)
    {
        int count = 0;
        for (int i = text.IndexOf(needle, StringComparison.Ordinal); i >= 0;
             i = text.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
        {
            count++;
        }
        return count;
    }
}
