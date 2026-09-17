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
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using TensorSharp.Cli;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Preparing the prompt every conversation shares, before anyone asks for it.
///
/// <para>
/// Measured on the shipped Qwen agent configuration: a new chat answered in 0.65 s while
/// the FIRST message of a process took 21.81 s, because the shared prefix is 6,459 tokens
/// of skills catalog and tool declarations and whoever crosses that boundary first pays
/// for all of it. These pin the two things that make moving that cost to startup safe:
/// the warm-up must render the prompt a real request renders, and the CLI must forward
/// only tokens that are genuinely common to every possible first turn.
/// </para>
/// </summary>
public class PrefixCacheStartupTests
{
    // ---- the server's warm-up request ------------------------------------

    /// <summary>
    /// Every field here is load-bearing and three of them are traps. A body that looks
    /// more realistic - an empty skills array, an empty tools array - renders a DIFFERENT
    /// and shorter system block, and because reuse is a longest-common-prefix match a
    /// block that differs by one token makes every token after it unshareable.
    /// </summary>
    [Fact]
    public void TheWarmUpRequestIsTheOneARealTurnSends()
    {
        JsonElement body = PrefixCacheWarmup.BuildRequest("warm-1", think: false);

        // A session, because the default session has no workspace and is therefore
        // offered a different set of tools.
        Assert.Equal("warm-1", body.GetProperty("sessionId").GetString());

        // One token: nothing wants the answer, only the K/V the prompt leaves behind.
        Assert.Equal(1, body.GetProperty("maxTokens").GetInt32());

        // Short, because the first real message has to be able to rewind past this
        // turn's framing and continue rather than re-prefill.
        Assert.Equal("hi", body.GetProperty("messages")[0].GetProperty("content").GetString());

        // NOT sent as empty arrays: an empty "skills" suppresses both the operator's
        // selection and the discovery catalog, and a "tools" array disables the host
        // skill router.
        Assert.False(body.TryGetProperty("skills", out _));
        Assert.False(body.TryGetProperty("tools", out _));

        // True would reset the session and release its workspace - deleting a
        // conversation's files in order to warm a cache.
        Assert.False(body.GetProperty("newChat").GetBoolean());
    }

    /// <summary>
    /// Thinking is part of the prefix, not a detail of the reply. Measured on Qwen 3.8:
    /// thinking off shares 6,459 tokens and thinking on shares 6,497, which are two
    /// different checkpoints - so the flag has to reach the request verbatim.
    /// </summary>
    [Fact]
    public void TheThinkingFlagReachesTheRequest()
    {
        Assert.False(PrefixCacheWarmup.BuildRequest("s", think: false).GetProperty("think").GetBoolean());
        Assert.True(PrefixCacheWarmup.BuildRequest("s", think: true).GetProperty("think").GetBoolean());
    }

    /// <summary>A session-less warm-up warms a prompt no real request sends, so it is refused.</summary>
    [Fact]
    public void AWarmUpWithoutASessionIsRefused()
    {
        Assert.Throws<ArgumentException>(() => PrefixCacheWarmup.BuildRequest("", think: false));
        Assert.Throws<ArgumentException>(() => PrefixCacheWarmup.BuildRequest(null, think: false));
    }

    /// <summary>
    /// The failure that does not throw. The chat service catches what it can and ends the
    /// stream with a done frame carrying an error, so a wrapper watching only for
    /// exceptions calls a warm-up that forwarded nothing "warm" - and the operator then
    /// believes the first message is fast when it is not.
    /// </summary>
    [Fact]
    public async Task AnErrorFrameIsNotAWarmCache()
    {
        PrefixCacheWarmup.Result result = await PrefixCacheWarmup.RunAsync(
            (_, _) => Frames(new { done = true, error = "No model loaded" }),
            "s", think: false, logger: null);

        Assert.False(result.Warmed);
        Assert.Contains("No model loaded", result.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AStreamThatEndsCleanlyIsAWarmCache()
    {
        PrefixCacheWarmup.Result result = await PrefixCacheWarmup.RunAsync(
            (_, _) => Frames(new { token = "o" }, new { done = true, error = (string)null }),
            "s", think: false, logger: null);

        Assert.True(result.Warmed, result.Detail);
    }

    [Fact]
    public async Task AnEmptyOrUnfinishedStreamDoesNotReportWarmupCompleted()
    {
        foreach (object[] frames in new[] { Array.Empty<object>(), new object[] { new { token = "partial" } } })
        {
            PrefixCacheWarmup.Result result = await PrefixCacheWarmup.RunAsync(
                (_, _) => Frames(frames), "s", think: false, logger: null);
            Assert.False(result.Warmed);
            Assert.Contains("without completing", result.Detail);
        }
    }

    [Fact]
    public async Task AnAbortedTerminalFrameDoesNotReportWarmupCompleted()
    {
        PrefixCacheWarmup.Result result = await PrefixCacheWarmup.RunAsync(
            (_, _) => Frames(new { done = true, aborted = true }), "s", think: false, logger: null);
        Assert.False(result.Warmed);
        Assert.Contains("aborted", result.Detail);
    }

    /// <summary>
    /// Several refusals throw before a single frame is yielded - no model loaded, an
    /// unknown session, a rejected backend. A startup latency optimisation must not be
    /// able to stop a server from serving.
    /// </summary>
    [Fact]
    public async Task AThrowingChatStreamIsReportedRatherThanPropagated()
    {
        PrefixCacheWarmup.Result result = await PrefixCacheWarmup.RunAsync(
            (_, _) => Throwing(), "s", think: false, logger: null);

        Assert.False(result.Warmed);
        Assert.Contains("boom", result.Detail, StringComparison.Ordinal);
    }

    private static async IAsyncEnumerable<object> Frames(params object[] frames)
    {
        foreach (object frame in frames)
        {
            await Task.Yield();
            yield return frame;
        }
    }

    private static async IAsyncEnumerable<object> Throwing()
    {
        await Task.Yield();
        throw new InvalidOperationException("boom");
#pragma warning disable CS0162
        yield break;
#pragma warning restore CS0162
    }

    // ---- where checkpoints are kept --------------------------------------

    /// <summary>
    /// Two models must never share a checkpoint directory. The store evicts beyond two
    /// files by last-access time across everything in the directory it was handed, with no
    /// notion of which model wrote one - so a shared directory turns a two-file-per-model
    /// budget into a two-file GLOBAL budget, and alternating launches of two configs from
    /// one install delete each other's several-hundred-megabyte checkpoints forever.
    /// </summary>
    [Fact]
    public void EachModelGetsItsOwnCheckpointDirectory()
    {
        string a = ServerOptionsBuilder.ResolvePrefixCacheDirectory("/srv/app", "/models/qwen-27b.gguf");
        string b = ServerOptionsBuilder.ResolvePrefixCacheDirectory("/srv/app", "/models/gemma-12b.gguf");

        Assert.NotEqual(a, b);
        Assert.StartsWith(Path.Combine("/srv/app", "prefix-cache"), a, StringComparison.Ordinal);
        Assert.StartsWith(Path.Combine("/srv/app", "prefix-cache"), b, StringComparison.Ordinal);
    }

    /// <summary>
    /// The readable half of the directory name is the file name, so an operator clearing
    /// one model's cache by hand can tell which is which.
    /// </summary>
    [Fact]
    public void TheCheckpointDirectoryNamesTheModel()
    {
        string key = ServerOptionsBuilder.ModelCacheKey("/models/Qwen3.8-27B-UD-Q4_K_XL.gguf");
        Assert.Contains("Qwen3.8-27B", key, StringComparison.Ordinal);
    }

    /// <summary>
    /// Two builds of one model can share a file name in different directories, so the
    /// full path decides, not the leaf.
    /// </summary>
    [Fact]
    public void SameFileNameInDifferentDirectoriesDoesNotCollide()
    {
        Assert.NotEqual(
            ServerOptionsBuilder.ModelCacheKey("/models/a/weights.gguf"),
            ServerOptionsBuilder.ModelCacheKey("/models/b/weights.gguf"));
    }

    /// <summary>The environment variable moves the root, and the per-model split survives it.</summary>
    [Fact]
    public void TheEnvironmentVariableMovesTheRootButKeepsThePerModelSplit()
    {
        const string name = "TENSORSHARP_PREFIX_CACHE_DIR";
        string previous = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, "/cache/prefix");
            string a = ServerOptionsBuilder.ResolvePrefixCacheDirectory("/srv/app", "/models/one.gguf");
            string b = ServerOptionsBuilder.ResolvePrefixCacheDirectory("/srv/app", "/models/two.gguf");

            Assert.StartsWith("/cache/prefix", a, StringComparison.Ordinal);
            Assert.NotEqual(a, b);
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, previous);
        }
    }

    /// <summary>A host with no startup model has nothing to name a directory after.</summary>
    [Fact]
    public void NoStartupModelStillResolvesToTheRoot()
    {
        Assert.Equal(
            Path.Combine("/srv/app", "prefix-cache"),
            ServerOptionsBuilder.ResolvePrefixCacheDirectory("/srv/app", null));
    }

    // ---- the CLI's warm prefix -------------------------------------------

    /// <summary>
    /// The whole safety argument of the CLI warm prefix. The system block is NOT a token
    /// prefix of a first turn by inspection: the tokenizer sees the rendered template and
    /// the user's text as one string, so a merge can span the boundary and change a token
    /// BEFORE it. Only tokens that every render agrees on may be forwarded.
    /// </summary>
    [Fact]
    public void OnlyTokensEveryRenderAgreesOnAreWarmed()
    {
        var renders = new IReadOnlyList<int>[]
        {
            new[] { 1, 2, 3, 4, 5, 100, 101 },
            new[] { 1, 2, 3, 4, 5, 200, 201 },
            new[] { 1, 2, 3, 4, 5, 300 },
        };

        // Five agree, less one for margin.
        Assert.Equal(4, InteractiveSession.WarmPrefixLength(renders, minimum: 1));
    }

    /// <summary>
    /// The case this defends against: a render whose LAST shared token differs because a
    /// merge reached backwards. If any render diverges earlier, everything from there on
    /// is unusable no matter how long the others agree.
    /// </summary>
    [Fact]
    public void OneDivergentRenderCutsThePrefixForEveryone()
    {
        var renders = new IReadOnlyList<int>[]
        {
            Enumerable.Range(0, 500).ToArray(),
            Enumerable.Range(0, 500).ToArray(),
            new[] { 0, 1, 9 }.Concat(Enumerable.Range(3, 400)).ToArray(),
        };

        Assert.Equal(1, InteractiveSession.WarmPrefixLength(renders, minimum: 1));
    }

    /// <summary>Below the floor there is nothing worth a startup pause, so nothing is warmed.</summary>
    [Fact]
    public void AShortAgreementIsNotWorthWarming()
    {
        var renders = new IReadOnlyList<int>[]
        {
            new[] { 1, 2, 3, 77 },
            new[] { 1, 2, 3, 88 },
        };

        Assert.Equal(0, InteractiveSession.WarmPrefixLength(renders, minimum: 64));
    }

    [Fact]
    public void NoRendersWarmNothing()
    {
        Assert.Equal(0, InteractiveSession.WarmPrefixLength(Array.Empty<IReadOnlyList<int>>(), minimum: 1));
        Assert.Equal(0, InteractiveSession.WarmPrefixLength(new IReadOnlyList<int>[] { Array.Empty<int>() }, minimum: 1));
    }

    /// <summary>
    /// Anti-vacuity: the helper must not simply return the first render's length. A single
    /// render still loses its last token to the margin.
    /// </summary>
    [Fact]
    public void TheMarginIsAlwaysTaken()
    {
        var one = new IReadOnlyList<int>[] { new[] { 1, 2, 3, 4, 5 } };
        Assert.Equal(4, InteractiveSession.WarmPrefixLength(one, minimum: 1));
    }
}
