using System.Text.Json;
using System.Runtime.CompilerServices;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryGlmStartupWarmupTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void KernelInitializationUsesTheSameShortUserTurnWithoutTheAgentCatalog(bool thinking)
    {
        JsonElement body = PrefixCacheWarmup.BuildKernelInitializationRequest("private-startup", thinking);
        Assert.Equal("private-startup", body.GetProperty("sessionId").GetString());
        Assert.Equal(thinking, body.GetProperty("think").GetBoolean());
        Assert.Equal(1, body.GetProperty("maxTokens").GetInt32());
        Assert.False(body.GetProperty("newChat").GetBoolean());
        Assert.Equal("hi", body.GetProperty("messages")[0].GetProperty("content").GetString());
        Assert.Equal("user", body.GetProperty("messages")[0].GetProperty("role").GetString());
        Assert.Equal(1, body.GetProperty("messages").GetArrayLength());
        Assert.Equal(0, body.GetProperty("tools").GetArrayLength());
        Assert.Equal(0, body.GetProperty("skills").GetArrayLength());
        Assert.False(body.GetProperty("skills_discovery").GetBoolean());
        Assert.False(body.GetProperty("multi_agent").GetBoolean());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DefaultPrefixWarmupKeepsItsRealChatCatalogContract(bool thinking)
    {
        JsonElement body = PrefixCacheWarmup.BuildRequest("prefix-startup", thinking);
        foreach (string key in new[] { "tools", "skills", "skills_discovery", "multi_agent" })
            Assert.False(body.TryGetProperty(key, out _));
        Assert.Equal(thinking, body.GetProperty("think").GetBoolean());
    }

    [Theory]
    [InlineData(false, "default-chat-prefix")]
    [InlineData(true, "kernel-initialization")]
    public void StartupStatusDistinguishesInitializationFromPrefixPreparation(bool kernelOnly, string scope)
    {
        var status = new StartupWarmupStatus([new(true, true, 10), new(false, true, 20)], true,
            null, false, null, kernelOnly);
        Assert.True(status.Completed);
        Assert.Equal(scope, status.Scope);
        Assert.Equal(30, status.ElapsedMilliseconds);
        Assert.False(status.JointCompleted);
    }

    [Fact]
    public async Task KernelRequestStillRunsThroughTheSuppliedChatStreamAndChecksCompletion()
    {
        JsonElement captured = default;
        async IAsyncEnumerable<object> Frames(JsonElement body, [EnumeratorCancellation] CancellationToken cancellation)
        {
            captured = body;
            await Task.CompletedTask;
            yield return new { done = true, error = "admission refused" };
        }
        var result = await PrefixCacheWarmup.RunKernelInitializationAsync(Frames, "startup", false, null!);
        Assert.False(result.Warmed);
        Assert.Contains("admission refused", result.Detail);
        Assert.False(captured.GetProperty("multi_agent").GetBoolean());
    }

    [Fact]
    public void ExistingPublicWarmupSignaturesRemainAvailableForCompiledCallers()
    {
        Assert.NotNull(typeof(PrefixCacheWarmup).GetMethod(nameof(PrefixCacheWarmup.BuildRequest),
            [typeof(string), typeof(bool), typeof(int)]));
        Assert.Equal(6, typeof(PrefixCacheWarmup).GetMethod(nameof(PrefixCacheWarmup.RunAsync))!.GetParameters().Length);
        Assert.Contains(typeof(StartupWarmupStatus).GetConstructors(), c => c.GetParameters().Length == 5);
    }
}
