
namespace InferenceWeb.Tests;

public class WebUiChatPolicyTests
{
    [Fact]
    public void TryValidateChatRequest_AllowsPlainChatRequests()
    {
        bool allowed = WebUiChatPolicy.TryValidateChatRequest(null, null, out string? error);

        Assert.True(allowed);
        Assert.Null(error);
    }

    [Fact]
    public void TryValidateChatRequest_RejectsPerTurnModelSelection()
    {
        bool allowed = WebUiChatPolicy.TryValidateChatRequest("Qwen3-4B-Q8_0.gguf", null, out string? error);

        Assert.False(allowed);
        Assert.Equal(WebUiChatPolicy.ModelSelectionLockedMessage, error);
    }

    [Fact]
    public void TryValidateChatRequest_RejectsPerTurnBackendSelection()
    {
        bool allowed = WebUiChatPolicy.TryValidateChatRequest(null, "ggml_cuda", out string? error);

        Assert.False(allowed);
        Assert.Equal(WebUiChatPolicy.ModelSelectionLockedMessage, error);
    }
}

