
namespace InferenceWeb.Tests;

public class ModelServiceHistoryTests
{
    [Fact]
    public void HasMultimodalContent_HistoryDetectsEarlierImageTurn()
    {
        var history = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "Describe this image.", ImagePaths = new List<string> { "first.png" } },
            new ChatMessage { Role = "assistant", Content = "It is a banner." },
            new ChatMessage { Role = "user", Content = "What text is on it?" }
        };

        Assert.True(ModelService.HasMultimodalContent(history));
        Assert.False(ModelService.HasMultimodalContent(history[^1]));
    }

    [Fact]
    public void GetImagePathsInPromptOrder_PreservesHistoricalOrder()
    {
        var history = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "First batch", ImagePaths = new List<string> { "a.png", "b.png" } },
            new ChatMessage { Role = "assistant", Content = "Two images received." },
            new ChatMessage { Role = "user", Content = "No image here." },
            new ChatMessage { Role = "user", Content = "Second batch", ImagePaths = new List<string> { "c.png" } }
        };

        var imagePaths = ModelService.GetImagePathsInPromptOrder(history);

        Assert.Equal(new[] { "a.png", "b.png", "c.png" }, imagePaths);
    }

    [Fact]
    public void PrepareHistoryForInference_NormalizesEarlierVideoTurns()
    {
        string? prior = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
        try
        {
            Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", "4");

            var history = new List<ChatMessage>
            {
                new ChatMessage
                {
                    Role = "user",
                    Content = "Describe this video.",
                    IsVideo = true,
                    ImagePaths = Enumerable.Range(1, 8).Select(i => $"frame{i:D2}.png").ToList()
                },
                new ChatMessage { Role = "assistant", Content = "It shows a moving banner." },
                new ChatMessage { Role = "user", Content = "What color dominates it?" }
            };

            var prepared = ModelService.PrepareHistoryForInference(history, "gemma4");

            Assert.NotSame(history, prepared);
            Assert.Equal(4, prepared[0].ImagePaths.Count);
            Assert.Equal(new[] { "frame01.png", "frame03.png", "frame06.png", "frame08.png" }, prepared[0].ImagePaths);
            Assert.Same(history[1], prepared[1]);
            Assert.Same(history[2], prepared[2]);
        }
        finally
        {
            Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", prior);
        }
    }
}


