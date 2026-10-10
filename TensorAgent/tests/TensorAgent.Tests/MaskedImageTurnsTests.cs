using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;

namespace TensorAgent.Tests;

public sealed class MaskedImageTurnsTests
{
    /// <summary>
    /// What a message with a photo or a selection asks for. Those are always what it edits,
    /// so the image model is never asked; the stand-in fails the test if it is.
    /// </summary>
    private static async Task<ImageTurns.ImagePlan> PlanAsync(JsonDocument body)
    {
        var planner = new ImageTurns.Planner(Path.GetTempPath(),
            (_, _) => throw new InvalidOperationException("an attached photo is never a question for the model"));
        return Assert.IsType<ImageTurns.ImagePlan>(await ImageTurns.PlanAsync(body.RootElement, planner, CancellationToken.None));
    }

    [Fact]
    public async Task SelectedAreaIsForwardedSeparatelyFromSourceAndReferences()
    {
        using JsonDocument body = JsonDocument.Parse("""
            {"messages":[{"role":"user","content":"change the scarf",
              "stillImagePaths":["source.png","reference.png"],"maskPath":"selection.png",
              "maskMode":"grayscale","maskInvert":true,"maskFeather":4,"maskCrop":true,"maskCropPadding":96}]}
            """);
        ImageTurns.ImagePlan request = await PlanAsync(body);
        JsonElement payload = JsonSerializer.SerializeToElement(request.Payload);
        Assert.True(request.Editing);
        Assert.Equal(new[] { "source.png", "reference.png" }, payload.GetProperty("imagePaths").EnumerateArray().Select(x => x.GetString()));
        Assert.Equal("selection.png", payload.GetProperty("maskPath").GetString());
        Assert.Equal("grayscale", payload.GetProperty("maskMode").GetString());
        Assert.True(payload.GetProperty("maskInvert").GetBoolean());
        Assert.Equal(4, payload.GetProperty("maskFeather").GetInt32());
        Assert.True(payload.GetProperty("maskCrop").GetBoolean());
        Assert.Equal(96, payload.GetProperty("maskCropPadding").GetInt32());
        // A selection keeps its canvas anyway; the field is the same for every edit.
        Assert.True(payload.GetProperty("keepSourceSize").GetBoolean());
    }

    [Fact]
    public async Task OldSelectionsDoNotLeakIntoNewImageRequests()
    {
        using JsonDocument body = JsonDocument.Parse("""
            {"messages":[{"role":"user","content":"edit","stillImagePaths":["old.png"],"maskPath":"old-mask.png"},
              {"role":"assistant","imageUrl":"/uploads/old-result.png"},
              {"role":"user","content":"new edit","stillImagePaths":["new.png"]}]}
            """);
        ImageTurns.ImagePlan request = await PlanAsync(body);
        Assert.False(JsonSerializer.SerializeToElement(request.Payload).TryGetProperty("maskPath", out _));
    }

    [Theory]
    [InlineData("\"mask.png\"")]
    [InlineData("123")]
    public async Task MissingSourceAndInvalidSelectionAreSentToServiceValidation(string mask)
    {
        using JsonDocument body = JsonDocument.Parse("""
            {"messages":[{"role":"user","content":"edit","maskPath":
            """ + mask + "}]}");
        ImageTurns.ImagePlan request = await PlanAsync(body);
        Assert.True(request.Editing);
        JsonElement payload = JsonSerializer.SerializeToElement(request.Payload);
        Assert.Empty(payload.GetProperty("imagePaths").EnumerateArray());
        Assert.Equal(mask, payload.GetProperty("maskPath").GetRawText());
    }

    [Fact]
    public void SavedConversationKeepsSelectionSettingsAndProtectsItsUpload()
    {
        var message = new StoredMessage
        {
            Content = "make the scarf red", StillImagePaths = ["source.png"],
            MaskPath = "mask.png", MaskMode = "grayscale", MaskFeather = 3, MaskCrop = true,
            Attachments = [new StoredAttachment { File = "source.png", MediaType = "image", MaskPath = "mask.png", MaskFeather = 3, MaskCrop = true }],
        };
        var loaded = JsonSerializer.Deserialize<StoredMessage>(JsonSerializer.Serialize(message))!;
        Assert.Equal("mask.png", loaded.MaskPath);
        Assert.Equal("grayscale", loaded.MaskMode);
        Assert.Equal(3, loaded.MaskFeather);
        Assert.True(loaded.MaskCrop);
        Assert.Equal("mask.png", Assert.Single(loaded.Attachments!).MaskPath);
        Assert.Contains("mask.png", loaded.ReferencedUploads);
    }

    [Fact]
    public void SavedConversationRetainsTheFullResolutionEditSourceAndUnavailableReason()
    {
        var message = new StoredMessage
        {
            StillImagePaths = ["camera.heic", "large.heic"],
            Attachments = [
                new StoredAttachment { File = "camera.heic", MediaType = "image", PreviewFile = "camera-preview.png", EditFile = "camera-edit.png" },
                new StoredAttachment { File = "large.heic", MediaType = "image", PreviewFile = "large-preview.png", EditUnavailableReason = "This image exceeds the editor's size limit." },
            ],
        };
        var loaded = JsonSerializer.Deserialize<StoredMessage>(JsonSerializer.Serialize(message))!;
        Assert.Equal("camera-edit.png", loaded.Attachments![0].EditFile);
        Assert.Equal("This image exceeds the editor's size limit.", loaded.Attachments[1].EditUnavailableReason);
        Assert.Equal(new[] { "camera-edit.png", "camera-preview.png", "camera.heic", "large-preview.png", "large.heic" },
            loaded.ReferencedUploads.Distinct().Order(StringComparer.Ordinal));
    }
}
