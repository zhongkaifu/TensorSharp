using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorSharp.Models.QwenImage;

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

    /// <summary>
    /// MiniMax-H3-Image (a model that takes masks and several pictures, with a 2048 x 2048 floor):
    /// a selection on a picture of the chat, with a second photo attached as a reference, reaches
    /// the model as one edit of [that picture, the photo] whose mask is the picture's, with every
    /// selection setting, the picture's own size kept, and the model's floor as the area; the
    /// picture's frame records the sources and the selection it was made with.
    /// </summary>
    [Fact]
    public async Task ASelectionOnAPictureOfTheChatWithAReferenceReachesAMiniMaxH3ImageEdit()
    {
        string root = Path.Combine(Path.GetTempPath(), "masked-turns-" + Guid.NewGuid().ToString("N")[..8]);
        string uploads = Path.Combine(root, "uploads");
        Directory.CreateDirectory(uploads);
        try
        {
            var (chat, model) = ImageTurnsTests.ChatWithPictureModel(root, uploads, 2048L * 2048);
            void Png(string name, int w, int h, Func<int, int, float> value)
            {
                var px = new float[w * h * 3];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        px[(y * w + x) * 3] = px[(y * w + x) * 3 + 1] = px[(y * w + x) * 3 + 2] = value(x, y);
                File.WriteAllBytes(Path.Combine(uploads, name), ImageIO.EncodePng(new RgbImage(w, h, px)));
            }
            Png("made.png", 64, 48, (_, _) => 0.5f);
            Png("lamp.png", 32, 32, (_, _) => 0.25f);
            Png("selection.png", 64, 48, (x, y) => x >= 16 && x < 48 && y >= 8 && y < 40 ? 1f : 0f);
            using JsonDocument body = JsonDocument.Parse("""
                {"sessionId":"s1","messages":[
                  {"role":"user","content":"a library table"},
                  {"role":"assistant","content":"","imageUrl":"/uploads/made.png"},
                  {"role":"user","content":"put the brass lamp from <Picture 2> on the table, in the selected area",
                   "stillImagePaths":["lamp.png"],"imageIntent":"edit","imageSource":"made.png",
                   "maskPath":"selection.png","maskMode":"grayscale","maskInvert":false,"maskFeather":2}]}
                """);
            var frames = new List<JsonElement>();
            await foreach (object frame in ImageTurns.StreamAsync(chat, body.RootElement, ImageTurns.Planner.For(chat, uploads), CancellationToken.None))
                frames.Add(JsonSerializer.SerializeToElement(frame));

            var call = Assert.Single(model.Calls);
            Assert.Equal(new[] { (64, 48), (32, 32) }, call.Inputs.Select(i => (i.Width, i.Height)));
            Assert.NotNull(call.P.Mask);
            Assert.Equal((64, 48), (call.P.Mask.Width, call.P.Mask.Height));
            Assert.Equal(QwenImageMaskMode.Grayscale, call.P.MaskMode);
            Assert.Equal(2, call.P.MaskFeather);
            Assert.True(call.P.KeepSourceSize);
            Assert.Equal(2048L * 2048, call.P.TargetArea);
            JsonElement picture = Assert.Single(frames, f => f.TryGetProperty("imageUrl", out _));
            Assert.Equal(new[] { "made.png", "lamp.png" }, picture.GetProperty("imageSources").EnumerateArray().Select(x => x.GetString()));
            Assert.Equal("selection.png", picture.GetProperty("imageMask").GetProperty("maskPath").GetString());
            Assert.Equal("asked", picture.GetProperty("imagePlanReason").GetString());
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
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
