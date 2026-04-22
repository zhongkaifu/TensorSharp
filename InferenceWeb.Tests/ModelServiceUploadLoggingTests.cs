// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Verifies the per-turn upload manifest baked into the chat audit log: the
/// <see cref="ModelService.SerializeUploadsForLog"/> helper for the latest user
/// turn, the per-message attachment paths inside
/// <see cref="ModelService.SerializeMessagesForLog"/>, and the WebUI request
/// parser's handling of the new <c>textFilePaths</c> field.
/// </summary>
public class ModelServiceUploadLoggingTests
{
    [Fact]
    public void SerializeUploadsForLog_NoMessage_ReturnsEmptyArray()
    {
        Assert.Equal("[]", ModelService.SerializeUploadsForLog(null));
    }

    [Fact]
    public void SerializeUploadsForLog_MessageWithoutAttachments_ReturnsEmptyArray()
    {
        var msg = new ChatMessage { Role = "user", Content = "Hello" };
        Assert.Equal("[]", ModelService.SerializeUploadsForLog(msg));
    }

    [Fact]
    public void SerializeUploadsForLog_MixedAttachments_EmitsPathNameAndMediaType()
    {
        var msg = new ChatMessage
        {
            Role = "user",
            Content = "Look at these",
            ImagePaths = new List<string> { "/uploads/abc.jpg", "/uploads/def.png" },
            AudioPaths = new List<string> { "/uploads/clip.mp3" },
            TextFilePaths = new List<string> { "/uploads/notes.txt" },
        };

        string json = ModelService.SerializeUploadsForLog(msg);
        using var doc = JsonDocument.Parse(json);
        var entries = doc.RootElement.EnumerateArray().ToList();

        Assert.Equal(4, entries.Count);

        Assert.Equal("/uploads/abc.jpg", entries[0].GetProperty("path").GetString());
        Assert.Equal("abc.jpg", entries[0].GetProperty("name").GetString());
        Assert.Equal("image", entries[0].GetProperty("mediaType").GetString());

        Assert.Equal("/uploads/def.png", entries[1].GetProperty("path").GetString());
        Assert.Equal("image", entries[1].GetProperty("mediaType").GetString());

        Assert.Equal("/uploads/clip.mp3", entries[2].GetProperty("path").GetString());
        Assert.Equal("clip.mp3", entries[2].GetProperty("name").GetString());
        Assert.Equal("audio", entries[2].GetProperty("mediaType").GetString());

        Assert.Equal("/uploads/notes.txt", entries[3].GetProperty("path").GetString());
        Assert.Equal("notes.txt", entries[3].GetProperty("name").GetString());
        Assert.Equal("text", entries[3].GetProperty("mediaType").GetString());
    }

    [Fact]
    public void SerializeUploadsForLog_VideoFrames_TaggedAsVideoFrame()
    {
        var msg = new ChatMessage
        {
            Role = "user",
            Content = "Describe video",
            IsVideo = true,
            ImagePaths = new List<string> { "/uploads/frame01.png", "/uploads/frame02.png" },
        };

        string json = ModelService.SerializeUploadsForLog(msg);
        using var doc = JsonDocument.Parse(json);
        var entries = doc.RootElement.EnumerateArray().ToList();

        Assert.Equal(2, entries.Count);
        Assert.All(entries, e => Assert.Equal("video_frame", e.GetProperty("mediaType").GetString()));
        Assert.Equal("frame01.png", entries[0].GetProperty("name").GetString());
        Assert.Equal("frame02.png", entries[1].GetProperty("name").GetString());
    }

    [Fact]
    public void SerializeUploadsForLog_NullOrEmptyPathsAreSkipped()
    {
        var msg = new ChatMessage
        {
            Role = "user",
            ImagePaths = new List<string> { "", null!, "/uploads/real.png" },
            AudioPaths = new List<string>(),
        };

        string json = ModelService.SerializeUploadsForLog(msg);
        using var doc = JsonDocument.Parse(json);
        var entries = doc.RootElement.EnumerateArray().ToList();

        Assert.Single(entries);
        Assert.Equal("/uploads/real.png", entries[0].GetProperty("path").GetString());
    }

    [Fact]
    public void SerializeMessagesForLog_IncludesPerTurnImageAudioAndTextFilePaths()
    {
        var history = new List<ChatMessage>
        {
            new ChatMessage { Role = "system", Content = "You are helpful." },
            new ChatMessage
            {
                Role = "user",
                Content = "What is in this picture?",
                ImagePaths = new List<string> { "/uploads/photo.jpg" },
            },
            new ChatMessage { Role = "assistant", Content = "A cat." },
            new ChatMessage
            {
                Role = "user",
                Content = "And this audio?",
                AudioPaths = new List<string> { "/uploads/clip.mp3" },
                TextFilePaths = new List<string> { "/uploads/transcript.txt" },
            },
        };

        string json = ModelService.SerializeMessagesForLog(history);
        using var doc = JsonDocument.Parse(json);
        var entries = doc.RootElement.EnumerateArray().ToList();

        Assert.Equal(4, entries.Count);

        Assert.False(entries[0].TryGetProperty("images", out _));
        Assert.False(entries[0].TryGetProperty("audios", out _));
        Assert.False(entries[0].TryGetProperty("textFiles", out _));

        var userImageEntry = entries[1];
        var images = userImageEntry.GetProperty("images").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Equal(new[] { "/uploads/photo.jpg" }, images);
        Assert.False(userImageEntry.TryGetProperty("audios", out _));

        var lastUser = entries[3];
        var audios = lastUser.GetProperty("audios").EnumerateArray().Select(e => e.GetString()).ToList();
        var textFiles = lastUser.GetProperty("textFiles").EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.Equal(new[] { "/uploads/clip.mp3" }, audios);
        Assert.Equal(new[] { "/uploads/transcript.txt" }, textFiles);
    }

    [Fact]
    public void SerializeMessagesForLog_OmitsAttachmentArraysWhenAbsent()
    {
        var history = new List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "Plain text" },
        };

        string json = ModelService.SerializeMessagesForLog(history);
        using var doc = JsonDocument.Parse(json);
        var entry = doc.RootElement.EnumerateArray().Single();

        Assert.False(entry.TryGetProperty("images", out _));
        Assert.False(entry.TryGetProperty("audios", out _));
        Assert.False(entry.TryGetProperty("textFiles", out _));
        Assert.False(entry.TryGetProperty("isVideo", out _));
    }

    [Fact]
    public void ChatMessageParser_ParseWebUi_ParsesTextFilePathsAlongsideImagesAndAudio()
    {
        const string body = """
            [
              {
                "role": "user",
                "content": "[File: notes.txt]\nfoo\n[End of file]\n\nWhat does this say?",
                "imagePaths": ["/uploads/a.png"],
                "audioPaths": ["/uploads/b.mp3"],
                "textFilePaths": ["/uploads/notes.txt"]
              }
            ]
            """;

        using var doc = JsonDocument.Parse(body);
        var messages = ChatMessageParser.ParseWebUi(doc.RootElement);

        var msg = Assert.Single(messages);
        Assert.Equal("user", msg.Role);
        Assert.Equal(new[] { "/uploads/a.png" }, msg.ImagePaths);
        Assert.Equal(new[] { "/uploads/b.mp3" }, msg.AudioPaths);
        Assert.Equal(new[] { "/uploads/notes.txt" }, msg.TextFilePaths);
    }

    [Fact]
    public void ChatMessageParser_ParseWebUi_NoTextFilePathsLeavesFieldNull()
    {
        const string body = """
            [
              { "role": "user", "content": "Just text", "imagePaths": ["/uploads/a.png"] }
            ]
            """;

        using var doc = JsonDocument.Parse(body);
        var messages = ChatMessageParser.ParseWebUi(doc.RootElement);

        Assert.Null(messages[0].TextFilePaths);
    }

    [Fact]
    public void PrepareHistoryForInference_PreservesTextFilePathsOnVideoDownsample()
    {
        string? prior = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
        try
        {
            Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", "3");

            var history = new List<ChatMessage>
            {
                new ChatMessage
                {
                    Role = "user",
                    Content = "Describe video",
                    IsVideo = true,
                    ImagePaths = Enumerable.Range(1, 6).Select(i => $"frame{i:D2}.png").ToList(),
                    AudioPaths = new List<string> { "/uploads/track.mp3" },
                    TextFilePaths = new List<string> { "/uploads/script.txt" },
                },
            };

            var prepared = ModelService.PrepareHistoryForInference(history, "gemma4");

            Assert.Equal(new[] { "/uploads/track.mp3" }, prepared[0].AudioPaths);
            Assert.Equal(new[] { "/uploads/script.txt" }, prepared[0].TextFilePaths);
        }
        finally
        {
            Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", prior);
        }
    }
}
