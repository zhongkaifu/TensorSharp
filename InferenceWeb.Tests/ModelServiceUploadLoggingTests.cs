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
    public void SerializeUploadsForLog_VideoFrames_AreOneEntryPerFolderBesideTheVideoFile()
    {
        var msg = new ChatMessage
        {
            Role = "user",
            Content = "Describe video",
            IsVideo = true,
            ImagePaths = new List<string> { "/tmp/clip/frame01.png", "/tmp/clip/frame02.png", "/uploads/still.png", "/tmp/clip/frame03.png" },
            // The third is a still image beside the clip's timed frames.
            ImageTimestamps = new List<double?> { 0.0, 0.5, null, 1.0 },
            VideoFilePaths = new List<string> { "/uploads/clip.mp4" },
        };

        string json = ModelService.SerializeUploadsForLog(msg);
        using var doc = JsonDocument.Parse(json);
        var entries = doc.RootElement.EnumerateArray().ToList();

        Assert.Equal(3, entries.Count);
        Assert.Equal(("image", "/uploads/still.png", "still.png"),
            (entries[0].GetProperty("mediaType").GetString(), entries[0].GetProperty("path").GetString(), entries[0].GetProperty("name").GetString()));
        Assert.Equal(("video", "/uploads/clip.mp4", "clip.mp4"),
            (entries[1].GetProperty("mediaType").GetString(), entries[1].GetProperty("path").GetString(), entries[1].GetProperty("name").GetString()));
        Assert.Equal("video_frames", entries[2].GetProperty("mediaType").GetString());
        Assert.Equal(System.IO.Path.GetDirectoryName("/tmp/clip/frame01.png"), entries[2].GetProperty("path").GetString());
        Assert.Equal(3, entries[2].GetProperty("count").GetInt32());
        Assert.False(entries[2].TryGetProperty("name", out _));
        Assert.DoesNotContain("frame01.png", json);
    }

    [Fact]
    public void SerializeUploadsForLog_FramesWithoutTimesAreVideoFramesToo()
    {
        var msg = new ChatMessage
        {
            Role = "user",
            IsVideo = true,
            ImagePaths = new List<string> { "/tmp/a/f1.png", "/tmp/a/f2.png", "/tmp/b/f3.png" },
        };

        using var doc = JsonDocument.Parse(ModelService.SerializeUploadsForLog(msg));
        var entries = doc.RootElement.EnumerateArray().ToList();

        Assert.Equal(2, entries.Count);
        Assert.All(entries, e => Assert.Equal("video_frames", e.GetProperty("mediaType").GetString()));
        Assert.Equal(new[] { 2, 1 }, entries.Select(e => e.GetProperty("count").GetInt32()));
    }

    [Fact]
    public void SerializeMessagesForLog_CountsAVideosFramesAndNamesTheVideoFile()
    {
        var history = new List<ChatMessage>
        {
            new ChatMessage
            {
                Role = "user",
                Content = "What happens?",
                IsVideo = true,
                ImagePaths = new List<string> { "/tmp/clip/frame01.png", "/tmp/clip/frame02.png", "/uploads/still.png" },
                ImageTimestamps = new List<double?> { 0.0, 0.5, null },
                VideoFilePaths = new List<string> { "/uploads/clip.mp4" },
            },
        };

        string json = ModelService.SerializeMessagesForLog(history);
        using var doc = JsonDocument.Parse(json);
        var entry = doc.RootElement.EnumerateArray().Single();

        Assert.Equal(new[] { "/uploads/still.png" }, entry.GetProperty("images").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal(2, entry.GetProperty("videoFrames").GetInt32());
        Assert.Equal(new[] { "/uploads/clip.mp4" }, entry.GetProperty("videos").EnumerateArray().Select(e => e.GetString()));
        Assert.True(entry.GetProperty("isVideo").GetBoolean());
        Assert.DoesNotContain("frame01.png", json);
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
        Assert.False(entry.TryGetProperty("videoFrames", out _));
        Assert.False(entry.TryGetProperty("videos", out _));
        Assert.False(entry.TryGetProperty("audios", out _));
        Assert.False(entry.TryGetProperty("textFiles", out _));
        Assert.False(entry.TryGetProperty("isVideo", out _));
    }

    [Fact]
    public void SerializeMessagesForLog_BoundsUploadedDocumentContent()
    {
        string document = new('x', 200_000);
        var history = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = document,
                TextFilePaths = new List<string> { "/uploads/book.pdf" },
            },
        };

        string json = ModelService.SerializeMessagesForLog(history);
        using var doc = JsonDocument.Parse(json);
        var entry = doc.RootElement.EnumerateArray().Single();

        Assert.Equal(200_000, entry.GetProperty("contentChars").GetInt32());
        Assert.True(entry.GetProperty("contentOmitted").GetBoolean());
        Assert.False(entry.TryGetProperty("contentTruncated", out _));
        Assert.True(entry.GetProperty("content").GetString()!.Length < 600);
        Assert.True(json.Length < 1_000);
    }

    [Fact]
    public void SerializeMessagesForLog_RedactsMarkerOnlyDocumentButKeepsInstruction()
    {
        string content = $"[File: book.pdf]\n{new string('q', 200_000)}\n[End of file]\n\nSummarize the book.";
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = content },
        };

        string json = ModelService.SerializeMessagesForLog(history);
        using var doc = JsonDocument.Parse(json);
        var entry = doc.RootElement.EnumerateArray().Single();

        Assert.Equal(content.Length, entry.GetProperty("contentChars").GetInt32());
        Assert.True(entry.GetProperty("contentOmitted").GetBoolean());
        Assert.Contains("Summarize the book.", entry.GetProperty("content").GetString());
        Assert.DoesNotContain(new string('q', 1_024), json);
        Assert.True(json.Length < 1_000);
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
    public void ChatMessageParser_ParseWebUi_RecognizesMetadataOnlyCsvAttachment()
    {
        const string body = """
            [
              {
                "role": "user",
                "content": "Please analyze this form.",
                "textFilePaths": ["stored.csv"],
                "textFileNames": ["responses.csv"],
                "attachments": [
                  {
                    "file": "stored.csv",
                    "fileName": "responses.csv",
                    "mediaType": "text",
                    "fileBacked": true
                  }
                ]
              }
            ]
            """;

        using var doc = JsonDocument.Parse(body);
        ChatMessage message = Assert.Single(ChatMessageParser.ParseWebUi(doc.RootElement));

        Assert.True(message.HasFileBackedTextAttachments);
        Assert.Equal(new[] { "stored.csv" }, message.TextFilePaths);
        Assert.Equal(new[] { "stored.csv" }, message.AttachmentPaths);
        Assert.Equal(new[] { "responses.csv" }, message.AttachmentNames);
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
                    TextFileNames = new List<string> { "script.txt" },
                    CacheControl = new CacheControlMarker(),
                    ContentCacheBreakpoints = new List<int> { 8 },
                },
            };

            var prepared = ModelService.PrepareHistoryForInference(history, "gemma4");

            Assert.Equal(new[] { "/uploads/track.mp3" }, prepared[0].AudioPaths);
            Assert.Equal(new[] { "/uploads/script.txt" }, prepared[0].TextFilePaths);
            Assert.Equal(new[] { "script.txt" }, prepared[0].TextFileNames);
            Assert.NotNull(prepared[0].CacheControl);
            Assert.Equal(new[] { 8 }, prepared[0].ContentCacheBreakpoints);
        }
        finally
        {
            Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", prior);
        }
    }
}
