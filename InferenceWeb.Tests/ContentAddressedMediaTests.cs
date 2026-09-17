// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using static InferenceWeb.Tests.TranscriptTestHelper;

namespace InferenceWeb.Tests;

/// <summary>
/// Media is identified by content at the parse boundary (repro-image-turn.md C2): an
/// OpenAI client resending the same data URI each turn used to get a new random file,
/// hence a new "image" for prompt reuse and the vision encoder, every turn.
/// </summary>
public sealed class ContentAddressedMediaTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-media-id-" + Guid.NewGuid().ToString("N"));

    public ContentAddressedMediaTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static readonly byte[] Png = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4, 5, 6, 7, 8 };

    private List<ChatMessage> ParseImageTurn(UploadStoragePolicy uploads, byte[] image)
    {
        string dataUri = "data:image/png;base64," + Convert.ToBase64String(image);
        using var doc = JsonDocument.Parse(
            "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"describe\"}," +
            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + dataUri + "\"}}]}]");
        return ChatMessageParser.ParseOpenAI(doc.RootElement, uploads);
    }

    [Fact]
    public void AResentBase64Image_LandsOnTheSameFile_WrittenOnce()
    {
        var uploads = new UploadStoragePolicy(_dir);

        string first = ParseImageTurn(uploads, Png)[0].ImagePaths[0];
        long usedAfterFirst = uploads.UsedBytes;
        string second = ParseImageTurn(uploads, Png)[0].ImagePaths[0];

        Assert.Equal(first, second);
        Assert.Equal(MediaContentId.OfBytes(Png) + ".png", Path.GetFileName(first));
        Assert.Single(Directory.GetFiles(_dir));
        Assert.Equal(usedAfterFirst, uploads.UsedBytes);

        byte[] other = (byte[])Png.Clone();
        other[^1] = 99;
        Assert.NotEqual(first, ParseImageTurn(uploads, other)[0].ImagePaths[0]);
    }

    [Fact]
    public void MediaContentId_IsTheSameForTheSameBytesUnderAnyName()
    {
        string a = Path.Combine(_dir, "one.png");
        string b = Path.Combine(_dir, "two.png");
        File.WriteAllBytes(a, Png);
        File.WriteAllBytes(b, Png);

        Assert.Equal(MediaContentId.OfFile(a), MediaContentId.OfFile(b));
        Assert.Equal(MediaContentId.OfBytes(Png), MediaContentId.OfFile(a));
        Assert.Null(MediaContentId.OfFile(Path.Combine(_dir, "missing.png")));
    }

    /// <summary>The transcript key covers media by content: the Web UI stores an upload
    /// per upload, an API client per resend, and neither changes what the picture is.</summary>
    [Fact]
    public void TranscriptSplice_MatchesMediaByContent_NotByPath()
    {
        string a = Path.Combine(_dir, "upload-a.png");
        string b = Path.Combine(_dir, "upload-b.png");
        string c = Path.Combine(_dir, "upload-c.png");
        File.WriteAllBytes(a, Png);
        File.WriteAllBytes(b, Png);
        byte[] other = (byte[])Png.Clone();
        other[0] = 1;
        File.WriteAllBytes(c, other);

        var store = new ConversationTranscriptStore(maxChains: 16, maxTokens: 10_000);
        store.Record(
            new List<ChatMessage> { new() { Role = "user", Content = "describe", ImagePaths = new List<string> { a } } },
            Generated("A cat.", new List<int> { 4, 5 }), Emitted("A cat."), scope: "img-chat");

        List<ChatMessage> Next(string imagePath) => new()
        {
            new() { Role = "user", Content = "describe", ImagePaths = new List<string> { imagePath } },
            new() { Role = "assistant", Content = "A cat." },
            new() { Role = "user", Content = "what colour?" },
        };

        Assert.Equal(new[] { 4, 5 }, store.Augment(Next(b)).History[1].RawOutputTokens);
        Assert.Null(store.Augment(Next(c)).History[1].RawOutputTokens);
    }
}
