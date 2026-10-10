// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;
using System.Text.Json;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

/// <summary>
/// What an attached file is to the two things that consume one: the model, and the
/// interpreter a program runs in.
///
/// <para>
/// The second half was missing, and the failure it produced looked like a stupid
/// model. Asked to turn a photo into a PDF, the app showed the picture to the vision
/// encoder and staged NOTHING into the working directory — only text uploads were
/// ever staged — so the model was told, truthfully, that it could run programs, and
/// then spent the turn guessing at a filename that was never going to exist.
/// </para>
/// <para>
/// The parser and the collector are reached by reflection because both are internal
/// to <c>TensorSharp.Chat</c>. That is worth the ugliness here: these two functions
/// are the entire contract, they are pure, and testing them through a loaded model
/// costs three minutes and cannot say which of them was wrong.
/// </para>
/// </summary>
public sealed class ChatAttachmentTests : IDisposable
{
    private readonly string _uploads = Path.Combine(Path.GetTempPath(), "tensoragent-att-" + Guid.NewGuid().ToString("N"));

    public ChatAttachmentTests() => Directory.CreateDirectory(_uploads);

    public void Dispose()
    {
        try { Directory.Delete(_uploads, true); } catch { }
    }

    private static readonly Assembly Chat = typeof(TensorSharp.Chat.WebUiChatService).Assembly;

    private static List<ChatMessage> Parse(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        Type parser = Chat.GetType("TensorSharp.Server.RequestParsers.ChatMessageParser")!;
        return (List<ChatMessage>)parser
            .GetMethod("ParseWebUi", BindingFlags.Public | BindingFlags.Static)!
            .Invoke(null, new object[] { document.RootElement })!;
    }

    private static string? Resolve(List<ChatMessage> messages, string uploadRoot)
    {
        Type parser = Chat.GetType("TensorSharp.Server.RequestParsers.ChatMessageParser")!;
        return (string?)parser
            .GetMethod("ResolveAttachmentPaths", BindingFlags.Public | BindingFlags.Static)!
            .Invoke(null, new object[] { messages, uploadRoot });
    }

    private static IReadOnlyList<CodeInputFile> Collect(List<ChatMessage> messages) =>
        (IReadOnlyList<CodeInputFile>)typeof(TensorSharp.Chat.WebUiChatService)
            .GetMethod("CollectCodeInputFiles", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, new object[] { messages })!;

    private string Upload(string name)
    {
        string path = Path.Combine(_uploads, name);
        File.WriteAllText(path, "x");
        return path;
    }

    /// <summary>
    /// A message whose optional lists are explicitly null, which is what a reopened
    /// chat sends back.
    ///
    /// <para>
    /// The whole history goes out with every message, and the history a resumed page
    /// holds is the one the app handed it -- <c>StoredMessage</c> serialized with its
    /// nine nullable lists written out as <c>null</c>. Four of the five reads here
    /// called <see cref="JsonElement.GetArrayLength"/> without checking the kind
    /// first, which THROWS on a null, so <c>/api/chat</c> answered 500 from inside the
    /// parser. From the phone that is a chat that works until you reopen it and then
    /// refuses everything, with nothing on screen to say why.
    /// </para>
    /// </summary>
    [Fact]
    public void AMessageWhoseOptionalListsAreNullIsReadRatherThanThrown()
    {
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "and what colour is the car?",
               "thinking": null, "imagePaths": null, "stillImagePaths": null,
               "videoFilePaths": null, "audioPaths": null, "textFilePaths": null,
               "textFileNames": null, "isVideo": null, "attachments": null,
               "artifacts": null, "imageUrl": null }]
            """);

        ChatMessage only = Assert.Single(messages);
        Assert.Equal("and what colour is the car?", only.Content);
        Assert.Null(only.ImagePaths);
        Assert.Null(only.AudioPaths);
        Assert.Null(only.TextFilePaths);
        Assert.Null(only.TextFileNames);
        Assert.Null(Resolve(messages, _uploads));
        Assert.Empty(Collect(messages));
    }

    /// <summary>
    /// A photo the user attached is a file the model can open, under the name the
    /// user knows it by.
    /// </summary>
    [Fact]
    public void AnAttachedPhotoIsStagedForTheInterpreterAndNotOnlyForTheEncoder()
    {
        Upload("g1.png");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "turn this into a pdf",
               "imagePaths": ["g1.png"], "stillImagePaths": ["g1.png"],
               "attachments": [{ "file": "g1.png", "fileName": "IMG_0004.png", "mediaType": "image" }] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        IReadOnlyList<CodeInputFile> staged = Collect(messages);
        CodeInputFile only = Assert.Single(staged);
        Assert.Equal("IMG_0004.png", only.Name);
        Assert.Equal(Path.Combine(_uploads, "g1.png"), only.SourcePath);
    }

    /// <summary>
    /// Every kind, once each, under the user's own names — and a frame the server
    /// extracted is not one of them.
    /// </summary>
    [Fact]
    public void EveryAttachmentIsStagedOnceAndDerivedFramesAreNot()
    {
        foreach (string name in new[] { "g1.png", "g2.wav", "g3.mp4", "g4.md", "g3_0001.png" })
            Upload(name);

        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "[File: notes.md]\nBODY\n[End of file]\n\nlook",
               "imagePaths": ["g1.png", "g3_0001.png"],
               "audioPaths": ["g2.wav"],
               "videoFilePaths": ["g3.mp4"],
               "textFilePaths": ["g4.md"], "textFileNames": ["notes.md"],
               "isVideo": true,
               "attachments": [
                 { "file": "g1.png", "fileName": "cat.png", "mediaType": "image" },
                 { "file": "g2.wav", "fileName": "note.wav", "mediaType": "audio" },
                 { "file": "g3.mp4", "fileName": "clip.mp4", "mediaType": "video" },
                 { "file": "g4.md", "fileName": "notes.md", "mediaType": "text" }
               ] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        Assert.Equal(
            new[] { "cat.png", "clip.mp4", "note.wav", "notes.md" },
            Collect(messages).Select(f => f.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray());
    }

    /// <summary>
    /// A client that sends no <c>attachments</c> — the desktop page, and any build
    /// older than this — keeps exactly the behaviour it had.
    /// </summary>
    [Fact]
    public void AClientThatSendsNoAttachmentsArrayStillGetsItsDocumentsStaged()
    {
        Upload("g4.md");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "summarise",
               "textFilePaths": ["g4.md"], "textFileNames": ["notes.md"] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        CodeInputFile only = Assert.Single(Collect(messages));
        Assert.Equal("notes.md", only.Name);
    }

    /// <summary>
    /// An attachment path is confined to the upload directory like every other one.
    /// It is a client-supplied path that now reaches the filesystem, so the check
    /// that was already applied to images has to cover it too.
    /// </summary>
    [Fact]
    public void AnAttachmentPathOutsideTheUploadDirectoryIsRefused()
    {
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "read this",
               "attachments": [{ "file": "../../etc/passwd", "fileName": "passwd", "mediaType": "text" }] }]
            """);
        Assert.NotNull(Resolve(messages, _uploads));
    }

    /// <summary>
    /// A repeated name keeps the file it first meant, so a name cannot change which bytes
    /// a program reads halfway through a conversation -- and the second file is staged
    /// under a name of its own rather than dropped. Two photos pasted on a phone are both
    /// "image.png"; dropping the second left the program opening the first when the user
    /// meant the second, and refused a text-only model the second image outright.
    /// </summary>
    [Fact]
    public void ARepeatedNameKeepsTheFileItFirstMeantAndTheNextFileGetsItsOwn()
    {
        Upload("g1.png");
        Upload("g9.png");
        Upload("g10.png");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "one",
               "attachments": [{ "file": "g1.png", "fileName": "photo.png", "mediaType": "image" }] },
             { "role": "assistant", "content": "ok" },
             { "role": "user", "content": "two",
               "attachments": [{ "file": "g9.png", "fileName": "photo.png", "mediaType": "image" },
                               { "file": "g1.png", "fileName": "photo.png", "mediaType": "image" },
                               { "file": "g10.png", "fileName": "photo.png", "mediaType": "image" }] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        IReadOnlyList<CodeInputFile> files = Collect(messages);
        Assert.Equal(
            new[]
            {
                new CodeInputFile("photo.png", Path.Combine(_uploads, "g1.png")),
                new CodeInputFile("photo-2.png", Path.Combine(_uploads, "g9.png")),
                new CodeInputFile("photo-3.png", Path.Combine(_uploads, "g10.png")),
            },
            files);
    }

    /// <summary>
    /// Two pasted photos are both "image.png". For a text-only model every image must be
    /// staged before its vision input is removed, and the second used to be dropped by
    /// name -- so the request was refused as "no vision" although a tool could open both.
    /// </summary>
    [Fact]
    public void TwoPhotosWithTheSameNameAreBothStagedForATextOnlyModel()
    {
        Upload("p1.png");
        Upload("p2.png");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "first", "imagePaths": ["p1.png"],
               "attachments": [{ "file": "p1.png", "fileName": "image.png", "mediaType": "image" }] },
             { "role": "assistant", "content": "ok" },
             { "role": "user", "content": "put both in a pdf", "imagePaths": ["p2.png"],
               "attachments": [{ "file": "p2.png", "fileName": "image.png", "mediaType": "image" }] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        IReadOnlyDictionary<string, string> staged = Collect(messages)
            .ToDictionary(file => file.SourcePath, file => file.Name, StringComparer.Ordinal);

        Assert.Equal("image.png", staged[Path.Combine(_uploads, "p1.png")]);
        Assert.Equal("image-2.png", staged[Path.Combine(_uploads, "p2.png")]);
        Assert.True(TensorSharp.Chat.WebUiChatService.TryUseImagesAsStagedFiles(messages, staged));
    }

    /// <summary>
    /// An iPhone photo is staged as the PNG it is converted to, so its HEIC name reserves
    /// that PNG name too: a real "IMG_1.png" attached beside "IMG_1.heic" must not be
    /// overwritten by the conversion.
    /// </summary>
    [Fact]
    public void AHeicPhotoReservesTheNameOfItsConvertedCopy()
    {
        Upload("h1.heic");
        Upload("p1.png");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "both",
               "attachments": [{ "file": "h1.heic", "fileName": "IMG_1.heic", "mediaType": "image" },
                               { "file": "p1.png", "fileName": "IMG_1.png", "mediaType": "image" }] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        Assert.Equal(new[] { "IMG_1.heic", "IMG_1-2.png" }, Collect(messages).Select(f => f.Name).ToArray());
    }

    /// <summary>
    /// A text-only checkpoint may transform an image with a host file tool, but it
    /// must never receive an image placeholder. Both the old image in history and the
    /// one on the current turn are therefore downgraded only after their exact upload
    /// paths are known to have been staged; attachment provenance stays intact for the
    /// tool declaration and conversation recorder.
    /// </summary>
    [Fact]
    public void StagedCurrentAndHistoricalImagesBecomeFilesInsteadOfVisionInputs()
    {
        Upload("old.png");
        Upload("new.png");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "keep this",
               "imagePaths": ["old.png"],
               "attachments": [{ "file": "old.png", "fileName": "before.png", "mediaType": "image" }] },
             { "role": "assistant", "content": "ready" },
             { "role": "user", "content": "put both photos in a pdf",
               "imagePaths": ["new.png"],
               "attachments": [{ "file": "new.png", "fileName": "after.png", "mediaType": "image" }] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        IReadOnlyDictionary<string, string> staged = Collect(messages)
            .ToDictionary(file => file.SourcePath, file => file.Name, StringComparer.Ordinal);
        List<string>[] attachmentPaths = messages
            .Where(message => message.AttachmentPaths != null)
            .Select(message => new List<string>(message.AttachmentPaths!))
            .ToArray();

        Assert.True(TensorSharp.Chat.WebUiChatService.TryUseImagesAsStagedFiles(messages, staged));
        Assert.All(messages, message => Assert.Null(message.ImagePaths));
        Assert.Equal(attachmentPaths[0], messages[0].AttachmentPaths);
        Assert.Equal(attachmentPaths[1], messages[2].AttachmentPaths);
    }

    /// <summary>
    /// The downgrade is all-or-nothing. If even a historical image was not an
    /// explicit attachment, was shadowed away from the host tool, or failed staging,
    /// no message is mutated and the caller can return the vision_not_ready refusal.
    /// </summary>
    [Fact]
    public void MissingStagedImageKeepsEveryVisionInputForAnHonestRefusal()
    {
        Upload("old.png");
        Upload("new.png");
        List<ChatMessage> messages = Parse("""
            [{ "role": "user", "content": "old",
               "imagePaths": ["old.png"],
               "attachments": [{ "file": "old.png", "fileName": "old.png", "mediaType": "image" }] },
             { "role": "assistant", "content": "ready" },
             { "role": "user", "content": "analyze both",
               "imagePaths": ["new.png"],
               "attachments": [{ "file": "new.png", "fileName": "new.png", "mediaType": "image" }] }]
            """);
        Assert.Null(Resolve(messages, _uploads));

        var onlyCurrentWasStaged = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [Path.Combine(_uploads, "new.png")] = "new.png",
        };

        Assert.False(TensorSharp.Chat.WebUiChatService.TryUseImagesAsStagedFiles(messages, onlyCurrentWasStaged));
        Assert.Equal(Path.Combine(_uploads, "old.png"), Assert.Single(messages[0].ImagePaths!));
        Assert.Equal(Path.Combine(_uploads, "new.png"), Assert.Single(messages[2].ImagePaths!));
    }
}
