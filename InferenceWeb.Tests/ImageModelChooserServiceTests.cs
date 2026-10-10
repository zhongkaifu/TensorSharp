// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// WebUiChatService.ChooseWithImageModelAsync: the host's way to put a planning question to the
// loaded image model. It must answer null for any other model, wait for and hold the lock
// picture requests take, count as media work while it waits or runs (a model switch asks
// IsGeneratingMedia before unloading), hold a use of the model an unload waits for and give
// way to an unload that has begun, and let a caller's mistake or a cancellation through
// rather than turning them into "no answer".
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models.QwenImage;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class ImageModelChooserServiceTests : IDisposable
{
    private static readonly string[] Answers = { "Change picture [1]", "Make a new picture" };
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-image-chooser-" + Guid.NewGuid().ToString("N"));
    private readonly ModelService _models = new();
    private readonly WebUiChatService _chat;

    public ImageModelChooserServiceTests()
    {
        Directory.CreateDirectory(_directory);
        var options = new ServerHostingOptions(
            startupModelPath: null, startupMmProjPath: null, defaultBackend: "cpu",
            supportedBackends: null, defaultMaxTokens: 100, maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
            defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
            uploadDirectory: _directory, logDirectory: _directory, fileLoggingEnabled: false,
            samplingDefaults: null);
        _chat = new WebUiChatService(_models, new SessionManager(), options,
            new UploadStoragePolicy(_directory), new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
    }

    // No constructor ran: ChooseAnswer checks its arguments first and must fail on anything past them.
    private QwenImageModel LoadUninitializedImageModel()
    {
        var model = (QwenImageModel)RuntimeHelpers.GetUninitializedObject(typeof(QwenImageModel));
        SetLoadedModel(model);
        return model;
    }

    private object ImageLock =>
        typeof(WebUiChatService).GetField("_imageEditLock", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;

    private void SetLoadedModel(object? model) =>
        typeof(ModelLifecycleService).GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(_models.LifecycleService, model);

    [Fact]
    public async Task WithoutAnImageModel_ThereIsNoAnswerAndNoMediaJob()
    {
        Assert.Null(await _chat.ChooseWithImageModelAsync("s", "u", Answers, CancellationToken.None));
        Assert.False(_chat.IsGeneratingMedia);
    }

    [Fact]
    public async Task ACallersMistake_PropagatesAndTheJobEnds()
    {
        LoadUninitializedImageModel();
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _chat.ChooseWithImageModelAsync("s", "u", new[] { "only one" }, CancellationToken.None));
        Assert.False(_chat.IsGeneratingMedia);
    }

    [Fact]
    public async Task WhileAPictureHoldsTheModel_TheQuestionWaitsAsMediaWorkAndCanBeCancelled()
    {
        LoadUninitializedImageModel();
        object imageLock = ImageLock;
        using var cancel = new CancellationTokenSource();
        Task<ImageIntentChoice?> question;
        Monitor.Enter(imageLock);
        try
        {
            question = _chat.ChooseWithImageModelAsync("s", "u", Answers, cancel.Token);
            Thread.Sleep(100);
            Assert.False(question.IsCompleted);
            Assert.True(_chat.IsGeneratingMedia);
            cancel.Cancel();
        }
        finally
        {
            Monitor.Exit(imageLock);
        }
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => question);
        Assert.False(_chat.IsGeneratingMedia);
    }

    /// <summary>A loaded image model that cannot score this question gives no answer, and no error.</summary>
    [Fact]
    public async Task AModelThatCannotScore_GivesNoAnswer()
    {
        SetLoadedModel(QwenImageIntentScorerTests.ModelWithoutWeights(TensorSharp.Runtime.BackendType.Cpu));
        Assert.Null(await _chat.ChooseWithImageModelAsync("s", "u", Answers, CancellationToken.None));
        Assert.False(_chat.IsGeneratingMedia);
    }

    /// <summary>
    /// A model already being unloaded is not asked: a host's model switch has started freeing
    /// it. Nothing of it is used, so the unload has nothing to wait for.
    /// </summary>
    [Fact]
    public async Task ARetiringModel_IsNotAsked()
    {
        QwenImageModel model = LoadUninitializedImageModel();
        model.BeginRetirement();

        Assert.Null(await _chat.ChooseWithImageModelAsync("s", "u", Answers, CancellationToken.None));
        Assert.False(_chat.IsGeneratingMedia);
        Assert.True(model.WaitForUsesToDrain(TimeSpan.Zero));
    }

    /// <summary>
    /// The question holds a use of the model from the moment it reads it, so an unload that
    /// starts while the question waits behind a picture waits for it; and the question, once
    /// it has the lock, gives way rather than running on a model about to be freed. (Asked,
    /// this model -- no constructor ran -- would fail past its argument checks.)
    /// </summary>
    [Fact]
    public async Task AModelUnloadedWhileTheQuestionWaits_IsNotAsked()
    {
        QwenImageModel model = LoadUninitializedImageModel();
        Task<ImageIntentChoice?> question;
        Monitor.Enter(ImageLock);
        try
        {
            question = _chat.ChooseWithImageModelAsync("s", "u", Answers, CancellationToken.None);
            Assert.False(model.WaitForUsesToDrain(TimeSpan.Zero));
            Assert.True(_chat.IsGeneratingMedia);
            model.BeginRetirement();
        }
        finally
        {
            Monitor.Exit(ImageLock);
        }

        Assert.Null(await question);
        Assert.True(model.WaitForUsesToDrain(TimeSpan.FromSeconds(5)));
        Assert.False(_chat.IsGeneratingMedia);
    }

    public void Dispose()
    {
        // Detach the intentionally uninitialized fixture; it owns no native model resources.
        SetLoadedModel(null);
        _models.Dispose();
        Directory.Delete(_directory, recursive: true);
    }
}
