// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Exercises the merged mask/LoRA request path with the real empty-mask pipeline.
/// No checkpoint or LoRA weights are loaded: these are request integration tests,
/// not evidence of real-model adapter quality or performance.
/// </summary>
public sealed class QwenImageMaskLoraRequestTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-mask-lora-" + Guid.NewGuid().ToString("N"));
    private readonly ModelService _models = new();
    private readonly QwenImage21DiffusionModel _model;
    private readonly WebUiChatService _chat;
    private readonly byte[] _source;
    private readonly byte[] _mask;
    private readonly JsonElement _request;

    public QwenImageMaskLoraRequestTests()
    {
        Directory.CreateDirectory(_directory);
        // The pipeline must return before touching model weights. An absent mask or lost
        // alpha/inversion option would enter inference and fail on this uninitialized model.
        _model = QwenImage21TurboTests.ModelOf(QwenImageVariant.Base);
        SetLoraMetadata(Array.Empty<LoraSpec>());
        typeof(ModelLifecycleService).GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(_models.LifecycleService, _model);
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
        _source = ImageIO.EncodePng(new RgbImage(2, 3,
            Enumerable.Range(0, 18).Select(i => i / 20f).ToArray(), [0, .2f, .4f, .6f, .8f, 1]));
        _mask = ImageIO.EncodePng(new RgbImage(2, 3, Enumerable.Repeat(1f, 18).ToArray(), new float[6]));
        File.WriteAllBytes(Path.Combine(_directory, "source.png"), _source);
        File.WriteAllBytes(Path.Combine(_directory, "mask.png"), _mask);
        _request = JsonSerializer.SerializeToElement(new
        {
            imagePaths = new[] { "source.png" }, maskPath = "mask.png",
            prompt = "Keep this empty selection unchanged", width = 32, height = 32,
            maskMode = "alpha", maskInvert = true, maskFeather = 8, maskCrop = true, maskCropPadding = 16,
        });
    }

    private void SetLoraMetadata(IReadOnlyList<LoraSpec> specs) =>
        typeof(QwenImage21DiffusionModel).GetProperty(nameof(QwenImage21DiffusionModel.LoraSpecs))!.SetValue(_model, specs);

    private void AssertOriginal(object response)
    {
        JsonElement result = JsonSerializer.SerializeToElement(response);
        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        Assert.Equal(2, result.GetProperty("width").GetInt32());
        Assert.Equal(3, result.GetProperty("height").GetInt32());
        string name = Uri.UnescapeDataString(result.GetProperty("url").GetString()!.Split('/').Last());
        RgbImage actual = ImageIO.Load(Path.Combine(_directory, name), preserveAlpha: true);
        RgbImage expected = ImageIO.Decode(_source, preserveAlpha: true);
        Assert.Equal(expected.Pixels, actual.Pixels);
        Assert.Equal(expected.Alpha, actual.Alpha);
        Assert.False(_chat.IsGeneratingMedia);
    }

    [Fact]
    public async Task MultipartMaskOverloadRetainsExistingLorasAndAllMaskOptions()
    {
        LoraSpec[] existing = [new(Path.Combine(_directory, "already-loaded.safetensors"), .7f)];
        SetLoraMetadata(existing);
        AssertOriginal(await _chat.ImageEditAsync(_request, new[] { _source }, _mask, CancellationToken.None));
        Assert.Same(existing, _model.LoraSpecs);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExplicitEmptyLorasClearsExistingSetWhileMaskRemainsEffective(bool stream)
    {
        SetLoraMetadata(new[] { new LoraSpec(Path.Combine(_directory, "already-loaded.safetensors"), .7f) });
        if (stream)
        {
            var frames = new List<object>();
            await foreach (object frame in _chat.ImageEditStreamAsync(_request, Array.Empty<LoraSpec>(), CancellationToken.None))
                frames.Add(frame);
            AssertOriginal(Assert.Single(frames));
        }
        else AssertOriginal(await _chat.ImageEditAsync(_request, Array.Empty<LoraSpec>(), CancellationToken.None));
        Assert.Empty(_model.LoraSpecs);
    }

    [Fact]
    public async Task UnchangedExplicitLorasAreReusedWithoutReloadingWeights()
    {
        LoraSpec[] existing = [new(Path.Combine(_directory, "already-loaded.safetensors"), .7f)];
        SetLoraMetadata(existing);
        AssertOriginal(await _chat.ImageEditAsync(_request, existing.ToArray(), CancellationToken.None));
        Assert.Same(existing, _model.LoraSpecs);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FailedLoraSelectionPreservesPreviousSetAndTheMaskedRequestCanRetry(bool stream)
    {
        LoraSpec[] existing = [new(Path.Combine(_directory, "already-loaded.safetensors"), .7f)];
        LoraSpec[] missing = [new(Path.Combine(_directory, "missing.safetensors"))];
        SetLoraMetadata(existing);
        string requestBefore = _request.GetRawText();
        if (stream)
        {
            var frames = new List<object>();
            await foreach (object frame in _chat.ImageEditStreamAsync(_request, missing, CancellationToken.None)) frames.Add(frame);
            var terminal = JsonSerializer.SerializeToElement(Assert.Single(frames));
            Assert.True(terminal.GetProperty("done").GetBoolean());
            Assert.Contains("LoRA plug-ins could not be applied", terminal.GetProperty("error").GetString());
        }
        else
        {
            var error = await Assert.ThrowsAsync<WebUiRequestRejectedException>(() =>
                _chat.ImageEditAsync(_request, missing, CancellationToken.None));
            Assert.Equal(500, error.StatusCode);
            Assert.Contains("LoRA plug-ins could not be applied", error.Message);
        }
        Assert.Same(existing, _model.LoraSpecs);
        Assert.Equal(requestBefore, _request.GetRawText());
        Assert.False(_chat.IsGeneratingMedia);
        Assert.Empty(Directory.GetFiles(_directory, "edit-*.png"));
        AssertOriginal(await _chat.ImageEditAsync(_request, CancellationToken.None));
        Assert.Same(existing, _model.LoraSpecs);
    }

    [Fact]
    public void ExplicitNullLoraSelectionIsRejectedWithoutAlteringMaskRequest()
    {
        Assert.Throws<ArgumentNullException>(() => { _ = _chat.ImageEditAsync(_request, (IReadOnlyList<LoraSpec>)null!, CancellationToken.None); });
        Assert.Throws<ArgumentNullException>(() => { _ = _chat.ImageEditStreamAsync(_request, null!, CancellationToken.None); });
        Assert.Equal("mask.png", _request.GetProperty("maskPath").GetString());
        Assert.Empty(_model.LoraSpecs);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task InvalidMaskGeometryDoesNotApplyOrClearPerRequestLoras(bool stream)
    {
        LoraSpec[] existing = [new(Path.Combine(_directory, "already-loaded.safetensors"), .7f)];
        SetLoraMetadata(existing);
        ImageIO.SavePng(Path.Combine(_directory, "wrong-size.png"), new RgbImage(1, 1, [1, 1, 1], [0]));
        var invalid = JsonSerializer.SerializeToElement(new
        {
            imagePaths = new[] { "source.png" }, maskPath = "wrong-size.png",
            maskMode = "alpha", maskInvert = true, width = 32, height = 32,
        });
        foreach (IReadOnlyList<LoraSpec> requested in new IReadOnlyList<LoraSpec>[]
            { Array.Empty<LoraSpec>(), new[] { new LoraSpec(Path.Combine(_directory, "missing.safetensors")) } })
        {
            if (stream)
            {
                var frames = new List<object>();
                await foreach (object frame in _chat.ImageEditStreamAsync(invalid, requested, CancellationToken.None)) frames.Add(frame);
                var terminal = JsonSerializer.SerializeToElement(Assert.Single(frames));
                Assert.True(terminal.GetProperty("done").GetBoolean());
                Assert.Contains("same width and height", terminal.GetProperty("error").GetString());
            }
            else
            {
                var error = await Assert.ThrowsAsync<WebUiRequestRejectedException>(() =>
                    _chat.ImageEditAsync(invalid, requested, CancellationToken.None));
                Assert.Equal(400, error.StatusCode);
                Assert.Contains("same width and height", error.Message);
            }
            Assert.Same(existing, _model.LoraSpecs);
            Assert.False(_chat.IsGeneratingMedia);
            Assert.Empty(Directory.GetFiles(_directory, "edit-*.png"));
        }
        AssertOriginal(await _chat.ImageEditAsync(_request, CancellationToken.None));
        Assert.Same(existing, _model.LoraSpecs);
    }

    public void Dispose()
    {
        // Detach the intentionally uninitialized fixture; it owns no native model resources.
        typeof(ModelLifecycleService).GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(_models.LifecycleService, null);
        _models.Dispose();
        Directory.Delete(_directory, recursive: true);
    }
}
