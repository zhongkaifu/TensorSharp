// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Primitives;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The OpenAI image routes (<c>/v1/images/generations</c>, <c>/v1/images/edits</c>) in the shapes SGLang serves: MiniMax's
/// examples/edit.py request, byte for byte as that script builds it, against a fake image model (no weights); the multipart
/// order of the <c>image</c> parts is the order the model receives the pictures in (<c>&lt;Picture n&gt;</c>); the field
/// mapping, <c>n</c> run one picture after another at seed + i, both response formats, and the refusals.
/// </summary>
public sealed class OpenAIImageRoutesTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-oai-images-" + Guid.NewGuid().ToString("N"));

    public OpenAIImageRoutesTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>An image model that records what it was asked and paints a picture of the asked size whose first pixel
    /// carries the first input's first byte.</summary>
    private sealed class RecordingImageModel : ModelBase, IImageGenerationModel
    {
        public RecordingImageModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "fake-image" };
        }

        public readonly List<(string Prompt, QwenImageParams P, IReadOnlyList<RgbImage> Inputs)> Calls = new();
        public string ImageModelFamily => "fake-image";
        public bool SupportsImageEdit => true;

        public RgbImage GenerateImage(string prompt, QwenImageParams p = null) => Make(prompt, p, Array.Empty<RgbImage>());
        public RgbImage EditImage(string prompt, IReadOnlyList<RgbImage> inputs, QwenImageParams p) => Make(prompt, p, inputs);

        private RgbImage Make(string prompt, QwenImageParams p, IReadOnlyList<RgbImage> inputs)
        {
            lock (Calls) Calls.Add((prompt, p, inputs.ToArray()));
            p.OnStep?.Invoke(1, 1, null);
            int w = p.Width > 0 ? p.Width : 64, h = p.Height > 0 ? p.Height : 32;
            var px = new float[w * h * 3];
            if (inputs.Count > 0) px[0] = inputs[0].Pixels[0];
            return new RgbImage(w, h, px);
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();
        protected override void ResetKVCacheCore() { }
    }

    private (WebUiAdapter Adapter, RecordingImageModel Model, ModelService Service) Build()
    {
        string gguf = Path.Combine(_dir, "fake.gguf");
        using (var writer = new BinaryWriter(File.Create(gguf)))
        {
            writer.Write(0x46554747u); writer.Write(3u); writer.Write(0UL); writer.Write(0UL); writer.Write(new byte[8]);
        }
        RecordingImageModel model = null;
        var service = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) => model = new RecordingImageModel(path));
        service.LoadModel(gguf, null, "cpu");
        var options = new ServerHostingOptions(
            startupModelPath: gguf, startupMmProjPath: null, defaultBackend: "cpu",
            supportedBackends: new[] { new BackendOption("cpu", "CPU") }, defaultMaxTokens: 100, maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0, defaultVideoHeight: 0, defaultVideoSteps: 0,
            defaultVideoMode: null, uploadDirectory: _dir, logDirectory: Path.Combine(_dir, "logs"), fileLoggingEnabled: false,
            samplingDefaults: null, skillsEnabled: false, skillsAllowScripts: false, skillsSandbox: SkillSandboxMode.Required,
            skillsAllowNetwork: false);
        var adapter = new WebUiAdapter(service, new SessionManager(), options, new UploadStoragePolicy(_dir),
            new SkillRegistry(new SkillRegistryOptions()), codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);
        return (adapter, model, service);
    }

    private static byte[] Png(byte first, int w = 8, int h = 8)
    {
        var px = new float[w * h * 3];
        px[0] = first / 255f;
        return ImageIO.EncodePng(new RgbImage(w, h, px));
    }

    /// <summary>examples/edit.py's multipart(): the text fields, then one image part per --image, in order.</summary>
    private static (byte[] Body, string ContentType) EditPy(IDictionary<string, string> fields, IReadOnlyList<(string Name, byte[] Bytes)> images)
    {
        string boundary = "minimax-h3-image-" + Guid.NewGuid().ToString("N");
        var parts = new List<byte[]>();
        foreach (var (name, value) in fields)
            parts.Add(Encoding.UTF8.GetBytes($"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"));
        foreach (var (name, bytes) in images)
        {
            parts.Add(Encoding.UTF8.GetBytes($"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n" +
                                             "Content-Type: image/png\r\n\r\n"));
            parts.Add(bytes.Concat(Encoding.ASCII.GetBytes("\r\n")).ToArray());
        }
        parts.Add(Encoding.UTF8.GetBytes($"--{boundary}--\r\n"));
        return (parts.SelectMany(p => p).ToArray(), $"multipart/form-data; boundary={boundary}");
    }

    private static HttpRequest Request(byte[] body, string contentType)
    {
        var context = new DefaultHttpContext();
        context.Request.Method = "POST";
        context.Request.ContentType = contentType;
        context.Request.Body = new MemoryStream(body);
        context.Request.ContentLength = body.Length;
        context.Response.Body = new MemoryStream();
        return context.Request;
    }

    private static async Task<(int Status, JsonElement Body)> Execute(IResult result, HttpRequest request)
    {
        var context = request.HttpContext;
        context.RequestServices = new ServiceCollection()
            .AddLogging().BuildServiceProvider();
        await result.ExecuteAsync(context);
        context.Response.Body.Position = 0;
        using var doc = await JsonDocument.ParseAsync(context.Response.Body);
        return (context.Response.StatusCode, doc.RootElement.Clone());
    }

    /// <summary>MiniMax's examples/edit.py, field for field: prompt, size (the script's --size), seed 1234, generator_device cpu,
    /// response_format b64_json, output_format png and the image parts; the answer is data[0].b64_json, a PNG of the size
    /// asked, and the model got the pictures in part order with the size and seed mapped.</summary>
    [Fact]
    public async Task EditPyRequestRunsAndTheImagePartsAreThePicturesInOrder()
    {
        var (adapter, model, service) = Build();
        using (service)
        {
            var fields = new Dictionary<string, string>
            {
                ["prompt"] = "Put <Picture 2> on the table of <Picture 1>.", ["size"] = "512x384", ["seed"] = "1234",
                ["generator_device"] = "cpu", ["response_format"] = "b64_json", ["output_format"] = "png",
            };
            var (body, type) = EditPy(fields, new[] { ("reference1.png", Png(11)), ("reference2.png", Png(22)), ("reference3.png", Png(33)) });
            HttpRequest request = Request(body, type);
            var (status, json) = await Execute(await adapter.OpenAIImageEditsAsync(request), request);
            Assert.Equal(200, status);
            string b64 = json.GetProperty("data")[0].GetProperty("b64_json").GetString();
            RgbImage png = ImageIO.Decode(Convert.FromBase64String(b64));
            Assert.Equal((512, 384), (png.Width, png.Height));
            Assert.Equal("512x384", json.GetProperty("size").GetString());
            var call = Assert.Single(model.Calls);
            Assert.Equal(fields["prompt"], call.Prompt);
            Assert.Equal((512, 384, 1234L), (call.P.Width, call.P.Height, call.P.Seed));
            Assert.Equal(new byte[] { 11, 22, 33 }, call.Inputs.Select(i => (byte)Math.Round(i.Pixels[0] * 255)).ToArray());
        }
    }

    /// <summary>JSON generations with n pictures made one after another at seed + i, answered as URLs (the default).</summary>
    [Fact]
    public async Task GenerationsRunNPicturesInSequenceAtConsecutiveSeeds()
    {
        var (adapter, model, service) = Build();
        using (service)
        {
            byte[] body = JsonSerializer.SerializeToUtf8Bytes(new
            {
                prompt = "a lighthouse", size = "1024x768", seed = 40, n = 3, num_inference_steps = 6, guidance_scale = 1.0,
                model = "MiniMaxAI/MiniMax-H3-Image",
            });
            HttpRequest request = Request(body, "application/json");
            var (status, json) = await Execute(await adapter.OpenAIImageGenerationsAsync(request), request);
            Assert.Equal(200, status);
            Assert.Equal(3, json.GetProperty("data").GetArrayLength());
            foreach (JsonElement d in json.GetProperty("data").EnumerateArray())
                Assert.StartsWith("/", d.GetProperty("url").GetString());
            Assert.Equal(new long[] { 40, 41, 42 }, model.Calls.Select(c => c.P.Seed).ToArray());
            Assert.All(model.Calls, c => Assert.Equal((1024, 768, 6, 1f, 0), (c.P.Width, c.P.Height, c.P.Steps, c.P.CfgScale, c.Inputs.Count)));
        }
    }

    [Theory]
    [InlineData("{\"size\":\"1024x1024\"}", "prompt is required")]
    [InlineData("{\"prompt\":\"x\",\"size\":\"big\"}", "size must be WIDTHxHEIGHT")]
    [InlineData("{\"prompt\":\"x\",\"size\":\"1000x1000\"}", "multiples of 32")]
    [InlineData("{\"prompt\":\"x\",\"n\":11}", "n must be an integer from 1 to 10")]
    [InlineData("{\"prompt\":\"x\",\"response_format\":\"base64\"}", "response_format must be url or b64_json")]
    [InlineData("{\"prompt\":\"x\",\"output_format\":\"webp\"}", "output_format must be png")]
    public async Task GenerationsRefuseMalformedRequestsInTheOpenAIEnvelope(string json, string message)
    {
        var (adapter, model, service) = Build();
        using (service)
        {
            HttpRequest request = Request(Encoding.UTF8.GetBytes(json), "application/json");
            var (status, body) = await Execute(await adapter.OpenAIImageGenerationsAsync(request), request);
            Assert.Equal(400, status);
            Assert.Contains(message, body.GetProperty("error").GetProperty("message").GetString());
            Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
            Assert.Empty(model.Calls);
        }
    }

    [Fact]
    public void FormFieldsMapToTheImageRoutesParameters()
    {
        var form = new FormCollection(new Dictionary<string, StringValues>
        {
            ["prompt"] = "x", ["size"] = "2048x1536", ["seed"] = "-5", ["num_inference_steps"] = "6", ["guidance_scale"] = "1",
            ["negative_prompt"] = "blurry", ["response_format"] = "url", ["quality"] = "high",
        });
        OpenAIImageRequest r = OpenAIImageRequest.FromForm(form);
        QwenImageParams p = WebUiChatService.ParseImageParameters(r.Parameters);
        Assert.Equal((2048, 1536, -5L, 6, 1f, "blurry"), (p.Width, p.Height, p.Seed, p.Steps, p.CfgScale, p.NegativePrompt));
        Assert.False(r.Base64);
        Assert.Equal(-3L, WebUiChatService.ParseImageParameters(r.ParametersFor(2)).Seed);
    }

    [Fact]
    public void EditFormsNeedAnImagePartAndTakeImageArrayPartsInOrder()
    {
        FormFile File(string name, byte b) => new(new MemoryStream(new[] { b }), 0, 1, name, name + ".png");
        var first = File("image[]", 1);
        var second = File("image[]", 2);
        var form = new FormCollection(new Dictionary<string, StringValues> { ["prompt"] = "x" }, new FormFileCollection { first, File("mask", 9), second });
        var (images, mask) = OpenAIImageRequest.FilesOf(form, 1024);
        Assert.Equal(new[] { first, second }, images);
        Assert.Equal("mask", mask.Name);
        var none = new FormCollection(new Dictionary<string, StringValues> { ["prompt"] = "x" }, new FormFileCollection { File("mask", 9) });
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => OpenAIImageRequest.FilesOf(none, 1024)).StatusCode);
    }
}
