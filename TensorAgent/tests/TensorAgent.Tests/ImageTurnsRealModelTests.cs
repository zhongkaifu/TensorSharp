// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using Xunit.Abstractions;

namespace TensorAgent.Tests;

/// <summary>
/// A fact that needs Qwen-Image-2.1 on Metal: <c>TENSORSHARP_QWEN21_DIT</c> names its DiT
/// GGUF, or a folder holding it, with the Qwen3-VL-8B text encoder and the VAE beside it (the
/// variable InferenceWeb.Tests gates its own Qwen-Image tests on). It says what is missing
/// rather than passing without weights.
/// </summary>
public sealed class QwenImageMetalFactAttribute : FactAttribute
{
    public QwenImageMetalFactAttribute() => Skip = ImageTurnsRealModelTests.Unavailable();
}

/// <summary>
/// The planner's own questions, put to the real image model through the chat service the app
/// uses (<see cref="ImageTurns.Planner.For"/>): a picture chat as the page sends it, and what
/// the turn decides to do with each follow-up. Every other planner test stands in for the
/// model; this is the one that says whether its answers are any good, and what a question
/// costs. It writes a row per follow-up -- the plan, each kind of reading's probability, the
/// pictures' when which one to change was asked, and the time the questions took -- and fails
/// on any follow-up planned wrong.
///
/// <para>
/// The follow-ups are plain requests a person makes after a picture, in two languages, one
/// for each thing a turn can do; none was chosen or reworded for how the model answers it.
/// </para>
/// </summary>
[Collection(LiveModelCollection.Name)]
public sealed class ImageTurnsRealModelTests : IDisposable
{
    private const string DitVariable = "TENSORSHARP_QWEN21_DIT";

    private readonly ITestOutputHelper _output;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "image-turns-real-" + Guid.NewGuid().ToString("N")[..8]);
    private readonly string _uploads;

    public ImageTurnsRealModelTests(ITestOutputHelper output)
    {
        _output = output;
        _uploads = Path.Combine(_root, "uploads");
        Directory.CreateDirectory(_uploads);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    /// <summary>Why the real model cannot be asked here, or null when it can.</summary>
    internal static string? Unavailable()
    {
        string? value = Environment.GetEnvironmentVariable(DitVariable);
        if (string.IsNullOrWhiteSpace(value))
            return $"Requires Qwen-Image-2.1 weights ({DitVariable} not set).";
        if (DitPath(value) is null)
            return $"Requires Qwen-Image-2.1 weights (no qwen-image-2.1 DiT GGUF at {DitVariable}={value}).";
        return MetalLifetimeTests.MetalUnavailable();
    }

    /// <summary>The DiT GGUF the variable names: the file itself, or the one in that folder.</summary>
    private static string? DitPath(string value)
    {
        if (File.Exists(value))
            return value;
        if (!Directory.Exists(value))
            return null;
        return Directory.EnumerateFiles(value, "*.gguf")
            .Where(path => Path.GetFileName(path).Replace('_', '-').Contains("qwen-image-2.1", StringComparison.OrdinalIgnoreCase)
                && !Path.GetFileName(path).Contains("vae", StringComparison.OrdinalIgnoreCase))
            .OrderBy(path => path, StringComparer.Ordinal)
            .FirstOrDefault();
    }

    private const string Fox = "draw a red fox in a snowy forest";
    private const string Lighthouse = "now draw a lighthouse at night";

    /// <summary>A picture the model drew from words, as the page sends it back: the host's record on it.</summary>
    private static object Drawn(string name, string words) => new
    {
        role = "assistant", content = "", imageUrl = "/uploads/" + name,
        imagePlan = "new", imagePlanReason = "first", imageSources = Array.Empty<string>(), imagePrompt = words, imageSeed = 0,
    };

    private static JsonElement Body(params object[] messages) =>
        JsonSerializer.SerializeToElement(new { sessionId = "s-real", messages });

    /// <summary>
    /// One follow-up, and the plan it should get -- what to do, and to which picture -- when
    /// it has one right answer.
    /// </summary>
    private sealed record Case(string Conversation, JsonElement Body, string FollowUp, string? Kind = null, string? Picture = null);

    /// <summary>
    /// What the turn made of a follow-up: its plan, the picture that plan acts on, and, in the
    /// answer to what to do, the likeliest kind of reading and how far it led the next.
    /// </summary>
    private sealed record Reading(ImageTurns.ImagePlan Plan, string? Picture, float Margin, string Kind);

    /// <summary>One question the turn asked, the image model's answer, and how long it took.</summary>
    private sealed record Asked(ImageTurns.PlanQuestion Question, ImageIntentChoice Answer, long Ms);

    /// <summary>The fox, and the fox then the lighthouse, as the page sends a chat back with the host's records on its pictures.</summary>
    private (object[] OnePicture, object[] TwoPictures) Conversations()
    {
        File.WriteAllBytes(Path.Combine(_uploads, "fox.png"), new byte[] { 1 });
        File.WriteAllBytes(Path.Combine(_uploads, "lighthouse.png"), new byte[] { 1 });
        object[] one = { new { role = "user", content = Fox }, Drawn("fox.png", Fox) };
        return (one, one.Concat(new[] { new { role = "user", content = Lighthouse }, Drawn("lighthouse.png", Lighthouse) }).ToArray());
    }

    private static Case Follow(string conversation, object[] before, string followUp, string? kind = null, string? picture = null) =>
        new(conversation, Body(before.Append(new { role = "user", content = followUp }).ToArray()), followUp, kind, picture);

    [QwenImageMetalFact]
    public async Task TheRealImageModelPlansFollowUpsToAPictureChat()
    {
        (object[] onePicture, object[] twoPictures) = Conversations();
        Case[] cases =
        {
            Follow("fox", onePicture, "make it brighter", "edit", "fox.png"),
            Follow("fox", onePicture, "把背景换成海滩", "edit", "fox.png"),
            Follow("fox", onePicture, Lighthouse, "new"),
            Follow("fox", onePicture, "another one", "again", "fox.png"),
            Follow("fox, lighthouse", twoPictures, "make the fox bigger", "edit", "fox.png"),
        };

        var wrong = new List<string>();
        await AskAsync(cases, (test, reading) =>
        {
            bool right = reading.Plan.Kind == test.Kind && reading.Picture == test.Picture;
            if (!right)
                wrong.Add($"\"{test.FollowUp}\": {reading.Plan.Kind} {reading.Picture}, expected {test.Kind} {test.Picture}");
            return right;
        });

        Assert.Empty(wrong);
    }

    /// <summary>
    /// A measurement, not a check of <see cref="ImageTurns.UnsureMargin"/>: follow-ups a
    /// person could mean more than one way, in several languages, written down before any was
    /// put to the model and not changed after, and how sure the model is of each. It writes how
    /// many fall under the margin, which is how often the turn asks with this model. There is
    /// no right plan to check, so it fails only when the model cannot answer or the turn does
    /// not do what the answer to what to do says -- plumbing the stand-in tests already pin
    /// down. What the turn does with an unsure answer is ImageTurnsTests' to check, with
    /// answers made to be unsure.
    /// </summary>
    [QwenImageMetalFact]
    public async Task HowOftenTheRealImageModelIsUnsureOfAFollowUp()
    {
        (object[] onePicture, object[] twoPictures) = Conversations();
        Case[] cases =
        {
            Follow("fox", onePicture, "a wolf"),
            Follow("fox", onePicture, "in watercolor"),
            Follow("fox", onePicture, "try something different"),
            Follow("fox", onePicture, "the same but at sunset"),
            Follow("fox", onePicture, "what about a cat?"),
            Follow("fox", onePicture, "again, but cuter"),
            Follow("fox", onePicture, "换一个"),
            Follow("fox", onePicture, "もっと可愛く"),
            Follow("fox", onePicture, "nochmal, aber im Sommer"),
            Follow("fox", onePicture, "not quite"),
            Follow("fox, lighthouse", twoPictures, "add snow to it"),
            Follow("fox, lighthouse", twoPictures, "both together"),
            Follow("fox, lighthouse", twoPictures, "the other one, but at dawn"),
            Follow("fox, lighthouse", twoPictures, "one more"),
        };

        var inconsistent = new List<string>();
        int unsure = 0;
        await AskAsync(cases, (test, reading) =>
        {
            if (reading.Margin < ImageTurns.UnsureMargin)
                unsure++;
            bool consistent = reading.Margin < ImageTurns.UnsureMargin ? reading.Plan.Kind == "ask" : reading.Plan.Kind == reading.Kind;
            if (!consistent)
                inconsistent.Add($"\"{test.FollowUp}\": {reading.Plan.Kind} with {reading.Kind} ahead by {reading.Margin:F3}");
            return null;
        });

        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"under the {ImageTurns.UnsureMargin:F2} margin, so asked: {unsure} of {cases.Length}"));
        Assert.Empty(inconsistent);
    }

    /// <summary>
    /// The whole of it once, as the app runs it: a picture from words, then a follow-up the
    /// image model reads as a change to it, and the change made to that picture. The turns'
    /// frames carry the plan before the first step and the record on the picture, which the
    /// next turn would plan from. Twelve to fifteen minutes of GPU on an M5 Pro at 1024 x
    /// 1024: in the two recorded runs, 325 s for the picture both times, and 403 s and 581 s
    /// for the change.
    /// </summary>
    [QwenImageMetalFact]
    public async Task AFollowUpChangesThePictureTheModelReadItAs()
    {
        string dit = DitPath(Environment.GetEnvironmentVariable(DitVariable)!)!;
        using var model = new QwenImageModel(dit, BackendType.GgmlMetal);
        var models = new ModelService();
        WebUiChatService chat = ChatWith(models, model);
        try
        {
            ImageTurns.Planner real = ImageTurns.Planner.For(chat, _uploads);
            long asked = -1;
            var planner = new ImageTurns.Planner(_uploads, async (question, cancellationToken) =>
            {
                var clock = Stopwatch.StartNew();
                ImageIntentChoice? answer = await real.Choose(question, cancellationToken).ConfigureAwait(false);
                asked = clock.ElapsedMilliseconds;
                return answer;
            });

            var first = Stopwatch.StartNew();
            List<JsonElement> drawn = await FramesAsync(ImageTurns.StreamAsync(chat, Body(new { role = "user", content = Fox }), planner, CancellationToken.None));
            first.Stop();
            JsonElement fox = Assert.Single(drawn, f => f.TryGetProperty("imageUrl", out _));
            Assert.Equal("new", drawn[0].GetProperty("image_plan").GetString());
            Assert.Equal("first", drawn[0].GetProperty("image_plan_reason").GetString());
            Assert.Equal(-1, asked);
            string foxName = fox.GetProperty("imageUrl").GetString()!["/uploads/".Length..];

            // The picture comes back as the page keeps it: the frame's record copied onto the entry.
            var picture = new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "" };
            foreach (JsonProperty field in fox.EnumerateObject())
                picture[field.Name] = field.Value;
            var second = Stopwatch.StartNew();
            List<JsonElement> changed = await FramesAsync(ImageTurns.StreamAsync(chat,
                Body(new { role = "user", content = Fox }, picture, new { role = "user", content = "make it brighter" }), planner, CancellationToken.None));
            second.Stop();

            JsonElement plan = changed[0];
            Assert.Equal("edit", plan.GetProperty("image_plan").GetString());
            Assert.Equal("model", plan.GetProperty("image_plan_reason").GetString());
            Assert.Equal(new[] { "/uploads/" + foxName }, plan.GetProperty("image_sources").EnumerateArray().Select(s => s.GetString()));
            JsonElement result = Assert.Single(changed, f => f.TryGetProperty("imageUrl", out _));
            Assert.Equal("edit", result.GetProperty("imagePlan").GetString());
            Assert.Equal(new[] { foxName }, result.GetProperty("imageSources").EnumerateArray().Select(s => s.GetString()));
            Assert.Equal("make it brighter", result.GetProperty("imagePrompt").GetString());
            Assert.False(changed[^1].TryGetProperty("error", out JsonElement error), error.ToString());

            static double Mean(string path) => ImageIO.Load(path).Pixels.Average();
            string resultName = result.GetProperty("imageUrl").GetString()!["/uploads/".Length..];
            _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"picture from words {first.ElapsedMilliseconds} ms; follow-up planned in {asked} ms, changed in {second.ElapsedMilliseconds} ms in all; " +
                $"mean pixel {Mean(Path.Combine(_uploads, foxName)):F3} -> {Mean(Path.Combine(_uploads, resultName)):F3}"));
        }
        finally
        {
            SetLoadedModel(models, null);
            models.Dispose();
        }
    }

    private static async Task<List<JsonElement>> FramesAsync(IAsyncEnumerable<object> frames)
    {
        var all = new List<JsonElement>();
        await foreach (object frame in frames)
            all.Add(JsonSerializer.SerializeToElement(frame));
        return all;
    }

    /// <summary>
    /// Plan each case through the chat service's real method, and write a row for it: the plan,
    /// each kind of reading's probability in the answer to what to do, the pictures' in the
    /// answer to which one to change when that was asked, the prompts' lengths and the time
    /// the questions took. <paramref name="judge"/> says whether the plan is right, or null
    /// when no single plan is.
    /// </summary>
    private async Task AskAsync(Case[] cases, Func<Case, Reading, bool?> judge)
    {
        string dit = DitPath(Environment.GetEnvironmentVariable(DitVariable)!)!;
        using var model = new QwenImageModel(dit, BackendType.GgmlMetal);
        var models = new ModelService();
        WebUiChatService chat = ChatWith(models, model);
        try
        {
            ImageTurns.Planner real = ImageTurns.Planner.For(chat, _uploads);
            _output.WriteLine($"{Path.GetFileName(dit)} on ggml_metal; a question is planned only when the conversation leaves a choice");
            _output.WriteLine("| conversation | follow-up | expected | plan | P(change) | P(new) | P(again) | margin | which picture (probability) | tokens | ms |");
            _output.WriteLine("|---|---|---|---|---|---|---|---|---|---|---|");
            foreach (Case test in cases)
            {
                var asked = new List<Asked>();
                var planner = new ImageTurns.Planner(_uploads, async (question, cancellationToken) =>
                {
                    var clock = Stopwatch.StartNew();
                    ImageIntentChoice? answer = await real.Choose(question, cancellationToken).ConfigureAwait(false);
                    Assert.True(answer is not null, $"the image model could not answer \"{test.FollowUp}\"");
                    Assert.Equal(1f, answer!.Probabilities.Sum(), 1e-3f);
                    asked.Add(new Asked(question, answer, clock.ElapsedMilliseconds));
                    return answer;
                });

                ImageTurns.ImagePlan plan = (await ImageTurns.PlanAsync(test.Body, planner, CancellationToken.None))!;

                // The first question is what to do, one option per kind of reading; a second,
                // when there is one, is which picture to change.
                Assert.NotEmpty(asked);
                Asked what = asked[0];
                List<string> kinds = ImageTurns.KindsOf(
                    ImageTurns.OptionsFor(ImageTurns.ReadConversation(test.Body, _uploads)!.Pictures, _uploads)).ToList();
                Assert.Equal(kinds.Count, what.Question.Options.Count);
                float P(string kind) => kinds.IndexOf(kind) is var i and >= 0 ? what.Answer.Probabilities[i] : 0f;
                string[] ranked = kinds.OrderByDescending(P).ToArray();
                float margin = P(ranked[0]) - P(ranked[1]);

                // The picture the turn acts on: an edit names the one it changes; another
                // version repeats the request that made the one it names.
                string? picture = plan.Kind switch
                {
                    "edit" => plan.Sources.FirstOrDefault(),
                    "again" => plan.Prompt == Fox ? "fox.png" : plan.Prompt == Lighthouse ? "lighthouse.png" : plan.Prompt,
                    _ => null,
                };
                bool? right = judge(test, new Reading(plan, picture, margin, ranked[0]));

                string which = asked.Count < 2 ? "-" : string.Join("; ", asked[1].Question.Options.Select((text, i) =>
                    string.Create(CultureInfo.InvariantCulture, $"{text} ({asked[1].Answer.Probabilities[i]:F3})")));
                string expected = test.Kind is null ? "(any)" : $"{test.Kind} {test.Picture}".Trim();
                string verdict = right switch { true => "", false => " WRONG", null => "" };
                _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"| {test.Conversation} | {test.FollowUp} | {expected} | {$"{plan.Kind} {picture}".Trim()} ({plan.Reason}){verdict} | {P("edit"):F3} | {P("new"):F3} | {P("again"):F3} | {margin:F3} | {which} | {string.Join(" + ", asked.Select(a => a.Answer.PromptTokens))} | {string.Join(" + ", asked.Select(a => a.Ms))} |"));
            }
        }
        finally
        {
            // The model is this test's to dispose, not the service's.
            SetLoadedModel(models, null);
            models.Dispose();
        }
    }

    /// <summary>A chat service with <paramref name="model"/> loaded, as a host's would be after a load.</summary>
    private WebUiChatService ChatWith(ModelService models, QwenImageModel model)
    {
        var options = new ServerHostingOptions(
            startupModelPath: null, startupMmProjPath: null, defaultBackend: "ggml_metal",
            supportedBackends: new[] { new BackendOption("ggml_metal", "GGML Metal") },
            defaultMaxTokens: 256, maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
            defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
            uploadDirectory: _uploads, logDirectory: Path.Combine(_root, "logs"), fileLoggingEnabled: false,
            samplingDefaults: null);
        SetLoadedModel(models, model);
        return new WebUiChatService(models, new SessionManager(), options,
            new UploadStoragePolicy(_uploads), new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
    }

    private static void SetLoadedModel(ModelService models, QwenImageModel? model) =>
        models.LifecycleService.GetType().GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(models.LifecycleService, model);
}
