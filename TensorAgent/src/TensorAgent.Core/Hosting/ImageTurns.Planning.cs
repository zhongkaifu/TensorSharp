// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;
using TensorSharp.Chat;
using TensorSharp.Models.QwenImage;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// What a picture turn does with the conversation it ends: the rules, and the questions the
/// image model is asked when the rules leave a choice.
///
/// <para>
/// The rules run in order. Photos or a selection on the newest message are what it edits.
/// A reading the user chose on the page (<c>imageIntent</c>) is followed. A conversation
/// with no earlier picture draws one from the words. Only then is there a choice, between
/// changing one of the pictures already in the chat, drawing a new one, and making another
/// version of the newest one, and the options are built from the conversation's structure
/// alone: no phrase lists, so what a follow-up means is read in whatever language it is
/// written, by the Qwen3-VL language model the image model already carries. What to do is
/// one question; which picture to change is a second, asked only when there is more than
/// one to change and the answer to the first may be a change.
/// </para>
/// </summary>
public static partial class ImageTurns
{
    /// <summary>
    /// How far the image model's likeliest reading must lead the next one, in probability,
    /// before the turn acts on it. A reading is a kind of answer -- change a picture, draw a
    /// new one, make another version -- and each kind is one option of the question that
    /// decides it, however many pictures could be changed (<see cref="Question"/>). An answer
    /// that rates every option alike therefore gives every kind the same share, and the turn
    /// asks; so does one that reads nothing but the options' positions, which the image model
    /// averages out by asking in every rotation. When the pictures were options of their own,
    /// a change took one share per picture and won without asking once a chat had two.
    /// Which picture to change is asked after, and not held to this margin: a model sure of
    /// a change but split between two pictures is not unsure of what to do, and a question
    /// could only offer the change anyway. Below 0.15 the two likeliest readings are close
    /// enough that acting is close to a coin toss, and a wrong guess costs a whole picture
    /// (on an M5 Pro at 1024 x 1024, about 60 s for a new one and 85 s for an edit with a
    /// 6-step speed LoRA; 5.5 and 7 minutes at the model's own 40 steps), against one tap on
    /// a question. A fixed policy, not fitted to any set of messages.
    /// </summary>
    public const float UnsureMargin = 0.15f;

    /// <summary>
    /// How close two pictures' shares are for the question of which to change to be a tie,
    /// which the newest wins. Not a margin of doubt (that question is not held to
    /// <see cref="UnsureMargin"/>): the rounding an answer that reads only positions is left
    /// with after averaging over every rotation of the options.
    /// </summary>
    public const float PictureTie = 0.02f;

    /// <summary>
    /// The most pictures offered to be changed, newest first. An older one is still reachable
    /// with its own Edit button or by attaching it again.
    /// </summary>
    internal const int MaxPicturesOffered = 5;

    /// <summary>
    /// The longest question the image model is asked, in tokens of the whole prompt it scores
    /// (<see cref="QwenImageModel.ChooseAnswerPrompt"/>: the chat template, the instruction,
    /// the conversation, and the lettered options and reply format the model adds). Its text
    /// trunk holds F32 attention scores and probabilities of [tokens, tokens, 32 heads], so its
    /// scratch grows with the square of the question: 256 MB at 1,024 tokens, 1 GB at 2,048,
    /// against the roughly 1,100-token pass the conditioner already makes for an edit. A long
    /// chat is fitted under it by dropping its oldest messages first (see <see cref="Question"/>);
    /// a question the estimate still lets through over it gets no answer, not a failure.
    /// </summary>
    internal const int QuestionTokenLimit = QwenImageModel.MaxQuestionTokens;

    /// <summary>
    /// How a turn reads the conversation: the folder its pictures are kept in, because a
    /// picture whose file is gone is not offered, and the image model's answer to a
    /// multiple-choice question, or null when it cannot answer one.
    /// </summary>
    public sealed record Planner(string UploadDirectory, Func<PlanQuestion, CancellationToken, Task<ImageIntentChoice?>> Choose)
    {
        /// <summary>The planner a host uses: the image model <paramref name="chat"/> has loaded answers.</summary>
        public static Planner For(WebUiChatService chat, string uploadDirectory)
        {
            ArgumentNullException.ThrowIfNull(chat);
            ArgumentException.ThrowIfNullOrWhiteSpace(uploadDirectory);
            return new Planner(uploadDirectory, (question, cancellationToken) =>
                chat.ChooseWithImageModelAsync(question.System, question.User, question.Options, cancellationToken));
        }
    }

    /// <summary>
    /// One question for the image model: the instruction, the conversation and what is asked
    /// about it, and the options' texts in option order. The options are not lettered here and
    /// the question does not say how to answer: the image model letters them, in every
    /// rotation, and adds that line itself (<see cref="QwenImageModel.ChooseAnswer"/>).
    /// </summary>
    public sealed record PlanQuestion(string System, string User, IReadOnlyList<string> Options);

    /// <summary>The newest user message: its words, the photos and selection it carries, and
    /// how the user told the page to read it when they did (with the picture that names).</summary>
    internal sealed record ImageMessage(
        string Text, IReadOnlyList<string> Photos, IReadOnlyDictionary<string, JsonElement> Mask,
        string? Intent, string? Source);

    /// <summary>
    /// A picture earlier in the conversation, by its upload name: what the user called it
    /// when they attached it, or how the model made it.
    /// </summary>
    internal sealed record Picture(string Name, string? Label, ImageRecipe? Recipe)
    {
        public bool Made => Recipe is not null;
    }

    /// <summary>What a picture the model made was made from: enough to make it again.</summary>
    internal sealed record ImageRecipe(
        string Plan, IReadOnlyList<string> Sources, string Prompt, long Seed, IReadOnlyDictionary<string, JsonElement> Mask);

    /// <summary>A message before the newest, as the question retells it: the user's words and
    /// the pictures they attached, or a picture the model made.</summary>
    internal sealed record EarlierMessage(string? Said, IReadOnlyList<string> Attached, string? Made);

    /// <summary>A conversation as a picture turn reads it. <see cref="Pictures"/> are oldest
    /// first and exist on disk.</summary>
    internal sealed record ImageConversation(ImageMessage Newest, IReadOnlyList<Picture> Pictures, IReadOnlyList<EarlierMessage> Earlier);

    /// <summary>One reading of the newest message: <c>edit</c> or <c>again</c> of the picture
    /// <see cref="Source"/> names, or <c>new</c>.</summary>
    internal sealed record ImageOption(string Intent, string? Source);

    /// <summary>
    /// What a turn does: <see cref="Kind"/> is <c>edit</c>, <c>new</c> or <c>again</c>, and
    /// <see cref="Reason"/> says what decided it (<c>attached</c>, <c>asked</c>, <c>first</c>,
    /// <c>model</c>, or <c>unavailable</c> when the image model could not answer). Or, with
    /// <see cref="Choices"/>, no picture yet: the turn asks which reading was meant.
    /// </summary>
    internal sealed record ImagePlan(
        string Kind, string Reason, IReadOnlyList<string> Sources, string Prompt, long Seed,
        IReadOnlyDictionary<string, JsonElement> Mask, IReadOnlyList<ImageOption>? Choices = null)
    {
        /// <summary>Whether the image service edits (sources or a selection) rather than draws.</summary>
        public bool Editing => Sources.Count > 0 || Mask.Count > 0;

        /// <summary>
        /// The image service's body. A selection is forwarded raw so the service rejects a
        /// malformed one rather than silently editing the whole picture. A new picture and a
        /// change keep the service's default seed of 0, so the same words and pictures give the
        /// same picture as before (ImageBench's baselines); only another version carries its own
        /// (see <see cref="Again"/>).
        ///
        /// <para>
        /// Every edit keeps its source picture's size (<c>keepSourceSize</c>), and so does
        /// another version of an edit, which is an edit of the same source. Without it the
        /// output was sized from <see cref="DefaultTargetArea"/> alone: a selection edit gave
        /// a 1600 x 1200 photo back at 1600 x 1200, and the next change of that result came
        /// back at 1184 x 896. The area still bounds what an edit samples at.
        /// </para>
        /// </summary>
        public object Payload
        {
            get
            {
                var payload = new Dictionary<string, object?> { ["prompt"] = Prompt };
                if (Editing)
                {
                    payload["imagePaths"] = Sources;
                    payload["keepSourceSize"] = true;
                }
                payload["targetArea"] = DefaultTargetArea;
                foreach ((string name, JsonElement value) in Mask)
                    payload[name] = value;
                if (Kind == "again" || Seed != 0)
                    payload["seed"] = Seed;
                return payload;
            }
        }

        public static ImagePlan Edit(IReadOnlyList<string> photos, string prompt, IReadOnlyDictionary<string, JsonElement> mask, string reason) =>
            new("edit", reason, photos, prompt, 0, mask);

        /// <summary>
        /// Change <paramref name="picture"/> as <paramref name="prompt"/> says, in the user's own
        /// words: an edit model takes the instruction as it is given. At the service's seed, as
        /// an attached photo is: the engine keys an edit's noise to its pictures
        /// (QwenImage21Sampling.ReferenceStream, with the evidence), so a change of a picture the
        /// model made never starts from the noise that picture was drawn from, whether it comes
        /// from this rule or from Edit on the picture, which attaches it.
        /// </summary>
        public static ImagePlan Change(Picture picture, string prompt, string reason) =>
            new("edit", reason, new[] { picture.Name }, prompt, 0, NoMask);

        public static ImagePlan New(string prompt, string reason) =>
            new("new", reason, Array.Empty<string>(), prompt, 0, NoMask);

        /// <summary>
        /// The request that made <paramref name="picture"/>, with the next seed. Qwen-Image 2.1
        /// is deterministic for a seed, so the same request would give back the same picture.
        /// </summary>
        public static ImagePlan Again(Picture picture, string reason)
        {
            ImageRecipe recipe = picture.Recipe ?? throw new ArgumentException("Only a picture the model made can be made again.", nameof(picture));
            return new("again", reason, recipe.Sources, recipe.Prompt, recipe.Seed + 1, recipe.Mask);
        }

        public static ImagePlan Ask(IReadOnlyList<ImageOption> choices) =>
            new("ask", "model", Array.Empty<string>(), string.Empty, 0, NoMask, choices);
    }

    private static readonly IReadOnlyDictionary<string, JsonElement> NoMask = new Dictionary<string, JsonElement>();

    private static readonly string[] MaskSettings = ["maskPath", "maskMode", "maskInvert", "maskFeather", "maskCrop", "maskCropPadding"];

    /// <summary>
    /// The newest user message and the pictures before it, or null when the message asks for
    /// nothing (no words, photo or selection).
    /// </summary>
    /// <param name="body">The <c>/api/chat</c> request: the page sends the whole history.</param>
    /// <param name="uploadDirectory">Where pictures are kept; one whose file is gone is left out.</param>
    internal static ImageConversation? ReadConversation(JsonElement body, string uploadDirectory)
    {
        if (body.ValueKind != JsonValueKind.Object
            || !body.TryGetProperty("messages", out JsonElement list)
            || list.ValueKind != JsonValueKind.Array)
            return null;
        JsonElement[] messages = list.EnumerateArray().Where(m => m.ValueKind == JsonValueKind.Object).ToArray();
        int newestAt = Array.FindLastIndex(messages, m => Text(m, "role") == "user");
        if (newestAt < 0)
            return null;

        JsonElement user = messages[newestAt];
        string text = Text(user, "content")?.Trim() ?? string.Empty;
        // Stills only: a video's sampled frames are not a photo to edit.
        string[] photos = Paths(user, "stillImagePaths");
        IReadOnlyDictionary<string, JsonElement> mask = MaskOf(user);
        if (photos.Length == 0 && mask.Count == 0 && text.Length == 0)
            return null;
        string? intent = Text(user, "imageIntent");
        if (intent is not ("edit" or "new" or "again"))
            intent = null;
        var newest = new ImageMessage(text, photos, mask, intent, UploadName(Text(user, "imageSource")));

        var pictures = new List<Picture>();
        var known = new HashSet<string>(StringComparer.Ordinal);
        var earlier = new List<EarlierMessage>();
        JsonElement? asked = null;
        for (int i = 0; i < newestAt; i++)
        {
            JsonElement message = messages[i];
            string? role = Text(message, "role");
            if (role == "user")
            {
                asked = message;
                var attached = new List<string>();
                foreach (string still in Paths(message, "stillImagePaths"))
                {
                    if (UploadName(still) is not { } name || !Kept(uploadDirectory, name))
                        continue;
                    // A picture attached again (Edit on a result attaches it) is the same picture.
                    if (known.Add(name))
                        pictures.Add(new Picture(name, LabelOf(message, still) ?? name, null));
                    attached.Add(name);
                }
                earlier.Add(new EarlierMessage(Text(message, "content")?.Trim() ?? string.Empty, attached, null));
            }
            else if (role == "assistant"
                && PictureName(Text(message, "imageUrl")) is { } made
                && Kept(uploadDirectory, made) && known.Add(made))
            {
                pictures.Add(new Picture(made, null, RecipeOf(message) ?? Reconstructed(asked)));
                earlier.Add(new EarlierMessage(null, Array.Empty<string>(), made));
            }
        }
        return new ImageConversation(newest, pictures, earlier);
    }

    /// <summary>The plan for <paramref name="body"/>, or null when it asks for nothing.</summary>
    internal static async Task<ImagePlan?> PlanAsync(JsonElement body, Planner planner, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(planner);
        return ReadConversation(body, planner.UploadDirectory) is { } conversation
            ? await PlanAsync(conversation, planner, cancellationToken).ConfigureAwait(false)
            : null;
    }

    /// <summary>
    /// What to do with the newest message. The image model is asked only by the last rule;
    /// its answer being unavailable (null) falls back to changing the newest picture, which
    /// the page says. Its faults and cancellation propagate.
    /// </summary>
    internal static async Task<ImagePlan> PlanAsync(ImageConversation conversation, Planner planner, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        ArgumentNullException.ThrowIfNull(planner);
        ImageMessage newest = conversation.Newest;

        // (a) Photos or a selection on the message are what it edits: the first photo is the
        // source, later ones are references, and the selection applies to the first.
        if (newest.Photos.Count > 0 || newest.Mask.Count > 0)
            return ImagePlan.Edit(newest.Photos, newest.Text, newest.Mask, "attached");

        IReadOnlyList<Picture> pictures = conversation.Pictures;

        // (b) The user said which reading they meant, with a button that sent the message again.
        if (newest.Intent is { } intent)
            return Asked(intent, newest, pictures, planner.UploadDirectory);

        // (c) Nothing to change: a picture from the words. Every first turn ends here, and so
        // does every single-message body (ImageBench), without asking the model anything.
        if (pictures.Count == 0)
            return ImagePlan.New(newest.Text, "first");

        // (d) A choice, put to the image model as a multiple-choice question about what to do,
        // with one option per kind of reading.
        IReadOnlyList<ImageOption> options = OptionsFor(pictures, planner.UploadDirectory);
        IReadOnlyList<string> kinds = KindsOf(options);
        ImageIntentChoice? choice = await planner.Choose(Question(conversation, options), cancellationToken).ConfigureAwait(false);

        // (f) No answer (no output head in the encoder file, a backend it does not run on, a
        // question over its length): changing the newest picture is the likeliest reading of a
        // follow-up to a picture, and the page says the turn guessed and offers the other reading.
        if (choice is null)
            return ImagePlan.Change(pictures[^1], newest.Text, "unavailable");
        IReadOnlyList<float> byKind = ProbabilitiesOf(choice, kinds.Count);
        int[] ranked = Enumerable.Range(0, kinds.Count).OrderByDescending(i => byKind[i]).ToArray();
        string kind = kinds[ranked[0]];
        bool unsure = byKind[ranked[0]] - byKind[ranked[1]] < UnsureMargin;

        // Which picture, when the turn may change one: a second question, among the pictures
        // alone, asked only when there is more than one to choose from.
        ImageOption? edit = kind == "edit" || unsure
            ? await PictureToChangeAsync(conversation, options, planner, cancellationToken).ConfigureAwait(false)
            : null;

        // (e) Too close to call: ask, rather than spend minutes on a picture that may be the
        // wrong one. Only what to do is asked; which picture to change is the model's best.
        if (unsure)
            return ImagePlan.Ask(ChoicesFor(options, edit, pictures));

        return kind switch
        {
            // A picture question with no answer leaves the newest picture, as (f) does.
            "edit" => ImagePlan.Change(pictures.Last(p => p.Name == (edit?.Source ?? pictures[^1].Name)), newest.Text, "model"),
            "again" => ImagePlan.Again(pictures.Last(p => p.Name == options.First(o => o.Intent == "again").Source), "model"),
            _ => ImagePlan.New(newest.Text, "model"),
        };
    }

    /// <summary>
    /// The kinds of reading <paramref name="options"/> offer, in option order: the options of
    /// the question about what to do (<see cref="Question"/>). Every picture that could be
    /// changed is one change.
    /// </summary>
    internal static IReadOnlyList<string> KindsOf(IReadOnlyList<ImageOption> options) =>
        options.Select(o => o.Intent).Distinct(StringComparer.Ordinal).ToArray();

    /// <summary>
    /// The picture the image model reads the newest message as changing: the only one offered,
    /// or its answer to <see cref="PictureQuestion"/>; null when it cannot answer that.
    /// </summary>
    private static async Task<ImageOption?> PictureToChangeAsync(
        ImageConversation conversation, IReadOnlyList<ImageOption> options, Planner planner, CancellationToken cancellationToken)
    {
        ImageOption[] edits = options.Where(o => o.Intent == "edit").ToArray();
        if (edits.Length < 2)
            return edits.FirstOrDefault();
        if (await planner.Choose(PictureQuestion(conversation, options), cancellationToken).ConfigureAwait(false) is not { } choice)
            return null;
        IReadOnlyList<float> probabilities = ProbabilitiesOf(choice, edits.Length);
        // Ties go to the newest, which is listed first. An answer that reads only the
        // options' positions averages out to a tie, give or take rounding, and a tie decided
        // by rounding changed the ORIGINAL fox where the user, a turn after "make it
        // brighter", was refining the brighter one (measured: 0.50 / 0.50).
        int best = 0;
        for (int i = 1; i < edits.Length; i++)
        {
            if (probabilities[i] > probabilities[best])
                best = i;
        }
        return probabilities[best] - probabilities[0] > PictureTie ? edits[best] : edits[0];
    }

    /// <summary>
    /// The answer's probability for each of the <paramref name="count"/> options it was asked
    /// about. An answer that does not fit its question (a probability per option) is a broken
    /// contract, not a missing answer, and is not guessed around.
    /// </summary>
    private static IReadOnlyList<float> ProbabilitiesOf(ImageIntentChoice choice, int count)
    {
        IReadOnlyList<float> probabilities = choice.Probabilities;
        if (choice.Index < 0 || choice.Index >= count)
            throw new InvalidOperationException($"The image model chose option {choice.Index} of {count}.");
        if (probabilities.Count != count || probabilities.Any(p => !float.IsFinite(p) || p < 0))
            throw new InvalidOperationException($"The image model gave {probabilities.Count} probabilities for {count} options.");
        return probabilities;
    }

    /// <summary>
    /// The reading the user chose on the page. The picture it names is used when the
    /// conversation still has it; otherwise the newest that fits, and a new picture when none does.
    /// </summary>
    private static ImagePlan Asked(string intent, ImageMessage newest, IReadOnlyList<Picture> pictures, string uploadDirectory)
    {
        Picture? named = newest.Source is { } source ? pictures.LastOrDefault(p => p.Name == source) : null;
        if (intent == "edit" && (named ?? pictures.LastOrDefault()) is { } target)
            return ImagePlan.Change(target, newest.Text, "asked");
        if (intent == "again"
            && (named is not null && CanMakeAgain(named, uploadDirectory) ? named : pictures.LastOrDefault(p => CanMakeAgain(p, uploadDirectory))) is { } original)
            return ImagePlan.Again(original, "asked");
        return ImagePlan.New(newest.Text, "asked");
    }

    /// <summary>
    /// The readings, from structure alone: change each of the newest
    /// <see cref="MaxPicturesOffered"/> pictures (newest first), draw a new one, and, when the
    /// newest picture is one the model made and can make again, another version of it. The
    /// question about what to do offers the changes as one option (<see cref="KindsOf"/>).
    /// </summary>
    internal static IReadOnlyList<ImageOption> OptionsFor(IReadOnlyList<Picture> pictures, string uploadDirectory)
    {
        var options = new List<ImageOption>();
        for (int i = pictures.Count - 1; i >= 0 && options.Count < MaxPicturesOffered; i--)
            options.Add(new ImageOption("edit", pictures[i].Name));
        options.Add(new ImageOption("new", null));
        if (pictures.Count > 0 && CanMakeAgain(pictures[^1], uploadDirectory))
            options.Add(new ImageOption("again", pictures[^1].Name));
        return options;
    }

    /// <summary>
    /// The buttons a question offers: one per kind of reading, not one per picture. The picture
    /// to change is the one the model found likeliest to be changed, otherwise the newest.
    /// </summary>
    private static IReadOnlyList<ImageOption> ChoicesFor(IReadOnlyList<ImageOption> options, ImageOption? edit, IReadOnlyList<Picture> pictures)
    {
        var choices = new List<ImageOption>
        {
            new("edit", edit?.Source ?? pictures[^1].Name),
            new("new", null),
        };
        if (options.FirstOrDefault(o => o.Intent == "again") is { } again)
            choices.Add(again);
        return choices;
    }

    /// <summary>Whether <paramref name="picture"/> can be made again: the model made it from
    /// words it recorded, and every file it was made from is still there.</summary>
    private static bool CanMakeAgain(Picture picture, string uploadDirectory) =>
        picture.Recipe is { Prompt.Length: > 0 } recipe
        && recipe.Sources.All(source => UploadName(source) is { } name && Kept(uploadDirectory, name))
        && (!recipe.Mask.TryGetValue("maskPath", out JsonElement mask)
            || (mask.ValueKind == JsonValueKind.String && UploadName(mask.GetString()) is { } maskName && Kept(uploadDirectory, maskName)));

    // ---- the question --------------------------------------------------------------

    /// <summary>
    /// What the model is told: what each kind of answer does, so that it can tell which one
    /// a message asks for. It says nothing about how any request is worded.
    /// </summary>
    internal const string PlannerInstruction =
        "You decide what an assistant that makes and changes pictures does with the user's newest message. "
        + "It can change one of the pictures already in the conversation, draw a new picture, or make another version of a picture it made. "
        + "Changing a picture keeps that picture and alters it as the newest message says. "
        + "A new picture is drawn from the newest message alone: nothing from earlier pictures or messages carries over. "
        + "Another version repeats the request that made a picture with a different random seed, and the newest message's words are not used. "
        + "Choose the option that does what the user most likely wants.";

    /// <summary>The longest quotation of a message or a picture's words before the question has to be shortened.</summary>
    private const int QuoteLimit = 160;

    /// <summary>How short a picture's quotation becomes when the question is still too long without the older messages.</summary>
    private const int ShortQuoteLimit = 40;

    /// <summary>The shortest the newest message is cut to, head and tail kept.</summary>
    private const int NewestFloor = 64;

    /// <summary>
    /// The question about what to do with the newest message, for the readings
    /// <paramref name="options"/> offer: one option per kind (<see cref="KindsOf"/>), the
    /// change naming every picture it could be of. One option per picture would give a change
    /// more of whatever share an answer spreads over the options it cannot tell apart.
    /// </summary>
    internal static PlanQuestion Question(ImageConversation conversation, IReadOnlyList<ImageOption> options) =>
        Compose(conversation, options, number => KindsOf(options).Select(kind => kind switch
        {
            "edit" => "Change picture " + Listed(options.Where(o => o.Intent == "edit").Select(o => $"[{number[o.Source!]}]").ToArray()),
            "again" => $"Make another version of picture [{number[options.First(o => o.Intent == "again").Source!]}] from the same request",
            _ => "Make a new picture from the newest message alone",
        }).ToArray(), "What should the assistant do with the newest message?");

    /// <summary>
    /// The question that follows when the turn may change a picture and more than one is
    /// offered: which, one option per picture, newest first.
    /// </summary>
    internal static PlanQuestion PictureQuestion(ImageConversation conversation, IReadOnlyList<ImageOption> options) =>
        Compose(conversation, options,
            number => options.Where(o => o.Intent == "edit").Select(o => $"Change picture [{number[o.Source!]}]").ToArray(),
            "The assistant will change one of the pictures as the newest message says. Which picture should it change?");

    /// <summary>"[2] or [1]", "[3], [2] or [1]".</summary>
    private static string Listed(IReadOnlyList<string> items) =>
        items.Count == 1 ? items[0] : string.Join(", ", items.Take(items.Count - 1)) + " or " + items[^1];

    /// <summary>
    /// A question about the readings <paramref name="options"/> offer. Pictures are numbered by
    /// where they appear in the conversation and retold with what made them; the user's words
    /// are quoted; then <paramref name="asking"/>, and the options' texts, which
    /// <paramref name="texts"/> writes from those numbers, for the image model to letter. When
    /// the prompt it scores is too long for <see cref="QuestionTokenLimit"/>, the oldest
    /// messages and pictures that are not options go first, then the options' quotations are
    /// cut short, then the newest message loses its middle.
    /// </summary>
    private static PlanQuestion Compose(ImageConversation conversation, IReadOnlyList<ImageOption> options,
        Func<IReadOnlyDictionary<string, int>, string[]> texts, string asking)
    {
        var number = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Picture picture in conversation.Pictures)
            number[picture.Name] = number.Count + 1;
        var offered = new HashSet<string>(options.Where(o => o.Intent == "edit").Select(o => o.Source!), StringComparer.Ordinal);
        Dictionary<string, Picture> byName = conversation.Pictures.ToDictionary(p => p.Name, StringComparer.Ordinal);

        var lines = new List<Line>();
        var introduced = new HashSet<string>(StringComparer.Ordinal);
        IReadOnlyList<EarlierMessage> earlier = conversation.Earlier;
        for (int i = 0; i < earlier.Count; i++)
        {
            EarlierMessage message = earlier[i];
            if (message.Made is { } made)
            {
                lines.Add(MadeLine(byName[made], number, droppable: !offered.Contains(made)));
                continue;
            }
            foreach (string name in message.Attached)
            {
                if (introduced.Add(name) && !byName[name].Made)
                    lines.Add(new Line($"[{number[name]}] a photo the user attached", byName[name].Label, !offered.Contains(name)));
            }
            // The words that made the next picture are quoted with it, once.
            if (string.IsNullOrEmpty(message.Said)
                || (i + 1 < earlier.Count && earlier[i + 1].Made is { } next
                    && byName[next].Recipe is { Plan: not "again" } recipe && recipe.Prompt == message.Said))
                continue;
            string attaching = message.Attached.Count == 0 ? string.Empty
                : ", attaching " + string.Join(" and ", message.Attached.Select(name => $"[{number[name]}]"));
            lines.Add(new Line($"The user said{attaching}", message.Said, Droppable: true));
        }

        string newest = Clean(conversation.Newest.Text);
        string[] optionTexts = texts(number);

        string Render(int newestLimit)
        {
            var user = new StringBuilder("Pictures and messages so far, oldest first:\n");
            foreach (Line line in lines)
                user.Append(line.Render()).Append('\n');
            user.Append("\nThe user's newest message: \"").Append(Shorten(newest, newestLimit)).Append("\"\n\n");
            user.Append(asking);
            return user.ToString();
        }

        // Measured on the whole prompt the image model scores, its option lines and reply
        // format included, not on this side's part of it.
        bool Fits(string user) => EstimateTokens(QwenImageModel.ChooseAnswerPrompt(PlannerInstruction, user, optionTexts)) <= QuestionTokenLimit;

        string question = Render(newest.Length);
        while (!Fits(question) && lines.FindIndex(l => l.Droppable) is var oldest and >= 0)
        {
            lines.RemoveAt(oldest);
            question = Render(newest.Length);
        }
        while (!Fits(question) && lines.FindIndex(l => l.Quote is not null && l.Limit > ShortQuoteLimit) is var quoted and >= 0)
        {
            lines[quoted] = lines[quoted] with { Limit = ShortQuoteLimit };
            question = Render(newest.Length);
        }
        if (!Fits(question) && newest.Length > NewestFloor)
        {
            // The longest cut of the newest message that fits, found by halving.
            int low = NewestFloor, high = newest.Length;
            while (low < high)
            {
                int middle = low + (high - low + 1) / 2;
                if (Fits(Render(middle))) low = middle; else high = middle - 1;
            }
            question = Render(low);
        }
        return new PlanQuestion(PlannerInstruction, question, optionTexts);
    }

    /// <summary>A picture the model made, retold with what made it.</summary>
    private static Line MadeLine(Picture picture, Dictionary<string, int> number, bool droppable)
    {
        string label = $"[{number[picture.Name]}]";
        ImageRecipe recipe = picture.Recipe!;
        if (recipe.Prompt.Length == 0)
            return new Line(label + " a picture the assistant made", null, droppable);
        string Named(string source) =>
            UploadName(source) is { } name && number.TryGetValue(name, out int n) ? $"[{n}]" : "an earlier picture";
        string head = recipe.Plan switch
        {
            "again" => " made again from an earlier request",
            "edit" when recipe.Sources.Count > 1 =>
                $" made by changing {Named(recipe.Sources[0])}, with {string.Join(" and ", recipe.Sources.Skip(1).Select(Named))} for reference, as the user asked",
            "edit" when recipe.Sources.Count == 1 => $" made by changing {Named(recipe.Sources[0])} as the user asked",
            "edit" => " made by changing the selected area as the user asked",
            _ => " drawn from the user's words",
        };
        return new Line(label + head, recipe.Prompt, droppable);
    }

    /// <summary>One line of the retold conversation: a head, then a quotation of at most <see cref="Limit"/> characters.</summary>
    private sealed record Line(string Head, string? Quote, bool Droppable)
    {
        public int Limit { get; init; } = QuoteLimit;

        public string Render() => Quote is null ? Head : $"{Head}: \"{Shorten(Clean(Quote), Limit)}\"";
    }

    /// <summary>
    /// Text as the question quotes it: on one line, and without the chat template's
    /// <c>&lt;|</c> and <c>|&gt;</c>, so that nothing a user typed or named a file can be read
    /// as the template's structure.
    /// </summary>
    internal static string Clean(string? text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;
        var clean = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text.Replace("<|", string.Empty, StringComparison.Ordinal).Replace("|>", string.Empty, StringComparison.Ordinal))
        {
            if (char.IsWhiteSpace(c))
            {
                space = clean.Length > 0;
                continue;
            }
            if (space)
                clean.Append(' ');
            space = false;
            clean.Append(c);
        }
        return clean.ToString();
    }

    /// <summary>
    /// <paramref name="text"/> in at most <paramref name="limit"/> characters, its head and
    /// tail kept and the middle elided: a request's subject tends to come first and the change
    /// asked for last. Never splits a surrogate pair.
    /// </summary>
    internal static string Shorten(string text, int limit)
    {
        if (text.Length <= limit)
            return text;
        const string gap = " … ";
        int keep = Math.Max(2, limit - gap.Length);
        int head = keep / 2;
        int tail = text.Length - (keep - head);
        if (char.IsHighSurrogate(text[head - 1]))
            head--;
        if (char.IsLowSurrogate(text[tail]))
            tail++;
        return string.Concat(text.AsSpan(0, head), gap, text.AsSpan(tail));
    }

    /// <summary>
    /// How many tokens <paramref name="text"/> takes, estimated from above: a third of a token
    /// per ASCII character, a whole one per digit (Qwen's tokenizer splits numbers into single
    /// digits), and half a token per UTF-8 byte otherwise. Only the scorer holds the tokenizer;
    /// this keeps what it is handed under its limit. Against the Qwen3-VL-8B-Instruct GGUF's own
    /// tokenizer it was never below the real count for any sample tried, and at most 2.8 times
    /// it: "make it brighter" 6 for 3, "把背景换成海滩" 11 for 4, <see cref="PlannerInstruction"/>
    /// 214 for 124, digits and a run of punctuation exactly, and Japanese, Korean, German, Thai,
    /// emoji and a URL in between.
    /// </summary>
    internal static int EstimateTokens(string text)
    {
        double tokens = 0;
        foreach (Rune rune in text.EnumerateRunes())
            tokens += !rune.IsAscii ? rune.Utf8SequenceLength / 2.0 : Rune.IsDigit(rune) ? 1 : 1.0 / 3;
        return (int)Math.Ceiling(tokens);
    }

    // ---- reading the messages --------------------------------------------------------

    /// <summary>
    /// What a picture saved before turns recorded it was made from. Then the request was the
    /// user message before it and nothing else: its photos or selection made it an edit of
    /// those photos, otherwise it was drawn from the words. That is exactly what the turn did
    /// then, so for such a picture this reading is exact.
    /// </summary>
    private static ImageRecipe Reconstructed(JsonElement? asked)
    {
        if (asked is not { } user)
            return new ImageRecipe("new", Array.Empty<string>(), string.Empty, 0, NoMask);
        string prompt = Text(user, "content")?.Trim() ?? string.Empty;
        string[] photos = Paths(user, "stillImagePaths");
        IReadOnlyDictionary<string, JsonElement> mask = MaskOf(user);
        return photos.Length > 0 || mask.Count > 0
            ? new ImageRecipe("edit", photos, prompt, 0, mask)
            : new ImageRecipe("new", Array.Empty<string>(), prompt, 0, NoMask);
    }

    /// <summary>What an assistant message recorded its picture was made from, or null when it recorded nothing.</summary>
    private static ImageRecipe? RecipeOf(JsonElement message)
    {
        string? plan = Text(message, "imagePlan");
        if (plan is not ("edit" or "new" or "again"))
            return null;
        long seed = message.TryGetProperty("imageSeed", out JsonElement s) && s.ValueKind == JsonValueKind.Number
            && s.TryGetInt64(out long value) ? value : 0;
        IReadOnlyDictionary<string, JsonElement> mask =
            message.TryGetProperty("imageMask", out JsonElement m) && m.ValueKind == JsonValueKind.Object ? MaskOf(m) : NoMask;
        return new ImageRecipe(plan, Paths(message, "imageSources"), Text(message, "imagePrompt") ?? string.Empty, seed, mask);
    }

    /// <summary>The selection settings on <paramref name="element"/>, as given.</summary>
    private static IReadOnlyDictionary<string, JsonElement> MaskOf(JsonElement element)
    {
        Dictionary<string, JsonElement>? mask = null;
        foreach (string name in MaskSettings)
        {
            if (element.TryGetProperty(name, out JsonElement value) && value.ValueKind != JsonValueKind.Null)
                (mask ??= new Dictionary<string, JsonElement>(StringComparer.Ordinal))[name] = value.Clone();
        }
        return mask ?? NoMask;
    }

    /// <summary>The name the user knows an attached photo by (its chip), when the message kept one.</summary>
    private static string? LabelOf(JsonElement message, string still)
    {
        if (!message.TryGetProperty("attachments", out JsonElement attachments) || attachments.ValueKind != JsonValueKind.Array)
            return null;
        foreach (JsonElement attachment in attachments.EnumerateArray())
        {
            if (attachment.ValueKind == JsonValueKind.Object && Text(attachment, "file") == still
                && Text(attachment, "fileName") is { Length: > 0 } label)
                return label;
        }
        return null;
    }

    /// <summary>An upload reference as the bare name the uploads folder keeps it under.</summary>
    private static string? UploadName(string? reference)
    {
        if (string.IsNullOrWhiteSpace(reference))
            return null;
        string name = Path.GetFileName(reference);
        return name.Length == 0 || name is "." or ".." ? null : name;
    }

    /// <summary>The upload name of a picture the image service made (<c>/uploads/&lt;name&gt;</c>), or null.</summary>
    private static string? PictureName(string? url)
    {
        const string prefix = "/uploads/";
        if (url is null || !url.StartsWith(prefix, StringComparison.Ordinal))
            return null;
        string name = Uri.UnescapeDataString(url[prefix.Length..]);
        return name.Contains('/') || name.Contains('\\') ? null : UploadName(name);
    }

    /// <summary>The URL the page loads an upload from, as the image service writes it.</summary>
    private static string UploadUrl(string reference) => "/uploads/" + Uri.EscapeDataString(UploadName(reference) ?? reference);

    private static bool Kept(string uploadDirectory, string name) => File.Exists(Path.Combine(uploadDirectory, name));
}
