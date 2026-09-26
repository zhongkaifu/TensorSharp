// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Maui.Hosting;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// The sandbox switches, and everything else the user gets to decide.
///
/// <para>
/// Two of these are the security surface of the whole app, so they are presented
/// as what they actually control rather than as feature names. "Run code" decides
/// whether the model may execute anything at all; with it off there is no shell,
/// no interpreter and no skill script, and the model is told so rather than
/// discovering it. "Allow network access" decides whether anything that runs may
/// open a socket. Both default to the safe answer — code on, because an agent that
/// cannot act is not an agent, and network off, because a model that can reach the
/// internet from inside a sandbox is a different risk entirely — and neither is
/// ever changed except from here.
/// </para>
/// <para>
/// A change takes effect on the next command the model runs. It used to take effect at
/// the next launch — the code runner and its policy were built once, at startup — and
/// the page said so, which was honest and useless: an iPhone app is not restarted by
/// leaving it, so the real instruction was "force-quit TensorAgent from the app
/// switcher" and nobody does that. The reported symptom was precisely what that
/// produces: network turned on, and <c>curl</c> still answering "network access is
/// disabled by the user". See <see cref="AgentAppHost.ApplySettings"/>.
/// </para>
/// </summary>
public sealed class SettingsPage : ContentPage
{
    private readonly AgentAppHost _app;
    private readonly VerticalStackLayout _body;
    private Label? _engine;

    public SettingsPage(LoopbackWebHost host)
    {
        _app = host.App;
        Title = "Settings";
        BackgroundColor = Theme.Background;

        _body = new VerticalStackLayout { Spacing = 4, Padding = new Thickness(0, 8, 0, 24) };
        Content = new ScrollView { Content = _body, BackgroundColor = Theme.Background };
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        try
        {
            Build();
        }
        catch (Exception ex)
        {
            // Build reads the settings file, sizes the models directory and asks the
            // host to describe the engine. Any of those can throw, and an exception
            // raised here cancels the push this page is being appeared for -- so the
            // user taps Settings and stays on the chat, with nothing said anywhere.
            Console.WriteLine("TensorAgent: the settings screen failed to appear: " + ex);
        }
    }

    /// <summary>
    /// Save a change and hand it to the running app in the same breath.
    ///
    /// <para>
    /// Saving alone is what made "Allow network access" a switch with no effect: the
    /// code runner, the installer and the skill planner all read the file once, at
    /// startup, so the model went on being told "network access is disabled by the
    /// user" until the app was force-quit from the app switcher. The engine line under
    /// the switches is repainted from the host's own answer, so it is evidence rather
    /// than a promise.
    /// </para>
    /// </summary>
    private void Apply(Action<AppSettings> change)
    {
        AppSettings settings = _app.Settings.Load();
        change(settings);
        _app.Settings.Save(settings);
        _app.ApplySettings(settings);
        if (_engine is not null)
            _engine.Text = "Now: " + _app.DescribeEngine();
    }


    private void Build()
    {
        AppSettings settings = _app.Settings.Load();
        _body.Clear();

        _body.Add(Section("Sandbox"));
        _body.Add(Switch(
            "Run code",
            "Let the model run shell commands and scripts. Everything runs inside the app, "
            + "confined to this chat's own folder; it can never write elsewhere on the device.",
            settings.AllowCodeExecution,
            on => Apply(s => s.AllowCodeExecution = on)));

        _body.Add(Switch(
            "Allow network access",
            "Let code the model runs reach the internet, and let it install packages. "
            + "Off by default: with it off, every attempt is refused and the model is told why.",
            settings.AllowNetwork,
            on => Apply(s => s.AllowNetwork = on)));

        // Here rather than under Generation because it decides what the model may do,
        // like the two above. There was no switch at all: delegation was simply on, on
        // a phone where every sub-agent is another conversation held in memory.
        _body.Add(Switch(
            "Sub-agents",
            "Let the model hand self-contained parts of a request to helper agents that run "
            + "on the same model and report back. Each helper is a separate conversation held "
            + "in memory, and while this is on every prompt also declares the tools for it.",
            settings.MultiAgentEnabled,
            on => Apply(s => s.MultiAgentEnabled = on)));

        // It used to say "the next time TensorAgent starts", which on a phone is not an
        // instruction anybody follows -- leaving an app does not restart it -- so the
        // switch read as one that did nothing. All three now take effect at once.
        _body.Add(Note("Changes here take effect straight away: the sandbox on the next command "
            + "the model runs, sub-agents on the next message."));
        _engine = new Label
        {
            Text = "Now: " + DescribeEngineSafely(),
            FontSize = 12,
            TextColor = Theme.Muted,
            Padding = new Thickness(16, 2),
        };
        _body.Add(_engine);

        _body.Add(Section("Generation"));
        int loadedContext = _app.ModelService.ContextTokens;
        int modelContext = _app.ModelService.ModelContextTokens;
        string contextNote = modelContext > loadedContext && loadedContext > 0
            ? "The model supports " + Describe(modelContext) + " input + output tokens; "
                + "TensorAgent currently keeps " + Describe(loadedContext) + " active to fit this device."
            : modelContext > 0 && loadedContext > modelContext
                ? "The model declares " + Describe(modelContext) + " input + output tokens; "
                    + "the configured active window is " + Describe(loadedContext) + " tokens."
                : modelContext > 0
                    ? "The loaded model's input + output context window is " + Describe(modelContext) + " tokens."
                    : loadedContext > 0
                        ? "The active input + output context window is " + Describe(loadedContext) + " tokens."
                        : "The loaded model determines the input + output context window.";
        _body.Add(Ladder("Reply output limit",
            "Maximum NEW tokens requested for one reply — this is not the context-window setting. "
            + contextNote + " A reply uses only what remains after the prompt.",
            settings.MaxTokens, ReplyLengthRungs,
            v => Apply(s => s.MaxTokens = v)));
        _body.Add(Choice("KV cache precision",
            "How the conversation's key/value cache is stored. On a phone this is often "
            + "the larger half of what a loaded model costs, so Q4 buys back more memory "
            + "than any other choice here.",
            settings.KvCacheDtype, KvCacheRungs,
            v => Apply(s => s.KvCacheDtype = v)));
        _body.Add(Note(
            "KV cache precision applies the next time a model is loaded — the cache is "
            + "allocated when the model is. Some models ignore it and always use FP16, "
            + "because their attention cannot read a quantized cache; the engine "
            + "substitutes rather than failing."));
        _body.Add(Stepper("Tool timeout", "Seconds before a command is stopped.",
            settings.ToolTimeoutSeconds, 10, 600, 10,
            v => Apply(s => s.ToolTimeoutSeconds = v)));
        _body.Add(Switch("Show reasoning by default",
            "Start each chat with the model's thinking visible.",
            settings.ThinkByDefault,
            on => { AppSettings s = _app.Settings.Load(); s.ThinkByDefault = on; _app.Settings.Save(s); }));
        // Through Apply like the sandbox switches. It used to save the file and nothing
        // else, so the engine that was standing kept the old policy until the next model
        // load, while AgentAppHost.ApplySpeculationSetting -- written to move the running
        // engine -- was only ever reached when some OTHER switch was flipped.
        _body.Add(Switch("Speculative decoding",
            "Guess a few tokens ahead and check them in one pass: the same answer, faster on "
            + "code and on replies that quote a file. The engine switches it off by itself "
            + "while it is not paying, and some models cannot do it at all. Applies from the "
            + "next reply.",
            settings.SpeculativeDecoding,
            on => Apply(s => s.SpeculativeDecoding = on)));

        _body.Add(Section("Downloads"));
        _body.Add(Switch("Download over cellular",
            "Model files are several gigabytes. Off by default so a download waits for Wi-Fi.",
            settings.AllowCellularDownloads,
            on => { AppSettings s = _app.Settings.Load(); s.AllowCellularDownloads = on; _app.Settings.Save(s); }));
        _body.Add(Switch("Include optional files",
            "The image projector and the speculative-decoding draft head. Larger downloads, but "
            + "without the projector a model cannot see pictures.",
            settings.DownloadOptionalFiles,
            on => { AppSettings s = _app.Settings.Load(); s.DownloadOptionalFiles = on; _app.Settings.Save(s); }));
        // Said here because it is the thing people worry about while a download runs,
        // and the model list can only say it while they are looking at the model list.
        _body.Add(Note(
            "A download keeps going while you use the rest of the app, and for a while "
            + "after you leave it. If the system stops it, it resumes from where it got to "
            + "the next time TensorAgent is open — nothing is fetched twice."));

        _body.Add(Section("Storage"));
        _body.Add(Note($"Models: {Gb(DirectorySize(_app.Paths.ModelsDirectory))} GB"));
        _body.Add(Note($"Chats: {_app.Conversations.List().Count}"));
        _body.Add(Note($"Skills: {_app.Skills.Skills.Count}"));

        var clear = new Button
        {
            Text = "Delete all chats",
            BackgroundColor = Theme.Surface,
            TextColor = Theme.Danger,
            CornerRadius = 10,
            Margin = new Thickness(16, 12, 16, 0),
        };
        clear.Clicked += async (_, _) =>
        {
            if (!await DisplayAlert("Delete all chats", "This cannot be undone.", "Delete", "Cancel"))
                return;
            // includeEmpty: "delete all chats" has to mean all of them, including the
            // untouched one the current session is sitting in.
            foreach (var summary in _app.Conversations.List(includeEmpty: true))
                _app.Conversations.Delete(summary.Id);
            Build();
        };
        _body.Add(clear);
    }

    /// <summary>
    /// The engine line, or why there isn't one.
    ///
    /// <para>
    /// A live call into the host, made while this page is being appeared for a push
    /// that has not completed. AboutPage already wraps the identical call; unguarded
    /// here it could take the whole Settings screen down with it and leave the user on
    /// the chat, which reads as the menu having ignored them.
    /// </para>
    /// </summary>
    private string DescribeEngineSafely()
    {
        try { return _app.DescribeEngine(); }
        catch (Exception ex) { return "unavailable (" + ex.GetType().Name + ")"; }
    }

    private static View Section(string text) => new Label
    {
        Text = text.ToUpperInvariant(),
        FontSize = 12,
        TextColor = Theme.Muted,
        FontAttributes = FontAttributes.Bold,
        Padding = new Thickness(16, 20, 16, 6),
    };

    private static View Note(string text) => new Label
    {
        Text = text,
        FontSize = 12,
        TextColor = Theme.Muted,
        Padding = new Thickness(16, 2),
    };

    private static View Switch(string title, string detail, bool value, Action<bool> onChanged)
    {
        var toggle = new Microsoft.Maui.Controls.Switch { IsToggled = value, OnColor = Theme.Accent, VerticalOptions = LayoutOptions.Center };
        toggle.Toggled += (_, e) => onChanged(e.Value);

        var grid = new Grid
        {
            ColumnDefinitions = { new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto) },
            Padding = new Thickness(16, 10),
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        }, 0, 0);
        grid.Add(toggle, 1, 0);
        return grid;
    }

    /// <summary>
    /// The reply-length rungs, up to 256K tokens.
    ///
    /// <para>
    /// A plain Stepper cannot express this range: 256 to 262,144 in steps of 256 is a
    /// thousand taps. The rungs double instead, so the whole range is eleven taps and
    /// the useful small values keep their resolution.
    /// </para>
    ///
    /// <para>
    /// The ceiling that actually applies is the CONTEXT, not this number:
    /// ChatGenerationPipeline.ClampGenerationReserve trims the reserve to what the
    /// window leaves after the prompt, so asking for 256K inside an 8,192-token context
    /// yields at most 8,192 minus the prompt. Raising this is what lets a long context
    /// be spent on one reply; it does not create context.
    /// </para>
    /// </summary>
    private static readonly int[] ReplyLengthRungs =
        { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };

    /// <summary>
    /// The K/V cache precisions, in the order the stepper walks them.
    ///
    /// <para>
    /// Widest first, so stepping right spends less memory — the same direction as every
    /// other stepper on this page. The stored values are the engine's own spellings;
    /// the labels are what the user is likely to have seen on a model card.
    /// </para>
    /// </summary>
    private static readonly (string Value, string Label)[] KvCacheRungs =
    {
        ("f16", "FP16"),
        ("q8_0", "Q8"),
        ("q4_0", "Q4"),
    };

    /// <summary>
    /// One of a short list of named values, on the same stepper the numbers use.
    ///
    /// <para>
    /// A Picker would open a modal wheel for three options. The stepper is already the
    /// page's idiom for "walk a small ordered range", and these ARE ordered: each step
    /// right halves the memory and loses a little precision.
    /// </para>
    /// </summary>
    private static View Choice(
        string title, string detail, string value,
        (string Value, string Label)[] options, Action<string> onChanged)
    {
        int index = 0;
        for (int i = 0; i < options.Length; i++)
        {
            if (string.Equals(options[i].Value, value, StringComparison.OrdinalIgnoreCase)) index = i;
        }

        var current = new Label
        {
            Text = options[index].Label,
            FontSize = 15,
            TextColor = Theme.Accent,
            VerticalOptions = LayoutOptions.Center,
        };
        var stepper = new Stepper(0, options.Length - 1, index, 1) { VerticalOptions = LayoutOptions.Center };
        stepper.ValueChanged += (_, e) =>
        {
            (string Value, string Label) picked = options[Math.Clamp((int)e.NewValue, 0, options.Length - 1)];
            current.Text = picked.Label;
            onChanged(picked.Value);
        };

        var grid = new Grid
        {
            ColumnDefinitions =
            {
                new ColumnDefinition(GridLength.Star),
                new ColumnDefinition(GridLength.Auto),
                new ColumnDefinition(GridLength.Auto),
            },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        });
        grid.Add(current, 1, 0);
        grid.Add(stepper, 2, 0);
        return grid;
    }

    private static View Ladder(string title, string detail, int value, int[] rungs, Action<int> onChanged)
    {
        int index = 0;
        for (int i = 0; i < rungs.Length; i++)
        {
            if (rungs[i] <= value) index = i;
        }

        var current = new Label
        {
            Text = Describe(rungs[index]),
            FontSize = 15,
            TextColor = Theme.Accent,
            VerticalOptions = LayoutOptions.Center,
        };
        var stepper = new Stepper(0, rungs.Length - 1, index, 1) { VerticalOptions = LayoutOptions.Center };
        stepper.ValueChanged += (_, e) =>
        {
            int chosen = rungs[Math.Clamp((int)e.NewValue, 0, rungs.Length - 1)];
            current.Text = Describe(chosen);
            onChanged(chosen);
        };

        var grid = new Grid
        {
            ColumnDefinitions =
            {
                new ColumnDefinition(GridLength.Star),
                new ColumnDefinition(GridLength.Auto),
                new ColumnDefinition(GridLength.Auto),
            },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        });
        grid.Add(current, 1, 0);
        grid.Add(stepper, 2, 0);
        return grid;
    }

    /// <summary>"1024" is harder to read at a glance than "1K"; the rungs are all
    /// powers of two, so the short form is exact rather than rounded.</summary>
    private static string Describe(int tokens) =>
        tokens >= 1024 && tokens % 1024 == 0 ? (tokens / 1024) + "K" : tokens.ToString();

    private static View Stepper(string title, string detail, int value, int min, int max, int step, Action<int> onChanged)
    {
        var current = new Label { Text = value.ToString(), FontSize = 15, TextColor = Theme.Accent, VerticalOptions = LayoutOptions.Center };
        var stepper = new Stepper(min, max, Math.Clamp(value, min, max), step) { VerticalOptions = LayoutOptions.Center };
        stepper.ValueChanged += (_, e) =>
        {
            int v = (int)e.NewValue;
            current.Text = v.ToString();
            onChanged(v);
        };

        var grid = new Grid
        {
            ColumnDefinitions =
            {
                new ColumnDefinition(GridLength.Star),
                new ColumnDefinition(GridLength.Auto),
                new ColumnDefinition(GridLength.Auto),
            },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        }, 0, 0);
        grid.Add(current, 1, 0);
        grid.Add(stepper, 2, 0);
        return grid;
    }

    private static long DirectorySize(string path)
    {
        try
        {
            long total = 0;
            foreach (string file in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories))
                total += new FileInfo(file).Length;
            return total;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return 0;
        }
    }

    private static string Gb(long bytes) => (bytes / 1e9).ToString("0.00");
}
