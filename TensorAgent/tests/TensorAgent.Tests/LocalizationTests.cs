// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Globalization;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;
using TensorAgent.Core.Shell;
using TensorAgent.Sharing;
using TensorAgent.Sharing.Localization;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// The process has one interface language (<see cref="Loc"/>), so the tests that change it
/// run alone: a page or route test running beside one would read another language's text.
/// Each puts English back as it ends.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class UiLanguageCollection
{
    public const string Name = "UI language";
}

/// <summary>
/// The interface's languages: which one a user gets, that every translation is complete
/// and keeps the English text's placeholders, that every key the code asks for exists and
/// every key that exists is asked for, and that the choice is saved and reaches the page.
/// </summary>
[Collection(UiLanguageCollection.Name)]
public sealed class LocalizationTests : IDisposable
{
    private static readonly string Repo = FindRepoRoot();
    private readonly Func<IReadOnlyList<string>> _systemLanguages = Loc.SystemLanguages;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-i18n-" + Guid.NewGuid().ToString("N"));

    public LocalizationTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        Loc.SystemLanguages = _systemLanguages;
        Loc.Use(UiLanguages.English);
        try { Directory.Delete(_root, true); } catch { }
    }

    [Fact]
    public void LanguageChangesRefreshVisiblePagesWithoutMutatingTheDisabledNativeFlyout()
    {
        // This project cannot host UIKit. Guard the application-side trigger from the
        // native crash report: ShellContent.Title -> BaseShellItem.PropertyChanged ->
        // UIContainerCell.ReloadRow -> NSArray containing a null IndexPath. The native
        // language validation tool also exercises the real picker and navigation.
        string appRoot = Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Maui");
        string shell = File.ReadAllText(Path.Combine(appRoot, "AppShell.cs"));
        Assert.Contains("FlyoutBehavior = FlyoutBehavior.Disabled", shell);
        Assert.DoesNotMatch(@"\bLoc\s*\.\s*Changed\s*\+=", shell);
        Assert.DoesNotMatch(@"\bmain\s*\.\s*(?:Title\s*=|SetBinding\s*\()", shell);

        // Removing the faulty hidden-cell refresh must not remove live localization
        // from the destinations that the user actually sees.
        foreach (string page in new[] { "MainPage.cs", "Pages/SessionsPage.cs", "Pages/ModelsPage.cs", "Pages/SettingsPage.cs", "Pages/AboutPage.cs" })
        {
            string source = File.ReadAllText(Path.Combine(appRoot, page));
            Assert.Matches(@"\bLoc\s*\.\s*Changed\s*\+=", source);
        }
    }

    // =====================================================================================
    // which language
    // =====================================================================================

    [Theory]
    [InlineData("en", "en")]
    [InlineData("en-GB", "en")]
    [InlineData("EN-us", "en")]
    [InlineData("zh-Hans-CN", "zh-Hans")]
    [InlineData("zh-CN", "zh-Hans")]
    [InlineData("zh_CN", "zh-Hans")]
    [InlineData("zh-SG", "zh-Hans")]
    [InlineData("zh", "zh-Hans")]
    [InlineData("zh-Hant-TW", "zh-Hant")]
    [InlineData("zh-TW", "zh-Hant")]
    [InlineData("zh-HK", "zh-Hant")]
    [InlineData("zh-MO", "zh-Hant")]
    [InlineData("zh-Hant", "zh-Hant")]
    [InlineData("yue-Hant-HK", "zh-Hant")]
    [InlineData("ja-JP", "ja")]
    [InlineData("ko-KR", "ko")]
    [InlineData("es-MX", "es")]
    [InlineData("fr-CA", "fr")]
    [InlineData("de-AT", "de")]
    public void ASystemOrSavedTagFindsTheLanguageItAsksFor(string tag, string expected) =>
        Assert.Equal(expected, UiLanguages.Match(tag)?.Tag);

    [Theory]
    [InlineData("pt-BR")]
    [InlineData("ru")]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(null)]
    [InlineData("-")]
    public void ALanguageWithNoTablesFindsNothing(string? tag) => Assert.Null(UiLanguages.Match(tag));

    [Fact]
    public void TheUsersChoiceWinsThenTheFirstSupportedSystemLanguageThenEnglish()
    {
        Assert.Equal("ja", UiLanguages.Resolve("ja", new[] { "zh-Hans-CN" }).Tag);
        // The first SUPPORTED one: a Portuguese speaker who also reads Chinese gets Chinese.
        Assert.Equal("zh-Hans", UiLanguages.Resolve("", new[] { "pt-BR", "zh-Hans-CN", "en-US" }).Tag);
        // A choice this build has no tables for (a newer build may have saved it) follows the system.
        Assert.Equal("zh-Hant", UiLanguages.Resolve("pt-BR", new[] { "zh-Hant-TW" }).Tag);
        Assert.Equal("en", UiLanguages.Resolve(null, new[] { "pt-BR", "ru-RU" }).Tag);
        Assert.Equal("en", UiLanguages.Resolve(null, null).Tag);
    }

    [Fact]
    public void AFirstLaunchFollowsTheSystemAndASavedChoiceWinsFromThenOn()
    {
        Loc.SystemLanguages = () => new[] { "ja-JP", "en-US" };

        Assert.Equal("ja", Loc.Apply(new AppSettings()).Tag);
        Assert.Equal("ja", Loc.Language.Tag);
        Assert.Equal(Table("ja", "settings")["settings.language.title"], Loc.T("settings.language.title"));

        Assert.Equal("zh-Hans", Loc.Apply(new AppSettings { UiLanguage = "zh-Hans" }).Tag);
        Assert.Equal(Table("zh-Hans", "settings")["settings.language.title"], Loc.T("settings.language.title"));
        Assert.Equal("zh-Hans", Loc.Culture.Name);
        // The app writes the drafts of shared items itself, with the share strings: they follow.
        Assert.Equal("zh-Hans", ShareStrings.Language.Tag);
        Assert.Equal("zh-Hans", ShareStrings.Culture.Name);
    }

    [Fact]
    public void ChangedIsRaisedWhenTheLanguageMovesAndNotWhenItStays()
    {
        int changes = 0;
        void Count() => changes++;
        Loc.Changed += Count;
        try
        {
            Loc.Apply(new AppSettings { UiLanguage = "en" });
            Assert.Equal(0, changes);
            Loc.Apply(new AppSettings { UiLanguage = "de" });
            Assert.Equal(1, changes);
            Loc.Apply(new AppSettings { UiLanguage = "de" });
            Assert.Equal(1, changes);
        }
        finally
        {
            Loc.Changed -= Count;
        }
    }

    [Fact]
    public void TheChoiceStartsEmptyAndIsSaved()
    {
        string path = Path.Combine(_root, "settings.json");
        var store = new SettingsStore(path);
        Assert.Equal(string.Empty, store.Load().UiLanguage);

        AppSettings settings = store.Load();
        settings.UiLanguage = "zh-Hans";
        store.Save(settings);

        Assert.Contains("\"uiLanguage\"", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal("zh-Hans", new SettingsStore(path).Load().UiLanguage);
    }

    [Theory]
    [InlineData("en")]
    [InlineData("zh-Hans")]
    [InlineData("zh-Hant")]
    [InlineData("ja")]
    [InlineData("ko")]
    [InlineData("es")]
    [InlineData("fr")]
    [InlineData("de")]
    public void TheRunningHostAppliesTheChoiceAndKeepsItAcrossLaunches(string choice)
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"))
        {
            ExecutionMode = AgentExecutionMode.InProcess,
        };
        Loc.SystemLanguages = () => new[] { "ja-JP", "en-US" };
        try
        {
            using (var first = new AgentAppHost(paths))
            {
                Assert.Equal("ja", Loc.Language.Tag);
                AppSettings settings = first.Settings.Load();
                Assert.Equal(string.Empty, settings.UiLanguage);
                settings.UiLanguage = choice;
                first.Settings.Save(settings);
                first.ApplySettings(settings);
                Assert.Equal(choice, Loc.Language.Tag);
                Assert.Equal(choice, ShareStrings.Language.Tag);
            }

            Loc.Use(UiLanguages.English);
            Loc.SystemLanguages = () => new[] { "fr-FR" };
            using (var next = new AgentAppHost(paths))
            {
                Assert.Equal(choice, next.Settings.Load().UiLanguage);
                Assert.Equal(choice, Loc.Language.Tag);
                Assert.Equal(choice, ShareStrings.Language.Tag);
                // Returning to System saves that choice too, rather than the current
                // resolved tag. A later launch then follows a changed system language.
                AppSettings settings = next.Settings.Load();
                settings.UiLanguage = string.Empty;
                next.Settings.Save(settings);
                next.ApplySettings(settings);
                Assert.Equal("fr", Loc.Language.Tag);
            }

            Loc.SystemLanguages = () => new[] { "zh-CN" };
            using var system = new AgentAppHost(paths);
            Assert.Equal(string.Empty, system.Settings.Load().UiLanguage);
            Assert.Equal("zh-Hans", Loc.Language.Tag);
        }
        finally
        {
            CodeEnvironment.Reset();
        }
    }

    /// <summary>
    /// The app writes the draft of what was shared (<see cref="ShareComposer"/>, called by
    /// the share importer) in its own process, with the share strings: the draft, its
    /// default question and its labels follow the language the app shows, not English.
    /// </summary>
    [Fact]
    public void TheAppWritesTheDraftOfAShareInTheInterfaceLanguage()
    {
        var payload = new SharePayload { Id = ShareIds.New(), Items = { ShareItem.ForText("the shared words") } };
        string english = ShareComposer.Compose(payload).Message;
        Assert.StartsWith("Summarize this and tell me what matters in it.", english, StringComparison.Ordinal);

        Loc.Apply(new AppSettings { UiLanguage = "de" });
        IReadOnlyDictionary<string, string> german = ShareStrings.Strings.LoadFiles("de")["share"];
        string message = ShareComposer.Compose(payload).Message;
        Assert.StartsWith(german["share.message.prompt.text"], message, StringComparison.Ordinal);
        Assert.Contains(german["share.message.text"], message, StringComparison.Ordinal);
        Assert.Contains("the shared words", message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheShareExtensionReadsTheChoiceTheAppLeftAndResolvesItTheSameWay()
    {
        Assert.Null(ShareStrings.ReadChoice(_root));
        Assert.Equal("ko", ShareStrings.UseChoiceIn(_root, new[] { "ko-KR" }).Tag);

        ShareStrings.WriteChoice(_root, "fr");
        Assert.Equal("fr", ShareStrings.ReadChoice(_root));
        Assert.Equal("fr", ShareStrings.UseChoiceIn(_root, new[] { "ko-KR" }).Tag);

        // "Follow the system" is written as nothing, and means the system in the extension too.
        ShareStrings.WriteChoice(_root, string.Empty);
        Assert.Equal("ko", ShareStrings.UseChoiceIn(_root, new[] { "ko-KR" }).Tag);
        Assert.Equal(UiLanguages.Resolve(string.Empty, new[] { "ko-KR" }), ShareStrings.Language);
    }

    // =====================================================================================
    // text
    // =====================================================================================

    [Fact]
    public void PlaceholdersAreFilledByNameAndDoubledBracesStayLiteral()
    {
        CultureInfo invariant = CultureInfo.InvariantCulture;
        Assert.Equal("3 of 5", StringCatalog.Format("{done} of {total}", new (string, object?)[] { ("done", 3), ("total", 5) }, invariant));
        // An unknown name shows, rather than vanishing.
        Assert.Equal("{name} stays", StringCatalog.Format("{name} stays", Array.Empty<(string, object?)>(), invariant));
        Assert.Equal("{literal}", StringCatalog.Format("{{literal}}", Array.Empty<(string, object?)>(), invariant));
        Assert.Equal(new[] { "count", "name" }, StringCatalog.Placeholders("{count} {name} {{not}}").OrderBy(n => n, StringComparer.Ordinal));
        // Numbers in the language's own format (grouping is the caller's to ask for).
        Assert.Equal("1234.5", StringCatalog.Format("{n}", new (string, object?)[] { ("n", 1234.5) }, CultureInfo.GetCultureInfo("en")));
        Assert.Equal("1234,5", StringCatalog.Format("{n}", new (string, object?)[] { ("n", 1234.5) }, CultureInfo.GetCultureInfo("de")));
    }

    [Theory]
    [InlineData("en", 1, "one")]
    [InlineData("en", 0, "other")]
    [InlineData("en", 2, "other")]
    [InlineData("de", 1, "one")]
    [InlineData("es", 1, "one")]
    [InlineData("es", 3, "other")]
    [InlineData("fr", 0, "one")]
    [InlineData("fr", 1, "one")]
    [InlineData("fr", 2, "other")]
    [InlineData("zh-Hans", 1, "other")]
    [InlineData("zh-Hant", 1, "other")]
    [InlineData("ja", 1, "other")]
    [InlineData("ko", 1, "other")]
    public void CountedTextFollowsEachLanguagesPluralRule(string tag, long count, string expected) =>
        Assert.Equal(expected, StringCatalog.PluralCategory(tag, count));

    /// <summary>
    /// A chat with nothing typed in it is titled by its date. The title is stored in English,
    /// as it always was, and shown in the interface language with that language's own way
    /// of writing a date; a title the user wrote is shown as written, whatever the language.
    /// </summary>
    [Fact]
    public void AChatsDateTitleIsShownInTheInterfaceLanguage()
    {
        var local = new DateTime(2026, 10, 2, 14, 30, 0, DateTimeKind.Unspecified);
        var created = new DateTimeOffset(local, TimeZoneInfo.Local.GetUtcOffset(local));
        string stored = Conversation.DeriveTitle(Array.Empty<StoredMessage>(), created);

        Assert.Equal("Chat Oct 2, 14:30", Conversation.DisplayTitle(stored, created));

        Loc.Use(UiLanguages.Find("zh-Hans")!);
        string shown = Conversation.DisplayTitle(stored, created);
        string[] around = Table("zh-Hans", "host")["host.conversations.untitled"].Split("{date}");
        Assert.StartsWith(around[0], shown, StringComparison.Ordinal);
        Assert.EndsWith(around[^1], shown, StringComparison.Ordinal);
        Assert.Contains("10月2日", shown, StringComparison.Ordinal);
        Assert.Contains("14:30", shown, StringComparison.Ordinal);
        // The user's own words are never translated.
        Assert.Equal("Plan a trip", Conversation.DisplayTitle("Plan a trip", created));
    }

    [WebJavaScriptFact]
    public void ThePagesRuntimeFormatsCountsAndTranslatesTheMarkupAsTheNativeSideDoes()
    {
        // The page's plural rule is a copy of StringCatalog.PluralCategory in JavaScript;
        // this runs it for every language and count the native rule is tested on.
        var cases = new List<object>();
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            foreach (int count in new[] { 0, 1, 2, 5 })
                cases.Add(new { lang = language.Tag, count, expected = StringCatalog.PluralCategory(language.Tag, count) });
        }
        string dom = File.ReadAllText(Path.Combine(Repo, "TensorAgent", "tests", "TensorAgent.Tests", "PageDom.js"));
        string source = dom + "\n"
            + "var label = document.createElement('span'); label.setAttribute('data-i18n', 'settings.language.title');\n"
            + "var field = document.createElement('input'); field.setAttribute('data-i18n-placeholder', 'settings.language.section');\n"
            + "document.body.appendChild(label); document.body.appendChild(field);\n"
            + PageStrings.Script() + "\n"
            + "var I = window.TensorAgentI18n, cases = " + JsonSerializer.Serialize(cases) + ", got = [];\n"
            + "I.strings['x.n.one'] = 'one:{count}'; I.strings['x.n.other'] = 'other:{count}';\n"
            + "cases.forEach(function (c) { I.lang = c.lang; got.push(I.tn('x.n', c.count).split(':')[0]); });\n"
            + "I.lang = 'en';\n"
            + "console.log('<<RESULT>>' + JSON.stringify({ got: got, thousand: I.tn('x.n', 1000),"
            + " label: label.textContent, placeholder: field.getAttribute('placeholder'),"
            + " filled: I.t('settings.language.system', { language: 'X' }),"
            + " missing: I.t('page.no.such.key'), shared: window.TensorSharpI18n.t('mask.no.such.key', 'Fallback {n}', { n: 2 }),"
            + " lang: document.documentElement.getAttribute('lang') }));\n";

        var policy = new TensorAgent.Core.Sandbox.ExecutionPolicy(
            AllowScripts: true, AllowNetwork: false, WorkRoot: _root,
            ReadableRoots: Array.Empty<string>(), TempRoot: _root)
        {
            DefaultTimeout = TimeSpan.FromSeconds(30),
        };
        var context = new TensorAgent.Core.Sandbox.InterpreterContext(_root, new Dictionary<string, string> { ["HOME"] = _root }, policy);
        TensorAgent.Core.Sandbox.ExecutionResult result = WebJavaScript.Run(source, _root, context);
        Assert.True(result.ExitCode == 0, result.Stdout + result.Stderr);
        int start = result.Stdout.IndexOf("<<RESULT>>", StringComparison.Ordinal);
        Assert.True(start >= 0, result.Stdout + result.Stderr);
        JsonElement page = JsonSerializer.Deserialize<JsonElement>(result.Stdout[(start + "<<RESULT>>".Length)..].Trim());

        Assert.Equal(cases.Select(c => (string)c.GetType().GetProperty("expected")!.GetValue(c)!),
            page.GetProperty("got").EnumerateArray().Select(e => e.GetString()!));
        // A count reads as the native side writes it, ungrouped.
        Assert.Equal("other:1000", page.GetProperty("thousand").GetString());
        Assert.Equal("1000", StringCatalog.Format("{count}", new (string, object?)[] { ("count", 1000L) }, CultureInfo.GetCultureInfo("en")));
        Assert.Equal(Loc.T("settings.language.title"), page.GetProperty("label").GetString());
        Assert.Equal(Loc.T("settings.language.section"), page.GetProperty("placeholder").GetString());
        Assert.Equal(Loc.T("settings.language.system", ("language", "X")), page.GetProperty("filled").GetString());
        Assert.Equal("page.no.such.key", page.GetProperty("missing").GetString());
        Assert.Equal("Fallback 2", page.GetProperty("shared").GetString());
        Assert.Equal("en", page.GetProperty("lang").GetString());
    }

    // =====================================================================================
    // the tables
    // =====================================================================================

    [WebJavaScriptFact]
    public void TheFirstPaintWaitsUntilTheBodyHasBeenTranslated()
    {
        Loc.Use(UiLanguages.Find("zh-Hans")!);
        string source = File.ReadAllText(Path.Combine(Repo, "TensorAgent", "tests", "TensorAgent.Tests", "PageDom.js"))
            + "\nvar label = document.createElement('span'); label.textContent = 'Language'; label.setAttribute('data-i18n', 'settings.language.title'); document.body.appendChild(label);"
            + "\nvar gate = document.getElementById('tensoragent-language-loading'); document.body.appendChild(gate); document.readyState = 'loading';\n"
            + PageStrings.Script()
            + "\nvar before = label.textContent, hidden = !!gate.parentNode; document.readyState = 'interactive'; document.dispatch('DOMContentLoaded');"
            + "\nconsole.log('<<RESULT>>' + JSON.stringify({before: before, hidden: hidden, after: label.textContent, revealed: !gate.parentNode, lang: document.documentElement.getAttribute('lang')}));\n";
        var policy = new TensorAgent.Core.Sandbox.ExecutionPolicy(true, false, _root, Array.Empty<string>(), _root);
        var context = new TensorAgent.Core.Sandbox.InterpreterContext(_root, new Dictionary<string, string> { ["HOME"] = _root }, policy);
        TensorAgent.Core.Sandbox.ExecutionResult result = WebJavaScript.Run(source, _root, context);
        Assert.True(result.ExitCode == 0, result.Stdout + result.Stderr);
        int start = result.Stdout.IndexOf("<<RESULT>>", StringComparison.Ordinal);
        Assert.True(start >= 0, result.Stdout + result.Stderr);
        JsonElement painted = JsonSerializer.Deserialize<JsonElement>(result.Stdout[(start + "<<RESULT>>".Length)..].Trim());
        Assert.Equal("Language", painted.GetProperty("before").GetString());
        Assert.True(painted.GetProperty("hidden").GetBoolean());
        Assert.Equal(Loc.T("settings.language.title"), painted.GetProperty("after").GetString());
        Assert.True(painted.GetProperty("revealed").GetBoolean());
        Assert.Equal("zh-Hans", painted.GetProperty("lang").GetString());
    }

    public static IEnumerable<object[]> Catalogs() => new[] { new object[] { "app" }, new object[] { "share" } };

    private static StringCatalog CatalogFor(string which) => which == "app" ? Loc.Tables : ShareStrings.Strings;

    private static IReadOnlyDictionary<string, string> Table(string tag, string area) => Loc.Tables.LoadFiles(tag)[area];

    [Theory]
    [MemberData(nameof(Catalogs))]
    public void EveryLanguageHasEveryEnglishKeyWithTheSamePlaceholders(string which)
    {
        StringCatalog catalog = CatalogFor(which);
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> english = catalog.LoadFiles("en");
        Assert.NotEmpty(english);

        var problems = new List<string>();
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> files = catalog.LoadFiles(language.Tag);
            foreach (string area in english.Keys.Except(files.Keys))
                problems.Add($"{language.Tag}: no {area}.json");
            foreach (string area in files.Keys.Except(english.Keys))
                problems.Add($"{language.Tag}: {area}.json has no English source");
            foreach ((string area, IReadOnlyDictionary<string, string> source) in english)
            {
                if (!files.TryGetValue(area, out IReadOnlyDictionary<string, string>? translated))
                    continue;
                foreach (string key in source.Keys.Except(translated.Keys))
                    problems.Add($"{language.Tag}/{area}.json: missing {key}");
                foreach (string key in translated.Keys.Except(source.Keys))
                    problems.Add($"{language.Tag}/{area}.json: {key} is not an English key");
                foreach ((string key, string text) in source)
                {
                    if (!translated.TryGetValue(key, out string? local))
                        continue;
                    if (string.IsNullOrWhiteSpace(local))
                        problems.Add($"{language.Tag}/{area}.json: {key} is empty");
                    string expected = string.Join(",", StringCatalog.Placeholders(text).OrderBy(n => n, StringComparer.Ordinal));
                    string actual = string.Join(",", StringCatalog.Placeholders(local).OrderBy(n => n, StringComparer.Ordinal));
                    if (expected != actual)
                        problems.Add($"{language.Tag}/{area}.json: {key} has placeholders [{actual}], English has [{expected}]");
                }
            }
        }
        Assert.True(problems.Count == 0, string.Join(Environment.NewLine, problems));
    }

    [Theory]
    [MemberData(nameof(Catalogs))]
    public void EveryKeyIsFiledUnderTheAreaItsFileIsNamedFor(string which)
    {
        var misfiled = new List<string>();
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            foreach ((string area, IReadOnlyDictionary<string, string> strings) in CatalogFor(which).LoadFiles(language.Tag))
                misfiled.AddRange(strings.Keys.Where(k => !k.StartsWith(area + ".", StringComparison.Ordinal)).Select(k => $"{language.Tag}/{area}.json: {k}"));
        }
        Assert.True(misfiled.Count == 0, string.Join(Environment.NewLine, misfiled));
    }

    [Fact]
    public void EveryLanguageDirectoryIsASupportedLanguageAndNoTableRepeatsAKey()
    {
        var problems = new List<string>();
        foreach (string root in TableRoots())
        {
            foreach (string directory in Directory.EnumerateDirectories(root))
            {
                string tag = Path.GetFileName(directory);
                if (UiLanguages.Find(tag) is not { } language || language.Tag != tag)
                    problems.Add($"{directory}: not a supported language tag");
                foreach (string file in Directory.EnumerateFiles(directory, "*.json"))
                    problems.AddRange(RepeatedKeys(file).Select(k => $"{file}: {k} appears more than once"));
            }
        }
        Assert.True(problems.Count == 0, string.Join(Environment.NewLine, problems));
    }

    /// <summary>
    /// Both directions, as with every other table in this repo held equal to the code. A key
    /// the code asks for that English lacks shows as the key itself on screen; a key nothing
    /// asks for is a translation every language pays for and nobody sees.
    /// </summary>
    [Theory]
    [MemberData(nameof(Catalogs))]
    public void EveryKeyTheCodeAsksForExistsAndEveryKeyIsAskedFor(string which)
    {
        HashSet<string> english = CatalogFor(which).LoadFiles("en").Values.SelectMany(t => t.Keys).ToHashSet(StringComparer.Ordinal);
        (HashSet<string> asked, HashSet<string> plural, HashSet<string> literals) = ScanSources(which);

        var problems = new List<string>();
        foreach (string key in asked.Where(k => !english.Contains(k)))
            problems.Add("asked for, but not in English: " + key);
        foreach (string key in plural)
        {
            foreach (string form in new[] { key + ".one", key + ".other" })
            {
                if (!english.Contains(form))
                    problems.Add("counted, but not in English: " + form);
            }
        }
        foreach (string key in english)
        {
            string? pluralBase = key.EndsWith(".one", StringComparison.Ordinal) || key.EndsWith(".other", StringComparison.Ordinal)
                ? key[..key.LastIndexOf('.')] : null;
            if (!literals.Contains(key) && (pluralBase is null || !literals.Contains(pluralBase)))
                problems.Add("in English, but nothing asks for it: " + key);
        }
        Assert.True(problems.Count == 0, string.Join(Environment.NewLine, problems.OrderBy(p => p, StringComparer.Ordinal)));
    }

    /// <summary>
    /// A translation is laid out as its English file is: the same keys in the same order,
    /// one to a line. A translator reads the two side by side, and a change to the English
    /// shows up at the same line in every language.
    /// </summary>
    [Fact]
    public void EveryTranslationKeepsTheEnglishOrderOneKeyToALine()
    {
        var problems = new List<string>();
        foreach (string root in TableRoots())
        {
            foreach (string directory in Directory.EnumerateDirectories(root))
            {
                foreach (string file in Directory.EnumerateFiles(directory, "*.json"))
                {
                    string text = File.ReadAllText(file);
                    List<string> keys = KeysInOrder(file);
                    string[] lines = text.Split('\n');
                    if (text.Contains('\r') || !text.EndsWith("}\n", StringComparison.Ordinal))
                        problems.Add($"{file}: not LF line endings with a final newline");
                    else if (keys.Count > 0 && lines.Length != keys.Count + 3)
                        problems.Add($"{file}: {keys.Count} keys on {lines.Length - 3} lines; one key to a line");
                    string english = Path.Combine(root, "en", Path.GetFileName(file));
                    if (Path.GetFileName(directory) == "en" || !File.Exists(english))
                        continue;
                    List<string> order = KeysInOrder(english).Where(keys.Contains).ToList();
                    if (!order.SequenceEqual(keys.Where(order.Contains)))
                        problems.Add($"{file}: keys are not in the English file's order");
                }
            }
        }
        Assert.True(problems.Count == 0, string.Join(Environment.NewLine, problems));
    }

    /// <summary>
    /// The page's markup is shown first and the page script then paints its own key over
    /// these elements, and a menu row names the screen it opens: in every language the two
    /// must read the same, or the text changes in front of the user.
    /// </summary>
    private static readonly (string First, string Then)[] SameWords =
    {
        ("markup.bar.noModel", "page.model.none"),
        ("markup.empty.onDevice", "page.empty.intro"),
        ("markup.empty.chooseModel", "page.empty.chooseModel"),
        ("markup.composer.placeholder", "page.composer.messageOrTalk"),
        ("markup.composer.holdToTalk", "page.voice.hold"),
        ("markup.skills.useSkills", "page.skills.on"),
        ("markup.model.lorasHint", "page.modelSheet.lorasHint"),
        ("markup.model.manage", "page.empty.manageModels"),
        ("markup.nav.models", "models.title"),
        ("markup.nav.settings", "settings.title"),
        ("markup.nav.about", "about.title"),
    };

    [Fact]
    public void TextThatIsRepaintedOrNamesAScreenReadsTheSameInEveryLanguage()
    {
        var problems = new List<string>();
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            Dictionary<string, string> strings = Loc.Tables.LoadFiles(language.Tag).Values
                .SelectMany(t => t).ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
            foreach ((string first, string then) in SameWords)
            {
                if (strings.TryGetValue(first, out string? a) && strings.TryGetValue(then, out string? b) && a != b)
                    problems.Add($"{language.Tag}: {first} \"{a}\" but {then} \"{b}\"");
            }
        }
        Assert.True(problems.Count == 0, string.Join(Environment.NewLine, problems));
    }

    // =====================================================================================
    // what the system shows for the app
    // =====================================================================================

    private static readonly string Platforms = Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Maui", "Platforms");
    private static readonly string ShareResources = Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.ShareExtension", "Resources");

    /// <summary>
    /// The app's bundles and the share extension's list the languages that have tables. The
    /// list is what lets iOS and the Mac pick a language's InfoPlist.strings, and draw their
    /// own controls in it.
    /// </summary>
    [Theory]
    [InlineData("TensorAgent.Maui/Platforms/iOS/Info.plist")]
    [InlineData("TensorAgent.Maui/Platforms/MacCatalyst/Info.plist")]
    [InlineData("TensorAgent.ShareExtension/Info.plist")]
    public void EveryBundleListsTheLanguagesThatHaveTables(string plist)
    {
        XElement dict = XDocument.Load(Path.Combine(Repo, "TensorAgent", "src", plist)).Root!.Element("dict")!;
        XElement key = dict.Elements("key").Single(k => k.Value == "CFBundleLocalizations");
        XElement array = (XElement)key.NextNode!;
        Assert.Equal("array", array.Name.LocalName);
        Assert.Equal(UiLanguages.Supported.Select(l => l.Tag), array.Elements("string").Select(e => e.Value));
        XElement region = dict.Elements("key").Single(k => k.Value == "CFBundleDevelopmentRegion");
        Assert.Equal(UiLanguages.English.Tag, ((XElement)region.NextNode!).Value);
    }

    /// <summary>
    /// What the system itself shows for the app -- the permission prompts, the share
    /// extension's name in the share sheet -- comes from each language's InfoPlist.strings,
    /// which no table holds. A missing or malformed file is an error nowhere: the system
    /// quietly shows English. So every language has the three files, each holding exactly
    /// the English entries, the iPhone's and the Mac's prompts alike, and the share sheet
    /// named as the extension's own screen is.
    /// </summary>
    [Fact]
    public void EveryLanguageHasThePromptsAndTheShareSheetNameTheSystemShows()
    {
        IReadOnlyDictionary<string, string> prompts = ReadStrings(PromptsFile("iOS", "en"));
        IReadOnlyDictionary<string, string> names = ReadStrings(Path.Combine(ShareResources, "en.lproj", "InfoPlist.strings"));
        var problems = new List<string>();
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            string ios = PromptsFile("iOS", language.Tag);
            string mac = PromptsFile("MacCatalyst", language.Tag);
            string share = Path.Combine(ShareResources, language.Tag + ".lproj", "InfoPlist.strings");
            string[] missing = new[] { ios, mac, share }.Where(f => !File.Exists(f)).ToArray();
            problems.AddRange(missing.Select(f => f + ": missing"));
            if (missing.Length > 0)
                continue;
            if (!File.ReadAllBytes(ios).AsSpan().SequenceEqual(File.ReadAllBytes(mac)))
                problems.Add($"{language.Tag}: the iPhone's and the Mac's InfoPlist.strings differ");
            try
            {
                IReadOnlyDictionary<string, string> local = ReadStrings(ios);
                if (!local.Keys.OrderBy(k => k, StringComparer.Ordinal).SequenceEqual(prompts.Keys.OrderBy(k => k, StringComparer.Ordinal)))
                    problems.Add($"{ios}: [{string.Join(", ", local.Keys)}], English has [{string.Join(", ", prompts.Keys)}]");
                problems.AddRange(local.Where(p => string.IsNullOrWhiteSpace(p.Value)).Select(p => $"{ios}: {p.Key} is empty"));

                IReadOnlyDictionary<string, string> name = ReadStrings(share);
                if (!name.Keys.OrderBy(k => k, StringComparer.Ordinal).SequenceEqual(names.Keys.OrderBy(k => k, StringComparer.Ordinal)))
                    problems.Add($"{share}: [{string.Join(", ", name.Keys)}], English has [{string.Join(", ", names.Keys)}]");
                ShareStrings.Strings.LoadFiles(language.Tag).TryGetValue("share", out IReadOnlyDictionary<string, string>? table);
                string? title = table is not null && table.TryGetValue("share.sheet.title", out string? t) ? t : null;
                foreach ((string key, string value) in name)
                {
                    if (value != title)
                        problems.Add($"{share}: {key} is \"{value}\", share.sheet.title is \"{title}\"");
                }
            }
            catch (FormatException e)
            {
                problems.Add(e.Message);
            }
        }
        Assert.True(problems.Count == 0, string.Join(Environment.NewLine, problems));
    }

    private static string PromptsFile(string platform, string tag) =>
        Path.Combine(Platforms, platform, "Resources", tag + ".lproj", "InfoPlist.strings");

    private static readonly Regex StringsEntry = new(@"""((?:[^""\\]|\\.)*)""\s*=\s*""((?:[^""\\]|\\.)*)""\s*;", RegexOptions.CultureInvariant);

    /// <summary>A .strings file's entries. Anything that is neither an entry nor a comment
    /// is an error, as is a key written twice: iOS would read the file as English.</summary>
    private static IReadOnlyDictionary<string, string> ReadStrings(string path)
    {
        string rest = Regex.Replace(File.ReadAllText(path), @"/\*.*?\*/", " ", RegexOptions.Singleline);
        rest = Regex.Replace(rest, @"(?m)^\s*//.*$", " ");
        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        string? repeated = null;
        rest = StringsEntry.Replace(rest, m =>
        {
            if (!entries.TryAdd(Unescape(m.Groups[1].Value), Unescape(m.Groups[2].Value)))
                repeated ??= m.Groups[1].Value;
            return " ";
        });
        if (repeated is not null)
            throw new FormatException($"{path}: {repeated} is written twice");
        if (!string.IsNullOrWhiteSpace(rest))
            throw new FormatException($"{path}: neither an entry nor a comment: {rest.Trim()}");
        return entries;

        static string Unescape(string text) => Regex.Replace(text, @"\\(U[0-9A-Fa-f]{4}|.)", m => m.Groups[1].Value switch
        {
            "n" => "\n",
            "t" => "\t",
            "r" => "\r",
            string u when u.Length == 5 => ((char)Convert.ToInt32(u[1..], 16)).ToString(),
            string other => other,
        });
    }

    // =====================================================================================
    // the page
    // =====================================================================================

    [Fact]
    public async Task ThePageIsServedTheStringsOfTheLanguageTheAppShows()
    {
        using var site = new Site(_root);
        Loc.Use(UiLanguages.Find("zh-Hans")!);

        string script = await site.Client.GetStringAsync("/i18n.js?lang=zh-Hans");
        const string head = "window.TensorAgentI18n = ";
        Assert.StartsWith(head, script, StringComparison.Ordinal);
        int end = script.IndexOf(";\n", StringComparison.Ordinal);
        JsonElement data = JsonSerializer.Deserialize<JsonElement>(script[head.Length..end]);
        Assert.Equal("zh-Hans", data.GetProperty("lang").GetString());
        Assert.Equal(Table("zh-Hans", "settings")["settings.language.title"],
            data.GetProperty("strings").GetProperty("settings.language.title").GetString());
        Assert.Contains("window.TensorSharpI18n", script, StringComparison.Ordinal);

        // The tag carries the language, so a reload after a switch never reuses a cached table.
        string page = await site.Client.GetStringAsync("/");
        Assert.Contains(PageStrings.Tag(), page, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThePagesSettingsSaveKeepsTheLanguageTheSettingsScreenChose()
    {
        using var site = new Site(_root);
        AppSettings stored = site.Settings.Load();
        stored.UiLanguage = "ko";
        site.Settings.Save(stored);

        // The page posts the whole copy it read at load; an old copy must not switch the user back.
        using HttpResponseMessage response = await site.Client.PostAsJsonAsync("/api/agent/settings",
            new { uiLanguage = "en", maxTokens = 4096 });
        response.EnsureSuccessStatusCode();

        AppSettings saved = site.Settings.Load();
        Assert.Equal("ko", saved.UiLanguage);
        Assert.Equal(4096, saved.MaxTokens);
    }

    // =====================================================================================
    // helpers
    // =====================================================================================

    private static IEnumerable<string> TableRoots() => new[]
    {
        Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Core", "Localization"),
        Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Sharing", "Localization"),
    };

    /// <summary>A table's keys in the order the file writes them.</summary>
    private static List<string> KeysInOrder(string file)
    {
        var keys = new List<string>();
        var reader = new Utf8JsonReader(File.ReadAllBytes(file), new JsonReaderOptions { CommentHandling = JsonCommentHandling.Disallow });
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.PropertyName && reader.CurrentDepth == 1)
                keys.Add(reader.GetString()!);
        }
        return keys;
    }

    /// <summary>Keys a JSON object names twice, which the loader would silently resolve to the last.</summary>
    private static IEnumerable<string> RepeatedKeys(string file)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var repeated = new List<string>();
        var reader = new Utf8JsonReader(File.ReadAllBytes(file), new JsonReaderOptions { CommentHandling = JsonCommentHandling.Disallow });
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.PropertyName && reader.CurrentDepth == 1 && !seen.Add(reader.GetString()!))
                repeated.Add(reader.GetString()!);
        }
        return repeated;
    }

    private static readonly Regex CsharpAsk = new(@"\b(?:Loc\.T|ShareStrings\.T)\(\s*""([^""]+)""", RegexOptions.CultureInvariant);
    private static readonly Regex CsharpCount = new(@"\b(?:Loc|ShareStrings)\.Plural\(\s*""([^""]+)""", RegexOptions.CultureInvariant);
    private static readonly Regex ScriptAsk = new(@"(?<![\w$])(?:t|tr|I18N\.t)\(\s*(['""])([^'""]+)\1", RegexOptions.CultureInvariant);
    private static readonly Regex ScriptCount = new(@"(?<![\w$])tn\(\s*(['""])([^'""]+)\1", RegexOptions.CultureInvariant);
    private static readonly Regex MarkupAsk = new(@"data-i18n(?:-placeholder|-title|-aria-label)?=""([^""]+)""", RegexOptions.CultureInvariant);
    private static readonly Regex Literal = new(@"""([A-Za-z][A-Za-z0-9_.\-]*)""|'([A-Za-z][A-Za-z0-9_.\-]*)'", RegexOptions.CultureInvariant);

    /// <summary>
    /// What the sources ask for. <c>asked</c> and <c>plural</c> come from the calls
    /// themselves (<c>Loc.T("…")</c>, <c>t('…')</c>, <c>data-i18n="…"</c>, …) and must all
    /// exist; <c>literals</c> is every quoted word that could be a key, wherever it is
    /// written, which is what "asked for" means for a key kept in a table and looked up
    /// later (the page's tool labels, a catalog entry's description).
    /// </summary>
    private static (HashSet<string> Asked, HashSet<string> Plural, HashSet<string> Literals) ScanSources(string which)
    {
        IEnumerable<string> files = which == "app"
            ? SourceFiles(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Core"), "*.cs", "*.js")
                .Concat(SourceFiles(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Maui"), "*.cs", "*.js", "*.html"))
                .Append(Path.Combine(Repo, "TensorSharp.Chat", "WebUi", "mask-editor.js"))
                .Append(Path.Combine(Repo, "TensorSharp.Chat", "WebUi", "studio.js"))
            : SourceFiles(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.Sharing"), "*.cs")
                .Concat(SourceFiles(Path.Combine(Repo, "TensorAgent", "src", "TensorAgent.ShareExtension"), "*.cs"));

        var asked = new HashSet<string>(StringComparer.Ordinal);
        var plural = new HashSet<string>(StringComparer.Ordinal);
        var literals = new HashSet<string>(StringComparer.Ordinal);
        foreach (string file in files)
        {
            string text = File.ReadAllText(file);
            bool script = file.EndsWith(".js", StringComparison.Ordinal) || file.EndsWith(".html", StringComparison.Ordinal);
            if (script)
            {
                foreach (Match m in ScriptAsk.Matches(text)) asked.Add(m.Groups[2].Value);
                foreach (Match m in ScriptCount.Matches(text)) plural.Add(m.Groups[2].Value);
                foreach (Match m in MarkupAsk.Matches(text)) asked.Add(m.Groups[1].Value);
            }
            else
            {
                foreach (Match m in CsharpAsk.Matches(text)) asked.Add(m.Groups[1].Value);
                foreach (Match m in CsharpCount.Matches(text)) plural.Add(m.Groups[1].Value);
            }
            foreach (Match m in Literal.Matches(text))
                literals.Add(m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value);
            foreach (Match m in MarkupAsk.Matches(text))
                literals.Add(m.Groups[1].Value);
        }
        return (asked, plural, literals);
    }

    private static IEnumerable<string> SourceFiles(string root, params string[] patterns) =>
        patterns.SelectMany(p => Directory.EnumerateFiles(root, p, SearchOption.AllDirectories))
            .Where(f =>
            {
                string relative = Path.GetRelativePath(root, f).Replace('\\', '/');
                return !relative.StartsWith("bin/", StringComparison.Ordinal) && !relative.Contains("/bin/", StringComparison.Ordinal)
                    && !relative.StartsWith("obj/", StringComparison.Ordinal) && !relative.Contains("/obj/", StringComparison.Ordinal);
            });

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException($"no TensorAgent above {AppContext.BaseDirectory}");
    }

    /// <summary>The loopback server with the app's own routes and a page to serve.</summary>
    private sealed class Site : IDisposable
    {
        private readonly LoopbackServer _server;

        public Site(string root)
        {
            string webRoot = Path.Combine(root, "webui");
            Directory.CreateDirectory(webRoot);
            File.WriteAllText(Path.Combine(webRoot, "index.html"), "<html><body><div id=\"chat\"></div></body></html>");
            Settings = new SettingsStore(Path.Combine(root, "settings.json"));
            var options = new ServerHostingOptions(
                startupModelPath: Path.Combine(root, "models", "none.gguf"),
                startupMmProjPath: null,
                defaultBackend: "ggml_cpu",
                supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
                defaultMaxTokens: 256,
                maxTokensPinned: false,
                defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
                defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
                uploadDirectory: root,
                logDirectory: Path.Combine(root, "logs"),
                fileLoggingEnabled: false,
                samplingDefaults: null);
            var chat = new WebUiChatService(
                new ModelService(), new SessionManager(), options,
                new UploadStoragePolicy(root), new SkillRegistry(new SkillRegistryOptions()),
                codeRunner: null, workspaces: null, codeArtifacts: null,
                NullLoggerFactory.Instance);
            _server = new LoopbackServer(NullLogger.Instance) { StaticRoot = webRoot };
            _server.MapWebUi(chat, root);
            _server.MapAgent(ModelCatalog.BuiltIn, new ModelStore(Path.Combine(root, "weights")),
                new ConversationStore(Path.Combine(root, "chats")), Settings, () => "test engine");
            _server.Start();
            Client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
            Client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_server.Token}");
        }

        public SettingsStore Settings { get; }

        public HttpClient Client { get; }

        public void Dispose()
        {
            Client.Dispose();
            _server.Dispose();
        }
    }
}
