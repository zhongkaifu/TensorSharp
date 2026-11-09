// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.RegularExpressions;
using TensorAgent.Core.Python;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Tests;

/// <summary>
/// The research skill, which had to be rebuilt around one missing thing: it could
/// not find a source.
///
/// <para>
/// The old skill's search was a client for an endpoint the USER was expected to
/// configure, with one unauthenticated fallback that answers a phone with a
/// challenge page more often than with results. So every research request began by
/// asking the person who wanted research to supply the URLs, which is the one thing
/// they do not have. The new one asks nine keyless indexes at once and merges what
/// they say.
/// </para>
/// <para>
/// Most of what decides whether that WORKS is not the network. It is the ranking (a
/// keyword index answers a sentence badly, and its confidence must not outrank
/// relevance), the challenge detection (an engine's own navigation must never be
/// reported as results), and the analysis (a sentence that denies a claim must not
/// be counted as support). All of those are pure, so they are tested here against
/// canned documents, through the app's own interpreter, with no socket opened.
/// </para>
/// </summary>
[Collection(LivePythonCollection.Name)]
public sealed class ResearchSkillTests : IDisposable
{
    private static readonly string Repo = FindRepoRoot();
    private static readonly string Skill = Path.Combine(Repo, "TensorAgent", "skills", "research");
    private static readonly string Scripts = Path.Combine(Skill, "scripts");

    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-research-" + Guid.NewGuid().ToString("N"));

    public ResearchSkillTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { }
    }

    // =====================================================================================
    // what the skill promises on paper
    // =====================================================================================

    [Fact]
    public void TheSkillDeclaresThatItFindsItsOwnSources()
    {
        string text = File.ReadAllText(Path.Combine(Skill, "SKILL.md")).Replace("\r\n", "\n");
        Assert.StartsWith("---\n", text, StringComparison.Ordinal);

        Match description = Regex.Match(text, @"(?m)^description:\s*(?<value>.+)$");
        Assert.True(description.Success, "SKILL.md has no description; the registry shows one per skill");
        string declared = description.Groups["value"].Value;

        // The description is what a model matches a request against, and the whole
        // change here is that a request no longer has to carry a URL. If that stops
        // being said, the skill stops being chosen for "look this up".
        Assert.Contains("without being given any URLs", declared, StringComparison.OrdinalIgnoreCase);
        // The declaration is read on every turn, from a catalog that a phone-sized
        // context shortens to its shortest rung (160 characters, see
        // SkillPrompt.FitCatalogDescriptions). Any longer and the cut falls on the
        // clauses that say when to use it; SkillCatalogVisibilityTests checks the line
        // the model is actually shown.
        Assert.True(declared.Length <= 160,
            $"{declared.Length} characters: a phone-sized catalog would cut the description short");

        // And the scripts it documents have to be the scripts that exist.
        var documented = new HashSet<string>(StringComparer.Ordinal);
        foreach (Match line in Regex.Matches(text, @"(?m)^python3 (?:scripts/)?(?<script>[A-Za-z0-9_]+\.py)(?<args>.*)$"))
        {
            string script = line.Groups["script"].Value;
            string path = Path.Combine(Scripts, script);
            Assert.True(File.Exists(path), $"SKILL.md runs {script}, which does not exist");
            documented.Add(script);

            string source = File.ReadAllText(path);
            foreach (Match option in Regex.Matches(line.Groups["args"].Value, @"(?<flag>--[a-z][a-z-]*)"))
            {
                string flag = option.Groups["flag"].Value;
                Assert.True(source.Contains($"\"{flag}\"", StringComparison.Ordinal),
                    $"SKILL.md passes {flag} to {script}, which does not define it");
            }
        }

        foreach (string path in Directory.GetFiles(Scripts, "*.py"))
        {
            if (!File.ReadAllText(path).Contains("__main__", StringComparison.Ordinal))
                continue;
            Assert.Contains(Path.GetFileName(path), documented);
        }

        // The endpoint the old skill asked the user to configure is gone, and nothing
        // may quietly reintroduce it: a skill that needs a key is a skill that does
        // not work on the phone it ships on.
        foreach (string path in Directory.GetFiles(Scripts, "*.py"))
        {
            string source = File.ReadAllText(path);
            Assert.DoesNotContain("RESEARCH_SEARCH_URL", source, StringComparison.Ordinal);
            Assert.DoesNotContain("api_key", source, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void EveryScriptRunsOnAPlatformWithNoProcessesAndNothingToInstall()
    {
        // Same invariant the documents skill holds, for the same reason: iOS starts no
        // processes, and nothing installs a package on the phone. Every import has to
        // be the standard library or a sibling of the script.
        string[] forbidden = { "subprocess", "multiprocessing", "webbrowser", "requests",
                               "bs4", "lxml", "httpx", "aiohttp", "selenium", "playwright", "ctypes" };
        string[] standard = { "__future__", "argparse", "collections", "csv", "dataclasses", "datetime",
                              "gzip", "html", "io", "json", "math", "os", "re", "statistics", "string",
                              "sys", "textwrap", "time", "typing", "unicodedata", "urllib", "zlib" };
        var siblings = Directory.GetFiles(Scripts, "*.py")
            .Select(Path.GetFileNameWithoutExtension).ToHashSet(StringComparer.Ordinal)!;

        foreach (string path in Directory.GetFiles(Scripts, "*.py"))
        {
            foreach (string module in ImportsOf(path))
            {
                Assert.DoesNotContain(module, forbidden);
                Assert.True(standard.Contains(module) || siblings.Contains(module),
                    $"{Path.GetFileName(path)} imports {module}, which is neither in the standard "
                    + "library nor shipped with the skill, and nothing installs packages on a phone");
            }
        }
    }

    // =====================================================================================
    // the parts that decide whether a run is any good, with no socket opened
    // =====================================================================================

    /// <summary>
    /// A keyword index's confidence must not outrank being about the question.
    ///
    /// <para>
    /// Observed, not invented: ask MediaWiki "what is the Kessler syndrome and is it
    /// happening" and its first result is the article on mental disorders, because
    /// the query is a sentence and its index is not. Wikipedia carries the highest
    /// provider weight here — it is usually right — so before relevance was part of
    /// the ranking, that article was the first thing the dossier read, and four
    /// thousand words about the DSM went in front of the model.
    /// </para>
    /// </summary>
    [LivePythonFact]
    public void RankingPutsThePageThatIsAboutTheQuestionAboveTheIndexThatIsConfident()
    {
        string report = RunPython("""
            import discover, json
            wrong = discover.Hit(url="https://en.wikipedia.org/wiki/Mental_disorder",
                                 title="Mental disorder", snippet="a psychological syndrome or pattern")
            right = discover.Hit(url="https://en.wikipedia.org/wiki/Kessler_syndrome",
                                 title="Kessler syndrome",
                                 snippet="a scenario in which the density of objects in low Earth orbit is high enough")
            terms = discover.webtext.content_words("what is the Kessler syndrome and is it happening")
            ranked = discover.merge({"wikipedia": [wrong, right]}, 5, terms)
            print(json.dumps([h.url for h in ranked]))
            """);
        Assert.StartsWith("[\"https://en.wikipedia.org/wiki/Kessler_syndrome\"", report.Trim(), StringComparison.Ordinal);
    }

    /// <summary>
    /// The same page named by two independent indexes beats one named by a single
    /// index, which is the only quality signal available without a ranker of our own.
    /// </summary>
    [LivePythonFact]
    public void AgreementBetweenTwoIndexesOutranksOne()
    {
        string report = RunPython("""
            import discover, json
            shared = discover.Hit(url="https://a.example/kessler-syndrome", title="The Kessler syndrome explained")
            alone = discover.Hit(url="https://b.example/kessler-syndrome-notes", title="Kessler syndrome notes")
            terms = discover.webtext.content_words("kessler syndrome")
            ranked = discover.merge({"duckduckgo": [shared, alone], "marginalia": [shared]}, 5, terms)
            print(json.dumps([[h.url, sorted(set(h.providers))] for h in ranked]))
            """);
        Assert.Contains("a.example", report.Split('\n')[0], StringComparison.Ordinal);
        Assert.StartsWith("[[\"https://a.example/kessler-syndrome\"", report.Trim(), StringComparison.Ordinal);
    }

    /// <summary>
    /// The same article reached from three indexes with three tracking parameters is
    /// one article. Counting it as three would destroy the agreement signal entirely.
    /// </summary>
    [LivePythonFact]
    public void TheSameArticleFromThreeIndexesIsOneSource()
    {
        string report = RunPython("""
            import discover, json
            a = discover.Hit(url="https://site.example/a-story", title="A story")
            b = discover.Hit(url="https://www.site.example/a-story/?utm_source=news", title="A story - Site")
            c = discover.Hit(url="https://site.example/a-story", title="A story")
            ranked = discover.merge({"duckduckgo": [a], "marginalia": [b], "hackernews": [c]}, 5,
                                    discover.webtext.content_words("a story"))
            print(json.dumps([[h.url, sorted(set(h.providers))] for h in ranked]))
            """);
        Assert.Single(Regex.Matches(report, "site.example"));
        Assert.Contains("duckduckgo", report, StringComparison.Ordinal);
        Assert.Contains("hackernews", report, StringComparison.Ordinal);
        Assert.Contains("marginalia", report, StringComparison.Ordinal);
    }

    /// <summary>
    /// An engine that has decided we are a robot must produce a named failure, never
    /// its own navigation dressed as results.
    ///
    /// <para>
    /// This is the specific way the old skill failed while looking like it worked:
    /// Mojeek now answers a plain fetch with "JavaScript is required to complete this
    /// challenge", and a generic link extractor happily returns Newsletter, About and
    /// Privacy as the sources for the user's question. The model then reads them.
    /// </para>
    /// </summary>
    [LivePythonFact]
    public void AChallengePageIsRefusedByNameRatherThanParsedForLinks()
    {
        string report = RunPython("""
            import discover, webtext
            page = webtext.Page(url="https://engine.example/search?q=x", status=200, title="Search",
                                text="Search\nAbout\nNewsletter\nJavaScript is required to complete this challenge.",
                                links=[("https://engine.example/about", "About"),
                                       ("https://buttondown.email/Engine", "Newsletter")])
            try:
                discover._refuse_if_challenged(page, "engine")
                print("NOT REFUSED")
            except discover.ProviderFailed as failed:
                print("REFUSED:", failed)
            """);
        Assert.StartsWith("REFUSED:", report.Trim(), StringComparison.Ordinal);
        Assert.Contains("rate-limiting", report, StringComparison.Ordinal);
    }

    /// <summary>
    /// A provider that throws does not take the run down with it. Nine services are
    /// asked on every question and several will be having a bad day.
    /// </summary>
    [LivePythonFact]
    public void OneProviderFailingLeavesTheOthersAnswering()
    {
        string report = RunPython("""
            import discover, webtext, json

            def broken(query, count, timeout):
                raise RuntimeError("upstream is on fire")

            def works(query, count, timeout):
                return [discover.Hit(url="https://ok.example/page", title="Kessler syndrome page")]

            discover.PROVIDERS["broken"] = broken
            discover.PROVIDERS["works"] = works
            hits, failures = discover.discover("kessler syndrome", ["broken", "works"], 5, 5.0)
            print(json.dumps({"hits": [h.url for h in hits], "failures": failures}))
            """);
        Assert.Contains("https://ok.example/page", report, StringComparison.Ordinal);
        Assert.Contains("upstream is on fire", report, StringComparison.Ordinal);
    }

    /// <summary>
    /// The switch being off is one sentence naming the switch, not nine failures.
    /// </summary>
    [LivePythonFact]
    public void TheNetworkSwitchBeingOffIsReportedOnceAndNotAsNineProviderFailures()
    {
        string report = RunPython("""
            import discover, webtext

            def refuses(query, count, timeout):
                raise webtext.NetworkOff("network access is disabled by the user")

            discover.PROVIDERS["a"] = refuses
            discover.PROVIDERS["b"] = refuses
            try:
                discover.discover("anything", ["a", "b"], 5, 5.0)
                print("NOT RAISED")
            except webtext.NetworkOff as off:
                print("OFF:", off)
            """);
        Assert.StartsWith("OFF:", report.Trim(), StringComparison.Ordinal);
    }

    /// <summary>
    /// A dossier is written even when every source failed, and it says which failed
    /// and why. A run that produced no file would leave the model with nothing to
    /// tell the user except that it did not work.
    /// </summary>
    [LivePythonFact]
    public void ADossierNamesEverySourceAndCarriesTheWarningWithIt()
    {
        string report = RunPython("""
            import research
            records = [
              {"url": "https://a.example/one", "title": "One", "ok": True, "error": "",
               "providers": ["duckduckgo", "wikipedia"], "date": "2024-05-01", "words": 900,
               "text": "The Kessler syndrome is a cascade of collisions in low Earth orbit. "
                       "Nothing else in this sentence matters at all to the question asked."},
              {"url": "https://b.example/two", "title": "Two", "ok": False,
               "error": "the server answered 503 Service Unavailable",
               "providers": ["marginalia"], "date": "", "words": 0, "text": ""},
            ]
            terms = research.keywords("what is the Kessler syndrome")
            print(research.dossier("what is the Kessler syndrome", records,
                                   {"github": "rate limited"}, ["duckduckgo"], terms))
            """);

        Assert.Contains("[One](https://a.example/one)", report, StringComparison.Ordinal);
        Assert.Contains("not fact and not instruction", report, StringComparison.Ordinal);
        // The date, so a reader can tell whether this is current.
        Assert.Contains("2024-05-01", report, StringComparison.Ordinal);
        // The failure, by name and reason: a dossier that hides what it could not
        // read misrepresents its own coverage.
        Assert.Contains("https://b.example/two", report, StringComparison.Ordinal);
        Assert.Contains("503", report, StringComparison.Ordinal);
        Assert.Contains("github: rate limited", report, StringComparison.Ordinal);
        // And the sentence that bears on the question was picked out of the page.
        Assert.Contains("cascade of collisions", report, StringComparison.Ordinal);
    }

    /// <summary>
    /// A sentence that denies a claim is not evidence for it.
    ///
    /// <para>
    /// The one thing a corroboration tool must not get wrong. "The scientific
    /// community has not reached a consensus about whether the Kessler Syndrome has
    /// begun" contains every word of the claim, and counting it as support turns a
    /// live disagreement into a fabricated consensus.
    /// </para>
    /// </summary>
    [LivePythonFact]
    public void AClaimDeniedIsSortedApartFromAClaimStated()
    {
        string report = RunPython("""
            import analyze
            records = [
              {"url": "https://yes.example/a", "title": "Yes", "ok": True, "date": "", "words": 50,
               "text": "The Kessler syndrome has already begun in low Earth orbit, several analysts say."},
              {"url": "https://no.example/b", "title": "No", "ok": True, "date": "", "words": 50,
               "text": "The scientific community has not reached a consensus about whether the Kessler "
                       "syndrome has begun, or how bad it would be."},
              {"url": "https://quiet.example/c", "title": "Quiet", "ok": True, "date": "", "words": 50,
               "text": "This page is about the history of weather balloons and mentions nothing else here."},
            ]
            print("\n".join(analyze.claim(records, "the Kessler syndrome has already begun", 2)))
            """);

        int plainly = report.IndexOf("Say it, plainly (1)", StringComparison.Ordinal);
        int hedged = report.IndexOf("qualification or a denial nearby (1)", StringComparison.Ordinal);
        int silent = report.IndexOf("Do not mention it (1)", StringComparison.Ordinal);
        Assert.True(plainly >= 0 && hedged > plainly && silent > hedged,
            "the three groups are not one source each:\n" + report);

        // The denial is marked where it is quoted, so a reader skimming sees it.
        Assert.Contains("⚠", report, StringComparison.Ordinal);
        Assert.Contains("https://quiet.example/c", report, StringComparison.Ordinal);
        // And the tool says out loud that silence is not disagreement.
        Assert.Contains("is not a source that disagrees", report, StringComparison.Ordinal);
    }

    /// <summary>
    /// Figures are grouped by the figure, which is the only way a disagreement
    /// between sources about a number is visible at all.
    /// </summary>
    [LivePythonFact]
    public void FiguresAreGroupedSoTwoSourcesDisagreeingIsVisible()
    {
        string report = RunPython("""
            import analyze
            records = [
              {"url": "https://a.example/a", "title": "A", "ok": True, "date": "", "words": 50,
               "text": "There are more than 27,000 tracked pieces of debris in orbit around Earth today."},
              {"url": "https://b.example/b", "title": "B", "ok": True, "date": "", "words": 50,
               "text": "Roughly 27,000 tracked objects are catalogued by the surveillance network."},
              {"url": "https://c.example/c", "title": "C", "ok": True, "date": "", "words": 50,
               "text": "The catalogue lists about 2,700 tracked objects larger than ten centimetres."},
            ]
            print("\n".join(analyze.numbers(records, 10)))
            """);
        Assert.Contains("**27,000** — stated by 2 source(s)", report, StringComparison.Ordinal);
        Assert.Contains("**2,700** — stated by 1 source(s)", report, StringComparison.Ordinal);
    }

    /// <summary>
    /// A table on a page comes out as columns, not as a run of words.
    /// </summary>
    [LivePythonFact]
    public void ATableOnAPageIsReadAsRowsAndColumns()
    {
        string report = RunPython("""
            import webtext, json
            markup = ("<html><body><table><tr><td>nav</td></tr></table>"
                      "<table><tr><th>Region</th><th>Units</th></tr>"
                      "<tr><td>Northgate</td><td>12</td></tr>"
                      "<tr><td>Bellhaven</td><td>19</td></tr></table></body></html>")
            print(json.dumps(webtext.tables(markup)))
            """);
        // Biggest first: the two-cell layout table at the top is not the one meant.
        Assert.StartsWith("[[[\"Region\", \"Units\"]", report.Trim(), StringComparison.Ordinal);
        Assert.Contains("Northgate", report, StringComparison.Ordinal);
    }

    // =====================================================================================
    // running python the way the app does
    // =====================================================================================

    /// <summary>
    /// Run a snippet with the skill's own scripts importable, under the app's real
    /// sandbox and with the network off — so a test that accidentally reaches for a
    /// socket fails rather than depending on somebody else's server.
    /// </summary>
    private string RunPython(string source)
    {
        var python = new EmbeddedPython(Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable));
        Assert.True(python.IsAvailable, python.UnavailableReason);

        var policy = new ExecutionPolicy(
            AllowScripts: true,
            AllowNetwork: false,
            WorkRoot: _root,
            ReadableRoots: new[] { Scripts },
            TempRoot: _root);
        var context = new InterpreterContext(_root, new Dictionary<string, string>
        {
            ["HOME"] = _root,
            ["PYTHONPATH"] = Scripts,
        }, policy);

        string prologue = "import sys\nsys.path.insert(0, " + Quote(Scripts) + ")\n";
        ExecutionResult result = python
            .RunCodeAsync(prologue + source, Array.Empty<string>(), context, CancellationToken.None)
            .GetAwaiter().GetResult();

        Assert.True(result.ExitCode == 0,
            $"the snippet failed with exit {result.ExitCode}:{Environment.NewLine}{result.Stdout}{result.Stderr}");
        return result.Stdout;
    }

    private static string Quote(string value) => "'" + value.Replace("'", "\\'") + "'";

    private static IEnumerable<string> ImportsOf(string path)
    {
        foreach (string raw in File.ReadAllLines(path))
        {
            string line = raw.Trim();
            Match import = Regex.Match(line, @"^import\s+(?<names>[A-Za-z_][\w., ]*)$");
            if (import.Success)
            {
                foreach (string name in import.Groups["names"].Value.Split(','))
                {
                    string trimmed = name.Trim().Split(' ')[0];
                    if (trimmed.Length > 0) yield return trimmed.Split('.')[0];
                }
                continue;
            }
            Match from = Regex.Match(line, @"^from\s+(?<module>[A-Za-z_][\w.]*)\s+import\s");
            if (from.Success)
                yield return from.Groups["module"].Value.Split('.')[0];
        }
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "TensorAgent", "skills")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException($"no TensorAgent/skills above {AppContext.BaseDirectory}");
    }
}
