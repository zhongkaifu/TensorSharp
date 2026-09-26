// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text.RegularExpressions;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// Every switch on the Settings screen reaches the running app, or is on a short list
/// of settings that are read at the moment they are used.
///
/// <para>
/// The "Speculative decoding" switch saved the file and nothing else, so the engine
/// that was standing kept the old policy until the next model load -- although
/// <c>AgentAppHost.ApplySpeculationSetting</c> exists precisely to move the running
/// engine, and was reached only when some other switch happened to be flipped. The
/// page is in the iOS head, which this project cannot reference, so the handlers are
/// read from the source: a switch that saves without <c>Apply</c> has to be named
/// here, with the reason it does not need to.
/// </para>
/// </summary>
public sealed class SettingsPageApplyTests
{
    /// <summary>Settings the page may save without applying, because nothing holds
    /// them: each is read from the store when it is used.</summary>
    private static readonly string[] ReadWhenUsed =
    {
        // Read when a chat is created (MainPage) and when the warm-up picks its prompt.
        nameof(AppSettings.ThinkByDefault),
        // Read by the Models page when a download is started.
        nameof(AppSettings.AllowCellularDownloads),
        nameof(AppSettings.DownloadOptionalFiles),
    };

    [Fact]
    public void EverySwitchIsAppliedUnlessItsSettingIsReadWhenUsed()
    {
        string page = File.ReadAllText(Path.Combine(
            FindRepoRoot(), "TensorAgent", "src", "TensorAgent.Maui", "Pages", "SettingsPage.cs"));

        var applied = Regex.Matches(page, @"Apply\(s => s\.(\w+) = ")
            .Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);
        var savedOnly = Regex.Matches(page, @"s\.(\w+) = \w+; _app\.Settings\.Save\(s\);")
            .Select(m => m.Groups[1].Value).ToHashSet(StringComparer.Ordinal);

        Assert.True(applied.Count > 0, "the page no longer routes its switches through Apply");
        Assert.Contains(nameof(AppSettings.SpeculativeDecoding), applied);
        Assert.Contains(nameof(AppSettings.MultiAgentEnabled), applied);
        Assert.Equal(ReadWhenUsed.OrderBy(n => n, StringComparer.Ordinal),
            savedOnly.OrderBy(n => n, StringComparer.Ordinal));

        // Every control the page builds is one of the two shapes above. A handler written
        // any other way (a differently named local, a block that saves and returns) would
        // otherwise fall into neither set and let a switch that never applies pass.
        int controls = Regex.Matches(page, @"_body\.Add\((?:Switch|Ladder|Choice|Stepper)\(").Count;
        Assert.True(controls > 0, "the page no longer builds its controls through Switch/Ladder/Choice/Stepper");
        Assert.Equal(controls, applied.Count + savedOnly.Count);
        // And nothing saves the settings except Apply itself and the read-when-used handlers.
        int saves = Regex.Matches(page, @"_app\.Settings\.Save\(").Count;
        Assert.Equal(savedOnly.Count + 1, saves);
    }

    /// <summary>The repository root, found by walking up for a known marker.</summary>
    private static string FindRepoRoot()
    {
        var here = new DirectoryInfo(AppContext.BaseDirectory);
        while (here is not null && !Directory.Exists(Path.Combine(here.FullName, "TensorSharp.Runtime")))
            here = here.Parent;
        return here?.FullName ?? throw new DirectoryNotFoundException("no repository root above " + AppContext.BaseDirectory);
    }
}
