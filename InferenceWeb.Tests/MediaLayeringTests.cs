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
using System.Xml.Linq;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;

namespace InferenceWeb.Tests;

/// <summary>
/// The seam TensorSharp.Models has for native media: everything that names OpenCvSharp,
/// ImageMagick or a spawned process lives under <c>Media/Desktop/</c>, which the iOS target
/// compiles out. These guards fail the moment a native using drifts back into shared code
/// (which would compile fine on desktop and throw DllNotFoundException on a phone), or the
/// csproj stops keeping the native packages off the ios target.
/// </summary>
public class MediaLayeringTests
{
    private static readonly Regex NativeMediaUse = new(
        @"using\s+OpenCvSharp\b|using\s+ImageMagick\b|\bOpenCvSharp\.[A-Z]|\bImageMagick\.[A-Z]|" +
        @"System\.Diagnostics\.Process\b|\bProcess\.Start\s*\(|\bnew\s+(System\.Diagnostics\.)?ProcessStartInfo\b",
        RegexOptions.Compiled);

    [Fact]
    public void NothingOutsideMediaDesktop_NamesOpenCvMagickOrProcess()
    {
        string? repoRoot = FindRepoRoot();
        if (repoRoot == null)
            return;

        string models = Path.Combine(repoRoot, "TensorSharp.Models");
        if (!Directory.Exists(models))
            return;

        string desktop = Path.Combine(models, "Media", "Desktop") + Path.DirectorySeparatorChar;
        var offenders = new List<string>();
        foreach (string file in Directory.EnumerateFiles(models, "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || file.StartsWith(desktop, StringComparison.Ordinal))
            {
                continue;
            }

            string[] lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                // Prose may mention the libraries; only code counts.
                string code = lines[i].Split("//", 2)[0];
                if (NativeMediaUse.IsMatch(code))
                    offenders.Add($"{Path.GetRelativePath(models, file)}:{i + 1}: {lines[i].Trim()}");
            }
        }

        Assert.True(
            offenders.Count == 0,
            "these reach a native media library or spawn a process outside Media/Desktop, which compiles on "
            + "desktop and fails at runtime on iOS (no ios-arm64 OpenCV/Magick binaries, no processes):\n  "
            + string.Join("\n  ", offenders));
    }

    [Fact]
    public void TheDesktopProvider_RegistersItselfBeforeAnyCallerRuns()
    {
        // The module initializer in Media/Desktop ran when the Models assembly loaded, so a
        // desktop host that never heard of MediaCodecs still gets OpenCV + Magick.NET.
        Assert.Same(DesktopMediaProvider.Instance, MediaCodecs.Image);
        Assert.Same(DesktopMediaProvider.Instance, MediaCodecs.Video);
        Assert.Same(DesktopMediaProvider.Instance, MediaCodecs.VideoEncoder);

        // Audio past the managed WAV/MP3/OGG readers (FLAC, M4A, a video's soundtrack) goes to ffmpeg.
        Assert.Same(DesktopMediaProvider.Instance, MediaCodecs.Audio);
        Assert.Contains("DesktopMediaProvider", MediaCodecs.Describe());
    }

    [Fact]
    public void MediaCodecs_RefuseNullProviders()
    {
        Assert.Throws<ArgumentNullException>(() => MediaCodecs.Image = null!);
        Assert.Throws<ArgumentNullException>(() => MediaCodecs.Video = null!);
        Assert.Throws<ArgumentNullException>(() => MediaCodecs.VideoEncoder = null!);
        Assert.Throws<ArgumentNullException>(() => MediaCodecs.Audio = null!);
    }

    [Fact]
    public void TheModelsCsproj_KeepsNativePackagesAndDesktopSourcesOffTheAppleTargets()
    {
        string? repoRoot = FindRepoRoot();
        if (repoRoot == null)
            return;
        string csproj = Path.Combine(repoRoot, "TensorSharp.Models", "TensorSharp.Models.csproj");
        if (!File.Exists(csproj))
            return;

        XDocument doc = XDocument.Load(csproj);

        // The Apple targets are opt-in, so desktop builds keep the single TFM and its output
        // path. The TensorAgent app asks for both: net10.0-ios for the phone, and
        // net10.0-maccatalyst for the Mac app, which uses the same ImageIO/AVFoundation provider.
        XElement? frameworks = doc.Descendants("TargetFrameworks").FirstOrDefault();
        Assert.NotNull(frameworks);
        Assert.Equal("net10.0;net10.0-ios;net10.0-maccatalyst", frameworks!.Value.Trim());
        Assert.Contains("TensorSharpAppleTargets", (string?)frameworks.Attribute("Condition") ?? string.Empty);
        XElement? framework = doc.Descendants("TargetFramework").FirstOrDefault();
        Assert.NotNull(framework);
        Assert.Equal("net10.0", framework!.Value);

        foreach (XElement package in doc.Descendants("PackageReference"))
        {
            string include = (string?)package.Attribute("Include") ?? string.Empty;
            string condition = (string?)package.Parent?.Attribute("Condition") ?? string.Empty;
            bool native = include.StartsWith("Magick.NET", StringComparison.OrdinalIgnoreCase)
                          || include.Contains("OpenCvSharp", StringComparison.OrdinalIgnoreCase);
            if (native)
            {
                Assert.True(condition.Contains("'$(TargetPlatformIdentifier)' != 'ios'", StringComparison.Ordinal)
                            && condition.Contains("'$(TargetPlatformIdentifier)' != 'maccatalyst'", StringComparison.Ordinal),
                    $"{include} has no Apple app binaries and must sit in the ItemGroup conditioned off both Apple targets");
            }
            else
            {
                Assert.True(condition.Length == 0, $"{include} is pure managed and should be referenced on every target");
            }
        }

        XElement? removal = doc.Descendants("Compile")
            .FirstOrDefault(c => ((string?)c.Attribute("Remove") ?? string.Empty).Replace('\\', '/') == "Media/Desktop/**");
        Assert.NotNull(removal);
        string removalCondition = (string?)removal!.Parent?.Attribute("Condition") ?? string.Empty;
        Assert.Contains("'$(TargetPlatformIdentifier)' == 'ios'", removalCondition);
        Assert.Contains("'$(TargetPlatformIdentifier)' == 'maccatalyst'", removalCondition);

        // The process-only guide transform is shared with the Mac app, without
        // pulling any OpenCV/Magick source into either Apple target or enabling
        // process spawning on the phone.
        XElement sharedTransform = Assert.Single(doc.Descendants("Compile").Where(c =>
            ((string?)c.Attribute("Include") ?? string.Empty).Replace('\\', '/') == "Media/Desktop/FfmpegImageConditioning.cs"));
        Assert.Equal("'$(TargetPlatformIdentifier)' == 'maccatalyst'",
            (string?)sharedTransform.Parent?.Attribute("Condition"));
    }

    /// <summary>
    /// Every source under <c>Media/Apple/</c> is gated to Apple platforms. A Mac-only
    /// process adapter uses the narrower <c>#if MACCATALYST</c> guard.
    ///
    /// <para>Without the guard the file compiles into the net10.0 build, where its
    /// <c>using AVFoundation;</c> does not resolve — that much fails loudly. The reason this
    /// test exists is the quieter case: a new Apple-folder file that happens to use no Apple
    /// types would compile on desktop, and a module initializer or a registration in it would
    /// then run on a server. The provider is registered by a module initializer, so "compiles
    /// on desktop" and "hijacks MediaCodecs on desktop" are the same thing.</para>
    /// </summary>
    [Fact]
    public void EverySourceUnderMediaApple_IsGatedToApplePlatforms()
    {
        string? repoRoot = FindRepoRoot();
        if (repoRoot == null)
            return;

        string apple = Path.Combine(repoRoot, "TensorSharp.Models", "Media", "Apple");
        Assert.True(Directory.Exists(apple), $"{apple} is missing; the iOS media provider lives there");

        var files = Directory.EnumerateFiles(apple, "*.cs", SearchOption.AllDirectories).ToList();
        Assert.NotEmpty(files);
        foreach (string file in files)
        {
            string text = File.ReadAllText(file);
            Assert.True(
                text.Contains("#if IOS || MACCATALYST", StringComparison.Ordinal)
                    || text.Contains("#if MACCATALYST", StringComparison.Ordinal),
                $"{Path.GetFileName(file)} is not gated to Apple platforms");
            Assert.True(
                text.TrimEnd().EndsWith("#endif", StringComparison.Ordinal),
                $"{Path.GetFileName(file)} does not close its platform guard at the end of the file");
        }
    }

    /// <summary>
    /// The iOS provider's on-device probe and the simulator harness that fails on it agree
    /// about which checks exist.
    ///
    /// <para><c>Media/Apple/</c> compiles only for <c>net10.0-ios</c>, so no test in this
    /// net10.0 assembly can execute a line of it — <see cref="MediaProviderParityTests"/> pins
    /// the contract against the two providers that do run here, and
    /// <c>TensorAgent.Maui.Hosting.MediaProbe</c> runs it for real on the device.
    /// <c>scripts/verify-sim.sh</c> is what turns that into a failure rather than a log line,
    /// and it matches on the check names — so a check renamed on one side and not the other
    /// would silently stop being verified. This is the guard for that.</para>
    /// </summary>
    [Fact]
    public void TheOnDeviceMediaProbe_AndTheSimulatorHarness_ListTheSameChecks()
    {
        string? repoRoot = FindRepoRoot();
        if (repoRoot == null)
            return;

        string probe = Path.Combine(repoRoot, "TensorAgent", "src", "TensorAgent.Maui", "Hosting", "MediaProbe.cs");
        string verify = Path.Combine(repoRoot, "TensorAgent", "scripts", "verify-sim.sh");
        string program = Path.Combine(repoRoot, "TensorAgent", "src", "TensorAgent.Maui", "MauiProgram.cs");
        if (!File.Exists(probe) || !File.Exists(verify) || !File.Exists(program))
            return;

        // The app installs the provider before anything can decode, and runs the probe.
        string mauiProgram = File.ReadAllText(program);
        Assert.Contains("AppleMediaProvider.Register()", mauiProgram);
        Assert.Contains("MediaProbe.Run()", mauiProgram);

        var declared = new SortedSet<string>(
            Regex.Matches(File.ReadAllText(probe), @"(?<![A-Za-z])Check\(""(?<name>[a-z0-9-]+)""")
                .Select(m => m.Groups["name"].Value));
        Assert.NotEmpty(declared);

        // The HEIC and orientation checks are the reason the provider exists; losing either
        // would leave the interesting half of it unverified on every platform.
        Assert.Contains("heic-decode", declared);
        Assert.Contains("exif-orientation", declared);

        Match loop = Regex.Match(File.ReadAllText(verify), @"for CHECK in (?<names>[a-z0-9 -]+); do");
        Assert.True(loop.Success, "verify-sim.sh no longer iterates the media probe's checks");
        var verified = new SortedSet<string>(loop.Groups["names"].Value.Split(' ', StringSplitOptions.RemoveEmptyEntries));

        Assert.Equal(declared, verified);
    }

    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }
}
