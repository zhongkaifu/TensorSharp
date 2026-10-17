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
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Claims the server's --help page made that the code did not back, pinned so they do not
/// come back: n-gram speculation "works on every model", a heredoc as the way files are
/// written, a MoE thread default of "one less than the CPUs", a fixed request-body limit,
/// a skills search order with the install directory in the wrong place, and an executable
/// called TensorSharp.Server that does not exist. The flag-set drift guards live in
/// ServerOptionsBuilderTests; these check what the page SAYS.
/// </summary>
public class ServerUsageAccuracyTests
{
    private static string Usage()
    {
        var writer = new StringWriter();
        ServerUsage.PrintUsage(writer);
        // Descriptions wrap at 100 columns, so compare against one flattened line.
        return Regex.Replace(writer.ToString(), @"\s+", " ");
    }

    [Fact]
    public void SpecType_DoesNotClaimNGramWorksEverywhere_AndSaysItDoesNotEnableSpeculation()
    {
        string usage = Usage();

        Assert.DoesNotContain("works on every model", usage, StringComparison.Ordinal);
        Assert.Contains("does not turn speculation on", usage, StringComparison.Ordinal);
        foreach (string family in new[] { "GPT-OSS", "Mistral 3", "Hunyuan Dense", "Nemotron-H" })
            Assert.Contains(family, usage, StringComparison.Ordinal);
    }

    [Fact]
    public void DraftModel_NamesEveryDrafterKind_AndTheEmbeddedNextNCheckpoints()
    {
        string usage = Usage();

        Assert.Contains("qwen4exp", usage, StringComparison.Ordinal);
        Assert.Contains("deepseek41-dspark", usage, StringComparison.Ordinal);
        Assert.Contains("Qwen 3.6, Qwen 3.8 27B, GLM-5.2 and GLM-5.3", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void CodeExec_NamesTheFileTools_AndNoLongerWritesFilesWithAHeredoc()
    {
        string usage = Usage();

        Assert.Contains("'read_file'", usage, StringComparison.Ordinal);
        Assert.Contains("'apply_patch'", usage, StringComparison.Ordinal);
        Assert.Contains("'write_file', which only CREATES a file", usage, StringComparison.Ordinal);
        Assert.DoesNotContain("write one with a heredoc", usage, StringComparison.Ordinal);
        Assert.DoesNotContain("One tool comes with it", usage, StringComparison.Ordinal);
        // edit_file is no longer offered; the page must not advertise it.
        Assert.DoesNotContain("edit_file", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void CpuMoeThreads_DescribesTheRealDefault()
    {
        string usage = Usage();

        Assert.DoesNotContain("Default: one less than", usage, StringComparison.Ordinal);
        Assert.Contains("capped at 64", usage, StringComparison.Ordinal);
        // Both native executors that size their own pool are named, so the rule above is
        // not read as applying to them.
        Assert.Contains("DeepSeek V4 / V4.1 and GLM-5.x executors are the exceptions", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void RedisUrl_IsDescribedAsTheResponsesStore()
    {
        string usage = Usage();

        Assert.DoesNotContain("for both the KV cache tier and the Responses API store", usage, StringComparison.Ordinal);
        Assert.Contains("Redis connection string for the Responses API store", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void Width_IsDocumentedAsTheQwenImageDefaultSizeToo()
    {
        string usage = Usage();

        Assert.Contains("TS_QWEN_IMAGE_WIDTH", usage, StringComparison.Ordinal);
        Assert.Contains("that default needs BOTH", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void UploadMaxMb_SaysTheRequestBodyLimitFollowsIt()
    {
        string usage = Usage();

        Assert.DoesNotContain("Default: 500, the request-body limit", usage, StringComparison.Ordinal);
        Assert.Contains("request-body limit of POST /api/upload follows it and never drops below 500 MB", usage, StringComparison.Ordinal);
        // The JSON routes keep the default: the help must not promise a bigger base64 attachment.
        Assert.Contains("Every other route keeps the 500 MB request-body limit", usage, StringComparison.Ordinal);
        Assert.Contains("about 375 MB whatever this cap is", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void SkillsDir_SaysTheInstallDirectoryIsScannedFirst()
    {
        string usage = Usage();

        Assert.Contains("ALWAYS scans skills/ next to the binary first", usage, StringComparison.Ordinal);
        Assert.Contains("POST /api/skills", usage, StringComparison.Ordinal);
    }

    [Fact]
    public void Examples_NameTheExecutableThatExists()
    {
        var writer = new StringWriter();
        ServerUsage.PrintUsage(writer);
        string usage = writer.ToString();

        Assert.Contains("  TensorSharp.Server.Host --model", usage, StringComparison.Ordinal);
        Assert.DoesNotMatch(new Regex(@"TensorSharp\.Server --"), usage);
    }

    private static string RawUsage()
    {
        var writer = new StringWriter();
        ServerUsage.PrintUsage(writer);
        return writer.ToString().Replace("\r\n", "\n", StringComparison.Ordinal);
    }

    /// <summary>The option flags listed under <paramref name="section"/>, in order.</summary>
    private static List<string> SectionFlags(string usage, string section)
    {
        string header = "\n" + section + ":\n";
        int start = usage.IndexOf(header, StringComparison.Ordinal);
        Assert.True(start >= 0, $"No usage section '{section}'.");
        int end = usage.IndexOf("\n\n", start + header.Length, StringComparison.Ordinal);
        return usage.Substring(start + header.Length, end - start - header.Length)
            .Split('\n')
            .Where(line => line.StartsWith("  --", StringComparison.Ordinal))
            .Select(line => line.Trim().Split(' ')[0])
            .ToList();
    }

    [Fact]
    public void LoraFlags_HaveTheirOwnSection_NamingBothFamilies()
    {
        string usage = RawUsage();

        Assert.Equal(LoraCliFlags.Flags, SectionFlags(usage, "LoRA plug-ins (Qwen-Image-2.1, MiniMax-H3 FL2VA)"));
        var companions = SectionFlags(usage, "Qwen-Image-2.1 companion models");
        Assert.Contains("--qwen-image-variant", companions);
        foreach (string flag in LoraCliFlags.Flags)
            Assert.DoesNotContain(flag, companions);
    }

    [Fact]
    public void Lora_NamesEveryShippedTurboConfig_AndWhatItLocksOnTheServer()
    {
        string lora = CliUsageTests.UsageEntry(RawUsage(), "--lora <path>");

        Assert.Contains("Qwen-Image-2.1 or MiniMax-H3 (FL2VA)", lora, StringComparison.Ordinal);
        foreach (var (path, plugin) in CliUsageTests.ShippedH3Plugins())
            Assert.Contains($"{path} ({plugin.Recipe.Steps} steps)", lora, StringComparison.Ordinal);
        foreach (string locked in new[] { "--video-steps", "\"steps\"", "\"flowShift\"", "\"sampler\"", "\"cfg\"" })
            Assert.Contains(locked, lora, StringComparison.Ordinal);
        // The CLI's spellings are not server flags.
        Assert.DoesNotContain("--diffusion-steps", lora, StringComparison.Ordinal);
        Assert.DoesNotContain("--flow-shift", lora, StringComparison.Ordinal);
        // A --config's model still to download is checked for its family after that download.
        Assert.Contains("refused at startup, before the plug-in downloads", lora, StringComparison.Ordinal);
        Assert.DoesNotContain("before anything downloads", lora, StringComparison.Ordinal);
        Assert.Contains("MiniMax-H3 refuses a negative strength", CliUsageTests.UsageEntry(RawUsage(), "--lora-scale <f>"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void VideoSteps_PointsAtTheTurboPlugin_NotAFourToEightStepOperatingPoint()
    {
        string steps = CliUsageTests.UsageEntry(RawUsage(), "--video-steps <N>");

        Assert.DoesNotContain("fast operating point", Usage(), StringComparison.Ordinal);
        Assert.DoesNotContain("4-8", steps, StringComparison.Ordinal);
        Assert.Contains($"{TensorSharp.Models.MiniMaxH3.MiniMaxH3Scheduler.DefaultSteps} for MiniMax-H3", steps, StringComparison.Ordinal);
        Assert.Contains("--lora config/lora/minimax-h3-fl2v-turbo-8step.json", steps, StringComparison.Ordinal);
        Assert.Contains("exactly the 8 or 4 steps it was distilled for", steps, StringComparison.Ordinal);
        Assert.Contains("any other value is refused at startup, and a request's own \"steps\" likewise", steps, StringComparison.Ordinal);
        Assert.Equal(new[] { 4, 8 }, CliUsageTests.ShippedH3Plugins().Values.Select(p => p.Recipe.Steps).OrderBy(s => s));
    }

    [Fact]
    public void Examples_IncludeAMiniMaxH3TurboServer()
    {
        string examples = RawUsage();
        examples = examples.Substring(examples.IndexOf("Examples:", StringComparison.Ordinal));

        Assert.Contains("  TensorSharp.Server.Host --config config/minimax-h3-fl2va.json --lora config/lora/minimax-h3-fl2v-turbo-8step.json ",
            examples, StringComparison.Ordinal);
        Assert.Contains("config/lora/minimax-h3-fl2v-turbo-8step.json", CliUsageTests.ShippedH3Plugins().Keys);
    }

    [Fact]
    public void HostedModelGuard_RefusalsNameTheExecutableThatExists()
    {
        var errors = new List<string>();
        Assert.False(HostedModelGuard.TryResolveHostedModelRequest("a.gguf", null, out _, out string noModel));
        errors.Add(noModel);
        Assert.False(HostedModelGuard.TryResolveHostedModelRequest("other.gguf", "/models/hosted.gguf", out _, out string wrongModel));
        errors.Add(wrongModel);
        Assert.False(HostedModelGuard.TryValidateHostedMmProjRequest("none", "/models/mmproj.gguf", out string unwantedProjector));
        errors.Add(unwantedProjector);
        Assert.False(HostedModelGuard.TryValidateHostedMmProjRequest("mmproj.gguf", null, out string noProjector));
        errors.Add(noProjector);
        Assert.False(HostedModelGuard.TryValidateHostedMmProjRequest("other.gguf", "/models/mmproj.gguf", out string wrongProjector));
        errors.Add(wrongProjector);

        foreach (string error in errors)
        {
            Assert.Contains("Restart TensorSharp.Server.Host ", error, StringComparison.Ordinal);
            Assert.DoesNotContain("Restart TensorSharp.Server ", error, StringComparison.Ordinal);
        }
    }
}
