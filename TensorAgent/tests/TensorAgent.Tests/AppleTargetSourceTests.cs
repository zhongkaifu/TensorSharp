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

namespace TensorAgent.Tests;

/// <summary>
/// The app's Apple targets (net10.0-ios, net10.0-maccatalyst) compile TensorSharp.Models against Apple's bindings,
/// whose global <c>ImageIO</c> namespace wins over a <c>using TensorSharp.Models.QwenImage;</c>: a bare
/// <c>ImageIO.Resize</c> outside that namespace compiles for the desktop and fails the Mac and iPhone app builds
/// (CS0234). Five such calls in the diffusion runtime broke both apps; the rest of the code says
/// <c>QwenImage.ImageIO</c>. Held here because no desktop build or test sees it.
/// </summary>
public sealed class AppleTargetSourceTests
{
    [Fact]
    public void TheModelsNameTheirImageIOByItsNamespace()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;
        Assert.NotNull(directory);
        string models = Path.Combine(directory!.FullName, "TensorSharp.Models");
        var bare = new Regex(@"(?<![.\w])ImageIO\.[A-Z]", RegexOptions.Compiled);
        var problems = new List<string>();
        foreach (string file in Directory.EnumerateFiles(models, "*.cs", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(models, file).Replace('\\', '/');
            // The desktop media provider is not compiled for the Apple targets; obj/ and bin/ are build output.
            if (relative.StartsWith("Media/Desktop/", StringComparison.Ordinal)
                || relative.StartsWith("obj/", StringComparison.Ordinal) || relative.StartsWith("bin/", StringComparison.Ordinal))
                continue;
            string[] lines = File.ReadAllLines(file);
            // Inside the namespace the class itself is found first; the Apple provider means Apple's ImageIO.
            if (lines.Any(l => l.TrimStart().StartsWith("namespace TensorSharp.Models.QwenImage", StringComparison.Ordinal)
                    || l.Trim() == "using ImageIO;"))
                continue;
            for (int i = 0; i < lines.Length; i++)
            {
                string code = lines[i].TrimStart();
                if (code.StartsWith("//", StringComparison.Ordinal))
                    continue;
                if (bare.IsMatch(code.Replace("cref=\"QwenImage.ImageIO", string.Empty, StringComparison.Ordinal)))
                    problems.Add($"{relative}:{i + 1}: {code}");
            }
        }
        Assert.True(problems.Count == 0,
            "Name it QwenImage.ImageIO (Apple's ImageIO namespace wins on the app's targets):" + Environment.NewLine
            + string.Join(Environment.NewLine, problems));
    }
}
