// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cli;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>--keep-source-size</c>, the CLI's spelling of the edit request's <c>keepSourceSize</c>.
/// The CLI's switch has no unknown-flag trap, so the page is the contract a user can see, and a
/// flag that cannot apply is refused rather than dropped.
/// </summary>
public sealed class CliKeepSourceSizeTests
{
    [Fact]
    public void TheUsagePageDocumentsTheFlag()
    {
        var page = new StringWriter();
        CliUsage.PrintUsage(page);
        Assert.Contains("--keep-source-size", page.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void OffIsAlwaysValid() =>
        TensorSharp.Cli.Program.ValidateKeepSourceSize(false, imageCount: 0, width: 1024, height: 768);

    [Fact]
    public void AnEditKeepsItsSourceSize() =>
        TensorSharp.Cli.Program.ValidateKeepSourceSize(true, imageCount: 2, width: 0, height: 0);

    [Fact]
    public void ItNeedsAnImage()
    {
        var error = Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.ValidateKeepSourceSize(true, 0, 0, 0));
        Assert.Contains("--image", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1024, 768)]
    [InlineData(1024, 0)]
    public void ItReplacesAnExplicitSizeRatherThanCombiningWithIt(int width, int height)
    {
        var error = Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.ValidateKeepSourceSize(true, 1, width, height));
        Assert.Contains("--width/--height", error.Message, StringComparison.Ordinal);
    }
}
