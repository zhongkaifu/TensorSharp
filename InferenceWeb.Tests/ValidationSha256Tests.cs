// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Security.Cryptography;
using TensorSharp.Validation;

namespace InferenceWeb.Tests;

public sealed class ValidationSha256Tests
{
    [Fact]
    public void LoadedUppercaseAndFileLowercaseDigestsIdentifyTheSameActualBytes()
    {
        byte[] image = "actual module fixture bytes\0\u0001\u0002"u8.ToArray();
        string loaded = Convert.ToHexString(SHA256.HashData(image));
        string file = loaded.ToLowerInvariant();
        Assert.NotEqual(loaded, file);
        Assert.True(ValidationSha256.Matches(loaded, file));
        Assert.True(ValidationSha256.Matches(file, loaded));
        Assert.Equal(file, ValidationSha256.Normalize(loaded));
        image[^1] ^= 1;
        Assert.False(ValidationSha256.Matches(loaded, Convert.ToHexString(SHA256.HashData(image))));
    }

    [Theory]
    [InlineData("")] [InlineData(" ")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")]
    [InlineData("gaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")]
    [InlineData(" aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")]
    public void CanonicalizationCannotTrimPadOrAcceptNonhexIdentity(string value)
        => Assert.Throws<ArgumentException>(() => ValidationSha256.Normalize(value));
}
