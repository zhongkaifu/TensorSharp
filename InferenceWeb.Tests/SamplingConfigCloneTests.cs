// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// SamplingConfig is shared across requests as a server-wide default; the
/// Clone() helper has to produce an independent instance so per-request
/// mutations don't bleed into other requests.
/// </summary>
public class SamplingConfigCloneTests
{
    [Fact]
    public void Clone_CopiesAllScalarFields()
    {
        var src = new SamplingConfig
        {
            Temperature = 0.42f,
            TopK = 17,
            TopP = 0.55f,
            MinP = 0.07f,
            RepetitionPenalty = 1.4f,
            PresencePenalty = 0.2f,
            FrequencyPenalty = 0.3f,
            Seed = 1234,
            MaxTokens = 999,
        };

        var clone = src.Clone();

        Assert.Equal(src.Temperature, clone.Temperature);
        Assert.Equal(src.TopK, clone.TopK);
        Assert.Equal(src.TopP, clone.TopP);
        Assert.Equal(src.MinP, clone.MinP);
        Assert.Equal(src.RepetitionPenalty, clone.RepetitionPenalty);
        Assert.Equal(src.PresencePenalty, clone.PresencePenalty);
        Assert.Equal(src.FrequencyPenalty, clone.FrequencyPenalty);
        Assert.Equal(src.Seed, clone.Seed);
        Assert.Equal(src.MaxTokens, clone.MaxTokens);
        Assert.NotSame(src, clone);
    }

    [Fact]
    public void Clone_DuplicatesStopSequencesList()
    {
        var src = new SamplingConfig
        {
            StopSequences = new List<string> { "</s>", "<|eot|>" },
        };

        var clone = src.Clone();

        Assert.Equal(src.StopSequences, clone.StopSequences);
        // Adding to the clone must not mutate the source list - that's the
        // whole point of clone semantics for the per-request override path.
        clone.StopSequences!.Add("EXTRA");
        Assert.Equal(2, src.StopSequences.Count);
        Assert.Equal(3, clone.StopSequences.Count);
    }

    [Fact]
    public void Clone_NullStopSequences_StaysNull()
    {
        var src = new SamplingConfig { StopSequences = null };

        var clone = src.Clone();

        Assert.Null(clone.StopSequences);
    }
}
