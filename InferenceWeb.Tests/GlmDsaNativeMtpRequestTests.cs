// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Whether the glm-dsa native loader pages its NextN/MTP draft block in. It
/// decides from the speculation enable variables BEFORE the scheduler exists,
/// so the two must read those variables the same way: a loader that treated
/// every non-"0" value as on paged ~3 GiB in for <c>TS_SPEC=false</c> while the
/// scheduler (which accepts only 1/true/yes/on) never drafted with it.
/// </summary>
public sealed class GlmDsaNativeMtpRequestTests : IDisposable
{
    private readonly EnvScope _env = new();

    public GlmDsaNativeMtpRequestTests()
    {
        _env.ClearSpeculationVars();
        _env.Set("TS_GLM_MTP", null);
    }

    public void Dispose() => _env.Dispose();

    [Theory]
    [InlineData("1", true)]
    [InlineData("true", true)]
    [InlineData("True", true)]
    [InlineData("yes", true)]
    [InlineData("on", true)]
    [InlineData("0", false)]
    [InlineData("false", false)]
    [InlineData("no", false)]
    [InlineData("off", false)]
    public void TsSpec_IsParsedExactlyAsTheSchedulerParsesIt(string value, bool expected)
    {
        _env.Set(SpeculationEnvVars.Enabled, value);

        Assert.Equal(expected, GlmDsaModel.NativeMtpRequested());
        Assert.Equal(SchedulerConfig.FromEnvironment().Speculation.Enabled, GlmDsaModel.NativeMtpRequested());
    }

    [Theory]
    [InlineData("1", true)]
    [InlineData("false", false)]
    [InlineData("0", false)]
    public void LegacyTsMtpSpec_IsHonouredWhenTsSpecIsUnset(string value, bool expected)
    {
        _env.Set(SpeculationEnvVars.LegacyEnabled, value);

        Assert.Equal(expected, GlmDsaModel.NativeMtpRequested());
        Assert.Equal(SchedulerConfig.FromEnvironment().Speculation.Enabled, GlmDsaModel.NativeMtpRequested());
    }

    [Fact]
    public void NoEnableVariable_LeavesTheDraftBlockUnloaded()
    {
        Assert.False(GlmDsaModel.NativeMtpRequested());
    }

    [Theory]
    [InlineData("1", "0", true)]
    [InlineData("0", "1", false)]
    public void TsGlmMtp_OverridesTheSharedSwitch(string glmMtp, string spec, bool expected)
    {
        _env.Set(SpeculationEnvVars.Enabled, spec);
        _env.Set("TS_GLM_MTP", glmMtp);

        Assert.Equal(expected, GlmDsaModel.NativeMtpRequested());
    }
}
