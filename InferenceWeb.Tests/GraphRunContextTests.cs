// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Graph;

namespace InferenceWeb.Tests;

/// <summary>D07: the attributes of the steps an executor runs are one immutable <see cref="GraphRunContext"/>, entered for a
/// scope and restored when it ends, instead of each caller setting and restoring the executor's properties.</summary>
public sealed class GraphRunContextTests
{
    private sealed class AttributeExecutor : IGraphExecutor
    {
        public GraphCapabilities Capabilities { get; } = new();
        public IGraphPlan Compile(GraphProgram program) => throw new NotSupportedException();
        public void SyncToHost(IntPtr data, long bytes) { }
        public void Invalidate(IntPtr data) { }
        public bool Drafting { get; set; }
        public int StepSequences { get; set; }
    }

    [Fact]
    public void AScopeRunsUnderItsContextAndRestoresTheOneBefore()
    {
        IGraphExecutor executor = new AttributeExecutor { StepSequences = 3 };
        Assert.Equal(new GraphRunContext(false, 3), executor.RunContext);
        using (executor.EnterRun(new GraphRunContext(Drafting: true, StepSequences: 1)))
        {
            Assert.Equal(new GraphRunContext(true, 1), executor.RunContext);
            using (executor.EnterRun(new GraphRunContext(Drafting: false, StepSequences: 4)))
                Assert.Equal(new GraphRunContext(false, 4), executor.RunContext);
            Assert.Equal(new GraphRunContext(true, 1), executor.RunContext);
        }
        Assert.Equal(new GraphRunContext(false, 3), executor.RunContext);
    }

    [Fact]
    public void AScopeRestoresItsContextWhenTheRunThrows()
    {
        IGraphExecutor executor = new AttributeExecutor();
        Action run = () =>
        {
            using (executor.EnterRun(new GraphRunContext(true, 2)))
                throw new InvalidOperationException("the step failed");
        };
        Assert.Throws<InvalidOperationException>(run);
        Assert.Equal(default, executor.RunContext);
        default(GraphRunScope).Dispose();   // an empty scope is harmless
    }
}
