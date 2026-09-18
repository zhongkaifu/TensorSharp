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
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The six editing rules that go in the system prompt — a channel this host had never
/// used for anything about files.
///
/// <para>
/// Two properties are worth defending and they pull in opposite directions. The block has
/// to NAME the tools, so a drift guard checks it against the dispatch table rather than
/// against a literal. And every byte of it has to be a pure function of the options,
/// because it sits at the front of the prompt where the KV prefix cache chains its hashes
/// from block zero: one varying byte drops prefix reuse to zero for the whole
/// conversation, not just for the part that changed, and it would show up as a
/// performance mystery rather than as a test failure.
/// </para>
/// </summary>
public class CodePromptTests
{
    [Fact]
    public void TheBlockIsByteIdenticalForIdenticalOptions()
    {
        // The KV-prefix guard. Nothing per-conversation, no timestamps, no paths, no
        // counters — asserted here because nothing else would ever catch it.
        Assert.Equal(CodePrompt.Block(true, true), CodePrompt.Block(true, true));
        Assert.Equal(CodePrompt.Block(true, false), CodePrompt.Block(true, false));
    }

    [Fact]
    public void TheBlockNamesEveryFileToolByItsRealName()
    {
        // Against SkillToolNames, not against literals: renaming a tool in one place and
        // not the other would leave the prompt telling the model to call something that
        // does not exist.
        string block = CodePrompt.Block(fileTools: true, hasPatch: true);

        Assert.Contains(SkillToolNames.ReadFile, block, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillToolNames.EditFile, block, StringComparison.Ordinal);
        Assert.Contains(SkillToolNames.WriteFile, block, StringComparison.Ordinal);
        Assert.Contains(SkillToolNames.ApplyPatch, block, StringComparison.Ordinal);
        Assert.Contains(SkillToolNames.Shell, block, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutThePatchTool_TheBlockDoesNotMentionIt()
    {
        // Naming a tool the model was not given is worse than saying nothing: it cannot
        // tell an inapplicable instruction from its own misreading of its tool list.
        string block = CodePrompt.Block(fileTools: true, hasPatch: false);

        Assert.DoesNotContain(SkillToolNames.ApplyPatch, block, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillToolNames.EditFile, block, StringComparison.Ordinal);
        Assert.Contains("`write_file` only to create a new file", block, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutTheFileTools_ThereIsNoBlockAtAll()
    {
        // The stateless endpoints have none of them, and their shell declaration already
        // says the honest thing about writing files there.
        Assert.Equal(string.Empty, CodePrompt.Block(fileTools: false, hasPatch: false));
        Assert.Equal(string.Empty, CodePrompt.Block(fileTools: false, hasPatch: true));
    }

    [Fact]
    public void TheBlockSaysTheThingThisWholeChangeIsAbout()
    {
        // The one rule that has to be known before the first call, because there is no
        // result to attach it to yet: do not re-type a file to change part of it.
        string block = CodePrompt.Block(fileTools: true, hasPatch: true);

        Assert.Contains("Never rewrite a whole file to change part of it", block, StringComparison.Ordinal);
        Assert.Contains("one file or multiple files", block, StringComparison.Ordinal);
        Assert.Contains("only tool for changing existing files", block, StringComparison.Ordinal);
        Assert.Contains("do not read the file back", block, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBlockIsShort()
    {
        // Six lines and no syntax teaching. Everything about HOW to call a tool lives in
        // that tool's declaration, and everything about recovering from a failure is
        // attached to the failing result — which is where this codebase has its only
        // evidence that guidance changes behaviour.
        string block = CodePrompt.Block(fileTools: true, hasPatch: true);

        Assert.True(block.Length < 1200, $"the block has grown to {block.Length} characters");
        Assert.StartsWith(CodePrompt.Heading, block, StringComparison.Ordinal);
    }
}
