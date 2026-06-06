// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Unit tests for the lightweight Jinja2 renderer used to evaluate GGUF chat
// templates. These focus on the features the Gemma 4 ("gemma4") template relies
// on that previously crashed or mis-rendered: the block-form set capture
// ({% set var %}...{% endset %}) and the multi-argument range() generator.
// They run without a model.

using System.Collections.Generic;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class Jinja2TemplateTests
{
    private static string Render(string template, Dictionary<string, object> ctx)
        => new Jinja2Template(template).Render(ctx);

    [Fact]
    public void SetBlock_CapturesRenderedBody()
    {
        // {% set x %}...{% endset %} captures the rendered body into x.
        string tmpl = "{% set captured %}Hello {{ name }}!{% endset %}[{{ captured }}]";
        string outp = Render(tmpl, new Dictionary<string, object> { ["name"] = "World" });
        Assert.Equal("[Hello World!]", outp);
    }

    [Fact]
    public void SetBlock_WithTrimFilter_AppliesFilterToCapture()
    {
        string tmpl = "{% set x | trim %}   padded   {% endset %}[{{ x }}]";
        string outp = Render(tmpl, new Dictionary<string, object>());
        Assert.Equal("[padded]", outp);
    }

    [Fact]
    public void SetBlock_HonorsWhitespaceControlInsideBody()
    {
        // The Gemma 4 template wraps the body in {%- ... -%} trims; the captured
        // value must reflect that trimming, not the raw inter-tag whitespace.
        string tmpl =
            "{%- set captured_content -%}\n" +
            "    {%- if msg == 'a' -%}AAA{%- else -%}BBB{%- endif -%}\n" +
            "{%- endset -%}<{{ captured_content }}>";
        Assert.Equal("<AAA>", Render(tmpl, new Dictionary<string, object> { ["msg"] = "a" }));
        Assert.Equal("<BBB>", Render(tmpl, new Dictionary<string, object> { ["msg"] = "z" }));
    }

    [Fact]
    public void InlineSet_StillWorks_NotMisparsedAsBlock()
    {
        // Regression: distinguishing block-set from inline-set must not break the
        // common "set var = expr" form (including values that contain '==').
        Assert.Equal("3", Render("{% set x = 1 + 2 %}{{ x }}", new Dictionary<string, object>()));
        Assert.Equal("True", Render("{% set x = a == a %}{{ x }}",
            new Dictionary<string, object> { ["a"] = "k" }));
    }

    [Fact]
    public void Range_SingleArg()
    {
        Assert.Equal("012", Render("{% for i in range(3) %}{{ i }}{% endfor %}", new Dictionary<string, object>()));
    }

    [Fact]
    public void Range_StartStop()
    {
        Assert.Equal("234", Render("{% for i in range(2, 5) %}{{ i }}{% endfor %}", new Dictionary<string, object>()));
    }

    [Fact]
    public void Range_StartStopStep_Descending()
    {
        // The Gemma 4 continuation scan uses range(loop.index0 - 1, -1, -1).
        Assert.Equal("3,2,1,0,",
            Render("{% for i in range(3, -1, -1) %}{{ i }},{% endfor %}", new Dictionary<string, object>()));
    }

    [Fact]
    public void Range_StartStopStep_PositiveStep()
    {
        Assert.Equal("0246", Render("{% for i in range(0, 8, 2) %}{{ i }}{% endfor %}", new Dictionary<string, object>()));
    }

    [Fact]
    public void NotFilterPrecedence_DefaultFalse()
    {
        // The Gemma 4 generation-prompt block hinges on
        // "{%- if not enable_thinking | default(false) -%}". Jinja binds the filter
        // tighter than 'not', i.e. not (enable_thinking | default(false)). With
        // enable_thinking undefined or false the result is true (emit the empty
        // <|channel>thought<channel|> priming); with true it is false (skip).
        const string tmpl = "{%- if not enable_thinking | default(false) -%}EMIT{%- endif -%}";
        Assert.Equal("EMIT", Render(tmpl, new Dictionary<string, object>()));
        Assert.Equal("EMIT", Render(tmpl, new Dictionary<string, object> { ["enable_thinking"] = false }));
        Assert.Equal("", Render(tmpl, new Dictionary<string, object> { ["enable_thinking"] = true }));
    }
}
