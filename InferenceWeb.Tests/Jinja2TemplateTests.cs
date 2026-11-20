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

    [Fact]
    public void ListPlusList_Concatenates()
    {
        // Jinja2 `+` on two sequences concatenates them. This used to fall through
        // to Stringify() and render the .NET type name
        // ("System.Collections.Generic.List`1[System.Object]").
        Assert.Equal("a,b,c",
            Render("{{ (['a', 'b'] + ['c']) | join(',') }}", new Dictionary<string, object>()));
    }

    [Fact]
    public void NamespaceListAccumulator_SurvivesJoin()
    {
        // Muse-Glimmer's render_system_meta() builds its recipient list with the
        // namespace accumulator idiom and then joins it. Before list concatenation
        // was implemented this emitted
        //   "# Valid recipients: System.Collections.Generic.List`1[System.Object]
        //    System.Collections.Generic.List`1[System.Object]."
        // into every rendered system prompt.
        const string tmpl =
            "{%- set ns = namespace(recipients=['\"self\"']) -%}" +
            "{%- for t in tools -%}{%- set ns.recipients = ns.recipients + ['\"' + t + '.*\"'] -%}{%- endfor -%}" +
            "{%- set ns.recipients = ns.recipients + ['\"user\"'] -%}" +
            "{{- '# Valid recipients: ' + ns.recipients | join(', ') + '.' -}}";
        Assert.Equal("# Valid recipients: \"self\", \"user\".",
            Render(tmpl, new Dictionary<string, object> { ["tools"] = new List<object>() }));
        Assert.Equal("# Valid recipients: \"self\", \"weather.*\", \"user\".",
            Render(tmpl, new Dictionary<string, object> { ["tools"] = new List<object> { "weather" } }));
    }

    [Fact]
    public void NumericAddition_StaysNumeric()
    {
        // int + int must not degrade to text, and a float on either side must stay
        // arithmetic rather than concatenate.
        Assert.Equal("3", Render("{{ 1 + 2 }}", new Dictionary<string, object>()));
        Assert.Equal("3.5", Render("{{ 1.5 + 2 }}", new Dictionary<string, object>()));
    }

    [Theory]
    [InlineData("{{ '<|eot|><image>' | string | replace('<|', '< |') | replace('<image>', '< image>') }}", "< |eot|>< image>")]
    [InlineData("{{ 'aaab' | replace('a', 'z', 2) }}", "zzab")]
    [InlineData("{{ 'aaab' | replace('a', 'z', -1) }}", "zzzb")]
    [InlineData("{{ false | default('xhigh', true) }}", "xhigh")]
    [InlineData("{{ (none or {'name': 'weather.get'}).name }}", "weather.get")]
    [InlineData("{{ ('abc' | upper)[:-1] }}", "AB")]
    [InlineData("{{ 'user\\'s reply; do not use for pre-tool updates or partial responses.' }}", "user's reply; do not use for pre-tool updates or partial responses.")]
    public void SparkExpressionFeaturesPreserveTextAndSchema(string template, string expected)
        => Assert.Equal(expected, Render(template, new Dictionary<string, object>()));

    /// <summary>
    /// <c>tojson</c> quotes strings as HF's chat-template filter (<c>json.dumps(...,
    /// ensure_ascii=False)</c>) and llama.cpp do: only quotes, backslashes and control
    /// characters are escaped, so a tool described with quotes, <c>&lt;</c>, <c>&amp;</c> or
    /// non-English text reaches the model as the publisher's engines print it.
    /// </summary>
    [Fact]
    public void ToJson_QuotesStringsLikePythonJsonDumpsWithoutAsciiEscapes()
    {
        var ctx = new Dictionary<string, object>
        {
            ["tool"] = new Dictionary<string, object>
            {
                ["name"] = "search",
                ["description"] = "Search \"docs\" — <b>A & B</b> 'x'+y\n\ttab\\ 搜索 🌍\u0001",
            },
        };
        Assert.Equal(
            "{\"name\": \"search\", \"description\": \"Search \\\"docs\\\" — <b>A & B</b> 'x'+y\\n\\ttab\\\\ 搜索 🌍\\u0001\"}",
            Render("{{ tool | tojson }}", ctx));
    }
}
