// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Security.Cryptography;
using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Literal prompt goldens: no model, tokenizer, image I/O or backend required.</summary>
public sealed class Qwen4ExpGenerationBoundaryTests
{
    // Unmodified tokenizer.chat_template from Qwen3.8-Flash-Next-UD-IQ1_M,
    // direct metadata extraction, SHA256 12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce.
    // Keep the complete publisher template, not a replacement renderer. Medium
    // reasoning effort avoids its optional system instruction in these goldens.
    // Git checkouts may use CRLF, while the GGUF string uses LF. Normalize only
    // this fixture representation; its independent full-byte SHA remains fixed.
    private static readonly string CheckpointTemplate = """
    {%- set image_count = namespace(value=0) %}
    {%- set video_count = namespace(value=0) %}
    {%- macro render_content(content, do_vision_count, is_system_content=false) %}
        {%- if content is string %}
            {{- content }}
        {%- elif content is iterable and content is not mapping %}
            {%- for item in content %}
                {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                    {%- if is_system_content %}
                        {{- raise_exception('System message cannot contain images.') }}
                    {%- endif %}
                    {%- if do_vision_count %}
                        {%- set image_count.value = image_count.value + 1 %}
                    {%- endif %}
                    {%- if add_vision_id %}
                        {{- 'Picture ' ~ image_count.value ~ ': ' }}
                    {%- endif %}
                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                {%- elif 'video' in item or item.type == 'video' %}
                    {%- if is_system_content %}
                        {{- raise_exception('System message cannot contain videos.') }}
                    {%- endif %}
                    {%- if do_vision_count %}
                        {%- set video_count.value = video_count.value + 1 %}
                    {%- endif %}
                    {%- if add_vision_id %}
                        {{- 'Video ' ~ video_count.value ~ ': ' }}
                    {%- endif %}
                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                {%- elif 'text' in item %}
                    {{- item.text }}
                {%- else %}
                    {{- raise_exception('Unexpected item type in content.') }}
                {%- endif %}
            {%- endfor %}
        {%- elif content is none or content is undefined %}
            {{- '' }}
        {%- else %}
            {{- raise_exception('Unexpected content type.') }}
        {%- endif %}
    {%- endmacro %}
    {%- if not messages %}
        {{- raise_exception('No messages provided.') }}
    {%- endif %}
    {%- set sysns = namespace(count=0, text='') %}
    {%- for message in messages %}
        {%- if sysns.count == loop.index0 and (message.role == 'system' or message.role == 'developer') %}
            {%- set sys_content = render_content(message.content, false, true)|trim %}
            {%- if sys_content %}
                {%- set sysns.text = sysns.text + ('\n' if sysns.text else '') + sys_content %}
            {%- endif %}
            {%- set sysns.count = sysns.count + 1 %}
        {%- endif %}
    {%- endfor %}
    {%- set num_sys = sysns.count %}
    {%- set merged_system = sysns.text %}
    {%- set reasoning_instructions = '' %}
    {%- if enable_thinking is undefined or enable_thinking is true %}
        {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
        {%- if resolved_reasoning_effort == 'high' %}
            {%- set resolved_reasoning_effort = 'xhigh' %}
        {%- endif %}
        {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
            {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
        {%- endif %}
        {%- if resolved_reasoning_effort == 'xhigh' %}
            {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
        {%- elif resolved_reasoning_effort == 'low' %}
            {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
        {%- endif %}
    {%- endif %}
    {%- if tools and tools is iterable and tools is not mapping %}
        {{- '<|im_start|>system\n' }}
        {%- if reasoning_instructions %}
            {{- reasoning_instructions + '\n\n' }}
        {%- endif %}
        {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
        {%- for tool in tools %}
            {{- "\n" }}
            {{- tool | tojson }}
        {%- endfor %}
        {{- "\n</tools>" }}
        {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
        {%- if merged_system %}
            {{- '\n\n' + merged_system }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- else %}
        {%- if merged_system %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + merged_system + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
    {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
    {%- for message in messages[::-1] %}
        {%- set index = (messages|length - 1) - loop.index0 %}
        {%- if ns.multi_step_tool and message.role == "user" %}
            {%- set content = render_content(message.content, false)|trim %}
            {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- for message in messages %}
        {%- if loop.index0 >= num_sys %}
        {%- set content = render_content(message.content, true)|trim %}
        {%- if message.role == "system" or message.role == "developer" %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- elif message.role == "user" %}
            {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
        {%- elif message.role == "assistant" %}
            {%- set reasoning_content = '' %}
            {%- if message.reasoning_content is string %}
                {%- set reasoning_content = message.reasoning_content %}
            {%- endif %}
            {%- set reasoning_content = reasoning_content|trim %}
            {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
            {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
                {%- for tool_call in message.tool_calls %}
                    {%- if tool_call.function is defined %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {%- if tool_call.name is not defined or tool_call.name is none %}
                        {{- raise_exception('Tool call is missing a function name.') }}
                    {%- endif %}
                    {%- if loop.first %}
                        {%- if content|trim %}
                            {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                        {%- else %}
                            {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                        {%- endif %}
                    {%- else %}
                        {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                    {%- if tool_call.arguments is mapping %}
                        {%- for args_name, args_value in tool_call.arguments|items %}
                            {{- '<parameter=' + args_name + '>\n' }}
                            {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                            {{- args_value }}
                            {{- '\n</parameter>\n' }}
                        {%- endfor %}
                    {%- elif tool_call.arguments is string %}
                        {%- if tool_call.arguments|trim %}
                            {{- raise_exception('Tool call arguments for function "' + (tool_call.name | string) + '" were passed as a JSON string. Parse them into an object before calling apply_chat_template.') }}
                        {%- endif %}
                    {%- elif tool_call.arguments is defined and tool_call.arguments is not none %}
                        {{- raise_exception('Tool call arguments for function "' + (tool_call.name | string) + '" must be an object/mapping or a JSON string.') }}
                    {%- endif %}
                    {{- '</function>\n</tool_call>' }}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\n<tool_response>\n' }}
            {{- content }}
            {{- '\n</tool_response>' }}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>\n' }}
            {%- elif loop.last %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- else %}
            {{- raise_exception('Unexpected message role.') }}
        {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\n' }}
        {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\n\n</think>\n\n' }}
        {%- else %}
            {{- '<think>\n' }}
        {%- endif %}
    {%- endif %}
    {#- Unsloth fixes - developer role, merged system messages, tool calling #}
    """.Replace("\r\n", "\n", StringComparison.Ordinal);

    private static List<ChatMessage> OneUser() => [new() { Role = "user", Content = "hello" }];

    [Fact]
    public void CheckpointTemplateFixture_HasTheExactIndependentMetadataIdentity()
        => Assert.Equal("12827F24B742EA4E80CDC12DBCF9622227056B9F797252A3149263D4F9AAADCE",
            Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(CheckpointTemplate))));

    [Theory]
    [InlineData(false, "<|im_start|>assistant\n<think>\n\n</think>\n\n")]
    [InlineData(true, "<|im_start|>assistant\n<think>\n")]
    public void PublisherTemplate_PreservesTheExactGenerationBoundary(bool thinking, string suffix)
    {
        string actual = ChatTemplate.RenderFromGgufTemplate(CheckpointTemplate, OneUser(),
            architecture: "qwen4exp", enableThinking: thinking, reasoningEffort: "medium");

        Assert.Equal("<|im_start|>user\nhello<|im_end|>\n" + suffix, actual);
        Assert.Equal(suffix["<|im_start|>assistant\n".Length..],
            ChatProtocolRegistry.For("qwen4exp")!.AssistantGenerationSuffix!(thinking));
    }

    [Theory]
    [InlineData(false, "<|im_start|>assistant\n<think>\n\n</think>\n\n")]
    [InlineData(true, "<|im_start|>assistant\n<think>\n")]
    public void PublisherTemplate_ToolAndImageHistoryKeepsTheSameBoundary(bool thinking, string suffix)
    {
        List<ChatMessage> history =
        [
            new() { Role = "user", Content = "Describe.", ImagePaths = ["unused-fixture.png"] },
            new() { Role = "assistant", Content = "Checking.", ToolCalls =
                [new() { Id = "call-inspect", Name = "inspect",
                    Arguments = new Dictionary<string, object?> { ["path"] = "flower.png" } }] },
            new() { Role = "tool", ToolCallId = "call-inspect", Content = "yellow" },
            new() { Role = "user", Content = "Summarize." },
        ];
        const string goldenHistory =
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\nChecking.\n\n" +
            "<tool_call>\n<function=inspect>\n<parameter=path>\nflower.png\n</parameter>\n</function>\n</tool_call><|im_end|>\n" +
            "<|im_start|>user\n<tool_response>\nyellow\n</tool_response><|im_end|>\n" +
            "<|im_start|>user\nSummarize.<|im_end|>\n";

        string actual = ChatTemplate.RenderFromGgufTemplate(CheckpointTemplate, history,
            architecture: "qwen4exp", enableThinking: thinking, reasoningEffort: "medium");

        Assert.Equal(goldenHistory + suffix, actual);
        Assert.Equal("Describe.", history[0].Content);
        Assert.Equal("unused-fixture.png", Assert.Single(history[0].ImagePaths!));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WithoutGenerationPrompt_PreservesExistingHistoryTrimming(bool thinking)
    {
        string actual = ChatTemplate.RenderFromGgufTemplate(CheckpointTemplate, OneUser(),
            addGenerationPrompt: false, architecture: "qwen4exp", enableThinking: thinking,
            reasoningEffort: "medium");

        Assert.Equal("<|im_start|>user\nhello<|im_end|>", actual);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AlternatePublisherBoundary_IsPreservedWithoutInventingAThinkingStub(bool thinking)
    {
        const string template = "<|im_start|>user\n{{ messages[0].content }}<|im_end|>\n" +
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        string actual = ChatTemplate.RenderFromGgufTemplate(template, OneUser(),
            architecture: "qwen4exp", enableThinking: thinking);

        // Adding qwen4exp to IsQwen35Family would violate this publisher choice.
        Assert.Equal("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", actual);
    }

    [Theory]
    [InlineData(false, "<|im_start|>assistant\n<think>\n\n</think>\n\n")]
    [InlineData(true, "<|im_start|>assistant\n<think>\n")]
    public void ExistingQwen35GenerationNormalization_IsUnchanged(bool thinking, string suffix)
    {
        string actual = ChatTemplate.RenderFromGgufTemplate(CheckpointTemplate, OneUser(),
            architecture: "qwen35", enableThinking: thinking, reasoningEffort: "medium");

        Assert.Equal("<|im_start|>user\nhello<|im_end|>\n" + suffix, actual);
    }

    [Theory]
    [InlineData("gemma4", "hello\n\t")]
    [InlineData("unrelated", "hello")]
    [InlineData(null, "hello")]
    public void OtherArchitectures_KeepTheirExistingWhitespacePolicy(string? architecture, string golden)
        => Assert.Equal(golden, ChatTemplate.RenderFromGgufTemplate("{{ messages[0].content }}\n\t",
            OneUser(), architecture: architecture));

    [Theory]
    [InlineData(false, "", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")]
    [InlineData(true, "", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n")]
    [InlineData(false, "template dropped the last user", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")]
    [InlineData(true, "template dropped the last user", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n")]
    public void MissingOrRejectedTemplate_KeepsTheExistingGenericChatMlFallback(bool thinking, string template, string golden)
    {
        // qwen4exp currently declares no own Render. Preserve this actual fallback
        // contract; the generation-boundary fix must not silently install Qwen35.
        Assert.Null(ChatProtocolRegistry.For("qwen4exp")!.Render);
        Assert.Equal(golden, ChatTemplate.RenderFromGgufTemplate(template, OneUser(),
            architecture: "qwen4exp", enableThinking: thinking));
    }

    [Theory]
    [InlineData(true, "<｜hy_begin▁of▁sentence｜><｜hy_User｜>hello<｜hy_Assistant｜>")]
    [InlineData(false, "<｜hy_begin▁of▁sentence｜><｜hy_User｜>hello<｜hy_place▁holder▁no▁8｜>")]
    public void PreferOwnRenderer_BypassesTheJinjaWhitespacePolicy(bool addGenerationPrompt, string golden)
        => Assert.Equal(golden, ChatTemplate.RenderFromGgufTemplate(CheckpointTemplate, OneUser(),
            addGenerationPrompt: addGenerationPrompt, architecture: "hunyuan-dense"));
}
