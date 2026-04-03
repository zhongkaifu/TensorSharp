using System;
using System.Collections.Generic;
using System.Text;

namespace InferenceEngine
{
    public class ChatMessage
    {
        public string Role { get; set; }
        public string Content { get; set; }
        /// <summary>
        /// Optional list of image file paths for multimodal messages.
        /// </summary>
        public List<string> ImagePaths { get; set; }
        /// <summary>
        /// Optional list of audio file paths for multimodal messages.
        /// </summary>
        public List<string> AudioPaths { get; set; }
        /// <summary>
        /// True if ImagePaths represent video frames (inserts &lt;|video&gt; before frame &lt;|image&gt; tokens).
        /// </summary>
        public bool IsVideo { get; set; }
    }

    public static class ChatTemplate
    {
        public static string RenderQwen3(List<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            var sb = new StringBuilder();
            foreach (var msg in messages)
            {
                sb.Append($"<|im_start|>{msg.Role}\n{msg.Content}<|im_end|>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|im_start|>assistant\n");
            }
            return sb.ToString();
        }

        /// <summary>
        /// Render Qwen3.5 template with optional image support.
        /// Matches the GGUF built-in chat template: for each image in a message,
        /// inserts <|vision_start|><|image_pad|><|vision_end|> markers.
        /// The single <|image_pad|> token is later expanded to N tokens based on image dimensions.
        /// </summary>
        public static string RenderQwen35(List<ChatMessage> messages, bool addGenerationPrompt = true,
            bool enableThinking = false)
        {
            var sb = new StringBuilder();
            foreach (var msg in messages)
            {
                sb.Append($"<|im_start|>{msg.Role}\n");
                if (msg.ImagePaths != null && msg.ImagePaths.Count > 0)
                {
                    foreach (var _ in msg.ImagePaths)
                    {
                        sb.Append("<|vision_start|><|image_pad|><|vision_end|>");
                    }
                }
                sb.Append($"{msg.Content}<|im_end|>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|im_start|>assistant\n");
                if (enableThinking)
                    sb.Append("<think>\n");
                else
                    sb.Append("<think>\n\n</think>\n\n");
            }
            return sb.ToString();
        }

        /// <summary>
        /// Render Gemma3 chat template.
        /// Uses &lt;start_of_turn&gt;/&lt;end_of_turn&gt; markers. Images use &lt;start_of_image&gt;.
        /// BOS token is prepended by the tokenizer (add_bos_token=true).
        /// </summary>
        public static string RenderGemma3(List<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < messages.Count; i++)
            {
                var msg = messages[i];
                string role = msg.Role == "assistant" ? "model" : msg.Role;
                sb.Append($"<start_of_turn>{role}\n");
                if (msg.ImagePaths != null)
                {
                    foreach (var _ in msg.ImagePaths)
                        sb.Append("<start_of_image>");
                }
                sb.Append($"{msg.Content}<end_of_turn>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<start_of_turn>model\n");
            }
            return sb.ToString();
        }

        public static string RenderFromGgufTemplate(string template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string architecture = null)
        {
            if (architecture == "gemma3")
                return RenderGemma3(messages, addGenerationPrompt);

            if (architecture == "gemma4")
                return RenderGemma4(messages, addGenerationPrompt);

            if (architecture == "qwen35" || architecture == "qwen35moe" || architecture == "qwen3next" ||
                architecture == "qwen3vl" || architecture == "qwen3vlmoe")
            {
                return RenderQwen35(messages, addGenerationPrompt);
            }

            return RenderQwen3(messages, addGenerationPrompt);
        }

        /// <summary>
        /// Render Gemma4 chat template.
        /// Uses &lt;|turn&gt;/&lt;turn|&gt; markers. Images use &lt;|image&gt;.
        /// </summary>
        public static string RenderGemma4(List<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            var sb = new StringBuilder();
            foreach (var msg in messages)
            {
                string role = msg.Role == "assistant" ? "model" : msg.Role;
                sb.Append($"<|turn>{role}\n");
                if (msg.ImagePaths != null)
                {
                    if (msg.IsVideo)
                        sb.Append("<|video>");
                    foreach (var _ in msg.ImagePaths)
                        sb.Append("<|image>");
                }
                if (msg.AudioPaths != null)
                {
                    foreach (var _ in msg.AudioPaths)
                        sb.Append("<|audio>");
                }
                sb.Append($"{msg.Content}<turn|>\n");
            }
            if (addGenerationPrompt)
            {
                sb.Append("<|turn>model\n");
            }
            return sb.ToString();
        }

        /// <summary>
        /// Expand image pad tokens in a token sequence.
        /// Replaces each single imagePadTokenId with tokenCounts[i] copies.
        /// </summary>
        public static List<int> ExpandImageTokens(List<int> tokens, int imagePadTokenId, int[] tokenCounts)
        {
            var result = new List<int>(tokens.Count + 1024);
            int imageIdx = 0;
            foreach (int token in tokens)
            {
                if (token == imagePadTokenId && imageIdx < tokenCounts.Length)
                {
                    int count = tokenCounts[imageIdx++];
                    for (int j = 0; j < count; j++)
                        result.Add(imagePadTokenId);
                }
                else
                {
                    result.Add(token);
                }
            }
            return result;
        }

        /// <summary>
        /// Expand Gemma3 image tokens: replace each &lt;start_of_image&gt; token with
        /// \n\n &lt;start_of_image&gt; [pad_tokens...] &lt;end_of_image&gt; \n\n
        /// </summary>
        public static List<int> ExpandGemma3ImageTokens(List<int> tokens, int startOfImageId,
            int endOfImageId, int newlineNewlineId, int padTokenId, int tokensPerImage)
        {
            var result = new List<int>(tokens.Count + tokensPerImage + 10);
            foreach (int token in tokens)
            {
                if (token == startOfImageId)
                {
                    result.Add(newlineNewlineId);
                    result.Add(startOfImageId);
                    for (int j = 0; j < tokensPerImage; j++)
                        result.Add(padTokenId);
                    result.Add(endOfImageId);
                    result.Add(newlineNewlineId);
                }
                else
                {
                    result.Add(token);
                }
            }
            return result;
        }
    }
}
