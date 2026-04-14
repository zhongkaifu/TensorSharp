using System;
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace TensorSharp.Server
{
    internal sealed class TextUploadResult
    {
        public string TextContent { get; init; }
        public bool Truncated { get; init; }
        public int TruncateLimit { get; init; }
        public string TruncateUnit { get; init; }
        public int? ModelContextLimit { get; init; }
        public int? OriginalTokenCount { get; init; }
        public int? ReturnedTokenCount { get; init; }
    }

    internal static class TextUploadHelper
    {
        internal static TextUploadResult PrepareTextContent(
            string textContent,
            ITokenizer tokenizer,
            int modelContextLimit,
            int fallbackCharLimit)
        {
            if (tokenizer != null && modelContextLimit > 0)
            {
                int tokenLimit = Math.Max(1, modelContextLimit / 2);
                List<int> tokens = tokenizer.Encode(textContent ?? string.Empty, addSpecial: false);
                bool truncated = tokens.Count > tokenLimit;

                if (truncated)
                    textContent = tokenizer.Decode(tokens.GetRange(0, tokenLimit));

                List<int> returnedTokens = tokenizer.Encode(textContent ?? string.Empty, addSpecial: false);
                return new TextUploadResult
                {
                    TextContent = textContent,
                    Truncated = truncated,
                    TruncateLimit = tokenLimit,
                    TruncateUnit = "tokens",
                    ModelContextLimit = modelContextLimit,
                    OriginalTokenCount = tokens.Count,
                    ReturnedTokenCount = returnedTokens.Count
                };
            }

            string safeText = textContent ?? string.Empty;
            bool charTruncated = safeText.Length > fallbackCharLimit;
            if (charTruncated)
                safeText = safeText.Substring(0, fallbackCharLimit);

            return new TextUploadResult
            {
                TextContent = safeText,
                Truncated = charTruncated,
                TruncateLimit = fallbackCharLimit,
                TruncateUnit = "characters",
                ModelContextLimit = null,
                OriginalTokenCount = null,
                ReturnedTokenCount = null
            };
        }
    }
}
