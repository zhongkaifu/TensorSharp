#!/usr/bin/env python3
"""Regenerate InferenceWeb.Tests/Fixtures/Glm4Tokenizer/reference.json.

Usage: python glm4-tokenizer-fixture.py <tokenizer.json> <reference.json>
The tokenizer.json is zai-org/GLM-5's (MIT); its vocabulary and merges are identical
to the GLM-5.2, GLM-5.3 and GLM-5.3-Flash GGUFs. Needs HF `tokenizers`.
"""
import json, random, sys
import tokenizers
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
coder = "\U0001F9D1\U0001F3FD\u200d\U0001F4BB"
hand = [
 "INV-472", "silver-4821", "juniper-5938", " is 17 + 25", "1", "12", "123", "1234", "12345", "123456", "1234567",
 "3.14159", "-0.000123", "1e10", "6.02E+23", "$1,234,567.89", "+1 (415) 555-0199", "2026-09-17T12:34:56Z",
 "0x1F4A9", "v10.0.401", "192.168.0.1:8080", "\uff11\uff12\uff13\uff14\uff15", "\u0664\u0665\u0666\u0667", "\u06f1\u06f2\u06f3\u06f4",
 "\u00b2\u00b3\u00b9\u2074", "\u216b\u2169\u2160", "\u2460\u2461\u2462\u2463", "\u00bc\u00bd\u00be",
 "\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u4ef7\u683c\u662f12345\u5143\u3002", "\u7b2c3\u7ae0\u7b2c45\u8282",
 "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8123\u3067\u3059", "\u30ab\u30bf\u30ab\u30ca\u3068\u3072\u3089\u304c\u306a",
 "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8 2026\ub144",
 "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645 123", "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! 42",
 "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac 7", "\u0939\u093f\u0928\u094d\u0926\u0940 \u092a\u093e\u0920 \u0967\u0968\u0969", "\u0e44\u0e17\u0e22 \u0e51\u0e52\u0e53",
 "don't", "DON'T", "we'll", "WE'LL", "I'm", "I'M", "they're", "THEY'RE", "you've", "it'd", "O'Neil's", "'s", "'S", "'\u017f", "rock 'n' roll",
 "def f(x):\n    return x**2 + 1\n", "for (int i = 0; i < 100; i++) {\n\tsum += a[i];\n}", "{\"name\": \"Mars\", \"moons\": 2}",
 "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
 "  leading", "trailing  ", "a  b   c    d", "\n\n\n", "\r\n\r\n", " \n \n ", "\t\t\tx", "x\u00a0y", "x\u3000y", "x\u2003\u2003y",
 "\U0001F600", "hello\U0001F600world", " \U0001F600", coder + " coding", "\U0001F1FA\U0001F1F8\U0001F1E8\U0001F1F3",
 "e\u0301cole", "na\u00efve caf\u00e9", "Kelvin \u212a", "\u0130stanbul", "\u01c5emal",
 "a/b\\c|d", "!!!???", "...", "--flag=value", "path/to/file.txt\n", "email@example.com", "https://example.com/a?b=1&c=22",
 "mixed\u4e2d\u6587English123\u6570\u5b57", "abc123def4567ghi", "x1y22z333w4444", "123abc", "abc 123 456 7890",
 "  12  345  ", "\n123\n", "a\n\n  b", "end with newline\n", "tab\tseparated\tvalues\t1\t22\t333",
 "SELECT * FROM t WHERE id = 42;", "#include <stdio.h>\nint main(){return 0;}", "\u03bbx.x+1", "\u2211_{i=1}^{n} i\u00b2 = n(n+1)/2",
 "\u200bzero\u200bwidth", "\ue000private", "control\u0001char", "\u00ad soft hyphen", "a\u2028b\u2029c", "\u0085nel",
 "\U0001D7CE\U0001D7CF\U0001D7D0\U0001D7D1", "\U00010140\U00010141", "\U00020000\U00020001abc", "\U0001F468\u200d\U0001F469\u200d\U0001F467 family 3",
]
frags = ["INV-", "472", "4821", "12345678", "3.5", " ", "  ", "\n", "\r\n", "\t", "\u4e2d\u6587", "\u30c6\u30b9\u30c8", "\ud55c\uad6d",
         "\U0001F600", coder, "'s", "'LL", "don't", "Hello", "world", "_x", "{", "}", "(", ")", "[", "]", ":", ";", ",", ".", "!", "?", "\"", "'",
         "\uff11\uff12", "\u0663", "\u00b2", "\u216b", "\u00e9", "e\u0301", "\u00a0", "\u3000", "return", "def", "var", "=", "==", "+=", "->", "/", "\\",
         "0", "00", "000", "0000", "#", "@", "$", "%", "&", "*", "~", "`", "^", "|", "<", ">", "\u017f", "K", "\u212a", "\u0130", "\u0131",
         "\U0001D7CE", "\U00020000"]
rng = random.Random(20260917)
gen = []
for _ in range(320):
    n = rng.randint(1, 12)
    gen.append("".join(rng.choice(frags) for _ in range(n)))
texts = []
seen = set()
for t in hand + gen:
    if t not in seen:
        seen.add(t); texts.append(t)
out = []
for t in texts:
    enc = tok.encode(t, add_special_tokens=False)
    pre = tok.pre_tokenizer.pre_tokenize_str(t)
    pieces = [t[a:b] for _, (a, b) in pre]
    assert "".join(pieces) == t, (t, pieces)
    out.append({"text": t, "pieces": pieces, "ids": enc.ids})
source = ("Generated from zai-org/GLM-5 tokenizer.json (MIT) with HF tokenizers " + tokenizers.__version__ +
          ": pieces are pre_tokenizer.pre_tokenize_str offsets, ids are encode(add_special_tokens=False). "
          "That vocabulary and merge list are identical to the GLM-5.2, GLM-5.3 and GLM-5.3-Flash GGUFs "
          "(tokenizer.ggml.pre = glm4).")
with open(sys.argv[2], "w") as f:
    f.write('{\n "source": ' + json.dumps(source) + ',\n "cases": [\n')
    f.write(",\n".join("  " + json.dumps(c, ensure_ascii=True) for c in out))
    f.write("\n ]\n}\n")
print(len(out))
