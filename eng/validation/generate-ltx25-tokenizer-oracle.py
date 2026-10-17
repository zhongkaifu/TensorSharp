#!/usr/bin/env python3
"""CPU-only multilingual LTX tokenizer oracle; reads embedded assets, never model weights.

Use transformers 5.10.1 and the unchanged official LTX checkout. The fixture
records the asset digest rather than duplicating its 32 MB vocabulary in Git.
"""
import argparse
import ast
from enum import Enum
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import tokenizers
from tokenizers import Tokenizer
import transformers
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"
USER_PROMPT = "请让图中的女人来个飞吻。她面向镜头微笑，先将右手指尖轻触嘴唇，噘嘴，再把手掌向镜头伸出，做一个清晰的飞吻动作，然后放下手并微笑。保持人物的面容、发型和服装，镜头稳定。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("InferenceWeb.Tests/Fixtures/Ltx25/tokenizer-multilingual.json"))
    args = parser.parse_args()
    assert transformers.__version__ == "5.10.1", "Use the pinned transformers 5.10.1 reference environment"
    assert subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip()
    with args.encoder.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > 16 * 1024 * 1024:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(stream.read(header_size))

        def asset(name):
            descriptor = header[name]
            begin, end = descriptor["data_offsets"]
            assert descriptor["dtype"] == "U8" and end - begin < 64 * 1024 * 1024
            stream.seek(8 + header_size + begin)
            data = stream.read(end - begin)
            assert len(data) == end - begin
            return data

        tokenizer_json = asset("tokenizer_json")
        config_json = asset("hf_asset__tokenizer_config.json")
    assets = SimpleNamespace(tokenizer_json=tokenizer_json, sidecars={"tokenizer_config.json": config_json})
    assets.sidecar_json = lambda name: json.loads(assets.sidecars[name])
    folder = args.reference / "packages/ltx-core/src/ltx_core/text_encoders/gemma"
    namespace = dict(json=json, Enum=Enum, GemmaAssets=SimpleNamespace, Tokenizer=Tokenizer,
                     PreTrainedTokenizerBase=PreTrainedTokenizerBase, PreTrainedTokenizerFast=PreTrainedTokenizerFast)
    hashes = {}
    for filename, names in {
        "gemma_assets.py": {"TOKENIZER_MAX_LENGTH", "_TOKENIZER_CONFIG_SKIP", "_chat_template_from_assets", "build_gemma_hf_tokenizer"},
        "tokenizer.py": {"PaddingSide", "LTXGemmaTokenizer"},
    }.items():
        source = (folder / filename).read_bytes()
        hashes[filename] = hashlib.sha256(source).hexdigest()
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if
                     isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names or
                     isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
        assert len(tree.body) == len(names)
        exec(compile(ast.fix_missing_locations(tree), str(folder / filename), "exec"), namespace)
    hf = namespace["build_gemma_hf_tokenizer"](assets)
    official = namespace["LTXGemmaTokenizer"](hf, max_length=1024)
    cases = []
    for name, prompt in [
        ("reported_chinese_prompt", USER_PROMPT),
        ("reported_prompt_surrounding_unicode_whitespace", "\u3000\t\n" + USER_PROMPT + "\r\n\u00a0"),
        ("mixed_chinese_english", "让她保持自然的五指手掌。A woman blows a kiss, then lowers her right hand."),
        ("japanese_korean", "彼女が笑顔で手を振る。 손가락 다섯 개가 자연스럽게 보인다."),
        ("unicode_emoji_and_accents", "Café / Cafe\u0301：微笑🙂，五根手指🖐🏽。"),
        ("interior_whitespace", "  镜头\t稳定。\n\nShe   smiles.\r\nThen lowers her hand.  "),
        ("explicit_bos", "<bos>" + USER_PROMPT),
        ("explicit_eos_inside_prompt", "微笑<eos>然后挥手"),
        ("empty", ""),
        ("whitespace_only", " \t\r\n\u3000\u00a0"),
        ("truncated_multilingual", (USER_PROMPT + " She gently lowers her hand.\n") * 30),
    ]:
        pairs = official.tokenize_with_weights(prompt)["gemma"]
        ids = [int(token) for token, mask in pairs if mask]
        raw = hf(prompt.strip(), padding=False, truncation=False).input_ids
        if not raw or raw[0] != hf.bos_token_id:
            raw = [hf.bos_token_id, *raw]
        cases.append(dict(name=name, prompt=prompt, trimmed_prompt=prompt.strip(), ids=ids,
                          untruncated_tokens=len(raw), truncated=len(raw) > 1024,
                          padded_ids=[int(token) for token, mask in pairs], attention_mask=[int(mask) for token, mask in pairs],
                          valid_position_ids=list(range(1024 - len(ids), 1024))))
    report = dict(source_revision=REVISION, source_sha256=hashes, transformers_version=transformers.__version__,
                  tokenizers_version=tokenizers.__version__, tokenizer_json_bytes=len(tokenizer_json),
                  tokenizer_json_sha256=hashlib.sha256(tokenizer_json).hexdigest(),
                  tokenizer_config_sha256=hashlib.sha256(config_json).hexdigest(),
                  model_tensor_bytes_read=0, maximum_length=1024, padding_side=hf.padding_side,
                  pad_token_id=hf.pad_token_id, bos_token_id=hf.bos_token_id, eos_token_id=hf.eos_token_id,
                  cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(cases=len(cases), tokenizer_bytes=len(tokenizer_json), model_tensor_bytes_read=0,
                         reported_prompt_tokens=len(cases[0]["ids"]), output=str(args.output))))


if __name__ == "__main__":
    main()
