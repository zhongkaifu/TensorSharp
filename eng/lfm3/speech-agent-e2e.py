#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Agentic spoken web chat turns against a running TensorSharp.Server.Host (LFM3-Omni with its mmproj, --code-exec):
each scenario (reasoning, a named skill, code run by the host, a file created and edited in the workspace, a multi-turn
session mixing them) runs through /api/chat as the Web UI sends it, once with speech off and once with speech on, each
in a fresh session, and the two are compared turn by turn: the answer text, the reasoning, the tool steps (tool_progress
and skill_step frames, with what ran), the token counts, the prompt sizes and the KV reuse. Per spoken turn: the frames'
order (audio during tool rounds), the WAV the done frame names (downloaded, parsed, as long as the audio frames), the
time to the first audio against the time to the first answer token, and what was spoken (read from the server's log
line "spoken turn done ... spoken=" when --server-log is given). Writes report.json and manifest.tsv (for
speech-whisper.py) to --output, which must be under the ignored artifacts/ or docs/validation/.

  python3 eng/lfm3/speech-agent-e2e.py --url http://127.0.0.1:18790 --output artifacts/lfm3-speech-agent/run1 \\
      --server-log artifacts/lfm3-speech-agent/run1/server.log [--lang en,zh] [--scenarios think,skill,code,file,multi]
"""
import argparse
import base64
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.request

SCENARIOS = {
    "en": {
        "plain": {"turns": ["What is the capital of France? Answer in one short sentence."], "expect": ["Paris"]},
        "think": {"think": True, "turns": [
            "Tom has 3 apples and buys 2 bags with 4 apples in each. How many apples does he have now? Answer briefly."],
            "expect": ["11"]},
        "skill": {"skills": ["internal-comms"], "turns": [
            "Use the internal-comms skill to write a very short 3P update for my team: we shipped the login page, "
            "next we build the settings page, and the build server is slow."], "expect": []},
        "code": {"turns": [
            "Use Python to compute the sum of the squares of the integers from 1 to 50, then tell me the result in one "
            "sentence."], "expect": ["42925", "42,925"]},
        "file": {"turns": [
            "In the workspace, create a file named shopping.txt that lists apples, bread and milk, one per line. Then "
            "change bread to rice in that file and tell me what the file contains now."], "expect": ["rice"],
            "file": ("shopping.txt", ["apples", "rice", "milk"])},
        "multi": {"turns": [
            "Use Python to compute 37 times 43.",
            "Now write that number into a file called answer.txt in the workspace.",
            "Thanks! In one sentence, what is in answer.txt, and what is the capital of France?"],
            "expect": ["1591"], "file": ("answer.txt", ["1591"])},
    },
    "zh": {
        "plain": {"turns": ["法国的首都是哪里？请用一句话回答。"], "expect": ["巴黎"]},
        "think": {"think": True, "turns": ["汤姆有3个苹果，又买了2袋，每袋4个。他现在一共有多少个苹果？请简要回答。"],
                  "expect": ["11", "十一"]},
        "skill": {"skills": ["internal-comms"], "turns": [
            "请用 internal-comms 技能为我的团队写一份很短的 3P 更新：我们上线了登录页，接下来做设置页，构建服务器很慢。"],
            "expect": []},
        "code": {"turns": ["请用 Python 计算 1 到 50 的整数的平方和，然后用一句话告诉我结果。"], "expect": ["42925", "42,925"]},
        "file": {"turns": [
            "在工作区里创建一个名为 shopping.txt 的文件，每行一个：苹果、面包、牛奶。然后把文件里的面包改成米饭，"
            "告诉我文件现在的内容。"], "expect": ["米饭"], "file": ("shopping.txt", ["苹果", "米饭", "牛奶"])},
        "multi": {"turns": [
            "请用 Python 计算 37 乘以 43。",
            "现在把这个数字写进工作区里一个叫 answer.txt 的文件。",
            "谢谢！用一句话说说 answer.txt 里是什么，以及法国的首都是哪里？"],
            "expect": ["1591"], "file": ("answer.txt", ["1591"])},
    },
}


# Single-turn English questions (speech-lang-eval.py's), each in a session of its own: answer quality, speech and time
# to first audio of the two ways an English reply can be spoken.
EVAL_EN = [
    "What is the capital of France?",
    "Give me one tip for sleeping better.",
    "Tell me a very short story about a cat.",
    "How many days are there in a leap year?",
    "What should I pack for a weekend trip to the beach?",
    "Explain what machine learning is in one sentence.",
]

# Conversations that switch Speak replies on and off between turns ("s" spoken, "t" text): each turn should reuse the
# previous turn's whole prompt and reply.
MIXED = {
    "mix12": "sst",
    "mix13": "stst" + "t",
    "mix24": "tstst",
}
MIXED_QUESTIONS = {
    "en": ["What is the capital of France? Answer in one short sentence.",
           "And what is the capital of Germany? Answer in one short sentence.",
           "Which of the two cities did you name first?",
           "How many people live in that city, roughly?",
           "Thanks. Say goodbye in one short sentence."],
    "zh": ["法国的首都是哪里？请用一句话回答。", "那德国的首都呢？请用一句话回答。", "你先说的是哪个城市？",
           "那个城市大约有多少人口？", "谢谢，用一句话说再见。"],
}


def run_pattern(base, lang, name, pattern, voice, max_tokens, out):
    """One conversation whose turn i is spoken when pattern[i] is "s": what each turn reused of its prompt."""
    session = json.load(post(base + "/api/sessions", {}))["sessionId"]
    history, turns = [], []
    for index, (mode, question) in enumerate(zip(pattern, MIXED_QUESTIONS[lang])):
        history.append({"role": "user", "content": question})
        body = {"messages": history, "maxTokens": max_tokens, "think": False, "replaceActive": True, "sessionId": session}
        if index == 0:
            body["newChat"] = True
        if mode == "s":
            body["speech"] = {"voice": voice} if voice else True
        frames, seconds = chat(base, body)
        done = next((f for f in frames if f.get("done")), {})
        text = "".join(f.get("token", "") for f in frames)
        message = {"role": "assistant", "content": text}
        for key in ("audioId", "audioUrl", "audioSeconds"):
            if done.get(key) is not None:
                message[key] = done[key]
        history.append(message)
        turns.append({"mode": mode, "question": question, "text": text, "promptTokens": done.get("promptTokens"),
                      "kvReusedTokens": done.get("kvReusedTokens"), "tokenCount": done.get("tokenCount"),
                      "error": done.get("error"), "seconds": round(seconds, 3),
                      "firstAudioSeconds": next((f["_t"] for f in frames if "audio" in f), None)})
        print("  %s %s t%d %s: prompt=%s reused=%s text=%r" % (lang, name, index + 1, mode, done.get("promptTokens"),
                                                             done.get("kvReusedTokens"), text[:60]), flush=True)
    # Each turn's prompt is the previous prompt, its reply and the new question: reused >= the previous prompt.
    for i in range(1, len(turns)):
        prev, cur = turns[i - 1], turns[i]
        cur["reusesPreviousPrompt"] = (cur["kvReusedTokens"] or 0) >= (prev["promptTokens"] or 0)
    return {"session": session, "pattern": pattern, "turns": turns}


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1200)


def get(url):
    return urllib.request.urlopen(url, timeout=600)


def chat(base, body):
    """The turn's frames, each with the client-side seconds since the request was sent."""
    frames = []
    t0 = time.monotonic()
    with post(base + "/api/chat", body) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("data: "):
                frame = json.loads(line[6:])
                frame["_t"] = time.monotonic() - t0
                frames.append(frame)
    return frames, time.monotonic() - t0


def wav_info(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    channels, rate = struct.unpack_from("<HI", data, 22)
    bits = struct.unpack_from("<H", data, 34)[0]
    size = struct.unpack_from("<I", data, 40)[0]
    return {"channels": channels, "rate": rate, "bits": bits, "dataBytes": size,
            "seconds": size / (rate * channels * bits / 8)}


def steps(frames):
    """The turn's tool steps as the page shows them: each tool's progress phases (writing, running with what ran,
    finished), collapsed, and every skill_step."""
    out = []
    for f in frames:
        item = None
        if "tool_progress" in f:
            phase = f["tool_progress"]
            item = "%s:%s:%s" % (phase, f.get("tool") or "", f.get("detail") or "" if phase == "running" else "")
        elif "skill_step" in f:
            item = "skill:%s:%s:%s" % (f.get("skill_step"), f.get("skill") or f.get("path") or "", f.get("ok"))
        if item is None:
            continue
        if out and (out[-1] == item or (item.startswith("writing:") and out[-1].startswith("writing:")
                                        and (out[-1] == "writing::" or out[-1] == item))):
            out[-1] = item
            continue
        out.append(item)
    return out


def order(frames):
    code = []
    for f in frames:
        c = ("t" if "token" in f else "a" if "audio" in f else "k" if "thinking" in f else "p" if "tool_progress" in f
             else "s" if "skill_step" in f else "r" if "replace" in f else "n" if "notice" in f
             else "d" if f.get("done") else "?")
        if not code or code[-1][0] != c:
            code.append([c, 1])
        else:
            code[-1][1] += 1
    return " ".join("%s%d" % (c, n) if n > 1 else c for c, n in code)


def spoken_from_log(path):
    spoken = {}
    if not path or not os.path.exists(path):
        return spoken
    pattern = re.compile(r'spoken turn done: .*?audio=(audio_[0-9a-f]+) .*? spoken="(.*)"\s*$')
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                spoken[m.group(1)] = m.group(2).replace(" | ", " ")
    return spoken


class TtsLoad:
    """Unrelated speech synthesis (/v1/audio/speech, one request at a time) for as long as it runs: a text turn whose
    engine steps are shared with TTS requests, as a spoken turn's are, without being spoken itself."""

    def __init__(self, base):
        self.base, self.stop, self.requests = base, threading.Event(), 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                post(self.base + "/v1/audio/speech", {"input": "This sentence is spoken while another request is generated.",
                                                      "voice": "female", "response_format": "pcm"}).read()
                self.requests += 1
            except Exception:  # noqa: BLE001 - load only
                time.sleep(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=60)


def run_arm(base, lang, name, scenario, speech, voice, max_tokens, out, load=False):
    if load:
        with TtsLoad(base) as tts:
            result = run_arm(base, lang, name, scenario, speech, voice, max_tokens, out)
        result["ttsLoadRequests"] = tts.requests
        return result
    session = json.load(post(base + "/api/sessions", {}))["sessionId"]
    history = []
    turns = []
    for index, question in enumerate(scenario["turns"]):
        history.append({"role": "user", "content": question})
        body = {"messages": history, "maxTokens": max_tokens, "think": bool(scenario.get("think")),
                "replaceActive": True, "sessionId": session}
        if index == 0:
            body["newChat"] = True
        if scenario.get("skills"):
            body["skills"] = scenario["skills"]
        if speech:
            body["speech"] = {"voice": voice} if voice else True
        frames, seconds = chat(base, body)
        done = next((f for f in frames if f.get("done")), {})
        text = "".join(f.get("token", "") for f in frames)
        for f in frames:
            if "replace" in f:
                text = f["replace"]
        thinking = "".join(f.get("thinking", "") for f in frames)
        audio = [f for f in frames if "audio" in f]
        pcm = b"".join(base64.b64decode(f["audio"]) for f in audio)
        first_token = next((f["_t"] for f in frames if f.get("token")), None)
        last_token = next((f["_t"] for f in reversed(frames) if f.get("token")), None)
        first_audio = audio[0]["_t"] if audio else None
        turn = {
            "question": question, "text": text, "thinking": thinking, "steps": steps(frames), "order": order(frames),
            "notices": [{"code": f.get("code"), "notice": f.get("notice")} for f in frames if "notice" in f],
            "toolCalls": [f["tool_calls"] for f in frames if "tool_calls" in f],
            "seconds": round(seconds, 3), "firstTokenSeconds": first_token, "lastTokenSeconds": last_token,
            "firstAudioSeconds": first_audio,
            "audioFrames": len(audio), "streamedAudioSeconds": len(pcm) / 2 / audio[0]["sampleRate"] if audio else 0,
            "audioDuringTools": any("audio" in f for f in frames[next((i for i, g in enumerate(frames) if "tool_progress" in g), len(frames)):
                                                                  max((i for i, g in enumerate(frames) if g.get("tool_progress") == "finished"), default=-1) + 1]),
        }
        for key in ("tokenCount", "elapsed", "tokPerSec", "promptTokens", "kvReusedTokens", "kvReusePercent", "truncated",
                    "error", "aborted", "audioId", "audioUrl", "audioSeconds"):
            turn[key] = done.get(key)
        if speech and done.get("audioUrl"):
            data = get(base + done["audioUrl"]).read()
            wav = os.path.join(out, "%s-%s-t%d.wav" % (lang, name, index + 1))
            with open(wav, "wb") as f:
                f.write(data)
            turn["wav"] = wav_info(data)
            turn["wavFile"] = wav
            turn["wavMatchesFrames"] = turn["wav"]["dataBytes"] == len(pcm)
        message = {"role": "assistant", "content": text}
        if thinking:
            message["thinking"] = thinking
        if done.get("audioId"):
            message["audioId"] = done["audioId"]
        if done.get("audioUrl"):
            message["audioUrl"] = done["audioUrl"]
            message["audioSeconds"] = done.get("audioSeconds")
        history.append(message)
        turns.append(turn)
        print("  %s %s %s t%d: %.1fs tok=%s tps=%s prompt=%s reused=%s first_tok=%s first_audio=%s audio=%ss steps=%d text=%r" % (
            lang, name, "on " if speech else "off", index + 1, seconds, turn["tokenCount"], turn["tokPerSec"] and round(turn["tokPerSec"], 1),
            turn["promptTokens"], turn["kvReusedTokens"], first_token and round(first_token, 3),
            first_audio and round(first_audio, 3), turn["audioSeconds"], len(turn["steps"]), text[:100]), flush=True)
    return {"session": session, "turns": turns}


def compare(off, on):
    rows = []
    for i, (a, b) in enumerate(zip(off["turns"], on["turns"])):
        rows.append({
            "turn": i + 1,
            "textIdentical": a["text"] == b["text"],
            "thinkingIdentical": a["thinking"] == b["thinking"],
            "stepsIdentical": a["steps"] == b["steps"],
            "tokenCountIdentical": a["tokenCount"] == b["tokenCount"],
            "promptTokens": [a["promptTokens"], b["promptTokens"]],
            "kvReused": [a["kvReusedTokens"], b["kvReusedTokens"]],
        })
    return rows


def check_file(root, spec, runs):
    """Whether each arm's session workspace holds the file the scenario asked for, with the expected lines."""
    if not root or not spec:
        return None
    name, needles = spec
    found = {}
    for arm, run in runs.items():
        hits = []
        for dirpath, _, files in os.walk(root):
            if run["session"] in dirpath and name in files:
                hits.append(os.path.join(dirpath, name))
        if not hits:
            found[arm] = {"found": False}
            continue
        with open(hits[0], encoding="utf-8", errors="replace") as f:
            content = f.read()
        found[arm] = {"found": True, "ok": all(n in content for n in needles), "content": content[:200]}
    return found


WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def openai_checks(base, model, out):
    """A client tool with audio on /v1/chat/completions: the reply's tool calls, and the speech of its content."""
    results = {}
    for lang, question in (("en", "What is the weather in Paris right now? Use the tool."),
                           ("zh", "巴黎现在天气怎么样？请使用工具。")):
        messages = [{"role": "user", "content": question}]
        body = {"model": model, "messages": messages, "tools": WEATHER_TOOL, "modalities": ["text", "audio"],
                "audio": {"voice": "female", "format": "pcm16"}, "stream": True, "max_tokens": 512}
        t0 = time.monotonic()
        chunks = []
        with post(base + "/v1/chat/completions", body) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if line.startswith("data: {"):
                    c = json.loads(line[6:])
                    c["_t"] = time.monotonic() - t0
                    chunks.append(c)
        deltas = [c["choices"][0].get("delta", {}) for c in chunks if c.get("choices")]
        calls = [d["tool_calls"] for d in deltas if d.get("tool_calls")]
        transcript = "".join((d.get("audio") or {}).get("transcript", "") for d in deltas)
        audio = [c for c, d in zip(chunks, deltas) if (d.get("audio") or {}).get("data")]
        finish = next((c["choices"][0].get("finish_reason") for c in reversed(chunks) if c.get("choices")), None)
        stream_result = {"toolCalls": calls, "transcript": transcript, "audioChunks": len(audio),
                         "firstAudioSeconds": audio[0]["_t"] if audio else None, "finish": finish,
                         "content": "".join(d.get("content") or "" for d in deltas)}
        # The client runs the tool and asks again, with audio: the answer is spoken.
        call = calls[0][0] if calls else None
        follow = None
        if call:
            history = messages + [{"role": "assistant", "content": transcript or None, "tool_calls": [{
                "id": call.get("id") or "call_1", "type": "function",
                "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]},
                {"role": "tool", "tool_call_id": call.get("id") or "call_1",
                 "content": json.dumps({"city": "Paris", "temperature_c": 18, "sky": "light rain"})}]
            whole = json.load(post(base + "/v1/chat/completions", {
                "model": model, "messages": history, "tools": WEATHER_TOOL, "modalities": ["text", "audio"],
                "audio": {"voice": "female", "format": "wav"}, "max_tokens": 512}))
            msg = whole["choices"][0]["message"]
            audio_obj = msg.get("audio") or {}
            wav = None
            if audio_obj.get("data"):
                wav = os.path.join(out, "openai-%s-answer.wav" % lang)
                with open(wav, "wb") as f:
                    f.write(base64.b64decode(audio_obj["data"]))
            follow = {"toolCalls": msg.get("tool_calls"), "transcript": audio_obj.get("transcript"), "wav": wav,
                      "wavInfo": wav_info(open(wav, "rb").read()) if wav else None,
                      "finish": whole["choices"][0].get("finish_reason"), "usage": whole.get("usage")}
        results[lang] = {"stream": stream_result, "afterTool": follow}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--lang", default="en,zh")
    parser.add_argument("--scenarios", default="think,skill,code,file,multi")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workspace-root", default=None, help="where session workspaces live, to check files")
    parser.add_argument("--repeat", type=int, default=1, help="run each scenario's off/on pair this many times")
    parser.add_argument("--arms", default="off,on",
                        help="off,on (compare), off (a model that does not speak), and load (speech off beside TTS requests)")
    parser.add_argument("--openai", action="store_true",
                        help="also ask /v1/chat/completions for audio with a client tool, streamed and whole")
    args = parser.parse_args()
    out = os.path.abspath(args.output)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not any(out.startswith(os.path.join(repo, d) + os.sep) for d in ("artifacts", "docs/validation")):
        sys.exit("--output must be under the ignored artifacts/ or docs/validation/ folders")
    os.makedirs(out, exist_ok=True)
    base = args.url.rstrip("/")
    models = json.load(get(base + "/api/models"))
    report = {"url": base, "model": models.get("loaded"), "mmproj": models.get("loadedMmProj"),
              "backend": models.get("loadedBackend"), "speechOutput": models.get("speechOutput"), "scenarios": []}
    report["eval"] = []
    report["mixed"] = []
    for lang in args.lang.split(","):
        for name in args.scenarios.split(","):
            if name in MIXED:
                report["mixed"].append({"lang": lang, "name": name,
                                        **run_pattern(base, lang, name, MIXED[name], args.voice, args.max_tokens, out)})
                continue
            if name == "eval":
                if lang != "en":
                    continue
                for qi, question in enumerate(EVAL_EN):
                    scenario = {"turns": [question]}
                    runs = {}
                    for speech in [False, True]:
                        if ("on" if speech else "off") in args.arms.split(","):
                            runs["on" if speech else "off"] = run_arm(base, lang, "eval%d-%s" % (qi + 1, args.voice or "female"),
                                                                     scenario, speech, args.voice, args.max_tokens, out)
                    report["eval"].append({"lang": lang, "question": question, "voice": args.voice or "female",
                                           "off": runs.get("off"), "on": runs.get("on"),
                                           "compare": compare(runs["off"], runs["on"]) if len(runs) == 2 else None})
                continue
            scenario = SCENARIOS[lang][name]
            for rep in range(args.repeat):
                # Alternate which arm goes first, so neither always meets the other's cached prefix. "load" is speech
                # off with unrelated TTS requests running beside it (the batching a spoken turn's text shares).
                order = ["off", "on", "load"] if rep % 2 == 0 else ["load", "on", "off"]
                runs = {}
                for arm in [a for a in order if a in args.arms.split(",")]:
                    runs[arm] = run_arm(base, lang, name + ("" if rep == 0 else "-r%d" % rep) + ("-load" if arm == "load" else ""),
                                        scenario, arm == "on", args.voice, args.max_tokens, out, load=arm == "load")
                entry = {"lang": lang, "scenario": name, "repeat": rep, "think": bool(scenario.get("think")),
                         "skills": scenario.get("skills"), "off": runs.get("off"), "on": runs.get("on"),
                         "load": runs.get("load"),
                         "compare": compare(runs["off"], runs["on"]) if "off" in runs and "on" in runs else None,
                         "compareLoad": compare(runs["on"], runs["load"]) if "on" in runs and "load" in runs else None}
                expect = scenario.get("expect") or []
                entry["expectFound"] = {}
                for arm, run in runs.items():
                    final = "".join(t["text"] for t in run["turns"])
                    entry["expectFound"][arm] = any(e in final for e in expect) if expect else None
                entry["workspaceFile"] = check_file(args.workspace_root, scenario.get("file"), runs)
                report["scenarios"].append(entry)
    if args.openai:
        report["openai"] = openai_checks(base, models.get("loaded"), out)
    spoken = spoken_from_log(args.server_log)
    with open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8") as m:
        for entry in report["eval"]:
            turn = (entry["on"] or {"turns": [{}]})["turns"][0]
            # An interleaved reply's speech says its transcript; one spoken from its text says what the log records.
            turn["spoken"] = spoken.get(turn.get("audioId")) or turn.get("text")
            if turn.get("wavFile") and turn["spoken"]:
                m.write("%s\t%s\t%s\t%s\n" % (os.path.splitext(os.path.basename(turn["wavFile"]))[0], entry["lang"],
                                              os.path.relpath(turn["wavFile"], out), turn["spoken"].replace("\t", " ").replace("\n", " ")))
        for entry in report["scenarios"]:
            for i, turn in enumerate((entry["on"] or {"turns": []})["turns"]):
                turn["spoken"] = spoken.get(turn.get("audioId"))
                if turn.get("wavFile") and turn["spoken"]:
                    m.write("%s-%s-r%d-t%d\t%s\t%s\t%s\n" % (entry["lang"], entry["scenario"], entry["repeat"], i + 1,
                                                          entry["lang"], os.path.relpath(turn["wavFile"], out),
                                                          turn["spoken"].replace("\t", " ")))
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for entry in report["scenarios"]:
        print(entry["lang"], entry["scenario"], "r%d" % entry["repeat"], json.dumps(entry["compare"]), "expect",
              entry["expectFound"], "file", entry["workspaceFile"])
        if entry.get("compareLoad"):
            print("   on vs off-under-TTS-load", json.dumps(entry["compareLoad"]))
    for entry in report["mixed"]:
        print("mixed", entry["lang"], entry["name"], entry["pattern"],
              ["%s %s/%s" % (t["mode"], t["kvReusedTokens"], t["promptTokens"]) for t in entry["turns"]])
    for entry in report["eval"]:
        if entry["compare"]:
            print("eval", entry["voice"], entry["question"][:40], json.dumps(entry["compare"][0]))
    if args.openai:
        print("openai", json.dumps(report["openai"], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
