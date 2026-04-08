#!/usr/bin/env python3
"""
Multi-Turn Chat Integration Tests for InferenceWeb

Runs comprehensive multi-turn conversation tests against all API surfaces,
validating response structure, context retention, and edge cases.

Usage:
    python3 test_multiturn.py [--model MODEL] [--url URL] [--max-tokens N]

Example:
    python3 test_multiturn.py --model gemma-4-E4B-it-Q8_0.gguf --url http://localhost:5000
"""

import argparse
import json
import sys
import time
import traceback
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class TestRunner:
    def __init__(self, base_url, model, max_tokens):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.passed = 0
        self.failed = 0

    def log(self, msg):
        print(f"{Colors.CYAN}[TEST]{Colors.RESET} {msg}")

    def ok(self, msg):
        print(f"{Colors.GREEN}[PASS]{Colors.RESET} {msg}")
        self.passed += 1

    def fail(self, msg):
        print(f"{Colors.RED}[FAIL]{Colors.RESET} {msg}")
        self.failed += 1

    def header(self, title):
        bar = "=" * 60
        print(f"\n{Colors.CYAN}{bar}\n {title}\n{bar}{Colors.RESET}")

    def _post_json(self, path, payload):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=120)
        return resp

    def _post_json_body(self, path, payload):
        resp = self._post_json(path, payload)
        return json.loads(resp.read().decode("utf-8"))

    def _post_sse_collect(self, path, payload):
        """POST and collect SSE events, return list of parsed JSON objects."""
        resp = self._post_json(path, payload)
        events = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def _post_ndjson_collect(self, path, payload):
        """POST and collect NDJSON lines, return list of parsed JSON objects."""
        resp = self._post_json(path, payload)
        items = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return items

    def _extract_sse_tokens(self, events):
        """Extract concatenated token text from SSE events."""
        tokens = ""
        for ev in events:
            if "token" in ev:
                tokens += ev["token"]
        return tokens

    def _extract_ndjson_content(self, items):
        """Extract concatenated content from Ollama NDJSON streaming response."""
        content = ""
        for item in items:
            msg = item.get("message", {})
            c = msg.get("content", "")
            if c:
                content += c
        return content

    def _has_done(self, events):
        return any(ev.get("done") is True for ev in events)

    # =========================================================================
    # Test: Web UI API - 5-turn basic conversation
    # =========================================================================
    def test_webui_5turn(self):
        self.header("Test 1: Web UI /api/chat - 5-Turn Conversation")

        messages = []
        turns = [
            "What is the tallest mountain in the world?",
            "How tall is it in meters?",
            "Has anyone climbed it without supplemental oxygen?",
            "Who was the first person to do so?",
            "What nationality was that person?",
        ]

        for i, question in enumerate(turns):
            turn = i + 1
            self.log(f"Turn {turn}/5: {question}")
            messages.append({"role": "user", "content": question})

            events = self._post_sse_collect("/api/chat", {
                "messages": messages,
                "maxTokens": self.max_tokens,
                "model": self.model,
            })

            content = self._extract_sse_tokens(events)
            has_done = self._has_done(events)

            if content and has_done:
                self.ok(f"Turn {turn}: {len(content)} chars, done=True")
                messages.append({"role": "assistant", "content": content})
            else:
                self.fail(f"Turn {turn}: content={len(content)} chars, done={has_done}")
                if content:
                    messages.append({"role": "assistant", "content": content})
                else:
                    break

        if len(messages) == 10:
            self.ok("Complete: 10 messages in history (5 user + 5 assistant)")
        else:
            self.fail(f"Incomplete: {len(messages)} messages (expected 10)")

    # =========================================================================
    # Test: Ollama API - context retention test
    # =========================================================================
    def test_ollama_context_retention(self):
        self.header("Test 2: Ollama API - Context Retention (4 Turns)")

        messages = []

        self.log("Turn 1: Establishing a fact")
        messages.append({"role": "user", "content": "Remember this number: 42. It is the answer to life, the universe, and everything."})
        r1 = self._post_json_body("/api/chat/ollama", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        })
        c1 = r1.get("message", {}).get("content", "")
        if c1 and r1.get("done"):
            self.ok(f"Turn 1: Established fact ({len(c1)} chars)")
            messages.append({"role": "assistant", "content": c1})
        else:
            self.fail("Turn 1: Failed")
            return

        self.log("Turn 2: Testing recall")
        messages.append({"role": "user", "content": "What number did I ask you to remember?"})
        r2 = self._post_json_body("/api/chat/ollama", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        })
        c2 = r2.get("message", {}).get("content", "")
        if c2:
            self.ok(f"Turn 2: Response received ({len(c2)} chars)")
            if "42" in c2:
                self.ok("Turn 2: Context retained - '42' found in response")
            else:
                self.fail(f"Turn 2: Context lost - '42' not found in: {c2[:200]}")
            messages.append({"role": "assistant", "content": c2})
        else:
            self.fail("Turn 2: No response")
            return

        self.log("Turn 3: Adding more context")
        messages.append({"role": "user", "content": "My name is Charlie. I live in Berlin."})
        r3 = self._post_json_body("/api/chat/ollama", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        })
        c3 = r3.get("message", {}).get("content", "")
        if c3:
            self.ok(f"Turn 3: Additional context set ({len(c3)} chars)")
            messages.append({"role": "assistant", "content": c3})
        else:
            self.fail("Turn 3: No response")
            return

        self.log("Turn 4: Multi-fact recall")
        messages.append({"role": "user", "content": "What is my name, where do I live, and what number did I ask you to remember?"})
        r4 = self._post_json_body("/api/chat/ollama", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        })
        c4 = r4.get("message", {}).get("content", "")
        if c4:
            self.ok(f"Turn 4: Multi-fact response ({len(c4)} chars)")
            checks = {"Charlie": False, "Berlin": False, "42": False}
            for key in checks:
                if key.lower() in c4.lower():
                    checks[key] = True
            for key, found in checks.items():
                if found:
                    self.ok(f"  Context check: '{key}' found")
                else:
                    self.fail(f"  Context check: '{key}' NOT found in response")
        else:
            self.fail("Turn 4: No response")

    # =========================================================================
    # Test: OpenAI API - 6-turn with system message
    # =========================================================================
    def test_openai_system_multiturn(self):
        self.header("Test 3: OpenAI API - System Message + 6 Turns")

        messages = [
            {"role": "system", "content": "You are a math tutor. Always show your work step by step. Keep answers concise."}
        ]
        turns = [
            "What is 7 times 8?",
            "Divide that by 4.",
            "Round that to the nearest whole number.",
            "Is that a prime number?",
            "What is the next prime number after it?",
            "What is the sum of all the numbers we computed?",
        ]

        for i, question in enumerate(turns):
            turn = i + 1
            self.log(f"Turn {turn}/6: {question}")
            messages.append({"role": "user", "content": question})

            resp = self._post_json_body("/v1/chat/completions", {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "max_tokens": self.max_tokens,
            })

            choices = resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                finish = choices[0].get("finish_reason", "")
                if content:
                    self.ok(f"Turn {turn}: {len(content)} chars (finish={finish})")
                    messages.append({"role": "assistant", "content": content})
                else:
                    self.fail(f"Turn {turn}: Empty content")
                    break
            else:
                self.fail(f"Turn {turn}: No choices in response")
                break

        expected = 1 + 6 * 2  # system + 6 user/assistant pairs
        if len(messages) == expected:
            self.ok(f"Complete: {len(messages)} messages (system + 6 pairs)")
        else:
            self.fail(f"Incomplete: {len(messages)} messages (expected {expected})")

    # =========================================================================
    # Test: Ollama streaming - verify done_reason and metrics
    # =========================================================================
    def test_ollama_streaming_metrics(self):
        self.header("Test 4: Ollama Streaming - Metrics Validation (3 Turns)")

        messages = []
        turns = [
            "Hello, how are you?",
            "Tell me a fun fact.",
            "Thank you, goodbye!",
        ]

        for i, question in enumerate(turns):
            turn = i + 1
            self.log(f"Turn {turn}/3: {question}")
            messages.append({"role": "user", "content": question})

            items = self._post_ndjson_collect("/api/chat/ollama", {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"num_predict": 60},
            })

            content = self._extract_ndjson_content(items)
            done_items = [it for it in items if it.get("done") is True]

            if not content:
                self.fail(f"Turn {turn}: No content")
                continue

            if not done_items:
                self.fail(f"Turn {turn}: No done event")
                messages.append({"role": "assistant", "content": content})
                continue

            done_item = done_items[-1]
            has_metrics = all(k in done_item for k in ["total_duration", "eval_count"])
            done_reason = done_item.get("done_reason", "")

            if has_metrics:
                self.ok(f"Turn {turn}: {len(content)} chars, reason={done_reason}, "
                        f"eval_count={done_item.get('eval_count', '?')}")
            else:
                self.fail(f"Turn {turn}: Missing metrics in done event")

            messages.append({"role": "assistant", "content": content})

    # =========================================================================
    # Test: OpenAI streaming with SSE parsing
    # =========================================================================
    def test_openai_streaming_multiturn(self):
        self.header("Test 5: OpenAI Streaming - 4-Turn Conversation")

        messages = []
        turns = [
            "What programming language was Python named after?",
            "Tell me more about that.",
            "When was Python first released?",
            "Who created it?",
        ]

        for i, question in enumerate(turns):
            turn = i + 1
            self.log(f"Turn {turn}/4: {question}")
            messages.append({"role": "user", "content": question})

            resp = self._post_json("/v1/chat/completions", {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "max_tokens": 80,
            })

            content = ""
            finish_reason = None
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        content += c
                    fr = chunk.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                except json.JSONDecodeError:
                    pass

            if content:
                self.ok(f"Turn {turn}: {len(content)} chars (finish={finish_reason})")
                messages.append({"role": "assistant", "content": content})
            else:
                self.fail(f"Turn {turn}: No content")
                break

    # =========================================================================
    # Test: OpenAI structured outputs
    # =========================================================================
    def test_openai_structured_outputs(self):
        self.header("Test 6: OpenAI Structured Outputs")

        resp = self._post_json_body("/v1/chat/completions", {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "What is 2+3? Return the result."}
            ],
            "max_tokens": 80,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": ["string", "number"]},
                            "confidence": {"type": ["string", "null"]}
                        },
                        "required": ["answer", "confidence"],
                        "additionalProperties": False
                    }
                }
            }
        })

        choices = resp.get("choices", [])
        if not choices:
            self.fail("Structured outputs: missing choices")
            return

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            self.fail("Structured outputs: empty content")
            return

        try:
            payload = json.loads(content)
        except Exception as e:
            self.fail(f"Structured outputs: invalid JSON ({e})")
            return

        if isinstance(payload, dict) and "answer" in payload and "confidence" in payload:
            self.ok(f"Structured outputs: parsed JSON object with keys {sorted(payload.keys())}")
        else:
            self.fail(f"Structured outputs: unexpected payload {payload}")

    # =========================================================================
    # Test: Very long conversation - 10 turns
    # =========================================================================
    def test_long_10turn(self):
        self.header("Test 7: Long Conversation - 10 Turns (Ollama Non-Streaming)")

        messages = [
            {"role": "system", "content": "You are a history expert. Keep answers to 1-2 sentences."}
        ]
        turns = [
            "When did World War II start?",
            "Which countries were the main Allied powers?",
            "Who led Germany at that time?",
            "What was D-Day?",
            "When did the war in Europe end?",
            "What about the Pacific theater?",
            "What event led to Japan's surrender?",
            "What organization was formed after the war?",
            "Where is its headquarters?",
            "Summarize the key dates we discussed.",
        ]

        for i, question in enumerate(turns):
            turn = i + 1
            self.log(f"Turn {turn}/10: {question}")
            messages.append({"role": "user", "content": question})

            resp = self._post_json_body("/api/chat/ollama", {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": 60},
            })

            content = resp.get("message", {}).get("content", "")
            if content and resp.get("done"):
                self.ok(f"Turn {turn}: {len(content)} chars")
                messages.append({"role": "assistant", "content": content})
            else:
                self.fail(f"Turn {turn}: Failed (content={len(content)}, done={resp.get('done')})")
                if content:
                    messages.append({"role": "assistant", "content": content})
                else:
                    break

        total = len(messages)
        self.log(f"Final message count: {total}")
        if total >= 18:
            self.ok(f"Long conversation completed with {total} messages")
        else:
            self.fail(f"Long conversation stalled at {total} messages")

    # =========================================================================
    # Test: Error handling
    # =========================================================================
    def test_error_handling(self):
        self.header("Test 8: Error Handling")

        self.log("Test: Missing model (Ollama)")
        try:
            req = Request(
                f"{self.base_url}/api/chat/ollama",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urlopen(req, timeout=10)
            self.fail(f"Missing model: expected 400, got {resp.status}")
        except HTTPError as e:
            if e.code == 400:
                self.ok("Missing model correctly returns 400")
            else:
                self.fail(f"Missing model: expected 400, got {e.code}")

        self.log("Test: Missing messages (OpenAI)")
        try:
            req = Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps({"model": self.model}).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urlopen(req, timeout=10)
            self.fail(f"Missing messages: expected 400, got {resp.status}")
        except HTTPError as e:
            if e.code == 400:
                self.ok("Missing messages correctly returns 400")
            else:
                self.fail(f"Missing messages: expected 400, got {e.code}")

        self.log("Test: Invalid structured output schema")
        try:
            req = Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bad_schema",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"}
                                },
                                "required": ["answer"]
                            }
                        }
                    }
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urlopen(req, timeout=10)
            self.fail(f"Invalid structured schema: expected 400, got {resp.status}")
        except HTTPError as e:
            if e.code == 400:
                self.ok("Invalid structured schema correctly returns 400")
            else:
                self.fail(f"Invalid structured schema: expected 400, got {e.code}")

    # =========================================================================
    # Test: Queue status
    # =========================================================================
    def test_queue_status(self):
        self.header("Test 9: Queue Status")

        resp = self._post_json_body("/api/queue/status", {}) if False else None
        url = f"{self.base_url}/api/queue/status"
        raw = urlopen(url, timeout=10).read().decode()
        data = json.loads(raw)

        required_fields = ["busy", "pending_requests", "total_processed"]
        for f in required_fields:
            if f in data:
                self.ok(f"Queue status has '{f}' = {data[f]}")
            else:
                self.fail(f"Queue status missing '{f}'")

    # =========================================================================
    # Run all tests
    # =========================================================================
    def run_all(self):
        self.header("InferenceWeb Multi-Turn Integration Tests (Python)")

        self.log(f"Server: {self.base_url}")
        self.log(f"Model:  {self.model}")
        self.log(f"Max tokens per turn: {self.max_tokens}")

        url = f"{self.base_url}/api/version"
        try:
            raw = urlopen(url, timeout=10).read().decode()
            self.log(f"Server version: {json.loads(raw).get('version', '?')}")
        except Exception as e:
            print(f"Error: Cannot connect to {self.base_url}: {e}")
            sys.exit(1)

        tests = [
            self.test_webui_5turn,
            self.test_ollama_context_retention,
            self.test_openai_system_multiturn,
            self.test_ollama_streaming_metrics,
            self.test_openai_streaming_multiturn,
            self.test_openai_structured_outputs,
            self.test_long_10turn,
            self.test_error_handling,
            self.test_queue_status,
        ]

        for test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                self.fail(f"Exception in {test_fn.__name__}: {e}")
                traceback.print_exc()

        self.header("Results")
        total = self.passed + self.failed
        print(f"{Colors.GREEN}PASSED: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}FAILED: {self.failed}{Colors.RESET}")
        print(f"TOTAL:  {total}")
        print()

        if self.failed == 0:
            print(f"{Colors.GREEN}All tests passed!{Colors.RESET}")
        else:
            print(f"{Colors.RED}{self.failed} test(s) failed.{Colors.RESET}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn chat integration tests for InferenceWeb")
    parser.add_argument("--model", type=str, default=None, help="Model filename (auto-detected if omitted)")
    parser.add_argument("--url", type=str, default="http://localhost:5000", help="InferenceWeb base URL")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max tokens per response")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    model = args.model
    if not model:
        try:
            raw = urlopen(f"{base_url}/api/models", timeout=10).read().decode()
            data = json.loads(raw)
            model = data.get("loaded") or (data.get("models", [None])[0])
            if not model:
                print("Error: No models found. Specify --model or set MODEL_DIR.")
                sys.exit(1)
            print(f"Auto-detected model: {model}")
        except Exception as e:
            print(f"Error auto-detecting model: {e}")
            sys.exit(1)

    runner = TestRunner(base_url, model, args.max_tokens)
    runner.run_all()


if __name__ == "__main__":
    main()
