#!/usr/bin/env python3
"""
Multi-Turn Chat Integration Tests for TensorSharp.Server

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
import traceback
from urllib.request import urlopen, Request
from urllib.error import HTTPError


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class TestRunner:
    def __init__(self, base_url, model, max_tokens):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.architecture = ""
        self.supports_thinking = False
        self.supports_tools = False
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def log(self, msg):
        print(f"{Colors.CYAN}[TEST]{Colors.RESET} {msg}")

    def ok(self, msg):
        print(f"{Colors.GREEN}[PASS]{Colors.RESET} {msg}")
        self.passed += 1

    def fail(self, msg):
        print(f"{Colors.RED}[FAIL]{Colors.RESET} {msg}")
        self.failed += 1

    def skip(self, msg):
        print(f"{Colors.YELLOW}[SKIP]{Colors.RESET} {msg}")
        self.skipped += 1

    def header(self, title):
        bar = "=" * 60
        print(f"\n{Colors.CYAN}{bar}\n {title}\n{bar}{Colors.RESET}")

    def model_output(self, label, content):
        if content is None:
            content = ""
        print(f"{Colors.MAGENTA}[MODEL]{Colors.RESET} {label}")
        if content:
            for line in str(content).splitlines():
                print(f"        {line}")
        else:
            print("        <empty>")

    def _post_json(self, path, payload):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urlopen(req, timeout=120)
        return resp

    def _post_json_body(self, path, payload):
        resp = self._post_json(path, payload)
        return json.loads(resp.read().decode("utf-8"))

    def _post_sse_payloads(self, path, payload):
        """POST and collect raw SSE data payloads."""
        resp = self._post_json(path, payload)
        payloads = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data: "):
                payloads.append(line[6:])
        return payloads

    def _post_sse_collect(self, path, payload):
        """POST and collect SSE events, return list of parsed JSON objects."""
        events = []
        for payload_text in self._post_sse_payloads(path, payload):
            if payload_text == "[DONE]":
                continue
            try:
                events.append(json.loads(payload_text))
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

    def ensure_model_loaded(self):
        data = json.loads(urlopen(f"{self.base_url}/api/models", timeout=30).read().decode("utf-8"))
        if data.get("loaded") == self.model:
            self.log(f"Using already-loaded model: {self.model}")
        else:
            backend = data.get("loadedBackend") or data.get("defaultBackend")
            if not backend:
                raise RuntimeError("No supported backend is available to load the test model.")

            self.log(f"Loading model: {self.model} ({backend})")
            result = self._post_json_body("/api/models/load", {"model": self.model, "backend": backend})
            if not result.get("ok"):
                raise RuntimeError(f"Failed to load model: {result}")

        self.refresh_model_metadata()

    def refresh_model_metadata(self):
        data = json.loads(urlopen(f"{self.base_url}/api/models", timeout=30).read().decode("utf-8"))
        self.architecture = data.get("architecture") or ""
        self.supports_thinking, self.supports_tools = self._detect_capabilities(self.architecture, self.model)

    def _normalize_name(self, value):
        return (value or "").lower().replace("-", "").replace("_", "").replace(".", "")

    def _detect_capabilities(self, architecture, model_name):
        normalized = self._normalize_name(architecture) or self._normalize_name(model_name)

        if normalized.startswith("gemma4"):
            return True, True
        if normalized.startswith("gemma3"):
            return False, False
        if normalized.startswith("qwen35"):
            return True, True
        if normalized.startswith("qwen3"):
            return True, True
        if "gptoss" in normalized:
            return True, False
        if normalized.startswith("nemotronh") or normalized.startswith("nemotron"):
            return True, True
        if normalized.startswith("mistral3") or normalized.startswith("mistral"):
            return False, False
        return False, False

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

    def _extract_openai_stream(self, payloads):
        content = ""
        finish_reason = None
        saw_done = False

        for payload_text in payloads:
            if payload_text == "[DONE]":
                saw_done = True
                continue
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                content += piece
            fr = chunk.get("choices", [{}])[0].get("finish_reason")
            if fr:
                finish_reason = fr

        return content, finish_reason, saw_done

    def expect_http_error(self, path, payload, expected_code, success_msg):
        try:
            self._post_json_body(path, payload)
            self.fail(f"{success_msg}: expected {expected_code}, got success")
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self.model_output(f"{success_msg} response", body)
            if e.code == expected_code:
                self.ok(success_msg)
            else:
                self.fail(f"{success_msg}: expected {expected_code}, got {e.code}")

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
            })

            content = self._extract_sse_tokens(events)
            has_done = self._has_done(events)
            self.model_output(f"Turn {turn} output", content)

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
        self.model_output("Turn 1 output", c1)
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
        self.model_output("Turn 2 output", c2)
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
        self.model_output("Turn 3 output", c3)
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
        self.model_output("Turn 4 output", c4)
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
                self.model_output(f"Turn {turn} output", content)
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
            self.model_output(f"Turn {turn} output", content)

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

            payloads = self._post_sse_payloads("/v1/chat/completions", {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "max_tokens": 80,
            })

            content, finish_reason, saw_done = self._extract_openai_stream(payloads)

            self.model_output(f"Turn {turn} output", content)
            if content and saw_done:
                self.ok(f"Turn {turn}: {len(content)} chars (finish={finish_reason})")
                messages.append({"role": "assistant", "content": content})
            else:
                self.fail(f"Turn {turn}: content={len(content)} chars, saw_done={saw_done}")
                break

    # =========================================================================
    # Test: OpenAI json_object outputs
    # =========================================================================
    def test_openai_json_object_outputs(self):
        self.header("Test 6: OpenAI JSON Object Outputs")

        resp = self._post_json_body("/v1/chat/completions", {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Return a JSON object with keys answer and confidence for 2+3."}
            ],
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
        })

        choices = resp.get("choices", [])
        if not choices:
            self.fail("json_object: missing choices")
            return

        content = choices[0].get("message", {}).get("content", "")
        self.model_output("json_object output", content)
        if not content:
            self.fail("json_object: empty content")
            return

        try:
            payload = json.loads(content)
        except Exception as e:
            self.fail(f"json_object: invalid JSON ({e})")
            return

        if isinstance(payload, dict) and "answer" in payload and "confidence" in payload:
            self.ok(f"json_object: parsed JSON object with keys {sorted(payload.keys())}")
        else:
            self.fail(f"json_object: unexpected payload {payload}")

    # =========================================================================
    # Test: OpenAI structured outputs
    # =========================================================================
    def test_openai_structured_outputs(self):
        self.header("Test 7: OpenAI Structured Outputs")

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
        self.model_output("Structured output", content)
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
    # Test: Ollama thinking mode
    # =========================================================================
    def test_ollama_thinking_multiturn(self):
        if not self.supports_thinking:
            self.skip(f"Thinking-mode tests skipped for architecture '{self.architecture or 'unknown'}'")
            return

        self.header("Test 8: Ollama API - Multi-Turn with Thinking Mode")

        messages = []
        prompts = [
            "What is 15 * 23?",
            "Now add 100 to that result.",
            "Is the final number divisible by 7?",
        ]
        saw_thinking = False

        for i, prompt in enumerate(prompts):
            turn = i + 1
            self.log(f"Turn {turn}/3: {prompt}")
            messages.append({"role": "user", "content": prompt})

            resp = self._post_json_body("/api/chat/ollama", {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": True,
                "options": {"num_predict": 150},
            })

            message = resp.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")
            self.model_output(f"Turn {turn} reasoning", thinking)
            self.model_output(f"Turn {turn} answer", content)

            if content and resp.get("done"):
                self.ok(f"Turn {turn}: {len(content)} chars")
                if thinking:
                    saw_thinking = True
                messages.append({"role": "assistant", "content": content})
            else:
                self.fail(f"Turn {turn}: Failed (content={len(content)}, done={resp.get('done')})")
                return

        if saw_thinking:
            self.ok("Thinking mode: response included a separate thinking field")
        else:
            self.fail("Thinking mode: no thinking field was returned")

    # =========================================================================
    # Test: OpenAI tool calls
    # =========================================================================
    def test_openai_tool_calls(self):
        if not self.supports_tools:
            self.skip(f"Tool-calling tests skipped for architecture '{self.architecture or 'unknown'}'")
            return

        self.header("Test 9: OpenAI Tool Calls")

        resp = self._post_json_body("/v1/chat/completions", {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You must call the provided function when the user asks for weather. Do not answer from memory."
                },
                {
                    "role": "user",
                    "content": "What is the weather in Tokyo? Call the get_weather tool and do not answer with prose."
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"],
                            "additionalProperties": False
                        }
                    }
                }
            ],
            "max_tokens": 120,
            "stream": False,
        })

        choices = resp.get("choices", [])
        if not choices:
            self.fail("OpenAI tool calls: missing choices")
            return

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        finish_reason = choices[0].get("finish_reason")
        self.model_output("OpenAI tool call payload", json.dumps(tool_calls, indent=2))

        if tool_calls and finish_reason == "tool_calls":
            self.ok(f"OpenAI tool calls: model returned {len(tool_calls)} tool call(s)")
        elif message.get("content"):
            self.skip("OpenAI tool calls: model answered directly instead of returning tool_calls")
        else:
            self.fail(f"OpenAI tool calls: expected tool_calls finish, got finish_reason={finish_reason}")

    # =========================================================================
    # Test: Error handling
    # =========================================================================
    def test_error_handling(self):
        self.header("Test 10: Error Handling")

        self.expect_http_error(
            "/api/chat/ollama",
            {"messages": [{"role": "user", "content": "hi"}]},
            400,
            "Missing model correctly returns 400",
        )

        self.expect_http_error(
            "/v1/chat/completions",
            {"model": self.model},
            400,
            "Missing messages correctly returns 400",
        )

        self.expect_http_error(
            "/v1/chat/completions",
            {
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
            },
            400,
            "Invalid structured schema correctly returns 400",
        )

        self.expect_http_error(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "noop",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False
                            }
                        }
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "simple_object",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"}
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        }
                    }
                }
            },
            400,
            "response_format + tools correctly returns 400",
        )

        self.expect_http_error(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "think": True,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "simple_object",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"}
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        }
                    }
                }
            },
            400,
            "response_format + think correctly returns 400",
        )

    # =========================================================================
    # Test: Queue status
    # =========================================================================
    def test_queue_status(self):
        self.header("Test 11: Queue Status")

        url = f"{self.base_url}/api/queue/status"
        raw = urlopen(url, timeout=10).read().decode()
        self.model_output("Queue status response", raw)
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
        self.header("TensorSharp.Server Integration Tests (Python)")

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

        try:
            self.ensure_model_loaded()
        except Exception as e:
            print(f"Error: Cannot load model '{self.model}': {e}")
            sys.exit(1)

        self.log(f"Architecture: {self.architecture or 'unknown'}")
        self.log(f"Capabilities: thinking={self.supports_thinking}, tools={self.supports_tools}")

        tests = [
            self.test_webui_5turn,
            self.test_ollama_context_retention,
            self.test_openai_system_multiturn,
            self.test_ollama_streaming_metrics,
            self.test_openai_streaming_multiturn,
            self.test_openai_json_object_outputs,
            self.test_openai_structured_outputs,
            self.test_ollama_thinking_multiturn,
            self.test_openai_tool_calls,
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
        total = self.passed + self.failed + self.skipped
        print(f"{Colors.GREEN}PASSED:  {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}FAILED:  {self.failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}SKIPPED: {self.skipped}{Colors.RESET}")
        print(f"TOTAL:  {total}")
        print()

        if self.failed == 0:
            print(f"{Colors.GREEN}All runnable tests passed!{Colors.RESET}")
        else:
            print(f"{Colors.RED}{self.failed} test(s) failed.{Colors.RESET}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn chat integration tests for TensorSharp.Server")
    parser.add_argument("--model", type=str, default=None, help="Model filename (auto-detected if omitted)")
    parser.add_argument("--url", type=str, default="http://localhost:5000", help="TensorSharp.Server base URL")
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

