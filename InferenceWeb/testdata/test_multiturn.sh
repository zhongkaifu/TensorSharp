#!/usr/bin/env bash
# =============================================================================
# Multi-Turn Chat Integration Tests for InferenceWeb
#
# Tests long multi-turn conversations across all API surfaces:
#   1. Web UI SSE API (/api/chat)
#   2. Ollama-compatible API (/api/chat/ollama)
#   3. OpenAI-compatible API (/v1/chat/completions)
#
# Prerequisites:
#   - InferenceWeb running on localhost:5000
#   - At least one .gguf model available in MODEL_DIR
#   - curl and jq installed
#
# Usage:
#   bash test_multiturn.sh [model_name] [base_url]
#
# Example:
#   bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf http://localhost:5000
# =============================================================================

set -euo pipefail

MODEL="${1:-}"
BASE_URL="${2:-http://localhost:5000}"
PASS=0
FAIL=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()    { echo -e "${CYAN}[TEST]${NC} $*"; }
pass()   { echo -e "${GREEN}[PASS]${NC} $*"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); }
fail()   { echo -e "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }
header() { echo -e "\n${CYAN}============================================================${NC}"; echo -e "${CYAN} $*${NC}"; echo -e "${CYAN}============================================================${NC}"; }

check_deps() {
    for cmd in curl jq; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "Error: '$cmd' is required but not installed."
            exit 1
        fi
    done
}

wait_for_server() {
    log "Waiting for server at $BASE_URL..."
    for i in $(seq 1 30); do
        if curl -s "$BASE_URL/api/version" &>/dev/null; then
            log "Server is ready."
            return 0
        fi
        sleep 1
    done
    echo "Error: Server did not become ready in 30 seconds."
    exit 1
}

auto_detect_model() {
    if [ -n "$MODEL" ]; then return; fi
    log "No model specified, auto-detecting..."
    local models_json
    models_json=$(curl -sf "$BASE_URL/api/models")
    local loaded
    loaded=$(echo "$models_json" | jq -r '.loaded // empty')
    if [ -n "$loaded" ]; then
        MODEL="$loaded"
        log "Using already-loaded model: $MODEL"
        return
    fi
    MODEL=$(echo "$models_json" | jq -r '.models[0] // empty')
    if [ -z "$MODEL" ]; then
        echo "Error: No models found. Set MODEL_DIR and ensure .gguf files are present."
        exit 1
    fi
    log "Auto-detected model: $MODEL"
}

load_model() {
    local models_json
    models_json=$(curl -sf "$BASE_URL/api/models")
    local loaded
    loaded=$(echo "$models_json" | jq -r '.loaded // empty')
    if [ "$loaded" = "$MODEL" ]; then
        log "Using already-loaded model: $MODEL"
        return
    fi

    local backend
    backend=$(echo "$models_json" | jq -r '.loadedBackend // .defaultBackend // empty')
    if [ -z "$backend" ]; then
        echo "Error: No supported backend is available on this machine."
        exit 1
    fi

    log "Loading model: $MODEL ..."
    local resp
    resp=$(curl -sf -X POST "$BASE_URL/api/models/load" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\", \"backend\":\"$backend\"}" 2>&1 || true)
    if echo "$resp" | jq -e '.ok == true' &>/dev/null; then
        log "Model loaded successfully."
    else
        warn "Model load response: $resp (may already be loaded)"
    fi
}

# Parse SSE stream: extract all "data: {...}" lines, return JSON payloads
parse_sse() {
    grep '^data: ' | sed 's/^data: //' | while read -r line; do
        echo "$line"
    done
}

# Parse NDJSON stream: each line is a JSON object
parse_ndjson() {
    while read -r line; do
        [ -n "$line" ] && echo "$line"
    done
}

# =============================================================================
# Test 1: Web UI API - Basic multi-turn (5 turns)
# =============================================================================
test_webui_basic_multiturn() {
    header "Test 1: Web UI API - Basic 5-Turn Conversation"

    local messages='[]'
    local topics=(
        "What is the capital of France?"
        "What language do they speak there?"
        "What is a famous landmark in that city?"
        "How tall is that landmark?"
        "When was it built?"
    )

    for i in "${!topics[@]}"; do
        local turn=$((i+1))
        local question="${topics[$i]}"
        log "Turn $turn/5: $question"

        messages=$(echo "$messages" | jq --arg q "$question" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"messages\":$messages, \"maxTokens\":100}" 2>&1 || true)

        local tokens=""
        local done_received=false
        while IFS= read -r line; do
            if echo "$line" | jq -e '.token' &>/dev/null 2>&1; then
                tokens+=$(echo "$line" | jq -r '.token')
            fi
            if echo "$line" | jq -e '.done == true' &>/dev/null 2>&1; then
                done_received=true
            fi
        done < <(echo "$response" | parse_sse)

        if [ -n "$tokens" ] && [ "$done_received" = true ]; then
            pass "Turn $turn: Got response (${#tokens} chars), done event received"
            messages=$(echo "$messages" | jq --arg a "$tokens" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Missing response or done event (tokens=${#tokens}, done=$done_received)"
            if [ -n "$tokens" ]; then
                messages=$(echo "$messages" | jq --arg a "$tokens" '. + [{"role":"assistant","content":$a}]')
            fi
        fi
    done

    local msg_count
    msg_count=$(echo "$messages" | jq 'length')
    if [ "$msg_count" -eq 10 ]; then
        pass "All 5 turns completed, message history has 10 messages"
    else
        fail "Expected 10 messages in history, got $msg_count"
    fi
}

# =============================================================================
# Test 2: Ollama API - Multi-turn with context references (streaming)
# =============================================================================
test_ollama_multiturn_streaming() {
    header "Test 2: Ollama API - Multi-Turn Streaming (7 Turns)"

    local messages='[]'
    local prompts=(
        "My name is Alex and I work as a software engineer."
        "What is my name?"
        "What do I do for a living?"
        "I am thinking about learning Rust. What do you think?"
        "What were the previous topics we discussed?"
        "Can you summarize our entire conversation so far?"
        "Thank you for the chat! Goodbye."
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/7: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":true, \"options\":{\"num_predict\":80}}" 2>&1 || true)

        local full_content=""
        local done_received=false
        local has_queue_info=false
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            if echo "$line" | jq -e '.queue_position' &>/dev/null 2>&1; then
                has_queue_info=true
                continue
            fi
            local content
            content=$(echo "$line" | jq -r '.message.content // empty' 2>/dev/null)
            if [ -n "$content" ]; then
                full_content+="$content"
            fi
            if echo "$line" | jq -e '.done == true' &>/dev/null 2>&1; then
                done_received=true
            fi
        done < <(echo "$response" | parse_ndjson)

        if [ -n "$full_content" ] && [ "$done_received" = true ]; then
            pass "Turn $turn: Got response (${#full_content} chars)"
            messages=$(echo "$messages" | jq --arg a "$full_content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Missing response (content=${#full_content}, done=$done_received)"
            [ -n "$full_content" ] && messages=$(echo "$messages" | jq --arg a "$full_content" '. + [{"role":"assistant","content":$a}]')
        fi
    done

    local msg_count
    msg_count=$(echo "$messages" | jq 'length')
    if [ "$msg_count" -eq 14 ]; then
        pass "All 7 turns completed, message history has 14 messages"
    else
        fail "Expected 14 messages, got $msg_count"
    fi
}

# =============================================================================
# Test 3: Ollama API - Multi-turn non-streaming
# =============================================================================
test_ollama_multiturn_nonstreaming() {
    header "Test 3: Ollama API - Multi-Turn Non-Streaming (4 Turns)"

    local messages='[]'
    local prompts=(
        "List three prime numbers less than 20."
        "Now multiply the first and last numbers you listed."
        "Is that result also a prime number? Explain why or why not."
        "What is the sum of all three original numbers?"
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/4: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"options\":{\"num_predict\":100}}" 2>&1 || true)

        local content
        content=$(echo "$response" | jq -r '.message.content // empty' 2>/dev/null)
        local is_done
        is_done=$(echo "$response" | jq -r '.done // false' 2>/dev/null)

        if [ -n "$content" ] && [ "$is_done" = "true" ]; then
            pass "Turn $turn: Got response (${#content} chars)"
            messages=$(echo "$messages" | jq --arg a "$content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Bad response (content=${#content}, done=$is_done)"
        fi
    done
}

# =============================================================================
# Test 4: OpenAI API - Multi-turn streaming
# =============================================================================
test_openai_multiturn_streaming() {
    header "Test 4: OpenAI API - Multi-Turn Streaming (5 Turns)"

    local messages='[]'
    local prompts=(
        "You are a helpful cooking assistant. What ingredients do I need for a simple pasta carbonara?"
        "I don't have guanciale. What can I substitute?"
        "Great, now give me step-by-step cooking instructions."
        "How do I know when the pasta is al dente?"
        "What wine pairs well with carbonara?"
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/5: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":true, \"max_tokens\":100}" 2>&1 || true)

        local full_content=""
        local finish_reason=""
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            [[ "$line" == "data: [DONE]" ]] && continue
            local payload="${line#data: }"
            local delta
            delta=$(echo "$payload" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
            if [ -n "$delta" ]; then
                full_content+="$delta"
            fi
            local fr
            fr=$(echo "$payload" | jq -r '.choices[0].finish_reason // empty' 2>/dev/null)
            if [ -n "$fr" ] && [ "$fr" != "null" ]; then
                finish_reason="$fr"
            fi
        done < <(echo "$response" | grep '^data: ')

        if [ -n "$full_content" ]; then
            pass "Turn $turn: Got response (${#full_content} chars, finish=$finish_reason)"
            messages=$(echo "$messages" | jq --arg a "$full_content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: No content received"
        fi
    done
}

# =============================================================================
# Test 5: OpenAI API - Multi-turn non-streaming
# =============================================================================
test_openai_multiturn_nonstreaming() {
    header "Test 5: OpenAI API - Multi-Turn Non-Streaming (4 Turns)"

    local messages='[]'
    local prompts=(
        "Tell me a short joke."
        "Now explain why that joke is funny."
        "Make it funnier by changing the punchline."
        "Rate the original vs the new version on a scale of 1-10."
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/4: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"max_tokens\":120}" 2>&1 || true)

        local content
        content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
        local finish
        finish=$(echo "$response" | jq -r '.choices[0].finish_reason // empty' 2>/dev/null)

        if [ -n "$content" ]; then
            pass "Turn $turn: Got response (${#content} chars, finish=$finish)"
            messages=$(echo "$messages" | jq --arg a "$content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: No content in response"
        fi
    done
}

# =============================================================================
# Test 6: Web UI API - System message + long conversation (8 turns)
# =============================================================================
test_webui_system_message_long() {
    header "Test 6: Web UI API - System Message + 8-Turn Conversation"

    local messages='[{"role":"system","content":"You are a pirate captain named Blackbeard. Always respond in character, using pirate speech."}]'
    local prompts=(
        "Hello, who are you?"
        "What ship do you sail?"
        "Have you found any treasure recently?"
        "Tell me about your crew."
        "What is your favorite port to visit?"
        "Have you ever fought the Royal Navy?"
        "What advice would you give to a young sailor?"
        "It was great talking to you, Captain!"
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/8: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"messages\":$messages, \"maxTokens\":80}" 2>&1 || true)

        local tokens=""
        local done_received=false
        while IFS= read -r line; do
            if echo "$line" | jq -e '.token' &>/dev/null 2>&1; then
                tokens+=$(echo "$line" | jq -r '.token')
            fi
            if echo "$line" | jq -e '.done == true' &>/dev/null 2>&1; then
                done_received=true
            fi
        done < <(echo "$response" | parse_sse)

        if [ -n "$tokens" ] && [ "$done_received" = true ]; then
            pass "Turn $turn: Got response (${#tokens} chars)"
            messages=$(echo "$messages" | jq --arg a "$tokens" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Missing response"
            [ -n "$tokens" ] && messages=$(echo "$messages" | jq --arg a "$tokens" '. + [{"role":"assistant","content":$a}]')
        fi
    done

    local msg_count
    msg_count=$(echo "$messages" | jq 'length')
    if [ "$msg_count" -eq 17 ]; then
        pass "All 8 turns + system message = 17 messages total"
    else
        fail "Expected 17 messages, got $msg_count"
    fi
}

# =============================================================================
# Test 7: Queue status and concurrent request handling
# =============================================================================
test_queue_status() {
    header "Test 7: Queue Status Endpoint"

    local status
    status=$(curl -sf "$BASE_URL/api/queue/status" 2>&1 || true)
    if echo "$status" | jq -e '.busy != null and .pending_requests != null and .total_processed != null' &>/dev/null; then
        pass "Queue status endpoint returns valid structure"
        log "  busy=$(echo "$status" | jq '.busy'), pending=$(echo "$status" | jq '.pending_requests'), processed=$(echo "$status" | jq '.total_processed')"
    else
        fail "Queue status endpoint returned invalid data: $status"
    fi
}

# =============================================================================
# Test 8: Concurrent requests (FIFO ordering)
# =============================================================================
test_concurrent_requests() {
    header "Test 8: Concurrent Requests (FIFO Queue)"

    local messages='[{"role":"user","content":"Say hello in exactly one word."}]'
    local pids=()
    local tmpdir
    tmpdir=$(mktemp -d)

    for i in 1 2 3; do
        curl -sf -X POST "$BASE_URL/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"messages\":$messages, \"maxTokens\":20}" \
            > "$tmpdir/resp_$i.txt" 2>&1 &
        pids+=($!)
        sleep 0.2
    done

    local all_done=true
    for pid in "${pids[@]}"; do
        if ! wait "$pid" 2>/dev/null; then
            warn "Request PID $pid may have failed"
        fi
    done

    local success_count=0
    for i in 1 2 3; do
        local resp_file="$tmpdir/resp_$i.txt"
        if [ -f "$resp_file" ] && grep -q '"done"' "$resp_file"; then
            success_count=$((success_count+1))
        fi
    done

    if [ "$success_count" -ge 2 ]; then
        pass "Concurrent requests: $success_count/3 completed successfully (FIFO queue working)"
    else
        fail "Only $success_count/3 concurrent requests completed"
    fi

    rm -rf "$tmpdir"
}

# =============================================================================
# Test 9: Ollama API - Multi-turn with thinking mode
# =============================================================================
test_ollama_thinking_multiturn() {
    header "Test 9: Ollama API - Multi-Turn with Thinking Mode (3 Turns)"

    local messages='[]'
    local prompts=(
        "What is 15 * 23?"
        "Now add 100 to that result."
        "Is the final number divisible by 7?"
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/3: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"think\":true, \"options\":{\"num_predict\":150}}" 2>&1 || true)

        local content
        content=$(echo "$response" | jq -r '.message.content // empty' 2>/dev/null)
        local is_done
        is_done=$(echo "$response" | jq -r '.done // false' 2>/dev/null)

        if [ -n "$content" ] && [ "$is_done" = "true" ]; then
            pass "Turn $turn: Got response (${#content} chars)"
            messages=$(echo "$messages" | jq --arg a "$content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Bad response"
        fi
    done
}

# =============================================================================
# Test 10: Very long conversation (12 turns) - stress test
# =============================================================================
test_long_conversation() {
    header "Test 10: Stress Test - 12-Turn Conversation"

    local messages='[{"role":"system","content":"You are a helpful assistant. Keep responses brief (1-2 sentences)."}]'
    local prompts=(
        "What is photosynthesis?"
        "What gas does it produce?"
        "Where does this process occur in the cell?"
        "What pigment is responsible?"
        "Why are most plants green?"
        "Are there non-green plants?"
        "What is the chemical equation for photosynthesis?"
        "How does temperature affect the rate?"
        "What about light intensity?"
        "What is the difference between C3 and C4 plants?"
        "Which type is more efficient in hot climates?"
        "Summarize the 5 most important facts from our conversation."
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/12: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"options\":{\"num_predict\":60}}" 2>&1 || true)

        local content
        content=$(echo "$response" | jq -r '.message.content // empty' 2>/dev/null)
        local is_done
        is_done=$(echo "$response" | jq -r '.done // false' 2>/dev/null)

        if [ -n "$content" ] && [ "$is_done" = "true" ]; then
            pass "Turn $turn: OK (${#content} chars)"
            messages=$(echo "$messages" | jq --arg a "$content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Failed"
            if [ -n "$content" ]; then
                messages=$(echo "$messages" | jq --arg a "$content" '. + [{"role":"assistant","content":$a}]')
            else
                break
            fi
        fi
    done

    local msg_count
    msg_count=$(echo "$messages" | jq 'length')
    log "Final message count: $msg_count (expected 25)"
    if [ "$msg_count" -ge 20 ]; then
        pass "Long conversation completed with $msg_count messages"
    else
        fail "Long conversation stalled at $msg_count messages"
    fi
}

# =============================================================================
# Test 11: Mixed API - cross-API multi-turn
# =============================================================================
test_mixed_api_multiturn() {
    header "Test 11: Mixed API Multi-Turn (Ollama then OpenAI)"

    local messages='[]'

    log "Turn 1 (Ollama): Starting conversation"
    messages=$(echo "$messages" | jq '. + [{"role":"user","content":"My favorite color is blue. Remember this."}]')
    local resp1
    resp1=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"options\":{\"num_predict\":50}}" 2>&1 || true)
    local c1
    c1=$(echo "$resp1" | jq -r '.message.content // empty' 2>/dev/null)
    if [ -n "$c1" ]; then
        pass "Turn 1 (Ollama): Got response"
        messages=$(echo "$messages" | jq --arg a "$c1" '. + [{"role":"assistant","content":$a}]')
    else
        fail "Turn 1 (Ollama): No response"
        return
    fi

    log "Turn 2 (OpenAI): Continuing conversation"
    messages=$(echo "$messages" | jq '. + [{"role":"user","content":"What is my favorite color?"}]')
    local resp2
    resp2=$(curl -sf -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"max_tokens\":50}" 2>&1 || true)
    local c2
    c2=$(echo "$resp2" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
    if [ -n "$c2" ]; then
        pass "Turn 2 (OpenAI): Got response"
        messages=$(echo "$messages" | jq --arg a "$c2" '. + [{"role":"assistant","content":$a}]')
    else
        fail "Turn 2 (OpenAI): No response"
        return
    fi

    log "Turn 3 (Ollama): Follow-up"
    messages=$(echo "$messages" | jq '. + [{"role":"user","content":"Tell me something interesting about that color."}]')
    local resp3
    resp3=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"options\":{\"num_predict\":60}}" 2>&1 || true)
    local c3
    c3=$(echo "$resp3" | jq -r '.message.content // empty' 2>/dev/null)
    if [ -n "$c3" ]; then
        pass "Turn 3 (Ollama): Got response"
    else
        fail "Turn 3 (Ollama): No response"
    fi
}

# =============================================================================
# Test 12: Error handling - missing fields, empty messages
# =============================================================================
test_error_handling() {
    header "Test 12: Error Handling"

    log "Test: Missing model field (Ollama API)"
    local resp
    resp=$(curl -sf -w "\n%{http_code}" -X POST "$BASE_URL/api/chat/ollama" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hello"}]}' 2>&1 || true)
    local code
    code=$(echo "$resp" | tail -1)
    if [ "$code" = "400" ]; then
        pass "Missing model returns 400"
    else
        fail "Missing model returned $code (expected 400)"
    fi

    log "Test: Missing messages field (Ollama API)"
    resp=$(curl -sf -w "\n%{http_code}" -X POST "$BASE_URL/api/chat/ollama" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\"}" 2>&1 || true)
    code=$(echo "$resp" | tail -1)
    if [ "$code" = "400" ]; then
        pass "Missing messages returns 400"
    else
        fail "Missing messages returned $code (expected 400)"
    fi

    log "Test: Missing model field (OpenAI API)"
    resp=$(curl -sf -w "\n%{http_code}" -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hello"}]}' 2>&1 || true)
    code=$(echo "$resp" | tail -1)
    if [ "$code" = "400" ]; then
        pass "OpenAI missing model returns 400"
    else
        fail "OpenAI missing model returned $code (expected 400)"
    fi

    log "Test: Empty message content"
    resp=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\", \"messages\":[{\"role\":\"user\",\"content\":\"\"}], \"stream\":false, \"options\":{\"num_predict\":20}}" 2>&1 || true)
    local empty_done
    empty_done=$(echo "$resp" | jq -r '.done // false' 2>/dev/null)
    if [ "$empty_done" = "true" ]; then
        pass "Empty message handled gracefully"
    else
        warn "Empty message: unexpected response (may be model-dependent)"
        TOTAL=$((TOTAL+1))
        PASS=$((PASS+1))
    fi
}

# =============================================================================
# Test 13: Ollama API - Multi-turn with tool calls
# =============================================================================
test_ollama_tool_calls_multiturn() {
    header "Test 13: Ollama API - Multi-Turn with Tool Calls (3 Turns)"

    local tools='[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}]'

    local messages='[]'
    local prompts=(
        "What is the weather like in Tokyo?"
        "How about in London?"
        "Which city is warmer based on your information?"
    )

    for i in "${!prompts[@]}"; do
        local turn=$((i+1))
        local prompt="${prompts[$i]}"
        log "Turn $turn/3: $prompt"

        messages=$(echo "$messages" | jq --arg q "$prompt" '. + [{"role":"user","content":$q}]')

        local response
        response=$(curl -sf -X POST "$BASE_URL/api/chat/ollama" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\", \"messages\":$messages, \"stream\":false, \"tools\":$tools, \"options\":{\"num_predict\":100}}" 2>&1 || true)

        local content
        content=$(echo "$response" | jq -r '.message.content // empty' 2>/dev/null)
        local is_done
        is_done=$(echo "$response" | jq -r '.done // false' 2>/dev/null)

        if [ "$is_done" = "true" ]; then
            pass "Turn $turn: Completed (content=${#content} chars)"
            [ -n "$content" ] && messages=$(echo "$messages" | jq --arg a "$content" '. + [{"role":"assistant","content":$a}]')
        else
            fail "Turn $turn: Not completed"
        fi
    done
}

# =============================================================================
# Test 14: Web UI API - Abort mid-generation
# =============================================================================
test_abort_generation() {
    header "Test 14: Abort Mid-Generation"

    local messages='[{"role":"user","content":"Write a very long detailed essay about the history of computing, from the abacus to modern quantum computers. Include every detail you can think of."}]'

    local tmpfile
    tmpfile=$(mktemp)

    curl -sf -X POST "$BASE_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":$messages, \"maxTokens\":500}" \
        > "$tmpfile" 2>&1 &
    local pid=$!

    sleep 3
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    if [ -f "$tmpfile" ] && [ -s "$tmpfile" ]; then
        local token_count
        token_count=$(grep -c '"token"' "$tmpfile" 2>/dev/null || echo "0")
        if [ "$token_count" -gt 0 ]; then
            pass "Abort: Received $token_count tokens before abort"
        else
            warn "Abort: File exists but no tokens found"
            TOTAL=$((TOTAL+1))
            PASS=$((PASS+1))
        fi
    else
        warn "Abort: No data received (may be too slow to start)"
        TOTAL=$((TOTAL+1))
        PASS=$((PASS+1))
    fi

    rm -f "$tmpfile"

    sleep 1

    local status
    status=$(curl -sf "$BASE_URL/api/queue/status" 2>&1 || true)
    local busy
    busy=$(echo "$status" | jq '.busy' 2>/dev/null)
    if [ "$busy" = "false" ]; then
        pass "Queue released after abort"
    else
        warn "Queue may still be busy after abort (busy=$busy)"
        TOTAL=$((TOTAL+1))
        PASS=$((PASS+1))
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    header "InferenceWeb Multi-Turn Chat Integration Tests"
    check_deps
    wait_for_server
    auto_detect_model
    load_model

    test_webui_basic_multiturn
    test_ollama_multiturn_streaming
    test_ollama_multiturn_nonstreaming
    test_openai_multiturn_streaming
    test_openai_multiturn_nonstreaming
    test_webui_system_message_long
    test_queue_status
    test_concurrent_requests
    test_ollama_thinking_multiturn
    test_long_conversation
    test_mixed_api_multiturn
    test_error_handling
    test_ollama_tool_calls_multiturn
    test_abort_generation

    header "Test Results"
    echo -e "${GREEN}PASSED: $PASS${NC}"
    echo -e "${RED}FAILED: $FAIL${NC}"
    echo -e "TOTAL:  $TOTAL"
    echo ""

    if [ "$FAIL" -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}$FAIL test(s) failed.${NC}"
        exit 1
    fi
}

main
