#!/usr/bin/env python3
"""Independent answer-only oracles for the fixed Direct probe tasks.

These rules never use a model's reference answer as ground truth, remove a
Markdown/JSON wrapper, or search prose for an embedded correct answer. Callers
must additionally establish process identity, complete generation and EOS.
"""
import ast
import json
import re


POLICY = 'fixed-answer-only-v2'
THINKING_POLICY = 'glm-qwen-angle-bracket-state-v1'
LEGACY_MAYA_POLICY = 'reference-final-text-equality-v1 (implicit historical policy)'
PROMPTS = {
    'math': 'Compute 17 + 25. Reply with only the integer.',
    'extraction': 'Inventory: cedar=3, maple=7, pine=2. Return only the names whose quantity is at least 3, comma-separated in the original order.',
    'squares': 'Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.',
    'code': 'Write a Python function square(x) that returns x*x. Return only the function, without Markdown.',
    'tool_json': 'Available tool: get_weather(city, unit), where unit is "celsius" or "fahrenheit". User asks: What is the weather in Hangzhou in Celsius? Return only one JSON object with exactly the keys "name" and "arguments", containing the tool name and its required arguments. Do not answer the weather question.',
}


def unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError('Duplicate JSON object keys')
    return value


def thinking_closed(prompt, raw):
    """Closure for GLM/Qwen angle-bracket thinking, not other channel formats."""
    if not isinstance(prompt, str) or not isinstance(raw, str):
        return False
    opened = prompt.rstrip().endswith('<think>')
    for tag in re.finditer(r'</?think>', raw):
        opened = tag.group() == '<think>'
    return not opened


def evaluate(name, prompt, answer):
    """Bind to the exact known question; unknown tasks/prompts fail closed."""
    known = isinstance(name, str) and name in PROMPTS
    bound = known and prompt == PROMPTS[name]
    text = answer.strip() if isinstance(answer, str) else None
    valid = False
    if bound and text is not None:
        if name == 'math':
            # Matches the original Direct probe's optional sentence punctuation.
            valid = re.fullmatch(r'42[.!]?', text) is not None
        elif name == 'extraction':
            valid = re.fullmatch(r'cedar\s*,\s*maple', text, re.IGNORECASE | re.ASCII) is not None
        elif name == 'squares':
            if re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+){19}', text, re.ASCII):
                # Bound every digit string before int(); malformed huge integer
                # input must fail, not throw or consume an unbounded conversion.
                parts = [part.strip().lstrip('0') or '0' for part in text.split(',')]
                valid = parts == [str(i * i) for i in range(1, 21)]
        elif name == 'code' and len(text) <= 65536:
            try:
                tree = ast.parse(text)
                expected = ast.parse('def square(x):\n    return x*x')
                valid = ast.dump(tree, include_attributes=False) == ast.dump(expected, include_attributes=False)
            except (SyntaxError, ValueError, RecursionError):
                valid = False
        elif name == 'tool_json':
            try:
                valid = json.loads(text, object_pairs_hook=unique_object) == {
                    'name': 'get_weather', 'arguments': {'city': 'Hangzhou', 'unit': 'celsius'}}
            except (ValueError, TypeError, RecursionError):
                valid = False
    return dict(policy=POLICY, case=name, known_case=known, prompt_matches=bound,
                passed=bool(valid), expected={'math': '42', 'extraction': ['cedar', 'maple'],
                    'squares': [i * i for i in range(1, 21)],
                    'code': 'def square(x):\n    return x*x',
                    'tool_json': {'name': 'get_weather', 'arguments': {'city': 'Hangzhou', 'unit': 'celsius'}}}.get(name) if known else None)


def parsed_answer(name, prompt, content, thinking_closed, tool_calls):
    """Grade supplied production-parser output without trusting a saved pass."""
    answer = evaluate(name, prompt, content)
    structure = thinking_closed is True and isinstance(tool_calls, list) and not tool_calls
    return dict(**answer, answer_passed=answer['passed'], structure_passed=structure) | {
        'passed': answer['passed'] and structure}
