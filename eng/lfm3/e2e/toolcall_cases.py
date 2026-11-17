# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Hand-written function-calling cases for toolcall_eval.py (LFM3 end-to-end suite).

Each case: id, tools (names from TOOLS), messages, expected and optional request extras.
expected is a list of calls, each {"name": ..., "args": {param: matcher}}; [] means the reply must
not call a tool. Parallel calls match in any order. A matcher is a literal (strings compare
case-insensitively after trimming; numbers numerically), {"contains": s}, {"any_of": [...]} or
{"present": True}. Arguments the schema does not declare, missing required arguments and values
of the wrong JSON type make a call schema-invalid (scored wrong).
"""


def fn(name, description, properties, required=None):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required if required is not None else list(properties)}}}


S = {"type": "string"}
TOOLS = {t["function"]["name"]: t for t in [
    fn("get_weather", "Get the current weather for a city.",
       {"city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}}, ["city"]),
    fn("get_time", "Get the current local time in an IANA time zone.",
       {"timezone": {"type": "string", "description": "IANA time zone, e.g. Europe/Paris"}}),
    fn("search_web", "Search the web and return the top results.",
       {"query": {"type": "string"}, "num_results": {"type": "integer", "description": "How many results (1-10)"}}, ["query"]),
    fn("send_email", "Send an email.",
       {"to": {"type": "string", "description": "Recipient email address"}, "subject": S, "body": S}),
    fn("create_event", "Create a calendar event.",
       {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
        "time": {"type": "string", "description": "HH:MM, 24-hour"},
        "duration_minutes": {"type": "integer"},
        "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email addresses"}},
       ["title", "date", "time"]),
    fn("convert_currency", "Convert an amount between currencies.",
       {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code"},
        "to_currency": {"type": "string", "description": "ISO code"}}),
    fn("get_stock_price", "Get the latest price of a stock by ticker symbol.",
       {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}}),
    fn("translate", "Translate text into another language.",
       {"text": S, "target_language": {"type": "string", "description": "Language name in English"}}),
    fn("set_timer", "Start a countdown timer.",
       {"minutes": {"type": "integer"}, "label": S}, ["minutes"]),
    fn("book_flight", "Book a flight.",
       {"origin": {"type": "string", "description": "IATA airport code"},
        "destination": {"type": "string", "description": "IATA airport code"},
        "date": {"type": "string", "description": "YYYY-MM-DD"}, "passengers": {"type": "integer"}},
       ["origin", "destination", "date"]),
    fn("lookup_order", "Look up an order by its id.", {"order_id": S}),
    fn("track_shipment", "Track a shipment by its tracking number.", {"tracking_number": S}),
    fn("set_lights", "Turn the lights of a room on or off and set their brightness.",
       {"room": S, "on": {"type": "boolean"}, "brightness": {"type": "integer", "description": "0-100"}}, ["room", "on"]),
    fn("run_sql", "Run a read-only SQL query on the sales database.", {"query": S}),
    fn("calculate", "Evaluate an arithmetic expression.", {"expression": S}),
    fn("play_music", "Play music.", {"artist": S, "song": S}, []),
]}


def U(text):
    return [{"role": "user", "content": text}]


def C(name, **args):
    return {"name": name, "args": args}


CASES = [
    # --- single calls
    {"id": "weather-basic", "tools": ["get_weather"], "messages": U("What's the weather like in Paris right now?"),
     "expected": [C("get_weather", city="Paris")]},
    {"id": "weather-unit", "tools": ["get_weather", "get_time"],
     "messages": U("How hot is it in Phoenix today? Give it to me in fahrenheit."),
     "expected": [C("get_weather", city="Phoenix", unit="fahrenheit")]},
    {"id": "time-tz", "tools": ["get_weather", "get_time"], "messages": U("What time is it in Tokyo?"),
     "expected": [C("get_time", timezone={"contains": "Tokyo"})]},
    {"id": "search", "tools": ["search_web", "get_weather"],
     "messages": U("Search the web for the latest news about the James Webb telescope."),
     "expected": [C("search_web", query={"contains": "James Webb"})]},
    {"id": "search-count", "tools": ["search_web"],
     "messages": U("Find me 3 web results about sourdough starter recipes."),
     "expected": [C("search_web", query={"contains": "sourdough"}, num_results=3)]},
    {"id": "email", "tools": ["send_email", "search_web"],
     "messages": U("Email bob@example.com with the subject 'Lunch' and say: Are we still on for noon tomorrow?"),
     "expected": [C("send_email", to="bob@example.com", subject="Lunch", body={"contains": "noon"})]},
    {"id": "stock", "tools": ["get_stock_price", "search_web"], "messages": U("What's Apple's stock price?"),
     "expected": [C("get_stock_price", symbol="AAPL")]},
    {"id": "currency", "tools": ["convert_currency", "calculate"],
     "messages": U("Convert 250 US dollars to euros."),
     "expected": [C("convert_currency", amount=250, from_currency="USD", to_currency="EUR")]},
    {"id": "translate", "tools": ["translate", "search_web"],
     "messages": U("Translate 'Where is the train station?' into German."),
     "expected": [C("translate", text={"contains": "train station"}, target_language="German")]},
    {"id": "timer", "tools": ["set_timer", "get_time"], "messages": U("Set a 15 minute timer for the pasta."),
     "expected": [C("set_timer", minutes=15)]},
    {"id": "order", "tools": ["lookup_order", "track_shipment"],
     "messages": U("Can you check the status of my order A-55821?"),
     "expected": [C("lookup_order", order_id="A-55821")]},
    {"id": "lights-bool", "tools": ["set_lights", "play_music"],
     "messages": U("Turn off the lights in the kitchen."),
     "expected": [C("set_lights", room={"contains": "kitchen"}, on=False)]},
    {"id": "lights-int", "tools": ["set_lights"],
     "messages": U("Turn the living room lights on at 40% brightness."),
     "expected": [C("set_lights", room={"contains": "living room"}, on=True, brightness=40)]},
    {"id": "flight", "tools": ["book_flight", "get_weather"],
     "messages": U("Book a flight from SFO to JFK on 2026-12-20 for 2 passengers."),
     "expected": [C("book_flight", origin="SFO", destination="JFK", date="2026-12-20", passengers=2)]},
    {"id": "event-array", "tools": ["create_event", "send_email"],
     "messages": U("Schedule a meeting titled 'Design review' on 2026-11-20 at 14:00 for 30 minutes with "
                   "ann@example.com and raj@example.com."),
     "expected": [C("create_event", title={"contains": "Design review"}, date="2026-11-20", time="14:00",
                    duration_minutes=30, attendees={"present": True})]},
    {"id": "sql", "tools": ["run_sql", "calculate"],
     "messages": U("How many orders were placed in 2025? Query the sales database; the table is called orders "
                   "and has a placed_at timestamp column."),
     "expected": [C("run_sql", query={"contains": "orders"})]},
    {"id": "music-optional", "tools": ["play_music", "set_lights"], "messages": U("Play something by Miles Davis."),
     "expected": [C("play_music", artist="Miles Davis")]},
    {"id": "unicode-arg", "tools": ["get_weather"], "messages": U("北京现在天气怎么样？"),
     "expected": [C("get_weather", city={"any_of": ["北京", "Beijing"]})]},
    {"id": "quotes-arg", "tools": ["translate"],
     "messages": U('Translate the sentence "Don\'t worry, be happy" into French.'),
     "expected": [C("translate", text={"contains": "be happy"}, target_language="French")]},
    {"id": "system-prompt", "tools": ["lookup_order", "track_shipment"],
     "messages": [{"role": "system", "content": "You are the support assistant of Acme Inc. Use tools to answer."},
                  {"role": "user", "content": "Where is my package? The tracking number is 1Z999AA10123456784."}],
     "expected": [C("track_shipment", tracking_number="1Z999AA10123456784")]},
    # --- parallel calls
    {"id": "parallel-weather", "tools": ["get_weather"],
     "messages": U("What's the weather in London and in Berlin?"),
     "expected": [C("get_weather", city="London"), C("get_weather", city="Berlin")]},
    {"id": "parallel-stocks", "tools": ["get_stock_price"],
     "messages": U("Get me the current stock prices of MSFT, NVDA and GOOGL."),
     "expected": [C("get_stock_price", symbol="MSFT"), C("get_stock_price", symbol="NVDA"),
                  C("get_stock_price", symbol="GOOGL")]},
    {"id": "parallel-mixed", "tools": ["get_weather", "get_time"],
     "messages": U("I'm calling a friend in Sydney: what's the local time there and what's the weather?"),
     "expected": [C("get_time", timezone={"contains": "Sydney"}), C("get_weather", city="Sydney")]},
    {"id": "parallel-timers", "tools": ["set_timer"],
     "messages": U("Set two timers: 5 minutes labelled eggs and 12 minutes labelled rice."),
     "expected": [C("set_timer", minutes=5, label={"contains": "egg"}), C("set_timer", minutes=12, label={"contains": "rice"})]},
    # --- no call
    {"id": "nocall-greeting", "tools": ["get_weather"], "messages": U("Hi! How are you today?"), "expected": []},
    {"id": "nocall-knowledge", "tools": ["get_stock_price", "get_weather"],
     "messages": U("What is the capital of Australia? Answer from your own knowledge."), "expected": []},
    {"id": "nocall-choice-none", "tools": ["get_weather"], "messages": U("What's the weather in Rome?"),
     "expected": [], "extra": {"tool_choice": "none"}},
    # --- tool_choice
    {"id": "required", "tools": ["get_weather", "get_time"], "messages": U("I'm heading to Oslo."),
     "expected": [C("get_weather", city="Oslo")], "alt_expected": [[C("get_time", timezone={"contains": "Oslo"})]],
     "extra": {"tool_choice": "required"}},
    {"id": "named", "tools": ["get_weather", "get_time"], "messages": U("Tell me about Lisbon."),
     "expected": [C("get_time", timezone={"contains": "Lisbon"})],
     "extra": {"tool_choice": {"type": "function", "function": {"name": "get_time"}}}},
    # --- multi-turn
    {"id": "after-result-answer", "tools": ["get_weather"],
     "messages": U("What's the weather in Madrid?") + [
         {"role": "assistant", "content": None, "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Madrid\"}"}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "{\"city\": \"Madrid\", \"temp_c\": 23, \"sky\": \"sunny\"}"}],
     "expected": [], "answer_contains": ["23"]},
    {"id": "chained-call", "tools": ["lookup_order", "track_shipment"],
     "messages": U("Where is order B-1093 right now?") + [
         {"role": "assistant", "content": None, "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "lookup_order", "arguments": "{\"order_id\": \"B-1093\"}"}}]},
         {"role": "tool", "tool_call_id": "c1",
          "content": "{\"order_id\": \"B-1093\", \"status\": \"shipped\", \"tracking_number\": \"TRK-77410\"}"}],
     "expected": [C("track_shipment", tracking_number="TRK-77410")]},
    {"id": "second-user-turn", "tools": ["get_weather", "get_stock_price"],
     "messages": U("What's the weather in Cairo?") + [
         {"role": "assistant", "content": None, "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Cairo\"}"}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 31}"},
         {"role": "assistant", "content": "It's 31°C in Cairo."},
         {"role": "user", "content": "Thanks. And what's Tesla's share price?"}],
     "expected": [C("get_stock_price", symbol="TSLA")]},
    {"id": "math-tool", "tools": ["calculate", "search_web"],
     "messages": U("Use the calculator tool to compute 1234 * 5678."),
     "expected": [C("calculate", expression={"contains": "1234"})]},
    {"id": "many-tools", "tools": list(TOOLS),
     "messages": U("Remind me in 25 minutes to call mom."),
     "expected": [C("set_timer", minutes=25)]},
    {"id": "many-tools-email", "tools": list(TOOLS),
     "messages": U("Send an email to team@example.org with subject 'Release' saying the build is green."),
     "expected": [C("send_email", to="team@example.org", subject="Release", body={"contains": "green"})]},
]
