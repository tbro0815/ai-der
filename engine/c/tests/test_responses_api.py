"""OpenAI Responses API (/v1/responses) as a translation layer over the chat path.

Codex CLI 0.149 is the reference client: it only speaks the Responses wire API,
streams by default, sends `instructions` + `input` items (developer/user messages,
function_call / function_call_output, reasoning with encrypted_content), function
tools plus `namespace`/`web_search` tool types, and keys off `type` in each SSE event.
"""
import base64
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from openai_server import (APIServer, APIError, responses_to_openai, responses_tools,
                           responses_reasoning_item, RESPONSES_REASONING_PREFIX,
                           render_chat_qwen38)


class FakeEngine:
    def __init__(self, script=("Hé", "llo"), length_limited=False, restored=None):
        self.script = script
        self.length_limited = length_limited
        self.restored = restored
        self.prompts = []

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None,
                 on_tool=None, presence_penalty=None, reasoning_budget=None, image=None):
        self.prompts.append(prompt)
        if on_accept:
            on_accept()
        for chunk in self.script:
            on_text(chunk)
            if stopped is not None and stopped():
                break
        stats = {"prompt_tokens": 11, "completion_tokens": 3,
                 "length_limited": self.length_limited}
        if self.restored is not None:
            stats["restored_tokens"] = self.restored
        return stats


class TranslationTest(unittest.TestCase):
    def test_instructions_and_leading_developer_become_one_system_block(self):
        messages = responses_to_openai({
            "instructions": "You are Codex.",
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "Skills: none."}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]},
            ]})
        self.assertEqual(messages, [{"role": "system", "content": "You are Codex.\n\nSkills: none."},
                                    {"role": "user", "content": "hi"}])

    def test_later_developer_message_becomes_a_user_turn(self):
        messages = responses_to_openai({"input": [
            {"type": "message", "role": "user", "content": "a"},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "MODE"}]},
            {"type": "message", "role": "user", "content": "b"},
        ]})
        self.assertEqual([m["role"] for m in messages], ["user", "user", "user"])
        self.assertEqual(messages[1]["content"], "MODE")
        # the renderer accepts it (a mid-conversation system message would be a 400)
        render_chat_qwen38(messages)

    def test_string_input_is_a_user_message(self):
        self.assertEqual(responses_to_openai({"input": "ping"}), [{"role": "user", "content": "ping"}])

    def test_function_call_round_trip_with_reasoning(self):
        thinking = "I should list the files."
        item = responses_reasoning_item(thinking)
        self.assertTrue(item["encrypted_content"].startswith(RESPONSES_REASONING_PREFIX))
        messages = responses_to_openai({"input": [
            {"type": "message", "role": "user", "content": "list files"},
            {"type": "reasoning", "id": item["id"], "summary": [],
             "encrypted_content": item["encrypted_content"]},
            {"type": "function_call", "call_id": "call_1", "name": "exec_command",
             "arguments": "{\"cmd\": \"ls\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "exec_command",
             "arguments": "{\"cmd\": \"pwd\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            {"type": "function_call_output", "call_id": "call_2",
             "output": [{"type": "input_text", "text": "/tmp"}]},
            {"type": "message", "role": "user", "content": "thanks"},
        ]})
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "tool", "user"])
        assistant = messages[1]
        self.assertEqual(assistant["reasoning_content"], thinking)
        self.assertEqual([c["id"] for c in assistant["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(json.loads(assistant["tool_calls"][1]["function"]["arguments"]), {"cmd": "pwd"})
        self.assertEqual(messages[2], {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"})
        self.assertEqual(messages[3]["content"], "/tmp")
        prompt = render_chat_qwen38(messages, tools=[{"type": "function", "function": {
            "name": "exec_command", "parameters": {"type": "object", "properties": {}}}}])
        self.assertIn(thinking, prompt)
        self.assertIn("<tool_response>", prompt)

    def test_assistant_output_text_message(self):
        messages = responses_to_openai({"input": [
            {"type": "message", "role": "user", "content": "q"},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "answer", "annotations": []}]},
            {"type": "message", "role": "user", "content": "more"},
        ]})
        self.assertEqual(messages[1], {"role": "assistant", "content": "answer"})

    def test_input_image_becomes_an_image_part(self):
        messages = responses_to_openai({"input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "what is this"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"}]}]})
        parts = messages[0]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"])
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_unsupported_items_are_refused_clearly(self):
        with self.assertRaises(APIError) as caught:
            responses_to_openai({"input": [{"type": "item_reference", "id": "x"}]})
        self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(APIError):
            responses_to_openai({"input": [{"type": "computer_call"}]})
        with self.assertRaises(APIError):
            responses_to_openai({"instructions": "only a system prompt", "input": []})

    def test_tools_translate_and_hosted_types_are_skipped(self):
        tools, choice, skipped, namespaces = responses_tools({
            "tools": [
                {"type": "function", "name": "exec_command", "description": "run",
                 "strict": False, "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
                {"type": "namespace", "name": "mcp__node_repl", "description": "Node REPL", "tools": [
                    {"type": "function", "name": "js", "description": "eval", "parameters": {"type": "object", "properties": {}}}]},
                {"type": "namespace", "name": "mcp__cua_repl", "tools": [
                    {"type": "function", "name": "js", "description": "eval too", "parameters": {"type": "object", "properties": {}}}]},
                {"type": "web_search", "external_web_access": False},
            ],
            "tool_choice": "auto"})
        self.assertEqual(tools[0], {"type": "function", "function": {
            "name": "exec_command", "description": "run",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}})
        # namespaced tools flatten to unique names; the mapping restores name + namespace
        self.assertEqual([t["function"]["name"] for t in tools[1:]], ["mcp__node_repl.js", "mcp__cua_repl.js"])
        self.assertEqual(tools[1]["function"]["description"], "[mcp__node_repl: Node REPL] eval")
        self.assertEqual(namespaces, {"mcp__node_repl.js": ("mcp__node_repl", "js"),
                                      "mcp__cua_repl.js": ("mcp__cua_repl", "js")})
        self.assertEqual(choice, "auto")
        self.assertEqual(skipped, ["web_search"])
        # namespaces beyond the schema budget are skipped, in request order
        with patch.dict("os.environ", {"COLI_RESPONSES_NAMESPACE_BUDGET_CHARS": "150"}):
            tools2, _, skipped2, ns2 = responses_tools({"tools": [
                {"type": "namespace", "name": "a", "tools": [
                    {"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}]},
                {"type": "namespace", "name": "b", "tools": [
                    {"type": "function", "name": "g", "parameters": {"type": "object", "properties": {}}}]}]})
        self.assertEqual([t["function"]["name"] for t in tools2], ["a.f"])
        self.assertEqual(list(ns2), ["a.f"])
        self.assertTrue(skipped2 and skipped2[0].startswith("namespace:b ("))
        self.assertEqual(responses_tools({"tool_choice": {"type": "function", "name": "f"}})[1],
                         {"type": "function", "function": {"name": "f"}})
        # a namespaced call coming back in the history renders under its flat name
        messages = responses_to_openai({"input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "call_id": "c1", "name": "js", "namespace": "mcp__node_repl",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "2"}]}, namespaces)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "mcp__node_repl.js")


class ResponsesHTTPTest(unittest.TestCase):
    def start(self, engine):
        self.engine = engine
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model", "secret", 64,
                                kv_slots=2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, body):
        head = {"Content-Type": "application/json", "Authorization": "Bearer secret"}
        body = {"model": "test-model", **body}
        return urlopen(Request(self.base + "/v1/responses", data=json.dumps(body).encode(),
                               headers=head), timeout=5)

    def events(self, raw):
        out = []
        for block in raw.split("\n\n"):
            lines = [l for l in block.split("\n") if l and not l.startswith(":")]
            if not lines:
                continue
            name = [l[7:] for l in lines if l.startswith("event: ")][0]
            data = json.loads([l[6:] for l in lines if l.startswith("data: ")][0])
            self.assertEqual(data["type"], name)
            out.append(data)
        return out

    def test_non_streaming_response_object(self):
        self.start(FakeEngine(("Hé", "llo")))
        with self.post({"input": "hi", "stream": False, "reasoning": {"effort": "none"}}) as r:
            body = json.load(r)
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["output"][0]["type"], "message")
        self.assertEqual(body["output"][0]["content"][0]["text"], "Héllo")
        self.assertEqual(body["usage"]["input_tokens"], 11)
        self.assertEqual(body["usage"]["output_tokens"], 3)
        self.assertEqual(body["usage"]["total_tokens"], 14)

    def test_streaming_text_event_sequence(self):
        self.start(FakeEngine(("Hé", "llo")))
        with self.post({"input": "hi", "stream": True, "reasoning": {"effort": "none"}}) as r:
            self.assertEqual(r.headers["Content-Type"], "text/event-stream")
            evs = self.events(r.read().decode())
        names = [e["type"] for e in evs]
        self.assertEqual(names[:2], ["response.created", "response.in_progress"])
        self.assertEqual(names[2:4], ["response.output_item.added", "response.content_part.added"])
        self.assertEqual(names[-1], "response.completed")
        deltas = [e["delta"] for e in evs if e["type"] == "response.output_text.delta"]
        self.assertEqual("".join(deltas), "Héllo")
        done = [e for e in evs if e["type"] == "response.output_text.done"][0]
        self.assertEqual(done["text"], "Héllo")
        self.assertEqual([e["sequence_number"] for e in evs], list(range(len(evs))))
        final = evs[-1]["response"]
        self.assertEqual(final["output"][0]["content"][0]["text"], "Héllo")
        self.assertEqual(final["usage"]["output_tokens"], 3)

    def test_streaming_tool_call_becomes_function_call_items(self):
        reply = ("<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls\n</parameter>\n"
                 "</function>\n</tool_call>")
        self.start(FakeEngine((reply[:20], reply[20:])))
        tools = [{"type": "function", "name": "exec_command", "description": "run",
                  "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}]
        with patch("openai_server.ARCH", "qwen38"), \
                self.post({"input": "list", "stream": True, "tools": tools,
                           "reasoning": {"effort": "none"}}) as r:
            evs = self.events(r.read().decode())
        names = [e["type"] for e in evs]
        self.assertNotIn("response.output_text.delta", names)   # tool markers never leak as text
        self.assertIn("response.function_call_arguments.done", names)
        done = [e for e in evs if e["type"] == "response.output_item.done" and
                e["item"]["type"] == "function_call"][0]["item"]
        self.assertEqual(done["name"], "exec_command")
        self.assertEqual(json.loads(done["arguments"]), {"cmd": "ls"})
        self.assertTrue(done["call_id"])
        final = evs[-1]["response"]
        self.assertEqual([i["type"] for i in final["output"]], ["function_call"])
        self.assertEqual(final["status"], "completed")

    def test_namespaced_tool_call_carries_the_namespace_field(self):
        reply = ("<tool_call>\n<function=mcp__node_repl.js>\n<parameter=code>\n1+1\n</parameter>\n"
                 "</function>\n</tool_call>")
        self.start(FakeEngine((reply,)))
        tools = [{"type": "namespace", "name": "mcp__node_repl", "tools": [
            {"type": "function", "name": "js", "description": "eval",
             "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}]}]
        with patch("openai_server.ARCH", "qwen38"), \
                self.post({"input": "compute", "stream": True, "tools": tools,
                           "reasoning": {"effort": "none"}}) as r:
            evs = self.events(r.read().decode())
        items = [e["item"] for e in evs if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"]
        self.assertEqual((items[0]["name"], items[0]["namespace"]), ("js", "mcp__node_repl"))
        self.assertEqual(json.loads(items[0]["arguments"]), {"code": "1+1"})
        added = [e["item"] for e in evs if e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"][0]
        self.assertEqual(added["namespace"], "mcp__node_repl")

    def test_reasoning_streams_as_summary_and_round_trips(self):
        self.start(FakeEngine(("think", "ing\n</think>\n\nanswer")))
        with patch("openai_server.ARCH", "qwen38"), \
                self.post({"input": "q", "stream": True, "reasoning": {"effort": "low"}}) as r:
            evs = self.events(r.read().decode())
        names = [e["type"] for e in evs]
        self.assertIn("response.reasoning_summary_text.delta", names)
        self.assertLess(names.index("response.reasoning_summary_text.done"),
                        names.index("response.output_text.delta"))
        items = evs[-1]["response"]["output"]
        self.assertEqual([i["type"] for i in items], ["reasoning", "message"])
        enc = items[0]["encrypted_content"]
        self.assertEqual(base64.b64decode(enc[len(RESPONSES_REASONING_PREFIX):]).decode(), "thinking")
        self.assertEqual(items[1]["content"][0]["text"], "answer")

    def test_length_limit_is_incomplete(self):
        self.start(FakeEngine(("cut",), length_limited=True))
        with self.post({"input": "q", "stream": True, "reasoning": {"effort": "none"},
                        "max_output_tokens": 3}) as r:
            evs = self.events(r.read().decode())
        self.assertEqual(evs[-1]["type"], "response.incomplete")
        self.assertEqual(evs[-1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})


class PrefixDivergenceTest(unittest.TestCase):
    def test_spans_mark_each_rendered_block(self):
        spans = []
        messages = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                    {"role": "tool", "tool_call_id": "c1", "content": "r2"},
                    {"role": "user", "content": "c"}]
        prompt = render_chat_qwen38(messages, tools=[{"type": "function", "function": {
            "name": "f", "parameters": {"type": "object", "properties": {}}}}], spans=spans)
        self.assertEqual([(i, r) for _, i, r in spans],
                         [(0, "system"), (1, "user"), (2, "assistant"), (3, "tool"), (5, "user")])
        offsets = [o for o, _, _ in spans]
        self.assertEqual(offsets, sorted(offsets))
        for offset, index, role in spans[1:]:
            self.assertTrue(prompt[offset:].startswith("<|im_start|>" + ("user" if role == "tool" else role)))


if __name__ == "__main__":
    unittest.main()
