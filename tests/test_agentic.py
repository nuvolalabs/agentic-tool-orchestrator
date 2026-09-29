"""Tests for the agentic tool orchestrator (stdlib unittest)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic import (  # noqa: E402
    Agent,
    ConversationMemory,
    Orchestrator,
    Supervisor,
    SummaryMemory,
    ToolRegistry,
    calculator,
    knowledge_lookup,
    word_count,
)
from agentic.agent import kb_policy, regex_policy  # noqa: E402
from agentic.orchestrator import Route  # noqa: E402
from agentic.tools import default_registry  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_default_registry_tools(self):
        reg = default_registry()
        self.assertEqual(reg.names(), ["calculator", "knowledge_lookup", "word_count"])

    def test_schema_shape(self):
        reg = default_registry()
        schema = reg.get("calculator").to_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculator")
        self.assertIn("expression", schema["function"]["parameters"]["properties"])

    def test_dispatch_dict_and_json(self):
        reg = default_registry()
        self.assertEqual(reg.call("calculator", {"expression": "2 + 3"}), 5.0)
        self.assertEqual(reg.call("calculator", '{"expression": "4 * 2"}'), 8.0)

    def test_unknown_tool_raises(self):
        with self.assertRaises(KeyError):
            default_registry().call("nope", {})


class TestTools(unittest.TestCase):
    def test_calculator_safe_eval(self):
        self.assertEqual(calculator("(2 + 3) * 4"), 20.0)
        self.assertEqual(calculator("10 % 3"), 1.0)

    def test_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            calculator("__import__('os').system('echo hi')")

    def test_knowledge_lookup(self):
        self.assertIn("5-7 business days", knowledge_lookup("refund policy"))

    def test_word_count(self):
        self.assertEqual(word_count("hello world")["words"], 2)


class TestMemory(unittest.TestCase):
    def test_rolling_window(self):
        mem = ConversationMemory(max_messages=3)
        for i in range(5):
            mem.add("user", f"m{i}")
        self.assertEqual(len(mem.messages), 3)
        self.assertEqual(mem.messages[-1].content, "m4")

    def test_summary_memory_keeps_summary(self):
        mem = SummaryMemory(max_messages=3)
        for i in range(6):
            mem.add("user", f"message {i}")
        self.assertTrue(mem.summary)


class TestAgent(unittest.TestCase):
    def test_calculator_agent_runs_tool_then_finishes(self):
        agent = Agent(
            name="math",
            instruction="do arithmetic",
            tools=default_registry(),
            policy=regex_policy("calculator", r"calculate\s+(.+)"),
        )
        res = agent.run("calculate 12 * (3 + 1)")
        self.assertEqual(res.output, "Result: 48.0")
        self.assertEqual(res.steps[0]["tool"], "calculator")
        self.assertEqual(res.stopped, "final")

    def test_agent_reports_unparseable(self):
        agent = Agent("math", "arith", default_registry(), regex_policy("calculator", r"calc (.+)"))
        res = agent.run("what is the weather")
        self.assertIn("Could not parse", str(res.output))

    def test_kb_agent(self):
        agent = Agent("support", "answer FAQ", default_registry(), kb_policy())
        res = agent.run("warranty")
        self.assertIn("12-month", str(res.output))

    def test_max_steps_stop(self):
        def never_finish(task, step, mem, tools):
            return {"tool": "word_count", "arguments": {"text": "x"}}

        agent = Agent("looper", "loop", default_registry(), never_finish, max_steps=3)
        res = agent.run("go")
        self.assertEqual(res.stopped, "max_steps")
        self.assertEqual(len(res.steps), 3)


class TestOrchestration(unittest.TestCase):
    def test_pipeline(self):
        a = Agent("extract", "-", default_registry(), regex_policy("calculator", r"calc (.+)"))
        b = Agent("support", "-", default_registry(), kb_policy())
        results = Orchestrator([a, b]).run("calc 5 + 5")
        self.assertEqual(len(results), 2)

    def test_supervisor_routes_by_keyword(self):
        math_agent = Agent("math", "-", default_registry(), regex_policy("calculator", r"calc (.+)"))
        kb_agent = Agent("support", "-", default_registry(), kb_policy())
        sup = Supervisor([Route(("calc", "math", "arithmetic"), math_agent),
                          Route(("refund", "shipping", "warranty"), kb_agent)])
        self.assertEqual(sup.route("please calc 7+7").name, "math")
        self.assertEqual(sup.route("what is the refund policy").name, "support")
        res = sup.dispatch("calc 9 + 1")
        self.assertEqual(res.output, "Result: 10.0")

    def test_supervisor_no_route(self):
        sup = Supervisor([])
        res = sup.dispatch("anything")
        self.assertIn("No agent", str(res.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
