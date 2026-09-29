"""Agent — a reason/act loop over a tool registry with a pluggable policy.

The "policy" is what decides the next action. The default is a deterministic,
rule-based policy so the framework runs with **no LLM** (great for tests and
CI); pass an LLM-backed policy to wire in a real model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .memory import ConversationMemory
from .registry import ToolRegistry

# A policy takes (task, step, memory, tools) -> an action dict, e.g.
#   {"tool": "calculator", "arguments": {"expression": "2+2"}}
#   {"final": "the answer"}
Policy = Callable[[str, int, ConversationMemory, ToolRegistry], dict]


@dataclass
class AgentResult:
    task: str
    output: Any
    steps: list[dict] = field(default_factory=list)
    stopped: str = "final"  # "final" | "max_steps"


@dataclass
class Agent:
    name: str
    instruction: str
    tools: ToolRegistry
    policy: Policy
    max_steps: int = 8

    def run(self, task: str) -> AgentResult:
        memory = ConversationMemory(system=f"{self.name}: {self.instruction}")
        memory.add("user", task)
        steps: list[dict] = []
        for i in range(self.max_steps):
            action = self.policy(task, i, memory, self.tools)
            if "final" in action:
                memory.add("assistant", str(action["final"]))
                return AgentResult(task, action["final"], steps, "final")
            tool_name = action.get("tool")
            args = action.get("arguments", {})
            try:
                result = self.tools.call(tool_name, args)
            except Exception as e:  # surface tool errors to the loop, don't crash
                result = {"error": f"{type(e).__name__}: {e}"}
            step = {"step": i, "tool": tool_name, "arguments": args, "result": result}
            steps.append(step)
            memory.add("tool", json.dumps({"tool": tool_name, "result": result}))
        return AgentResult(task, steps[-1]["result"] if steps else None, steps, "max_steps")


def regex_policy(tool_name: str, pattern: str, final_template: str = "Result: {result}") -> Policy:
    """A simple policy: extract an argument via regex, call one tool, then finish."""

    def policy(task: str, step: int, memory: ConversationMemory, tools: ToolRegistry) -> dict:
        if step == 0:
            m = re.search(pattern, task, re.IGNORECASE)
            if not m:
                return {"final": f"Could not parse a {tool_name} request from: {task}"}
            return {"tool": tool_name, "arguments": {"expression": m.group(1)}}
        # step 1: finish using the previous tool result
        prev = json.loads(memory.messages[-1].content)["result"]
        return {"final": final_template.format(result=prev)}

    return policy


def kb_policy() -> Policy:
    """Policy that routes knowledge questions to the knowledge_lookup tool."""

    def policy(task: str, step: int, memory: ConversationMemory, tools: ToolRegistry) -> dict:
        if step == 0:
            return {"tool": "knowledge_lookup", "arguments": {"topic": task}}
        prev = json.loads(memory.messages[-1].content)["result"]
        return {"final": prev}

    return policy
