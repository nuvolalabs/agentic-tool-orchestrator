"""Agentic tool orchestrator — a tiny multi-agent framework (stdlib only).

Demonstrates: agent loops, tool/function calling, a planner, conversation
memory, and a supervisor that routes between specialist agents.
"""
from .registry import ToolRegistry, Tool, tool
from .agent import Agent, AgentResult
from .memory import ConversationMemory, SummaryMemory
from .orchestrator import Orchestrator, Supervisor
from .tools import calculator, knowledge_lookup, word_count

__all__ = [
    "ToolRegistry", "Tool", "tool",
    "Agent", "AgentResult",
    "ConversationMemory", "SummaryMemory",
    "Orchestrator", "Supervisor",
    "calculator", "knowledge_lookup", "word_count",
]
__version__ = "1.0.0"
