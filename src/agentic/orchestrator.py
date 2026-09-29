"""Orchestrator + Supervisor — route tasks to specialist agents."""
from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent, AgentResult
from .registry import ToolRegistry


class Orchestrator:
    """Runs a fixed pipeline of agents, passing each result to the next."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def run(self, task: str) -> list[AgentResult]:
        results, current = [], task
        for agent in self.agents:
            res = agent.run(current)
            results.append(res)
            current = str(res.output)
        return results


@dataclass
class Route:
    keywords: tuple[str, ...]
    agent: Agent


class Supervisor:
    """Keyword-routing supervisor: picks the best specialist agent per task.

    A deterministic stand-in for an LLM router — same interface, so you can swap
    in a model-based classifier without touching callers.
    """

    def __init__(self, routes: list[Route], fallback: Agent | None = None):
        self.routes = routes
        self.fallback = fallback

    def route(self, task: str) -> Agent | None:
        t = task.lower()
        best, best_score = None, 0
        for r in self.routes:
            score = sum(1 for k in r.keywords if k in t)
            if score > best_score:
                best, best_score = r.agent, score
        return best or self.fallback

    def dispatch(self, task: str) -> AgentResult:
        agent = self.route(task)
        if agent is None:
            return AgentResult(task, "No agent could handle this task.", [], "final")
        return agent.run(task)
