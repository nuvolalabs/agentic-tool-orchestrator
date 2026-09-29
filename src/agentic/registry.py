"""Tool registry — function/JSON-schema tool definitions and dispatch."""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """A callable tool with a JSON-schema description (LLM-tool-calling shape)."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def to_schema(self) -> dict:
        """OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def _schema_from_signature(fn: Callable) -> dict:
    """Build a minimal JSON schema from a function signature."""
    props, required = {}, []
    for pname, p in inspect.signature(fn).parameters.items():
        if pname in ("self", "cls"):
            continue
        typ = "string"
        if p.annotation in (int, float):
            typ = "number"
        elif p.annotation is bool:
            typ = "boolean"
        props[pname] = {"type": typ}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def tool(description: str, name: str | None = None):
    """Decorator: register a function as a Tool."""

    def wrap(fn: Callable) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=description,
            parameters=_schema_from_signature(fn),
            func=fn,
        )

    return wrap


class ToolRegistry:
    """Holds tools and dispatches calls by name."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def call(self, name: str, arguments: dict | str) -> Any:
        """Dispatch a tool call; accepts a dict or a JSON string of args."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        return self.get(name)(**arguments)
