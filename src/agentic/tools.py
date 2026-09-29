"""Built-in example tools for the orchestrator."""
from __future__ import annotations

import ast
import operator

from .registry import Tool, tool

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool("Safely evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.")
def calculator(expression: str) -> float:
    return float(_eval_node(ast.parse(expression, mode="eval")))


_KB = {
    "refund": "Refunds are processed within 5-7 business days to the original payment method.",
    "shipping": "Standard shipping takes 3-5 business days; express is next-day.",
    "warranty": "Products carry a 12-month manufacturer warranty covering defects.",
}


@tool("Look up a knowledge-base answer by keyword (topic).")
def knowledge_lookup(topic: str) -> str:
    t = (topic or "").lower().strip()
    for k, v in _KB.items():
        if k in t:
            return v
    return "No matching knowledge-base entry."


@tool("Count words and characters in a piece of text.")
def word_count(text: str) -> dict:
    words = text.split()
    return {"words": len(words), "characters": len(text)}


def default_registry():
    from .registry import ToolRegistry

    return ToolRegistry([calculator, knowledge_lookup, word_count])
