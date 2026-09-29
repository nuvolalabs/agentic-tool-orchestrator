# Agentic Tool Orchestrator

A small, dependency-free **multi-agent framework** demonstrating the agent
patterns that applied-GenAI JDs ask for: tool/function calling, a reason–act
loop, pluggable policies, conversation memory, and supervisor routing between
specialist agents.

## Why

Most "agent frameworks" are black boxes. This one is ~400 lines of readable
Python that shows exactly how an agent loop works — and runs with **no LLM and
no dependencies**, so the behaviour is deterministic and testable.

## Architecture

```
                ┌─────────────┐
   task ───────▶│ Supervisor  │  keyword/score routing
                └──────┬──────┘
                       │ picks
                ┌──────▼──────┐   reason/act loop     ┌──────────────┐
                │   Agent     │◀─────────────────────▶│ ToolRegistry │
                │  (policy)   │  tool calls / results  │  (schemas)   │
                └──────┬──────┘                        └──────────────┘
                       │
                ┌──────▼──────┐
                │   Memory    │  rolling window + summarisation
                └─────────────┘

   Orchestrator: run a fixed pipeline of agents, chaining each output into the next.
```

## Quickstart

```bash
python3 tests/test_agentic.py        # 17 tests, stdlib only

PYTHONPATH=src python3 - <<'PY'
from agentic import Agent, Supervisor
from agentic.agent import regex_policy, kb_policy
from agentic.orchestrator import Route
from agentic.tools import default_registry

math = Agent("math", "arithmetic", default_registry(), regex_policy("calculator", r"calc (.+)"))
kb   = Agent("support", "FAQ", default_registry(), kb_policy())
sup  = Supervisor([Route(("calc", "math"), math), Route(("refund", "warranty"), kb)])

print(sup.dispatch("calc 12 * (3 + 1)").output)   # Result: 48.0
print(sup.dispatch("what is the refund policy").output)
PY
```

## Design notes

- **Policies are pluggable functions** — swap the deterministic rule policy for an
  LLM-backed one without changing the agent loop, memory, or tools.
- **Tools carry OpenAI-style JSON schemas** built from the function signature, so
  they drop straight into a real function-calling request.
- **Safe calculator** parses an AST and whitelists arithmetic ops — it rejects
  code injection rather than `eval`-ing input.
- **Tool errors never crash the loop** — they're surfaced as a step result.

## Skills demonstrated

`agent architecture` · `tool/function calling` · `prompt/policy design`
· `multi-agent orchestration` · `supervisor routing` · `conversation memory`
· `safe expression evaluation` · `Python packaging + tests`

## License

MIT
