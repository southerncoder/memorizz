# Shared Memory

Shared memory powers coordination between multiple agents. It sits in `src/memorizz/coordination/shared_memory/` and corresponds to `MemoryType.SHARED_MEMORY`.

## Why It Exists

Complex workflows often split responsibilities across researcher, analyst, and writer agents. Shared memory provides a blackboard-like store where agents can exchange artifacts, delegate tasks, and keep track of global progress.

## Creating a Session

```python
from memorizz.coordination.shared_memory import SharedMemory

shared = SharedMemory(memory_provider)
session_id = shared.create_shared_session(
    root_agent_id="orchestrator",
    delegate_agent_ids=["researcher", "writer"],
)
```

Each session keeps:

- Participants and roles
- Messages and artifacts exchanged between agents
- Links to the originating episodic/semantic records for traceability

## Patterns

- Orchestrator + delegate setups (research, summarization, QA)
- Human-in-the-loop review queues where both agents and operators inspect shared state
- Multi-modal agents handing off voice, vision, or text data through a common buffer

Shared memory complements the per-agent stores so everyone observes the same document trail without duplicating data.
