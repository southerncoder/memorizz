# Workflow Mode

Workflow mode targets deterministic task execution (think onboarding checklists, ticket triage, or knowledge-base upkeep). It favors procedural memory and tools over conversational depth.

## Memory Stack

- `MemoryType.WORKFLOW_MEMORY`
- `MemoryType.TOOLBOX`
- `MemoryType.LONG_TERM_MEMORY`
- `MemoryType.SHORT_TERM_MEMORY`

## Sample Flow

```python
from memorizz.enums import ApplicationMode

agent = (MemAgentBuilder()
    .with_application_mode(ApplicationMode.WORKFLOW)
    .with_memory_provider(provider)
    .with_tool(module_path="memorizz.tools.workflow")
    .build())

agent.run("Process ticket 12491 and update the changelog")
```

Workflow mode keeps episodic memory minimal so the agent can stay focused on the currently executing process. Pair it with shared memory if you need a supervisor agent to inspect progress.
